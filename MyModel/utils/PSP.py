import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class GradualStyleEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=64):
        super(GradualStyleEncoder, self).__init__()
        blocks = get_blocks(50)
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
            )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR_SE(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride
                    ))
        self.body = nn.Sequential(*modules)

        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.output_layer = nn.Conv2d(64, out_dim, 1, 1, 0, bias=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x): # b, 3, 256, 256
        x = self.input_layer(x) # b 64 256 256

        modulelist = list(self.body._modules.values())
        c0 = x
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 0:
                c1 = x
            elif i == 6:
                c2 = x
                
            elif i == 20:
                c3 = x
            elif i == 23:
                c4 = x
        
        p3 = self._upsample_add(self.latlayer1(c4), c3) # 256 32 32
        p2 = self._upsample_add(self.latlayer2(p3), c2) # 128 64 64
        p1 = self._upsample_add(self.latlayer3(p2), c1) # 64 128 128
        p0 = self._upsample_add(self.latlayer4(p1), c0) # 64 256 256
        out= self.output_layer(p0)

        return out # b 64 256 256