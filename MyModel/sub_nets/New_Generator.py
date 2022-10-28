import math
import torch.nn as nn
import torch.nn.functional as F

from MyModel.utils.blocks import NewResnetBlock

# input(style vectors) shape : b*512*8*8
# output shape : b*c*opt.output_size*opt.output_size
class My_Generator(nn.Module):
    def __init__(self, input_ch=67, input_size=32, output_size=256):
        super(My_Generator, self).__init__()

        self.input_ch = input_ch
        self.input_size = input_size
        self.output_size = output_size
        self.num_up_layers = math.log2(output_size // self.input_size) # 32 -> 256 : 3

        self.ch ={}
        # 32:512, 64:256, 128: 128, 256:64 \ , 512:32 1024:16
        # 4
        for i in range(int(self.num_up_layers)+1):
            self.ch[input_size*(2**(i))] = 256 // (2**(i))
        # 3
        self.ch_match = nn.ModuleList([nn.Conv2d(self.input_ch,o,3,padding=1) for i,o in self.ch.items() if i < output_size])


        in_ch = 512
        self.res_list = []
        for in_size , in_ch in self.ch.items(): # 
            if in_size != output_size:
                res_block = NewResnetBlock(in_ch, in_ch//2, in_ch)
                self.res_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_list)

        self.conv_img = nn.Conv2d(self.ch[output_size], 3, 3, padding=1)

    # x == mix features
    def forward(self, x):
        # 3
        mid_features = []
        for idx, conv_layer in enumerate(self.ch_match):
            size = int(2**(math.log2(self.output_size) - self.num_up_layers + idx))
            _x = F.interpolate(x, (size, size), mode='bilinear')
            mid_feature = conv_layer(_x)
            mid_features.append(mid_feature)

        x = mid_features[0]
        for mid_feature, res_layer in zip(mid_features, self.res_blocks):
            x = res_layer(x, mid_feature)
            x = F.interpolate(x,scale_factor=2, mode='bilinear')

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
