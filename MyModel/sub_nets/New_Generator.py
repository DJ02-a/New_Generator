import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from MyModel.utils.blocks import NewResnetBlock

# input(style vectors) shape : b*512*8*8
# output shape : b*c*opt.output_size*opt.output_size


# # input(style vectors) shape : b*512*8*8
# # output shape : b*c*opt.output_size*opt.output_size
# class My_Generator(nn.Module):
#     def __init__(self, input_ch=67, input_size=64, output_size=512):
#         super(My_Generator, self).__init__()
        
#         self.ch ={}
#         self.input_ch = input_ch
#         self.input_size = input_size
#         self.output_size = output_size
#         # 32:512, 64:256, 128: 128, 256:64 , 512:32 1024:16
#         # input: 64 output : 512 -> 64:256, 128:128, 256:64 512:32
#         self.num_up_layers = math.log2(output_size // self.input_size) # 64 -> 512 : 3
#         for i in range(int(self.num_up_layers)+1):
#             self.ch[input_size*(2**(i))] = 256 // (2**(i))
            
        
#         self.ch_match = nn.ModuleList([nn.Conv2d(self.input_ch,o,3,padding=1) for res,o in self.ch.items() if res < output_size])

#         self.up_0 = NewResnetBlock(256, 128, 256)   # 64
#         self.up_1 = NewResnetBlock(128, 64, 128)    # 128
#         self.up_2 = NewResnetBlock(64, 32, 64)  # 256
#         self.conv_img = nn.Conv2d(32, 3, 3, padding=1)

#         self.up = nn.Upsample(scale_factor=2)

#     def forward(self, x):


#         mid_features = []
#         for idx, conv_layer in enumerate(self.ch_match):
#             size = int(2**(math.log2(self.output_size) - self.num_up_layers + idx))
#             _x = F.interpolate(x, (size, size), mode='bilinear')
#             mid_feature = conv_layer(_x)
#             mid_features.append(mid_feature)
#         import pdb;pdb.set_trace()
#         x = self.up_0(mid_features[0], mid_features[0])                           # b, 512, 64, 64
#         x = self.up(x)

#         x = self.up_1(x, mid_features[1])                           # b, 256, 128, 128
#         x = self.up(x)

#         x = self.up_2(x, mid_features[2])                           # b, 128, 256, 256
#         x = self.up(x)

#         x = self.conv_img(F.leaky_relu(x, 2e-1))
#         x = F.tanh(x)

#         return x

class My_Generator(nn.Module):
    def __init__(self, input_ch=67, input_size=64, output_size=512):
        super(My_Generator, self).__init__()

        self.input_ch = input_ch
        self.input_size = input_size
        self.output_size = output_size
        self.num_up_layers = math.log2(output_size // self.input_size) # 32 -> 256 : 3

        self.ch ={}
        # 32:512, 64:256, 128: 128, 256:64 , 512:32 1024:16
        # input: 64 output : 512 -> 64:256, 128:128, 256:64 512:32
        for i in range(int(self.num_up_layers)+1):
            self.ch[input_size*(2**(i))] = 256 // (2**(i))
            
        # 3
        self.ch_match = nn.ModuleList([nn.Conv2d(self.input_ch,o,3,padding=1) for res,o in self.ch.items() if res < output_size])


        in_ch = 256
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
            _x = F.interpolate(x, (size, size), mode='bilinear', align_corners=False)
            mid_feature = conv_layer(_x)
            mid_features.append(mid_feature)
        

        x = mid_features[0]
        for mid_feature, res_layer in zip(mid_features, self.res_blocks):
            
            x = res_layer(x, mid_feature)
            x = F.interpolate(x,scale_factor=2, mode='bilinear', align_corners=False)
            
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x
