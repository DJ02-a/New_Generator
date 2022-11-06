import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from MyModel.utils.blocks import NewResnetBlock, ConvBlock

class My_Generator(nn.Module):
    def __init__(self, input_ch=64, input_size=128, output_size=512):
        super(My_Generator, self).__init__()

        self.input_ch = input_ch
        self.input_size = input_size
        self.output_size = output_size
        self.feature_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.feature_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.resblock_0 = NewResnetBlock(self.input_ch, 256, 64)
        self.resblock_1 = NewResnetBlock(256, 256, 64)
        self.resblock_2 = NewResnetBlock(256, 256, 64)
        self.resblock_3 = NewResnetBlock(256, 256, 64)
        
        # up size
        self.up_conv_0 = ConvBlock(256, 128, 3, 2, 1, norm='bn', activation='lrelu', transpose=True)
        self.up_conv_1 = ConvBlock(128, 32, 3, 2, 1, norm='bn', activation='lrelu', transpose=True)
        self.color_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        # self.activation = nn.Tanh()

    # x == mix features
    def forward(self, x, mid_feature):
        x = self.feature_conv1(x)
        mid_feature = self.feature_conv2(mid_feature)
        
        x = self.resblock_0(x, mid_feature)
        x = self.resblock_1(x, mid_feature)
        x = self.resblock_2(x, mid_feature)
        x = self.resblock_3(x, mid_feature)
        
        x = self.up_conv_0(x)
        x = self.up_conv_1(x)
        
        x = self.color_conv(x)
        # x = self.activation(x)
        return x
       