import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from .resnet_encoder import *


class FusionUNet(nn.module):
    """Constructs a Depth Fusion Model
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, inputf, up_conv1_inplanes, up_conv1_outplanes, ngf=64, deepBlender=False):
        super(FusionUNet, self).__init__()  # TODO
        self.up_conv1_inplanes = up_conv1_inplanes
        self.up_conv1_outplanes = up_conv1_outplanes
        assert type(up_conv1_inplanes) in [
            tuple, list] and len(up_conv1_inplanes) == 5
        assert type(up_conv1_outplanes) in [
            tuple, list] and len(up_conv1_outplanes) == 5

        # * constructin UNET, from innermost to outermost
        unet_block = UNetBlock(down_conv1_in=ngf*8, down_conv2_in=ngf*8,
                               up_conv1_in=self.up_conv1_inplanes[4], up_conv1_out=self.up_conv1_outplanes[4], up_conv2_out=256, innermost=True, deepBlender=deepBlender)  # * inner most layer
        unet_block = UNetBlock(down_conv1_in=ngf*4, down_conv2_in=ngf*8,
                               up_conv1_in=self.up_conv1_inplanes[3], up_conv1_out=self.up_conv1_outplanes[3], up_conv2_out=256, submodule=unet_block, deepBlender=deepBlender)
        unet_block = UNetBlock(down_conv1_in=ngf*2, down_conv2_in=ngf*4,
                               up_conv1_in=self.up_conv1_inplanes[2], up_conv1_out=self.up_conv1_outplanes[2], up_conv2_out=256, submodule=unet_block, deepBlender=deepBlender)
        unet_block = UNetBlock(down_conv1_in=ngf, down_conv2_in=ngf*2,
                               up_conv1_in=self.up_conv1_inplanes[1], up_conv1_out=self.up_conv1_outplanes[1], up_conv2_out=64 if deepBlender else 128, submodule=unet_block, deepBlender=deepBlender)
        self.model = UNetBlock(down_conv1_in=inputf, down_conv2_in=ngf,
                               up_conv1_in=self.up_conv1_inplanes[0], up_conv1_out=self.up_conv1_outplanes[0], up_conv2_out=32, outermost=True, deepBlender=deepBlender)  # * outer most

        self.finalConv1 = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=3)
        self.LReLU = nn.LeakyReLU(0.2, True)
        self.finalConv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=3)

        # * kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_dm, x_ds):
        output = self.model(x_dm, x_ds)
        return self.finalConv2(self.LReLU(self.finalConv1(output)))


class UNetBlock(nn.module):
    def __init__(self, down_conv1_in, down_conv2_in, up_conv1_in, up_conv1_out, up_conv2_out, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, deepBlender=False):
        self.innermost = innermost
        self.outermost = outermost
        self.deepBlender = deepBlender  # * cat both output in forward
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        LReLU = nn.LeakyReLU(0.2, inplace=True)
        down_conv1 = nn.Conv2d(down_conv1_in, down_conv2_in,
                               kernel_size=3, stride=1, padding=3)
        # ? change the same occurence ?
        down_conv2 = nn.Conv2d(down_conv2_in, down_conv2_in,
                               kernel_size=3, stride=1, padding=3)
        up_norm = nn.InstanceNorm2d  # TODO: replace with PixNorm
        # FIXME
        up_conv1 = nn.Conv2d(in_channels=up_conv1_in,
                             out_channels=up_conv1_out, kernel_size=3, stride=1)
        up_conv2 = nn.ConvTranspose2d(
            kernel_size=4, stride=2, in_channels=up_conv1_out, out_channels=up_conv2_out, padding=1)
        self.down = [down_conv1, LReLU, down_conv2, LReLU]
        self.up = [up_conv1, LReLU, up_conv2,
                   up_norm(up_conv2_out), LReLU]  # TODO: position of pixel normalization
        if innermost:
            model = self.up
        else:
            model = [submodule] + self.up

        self.model = nn.Sequential(*model)

    def forward(self, x_dm, x_ds):
        x_dm = self.down(x_dm)
        x_ds = self.down(x_ds)
        if self.deepBlender or self.innermost:
            # * cat both output
            return torch.cat([x_ds, x_dm, self.model(torch.cat([self.maxpool(x_dm), self.maxpool(x_ds)])), -1])
        else:
            # FIXME: cat dimension
            return torch.cat([x_ds, self.model(self.maxpool(x_dm), self.maxpool(x_ds))], -1)


class PixelNormLayer(nn.Module):  # can be replaced with IN
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__
