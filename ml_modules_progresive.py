import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from math import sqrt


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None,
                 pixel_norm=False, batch_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]

        if batch_norm:
            convs.append(nn.BatchNorm2d(out_channel))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.ReLU())
        convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))

        if batch_norm:
            convs.append(nn.BatchNorm2d(out_channel))


        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.ReLU())

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)




class EncoderProgressive(nn.Module):
    def __init__(self, args, in_channels=1, global_avg_pool=True, batch_norm=True):
        super().__init__()
        feature=1024
        self.global_avg_pool=global_avg_pool
        self.in_channels=in_channels
        self.progression = nn.ModuleList([
                                          ConvBlock(feature//64,feature//32, 3, 1, batch_norm=batch_norm),
                                          ConvBlock(feature//32 ,feature//16, 3, 1,  batch_norm=batch_norm),
                                          ConvBlock(feature//16, feature//8, 3, 1,  batch_norm=batch_norm),
                                          ConvBlock(feature//8, feature//4, 3, 1,  batch_norm=batch_norm),
                                          ConvBlock(feature//4, feature//2, 3, 1,  batch_norm=batch_norm),
                                          ConvBlock(feature//2, feature, 3, 1,  batch_norm=batch_norm)])

        self.from_rgb = nn.ModuleList([EqualConv2d(1, feature//64, 1),
                                       EqualConv2d(1, feature//32, 1),
                                       EqualConv2d(1,  feature//16, 1),
                                       EqualConv2d(1, feature//8, 1),
                                       EqualConv2d(1, feature//4, 1),
                                       EqualConv2d(1, feature//2, 1)])

        self.n_layer = len(self.progression)

    def forward(self, input):
        scale=self.scale
        alpha=self.alpha
        step=scale-3
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1
            if i == step:
                out = self.from_rgb[index](input)
            out = self.progression[index](out)
            out = torch.nn.AvgPool2d(kernel_size=2)(out)
            if i == step and 0 <= alpha < 1:
                skip_rgb = torch.nn.AvgPool2d(kernel_size=2)(input)
                skip_rgb = self.from_rgb[index + 1](skip_rgb)
                out = (1 - alpha) * skip_rgb + alpha * out
        if self.global_avg_pool:
            out = torch.nn.AvgPool2d((out.shape[2], out.shape[3]))(out)
        return out

    def get_out_shape(self, h=None, w=None):
        if hasattr(self, "scale"):
            scale_save=self.scale
            memory_scale=True
        else:
            memory_scale=False
        if hasattr(self, "alpha"):
            alpha_save=self.alpha
            memory_alpha=True
        else:
            memory_alpha=False
        self.scale=5
        self.alpha=1
        out= self.forward(torch.rand(1, self.in_channels, 32, 32)).shape[1:]

        if memory_scale:
            self.scale=scale_save
        if memory_alpha:
            self.alpha=alpha_save
        return out