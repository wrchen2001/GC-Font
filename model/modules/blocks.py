from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import spectral_norm
import math

# NOTE for nsml pytorch 1.1 docker

class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    

def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = 'none'

        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def norm_dispatch(norm):
    return {
        'none': nn.Identity,
        'in': partial(nn.InstanceNorm2d, affine=False),  # false as default
        'bn': nn.BatchNorm2d,
    }[norm.lower()]

@dispatcher
def w_norm_dispatch(w_norm):
    # NOTE Unlike other dispatcher, w_norm is function, not class.
    return {
        'spectral': spectral_norm,
        'none': lambda x: x
    }[w_norm.lower()]

@dispatcher
def activ_dispatch(activ, norm=None):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
    }[activ.lower()]

@dispatcher
def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d
    }[pad_type.lower()]


class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, n_groups = None):
        super().__init__()

        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1)/2    
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, kernel_size),
                              padding = (math.ceil(padding), math.floor(padding)), padding_mode = "reflect")

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1)/2
        pad = (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding))

        x = F.pad(x, pad = pad, mode = "reflect")
        
        w_spatial = w_spatial.view(self.n_groups, 1, w_spatial.size(-2), w_spatial.size(-1)).to(x.device)   
        x = F.conv2d(x, w_spatial, stride = 1, padding = 0, groups = self.n_groups)
        
        w_pointwise = w_pointwise.view(self.n_groups,1 , 1, 1).to(x.device)
        x = F.conv2d(x, w_pointwise, stride = 1, padding = 0, groups = self.n_groups, bias = bias.to(x.device))

        return x
    

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)  
        
        ys = []
        for i in range(len(x)):   
            y = self._forward_single(x[i:i+1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim = 0)
        ys = self.conv(ys)

        return ys


class AdaConvBlock(nn.Module):
    """ pre-active conv block """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none',
                 activ='relu', bias = True, upsample=False, downsample=False, w_norm='none',
                 pad_type='zero', dropout=0., size=None):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride

        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)

        self.upsample = upsample
        self.downsample = downsample

        self.norm = norm(C_in)
        self.activ = activ()

        if dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(AdaConv2d(C_in, C_out, kernel_size))

    def forward(self, x):
        B, C, H, W = x.shape
        w_spatial = torch.randn(B, C, self.kernel_size, self.kernel_size)
        w_pointwise = torch.randn(B, C, self.stride, self.stride)
        bias = torch.randn(B, C)

        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(x, w_spatial, w_pointwise, bias)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ConvBlock(nn.Module):
    """ pre-active conv block """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none',
                 activ='relu', bias=True, upsample=False, downsample=False, w_norm='none',
                 pad_type='zero', dropout=0., size=None):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample

        self.norm = norm(C_in)
        self.activ = activ()

        if dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """ Pre-activate ResBlock with spectral normalization """
    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False,
                 norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.,
                 scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ,
                               upsample=upsample, w_norm=w_norm, pad_type=pad_type,
                               dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ,
                               w_norm=w_norm, pad_type=pad_type, dropout=dropout)

        if C_in != C_out or upsample or downsample:    
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))   

    
    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x  
        if self.scale_var:
            out = out / np.sqrt(2)
        return out


class ResAdaConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False,
                 norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.,
                 scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = AdaConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ,
                               upsample=upsample, w_norm=w_norm, pad_type=pad_type,
                               dropout=dropout)
        self.conv2 = AdaConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ,
                               w_norm=w_norm, pad_type=pad_type, dropout=dropout)

        if C_in != C_out or upsample or downsample:  
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1)) 

    
    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x  
        if self.scale_var:
            out = out / np.sqrt(2)
        return out


