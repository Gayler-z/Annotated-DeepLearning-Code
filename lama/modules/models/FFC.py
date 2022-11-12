"""
Fast Fourier Convolution NeurIPS2020
https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
Modules of FFC, FFC ResNet
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.SENet import SELayer
from modules.models.SpatialTransform import LearnableSpatialTransformWrapper
from modules.models.base import BaseDiscriminator
from modules.models.utils import get_activation


class FourierUnit(nn.Module):
    """
    Constituent of SpectralTransform
    """
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        # (batch, c, h, w)
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            # h, w
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        # Fast Fourier Transform
        # (batch, c, h, w/2+1) complex matrix
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        # (batch, c, h, w/2+1, 2) real matrix(concatenation of complex matrix)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # (batch, 2*c, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # not in Fast Fourier Convolution Paper
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)
        # end of not in Fast Fourier Convolution Paper

        # Convolution in the frequency space
        # (batch, c*2, h, w/2+1)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # (batch, c, 2, h, w/2+1) -> (batch, c, h, w/2+1, 2)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        # split into complex matrix
        # (batch, c, h, w, w/2+1)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # (h, w) if 2D FFT
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]

        # Inverted Fast Fourier Transform
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        # not in Fast Fourier Convolution Paper
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        # end of not in Fast Fourier Convolution Paper

        return output


class SpectralTransform(nn.Module):
    """
    Constituent of FFC
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        # downsample(optional)
        x = self.downsample(x)
        # channel reduction(mimic ResNet BottleNeck Structure)
        x = self.conv1(x)
        # Fourier Unit
        output = self.fu(x)

        # Local Fourier Unit
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            # split size(h == w)
            split_s = h // split_no

            # when h != w
            # split_h = h // split_no
            # split_w = w // split_no

            # split into four parts and concatenate in the channel dimension
            # (batch, c/4, h, w) -> (batch, c/2, h/2, w)
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            # (batch, c/2, h/2, w) -> (batch, c, h/2, w/2)
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            # Fourier Unit
            xs = self.lfu(xs)
            # four copies and concatenate from (h/2, w/2) to　(h, w)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        # channel promotion and residual connection
        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    """
    FFC: consists of l2g, l2l, g2g(Vanilla Dilated Convolution), g2l(SpectralTransform)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        # for downsample
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        # number of global input channels
        in_cg = int(in_channels * ratio_gin)
        # number of local input channels
        in_cl = in_channels - in_cg
        # number of global output channels
        out_cg = int(out_channels * ratio_gout)
        # number of local output channels
        out_cl = out_channels - out_cg

        # group convolution parameter
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # Dilated Convolution
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)

        # SpectralTransform
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):

        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            # generate gate
            gates = torch.sigmoid(self.gate(total_input))
            # divide into 2 parts
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            # pass through all
            g2l_gate, l2g_gate = 1, 1

        # local branch output
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        # global branch output
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    """
    FFC with BatchNorm and Activation, no Residual Connection
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect', enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        # Fast Fourier Convolution
        x_l, x_g = self.ffc(x)
        # BatchNorm and Activation of local part of FFC output
        x_l = self.act_l(self.bn_l(x_l))
        # BatchNorm and Activation of global part of FFC output
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    """
    Residual Block with 2 FFC Convolution
    """
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        # spatial transform on input -> operate on transformed input -> undo spatial transform
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        # maybe inline means input x is torch.Tensor
        # default: input x is tuple
        self.inline = inline

    def forward(self, x):
        # split channel dimension(when inline is True, i.e. x is torch.Tensor)
        # else x is in form of tuple(x_l, x_g) which is default
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        # residual connection
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g

        # concatenate local and global part(when inline is True) and return torch.Tensor
        # else return tuple(default)
        if self.inline:
            out = torch.cat(out, dim=1)
        return out
