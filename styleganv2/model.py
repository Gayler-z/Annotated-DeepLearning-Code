import math
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class Smooth(nn.Module):
    """
    layer blur each channel
    """
    def __init__(self):
        super().__init__()
        # blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # to tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # normalize
        kernel /= kernel.sum()
        # fixed parameter
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        # x: (batch * in_channels, 1, h, w)
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)


class UpSample(nn.Module):
    # https://arxiv.org/pdf/1904.11486.pdf
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smmoth(self.up_sample(x))


class DownSample(nn.Module):
    # https://arxiv.org/pdf/1904.11486.pdf
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        """
        shape is the shape of the weight parameter
        """
        super().__init__()
        
        # HeKaiming initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        # return initialization with HeKaiming constant
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_channels, in_channels, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_channels))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_channels, in_channels, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        # x: input feature map (batch, in_channels, height, width)
        # s: style based scaling tensor (batch, in_channels)
        b, _, h, w = x.shape
        # s: (batch, in_channels) -> (batch, 1, in_channels, 1, 1) broadcast
        s = s[:, None, :, None, None]
        # weights: (1, out_channels, in_channels, kernel_size, kernel_size)
        weights = self.weight()[None, :, :, :, :]
        # modulate
        # 对每个 sample，卷积核权重都不一样
        # weights: (batch, out_channels, in_channels, kernel_size, kernel_size)
        weights = weights * s
        # demodulate
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv
        # x: (1, batch * in_channels, h, w)
        x = x.reshape(1, -1, h, w)
        # ws: (in_channels, kernel_size, kernel_size)
        _, _, *ws = weights.shape
        # weights: (batch * out_channels, in_channels, kernel_size, kernel_size)
        weights = weights.reshape(b * self.out_features, *ws)
        # 用分组卷积计算更高效（每个 sample 的卷积核参数都不一样）
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.reshape(-1, self.out_channels, h, w)


class ToRGB(nn.Module):
    def __init__(self, d_latent: int, in_channels: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_channels, bias=1.0)
        self.conv = Conv2dWeightModulate(in_channels, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        # generate RGB image using 1*1 convolution
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # LayerNorm
        z = F.normalize(z, dim=1)
        return self.net(z)


class StyleBlock(nn.Module):
    def __init__(self, d_latent: int, in_channels: int, out_channels: int):
        super().__init__()
        # style vector from w
        self.to_style = EqualizedLinear(d_latent, in_channels, bias=1.0)
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        # modulate and demodulate convolution
        s = self.to_style(w)
        x = self.conv(x, s)
        # inject noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class GeneratorBlock(nn.Module):
    def __init__(self, d_latent: int, in_channels: int, out_channels: int):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_channels, out_channels)
        self.style_block2 = StyleBlock(d_latent, out_channels, out_channels)
        self.to_rgb = ToRGB(d_latent, out_channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        x = self.styleblock1(x, w, noise[0])
        x = self.styleblock2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb


class Generator(nn.Module):
    def __init__(self, log_resolution: int, d_latent: int, n_channels: int = 32, max_channels: int = 512):
        """
        :param log_resolution:
        :param d_latent:
        :param n_channels: 最后一个 block 的通道数
        :param max_channels:
        """
        super().__init__()
        # number of channels for each block
        # e.g. [512, 512, 256, 128, 64, 32]
        channels = [min(max_channels, n_channels * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(channels)
        # learned constant
        self.initial_constant = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        # first style block for 4*4 resolution and toRGB layer
        self.style_block = StyleBlock(d_latent, channels[0], channels[0])
        self.to_rgb = ToRGB(d_latent, channels[0])
        # generator blocks
        blocks = [GeneratorBlock(d_latent, channels[i - 1], channels[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        :param w: (n_blocks, batch_size, d_latent), n_blocks for style mixing
        :param input_noise: each block has two noise inputs(except initial layer)
        :return:
        """

        batch_size = w.shape[1]
        # 常量输入复制 batch_size 份
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            # skip connections
            rgb = self.up_sample(rgb) + rgb_new
        return rgb


class MiniBatchStdDev(nn.Module):
    """
    计算一个 minibacth 的所有特征的标准差，然后取均值
    """
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        assert x.shape[0] % self.group_size == 0
        # 分组
        grouped = x.view(self.group, -1)
        # 算标准差
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([x, std], dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_channels, out_channels, kernel_size=1))
        self.block = nn.Sequential(
            EqualizedConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.down_sample = DownSample()
        # 残差连接会导致 sqrt(2) 倍标准差，缩放回去（resnet 有 BatchNorm 所以问题不大）
        self.scale = 1 / math.sqrt(2)

    def forward(self, x: torch.Tensor):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class Discriminator(nn.Module):
    def __init__(self, log_resolution: int, n_channels: int = 64, max_channels: int = 512):
        """
        :param log_resolution:
        :param n_channels: 第一个 block 的通道数
        :param max_channels:
        """
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_channels, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # e.g. [64, 128, 512, 512, 512]
        channels = [min(max_channels, n_channels * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(channels) - 1
        blocks = [DiscriminatorBlock(channels[i], channels[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.std_dev = MiniBatchStdDev()
        final_channels = channels[-1] + 1
        # 得到最后特征图再卷积一次
        self.conv = EqualizedConv2d(final_channels, final_channels, 3)
        # 展平，全连接
        self.final = EqualizedLinear(2 * 2 * final_channels, 1)

    def forward(self, x: torch.Tensor):
        # optional: normalize
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)


class GradientPenalty(nn.Module):
    # https://arxiv.org/pdf/1801.04406.pdf
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        :param x: x~D
        :param d: D(x)
        :return:
        """
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float):
        """
        :param beta: exponential moving average coefficient
        """
        super().__init__()
        self.beta = beta
        # number of steps
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        :param w: (batch, d_latent)
        :param x: (batch, 3, h, w)
        :return:
        """
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)
        # g(w) * y and normalize
        output = (x * y).sum() / math.sqrt(image_size)
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)
        # L2 norm
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()
        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)
        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)
        return loss
