import torch.nn as nn


# Source: DepthWiseSeperableConv
class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        if 'groups' in kwargs:
            # ignoring groups for Depthwise Separable Conv
            del kwargs['groups']

        self.depthwise = nn.Conv2d(in_dim, in_dim, *args, groups=in_dim, **kwargs)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # depth wise convolution(each input feature map is convoluted by only one kernel so as the kernel)
        # depth wise convolution lacks communication between channels
        out = self.depthwise(x)
        # communicate between channels(by vanilla 1x1 convolution)
        out = self.pointwise(out)
        return out
