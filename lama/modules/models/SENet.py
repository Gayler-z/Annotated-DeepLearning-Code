import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze and Excitation Block Layer
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # calculate gate(sigmoid)
        y = self.fc(y).view(b, c, 1, 1)
        # pass through partial feature map(according to gate)
        res = x * y.expand_as(x)
        return res