import torch
import torch.nn as nn


class ConcatTupleLayer(nn.Module):
    """
    concatenate tuple(torch.Tensor, torch.Tensor) into torch.Tensor
    where tuple(torch.Tensor, torch.Tensor) is (x_l, x_g)
    """
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')
