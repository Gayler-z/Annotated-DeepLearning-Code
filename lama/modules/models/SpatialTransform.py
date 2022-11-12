import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate


class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            # 1. spatial transform on input feature maps
            # 2. operate on the transformed feature maps(FFC Block)
            # 3. undo spatial transform
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        # (batch, c, h, w)
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        # (batch, c, h + 2 * pad_h, w + 2 * pad_w)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        # rotate
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h: y_height - pad_h, pad_w: y_width - pad_w]
        return y
