"""
original: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
"""
import collections
import functools
import logging
import torch
import numpy as np
import torch.nn as nn

from collections import defaultdict
from functools import partial
from modules.models.base import BaseDiscriminator
from modules.models.MultiDialatedConv import MultidilatedConv
from modules.models.FFC import FFC_BN_ACT
from modules.models.utils import ConcatTupleLayer


class NLayerDiscriminator(BaseDiscriminator):
    """
    PatchGAN Discriminator
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        # pad to keep the size of feature map
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf

        # each time half the size of feature map(done by stride of 2) and double the number of channels
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        # get final feature map(with channel of 1) to discriminate every element
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        # act consists of outputs of every activation
        return act[-1], act[:-1]


class MultidilatedNLayerDiscriminator(BaseDiscriminator):
    """
    PatchGAN Discriminator with MultidilatedConv
    The same as NLayerDiscriminator except MultidilatedConv
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, multidilation_kwargs=None):
        if multidilation_kwargs is None:
            multidilation_kwargs = {}
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf

        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                MultidilatedConv(nf_prev, nf, kernel_size=kw, stride=2, padding=[2, 3], **multidilation_kwargs),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class FFCNLayerDiscriminator(BaseDiscriminator):
    """
    TODO: Purpose of FFCNLayerDiscriminator?
    None of the Network Use this as discriminator
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
                 init_conv_kwargs=None, conv_kwargs=None):
        if init_conv_kwargs is None:
            init_conv_kwargs = {}
        if conv_kwargs is None:
            conv_kwargs = {}

        super().__init__()

        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
                                activation_layer=_act_ctor, **init_conv_kwargs)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)

            # half the size of feature map and double the number of channels
            cur_model = [FFC_BN_ACT(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer,
                                    activation_layer=_act_ctor, **conv_kwargs)]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = [
            FFC_BN_ACT(nf_prev, nf,
                       kernel_size=kw, stride=1, padding=padw,
                       norm_layer=norm_layer,
                       activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
                       **conv_kwargs),
            ConcatTupleLayer()
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return act[-1], feats
