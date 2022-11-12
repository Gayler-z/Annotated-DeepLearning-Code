import torch.nn as nn
from modules.models.FFC import FFC_BN_ACT, FFCResnetBlock
from modules.models.SpatialTransform import LearnableSpatialTransformWrapper
from modules.models.utils import ConcatTupleLayer
from modules.models.utils import get_activation


class FFCResNetGenerator(nn.Module):
    """
    Generator with FFC ResNet
    TODO: consider StyleGAN, use skip connection to replace residual connection in generator
    """
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d,
                 up_activation=nn.ReLU(True), init_conv_kwargs=None, downsample_conv_kwargs=None,
                 resnet_conv_kwargs=None, spatial_transform_layers=None, spatial_transform_kwargs=None,
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs=None):
        if init_conv_kwargs is None:
            init_conv_kwargs = {}
        if downsample_conv_kwargs is None:
            downsample_conv_kwargs = {}
        if resnet_conv_kwargs is None:
            resnet_conv_kwargs = {}
        if spatial_transform_kwargs is None:
            spatial_transform_kwargs = {}
        if out_ffc_kwargs is None:
            out_ffc_kwargs = {}

        assert (n_blocks >= 0)
        super().__init__()

        # ngf: number of generator's input/output channels(output(ngf channels) will be converted to RGB)
        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            # FFC Convolution with stride of 2 to downsample
            # each time downsample, double number of channels
            model += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2),
                      kernel_size=3, stride=2, padding=1, norm_layer=norm_layer,
                      activation_layer=activation_layer, **cur_conv_kwargs)]

        # number of FFC ResNet's input channels
        mult = 2 ** n_downsampling
        # Is this a bottleneck structure?(bottleneck is in the SpectralTransform)
        feats_num_bottleneck = min(max_features, ngf * mult)

        # resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer, norm_layer=norm_layer,
                                          **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        # default format of output of ResNet Blocks is tuple(x_l, x_g) which needs to be converted to torch.Tensor for
        # operation afterwards
        model += [ConcatTupleLayer()]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # Transposed Convolution(along with normalization and activation) with stride of 2 to upsample
            # each time upsample, half number of channels
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        # TODO: purpose of out_ffc layer?
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        # generate final output(maybe RGB)
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # whether to add final activation(default: False)
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)