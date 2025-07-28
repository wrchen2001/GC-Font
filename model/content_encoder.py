import torch.nn as nn
from functools import partial
from model.modules.blocks import ConvBlock, AdaConvBlock, ResBlock, ResAdaConvBlock

"""
    Basic Content Encoder module used during RQ-VAE pretraining phase.
    This is the original content encoder implementation without attention integration.
"""
# class ContentEncoder(nn.Module):
#     """
#     ContentEncoder
#     """

#     def __init__(self, layers, sigmoid=False):
#         super().__init__()
#         self.net = nn.Sequential(*layers)
#         self.sigmoid = sigmoid

#     def forward(self, x):
#         out = self.net(x)
#         if self.sigmoid:
#             out = nn.Sigmoid()(out)
#         return out
    

"""
    Enhanced content encoder module used during main model training phase.
    Note: This should be loaded after RQ-VAE pretraining is complete.
"""
class ContentEncoder(nn.Module):
    """
    ContentEncoder
    """

    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):

        outputs = []
        for layer in self.net:
            x = layer(x)
            outputs.append(x)
        if self.sigmoid:
            outputs[-1] = nn.Sigmoid()(outputs[-1])
        return outputs


def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', pad_type='reflect', content_sigmoid=False):
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    AdaConvBlk = partial(AdaConvBlock, norm = norm, activ = activ, pad_type = pad_type)

    layers = [
        AdaConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'),
        AdaConvBlk(C * 1, C * 2, 3, 1, 1, downsample=True),  # 64x64
        AdaConvBlk(C * 2, C * 4, 3, 1, 1, downsample=True),  # 32x32
        ConvBlk(C * 4, C * 8, 3, 2, 1),
        ConvBlk(C * 8, C_out, 3, 1, 1)
    ]

    return ContentEncoder(layers, content_sigmoid)
