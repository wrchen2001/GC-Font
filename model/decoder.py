from functools import partial
import torch
import torch.nn as nn
from .modules import ConvBlock, ResBlock, AdaConvBlock, ResAdaConvBlock

class Integrator(nn.Module):
    def __init__(self, C, norm='none', activ='none', C_content=0, C_reference=0):
        super().__init__()
        C_in = C + C_content + C_reference  
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps, content=None, reference=None):
        """
        Args:
            comps [B, 3, mem_shape]: component features
        """
        if reference==None:  
            inputs = torch.cat((comps, content), 1)
            out = self.integrate_layer(inputs)
            return out
        else:    
            inputs = torch.cat((comps, content, reference), 1)
            out = self.integrate_layer(inputs)
            return out


"""
    Basic Decoder module used during RQ-VAE pretraining phase.
    This is the original decoder implementation without attention integration.
"""
# class Decoder(nn.Module):
#     def __init__(self, layers, skips=None, out='sigmoid'):
#         super().__init__()
#         self.layers = nn.ModuleList(layers)

#         if out == 'sigmoid':
#             self.out = nn.Sigmoid()
#         elif out == 'tanh':
#             self.out = nn.Tanh()
#         else:
#             raise ValueError(out)

#     def forward(self, x):
#         """
#         forward
#         """
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#         return self.out(x)


"""
    Enhanced Decoder module used during main model training phase.
    This version integrates attention features at early layers (0-2).
    Note: This should be loaded after RQ-VAE pretraining is complete.
"""

class Decoder(nn.Module):
    def __init__(self, layers, skips=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x, atten):
        """
        forward
        """
        for i, layer in enumerate(self.layers):  
            if i==0 or i==1 or i==2:
                x = layer(x)
                x = x + atten
            else:
                x = layer(x)
        return self.out(x)


def dec_builder(C, C_out, norm='none', activ='relu', out='sigmoid'):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ)
    ResBlk = partial(ResBlock, norm=norm, activ=activ)
    AdaConvBlk = partial(AdaConvBlock, norm = norm, activ = activ)
    ResAdaConvBlk = partial(ResAdaConvBlock, norm = norm, activ = activ)

    layers = [
        ResBlk(C * 8, C * 8, 3, 1),
        ResBlk(C * 8, C * 8, 3, 1),
        ResBlk(C * 8, C * 8, 3, 1),
        ConvBlk(C * 8, C * 4, 3, 1, 1, upsample=True),  # 32x32
        ConvBlk(C * 4, C * 2, 3, 1, 1, upsample=True),  # 64x64
        ConvBlk(C * 2, C * 1, 3, 1, 1, upsample=True),  # 128x128
        ConvBlk(C * 1, C_out, 3, 1, 1)
    ] 


    return Decoder(layers, out=out)



