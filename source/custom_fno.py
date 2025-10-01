import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from neuralop.models import FNO


class CustomFNO(FNO):
    def __init__(self, *args, **kwargs):
        super(CustomFNO, self).__init__(*args, **kwargs)
        cnn = [nn.Conv1d, nn.Conv2d, nn.Conv3d][self.n_dim - 1]
        self.padding = partial(F.pad, pad=(1, 1)*self.n_dim, mode='circular')
        self.cnn_layers = nn.ModuleList([cnn(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=0) for _ in range(2 * self.n_layers)])

    def forward(self, x, output_shape=None, **kwargs):
        """
        1. Applies optional positional encoding
        2. Sends inputs through a lifting layer to a high-dimensional latent space
        3. Applies optional domain padding to high-dimensional intermediate function representation
        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity)
        5. If domain padding was applied, domain padding is removed
        6. Projection of intermediate function representation to the output channels
        """
        # x is shape (batch, channels, x, y, ...)

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x
