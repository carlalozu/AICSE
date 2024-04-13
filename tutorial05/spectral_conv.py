"""Spectral (or Fourier) layer in 1d"""
import torch
from torch import nn


class SpectralConv1d(nn.Module):
    """Spectral Convolution Layer in 1d"""

    def __init__(self, in_channels, out_channels, modes1):
        """1D Fourier layer. It does FFT, linear transform, and Inverse FFT."""
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, inputs, weights):
        """Complex multiplication"""
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", inputs, weights)

    def forward(self, x):
        """Forward pass of the Fourier layer"""

        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]

        ##########################################
        # TO DO: Implement the forward method by:
        ##########################################
        # 1) Compute Fourier coefficients
        # 2) Multiply relevant Fourier modes
        # 3) Transform the data to physical space
        # HINT: Use torch.fft library torch.fft.rfft

        # Compute Fourier coefficients

        # Multiply relevant Fourier modes

        # Return to physical space

        return x
