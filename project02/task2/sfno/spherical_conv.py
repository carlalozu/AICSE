"""Spectral (or Fourier) layer in 1d"""
import torch
from torch import nn
import torch_harmonics as th


class SphericalConv2d(nn.Module):
    """2D Spherical Convolutional Layer.
    Implemented using the RealSHT and InverseRealSHT modules from
    https://github.com/NVIDIA/torch-harmonics.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels,
                self.modes1, self.modes2,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels,
                self.modes1, self.modes2,
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, inputs, weights):
        """Complex multiplication"""
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x):
        """Implement the forward method using the torch harmonics from NVidia."""
        batchsize = x.shape[0]

        # forward and inverse data transform on an equiangular grid
        sht = th.RealSHT(x.size(-2), x.size(-1), grid="equiangular")
        isht = th.InverseRealSHT(x.size(-2), x.size(-1), grid="equiangular")

        # Spherical harmonic transform
        x_ft = sht(x)

        # Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        return isht(out_ft)
