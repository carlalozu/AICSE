"""Spectral (or Fourier) layer in 1d"""
import torch
import torch_harmonics as th
from spectral_conv import SpectralConv2d


class SphericalConv2d(SpectralConv2d):
    """2D Spherical Convolutional Layer.
    Implemented using the RealSHT and InverseRealSHT modules from
    https://github.com/NVIDIA/torch-harmonics.
    """

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
