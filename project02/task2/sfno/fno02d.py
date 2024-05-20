"""Fourier Neural Operator for parametrized PDEs."""
from torch import nn
import torch.nn.functional as F

from spectral_conv import SpectralConv2d
from spherical_conv import SphericalConv2d


class FNO2d(nn.Module):
    """The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .

    input: the solution of the coefficient function and locations (a(x, y), x, y)
    input shape: (batchsize, x=s, y=s, c=3)
    output: the solution 
    output shape: (batchsize, x=s, y=s, c=1)
    """

    def __init__(self, modes1, modes2,  width):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.padding = 9  # pad the domain if input is non-periodic
        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def __str__(self) -> str:
        return "FNO2d"

    def forward(self, x):
        """Forward pass of the network."""
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class SFNO2d(FNO2d):
    """The overall network. It contains 4 layers of the Spherical harmonics layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .

    input: the solution of the coefficient function and locations (a(x, y), x, y)
    input shape: (batchsize, x=s, y=s, c=3)
    output: the solution 
    output shape: (batchsize, x=s, y=s, c=1)
    """

    def __init__(self, modes1, modes2,  width):
        super().__init__(modes1, modes2,  width)

        self.conv0 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)

    def __str__(self) -> str:
        return "SFNO2d"
