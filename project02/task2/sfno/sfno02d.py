"""Fourier Neural Operator for parametrized PDEs."""
from torch import nn
import torch.nn.functional as F
from spherical_conv import SphericalConv2d


class SFNO2d(nn.Module):
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
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SphericalConv2d(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)
                     ).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)
                     ).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)
                     ).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)
                     ).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
