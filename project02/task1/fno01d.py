"""Fourier Neural Operator for parametrized PDEs."""
import torch
from torch import nn

from spectral_conv import SpectralConv1d


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1D PDEs."""

    def __init__(self, modes, width):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later time step
        output shape: (batchsize, x=s, c=1)
        """
        super().__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        # input channel is 2: (u0(x), x) --> GRID IS INCLUDED!
        self.linear_p = nn.Linear(1, self.width)

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        """Implement the Fourier layer"""
        ##########################################
        return spectral_layer(x) + conv_layer(x)
        ##########################################

    def linear_layer(self, x, linear_transformation):
        """Implement the Linear layer"""
        ##########################################
        return linear_transformation(x)
        ##########################################

    def forward(self, x):
        """Implement the forward method using the Fourier 
        and the Linear layer"""
        #################################################
        # Lifting layer
        x = self.linear_layer(x, self.linear_p)
        # x = x.permute(0, 2, 1)
        # 3 layers of the integral operators
        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.activation(x)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.activation(x)
        x = self.fourier_layer(x, self.spect3, self.lin2)
        x = self.activation(x)
        # Projection layer
        # x = x.permute(0, 2, 1)

        x = self.linear_layer(x, self.linear_q)
        x = self.activation(x)
        x = self.output_layer(x)
        #################################################
        return x
