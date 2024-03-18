"""Neural net"""
import torch


class NNAnsatz(torch.nn.Module):
    """Feed-forward neural net."""

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 n_hidden_layers,  # TODO: how to incorporate hidden layers?
                 hidden_size):
        """Setup your layers."""
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, output_dimension),
        )

    def forward(self, x):
        """Do a forward pass on `x`."""
        return self.layers(x)