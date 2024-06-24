"""Neural net"""
import torch


class NNAnsatz(torch.nn.Module):
    """Feed-forward neural net."""

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 n_hidden_layers,
                 hidden_size):
        """Setup your layers."""
        super().__init__()
        # Define a list to store the hidden layers
        self.hidden_layers = torch.nn.ModuleList()

        # Add the input layer
        self.hidden_layers.append(
            torch.nn.Linear(input_dimension, hidden_size))
        self.hidden_layers.append(torch.nn.SiLU())

        # Add the hidden layers
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(torch.nn.SiLU())

        # Add the output layer
        self.output_layer = torch.nn.Linear(hidden_size, output_dimension)

    def forward(self, x):
        """Do a forward pass on `x`."""
        # Pass the input through all hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Pass through the output layer
        x = self.output_layer(x)

        return x
