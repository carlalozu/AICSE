"""Module with NN"""
import torch
from torch import nn

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


class NeuralNet(nn.Module):
    """Neural Net architecture"""

    def __init__(self,
                input_dimension,
                output_dimension,
                n_hidden_layers,
                neurons,
                retrain_seed):
        super().__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function, we only want positive values
        self.activation = nn.Tanh()

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed

        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        """The forward function performs the set of affine and non-linear
        transformations defining the network"""

        ##############
        # Implement forward pass through the network
        x = self.input_layer(x)
        x = self.activation(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        ##############
        return x

    def init_xavier(self):
        """Xavier initialization"""
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if isinstance(type(m), nn.Linear) and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)
