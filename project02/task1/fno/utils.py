"""Utils for the tutorial05."""
from torch import nn


def activation(name):
    """Activation function factory."""
    name = name.lower()
    activation_functions = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(inplace=True),
        'lelu': nn.LeakyReLU(inplace=True),
        'sigmoid': nn.Sigmoid(),
        'softplus': nn.Softplus(beta=4),
        'celu': nn.CELU(),
        'elu': nn.ELU(),
        'mish': nn.Mish(),
    }
    if name in activation_functions:
        return activation_functions[name]

    raise ValueError('Unknown activation function')
