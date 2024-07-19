import jax
import equinox as eqx


class FCN(eqx.Module):
    "Fully-connected neural network, with 1 hidden layer"

    input_layer: eqx.nn.Linear
    hidden_layer: eqx.nn.Linear
    output_layer: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, key):
        "Initialise network parameters"

        key1, key2, key3 = jax.random.split(key, 3)

        self.input_layer = eqx.nn.Linear(in_size, hidden_size, key=key1)
        self.hidden_layer = eqx.nn.Linear(hidden_size, hidden_size, key=key2)
        self.output_layer = eqx.nn.Linear(hidden_size, out_size, key=key3)

    def __call__(self, x):
        "Defines forward model"
        x = jax.nn.tanh(self.input_layer(x))
        x = jax.nn.tanh(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
