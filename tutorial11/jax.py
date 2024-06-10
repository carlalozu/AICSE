import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import diffrax as dfx
import equinox as eqx
from equinox import nn
import matplotlib.pyplot as plt
import numpy as np


class Jax():

    def __init__(self):
        self.x_batch, self.y_label_batch = self.assemble_datasets()
        forward_batch = None  # TODO
        grad = None  # TODO
        jit_step = None  # TODO

    def assemble_datasets(self):
        # TODO
        key = jr.key(0)
        x_batch = jnp.linspace(-1, 1, 100).reshape((100, 1))
        y_label_batch = 3*x_batch**3 - x_batch**2 - 3 * \
            x_batch + 2 + 0.2*jax.random.normal(key, (100, 1))
        return x_batch, y_label_batch

    def forward(theta, x):
        "Returns model prediction, for a single example input"
        # TODO
        return y

    def loss(theta, x_batch, y_label_batch):
        "Computes mean squared error between model prediction and training data"
        # TODO

        return loss

    def step(lrate, theta, x_batch, y_label_batch):
        "Performs one gradient descent step on model parameters, given training data"
        # TODO

        return theta, lossval

    def plot(self):
        plt.scatter(self.x_batch, self.y_label_batch, label="training data")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
