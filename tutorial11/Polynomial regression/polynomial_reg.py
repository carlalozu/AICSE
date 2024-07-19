"""Polynomial regression class using JAX"""
import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt


class PolynomialRegression():
    """Polynomial regression class using JAX and JIT compilation"""

    def __init__(self):
        self.x_batch, self.y_label_batch = self.assemble_datasets()
        self.forward_batch = jax.vmap(self.forward, in_axes=(None, 0))
        self.grad = jax.value_and_grad(self.loss, argnums=0)
        self.jit_step = jax.jit(self.step)

    def assemble_datasets(self):
        """Generates test data for polynomial regression"""
        key = jr.key(0)
        x_batch = jnp.linspace(-1, 1, 100).reshape((100, 1))
        y_label_batch = 3*x_batch**3 - x_batch**2 - 3 * \
            x_batch + 2 + 0.2*jax.random.normal(key, (100, 1))
        return x_batch, y_label_batch

    @staticmethod
    def forward(theta, x):
        "Returns model prediction, for a single example input"
        y = theta[0]*x**3 + theta[1]*x**2 + theta[2]*x + theta[3]
        return y

    def loss(self, theta, x_batch, y_label_batch):
        "Computes mean squared error between model prediction and training data"
        y_pred = self.forward_batch(theta, x_batch)
        loss = jnp.mean((y_pred - y_label_batch)**2)
        return loss

    def step(self, lrate, theta, x_batch, y_label_batch):
        "Performs one gradient descent step on model parameters, given training data"
        # Compute gradient using jax.value_and_grad
        lossval, dldt = self.grad(theta, x_batch, y_label_batch)

        # Update theta
        theta = jax.tree_util.tree_map(lambda t, dt: t-lrate*dt, theta, dldt)

        return theta, lossval

    def gradient_descent(self, theta, lrate, nsteps):
        """Performs gradient descent on model parameters"""
        for i in range(nsteps):
            theta, lossval = self.jit_step(lrate, theta,
                                           self.x_batch, self.y_label_batch)
            print(f"Step {i}, Loss: {lossval}")

        return theta

    def plot(self, theta=None):
        """Plots training data"""
        plt.scatter(self.x_batch, self.y_label_batch, label="training data")
        if theta is not None:
            x = self.x_batch
            y = self.forward_batch(theta, x)
            plt.plot(x, y, color="tab:orange", lw=3, label="Model prediction")
            print(f"theta={theta}")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
