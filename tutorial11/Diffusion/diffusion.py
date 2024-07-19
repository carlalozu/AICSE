"""Diffusion model class and training code"""
import jax
import jax.numpy as jnp
import jax.random as jr

import optax
import equinox as eqx
import diffrax as dfx

import numpy as np
import matplotlib.pyplot as plt
from minst import MNIST


class DiffusionModel():
    "Diffusion model class trainer"

    def __init__(self):
        self.data = self.load_data()
        self.x_0_shape = self.data[0].shape  # x_0_shape=(1, 28, 28)

    def load_data(self):
        "Get the observed data"
        mnist = MNIST()
        mnist.normalize()
        return mnist.data

    def loss(self, model, int_beta, t1, x_0, t, key):
        "Loss value for a single x_0"

        mu = jnp.exp(-0.5 * int_beta(t))  # mean
        sigma_2 = jnp.maximum(1 - jnp.exp(-int_beta(t)),
                              1e-5)  # for numerical stability
        sigma = jnp.sqrt(1 - jnp.exp(-int_beta(t)))  # std

        # sample z ~ N(0, 1) noise
        z = jr.normal(key, x_0.shape)

        score = model(mu*x_0 + sigma * z, t, t1)

        # compute grad log p(x_t|x_0)
        glp = - z / sigma

        # compute loss
        loss = sigma_2 * jnp.mean((score - glp)**2)

        return loss

    def loss_batch(self, model, int_beta, t1, x_0_batch, key):
        "Compute loss over batch of x_0, using random t for each example"

        batch_size = x_0_batch.shape[0]
        key, subkey = jr.split(key)

        # sample t values with low-discrepancy
        t_batch = jr.uniform(subkey, (batch_size,),
                             minval=0, maxval=t1 / batch_size)
        t_batch = t_batch + (t1 / batch_size) * jnp.arange(batch_size)

        # evaluate losses
        key_batch = jr.split(key, batch_size)
        loss_batch = jax.vmap(self.loss, in_axes=(
            None, None, None, 0, 0, 0))(
            model, int_beta, t1, x_0_batch, t_batch, key_batch)

        return jnp.mean(loss_batch)

    @eqx.filter_jit
    def make_step(self, opt_state, opt_update, model, *fargs):
        "Carries out a gradient descent step to train score model"

        grad = eqx.filter_value_and_grad(self.loss_batch)
        lossval, grads = grad(model, *fargs)
        updates, opt_state = opt_update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return lossval, model, opt_state

    @eqx.filter_jit
    def single_sample_fn(self, model, int_beta, t1, x_0_shape, dt0, key):
        "Sample diffusion model (solve NDE backwards in time)"

        def f(t, x, args):
            _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
            return -0.5 * beta * (x + model(x, jnp.array(t), t1))

        term = dfx.ODETerm(f)
        solver = dfx.Tsit5()
        x_1 = jr.normal(key, x_0_shape)
        sol = dfx.diffeqsolve(term, solver, t1, 0, -dt0, x_1)
        x_0 = sol.ys[0]
        return x_0

    def train(self, model, int_beta, t1, dt0, key, n_steps=10000, batch_size=64):
        "Train the diffusion model"
        # define optimiser
        optimiser = optax.adam(learning_rate=3e-4)
        opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
        opt_update = optimiser.update

        # training loop
        lossvals = []
        for i in range(n_steps):

            # sample batch
            key, key1, key2 = jr.split(key, 3)
            x_0_batch = self.data[jr.randint(
                key1, (batch_size,), 0, self.data.shape[0])]

            keys_test = jax.random.split(key2, 4*4)

            # make update step
            lossval, model, opt_state = self.make_step(
                opt_state, opt_update, model, int_beta, t1, x_0_batch, key2)
            lossvals.append(lossval)

            # print/plot results
            if (i+1) % 2000 == 0 or i == 0:
                print(f"[{i+1}/{n_steps}] loss: {lossval}")
            if (i+1) % 10000 == 0:
                x_0s = jax.vmap(self.single_sample_fn, in_axes=(None, None, None, None, None, 0))(
                    model, int_beta, t1, self.x_0_shape, dt0, keys_test)
                self.plot_results(lossvals, x_0s)

    @staticmethod
    def plot_results(lossvals, x_0s):
        "Plots results of training"
        plt.figure()
        plt.plot(np.array(lossvals))
        plt.yscale("log")
        plt.ylabel("Loss")
        plt.xlabel("Training step")
        plt.show()
        plt.figure(figsize=(8, 8))
        for i in range(4*4):
            plt.subplot(4, 4, i+1)
            plt.imshow(x_0s[i, 0])
            plt.axis("off")
        plt.show()

    @staticmethod
    def plot_p(t1, int_beta):
        "Plots p(x_t|x_0) mean and variance"
        t_ = jnp.linspace(0, t1, 1000)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("$p(x_t|x_0)$ mean")
        plt.plot(t_, jnp.exp(-0.5 * int_beta(t_)))
        plt.subplot(1, 2, 2)
        plt.title("$p(x_t|x_0)$ variance")
        plt.plot(t_, 1 - jnp.exp(-int_beta(t_)))
        plt.show()
