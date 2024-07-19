import jax
import jax.numpy as jnp

import equinox as eqx
import optax

import numpy as np
import matplotlib.pyplot as plt

BETA = 0.2
GAMMA = 0.1


class Trainer():
    """Trainer for the Lotka-Volterra equations using JAX"""

    def __init__(self):
        self.X_obs = self.assemble_datasets()
        print(self.X_obs.shape)

        # initial conditions
        self.N = self.X_obs.shape[0]-1
        self.dt = 0.1
        self.x0 = self.X_obs[0]

    def assemble_datasets(self):
        """Get the observed data"""
        return jnp.array(np.load("data/X_obs.npy"))

    def plot_inputs(self):
        "Plots Lotka-Volterra solution"
        plt.figure()
        plt.scatter(jnp.arange(self.X_obs.shape[0]), self.X_obs[:,0], alpha=0.8, label="prey")
        plt.scatter(jnp.arange(self.X_obs.shape[0]), self.X_obs[:,1], alpha=0.8, label="predator")
        plt.ylabel("Population")
        plt.xlabel("Time step")
        plt.legend()

    @staticmethod
    def f(state, model, delta):
        """New RHS including neural network"""
        x, y = state[0:1], state[1:2]
        return jnp.concatenate([model(x) - BETA*x*y, GAMMA*x*y-delta*y], axis=0)

    def ode_solver(self, model, delta):
        """Generic 4th order explicit RK ODE solver. f(x, model, delta) is a function
        which computes the RHS of the ODE using a NN."""

        def runge_kutta4(state, i):
            "4th order Runge-Kutta ODE solver"
            k1 = self.f(state, model, delta)
            k2 = self.f(state + 0.5*self.dt*k1, model, delta)
            k3 = self.f(state + 0.5*self.dt*k2, model, delta)
            k4 = self.f(state + self.dt*k3, model, delta)
            state = state + self.dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

            return state, state

        _, x = jax.lax.scan(runge_kutta4, self.x0, jnp.arange(self.N))
        x = jnp.concatenate([self.x0.reshape((1, -1)), x], axis=0)

        return x

    def loss(self, fargs):
        "Computes mean squared error between model prediction and training data"
        X_pred = self.ode_solver(*fargs)
        loss = jnp.mean((X_pred - self.X_obs)**2)
        return loss

    def grad(self, fargs):
        "Computes gradient of loss function wrt fargs"
        return eqx.filter_value_and_grad(self.loss)(fargs)

    @eqx.filter_jit
    def step(self, fargs, opt_state, opt_update):
        "Performs one gradient descent step on fargs"
        lossval, grads = self.grad(fargs)
        updates, opt_state = opt_update(grads, opt_state)
        fargs = eqx.apply_updates(fargs, updates)
        return lossval, fargs, opt_state

    def train(self, model, delta, lr=1e-2, n_steps=2000):
        """Training loop for Lotka-Volterra NDE"""
        fargs = (model, delta)

        # define optimiser
        optimiser = optax.adam(learning_rate=lr)

        opt_state = optimiser.init(eqx.filter(fargs, eqx.is_array))
        opt_update = optimiser.update

        # train NDE
        lossvals = []
        for i in range(n_steps):
            lossval, fargs, opt_state = self.step(
                fargs, opt_state, opt_update)
            lossvals.append(lossval)
            if (i+1) % 500 == 0 or i == 0:
                print(f"[{i+1}/{n_steps}] loss: {lossval}, delta:{fargs[1]}")

        return fargs

    def plot(self, model, delta, show=True):
        "Plots Lotka-Volterra solution"
        if show:
            self.plot_inputs()
        X = self.ode_solver(model, delta)
        plt.plot(X[:, 0], label='Prey predicted')
        plt.plot(X[:, 1], label='Predator predicted')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()

    @staticmethod
    def plot_learned_model(model):
        """Plots the learned model against the true function"""
        x = jnp.linspace(0.5,6,1000).reshape(-1,1)
        plt.figure()
        plt.plot(x, jax.vmap(model)(x), label="learned function")
        plt.plot(x, jax.nn.tanh(x), label="true function")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("value")
        plt.show()
