"""Lotka-Volterra"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class LotkaVolterra():
    """Class for solving the Lotka-Volterra equations using JAX"""

    def __init__(self):
        pass

    @staticmethod
    def f(state, alpha, beta, delta, gamma):
        "Computes RHS of Lotka-Volterra NDE"
        x, y = state

        # Returns (x', y')
        return jnp.array([alpha*x - beta*x*y, gamma*x*y-delta*y])

    def ode_solver(self, x0, N, dt, fargs):
        """Generic 4th order explicit RK ODE solver. f(x, *fargs) is a function
        which computes the RHS of the ODE."""

        def runge_kutta4(state, i):
            "4th order Runge-Kutta ODE solver"
            k1 = self.f(state, *fargs)
            k2 = self.f(state + 0.5*dt*k1, *fargs)
            k3 = self.f(state + 0.5*dt*k2, *fargs)
            k4 = self.f(state + dt*k3, *fargs)
            state = state + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

            return state, state

        _, x = jax.lax.scan(runge_kutta4, x0, jnp.arange(N))
        x = jnp.concatenate([x0.reshape((1, -1)), x], axis=0)

        return x

    def plot(self, x):
        "Plots Lotka-Volterra solution"
        plt.plot(x[:, 0], label='Prey')
        plt.plot(x[:, 1], label='Predator')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
