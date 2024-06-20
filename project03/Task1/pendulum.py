"""Inverted pendulum solved using JAX autodiff"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class InvertedPendulum:
    """Inverted pendulum class"""

    def __init__(self, mass_cart=1.0, mass_pendulum=0.1, length=1.0, gravity=9.81):
        self.M = mass_cart      # kg
        self.m = mass_pendulum  # kg
        self.l = length         # m
        self.g = gravity        # m/s^2

    def dynamics(self, state, action):
        """Returns the state derivative"""
        _, x_dot, theta, theta_dot = state

        # force
        force = jnp.clip(action, -10.0, 10.0)

        # cart acceleration
        x_dot_dot_num = force - self.m*self.g * \
            jnp.cos(theta)*jnp.sin(theta) + self.m * \
            self.l * theta_dot ** 2 * jnp.sin(theta)
        x_dot_dot_den = self.M + self.m*(1 - jnp.cos(theta) ** 2)
        x_dot_dot = x_dot_dot_num / x_dot_dot_den

        # angular acceleration
        theta_dot_dot = (self.g * jnp.sin(theta) -
                         x_dot_dot*jnp.cos(theta)) / self.l

        return jnp.array([x_dot, x_dot_dot, theta_dot, theta_dot_dot])

    def step(self, state, action, dt):
        """Returns the next state"""
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + dt/2*k1, action)
        k3 = self.dynamics(state + dt/2*k2, action)
        k4 = self.dynamics(state + dt*k3, action)
        return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def cost(self, state, action):
        """Returns the cost of the state-action pair"""
        x, _, theta, _ = state
        return x ** 2 + 0.1*theta ** 2 + 0.001*action ** 2
    
    def rollout(self, initial_state, actions, dt):
        """Returns the states and costs of a rollout"""
        states = [initial_state]
        costs = []
        for action in actions:
            state = self.step(states[-1], action, dt)
            cost = self.cost(states[-1], action)

            states.append(state)
            costs.append(cost)
        return states, costs

    def animate(self, states):
        """Animates the pendulum"""
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid()
        cart = plt.Rectangle((-0.1, -0.05), 0.2, 0.1, fill=True)
        pendulum, = ax.plot([], [], 'r-', lw=2)

        def init():
            ax.add_patch(cart)
            pendulum.set_data([], [])
            return cart, pendulum

        def update(frame):
            x, _, theta, _ = states[frame]
            y = 0

            cart.set_xy((x - 0.1, y - 0.05))
            pendulum.set_data([x, x + self.l*jnp.sin(theta)],
                              [y, self.l*jnp.cos(theta)])
            return cart, pendulum

        anim = FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True)
        return anim

    @staticmethod
    def plot(states, force):
        x_cart = [frame[0] for frame in states]
        x_dot = [frame[1] for frame in states]
        theta = [frame[2]*180/jnp.pi for frame in states]
        theta_dot = [frame[3]*180/jnp.pi for frame in states]

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(x_cart, label=r'$x$')
        axs[0].plot(x_dot, label=r'$\dot{x}$')
        axs[0].set_ylabel('Position (m)')
        axs[0].legend()
        axs[0].set_title('Cart', loc='left')

        axs[1].plot(theta, label=r'$\theta$')
        axs[1].plot(theta_dot, label=r'$\dot{\theta}$')
        axs[1].set_ylabel('Angle (deg)')
        axs[1].legend()
        axs[1].set_title('Pendulum', loc='left')

        axs[2].plot(force)
        axs[2].legend()
        axs[2].set_title('External Force')

        fig.tight_layout()
        plt.show()
