"""Inverted pendulum solved using JAX autodiff"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter

class InvertedPendulum:
    """Inverted pendulum class"""

    def __init__(self, mass_cart=1.0, mass_pendulum=0.1, length=1.0, gravity=9.81):
        self.M = mass_cart      # kg
        self.m = mass_pendulum  # kg
        self.l = length         # m
        self.g = gravity        # m/s^2

    def dynamics(self, state, force):
        """Returns the state derivative"""
        _, x_dot, theta, theta_dot = state

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

    def RK_step(self, state, force, dt):
        """Returns the next state using 4th order Runge Kutta method"""
        k1 = self.dynamics(state, force)
        k2 = self.dynamics(state + dt/2*k1, force)
        k3 = self.dynamics(state + dt/2*k2, force)
        k4 = self.dynamics(state + dt*k3, force)
        return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def rollout(self, initial_state, external_force, dt):
        """Returns the states of a rollout"""
        states = [initial_state]

        for force in external_force:
            state = self.RK_step(states[-1], force, dt)
            states.append(state)

        return states

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
        theta = [frame[2] for frame in states]
        theta_dot = [frame[3] for frame in states]

        plt.figure()
        fig, axs = plt.subplots(3, 1, figsize=(7, 6)) # frameon=False)
        
        axs[0].set_title('Cart', loc='left')
        axs[0].plot(x_cart, label=r'$x$')
        axs[0].set_ylabel('Position (m)', color='tab:blue')
        axs[0].tick_params(axis='y', labelcolor='tab:blue')

        # Second y-axis for velocity
        ax2 = axs[0].twinx()
        ax2.plot(x_dot, label=r'$\dot{x}$', color='tab:orange')
        ax2.set_ylabel('Velocity (m/s)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        axs[1].set_title('Pendulum', loc='left')
        axs[1].plot(theta, label=r'$\theta$')
        axs[1].set_ylabel('Angle (rad)', color='tab:blue')
        axs[1].tick_params(axis='y', labelcolor='tab:blue')

        # Second y-axis for angular velocity
        ax3 = axs[1].twinx()
        ax3.plot(theta_dot, label=r'$\dot{\theta}$', color='tab:orange')
        ax3.set_ylabel('Angular Velocity (rad/s)', color='tab:orange')
        ax3.tick_params(axis='y', labelcolor='tab:orange')

        axs[2].set_title('External Force', loc='left')
        axs[2].plot(force)
        axs[2].set_xlabel('Timesteps')
        axs[2].set_ylabel('Force (N)', color='tab:blue')
        axs[2].tick_params(axis='y', labelcolor='tab:blue')

        fig.tight_layout()
        plt.show()
        return fig
