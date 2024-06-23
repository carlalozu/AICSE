"""Inverted pendulum solved using JAX autodiff"""

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
sns.set()

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

        if theta < 0:
            theta = theta + 2*torch.pi

        if theta > 2*torch.pi:
            theta = theta - 2*torch.pi

        # cart acceleration
        x_dot_dot_num = force - self.m*self.g * \
            torch.cos(theta)*torch.sin(theta) + self.m * \
            self.l * theta_dot ** 2 * torch.sin(theta)
        x_dot_dot_den = self.M + self.m*(1 - torch.cos(theta) ** 2)
        x_dot_dot = x_dot_dot_num / x_dot_dot_den

        # angular acceleration
        theta_dot_dot = (self.g * torch.sin(theta) -
                         x_dot_dot*torch.cos(theta)) / self.l

        return torch.stack((x_dot, x_dot_dot, theta_dot, theta_dot_dot))

    def RK_step(self, state, force, dt):
        """Returns the next state using 4th order Runge Kutta method"""
        k1 = self.dynamics(state, force)
        k2 = self.dynamics(state + dt/2*k1, force)
        k3 = self.dynamics(state + dt/2*k2, force)
        k4 = self.dynamics(state + dt*k3, force)
        return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def rollout(self, initial_state, external_force, dt):
        """Returns the states of a rollout"""
        state = initial_state.detach().clone()
        states = []
        for force in external_force:
            state = self.RK_step(state, force, dt)
            states.append(state)

        return torch.stack(states)

    def animate(self, states, dt):
        """Animates the pendulum"""
        fig, ax = plt.subplots( figsize=(3, 3))
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True)
        # Axis on y only on integer values
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))

        # time
        time = torch.arange(0, len(states)*dt, dt)

        cart = plt.Rectangle((-0.1, -0.05), 0.2, 0.1, fill=True)
        pendulum, = ax.plot([], [], '-', lw=1.5, color='tab:orange')

        def init():
            # Initial axis lims
            ax.set_xlim(-2, 2)
            ax.set_title('Time: 0s')

            ax.add_patch(cart)
            pendulum.set_data([], [])
            return cart, pendulum

        def update(frame):
            ax.set_title(f'Time: {time[frame]:.2f} s')
            x, _, theta, _ = states[frame]
            y = 0
            # Follow the cart
            ax.set_xlim(x - 1.5, x + 1.5)
            cart.set_xy((x - 0.1, y - 0.05))
            pendulum.set_data([x, x + self.l*torch.sin(theta)],
                              [y, self.l*torch.cos(theta)])
            return cart, pendulum

        anim = FuncAnimation(fig, update, frames=len(
            states), init_func=init, blit=True)
        return anim

    @staticmethod
    def plot(states, force, dt, line=False):
        """Plots the relevant variables of the pendulum"""
        x = states[:, 0]
        x_dot = states[:, 1]
        theta = states[:, 2]
        theta_dot = states[:, 3]

        # Steps
        time = torch.arange(0, len(states)*dt, dt)

        plt.figure()
        fig, axs = plt.subplots(3, 1, figsize=(7, 6))  # frameon=False)

        # Cart variables
        axs[0].set_title('Cart', loc='left')
        axs[0].plot(time, x, label=r'$x$', zorder=3)
        axs[0].set_ylabel('Position (m)', color='tab:blue')
        axs[0].tick_params(axis='y', labelcolor='tab:blue')

        # Second y-axis for velocity but delete grid
        ax2 = axs[0].twinx()
        ax2.plot(time, x_dot, label=r'$\dot{x}$', color='tab:orange', zorder=2)
        ax2.set_ylabel('Velocity (m/s)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.grid(False)

        # Pendulum variables
        axs[1].set_title('Pendulum', loc='left')
        axs[1].plot(time, theta, label=r'$\theta$')
        axs[1].set_ylabel('Angle (rad)', color='tab:blue')
        axs[1].set_ylim(0, 2*torch.pi+0.2)
        axs[1].tick_params(axis='y', labelcolor='tab:blue')

        if line:
            # vertical line at 3/4 of the plot in axis 
            axs[1].axvline(x=3*time[-1]/4, color='k', linestyle='--')

        # Second y-axis for angular velocity
        ax3 = axs[1].twinx()
        ax3.plot(time, theta_dot, label=r'$\dot{\theta}$', color='tab:orange')
        ax3.set_ylabel('Angular Velocity (rad/s)', color='tab:orange')
        ax3.tick_params(axis='y', labelcolor='tab:orange')
        ax3.grid(False)

        # External force
        axs[2].set_title('External Force', loc='left')
        axs[2].plot(time, force)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Force (N)', color='tab:blue')
        axs[2].tick_params(axis='y', labelcolor='tab:blue')


        fig.tight_layout()
        plt.show()
        return fig
