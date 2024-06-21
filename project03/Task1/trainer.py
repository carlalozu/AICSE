"""Trainer for the external Force function"""

import matplotlib.pyplot as plt
from neural_net import NNAnsatz
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


class Trainer():
    """Trainer class for the external force function"""

    def __init__(self, pendulum):

        self.model = NNAnsatz(4, 1, 4, 32)
        self.pendulum = pendulum

    def loss(self, state):
        """Residual of the state of the pendulum"""
        x, x_dot, theta, theta_dot = state

        # Combine the losses
        return 10*torch.sin(theta)**2 + theta_dot**2

    @staticmethod
    def norm_state(state):
        """Normalize the state"""
        # Normalize the state
        state_normed = state
        state_normed = (state_normed - state_normed.mean(dim=0)
                        ) / state_normed.std(dim=0)
        return state_normed

    def train(self, initial_state, dt, epochs, steps, lr=1e-3):
        """Trains the model"""
        hist_train = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(epochs):
            state = initial_state.detach().clone().requires_grad_(True)
            loss_sum = 0

            for step in range(steps):
                optimizer.zero_grad()

                # Normalize the state
                state_normed = self.norm_state(state)

                # Get new force
                external_force = self.model(state_normed)[0]
                new_state = self.pendulum.RK_step(state, external_force, dt)

                loss = self.loss(new_state)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                state = new_state.detach()

            loss_sum /= steps
            hist_train.append(loss_sum)
            if i % 10 == 0:
                print(f"##### Epoch: {i}, Loss: {loss_sum}")

        return hist_train

    def test(self, initial_state, dt, steps):
        """Tests the model"""
        states = []
        external_forces = []
        state = initial_state.detach().clone()
        for _ in range(steps):
            state_normed = self.norm_state(state)

            external_force = self.model(state_normed)[0]
            state = self.pendulum.RK_step(state, external_force, dt)
            states.append(state)
            external_forces.append(external_force)

        return torch.stack(states).detach(), torch.stack(external_forces).detach()

    def plot_loss_function(self, hist_train, save=False):
        """Plot the loss function normalized in log scale."""
        hist_train = np.array(hist_train)

        fig = plt.figure(dpi=100, figsize=(7, 4))
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist_train)+1),
                 hist_train, label="Train", linewidth=2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig('loss_function.pdf')
        plt.show()
        return fig
