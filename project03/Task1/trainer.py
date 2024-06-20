"""Trainer for the external Force function"""

import matplotlib.pyplot as plt
from neural_net import NNAnsatz
import torch
from torch.optim import Adam
import numpy as np


class Trainer():
    """Trainer class for the external force function"""

    def __init__(self, pendulum):

        self.model = NNAnsatz(5, 1, 2, 32)
        self.optimizer = Adam(self.model.parameters(), lr=1e-2)
        self.pendulum = pendulum

    def loss(self, states):
        """Residual of the angle"""
        theta = states[:, 3]
        return torch.mean(torch.abs(theta))

    def train(self, initial_state, initial_force, dt, iterations):
        """Trains the model"""
        hist_train = []
        external_force = initial_force
        for _ in range(iterations):
            states = self.pendulum.rollout(initial_state, external_force, dt)
            states.requires_grad = True
            model_input = torch.stack(
                (states[:, 0], states[:, 1], states[:, 2], states[:, 3], external_force.reshape(-1,))).T
            external_force = self.model(model_input)

            loss = self.loss(states)
            hist_train.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Loss: {loss.item()}")

        return hist_train

    def plot_loss_function(self, hist_train):
        """Plot the loss function normalized in log scale."""
        hist_train = np.array(hist_train)

        fig = plt.figure(dpi=100, figsize=(7, 4), frameon=False)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist_train)+1),
                 hist_train, label="Train", linewidth=2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss_function.pdf')
        plt.show()
        return fig
