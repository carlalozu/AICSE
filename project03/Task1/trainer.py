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
        self.model = NNAnsatz(1, 1, 1, 32)
        self.pendulum = pendulum

    def loss(self, states):
        """Residual of the state of the pendulum"""
        thetas = states[:, 2]
        theta_dots = states[:, 3]

        # our goal is to stabilize the pendulum at 3/4s of the steps
        steps = len(thetas)
        tau = int(3*steps/4)

        # Angle with respect to the vertical position
        angle_loss = torch.mean(torch.sin(thetas[tau:]/2)**2)

        # Angular velocity loss
        angular_velocity_loss = torch.mean((theta_dots[tau:]) ** 2)

        return angle_loss + 0.1 * angular_velocity_loss

    def train(self, initial_state, dt, epochs, steps, lr=1e-3, step_size=100, gamma=0.1):
        """Trains the model"""
        hist_train = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

        for i in range(epochs):
            state = initial_state.detach().clone().requires_grad_(True)

            states = []
            optimizer.zero_grad()
            for step in range(steps):
                external_force = self.model(step*dt*torch.ones(1))[0]
                new_state = self.pendulum.RK_step(state, external_force, dt)
                states.append(new_state)
                state = new_state

            states = torch.stack(states)
            loss = self.loss(states)
            loss.backward()
            optimizer.step()
            scheduler.step()

            hist_train.append(loss.item())
            if i % 10 == 0:
                print(f"##### Epoch: {i}, Loss: {loss.item()}")

        return hist_train

    def test(self, initial_state, dt, steps):
        """Tests the model"""
        states = []
        external_forces = []
        state = initial_state.detach().clone()
        for step in range(steps):
            external_force = self.model(step*dt*torch.ones(1))[0]
            state = self.pendulum.RK_step(state, external_force, dt)
            states.append(state)
            external_forces.append(external_force)

        return torch.stack(states).detach(), torch.stack(external_forces).detach()

    def plot_loss_function(self, hist_train):
        """Plot the loss function normalized in log scale."""
        hist_train = np.array(hist_train)

        fig = plt.figure(dpi=100, figsize=(7, 4))
        plt.plot(np.arange(1, len(hist_train)+1),
                 hist_train, label="Train", linewidth=2)
        plt.grid(True)
        # plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return fig
