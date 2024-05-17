"""Trainer for the FNO1d model."""
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from neuralop.datasets import load_spherical_swe
from fno02d import FNO2d
from utils import LpLoss

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer():
    """Trainer for the FNO3d model."""

    def __init__(self, n_train, modes=32, width=64):
        self.n_train = n_train
        self.training_set, self.testing_set = self.assemble_datasets()

        self.fno = FNO2d(modes, modes, width)  # model
        # self.fno = FNO3d(n_modes=(32, 32), in_channels=3, out_channels=3,
        # hidden_channels=32, projection_channels=64, factorization='dense')
        self.fno = self.fno.to(device)

    def assemble_datasets(self):
        """Load the data and prepare the datasets."""

        training_set, testing_set = load_spherical_swe(
            n_train=200,
            batch_size=4,
            train_resolution=(32, 64),
            test_resolutions=[(32, 64), (64, 128)],
            n_tests=[50, 50],
            test_batch_sizes=[10, 10],
        )
        return training_set, testing_set

    def plot_inputs(self):
        """Plot the input and output functions."""
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        fig.suptitle('Inputs and ground-truth output')

        test_samples = self.training_set.dataset
        data = test_samples[0]

        # Input x
        x = data['x'][0, ...].detach().numpy()
        # Ground-truth
        y = data['y'][0, ...].numpy()

        ax[0].imshow(x)
        ax[0].set_title('Input x')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(y)
        ax[1].set_title('Ground-truth y')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()
        plt.show()

    def train(self, epochs, learning_rate=8e-4):
        """Train the model."""

        optimizer = Adam(self.fno.parameters(),
                         lr=learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)

        loss = LpLoss(d=2, p=2, reduce_dims=(0, 1))
        freq_print = 1
        for epoch in range(epochs):
            train_mse = 0.0
            for input_batch, output_batch in self.training_set:
                optimizer.zero_grad()
                output_pred_batch = self.fno(input_batch).squeeze(2)
                loss_f = loss(output_pred_batch, output_batch)
                loss_f.backward()
                optimizer.step()
                train_mse += loss_f.item()
            train_mse /= len(self.training_set)

            scheduler.step()

            with torch.no_grad():
                self.fno.eval()
                test_relative_l2 = 0.0
                for input_batch, output_batch in self.testing_set:
                    output_pred_batch = self.fno(input_batch).squeeze(2)
                    loss_f = self.error(output_pred_batch, output_batch)
                    test_relative_l2 += loss_f.item()
                test_relative_l2 /= len(self.testing_set)

            if epoch % freq_print == 0:
                print("######### Epoch:", epoch, " ######### Train Loss:",
                      train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)

    @staticmethod
    def error(y, y_, p=2):
        """Relative L2 error."""
        err = (torch.mean(abs(y.detach().reshape(-1, ) - y_.detach(
        ).reshape(-1, )) ** p) / torch.mean(abs(y.detach()) ** p)) ** (1 / p) * 100
        return err

    def plot(self):
        """Plot results"""
        fig = plt.figure(figsize=(7, 7))
        for index, resolution in enumerate([(32, 64), (64, 128)]):
            test_samples = self.testing_set[resolution].dataset
            data = test_samples[0]
            # Input x
            x = data['x']
            # Ground-truth
            y = data['y'][0, ...].numpy()
            # Model prediction
            x_in = x.unsqueeze(0).to(device)
            out = self.fno(x_in).squeeze()[0, ...].detach().cpu().numpy()
            x = x[0, ...].detach().numpy()

            ax = fig.add_subplot(2, 3, index*3 + 1)
            ax.imshow(x)
            ax.set_title(f'Input x {resolution}')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(2, 3, index*3 + 2)
            ax.imshow(y)
            ax.set_title('Ground-truth y')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(2, 3, index*3 + 3)
            ax.imshow(out)
            ax.set_title('Model prediction')
            plt.xticks([], [])
            plt.yticks([], [])

        fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
        plt.tight_layout()
        fig.show()
