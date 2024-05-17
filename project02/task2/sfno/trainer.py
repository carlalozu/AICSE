"""Trainer for the FNO1d model."""
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from neuralop.datasets import load_spherical_swe
from fno02d import FNO2d
from utils import LpLoss

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer():
    """Trainer for the FNO3d model."""

    def __init__(self, n_train, modesx=32, modesy=32, width=64):
        self.n_train = n_train
        self.training_set, self.testing_set = self.assemble_datasets()

        self.fno = FNO2d(modesx, modesy, width)  # model
        # self.fno = FNO3d(n_modes=(32, 32), in_channels=3, out_channels=3,
        # hidden_channels=32, projection_channels=64, factorization='dense')
        self.fno = self.fno.to(device)

    def assemble_datasets(self):
        """Load the data and prepare the datasets."""

        training_set, testing_set = load_spherical_swe(
            n_train=self.n_train,
            batch_size=10,
            train_resolution=(32, 64),
            test_resolutions=[(32, 64), (64, 128)],
            n_tests=[50, 50],
            test_batch_sizes=[10, 10],
        )

        data_in = []
        data_out = []
        for i, data in enumerate(training_set.dataset):
            if i >= len(training_set.dataset):
                break
            data_in.append(data['x'])
            data_out.append(data['y'])

        data_in = torch.stack(data_in).permute(0, 3, 2, 1).transpose(1,2)
        data_out = torch.stack(data_out).permute(0, 3, 2, 1).transpose(1,2)

        training_set_ = DataLoader(TensorDataset(data_in, data_out),
                                  batch_size=4, shuffle=False)

        testing_set_ = {}
        for resolution in [(32, 64), (64, 128)]:
            data_in = []
            data_out = []
            for i, data in enumerate(testing_set[resolution].dataset):
                if i >= len(testing_set[resolution].dataset):
                    break
                data_in.append(data['x'])
                data_out.append(data['y'])

            data_in = torch.stack(data_in).permute(0, 3, 2, 1).transpose(1,2)
            data_out = torch.stack(data_out).permute(0, 3, 2, 1).transpose(1,2)

            testing_set_[resolution] = DataLoader(TensorDataset(data_in, data_out),
                                                batch_size=10, shuffle=False)

        return training_set_, testing_set_

    def plot_inputs(self, idx_sample=0):
        """Plot the input and output functions."""
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        fig.suptitle('Inputs and ground-truth output')

        for inputs, outputs in self.training_set:

            inputs = inputs[idx_sample].numpy()
            outputs = outputs[idx_sample].numpy()

            ax[0].imshow(inputs[:,:,0])
            ax[0].set_title('Input x')
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            ax[1].imshow(outputs[:,:,0])
            ax[1].set_title('Ground-truth y')
            ax[1].set_xticks([])
            ax[1].set_yticks([])

            plt.tight_layout()
            plt.show()
            break

    def train(self, epochs, learning_rate=8e-4):
        """Train the model."""

        optimizer = Adam(self.fno.parameters(),
                         lr=learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)

        loss = LpLoss(d=2, p=2, reduction=True)
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
                for resolution in [(32, 64), (64, 128)]:
                    test_relative_ = 0.0
                    for input_batch, output_batch in self.testing_set[resolution]:

                        output_pred_batch = self.fno(input_batch).squeeze(2)
                        loss_f = self.error(output_pred_batch, output_batch)
                        test_relative_ += loss_f.item()
                    test_relative_ /= len(self.testing_set)
                    test_relative_l2 += test_relative_

            if epoch % freq_print == 0:
                print("######### Epoch:", epoch, " ######### Train Loss:",
                      train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)

    @staticmethod
    def error(y, y_, p=2):
        """Relative L2 error."""
        err = (torch.mean(abs(y.detach().reshape(-1, ) - y_.detach(
        ).reshape(-1, )) ** p) / torch.mean(abs(y.detach()) ** p)) ** (1 / p)
        return err


    def plot(self, idx_sample=0):
        """Plot results"""
        fig = plt.figure(figsize=(8, 4))
        for index, resolution in enumerate([(32, 64), (64, 128)]):
            for inputs, outputs in self.testing_set[resolution]:
                output_pred_batch = self.fno(inputs).squeeze(2)
                output_pred_batch = output_pred_batch.detach().cpu()

                output_pred_batch = output_pred_batch[idx_sample].numpy()
                inputs = inputs[idx_sample].numpy()
                outputs = outputs[idx_sample].numpy()


                ax = fig.add_subplot(2, 3, index*3 + 1)
                ax.imshow(inputs[:,:,0])
                ax.set_title(f'Input x {resolution}')
                plt.xticks([], [])
                plt.yticks([], [])

                ax = fig.add_subplot(2, 3, index*3 + 2)
                ax.imshow(outputs[:,:,0])
                ax.set_title('Ground-truth y')
                plt.xticks([], [])
                plt.yticks([], [])

                ax = fig.add_subplot(2, 3, index*3 + 3)
                ax.imshow(output_pred_batch[:,:,0])
                ax.set_title('Model prediction')
                plt.xticks([], [])
                plt.yticks([], [])

                break

        fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
        plt.tight_layout()
        fig.show()
