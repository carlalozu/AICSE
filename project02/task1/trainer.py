"""Trainer for the FNO1d model."""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd

from fno01d import FNO1d

torch.manual_seed(0)
np.random.seed(0)


class Trainer():
    """Trainer for the FNO1d model."""

    def __init__(self, n_train, modes=16, width=64, batch_size=10):
        self.n_train = n_train
        self.training_set, self.testing_set = self.assemble_datasets(
            batch_size)

        self.fno = FNO1d(modes, batch_size)  # model



    def assemble_datasets(self, batch_size=10):
        """Load the data and prepare the datasets."""

        # Load the data
        data = pd.read_csv('TrainingData.txt', sep=',')

        x_data = torch.tensor(data['t'], dtype=torch.float32).reshape(-1, 1)
        y_data = torch.tensor(data[['tf0', 'ts0']].values, dtype=torch.float32)

        self.input_function_train = x_data[:self.n_train]
        self.output_function_train = y_data[:self.n_train, :]

        self.input_function_test = x_data[self.n_train:]
        self.output_function_test = y_data[self.n_train:, :]

        training_set = DataLoader(TensorDataset(
            self.input_function_train, self.output_function_train),
            batch_size=batch_size, shuffle=True)
        testing_set = DataLoader(TensorDataset(
            self.input_function_test, self.output_function_test),
            batch_size=batch_size, shuffle=False)

        return training_set, testing_set

    def plot_inputs(self):
        """Plot the input and output functions."""
        plt.figure()
        plt.plot(
            self.input_function_train,
            self.output_function_train[:,0],
            label="Training Data Fluid Phase")
        plt.plot(
            self.input_function_train,
            self.output_function_train[:,1],
            label="Training Data Solid Phase")
        plt.grid(True, which="both", ls=":")
        plt.legend()

    def train(self, epochs, learning_rate, step_size, gamma):
        """Train the model."""

        optimizer = Adam(self.fno.parameters(),
                         lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

        loss = torch.nn.MSELoss()
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
        input_function_test_n = self.input_function_test
        output_function_test_n = self.output_function_test
        # print(input_function_test_n.shape)
        # print(output_function_test_n.shape)

        output_function_test_pred_n = self.fno(input_function_test_n)
        # print(output_function_test_pred_n.shape)
        # print(input_function_test_n[0,:,1])
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.plot(
            input_function_test_n,
            output_function_test_n[0:,0],
            label="True Solution", c="C0", lw=2)
        plt.scatter(
            input_function_test_n,
            output_function_test_pred_n[:,0],
            label="Approximate Solution", s=8, c="C1")

        err = self.error(output_function_test_n, output_function_test_pred_n)
        print("Relative L2 error: ", err.item())
        plt.legend()
