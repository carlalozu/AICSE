"""Trainer for the FNO1d model."""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import LpLoss
from window_generator import WindowGenerator
from fno01d import FNO1d

sns.set(style="white")
torch.manual_seed(0)
np.random.seed(0)


class Trainer():
    """Trainer for the FNO1d model."""

    def __init__(self, modes=16, width=64, batch_size=10, window_length=34):
        """Initialize the trainer."""

        self.batch_size = batch_size
        self.window_length_in = window_length
        self.window_length_out = window_length

        self.scaler_tf0 = MinMaxScaler()
        self.scaler_ts0 = MinMaxScaler()
        self.scaler_time = MinMaxScaler()

        self.training_set, self.testing_set = self.assemble_datasets()
        self.fno = FNO1d(modes, width)  # model

    def load_data(self):
        """Load the data and normalize it."""
        # Retrieve data
        data = pd.read_csv('../TrainingData.txt', sep=',')

        # Normalize the data using min-max scaling
        data['tf0'] = self.scaler_tf0.fit_transform(
            data['tf0'].values.reshape(-1, 1))
        data['ts0'] = self.scaler_ts0.fit_transform(
            data['ts0'].values.reshape(-1, 1))
        data['t'] = self.scaler_time.fit_transform(
            data['t'].values.reshape(-1, 1))

        # Transform data
        x_data = torch.tensor(data[['tf0', 'ts0', 't']].values, dtype=torch.float32)[
            :-self.window_length_out, :]
        y_data = torch.tensor(data[['tf0', 'ts0']].values, dtype=torch.float32)[
            self.window_length_in:, :]

        return x_data, y_data

    def assemble_datasets(self):
        """Prepare the datasets, without splitting into train and test sets."""

        x_data, y_data = self.load_data()

        window_generator = WindowGenerator(
            x_data, y_data,
            self.window_length_in,
            self.window_length_out,
            shift=1, stride=1
        )

        inputs, outputs = zip(*window_generator)

        data_train, data_test, targets_train, targets_test = train_test_split(
            torch.stack(list(inputs)),
            torch.stack(list(outputs)),
            test_size=0.2, shuffle=True,
        )

        self.input_function_train = data_train
        self.output_function_train = targets_train

        self.input_function_test = data_test
        self.output_function_test = targets_test

        training_set = DataLoader(
            TensorDataset(data_train, targets_train),
            batch_size=self.batch_size, shuffle=False)
        testing_set = DataLoader(
            TensorDataset(data_test, targets_test),
            batch_size=self.batch_size, shuffle=False)

        return training_set, testing_set

    def plot_inputs(self):
        """Plot the input and output functions."""

        x_data, y_data = self.load_data()
        y_axis = x_data[:len(y_data[:, 1]), 2] + \
            x_data[self.window_length_in, 2]

        plt.figure()
        plt.plot(x_data[:, 2], x_data[:, 0],
                 label="inputs for fluid phase")

        plt.plot(x_data[:, 2], x_data[:, 1],
                 label="inputs for solid phase")

        plt.plot(y_axis, y_data[:, 0],
                 label="outputs for fluid phase", linestyle='--')

        plt.plot(y_axis, y_data[:, 1],
                 label="outputs for solid phase", linestyle='--')

        plt.grid(True, which="both", ls=":")
        plt.xlabel('Time increments')
        plt.ylabel('Temperature T(0,t)')
        plt.title('Fluid phase, x=0')
        plt.legend()

    def train(self, epochs, learning_rate, step_size, gamma):
        """Train the model."""

        optimizer = Adam(self.fno.parameters(),
                         lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

        loss = LpLoss(size_average=False)
        freq_print = 1
        hist_train, hist_test = [], []
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
            hist_train.append(train_mse)

            scheduler.step()

            with torch.no_grad():
                self.fno.eval()
                test_relative_l2 = 0.0
                for input_batch, output_batch in self.testing_set:
                    output_pred_batch = self.fno(input_batch).squeeze(2)
                    loss_f = self.relative_lp_norm(output_pred_batch, output_batch)
                    test_relative_l2 += loss_f.item()
                test_relative_l2 /= len(self.testing_set)
                hist_test.append(test_relative_l2)

            if epoch % freq_print == 0:
                print("######### Epoch:", epoch, " ######### Train Loss:",
                      train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)

        return hist_train, hist_test

    @staticmethod
    def relative_lp_norm(y, y_, p=2):
        """Relative Lp norm."""
        err = (torch.mean(abs(y.reshape(-1, ) - y_.detach(
        ).reshape(-1, )) ** p) / torch.mean(abs(y.detach()) ** p)) ** (1 / p) * 100
        return err

    def plot(self, idx_=-1):
        """Plot results."""
        for input_batch, output_batch in self.testing_set:

            if idx_ == -1:
                range_idx = range(input_batch.shape[0])
            else:
                range_idx = [idx_]

            for idx in range_idx:
                plt.figure()
                output_pred_batch = self.fno(input_batch).detach().numpy()

                x_ax_output = input_batch[idx, -1, 2] + \
                    input_batch[idx, :, 2]-input_batch[idx, 0, 2]

                plt.plot(input_batch[idx, :, 2], input_batch[idx, :, 0])
                plt.plot(x_ax_output, output_batch[idx, :, 0])
                plt.plot(x_ax_output, output_pred_batch[idx, :, 0],
                         label='predicted ft', linestyle='--')

                plt.plot(input_batch[idx, :, 2], input_batch[idx, :, 1])
                plt.plot(x_ax_output, output_batch[idx, :, 1])
                plt.plot(x_ax_output, output_pred_batch[idx, :, 1],
                         label='predicted fs', linestyle='--')

                plt.legend()


    def plot_loss_function(self, hist_train, hist_test):
        """Function to plot the loss function"""
        hist_train = np.array(hist_train)
        hist_test = np.array(hist_test)

        plt.figure(dpi=100, figsize=(7, 4), frameon=False)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist_train)+1), hist_train/hist_train[0], label = "Train")
        plt.plot(np.arange(1, len(hist_test)+1), hist_test/hist_test[0], label = "Test")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
