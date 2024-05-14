"""Trainer for the FNO1d model."""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd
from window_generator import WindowGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import LpLoss

from fno01d import FNO1d

torch.manual_seed(0)
np.random.seed(0)


class Trainer():
    """Trainer for the FNO1d model."""

    def __init__(self, modes=16, width=64, batch_size=10, window_length=34):

        self.batch_size = batch_size
        self.window_length = window_length

        self.scaler_tf0 = MinMaxScaler()
        self.scaler_ts0 = MinMaxScaler()
        self.scaler_time = MinMaxScaler()

        self.training_set, self.testing_set = self.assemble_datasets()
        self.fno = FNO1d(modes, width)  # model

    def load_data(self):
        """Load the data and normalize it."""
        #Retrieve data
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
            :-self.window_length, :]
        y_data = torch.tensor(data[['tf0', 'ts0']].values, dtype=torch.float32)[
            self.window_length:, :]

        return x_data, y_data

    def assemble_datasets(self):
        """Prepare the datasets, without splitting into train and test sets."""

        x_data, y_data = self.load_data()
        self.input_function_train = x_data
        self.output_function_train = y_data

        self.input_function_test = x_data
        self.output_function_test = y_data

        # Call the WindowGenerator class
        window_generator = WindowGenerator(
            x_data, y_data, self.window_length, self.window_length, 1, 1)
        training_set = DataLoader(
            window_generator, batch_size=self.batch_size, shuffle=True)

        testing_set = DataLoader(
            window_generator, batch_size=self.batch_size, shuffle=True)

        return training_set, testing_set

    def assemble_datasets_with_split(self):
        """Load the data and prepare the datasets."""

        x_data, y_data = self.load_data()

        data_train, data_test, targets_train, targets_test = train_test_split(
            x_data, y_data, test_size=0.3, shuffle=False)

        self.input_function_train = data_train
        self.output_function_train = targets_train

        self.input_function_test = data_test
        self.output_function_test = targets_test

        window_generator = WindowGenerator(
            data_train, targets_train, self.window_length, self.window_length, 1, 1)
        training_set = DataLoader(
            window_generator, batch_size=self.batch_size, shuffle=True)

        window_generator = WindowGenerator(
            data_test, targets_test, self.window_length, self.window_length, 1, 1)
        testing_set = DataLoader(
            window_generator, batch_size=self.batch_size, shuffle=True)

        return training_set, testing_set

    def plot_inputs(self):
        """Plot the input and output functions."""
        plt.figure()
        plt.plot(
            self.input_function_train[:, 2],
            self.input_function_train[:, 0],
            label="train")
        plt.plot(
            self.input_function_test[:, 2],
            self.input_function_test[:, 0],
            label="test inputs")

        plt.plot(
            self.input_function_train[:, 2] +
            self.input_function_train[self.window_length, 2],
            self.output_function_train[:, 0],
            label="train outputs", linestyle='--')

        plt.plot(
            self.input_function_test[:, 2] +
            self.input_function_train[self.window_length, 2],
            self.output_function_test[:, 0],
            label="test outputs", linestyle='--')

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
        err = (torch.mean(abs(y.reshape(-1, ) - y_.detach(
        ).reshape(-1, )) ** p) / torch.mean(abs(y.detach()) ** p)) ** (1 / p)
        return err

    def plot(self, idx=-1):
        for input_batch, output_batch in self.testing_set:
            plt.figure()
            output_pred_batch = self.fno(input_batch)

            x_ax_output = input_batch[idx, -1, 2] + \
                input_batch[idx, :, 2]-input_batch[idx, 0, 2]

            plt.plot(input_batch[idx, :, 2], input_batch[idx, :, 0])
            plt.plot(x_ax_output, output_batch[idx, :, 0])
            plt.plot(
                x_ax_output, output_pred_batch[idx, :, 0].detach().numpy(), label='predicted', linestyle='--')

            plt.plot(input_batch[idx, :, 2], input_batch[idx, :, 1])
            plt.plot(x_ax_output, output_batch[idx, :, 1])
            plt.plot(
                x_ax_output, output_pred_batch[idx, :, 1].detach().numpy(), label='predicted', linestyle='--')

            plt.legend()
