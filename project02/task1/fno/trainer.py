"""Trainer for the FNO1d model."""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

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

        # Store the data in tensors
        x_data_ = torch.tensor(
            data[['tf0', 'ts0', 't']].values, dtype=torch.float32)

        x_data = x_data_[:-self.window_length_out, :]
        y_data = torch.tensor(data[['tf0', 'ts0']].values, dtype=torch.float32)[
            self.window_length_in:, :]

        self.input_function_predict = x_data_[-self.window_length_out:, :]

        return x_data, y_data

    def assemble_datasets(self):
        """Prepare the datasets, split into train and test and create dataloaders."""
        x_data, y_data = self.load_data()

        # Generate windows using the window generator
        window_generator = WindowGenerator(
            x_data, y_data,
            self.window_length_in,
            self.window_length_out,
            shift=1, stride=1
        )
        inputs, outputs = zip(*window_generator)

        # Split the data into train and test
        data_train, data_test, targets_train, targets_test = train_test_split(
            torch.stack(list(inputs)),
            torch.stack(list(outputs)),
            test_size=0.2, shuffle=True,
        )
        self.input_function_train = data_train
        self.output_function_train = targets_train

        self.input_function_test = data_test
        self.output_function_test = targets_test

        # Create batches with dataloaders
        training_set = DataLoader(
            TensorDataset(data_train, targets_train),
            batch_size=self.batch_size, shuffle=False)
        testing_set = DataLoader(
            TensorDataset(data_test, targets_test),
            batch_size=self.batch_size, shuffle=False)

        return training_set, testing_set

    def plot_prediction(self):
        """Plot the input and output functions that will feed the model."""

        x_data, y_data = self.load_data()
        x_data = x_data.reshape(1, -1, 3)
        y_data = y_data.reshape(1, -1, 2)

        #  Inverse transform the data
        scalers = [self.scaler_tf0, self.scaler_ts0, self.scaler_time]
        x_data = self._inverse_transform_data(x_data, scalers).squeeze(0)
        scalers.pop()
        y_data = self._inverse_transform_data(y_data, scalers).squeeze(0)

        input_batch, output_pred_batch = self.predict_future()
        x_data = torch.cat([x_data, input_batch], dim=0)
        x_data = torch.cat([x_data, output_pred_batch], dim=0)

        # Plots
        _, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 4), frameon=False)
        ax[0].plot(x_data[:, 2], x_data[:, 0], linewidth=2)
        ax[0].plot(output_pred_batch[:, 2],
                   output_pred_batch[:, 0], linewidth=2)
        ax[0].set_ylabel(r'$T_f(x=0, t)$')
        ax[0].grid(True, which="both", ls=":")
        ax[0].set_ylim([300, 900])

        ax[1].plot(x_data[:, 2], x_data[:, 1], linewidth=2,
                   label='Measurements')
        ax[1].plot(output_pred_batch[:, 2], output_pred_batch[:, 1], linewidth=2,
                   label='Prediction')
        ax[1].set_ylabel(r'$T_s(x=0, t)$')
        ax[1].grid(True, which="both", ls=":")
        ax[1].set_ylim([300, 900])

        plt.xlabel('Time')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('plot_complete.pdf')
        plt.show()
        return output_pred_batch


    def train(self, epochs, learning_rate, step_size, gamma, freq_print=1):
        """Train the model."""
        # Set up training parameters
        optimizer = Adam(self.fno.parameters(),
                         lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

        loss = LpLoss(d=2, p=2, size_average=False)
        loss_test = LpLoss(d=2, p=2, size_average=False)

        # Training loop
        hist_train, hist_test = [], []
        for epoch in range(epochs):

            # Train the model
            train_loss = 0.0
            for input_batch, output_batch in self.training_set:
                optimizer.zero_grad()
                output_pred_batch = self.fno(input_batch).squeeze(2)
                loss_f = loss(output_pred_batch, output_batch)
                loss_f.backward()
                optimizer.step()
                train_loss += loss_f.item()
            train_loss /= len(self.training_set)
            hist_train.append(train_loss)
            scheduler.step()

            # Test the model
            with torch.no_grad():
                self.fno.eval()
                test_loss = 0.0
                for input_batch, output_batch in self.testing_set:
                    output_pred_batch = self.fno(input_batch).squeeze(2)
                    loss_t = loss_test(output_pred_batch, output_batch)
                    test_loss += loss_t.item()
                test_loss /= len(self.testing_set)
                hist_test.append(test_loss)

            # Print results
            if epoch % freq_print == 0:
                print("######### Epoch:", epoch, " ######### Train Loss:",
                      train_loss, " ######### Relative L2 Test Norm:", test_loss)

        return hist_train, hist_test

    @staticmethod
    def _inverse_transform_data(data, scalers):
        """Inverse transform the data per channel."""
        for i, scaler in enumerate(scalers):
            data[:, :, i] = torch.tensor(scaler.inverse_transform(
                data[:, :, i].reshape(-1, 1)).reshape(data[:, :, i].shape))
        return data

    def plot_window(self, idx=0):
        """Plot results."""
        # Retrieve data
        inputs = self.input_function_test
        outputs = self.output_function_test

        input_batch = deepcopy(inputs[idx, ...].unsqueeze(0))
        output_batch = deepcopy(outputs[idx, ...].unsqueeze(0))

        # Predict
        output_pred_batch = self.fno(input_batch).detach().numpy()

        #  Inverse transform the data
        scalers = [self.scaler_tf0, self.scaler_ts0, self.scaler_time]
        input_batch = self._inverse_transform_data(input_batch, scalers)
        scalers.pop()
        output_batch = self._inverse_transform_data(output_batch, scalers)
        output_pred_batch = self._inverse_transform_data(
            output_pred_batch, scalers)

        x_ax_output = input_batch[0, -1, 2] + \
            input_batch[0, :, 2]-input_batch[0, 0, 2]

        # Plots
        _, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 4), frameon=False)
        ax[0].plot(input_batch[0, :, 2], input_batch[0, :, 0],
                   label='Input', linewidth=2)
        ax[0].plot(x_ax_output, output_batch[0, :, 0],
                   label='Ground truth', linewidth=2)
        ax[0].plot(x_ax_output, output_pred_batch[0, :, 0],
                   label='Predicted', linestyle='--', linewidth=2)
        ax[0].set_ylabel(r'$T_f(x=0, t)$')
        ax[0].grid(True, which="both", ls=":")
        ax[0].set_ylim([300, 900])

        ax[1].plot(input_batch[0, :, 2], input_batch[0, :, 1],
                   label='Input', linewidth=2)
        ax[1].plot(x_ax_output, output_batch[0, :, 1],
                   label='Ground truth', linewidth=2)
        ax[1].plot(x_ax_output, output_pred_batch[0, :, 1],
                   label='Predicted', linestyle='--', linewidth=2)
        ax[1].set_ylabel(r'$T_s(x=0, t)$')
        ax[1].grid(True, which="both", ls=":")
        ax[1].set_ylim([300, 900])

        plt.xlabel('Time')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(f'plot_window_{idx}.pdf')
        plt.show()

    def plot_loss_function(self, hist_train, hist_test):
        """Plot the loss function normalized in log scale."""
        hist_train = np.array(hist_train)
        hist_test = np.array(hist_test)

        plt.figure(dpi=100, figsize=(7, 4), frameon=False)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist_train)+1),
                 hist_train, label="Train", linewidth=2)
        plt.plot(np.arange(1, len(hist_test)+1),
                 hist_test, label="Validation", linewidth=2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss_function.pdf')
        plt.show()

    def predict_future(self):
        """Predict the future."""
        # Make a prediction
        input_batch = self.input_function_predict.reshape(1, -1, 3)
        output_pred_batch = self.fno(input_batch).detach()

        # Inverse transform the data
        scalers = [self.scaler_tf0, self.scaler_ts0, self.scaler_time]
        input_batch = self._inverse_transform_data(
            input_batch, scalers).squeeze(0)
        scalers.pop()
        output_pred_batch = self._inverse_transform_data(
            output_pred_batch, scalers).squeeze(0)

        # Add the axis for the prediction
        pred_axis = input_batch[:, 2] - input_batch[0, 2]
        pred_axis += input_batch[-1, 2] + pred_axis[1]
        output_pred_batch = torch.cat(
            [output_pred_batch, pred_axis.reshape(-1, 1)], axis=1)

        return input_batch, output_pred_batch

    def plot_test(self):
        """Plot results."""
        # Retrieve data
        inputs = self.input_function_test
        outputs = self.output_function_test
        _, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 4), frameon=False)

        for idx in range(len(inputs)):
            if idx//7 == 0:
                continue

            input_batch = deepcopy(inputs[idx, ...].unsqueeze(0))
            output_batch = deepcopy(outputs[idx, ...].unsqueeze(0))

            # Predict
            output_pred_batch = self.fno(input_batch).detach().numpy()

            #  Inverse transform the data
            scalers = [self.scaler_tf0, self.scaler_ts0, self.scaler_time]
            input_batch = self._inverse_transform_data(input_batch, scalers)
            scalers.pop()
            output_batch = self._inverse_transform_data(output_batch, scalers)
            output_pred_batch = self._inverse_transform_data(
                output_pred_batch, scalers)

            x_ax_output = input_batch[0, -1, 2] + \
                input_batch[0, :, 2]-input_batch[0, 0, 2]

            # Plots
            ax[0].plot(x_ax_output, output_batch[0, :, 0], linewidth=2)
            ax[0].plot(x_ax_output, output_pred_batch[0, :, 0],
                       linestyle='--', linewidth=2)
            ax[0].set_ylabel(r'$T_f(x=0, t)$')
            ax[0].set_ylim([500, 900])
            ax[0].grid(True, which="both", ls=":")

            ax[1].plot(x_ax_output, output_batch[0, :, 1], linewidth=2)
            ax[1].plot(x_ax_output, output_pred_batch[0, :, 1],
                       linestyle='--', linewidth=2)
            ax[1].set_ylabel(r'$T_s(x=0, t)$')
            ax[1].set_ylim([500, 900])
            ax[1].grid(True, which="both", ls=":")

        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig('plot_predictions.pdf')
        plt.show()
