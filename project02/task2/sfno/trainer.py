"""Trainer for the FNO1d model."""
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from neuralop.datasets import load_spherical_swe
from sklearn.preprocessing import StandardScaler

from utils import LpLoss

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer():
    """Trainer for the FNO2d or SFN02d model."""

    def __init__(self, n_train, model):
        self.n_train = n_train
        self.n_test = 50

        self.scalers_inputs = [
            StandardScaler(), StandardScaler(), StandardScaler()]
        self.scalers_outputs = [
            StandardScaler(), StandardScaler(), StandardScaler()]

        self.train_resolution = (32, 64)
        self.test_resolutions = [(32, 64), (64, 128)]
        self.training_set, self.testing_set = self.assemble_datasets()


        self.model = model.to(device)

    @staticmethod
    def _arrange_data(dataset):
        """Arrange the data."""
        data_in = []
        data_out = []
        for i, data in enumerate(dataset):
            if i >= len(dataset):
                break

            data_in.append(data['x'])
            data_out.append(data['y'])

        data_in = torch.stack(data_in).permute(0, 3, 2, 1).transpose(1, 2)
        data_out = torch.stack(data_out).permute(0, 3, 2, 1).transpose(1, 2)

        return data_in, data_out

    @staticmethod
    def _fit_scaler(data, scalers):
        for i, scaler in enumerate(scalers):
            scaler.fit(data[:, :, :, i].reshape(-1, 1))

    @staticmethod
    def _transform_data(data, scalers):
        for i, scaler in enumerate(scalers):
            data[:, :, :, i] = torch.tensor(scaler.transform(
                data[:, :, :, i].reshape(-1, 1)).reshape(data[:, :, :, i].shape))
        return data

    @staticmethod
    def _inverse_transform_data(data, scalers):
        for i, scaler in enumerate(scalers):
            data[:, :, :, i] = torch.tensor(scaler.inverse_transform(
                data[:, :, :, i].reshape(-1, 1)).reshape(data[:, :, :, i].shape))
        return data

    def assemble_datasets(self):
        """Load the data and prepare the datasets."""

        training_set, testing_set = load_spherical_swe(
            n_train=self.n_train,
            batch_size=None,
            train_resolution=self.train_resolution,
            test_resolutions=self.test_resolutions,
            n_tests=[self.n_test, self.n_test],
            test_batch_sizes=[None, None],
        )

        # Training set
        data_in, data_out = self._arrange_data(training_set.dataset)

        self.input_function_train = data_in
        self.output_function_train = data_out

        self._fit_scaler(data_in, self.scalers_inputs)
        self._fit_scaler(data_out, self.scalers_outputs)

        data_in = self._transform_data(data_in, self.scalers_inputs)
        data_out = self._transform_data(data_out, self.scalers_outputs)

        training_set_ = DataLoader(
            TensorDataset(data_in, data_out), batch_size=4, shuffle=False)

        # Testing set
        testing_set_ = {}
        self.input_function_test = {}
        self.output_function_test = {}

        for resolution in self.test_resolutions:
            data_in, data_out = self._arrange_data(
                testing_set[resolution].dataset)

            data_in = self._transform_data(data_in, self.scalers_inputs)
            data_out = self._transform_data(data_out, self.scalers_outputs)

            self.input_function_test[resolution] = data_in
            self.output_function_test[resolution] = data_out
            testing_set_[resolution] = DataLoader(
                TensorDataset(data_in, data_out), batch_size=10, shuffle=False)

        return training_set_, testing_set_

    def plot_inputs(self, idx_sample=0):
        """Plot the input and output functions."""

        inputs_, outputs_ = self.input_function_train, self.output_function_train

        _, ax = plt.subplots(1, 2, figsize=(7, 3))
        inputs = inputs_[idx_sample].unsqueeze(0)
        outputs = outputs_[idx_sample].unsqueeze(0)

        inputs = self._inverse_transform_data(inputs, self.scalers_inputs)
        outputs = self._inverse_transform_data(
            outputs, self.scalers_outputs)

        ax[0].imshow(inputs[0, :, :, 0])
        ax[0].set_title('Input')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(outputs[0, :, :, 0])
        ax[1].set_title('Ground-truth')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()

    def train(self, epochs, learning_rate=8e-4):
        """Train the model."""

        optimizer = Adam(self.model.parameters(),
                         lr=learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)

        loss = LpLoss(d=2, p=2, reduction=True)
        loss_test = LpLoss(d=2, p=2, reduction=True)
        freq_print = 1
        hist = []
        hist_train = []
        for epoch in range(epochs):
            train_loss = 0.0
            for input_batch, output_batch in self.training_set:
                optimizer.zero_grad()
                output_pred_batch = self.model(input_batch).squeeze(2)
                loss_f = loss(output_pred_batch, output_batch)
                loss_f.backward()
                optimizer.step()
                train_loss += loss_f.item()
            train_loss /= len(self.training_set)
            hist.append(train_loss)
            scheduler.step()

            with torch.no_grad():
                self.model.eval()
                test_loss = 0.0
                for resolution in self.test_resolutions:
                    test_loss_ = 0.0
                    for input_batch, output_batch in self.testing_set[resolution]:
                        output_pred_batch = self.model(input_batch).squeeze(2)
                        loss_t = loss_test(
                            output_pred_batch, output_batch)
                        test_loss_ += loss_t.item()
                    test_loss_ /= len(self.testing_set)
                    test_loss += test_loss_
                test_loss /= len(self.test_resolutions)
                hist_train.append(test_loss)

            if epoch % freq_print == 0:
                print("######### Epoch:", epoch, " ######### Train Loss:",
                      train_loss, " ######### Relative L2 Test Norm:", test_loss)
        return hist, hist_train

    def plot(self, idx_sample=0):
        """Plot results"""
        fig = plt.figure(figsize=(9, 5))
        for index, resolution in enumerate(self.test_resolutions):
            inputs = self.input_function_test[resolution]
            outputs = self.output_function_test[resolution]

            inputs = inputs[idx_sample, ...].unsqueeze(0)
            outputs = outputs[idx_sample, ...].unsqueeze(0)

            output_pred_batch = self.model(inputs)
            output_pred_batch = output_pred_batch.detach().cpu()

            inputs = self._inverse_transform_data(inputs, self.scalers_inputs)
            outputs = self._inverse_transform_data(
                outputs, self.scalers_outputs)
            output_pred_batch = self._inverse_transform_data(
                output_pred_batch, self.scalers_outputs)

            ax = fig.add_subplot(2, 3, index*3 + 1)
            ax.imshow(inputs[0, :, :, 0])
            ax.set_title(f'Input {resolution}')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(2, 3, index*3 + 2)
            ax.imshow(outputs[0, :, :, 0])
            ax.set_title('Ground-truth')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(2, 3, index*3 + 3)
            ax.imshow(output_pred_batch[0, :, :, 0])
            ax.set_title('Model prediction')
            plt.xticks([], [])
            plt.yticks([], [])

            plt.tight_layout()
            plt.savefig(f"results_{idx_sample}_{self.model}.png")

    def plot_loss_function(self, hist_train, hist_test):
        """Function to plot the loss function"""
        hist_train = np.array(hist_train)
        hist_test = np.array(hist_test)

        plt.figure(dpi=100, figsize=(7, 4), frameon=False)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist_train)+1), hist_train, label = "Train")
        plt.plot(np.arange(1, len(hist_test)+1), hist_test, label = "Validation")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.xticks(np.arange(0, len(hist_train) + 1, 2))
        plt.ylabel("Log loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"loss_function_{self.model}.pdf")
        plt.show()
