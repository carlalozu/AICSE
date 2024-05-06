"""Trainer for the CNO1d model."""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
from cno1d import CNO1d

torch.manual_seed(0)
np.random.seed(0)


class Trainer():
    """Trainer for the CNO1d model."""

    def __init__(self, n_train, s, N_layers=4,
                 N_res=4, N_res_neck=4, channel_multiplier=16, batch_size=10):
        self.n_train = n_train
        self.s = s

        self.training_set, self.testing_set = self.assemble_datasets(
            batch_size)

        self.cno = CNO1d(
            # Number of input channels.
            in_dim=2,
            # Number of output channels.
            out_dim=1,
            # Input and Output spatial size (required )
            size=s,
            # Number of (D) or (U) blocks in the network
            N_layers=N_layers,
            # Number of (R) blocks per level (except the neck)
            N_res=N_res,
            # Number of (R) blocks in the neck
            N_res_neck=N_res_neck,
            # How the number of channels evolve?
            channel_multiplier=channel_multiplier,
            use_bn=False)

    def assemble_datasets(self, batch_size=10):
        """Load the data and prepare the datasets."""

        # Load the data
        # - AC_data_input.npy
        # - AC_data_output.npy
        x_data = torch.from_numpy(
            np.load("AC_data_input.npy")).type(torch.float32)
        y_data = torch.from_numpy(
            np.load("AC_data_output.npy")).type(torch.float32)

        x_data = x_data.permute(0, 2, 1)
        y_data = y_data.unsqueeze(1)
        x_data = F.interpolate(x_data.unsqueeze(
            2), size=(1, self.s), mode="bicubic")[:, :, 0]
        y_data = F.interpolate(y_data.unsqueeze(
            2), size=(1, self.s), mode="bicubic")[:, :, 0]

        self.input_function_train = x_data[:self.n_train, :]
        self.output_function_train = y_data[:self.n_train, :]
        self.input_function_test = x_data[self.n_train:, :]
        self.output_function_test = y_data[self.n_train:, :]

        training_set = DataLoader(TensorDataset(
            self.input_function_train, self.output_function_train),
            batch_size=batch_size, shuffle=True)
        testing_set = DataLoader(TensorDataset(
            self.input_function_test, self.output_function_test),
            batch_size=batch_size, shuffle=False)

        return training_set, testing_set

    def plot_inputs(self, id_sample=32):
        """Plot the input and output functions."""
        plt.figure()
        plt.plot(
            np.linspace(-1, 1, self.s),
            self.input_function_train[id_sample, 1, :],
            label="input:  $u(t = 0)$")
        plt.plot(
            np.linspace(-1, 1, self.s),
            self.output_function_train[id_sample, 0, :],
            label="output: $u(t = 1)$")
        plt.grid(True, which="both", ls=":")
        plt.legend()

    def train(self, epochs, learning_rate, step_size, gamma):
        """Train the model."""

        optimizer = AdamW(self.cno.parameters(),
                          lr=learning_rate, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

        loss = torch.nn.L1Loss()
        freq_print = 1
        for epoch in range(epochs):
            train_mse = 0.0
            for input_batch, output_batch in self.training_set:
                optimizer.zero_grad()
                output_pred_batch = self.cno(input_batch).squeeze(2)
                loss_f = loss(output_pred_batch, output_batch)
                loss_f.backward()
                optimizer.step()
                train_mse += loss_f.item()
            train_mse /= len(self.training_set)

            scheduler.step()

            with torch.no_grad():
                self.cno.eval()
                test_relative_l2 = 0.0
                for input_batch, output_batch in self.testing_set:
                    output_pred_batch = self.cno(input_batch).squeeze(2)
                    loss_f = self.error(output_pred_batch, output_batch)
                    test_relative_l2 += loss_f.item()
                test_relative_l2 /= len(self.testing_set)

            if epoch % freq_print == 0:
                print("######### Epoch:", epoch, " ######### Train Loss:",
                      train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)

    @ staticmethod
    def error(y, y_, p=2):
        """Relative L2 error."""
        err = (torch.mean(abs(y.detach().reshape(-1, ) - y_.detach(
        ).reshape(-1, )) ** p) / torch.mean(abs(y.detach()) ** p)) ** (1 / p) * 100
        return err

    def plot(self, idx_data=32):
        """Plot results"""
        input_function_test_n = self.input_function_test[idx_data].unsqueeze(
            0)
        output_function_test_n = self.output_function_test[idx_data].unsqueeze(
            0)
        # print(input_function_test_n.shape)
        # print(output_function_test_n.shape)

        output_function_test_pred_n = self.cno(input_function_test_n)
        # print(output_function_test_pred_n.shape)
        # print(input_function_test_n[0,:,1])
        plt.figure()
        plt.grid(True, which="both", ls=":")
        X = np.linspace(-1, 1, self.s)
        plt.plot(
            X,
            output_function_test_n[0, 0].detach(),
            label="True Solution", c="C0", lw=2)
        plt.scatter(
            X,
            output_function_test_pred_n[0, 0].detach(),
            label="Approximate Solution", c="C1")
        plt.legend()

        err = self.error(output_function_test_n, output_function_test_pred_n)
        print("Relative L2 error: ", err.item())
