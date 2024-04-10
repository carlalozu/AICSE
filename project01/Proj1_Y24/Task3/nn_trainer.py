"""Simple class to make a predictor using a neural network"""

import torch
from torch.utils.data import DataLoader
from torch import nn
from neural_net import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error)


class NNTrainer:
    """Class to solve a regression problem using a neural net"""

    def __init__(self, dataset_train, dataset_test):

        # Assemble the datasets

        self.n_dims = dataset_train.shape[1]

        self.minmax_scaler = MinMaxScaler()
        self.dataset_train, self.dataset_test = self.assemble_datasets(
            dataset_train, dataset_test)

        self.loss_function = nn.MSELoss(
            reduction='mean',
        )

        self.network = NeuralNet(
            input_dimension=self.n_dims-1,
            output_dimension=1,
            n_hidden_layers=3,
            neurons=(self.n_dims-1)*2,
            retrain_seed=42
        )

    @staticmethod
    def convert_df_to_tensor(df):
        """Function to convert a pandas dataframe to a PyTorch tensor"""
        a = np.vstack(df.values).astype(np.float32)
        return torch.from_numpy(a)

    def assemble_datasets(self, dataset_train, dataset_test):
        """Function to assemble the datasets"""

        input_train = self.convert_df_to_tensor(
            dataset_train.drop('median_house_value', axis=1))

        output_train = torch.tensor(
            dataset_train['median_house_value'].values, dtype=torch.float32).reshape(-1, 1)
        output_train_minmax = self.minmax_scaler.fit_transform(output_train)
        output_train_minmax = torch.tensor(
            output_train_minmax.flatten(), dtype=torch.float32)

        input_test = self.convert_df_to_tensor(
            dataset_test.drop('median_house_value', axis=1))

        output_test = torch.tensor(
            dataset_test['median_house_value'].values, dtype=torch.float32).reshape(-1, 1)

        output_test_minmax = self.minmax_scaler.transform(output_test)
        output_test_minmax = torch.tensor(
            output_test_minmax.flatten(), dtype=torch.float32)

        dataset_train_ = torch.utils.data.TensorDataset(
            input_train, output_train_minmax)
        dataset_test_ = torch.utils.data.TensorDataset(
            input_test, output_test_minmax)

        dataloader_params = {
            "batch_size": 150,
            "shuffle": True,
            "pin_memory": True,
        }
        training_set = DataLoader(dataset_train_, **dataloader_params)
        testing_set = DataLoader(dataset_test_, **dataloader_params)

        return training_set, testing_set

    def unscale(self, x):
        """Function to unscale the data"""
        is_tensor = False
        if isinstance(x, torch.Tensor):
            is_tensor = True
            x = torch.clone(x).detach().numpy()

        # Invert scaling
        x_ = self.minmax_scaler.inverse_transform(x)

        if is_tensor:
            # Convert back to PyTorch tensor
            x_ = torch.tensor(x_.flatten(), dtype=torch.float32)
        return x_

    def compute_loss(self, x, y, verbose=False):
        """Function to compute the loss"""
        outputs = self.network(x)
        y = y.reshape(-1, 1)

        # MSE directly from the loss function
        loss = self.loss_function(outputs, y)

        # Inverse transform using the scalers
        outputs_ = self.unscale(outputs)
        y_ = self.unscale(y)
        # RMSE of the not normalized values
        loss_not_normalized = root_mean_squared_error(outputs_, y_)

        if verbose:
            print("Total loss (MSE): ", round(loss.item(), 4))
            print("Total loss not normalized (RMSE): ",
                  round(loss_not_normalized.item(), 4))

        return loss, loss_not_normalized

    def fit(self, num_epochs, verbose=True):
        """Function to fit the PINN"""

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-2,
            weight_decay=1e-6,
        )

        inp_train = None
        outputs_train = None

        history = []
        history_not_n = []

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose:
                print("################################ ",
                      epoch, " ################################")

            for inp_train, outputs_train in self.dataset_train:

                def closure():
                    optimizer.zero_grad()
                    loss, loss_not_n = self.compute_loss(
                        inp_train,
                        outputs_train,
                        verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    history_not_n.append(loss_not_n.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history, history_not_n

    def plot_loss_function(self, hist, **kwargs):
        """Function to plot the loss function"""
        plt.figure(dpi=100)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, **kwargs)
        plt.xscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.legend()
        plt.show()

    def test_model(self):
        """Function to test the model on the test set for a fold"""
        y_hat = []
        y = []
        with torch.no_grad():
            for data in self.dataset_test:
                inputs, targets = data
                targets = targets.reshape(-1, 1)
                outputs = self.network(inputs)

                y_hat.extend(outputs.numpy())
                y.extend(targets.numpy())

        y_hat_ = self.unscale(y_hat)
        y_ = self.unscale(y)

        # Normalized
        mse = mean_squared_error(y, y_hat)
        rmse = root_mean_squared_error(y, y_hat)
        mae = mean_absolute_error(y, y_hat)
        r2 = r2_score(y, y_hat)

        # Unnormalized
        mse_ = mean_squared_error(y_, y_hat_)
        rmse_ = root_mean_squared_error(y_, y_hat_)
        mae_ = mean_absolute_error(y_, y_hat_)
        r2_ = r2_score(y_, y_hat_)

        print("Normalized values")
        print("Mean Squared Error (MSE):", round(mse, 4))
        print("Root Mean Squared Error (RMSE):", round(rmse, 4))
        print("Mean Absolute Error (MAE):", round(mae, 4))
        print("R-squared (R2):", round(r2, 4), "\n")

        print("Unnormalized values")
        print("Mean Squared Error (MSE):", round(mse_, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse_, 2))
        print("Mean Absolute Error (MAE):", round(mae_, 2))
        print("R-squared (R2):", round(r2_, 4))
