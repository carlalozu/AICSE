"""Simple class to make a predictor using a neural network"""

import torch
from torch.utils.data import DataLoader
from torch import nn
from neural_net import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
class PinnTrainer:
    """Class to solve an inverse problem using physics-informed neural network (PINN)"""

    def __init__(self, dataset_train, dataset_test):
        # Assemble the datasets

        self.n_dims = dataset_train.shape[1]

        self.minmax_scaler = StandardScaler()
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

    def assemble_datasets(self, dataset_train, dataset_test):
        """Function to assemble the datasets"""
        a = dataset_train.drop('median_house_value', axis=1).values
        a = np.vstack(a).astype(np.float32)
        input_train = torch.from_numpy(a)

        output_train = torch.tensor(
            dataset_train['median_house_value'].values, dtype=torch.float32).reshape(-1, 1)
        output_train_minmax = self.minmax_scaler.fit_transform(output_train)
        output_train_minmax = torch.tensor(output_train_minmax.flatten(), dtype=torch.float32)

        a = dataset_test.drop('median_house_value', axis=1).values
        a = np.vstack(a).astype(np.float32)
        input_test = torch.from_numpy(a)
        output_test = torch.tensor(
            dataset_test['median_house_value'].values, dtype=torch.float32).reshape(-1, 1)
        output_test_minmax = self.minmax_scaler.transform(output_test)
        output_test_minmax = torch.tensor(output_test_minmax.flatten(), dtype=torch.float32)

        dataset_train = torch.utils.data.TensorDataset(
            input_train, output_train_minmax)
        dataset_test = torch.utils.data.TensorDataset(
            input_test, output_test_minmax)

        training_set = DataLoader(dataset_train,
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)

        testing_set = DataLoader(dataset_test,
                                 batch_size=32,
                                 shuffle=True,
                                 pin_memory=True)

        return training_set, testing_set

    def un_normalize(self, y_hat, y):
        """Function to unnormalize the data"""
        # Inverse transform using the scalers
        y_hat_inverse = self.minmax_scaler.inverse_transform(y_hat.detach().numpy())
        y_inverse = self.minmax_scaler.inverse_transform(y)

        # Convert back to PyTorch tensor
        y_hat_inverse = torch.tensor(y_hat_inverse.flatten(), dtype=torch.float32)
        y_inverse = torch.tensor(y_inverse.flatten(), dtype=torch.float32)
        return y_hat_inverse, y_inverse

    def compute_loss(self, x, y, verbose=False):
        """Function to compute the loss"""
        y_hat = self.network(x)
        y = y.reshape(-1, 1)

        loss = self.loss_function(y_hat, y)

        # Inverse transform using the scalers
        y_hat_inverse, y_inverse = self.un_normalize(y_hat, y)

        # MSE of not normalized values
        loss_not_normalized = torch.sqrt(torch.mean((y_hat_inverse-y_inverse)**2))

        if verbose:
            print("Total loss: ", round(loss.item(), 4))
            print("Total loss nt normalized: ", round(loss_not_normalized.item(), 4))
        return loss, loss_not_normalized


    def test_model(self):
        """Function to test the model on the test set for a fold"""
        total_rmse_, total_rmse = 0, 0

        n = 0
        with torch.no_grad():
            for data in self.dataset_test:
                inputs, targets = data
                targets = targets.reshape(-1, 1)
                y_hat = self.network(inputs)
                y_hat_, targets_ = self.un_normalize(y_hat, targets)

                rmse = torch.sqrt(self.loss_function(y_hat, targets))
                rmse_ = torch.sqrt(torch.mean((y_hat_-targets_)**2))

                total_rmse += rmse.item()
                total_rmse_ += rmse_.item()

                n += 1

        print(f'RMSE: {total_rmse/n:.2f}')
        print(f'RMSE not normalized: {total_rmse_:.2f}')
        return total_rmse, total_rmse_

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

    def plot_loss_function(self, hist):
        """Function to plot the loss function"""
        plt.figure(dpi=100)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        plt.show()
