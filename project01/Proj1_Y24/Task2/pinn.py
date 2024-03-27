"""Physics-Informed Neural Networks (PINNs) for the heat equation."""
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from neural_net import NeuralNet
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


class PINNTrainer:
    """Trainer for the Physics-Informed Neural Network (PINN) for the heat equation.
    """

    def __init__(self, n_int_, n_sb_, n_tb_, alpha_f, h_f,
                 T_hot, u_f, T0, T_cold):
        """Initialize the PINN trainer."""
        # Number of spatial and temporal boundary points
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Parameters of the reaction-convection-diffusion equation
        self.alpha_f = alpha_f
        self.h_f = h_f
        self.T_hot = T_hot
        self.u_f = u_f
        self.T0 = T0
        self.T_cold = T_cold

        # Extrema of the solution domain (t, x) in [0,8] x [0,1]
        self.domain_extrema = torch.tensor([[0, 8],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1
        self.time_dimensions = 1
        self.output_dimensions = 2

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # F Dense NN to approximate the solution of the underlying equation
        # Write the `NNAnsatz` class to be a feed-forward neural net
        # that you'll train to approximate your solution.
        # check values of parameters
        self.approximate_solution = NeuralNet(
            input_dimension=self.space_dimensions+self.time_dimensions,
            output_dimension=self.output_dimensions,
            n_hidden_layers=2,
            neurons=100,
            retrain_seed=42,
        )

        # Setup optimizer
        self.optimizer = torch.optim.LBFGS(
            self.approximate_solution.parameters(),
            lr=float(0.5),
            max_iter=100,
            max_eval=10000,
            history_size=150,
            line_search_fn="strong_wolfe",
            tolerance_change=1.0 * np.finfo(float).eps)

        # Generator of Sobol sequences.
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = (
            self.assemble_datasets())

    def convert(self, tens):
        """Function to linearly transform a tensor whose value are between 0 and 1
        to a tensor whose values are between the domain extrema"""
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return (
            tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0])
            + self.domain_extrema[:, 0])

    def initial_condition(self, x):
        """Initial condition to solve the equation at t=0"""
        return torch.zeros((x.shape[0], self.output_dimensions)) + self.T0

    def exact_solution(self, inputs):
        """Exact solution for the heat equation"""
        pass

    def add_temporal_boundary_points(self):
        """Function returning the input-output tensor required to
        assemble the training set S_tb corresponding to the temporal boundary.
        """
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1])

        return input_tb, output_tb

    def get_measurement_data(self):
        """Take measurements every 0.001 sec on a set of randomly or
        uniformly placed (in space) sensors 
        # TODO: Think about logic on measurements
        """
        data = pd.read_csv('DataSolution.txt', sep=",", header=0)

        # Extract the data
        t = torch.tensor(data['t'].unique(), dtype=torch.float)
        x = torch.tensor(data['x'].unique(), dtype=torch.float)

        # Define the input-output tensor required to assemble the training
        input_meas = torch.cartesian_prod(t, x)
        output_meas = torch.tensor(data['tf'].values, dtype=torch.float)

        return input_meas, output_meas

    def add_spatial_boundary_points(self):
        """Function returning the input-output tensor required to
        assemble the training set S_sb corresponding to the spatial boundary.
        # TODO: implement boundary points
        """
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # spatial boundary condition for Ts and Tf at x=0
        output_sb_0 = torch.zeros((input_sb.shape[0], self.output_dimensions))
        # spatial boundary condition for Tf at x=0
        output_sb_0[:, 0] = torch.zeros(input_sb.shape[0]) + self.T_hot

        # spatial boundary condition for Ts and Tf at x=L
        output_sb_L = torch.zeros((input_sb.shape[0], self.output_dimensions))

        return (
            torch.cat([input_sb_0, input_sb_L], 0),
            torch.cat([output_sb_0, output_sb_L], 0)
        )

    def add_interior_points(self):
        """Function returning the input-output tensor required to assemble
        the training set S_int corresponding to the interior domain
        where the PDE is enforced."""
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], self.output_dimensions))
        return input_int, output_int

    def assemble_datasets(self):
        """Function returning the training sets S_sb, S_tb, S_int as dataloader"""
        input_sb, output_sb = self.add_spatial_boundary_points()  # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(
            input_sb, output_sb), batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(
            input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(
            input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int


    def eval_initial_condition(self, input_tb):
        """Function to compute the terms required in the definition of
        the TEMPORAL boundary residual.
        """
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    def eval_boundary_conditions(self, input_sb):
        """Function to compute the terms required in the definition
        of the SPATIAL boundary residual.
        TODO: Implement boundary conditions depending on phase
        """
        input_sb.requires_grad = True 
        u_pred_sb = self.approximate_solution(input_sb)

        # Apply Von Neumann boundary conditions
        grad_u_tf = torch.autograd.grad(
            u_pred_sb[:, 0].sum(), input_sb, create_graph=True)[0]

        # Only to the second half of the points (x=L) for Tf
        grad_u_tf_x = grad_u_tf[int(len(u_pred_sb) / 2):, 1]
        grad_u_tf_x = torch.cat(
            [u_pred_sb[: int(len(u_pred_sb) / 2), 0], grad_u_tf_x], 0)

        grad_u_ts = torch.autograd.grad(
            u_pred_sb[:, 1].sum(), input_sb, create_graph=True)[0]
        grad_u_ts_x = grad_u_ts[:, 1]

        return torch.stack((grad_u_tf_x, grad_u_ts_x), dim=1)

    def compute_pde_residual(self, input_int):
        """Function to compute the PDE residuals"""
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        # compute `grad_u` w.r.t (t, x) (time + 1D space).
        grad_u_tf = torch.autograd.grad(
            u[:,0].sum(), input_int, create_graph=True)[0]

        # Extract time and space derivative at all input points.
        grad_u_tf_t = grad_u_tf[:, 0]
        grad_u_tf_x = grad_u_tf[:, 1]

        # Compute `grads` again across the spatial dimension
        grad_u_tf_xx = torch.autograd.grad(
            grad_u_tf_x.sum(), input_int, create_graph=True)[0][:, 1]

        # Residual of the PDE
        residual = grad_u_tf_t + self.u_f * grad_u_tf_x - \
            self.alpha_f * grad_u_tf_xx + self.h_f * (u[:,0] - u[:,1])

        return residual

    def compute_loss(
        self, inp_train_sb, u_train_sb,
            inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        """Function to compute the total loss (weighted sum of spatial boundary loss,
        temporal boundary loss and interior loss)."""
        u_pred_sb = self.eval_boundary_conditions(inp_train_sb)
        u_pred_tb = self.eval_initial_condition(inp_train_tb)

        inp_train_meas, u_train_meas = self.get_measurement_data()
        u_pred_meas = self.approximate_solution(inp_train_meas)

        # Check dimensions
        assert u_pred_sb.shape[1] == u_train_sb.shape[1]
        assert u_pred_tb.shape[1] == u_train_tb.shape[1]
        assert u_pred_meas.shape[0] == u_train_meas.shape[0]

        r_int = self.compute_pde_residual(inp_train_int)

        # Compute spatial boundary residual.
        r_sb = (u_pred_sb - u_train_sb).reshape(-1,)

        # Compute temporal boundary residual
        r_tb = (u_pred_tb - u_train_tb).reshape(-1,)

        # Compute measurement residual
        r_meas = (u_pred_meas[:,0] - u_train_meas).reshape(-1,)

        # Compute losses based on these residuals. Integrate using quadrature rule
        loss_sb = torch.mean(abs(r_sb)**2)
        loss_tb = torch.mean(abs(r_tb)**2)
        loss_int = torch.mean(abs(r_int)**2)
        loss_meas = torch.mean(abs(r_meas)**2)

        loss_u = loss_sb + loss_tb + loss_meas
        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        if verbose:
            print(
                "Total loss: ", round(loss.item(), 4),
                "| PDE Loss: ", round(torch.log10(loss_u).item(), 4),
                "| Function Loss: ", round(torch.log10(loss_int).item(), 4)
            )

        return loss

    def fit(self, num_epochs, verbose):
        """Function to fit the PINN"""
        history = []
        inp_train_sb = None
        u_train_sb = None
        inp_train_tb = None
        u_train_tb = None
        inp_train_int = None

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose:
                print("################################ ",
                      epoch, " ################################")

                for inputs_and_outputs in zip(self.training_set_sb,
                                              self.training_set_tb,
                                              self.training_set_int):
                    (
                        (inp_train_sb, u_train_sb),
                        (inp_train_tb, u_train_tb),
                        (inp_train_int, _)
                    ) = inputs_and_outputs

                    def closure():
                        self.optimizer.zero_grad()
                        loss = self.compute_loss(
                            inp_train_sb,
                            u_train_sb,
                            inp_train_tb,
                            u_train_tb,
                            inp_train_int,
                            verbose=verbose)
                        loss.backward()

                        history.append(loss.item())
                        return loss

                    self.optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    def plot(self):
        """Create plot"""
        inputs = self.soboleng.draw(100000)
        output = self.approximate_solution(inputs)

        labels = ["T_f", "T_s"]
        _ , axs = plt.subplots(1, 3, figsize=(16, 6), dpi=150)

        for i in range(2):
            im = axs[i].scatter(
                inputs[:, 1].detach(),
                inputs[:, 0].detach(),
                c=output[:, i].detach(),
                cmap="jet",
                clim=(1, 4)
            )
            axs[i].set_xlabel("x")
            axs[i].set_ylabel("t")
            axs[i].grid(True, which="both", ls=":")
            axs[i].set_title(f"Approximate Solution {labels[i]}")

        im = axs[i].scatter(
            inputs[:, 1].detach(),
            inputs[:, 0].detach(),
            c=output[:, i].detach(),
            cmap="jet",
            clim=(1, 4)
        )
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("t")
        axs[i].grid(True, which="both", ls=":")
        axs[i].set_title(f"Approximate Solution {labels[i]}")

        plt.colorbar(im, ax=axs)
        plt.show()
