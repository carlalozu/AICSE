"""Inverse problem using physics-informed neural network (PINN)"""
import torch
from torch.utils.data import DataLoader
from neural_net import NeuralNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Pinns:
    """Class to solve an inverse problem using physics-informed neural network (PINN)"""

    def __init__(self, n_int_, n_sb_, n_tb_, t0, tf, alpha_f, h_f,
                 T_hot, T0, T_cold):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Parameters of the reaction-convection-diffusion equation
        self.alpha_f = alpha_f
        self.h_f = h_f
        self.T_hot = T_hot
        self.T0 = T0
        self.T_cold = T_cold

        self.t0 = t0
        self.tf = tf

        # Extrema of the solution domain (t,x)
        self.domain_extrema = torch.tensor([[t0, tf],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of cycles
        self.n_cycles = self.domain_extrema[0, 1] - self.domain_extrema[0, 0]
        self.n_sb_cycles = self.n_sb // self.n_cycles

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        ########################################################
        # Create FF Dense NNs for approximate solution and approximate coefficient

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=4,
            neurons=20,
            retrain_seed=42
        )

        # FF Dense NN to approximate the solid temperature we wish to infer
        self.approximate_coefficient = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=4,
            neurons=20,
            retrain_seed=42
        )

        ########################################################
        max_iter = 5000
        self.optimizer = torch.optim.LBFGS(
            list(self.approximate_solution.parameters()) +
            list(self.approximate_coefficient.parameters()),
            lr=float(0.5),
            max_iter=max_iter,
            max_eval=50000,
            history_size=150,
            line_search_fn="strong_wolfe",
            tolerance_change=1.0 * np.finfo(float).eps
        )

        # Generator of Sobol sequences --> Sobol sequences (see
        # https://en.wikipedia.org/wiki/Sobol_sequence)
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()


    def convert(self, tens):
        """Function to linearly transform a tensor whose value are between 0 and
        1 to a tensor whose values are between the domain extrema"""
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]
                       ) + self.domain_extrema[:, 0]

    def initial_condition(self, x):
        """Initial condition to solve the equation, T0"""
        return torch.ones_like(x)*self.T0

    def add_temporal_boundary_points(self):
        """Function returning the input-output tensor required to assemble the
        training set S_tb corresponding to the temporal boundary """

        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, self.t0)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)

        return input_tb, output_tb

    def add_spatial_boundary_points_charging(self, x):
        """Boundary points for charging phase"""
        # Dirichlet boundary condition
        output_sb_0 = torch.zeros((x.shape[0], 1)) + self.T_hot
        # Zero Neumann boundary condition
        output_sb_L = torch.zeros((x.shape[0], 1))
        return output_sb_0, output_sb_L

    def add_spatial_boundary_points_discharging(self, x):
        """Boundary points for discharging phase"""
        # Dirichlet boundary condition
        output_sb_L = torch.zeros((x.shape[0], 1)) + self.T_cold
        # Zero Neumann boundary condition
        output_sb_0 = torch.zeros((x.shape[0], 1))
        return output_sb_0, output_sb_L

    def add_spatial_boundary_points_idle(self, x):
        """Boundary points for idle phase"""
        # Zero Neumann boundary condition
        output_sb_L = torch.zeros((x.shape[0], 1))
        output_sb_0 = torch.zeros((x.shape[0], 1))
        return output_sb_0, output_sb_L

    def add_spatial_boundary_points(self):
        """Function returning the input-output tensor required to assemble the
        training set S_sb corresponding to the spatial boundary"""

        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        # Add input coordinates
        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # Add output coordinates
        for i in range(self.t0, self.tf):
            if i % 4 == 0:      # Multiple of 4
                print("Charging")
                output_sb_0, output_sb_L = self.add_spatial_boundary_points_charging(
                    input_sb_0)
            elif i % 2 != 0:    # Odd number
                print("Idle")
                output_sb_0, output_sb_L = self.add_spatial_boundary_points_idle(
                    input_sb_0)
            else:               # Else
                print("Discharging")
                output_sb_0, output_sb_L = self.add_spatial_boundary_points_discharging(
                    input_sb_0)

        return torch.cat([input_sb_0, input_sb_L], 0), torch.cat([output_sb_0, output_sb_L], 0)

    def add_interior_points(self):
        """Function returning the input-output tensor required to assemble the
        training set S_int corresponding to the interior domain where the PDE is
        enforced"""
        # Return input-output tensor required to assemble the training set S_int
        # corresponding to the interior domain where the PDE is enforced
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))

        return input_int, output_int

    def get_measurement_data(self):
        """Read measurements from data file and return the input-output tensor
        # TODO: Think about logic on measurements
        """
        data = pd.read_csv('DataSolution.txt', sep=",", header=0)
        data_filtered = data[(data['t'] < self.tf) & (data['t'] >= self.t0)]

        # Extract the data
        x = torch.tensor(data_filtered['x'].unique(), dtype=torch.float)
        t = torch.tensor(data_filtered['t'].unique(), dtype=torch.float)

        # Define the input-output tensor required to assemble the training
        input_meas = torch.cartesian_prod(x, t)
        output_meas = torch.tensor(
            data_filtered['tf'].values, dtype=torch.float)

        input_meas_ = torch.zeros(input_meas.shape)
        input_meas_[:, 0] = input_meas[:, 1]  # t in position 0
        input_meas_[:, 1] = input_meas[:, 0]  # x in position 1

        return input_meas_, output_meas.reshape(-1, 1)

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

    def apply_initial_condition(self, input_tb):
        """Function to compute the terms required in the definition of the
        TEMPORAL boundary residual"""
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    def apply_boundary_conditions(self, input_sb):
        """Compute the terms required in the definition of the SPATIAL boundary
        residual"""
        input_sb.requires_grad = True
        u_pred_sb = self.approximate_solution(input_sb)

        bound_x0 = None
        bound_xL = None
        for i in range(self.t0, self.tf):
            u_x0 = u_pred_sb[:self.n_sb]
            u_xL = u_pred_sb[self.n_sb:]

            # Differentiate u_x0 and u_xL with input_sb
            u_x0_x = torch.autograd.grad(
                u_x0.sum(), input_sb, create_graph=True)[0][:, 1]    
            u_x0_x = u_x0_x.reshape(-1, 1)[:self.n_sb]

            u_xL_x = torch.autograd.grad(
                u_xL.sum(), input_sb, create_graph=True)[0][:, 1]
            u_xL_x = u_xL_x.reshape(-1, 1)[self.n_sb:]

            if i % 4 == 0: # Charging
                print("BC for charging phase")
                bound_x0 = u_x0
                bound_xL = u_xL_x

            elif i % 2 != 0: # Idle
                print("BC for idle phase")
                bound_x0 = u_x0_x
                bound_xL = u_xL_x

            else: # Discharging
                print("BC for discharging phase")
                bound_x0 = u_x0_x
                bound_xL = u_xL

        # Concat
        u_bound = torch.cat([bound_x0, bound_xL], 0)
        return u_bound

    def compute_pde_residual(self, input_int):
        """Function to compute the PDE residuals"""
        input_int.requires_grad = True
        u = self.approximate_solution(input_int).reshape(-1,)
        k = self.approximate_coefficient(input_int).reshape(-1,)

        # grad compute the gradient of a "SCALAR" function L with respect to
        # some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3,
        # dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial
        # function whereas sum_u = u1 + u2 u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2,
        # dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2)
        # u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]

        ##############
        # Compute the second derivative (HINT: Pay attention to the
        # dimensions! --> torch.autograd.grad(..., ..., ...)[...][...]
        ##############
        grad_u_xx = torch.autograd.grad(
            grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        u_f = 0
        if self.t0 % 4 == 0:
            u_f = 1
        elif self.t0 % 2 != 0:
            u_f = -1

        residual = grad_u_t + u_f*grad_u_x - \
            self.alpha_f*grad_u_xx + self.h_f*(u - k)

        return residual.reshape(-1, )

    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        """Function to compute the total loss (weighted sum of spatial boundary
        loss, temporal boundary loss and interior loss)"""
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        inp_train_meas, u_train_meas = self.get_measurement_data()
        u_pred_meas = self.approximate_solution(inp_train_meas)

        assert u_pred_sb.shape[1] == u_train_sb.shape[1]
        assert u_pred_tb.shape[1] == u_train_tb.shape[1]
        assert u_pred_meas.shape[1] == u_train_meas.shape[1]

        ##############
        # Define respective resiudals and loss values

        # Compute interior PDE residual.
        r_int = self.compute_pde_residual(inp_train_int)
        # Compute spatial boundary residual.
        r_sb = u_pred_sb - u_train_sb
        # Compute temporal boundary residual
        r_tb = u_pred_tb - u_train_tb
        # Compute measurement residual
        r_meas = u_pred_meas - u_train_meas

        # Compute losses based on these residuals. Integrate using quadrature rule
        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)
        loss_meas = torch.mean(abs(r_meas) ** 2)

        ##############

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        if verbose:
            print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(
                loss_int).item(), 4), "| Function Loss: ", round(torch.log10(loss_u).item(), 4))

        return loss

    def fit(self, num_epochs, verbose=True):
        """Function to fit the PINN"""
        history = []
        inp_train_sb = None
        u_train_sb = None
        inp_train_tb = None
        u_train_tb = None
        inp_train_int = None

        history = []

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose:
                print("################################ ",
                      epoch, " ################################")

            for inputs_and_outputs in zip(
                self.training_set_sb,
                self.training_set_tb,
                self.training_set_int
            ):
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
        output_tf = self.approximate_solution(inputs)
        output_ts = self.approximate_coefficient(inputs)

        output = torch.cat([output_tf, output_ts], 1)
        labels = ["T_f", "T_s"]
        _, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        lims = [(1, 4), (1, 3)]
        for i in range(2):
            im = axs[i].scatter(
                inputs[:, 0].detach(),
                inputs[:, 1].detach(),
                c=output[:, i].detach(),
                cmap="jet",
                clim=lims[i]
            )
            axs[i].set_xlabel("t")
            axs[i].set_ylabel("x")
            axs[i].grid(True, which="both", ls=":")
            axs[i].set_title(f"Approximate Solution {labels[i]}")
            plt.colorbar(im, ax=axs[i])
        plt.show()
