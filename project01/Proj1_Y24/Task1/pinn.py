"""Physics-Informed Neural Networks (PINNs) for the heat equation."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from nn import NNAnsatz
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


class PINNTrainer:
    """Trainer for the Physics-Informed Neural Network (PINN) for the heat equation.
    """

    def __init__(self, n_sb_, n_tb_, alpha_f, h_f,
                t_hot, u_f, alpha_s, h_s, t0):
        """Initialize the PINN trainer."""
        # Number of spatial and temporal boundary points
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Parameters of the reaction-convection-diffusion equation
        self.alpha_f = alpha_f
        self.h_f = h_f
        self.t_hot = t_hot
        self.u_f = u_f
        self.alpha_s = alpha_s
        self.h_s = h_s
        self.t0 = t0

        # Extrema of the solution domain (t, x) in [0,1] x [0,1]
        self.domain_extrema = torch.tensor([[0, 1],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1
        self.time_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # F Dense NN to approximate the solution of the underlying heat equation
        # Write the `NNAnsatz` class to be a feed-forward neural net
        # that you'll train to approximate your solution.
        # check values of parameters
        self.approximate_solution = NNAnsatz(
            input_dimension=self.space_dimensions+self.time_dimensions,
            output_dimension=self.space_dimensions*2,
            n_hidden_layers=1,
            hidden_size=100,
        )

        # Setup optimizer.
        self.optimizer = torch.optim.LBFGS(
            self.approximate_solution.parameters())

        # Generator of Sobol sequences.
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0])

        # Load training data
        self.data_train = np.genfromtxt('SubExample.csv', delimiter=',', skip_header=1)
        self.n_int = self.data_train.shape[0]

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
        """Initial condition to solve the heat equation u0(x)=-sin(pi x)
        """
        return torch.zeros((x.shape[0], 2)) + self.t_hot

    def exact_solution(self, inputs):
        """Exact solution for the heat equation ut = u_xx with the IC above.
        This is placeholder, we do not have exact solution for this problem.
        """
        t = inputs[:, 0]
        x = inputs[:, 1]

        u = -torch.exp(-np.pi ** 2 * t) * torch.sin(np.pi * x)
        return torch.stack((u, u), dim=1)

    def add_temporal_boundary_points(self):
        """Function returning the input-output tensor required to
        assemble the training set S_tb corresponding to the temporal boundary.
        TODO: Check correct implementation
        """
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1])

        return input_tb, output_tb

    def add_spatial_boundary_points(self):
        """Function returning the input-output tensor required to
        assemble the training set S_sb corresponding to the spatial boundary.
        TODO: Check correct implementation
        """
        x0 = self.domain_extrema[1, 0]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        output_sb_0 = torch.zeros((input_sb.shape[0], 2)) + self.t0

        return (input_sb_0, output_sb_0)

    def add_interior_points(self):
        """Function returning the input-output tensor required to assemble
        the training set S_int corresponding to the interior domain
        where the PDE is enforced.
        """
        input_int  = torch.Tensor(self.data_train[:, 0:2])
        output_int = torch.Tensor(self.data_train[:, 2:4])
        return input_int, output_int

    def assemble_datasets(self):
        """Function returning the training sets S_sb, S_tb, S_int as dataloader
        """
        input_sb, output_sb = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int

        training_set_sb = DataLoader(
            torch.utils.data.TensorDataset(input_sb, output_sb),
            batch_size=2 * self.space_dimensions*self.n_sb, shuffle=False)
        training_set_tb = DataLoader(
            torch.utils.data.TensorDataset(input_tb, output_tb),
            batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(
            torch.utils.data.TensorDataset(input_int, output_int),
            batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    def eval_initial_condition(self, input_tb):
        """Function to compute the terms required in the definition of
        the `TEMPORAL` boundary residual.
        """
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    def eval_boundary_conditions(self, input_sb):
        """Function to compute the terms required in the definition
        of the `SPATIAL` boundary residual.
        """
        u_pred_sb = self.approximate_solution(input_sb)

        return u_pred_sb

    def compute_pde_residual(self, input_int):
        """Function to compute the PDE residuals"""
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        # TODO: Implement the computation of the PDE residual
        # `grad` computes the gradient of a SCALAR function `L` w.r.t
        # some input (n x m) tensor  [[x1, y1], ...,[xn ,yn]] (here `m` = 2).
        # it returns grad_L = [[dL/dx1, dL/dy1]...,[dL/dxn, dL/dyn]]
        # NOTE: pytorch considers a tensor [u1, u2,u3, ... ,un] a vector
        # whereas `sum_u = u1 + u2 + u3 + u4 + ... + un` as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2], ...]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un) /dxi ==
        #  d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi == dui / dxi.

        tf = input_int[:,2]
        ts = input_int[:,3]

        u_tf = u[:,0]
        u_ts = u[:,1]

        # compute `grad_u` w.r.t (t, x) (time + 1D space).
        grad_u = torch.autograd.grad(
            u, input_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Extract time and space derivative at all input points.
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]

        # Compute `grads` again across the spatial dimension --
        # here you should reuse something you just computed.
        grad_u_xx = torch.autograd.grad(
            grad_u_x, input_int, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0][:, 1]

        # Compute the residual term you're getting.
        residual = grad_u_t - grad_u_xx
        return residual.reshape(-1,)

    def compute_loss(
        self, inp_train_sb, u_train_sb,
            inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        """Function to compute the total loss (weighted sum of spatial boundary loss,
        temporal boundary loss and interior loss)."""
        u_pred_sb = self.eval_boundary_conditions(inp_train_sb)
        u_pred_tb = self.eval_initial_condition(inp_train_tb)

        # Check dimensions
        assert u_pred_sb.shape[1] == u_train_sb.shape[1]
        assert u_pred_tb.shape[1] == u_train_tb.shape[1]

        # Compute interior PDE residual.
        r_int = self.compute_pde_residual(inp_train_int)

        # Compute spatial boundary residual.
        r_sb = (u_pred_sb - u_train_sb).reshape(-1,)

        # Compute temporal boundary residual
        r_tb = (u_pred_tb - u_train_tb).reshape(-1,)

        # Compute losses based on these residuals. Integrate using quadrature rule
        loss_sb = (r_sb*r_sb).sum()/len(r_sb)
        loss_tb = (r_tb*r_tb).sum()/len(r_tb)
        loss_int = (r_int*r_int).sum()/len(r_int)

        loss_u = loss_sb + loss_tb
        loss = torch.log10(self.lambda_u * (loss_sb + loss_tb) + loss_int)
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
        exact_output = self.exact_solution(inputs)

        labels = ["T_f", "T_s"]
        _, axs = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
        for i in range(2):
            im1 = axs[i][0].scatter(
                inputs[:, 1].detach(),
                inputs[:, 0].detach(),
                c=exact_output[:,i].detach(),
                cmap="jet"
            )
            axs[i][0].set_xlabel("x")
            axs[i][0].set_ylabel("t")
            plt.colorbar(im1, ax=axs[i][0])
            axs[i][0].grid(True, which="both", ls=":")
            axs[i][0].set_title(f"Exact Solution {labels[i]}")

            im2 = axs[i][1].scatter(
                inputs[:, 1].detach(),
                inputs[:, 0].detach(),
                c=output[:,i].detach(),
                cmap="jet"
            )
            axs[i][1].set_xlabel("x")
            axs[i][1].set_ylabel("t")
            plt.colorbar(im2, ax=axs[i][1])
            axs[i][1].grid(True, which="both", ls=":")
            axs[i][1].set_title(f"Approximate Solution {labels[i]}")

        plt.show()

        err = (
            torch.mean((output - exact_output) ** 2) /
            torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")
