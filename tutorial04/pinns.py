"""Inverse problem using physics-informed neural network (PINN)"""
import torch
from torch.utils.data import DataLoader
from neural_net import NeuralNet
from utils import initial_condition, exact_solution, source

class InversePinns:
    """Class to solve an inverse problem using physics-informed neural network (PINN)"""
    def __init__(self, n_int_, n_sb_, n_tb_):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Extrema of the solution domain (t,x) in [0,0.1] x [-1,1]
        self.domain_extrema = torch.tensor([[0, 0.1],  # Time dimension
                                            [-1, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        ########################################################
        # Create FF Dense NNs for approxiamte solution and approximate coefficient

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=3,
            neurons=20,
            retrain_seed=42
        )

        # FF Dense NN to approximate the conductivity we wish to infer
        self.approximate_coefficient = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=3,
            neurons=20,
            retrain_seed=42
        )

        ########################################################

        # Generator of Sobol sequences --> Sobol sequences (see
        # https://en.wikipedia.org/wiki/Sobol_sequence)
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

        # number of sensors to record temperature
        self.n_sensor = 50

    #######################################################################
    def convert(self, tens):
        """Function to linearly transform a tensor whose value are between 0 and
        1 to a tensor whose values are between the domain extrema"""
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]
                       ) + self.domain_extrema[:, 0]

    #######################################################################
    def add_temporal_boundary_points(self):
        """Function returning the input-output tensor required to assemble the
        training set S_tb corresponding to the temporal boundary """

        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = initial_condition(input_tb[:, 1]).reshape(-1, 1)

        return input_tb, output_tb

    def add_spatial_boundary_points(self):
        """Function returning the input-output tensor required to assemble the
        training set S_sb corresponding to the spatial boundary"""

        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_0 = torch.zeros((input_sb.shape[0], 1))
        output_sb_L = torch.zeros((input_sb.shape[0], 1))

        return torch.cat([input_sb_0, input_sb_L], 0), torch.cat([output_sb_0, output_sb_L], 0)

    def add_interior_points(self):
        """Function returning the input-output tensor required to assemble the
        training set S_int corresponding to the interior domain where the PDE is
        enforced"""

        ########################################################
        # Return input-output tensor required to assemble the training set S_int
        # corresponding to the interior domain where the PDE is enforced

        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))

        ########################################################

        return input_int, output_int

    def get_measurement_data(self):
        """Take measurments every 0.001 sec on a set of randomly or
        uniformly placed (in space) sensors """
        torch.random.manual_seed(42)

        ##########################################
        # TODO: Define the input-output tensor required to assemble the training
        t = None
        x = None

        ##########################################

        input_meas = torch.cartesian_prod(t, x)
        output_meas = exact_solution(input_meas).reshape(-1, 1)
        noise = 0.01*torch.randn_like(output_meas)
        output_meas = output_meas + noise

        return input_meas, output_meas

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

    #######################################################################
    def apply_initial_condition(self, input_tb):
        """Function to compute the terms required in the definition of the
        TEMPORAL boundary residual"""
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    def apply_boundary_conditions(self, input_sb):
        """Compute the terms required in the definition of the SPATIAL boundary residual"""

        ##############
        # TODO

        u_pred_sb = None

        ##############

        return u_pred_sb

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
        # TODO: Compute the second derivative (HINT: Pay attention to the
        # dimensions! --> torch.autograd.grad(..., ..., ...)[...][...]
        ##############
        grad_u_xx = None

        ##############

        s = source(input_int)

        residual = grad_u_t - k*grad_u_xx - s

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
        # TODO: Define respective resiudals and loss values
        ##############

        r_int = None
        r_sb = None
        r_tb = None
        r_meas = None

        loss_sb = None
        loss_tb = None
        loss_int = None
        loss_meas = None

        ##############

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        if verbose:
            print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(
                loss_int).item(), 4), "| Function Loss: ", round(torch.log10(loss_u).item(), 4))

        return loss
