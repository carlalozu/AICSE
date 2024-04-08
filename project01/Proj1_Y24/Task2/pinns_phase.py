from pinn import Pinns
import torch

class PinnLaterPhase(Pinns):

    def __init__(self, n_int_, n_sb_, n_tb_, t0, tf, reference_pinn, alpha_f, h_f, T_hot, T0, T_cold):
        self.reference_pinn = reference_pinn
        super().__init__(n_int_, n_sb_, n_tb_, t0, tf, alpha_f, h_f, T_hot, T0, T_cold)

    def initial_condition(self, x):
        """Initial condition to solve the equation, T0"""
        prev_output = self.reference_pinn.approximate_solution(x)
        return torch.clone(prev_output.detach())
