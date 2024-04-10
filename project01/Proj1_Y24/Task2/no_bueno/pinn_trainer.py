from pinns_phase import Pinns, PinnLaterPhase
import torch

kwargs = {
    "alpha_f" : 0.005,
    "h_f" : 5,
    "T_hot" : 4,
    "T0" : 1,
    "T_cold" : 1,
}

n_int = 128
n_sb = 64
n_tb = 64
max_iter = 5000

pins_phases = []

# Train Pinn for phase 0
pinn = Pinns(n_int, n_sb, n_tb, 0, 1, **kwargs)
hist = pinn.fit(num_epochs=1, max_iter=max_iter, verbose=False)
torch.save(pinn.approximate_solution.state_dict(), 'saved_models/pinn_0.pth')
pins_phases.append(pinn)
# Time: around 3:30 mins each

# Train Pinn for phase 1 to 7
for i in range(1, 8):
    t0 = i
    tf = i+1

    old_pinn = pins_phases[-1]
    new_pinn = PinnLaterPhase(n_int, n_sb, n_tb, t0, tf, old_pinn, **kwargs)
    hist = new_pinn.fit(num_epochs=1, max_iter=max_iter, verbose=False)
    torch.save(new_pinn.approximate_solution.state_dict(), f'saved_models/pinn_{t0}.pth')
    pins_phases.append(new_pinn)
