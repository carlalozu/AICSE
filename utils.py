import numpy as np
import matplotlib.pyplot as plt
import torch


def initial_condition(x):
    # Initial condition to solve the heat equation u0(x)=-sin(pi x)
    return -torch.sin(np.pi * x)

def exact_solution(inputs):
    # Exact solution for the heat equation ut = u_xx with the IC above
    t = inputs[:, 0]
    x = inputs[:, 1]

    u = -torch.exp(-np.pi ** 2 * t) * torch.sin(np.pi * x)
    return u


def exact_conductivity(inputs):
    # t = inputs[:, 0]
    x = inputs[:, 1]
    k = torch.sin(np.pi * x) + 1.1

    return k


def source(inputs):
    s = -np.pi**2*exact_solution(inputs)*(1 - exact_conductivity(inputs))
    return s


def fit(model, num_epochs, optimizer, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print("################################ ",
                  epoch, " ################################")

        for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(model.training_set_sb, model.training_set_tb, model.training_set_int)):
            def closure():
                optimizer.zero_grad()
                loss = model.compute_loss(
                    inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                loss.backward()

                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)

    print('Final Loss: ', history[-1])

    return history


def plotting(model):
    inputs = model.soboleng.draw(100000)
    inputs = model.convert(inputs)

    output = model.approximate_solution(inputs).reshape(-1, )
    exact_output = exact_solution(inputs).reshape(-1, )

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:,
                         0].detach(), c=exact_output.detach(), cmap="jet")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    plt.colorbar(im1, ax=axs[0])
    axs[0].grid(True, which="both", ls=":")
    im2 = axs[1].scatter(inputs[:, 1].detach(),
                         inputs[:, 0].detach(), c=output.detach(), cmap="jet")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")
    plt.colorbar(im2, ax=axs[1])
    axs[1].grid(True, which="both", ls=":")
    axs[0].set_title("Exact Solution")
    axs[1].set_title("Approximate Solution")

    err = (torch.mean((output - exact_output) ** 2) /
           torch.mean(exact_output ** 2)) ** 0.5 * 100
    print("L2 Relative Error Norm U: ", err.item(), "%")

    approx_cond = model.approximate_coefficient(inputs).reshape(-1, )
    exact_cond = exact_conductivity(inputs).reshape(-1, )

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:,
                         0].detach(), c=exact_cond.detach(), cmap="jet")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    plt.colorbar(im1, ax=axs[0])
    axs[0].grid(True, which="both", ls=":")
    im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:,
                         0].detach(), c=approx_cond.detach(), cmap="jet")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")
    plt.colorbar(im2, ax=axs[1])
    axs[1].grid(True, which="both", ls=":")
    axs[0].set_title("Exact Conductivity")
    axs[1].set_title("Approximate Conductivity")

    err = (torch.mean((approx_cond - exact_cond) ** 2) /
           torch.mean(exact_cond ** 2)) ** 0.5 * 100
    print("L2 Relative Error Norm K: ", err.item(), "%")

    plt.show()
