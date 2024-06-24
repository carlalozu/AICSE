from neural_net import NNAnsatz
import torch


def loss_fn(y_pred, y_true):
    """MSE loss function"""
    diff = torch.mean((y_pred - y_true)**2)*100
    # try to minimize the number of non-zero entries
    non_zero = torch.mean(torch.where(y_pred > 0.01, torch.ones_like(y_pred), torch.zeros_like(y_pred)))

    return diff + 0.1*non_zero

def train(library, y_true, epochs=1000):
    """Find weights for the differential equations"""
    dim = len(library)

    components_ = torch.stack(library, dim=1).squeeze()

    model = NNAnsatz(input_dimension=dim,
                     output_dimension=dim,
                     n_hidden_layers=2,
                     hidden_size=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    # Initialize weights
    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = torch.ones(dim, requires_grad=True)        
        weights = model(weights)

        y_pred = torch.sum(weights.view(dim, 1, 1) * components_, dim=0)
        loss = loss_fn(y_pred, y_true.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, loss {loss.item()}')

    return weights

def test(library, weights):
    """Test the neural net"""
    dim = len(library)
    library_ = torch.stack(library, dim=1).squeeze()
    y_pred = torch.sum(weights.view(dim, 1, 1) * library_, dim=0)
    return y_pred.reshape(1, *y_pred.shape)
