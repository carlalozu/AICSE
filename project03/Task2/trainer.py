from neural_net import NNAnsatz
import torch


class Trainer():
    """Train the neural net to find the weights for the differential equation."""

    def __init__(self, library, ground_truth) -> None:
        self.library = library
        self.dim = len(library)
        self.model = NNAnsatz(input_dimension=self.dim,
                              output_dimension=self.dim,
                              n_hidden_layers=2,
                              hidden_size=10)
        self.ground_truth = ground_truth

    @staticmethod
    def _loss_fn(y_pred, y_true):
        """MSE loss function"""
        diff = torch.mean((y_pred - y_true)**2)*100
        # try to minimize the number of non-zero entries
        non_zero = torch.mean(torch.where(
            y_pred > 0.01, torch.ones_like(y_pred), torch.zeros_like(y_pred)))

        return diff + 0.1*non_zero

    def train(self, epochs=1000, lr=0.01, step_size=1000, gamma=0.5):
        """Find weights for the differential equations"""
        components_ = torch.stack(self.library, dim=1).squeeze()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if step_size:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            optimizer.zero_grad()
            weights = torch.ones(self.dim, requires_grad=True)
            weights = self.model(weights)

            y_pred = torch.sum(weights.view(self.dim, 1, 1)
                               * components_, dim=0)
            loss = self._loss_fn(y_pred, self.ground_truth.squeeze())
            loss.backward()
            optimizer.step()
            if step_size:
                scheduler.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, loss {loss.item()}')

        return weights

    def test(self, weights):
        """Test the neural net"""
        library_ = torch.stack(self.library, dim=1).squeeze()
        y_pred = torch.sum(weights.view(self.dim, 1, 1) * library_, dim=0)
        return y_pred.reshape(1, *y_pred.shape)

    def print_eq(self, weights):
        """Print the differential equation"""
        used = [f'{weight:.2e}*{term}' for term,
                weight in zip(self.library['terms'], weights) if weight != 0]
        print(' + '.join(used), '=d/dt')
