import time

import torch
from torch import tensor
from torch.optim import Adam
from torch.nn import functional as F
from sage_net import SAGENet
import os.path as osp
import os

from torch_geometric.datasets import Planetoid



class Trainer():

    def __init__(self, aggr):
        """Implement and train a basic GNN  to do vertex classification on the Cora dataset."""
        self.dataset = self.assemble_dataset()
        self.model = SAGENet(self.dataset, aggr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def assemble_dataset(self):
        path = osp.join(os.getcwd(), 'data', 'Cora')

        return Planetoid(path, 'Cora')

    def run(self, runs, epochs, lr, weight_decay,
        early_stopping):

        val_losses, accs, durations = [], [], []
        for _ in range(runs):
            data = self.dataset[0]
            data = data.to(self.device)

            self.model.to(self.device).reset_parameters()
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            best_val_loss = float('inf')
            test_acc = 0
            val_loss_history = []

            for epoch in range(1, epochs + 1):
                self.train(optimizer, data)
                eval_info = self.evaluate(self.model, data)
                eval_info['epoch'] = epoch

                if eval_info['val_loss'] < best_val_loss:
                    best_val_loss = eval_info['val_loss']
                    test_acc = eval_info['test_acc']

                val_loss_history.append(eval_info['val_loss'])
                if early_stopping > 0 and epoch > epochs // 2:
                    tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                    if eval_info['val_loss'] > tmp.mean().item():
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            val_losses.append(best_val_loss)
            accs.append(test_acc)
            durations.append(t_end - t_start)

        loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

        print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
            format(loss.mean().item(),
                    acc.mean().item(),
                    acc.std().item(),
                    duration.mean().item()))


    def train(self, optimizer, data):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


    def evaluate(self, model, data):
        model.eval()

        with torch.no_grad():
            logits = model(data)

        outs = {}
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            loss = F.nll_loss(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            outs['{}_loss'.format(key)] = loss
            outs['{}_acc'.format(key)] = acc

        return outs
