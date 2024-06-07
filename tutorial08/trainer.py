import random
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import AllenCahnDataset
from vit import ViT

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Trainer:
    def __init__(self):
        train_loader, val_loader = self.assemble_datasets()
        self.train_loader = train_loader
        self.val_loader = val_loader

        image_size = 64
        patch_size = 16
        dim = 128
        depth = 4
        heads = 4
        dim_head = 32
        emb_dropout = 0.0

        self.model = ViT(image_size=image_size,
                         patch_size=patch_size,
                         dim=dim,
                         depth=depth,
                         heads=heads,
                         mlp_dim=256,
                         channels=1,
                         dim_head=dim_head,
                         emb_dropout=emb_dropout)

        self.model.print_size()

    def assemble_datasets(self):
        training_samples = 256
        batch_size = 16

        train_dataset = AllenCahnDataset(
            which="train", training_samples=training_samples)
        val_dataset = AllenCahnDataset(which="val")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def plot_inputs(self):
        inputs, outputs = next(iter(self.train_loader))
        inputs = inputs[0, 0].numpy()
        outputs = outputs[0, 0].numpy()

        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(inputs, cmap="gist_ncar", extent=(0, 1, 0, 1),)
        axes[0].set_title("Input")
        axes[1].imshow(outputs, cmap="gist_ncar", extent=(0, 1, 0, 1),)
        axes[1].set_title("Output")

    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, eta_min=10**-6)
        l = nn.L1Loss()

        epoch = 200

        freq_print = 1
        for epoch in range(epoch):
            train_mse = 0.0
            for input_batch, output_batch in self.train_loader:
                optimizer.zero_grad()
                output_pred_batch = self.model(input_batch)
                loss_f = l(output_pred_batch, output_batch) / \
                    l(output_batch, torch.zeros_like(output_batch))
                loss_f.backward()
                optimizer.step()
                train_mse += loss_f.item()
            train_mse /= len(self.train_loader)

            scheduler.step()
            with torch.no_grad():
                self.model.eval()
                test_relative_l1 = 0.0
                for step, (input_batch, output_batch) in enumerate(self.val_loader):
                    output_pred_batch = self.model(input_batch)
                    loss_f = (torch.mean(
                        (abs(output_pred_batch - output_batch))) / torch.mean(abs(output_batch))) * 100
                    test_relative_l1 += loss_f.item()
                test_relative_l1 /= len(self.val_loader)

            if epoch % freq_print == 0:
                print("## Epoch:", epoch, " ## Train Loss:", train_mse,
                      "## Rel L1 Test Norm:", test_relative_l1, "LR: ", scheduler.get_lr())

    def plot(self):
        inputs, outputs = next(iter(self.val_loader))
        pred = self.model(inputs)
        inputs = inputs[0, 0].numpy()
        outputs = outputs[0, 0].numpy()
        pred = pred[0, 0].detach().numpy()

        vmin = np.min(outputs)
        vmax = np.max(outputs)

        _, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(inputs, cmap="gist_ncar", extent=(0, 1, 0, 1),)
        axes[0].set_title("Input")
        axes[1].imshow(outputs, cmap="gist_ncar", extent=(
            0, 1, 0, 1), vmin=vmin, vmax=vmax)
        axes[1].set_title("Output")
        axes[2].imshow(pred, cmap="gist_ncar", extent=(
            0, 1, 0, 1), vmin=vmin, vmax=vmax)
        axes[2].set_title("Prediction")
