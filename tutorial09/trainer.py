
import torch
import torchvision
from myconv import MyTinyUNet
from denoiser import DDPM
from tqdm import tqdm
from torch import nn


class Trainer():
    """Trainer for DDPM"""

    def __init__(self, num_timesteps, verbose=False):
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dataloader = self.assemble_datasets()

        self.network = MyTinyUNet().to(self.device)

        self.model = DDPM(self.network, num_timesteps, beta_start=0.0001,
                          beta_end=0.02, device=self.device)

        if self.verbose:
            for n, p in self.model.named_parameters():
                print(n, p.shape)

    def plot(self, title=""):
        """Plots the provided images in a square"""
        for b in self.dataloader:
            batch = b[0]
            break

        images = list(batch[:100])
        self.model.show_images(images, title=title)

    def assemble_datasets(self):
        """Get and download dataset"""
        root_dir = './data/'
        transform01 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])
        dataset = torchvision.datasets.MNIST(
            root=root_dir, train=True, transform=transform01, download=True)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=4096, shuffle=True, num_workers=10)
        return dataloader

    def train(self, num_epochs, learning_rate):
        """Training loop for DDPM"""
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate)
        l = nn.L1Loss()

        global_step = 0
        losses = []
        for epoch in range(num_epochs):
            train_mse = 0.0
            self.model.train()
            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for input_batch, output_batch in self.dataloader:
                ########################################
                optimizer.zero_grad()
                output_pred_batch = self.model(input_batch)
                loss = l(output_pred_batch, output_batch) / \
                    l(output_batch, torch.zeros_like(output_batch))
                loss.backward()
                train_mse += loss.item()
                ########################################

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                global_step += 1

            train_mse /= len(self.dataloader)

            if self.verbose:
                print("###",
                      "Epoch: ", epoch,
                      "Loss:", losses[-1],
                      "Train Loss:", train_mse.end,
                      "Global Step:", global_step,
                      "###")

            progress_bar.close()

        return losses
