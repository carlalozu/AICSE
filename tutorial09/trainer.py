
import torch
import torchvision
from myconv import MyTinyUNet
from denoiser import DDPM
from tqdm import tqdm


class Trainer():

    def __init__(self, num_timesteps, verbose=False):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dataloader = self.assemble_datasets()

        self.network = MyTinyUNet().to(self.device)

        self.model = DDPM(self.network, num_timesteps, beta_start=0.0001,
                          beta_end=0.02, device=self.device)

        if verbose:
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

        global_step = 0
        losses = []

        for epoch in range(num_epochs):
            self.model.train()
            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for _, batch in enumerate(self.dataloader):
                batch = batch[0].to(self.device)
                #
                # your code here
                #

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                global_step += 1

            progress_bar.close()
