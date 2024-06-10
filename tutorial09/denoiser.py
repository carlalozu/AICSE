import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class DDPM(nn.Module):
    def __init__(
            self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=None) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (
            1 - self.alphas_cumprod) ** 0.5  # used in add_noise and step

    def add_noise(self, x_start, noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        #
        # your code here
        pass

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)

    def step(self, predicted_noise, timestep, sample):
        # one step of sampling
        # timestep (1)
        #
        # your code here
        pass

    def generate_image(self, sample_size=100, channel=1, size=32):
        """Generate the image from the Gaussian noise"""

        frames = []
        frames_mid = []
        self.eval()
        with torch.no_grad():
            timesteps = list(range(self.num_timesteps))[::-1]
            sample = torch.randn(sample_size, channel, size, size)

            for i, t in enumerate(tqdm(timesteps)):
                #
                # your code here
                #

                if t == 500:
                    for i in range(sample_size):
                        frames_mid.append(sample[i].detach().cpu())

            for i in range(sample_size):
                frames.append(sample[i].detach().cpu())
        return frames, frames_mid

    @staticmethod
    def show_images(images, title=""):
        """Shows the provided images as sub-pictures in a square"""

        images = [im.permute(1, 2, 0).numpy() for im in images]

        # Defining number of rows and columns
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        # Populating figure with sub-plots
        idx = 0
        for _ in range(rows):
            for _ in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx], cmap="gray")
                    plt.axis('off')
                    idx += 1
        fig.suptitle(title, fontsize=30)

        # Showing the figure
        plt.show()
