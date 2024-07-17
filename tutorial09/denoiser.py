"""Denoiser module"""
import torch
from torch import nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from myconv import MyTinyUNet


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""

    def __init__(
            self, num_timesteps, beta_start=0.0001, beta_end=0.02, device=None) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32)  # schedule for beta
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (
            1 - self.alphas_cumprod) ** 0.5  # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        """The forward process"""
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)

        ########################################
#         mu = 1/self.alphas[timesteps]*(x_start-(1-self.alphas[timesteps]) /
#                                        self.sqrt_one_minus_alphas_cumprod[timesteps]*x_noise)
#         gamma = self.beta[timesteps]*(1-self.alphas_cumprod[timesteps-1]) / \
#             (1-self.alphas_cumprod[timesteps])
# 
#         return mu + gamma**0.5*torch.randn_like(x_start)
    
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise
        ########################################

    def step(self, predicted_noise, timestep, sample):
        """One step of sampling"""
        # timestep (1)
        # predicted_noise is epsilon_not

        ########################################
        inner_term = sample - \
            (1-self.alphas[timestep]) / \
            self.sqrt_one_minus_alphas_cumprod[timestep]
        inner_term_t = inner_term.reshape(-1,1,1,1)[timestep]

        coeff_term_t = (1/self.alphas**0.5).reshape(-1,1,1,1)[timestep]

        epsilon = 0
        if timestep:
            epsilon = torch.rand_like(predicted_noise)
        wiener_process = self.betas[timestep]**0.5 * epsilon

        return coeff_term_t*(sample-inner_term_t*predicted_noise)+wiener_process
        ########################################

    def generate_image(self, sample_size=100, channel=1, size=32):
        """Generate the image from the Gaussian noise"""

        frames = []
        frames_mid = []
        self.eval()
        with torch.no_grad():
            timesteps = list(range(self.num_timesteps))[::-1]
            sample = torch.randn(sample_size, channel, size, size)
            sample_denoised = sample
            for i, t in enumerate(tqdm(timesteps)):
                ########################################
                sample = self.step(sample_denoised, t, sample)
                ########################################

                if t == 500:
                    for i in range(sample_size):
                        frames_mid.append(sample_denoised[i].detach().cpu())

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
