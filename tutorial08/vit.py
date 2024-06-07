"""ViT model implementation"""
import torch
import torch.nn as nn
from transformer import TransformerBlock
from einops.layers.torch import Rearrange


def pair(t):
    """Returns a tuple of the same element if t is not a tuple, otherwise
    returns t"""
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    """ Takes an image of size (n, c, h, w)
    Finds patch sizes (p_h, p_w) & number of patches (n_h, n_w)
    NOTE: It must hold that h%p_h == 0

    1. Applies to_patch_embedding:
        a. (n, c, p_h*p1, p_w*p2) -> (n, n_h*n_w, p_h*p_w*c)
         b. LayerNorm
         c. Linear embedding p_h*p_w*c -> dim
         d. LayerNorm
     2. Add positional embedding
     3. Apply Transformer Block
     4. Depatchify
    """

    def __init__(self, image_size, patch_size,
                 dim, depth, heads, mlp_dim=256,
                 channels=1, dim_head=32, emb_dropout=0.,):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.patch_to_image = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      p1=patch_height, p2=patch_width, h=image_height // patch_height)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerBlock(
            dim, depth, heads, dim_head, mlp_dim)

        self.conv_last = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=3, padding=1)

    def forward(self, img):
        """Forward pass of the ViT model"""
        ###############################
        # Implement forward pass
        x = self.to_patch_embedding(img)
        x += self.pos_embedding
        x = self.transformer(x)
        x = self.patch_to_image(x)
        ###############################
        return x

    def print_size(self):
        """Prints the number of parameters of the model"""
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams}')

        return nparams
