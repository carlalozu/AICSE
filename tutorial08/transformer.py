"""Transformer model implementation"""
import torch
from torch import nn
from einops import rearrange


class TransformerBlock(nn.Module):
    """Takes sequence of embeddings of dimension dim
    1. Applies depth times:
       a) Attention block: dim->dim (in the last dimension)
       b) MLP block:       dim->dim (in the last dimension)
    2. Applies LayerNorm
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionBlock(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        """Forward pass of the transformer block"""
        ###############################
        # Implement forward pass
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        ###############################
        return self.norm(x)


class AttentionBlock(nn.Module):
    """Takes a sequence of embedding of dimension dim
    1. Applies LayerNorm
    2. Applies linear layer dim -> 3x inner_dim
        NOTE: inner_dim = dim_head x heads
    3. Applies attention
    4. Projects inner -> dim
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """Forward pass of the attention block"""
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    """ Appplies following operations:
    1. LayerNorm
    2. nn.Linear (dim --> hidden_dim)
    3. GELU
    4. Dropout
    5. nn.Linear (hidden_dim --> dim)
    6. Dropout
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            #############################################
            # Implement sequence of 6 operations
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
            #############################################
        )

    def forward(self, x):
        """Forward pass of the feed forward block"""
        return self.net(x)
