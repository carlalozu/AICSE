"""
In the following module, we will define
* **CNO LReLu** activation fucntion
* **CNO building block** (CNOBlock) → Conv1d - BatchNorm - Activation
* **Lift/Project Block** (Important for embeddings)
* **Residual Block** → Conv1d - BatchNorm - Activation - Conv1d - BatchNorm - *Skip Connection*
* **ResNet** → Stacked ResidualBlocks (several blocks applied iteratively)
"""
import torch
from torch import nn


class CNO_LReLu(nn.Module):
    """Activation Function"""

    def __init__(self,
                 in_size,
                 out_size
                 ):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        """
        Implement CNO activation function (3 steps)
            (1) Upsample the signal x.unsqueeze(2) to the resolution 2 x in_size
                Use F.interpolate in 'bicubic' mode with antialis = True
            (2) Apply activation
            (3) Downsample the signal x to the resolution out_size (similar to (1))
            Don't forget to return x[:,:,0] --> You unsqueezed the signal in (1)
        """

        x = x.unsqueeze(2)
        x = nn.functional.interpolate(x, size = (1,2 * self.in_size), mode='bicubic', antialias=True)
        x = self.act(x)
        x = nn.functional.interpolate(x, size = (1,self.out_size), mode='bicubic', antialias=True)
        return x[:,:,0]


class CNOBlock(nn.Module):
    """CNO Block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 use_bn=True
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size

        # -----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = torch.nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=3,
                                           padding=1)

        if use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()
        self.act = CNO_LReLu(in_size=self.in_size,
                             out_size=self.out_size)

    def forward(self, x):
        """
        Implement CNOBlock forward
            Conv -> BN -> Activation
        """
        x = self.convolution(x)
        x = self.batch_norm(x)
        x = self.act(x)

        return x


class LiftProjectBlock(nn.Module):
    """LiftProject Block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 size,
                 latent_dim=64
                 ):
        super().__init__()

        self.inter_CNOBlock = CNOBlock(in_channels=in_channels,
                                       out_channels=latent_dim,
                                       in_size=size,
                                       out_size=size,
                                       use_bn=False)

        self.convolution = torch.nn.Conv1d(in_channels=latent_dim,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           padding=1)

    def forward(self, x):
        """
        Implement LiftProjectBlock forward
         inter_CNOBlock -> Conv
        """
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)

        return x


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self,
                 channels,
                 size,
                 use_bn=True
                 ):
        super().__init__()

        self.channels = channels
        self.size = size

        # -----------------------------------------

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation

        self.convolution1 = torch.nn.Conv1d(in_channels=self.channels,
                                            out_channels=self.channels,
                                            kernel_size=3,
                                            padding=1)
        self.convolution2 = torch.nn.Conv1d(in_channels=self.channels,
                                            out_channels=self.channels,
                                            kernel_size=3,
                                            padding=1)

        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(self.channels)
            self.batch_norm2 = nn.BatchNorm1d(self.channels)

        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNO_LReLu(in_size=self.size,
                             out_size=self.size)

    def forward(self, x):
        """
        Implement ResidualBlock forward
          Conv -> BN -> Activation -> Conv -> BN -> Skip Connection
        """
        x_ = self.convolution1(x)
        x_ = self.batch_norm1(x_)
        x_ = self.act(x_)

        x_ = self.convolution2(x_)
        x_ = self.batch_norm2(x_)
        x_ = self.act(x_)

        x = x_ + x
        return x


class ResNet(nn.Module):
    """ResNet"""

    def __init__(self,
                 channels,
                 size,
                 num_blocks,
                 use_bn=True
                 ):
        super().__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels=channels,
                                               size=size,
                                               use_bn=use_bn))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        """
        Implement ResNet forwards
          Apply ResidualBlocks num_blocks time
        """
        for res_net in self.res_nets:
            x = res_net(x)

        return x
