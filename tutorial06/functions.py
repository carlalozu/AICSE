import torch
from torch import nn


class CNO_LReLu(nn.Module):
    """Activation Function"""

    def __init__(self,
                 in_size,
                 out_size
                 ):
        super(CNO_LReLu, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        """
        TO DO: Implement CNO activation function (3 steps)
               (1) Upsample the signal x.unsqueeze(2) to the resolution 2 x in_size
                   HINT: Use F.interpolate in 'bicubic' mode with antialis = True
               (2) Apply activation
               (3) Downsample the signal x to the resolution out_size (similar to (1))
               Don't forget to return x[:,:,0] --> You unsqueezed the signal in (1)
        """
        # Check you dimensions in the code block below (apply CNO_LReLu to a random signal)

        return None


class CNOBlock(nn.Module):
    """CNO Block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 use_bn=True
                 ):
        super(CNOBlock, self).__init__()

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
        TO DO: Implement CNOBlock forward
        Hint: Conv -> BN -> Activation
        """
        # Check you dimensions in the code block below (apply CNOBlock to a random signal)

        return None


class LiftProjectBlock(nn.Module):
    """LiftProject Block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 size,
                 latent_dim=64
                 ):
        super(LiftProjectBlock, self).__init__()

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
        TO DO: Implement LiftProjectBlock forward
        Hint: inter_CNOBlock -> Conv
        """
        # Check you dimensions in the code block below (apply LiftProjectBlock to a random signal)

        return None


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self,
                 channels,
                 size,
                 use_bn=True
                 ):
        super(ResidualBlock, self).__init__()

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
        TO DO: Implement ResidualBlock forward
        Hint: Conv -> BN -> Activation -> Conv -> BN -> Skip Connection
        """
        # Check you dimensions in the code block below (apply ResidualBlock to a random signal)

        return None


class ResNet(nn.Module):
    """ResNet"""

    def __init__(self,
                 channels,
                 size,
                 num_blocks,
                 use_bn=True
                 ):
        super(ResNet, self).__init__()

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
        TO DO: Implement ResNet forwards
        Hint: Apply ResidualBlocks num_blocks time
        """
        # Check you dimensions in the code block below (apply ResNet to a random signal)

        return None
