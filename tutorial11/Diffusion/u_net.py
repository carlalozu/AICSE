"""UNet model"""
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn


def relu_layer(x, key=None):
    return jax.nn.relu(x)


class UNet(eqx.Module):
    down_conv_blocks: list
    max_pools: list
    conv_transposes: list
    up_conv_blocks: list
    conv_final: nn.Conv2d
    n_levels: int

    def __init__(self, in_channels, out_channels, n_hidden, n_levels, n_width, key):

        key1, key2, key3, key4 = jr.split(key, 4)
        self.n_levels = n_levels

        # downsampling and middle convolutional blocks
        self.down_conv_blocks = [
            nn.Sequential(
                ([nn.Conv2d(n_hidden*(2**(il-1)) if il != 0 else in_channels, n_hidden*(2**(il)), kernel_size=3, stride=1, padding=1, key=key1)] +
                 [relu_layer] +
                 ([nn.Sequential(
                     ([nn.Conv2d(n_hidden*(2**(il)), n_hidden*(2**(il)), kernel_size=3, stride=1, padding=1, key=key2)] +
                      [relu_layer])
                 ) for key1, key2 in jr.split(key, (n_width, 2))])
                 )
            )
            for il, key in enumerate(jr.split(key1, n_levels+1))]

        # pooling operations
        self.max_pools = [
            nn.MaxPool2d(kernel_size=2, stride=2)
            for il in range(n_levels)]

        # transposed convolutions
        self.conv_transposes = [
            nn.ConvTranspose2d(n_hidden*(2**(n_levels-il)), n_hidden *
                               (2**(n_levels-il-1)), kernel_size=2, stride=2, key=key)
            for il, key in enumerate(jr.split(key2, n_levels))]

        # upsampling convolutional blocks
        self.up_conv_blocks = [
            nn.Sequential(
                ([nn.Conv2d(n_hidden*(2**(n_levels-il)), n_hidden*(2**(n_levels-il-1)), kernel_size=3, stride=1, padding=1, key=key1)] +
                 [relu_layer] +
                 ([nn.Sequential(
                     ([nn.Conv2d(n_hidden*(2**(n_levels-il-1)), n_hidden*(2**(n_levels-il-1)), kernel_size=3, stride=1, padding=1, key=key2)] +
                      [relu_layer])
                 ) for key1, key2 in jr.split(key, (n_width, 2))])
                 )
            )
            for il, key in enumerate(jr.split(key3, n_levels))]

        self.conv_final = nn.Conv2d(
            n_hidden, out_channels, kernel_size=1, stride=1, key=key4)

    def __call__(self, x, t, t1):
        print(x.shape, t.shape)
        # simply add time as an extra channel
        x = jnp.concatenate([x, jnp.tile(t/t1, x.shape)], axis=0)
        print(x.shape)

        # downsampling blocks
        skips = []
        for il in range(self.n_levels):
            x = self.down_conv_blocks[il](x)
            skips.append(x)
            x = self.max_pools[il](x)
            print(x.shape)

        # middle block
        x = self.down_conv_blocks[-1](x)
        print(x.shape)

        # upsampling blocks
        for il in range(self.n_levels):
            x = self.conv_transposes[il](x)
            x = jnp.concatenate([x, skips[self.n_levels-il-1]], axis=0)
            x = self.up_conv_blocks[il](x)
            print(x.shape)

        x = self.conv_final(x)
        print(x.shape)

        return x
