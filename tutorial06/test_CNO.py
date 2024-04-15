""" Let's check our code:"""
import torch
from functions import CNO_LReLu, CNOBlock, LiftProjectBlock, ResidualBlock, ResNet

X = torch.rand((1, 2, 128))


def test_CNO_LReLu():
    """Check your activation"""
    cno_lrelu = CNO_LReLu(in_size=128,
                          out_size=128)
    Y = cno_lrelu(X)
    assert tuple(Y.shape) == (1, 2, 128)


def test_CNOBlock():
    """Check your CNOBlock"""
    cno_block = CNOBlock(in_channels=2,
                         out_channels=4,
                         in_size=128,
                         out_size=64,
                         use_bn=True)
    Y = cno_block(X)
    assert tuple(Y.shape) == (1, 4, 64)


def test_LiftProjectBlock():
    """Check your LiftProjectBlock"""
    lift_project = LiftProjectBlock(in_channels=2,
                                    out_channels=4,
                                    size=128)
    Y = lift_project(X)
    assert tuple(Y.shape) == (1, 4, 128)


def test_ResidualBlock():
    """Check your ResidualBlock"""
    residual_bl = ResidualBlock(channels=2,
                                size=128,
                                use_bn=True)
    Y = residual_bl(X)
    assert tuple(Y.shape) == (1, 2, 128)


def test_ResNet():
    """Check your ResNet"""
    res_net = ResNet(channels=2,
                     size=128,
                     num_blocks=3,
                     use_bn=True)
    Y = res_net(X)
    assert tuple(Y.shape) == (1, 2, 128)
