"""SAGEConv implementation"""
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

class SAGEConv(MessagePassing):
    """SAGEConv layer"""

    def __init__(self, in_channels, out_channels, aggr):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(2 * in_channels, out_channels))

        self.reset_parameters()

    @staticmethod
    def uniform(size, tensor):
        """Initialize tensor uniformly"""
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        """Reset the parameters of the layer"""
        self.uniform(self.weight.size(0), self.weight)

    def forward(self, x, edge_index):
        """Forward pass of the SAGEConv layer"""
        ########################################################################
        return self.propagate(edge_index, x=x)
        ########################################################################

    def message(self, x_j, edge_weight):
        """Message passing"""
        ########################################################################
        return x_j if edge_weight else edge_weight.view(-1, 1) * x_j
        ########################################################################

    def update(self, aggr_out, x):
        """Update node embedding"""
        ########################################################################
        aggr_out = torch.cat([x, aggr_out], dim=-1)
        aggr_out = torch.matmul(aggr_out, self.weight)
        aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        ########################################################################
        return aggr_out