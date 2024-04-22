import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(2 * in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.weight.size(0), self.weight)

    def forward(self, x, edge_index):
        """Fprward pass of the SAGEConv layer"""
        ########################################################################
        #      START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)             #
        ########################################################################
        return self.propagate(edge_index, x=x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################



    def message(self, x_j, edge_weight):
        ########################################################################
        #      START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)             #
        ########################################################################
        return self.message(x_j, edge_weight)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################



    def update(self, aggr_out, x):
        """Update node embedding"""
        ########################################################################
        #      START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)             #
        ########################################################################
        return self.update(aggr_out, x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

