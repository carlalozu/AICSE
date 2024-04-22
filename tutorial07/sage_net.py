import torch
from sage_conv import SAGEConv
import copy

class SAGENet(torch.nn.Module):
    def __init__(self, dataset, aggr='mean'):
        """Implement your own GraphSAGE (with mean/sum/max aggregation)
        message passing scheme as the core mechanism of your GNN."""

        super().__init__()
        ########################################################################
        #      START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)             #
        ########################################################################

        self.in_channels = dataset.num_features
        self.out_channels = dataset.num_classes

        self.linear = torch.nn.Linear(self.in_channels, self.out_channels)

        self.conv1 = SAGEConv(self.in_channels, 16, aggr)
        self.conv2 = SAGEConv(16, self.out_channels, aggr)

        self.activation = torch.nn.Tanh()
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        self.reset_parameters()


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        """Data is an object of torch_geometric.data.Data class, we modify its x"""

        ########################################################################
        #      START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)             #
        ########################################################################
        data.x = self.linear(data.x)
        data = self.conv1(data.x, data.edge_index)
        data.x = self.activation(data.x)
        data = self.conv2(data.x, data.edge_index)
        data.x = self.activation(data.x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
