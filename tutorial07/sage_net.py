"""Own SAGENet implementation"""
import torch
from sage_conv import SAGEConv
import torch.nn.functional as F

class SAGENet(torch.nn.Module):
    """Own SAGENet implementation"""

    def __init__(self, dataset, aggr='mean', hidden_channels=16):
        """Implement your own GraphSAGE (with mean/sum/max aggregation)
        message passing scheme as the core mechanism of your GNN."""

        super().__init__()
        ########################################################################
        self.in_channels = dataset.num_features
        self.out_channels = dataset.num_classes

        self.linear = torch.nn.Linear(self.in_channels, self.out_channels)

        self.conv1 = SAGEConv(self.in_channels, hidden_channels, aggr)
        self.conv2 = SAGEConv(hidden_channels, self.out_channels, aggr)

        self.activation = torch.nn.Tanh()
        ########################################################################

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the network"""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        """Data is an object of torch_geometric.data.Data class, we modify its x"""
        ########################################################################
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        ########################################################################
