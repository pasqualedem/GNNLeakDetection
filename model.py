#%% Enhanced Model Definition
import torch
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class LeakDetector(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=1, num_layers=4, edge_in=None):
        super().__init__()
        self.edge_in = edge_in
        self.encoder = torch.nn.ModuleList()
        self.encoder.append(GATConv(in_dim, hid_dim, heads=4, edge_dim=edge_in))  # GAT with attention
        self.encoder.append(BatchNorm(hid_dim*4))
        for _ in range(num_layers-2):
            self.encoder.append(GATConv(hid_dim*4, hid_dim, heads=4, edge_dim=edge_in))
            self.encoder.append(BatchNorm(hid_dim*4))
        self.encoder.append(GATConv(hid_dim*4, hid_dim, heads=1, edge_dim=edge_in))  # Single head for final
        
        self.decoder = torch.nn.ModuleList([
            GATConv(hid_dim, hid_dim, edge_dim=edge_in),
            GATConv(hid_dim, in_dim, edge_dim=edge_in)
        ])
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.edge_in else None
        for layer in self.encoder:
            if isinstance(layer, BatchNorm):
                x = layer(x)
            else:
                x = layer(x, edge_index, edge_attr)
                x = torch.relu(x)
                x = self.dropout(x)
        
        x_recon = x
        for conv in self.decoder[:-1]:
            x_recon = torch.relu(conv(x_recon, edge_index, edge_attr))
        return self.decoder[-1](x_recon, edge_index, edge_attr)