#%% Enhanced Model Definition
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm


class FirstLayer(nn.Module):
    def __init__(self, node_in, window_size, hidden_size, lstm_layers=None, edge_in=None):
        super(FirstLayer, self).__init__()
        
        # LSTM layer or GATConv
        self.window_size = window_size
        self.lstm_layers = lstm_layers
        if lstm_layers is not None:
            self.layer = nn.LSTM(input_size=node_in // window_size, 
                                hidden_size=hidden_size*4, 
                                num_layers=lstm_layers, 
                                batch_first=True)
        else:
            self.layer = GATConv(node_in, hidden_size, heads=4, edge_dim=edge_in)
        
    
    def forward(self, x, edge_index, edge_attr):
        # Input x shape: (batch_size, L, node_in)
        
        # Reshape input for LSTM (if needed)
        # Assuming x is already in the shape (batch_size, L, node_in)
        
        if self.lstm_layers is not None:
            x = x.view(x.shape[0], self.window_size, -1)
            out = self.layer(x)  # lstm_out shape: (batch_size, L, hidden_size)
            lstm_out, (h_n, c_n) = out
            out = h_n[-1, :, :]  # Shape: (batch_size, hidden_size)
        else:
            out = self.layer(x, edge_index, edge_attr)
        
        return out


class LeakDetector(torch.nn.Module):
    def __init__(self, node_in, hid_dim=1, num_layers=4, hidden_dims=None, edge_in=None, decoder_dims=None, lstm_layers=None, window_size=None):
        super().__init__()
        self.edge_in = edge_in
        
        if hid_dim is not None and hidden_dims is not None:
            raise ValueError("Cannot specify both hid_dim and hidden_dims")
        
        if hidden_dims is None:
            hidden_dims = [hid_dim] * num_layers
            
        decoder_dims = decoder_dims or hidden_dims[::-1]
        
        # Encoder
        self.encoder = torch.nn.ModuleList()
        self.encoder.append(FirstLayer(node_in=node_in, hidden_size=hidden_dims[0], edge_in=edge_in, window_size=window_size, lstm_layers=lstm_layers))  # First layer
        self.encoder.append(BatchNorm(hidden_dims[0] * 4))  # BatchNorm after first layer
        
        # Intermediate layers
        for i in range(1, len(hidden_dims)):
            self.encoder.append(GATConv(hidden_dims[i-1] * 4, hidden_dims[i], heads=4, edge_dim=edge_in))
            self.encoder.append(BatchNorm(hidden_dims[i] * 4))
        
        # Final layer (single head)
        self.encoder.append(GATConv(hidden_dims[-1] * 4, hidden_dims[-1], heads=1, edge_dim=edge_in))
        
        # Decoder (reverse of encoder)
        self.decoder = torch.nn.ModuleList()
        for i in range(len(decoder_dims[:-1])):
            self.decoder.append(GATConv(decoder_dims[i], decoder_dims[i-1], edge_dim=edge_in))
            self.decoder.append(BatchNorm(decoder_dims[i-1]))
        
        # Final decoder layer to reconstruct input
        self.decoder.append(GATConv(decoder_dims[-1], node_in, edge_dim=edge_in))
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.edge_in else None
        
        # Encoder forward pass
        for layer in self.encoder:
            if isinstance(layer, BatchNorm):
                x = layer(x)
            else:
                x = layer(x, edge_index, edge_attr)
                x = torch.relu(x)
                x = self.dropout(x)
        
        # Decoder forward pass
        x_recon = x
        for i, layer in enumerate(self.decoder[:-1]):
            if isinstance(layer, BatchNorm):
                x_recon = layer(x_recon)
            else:
                x_recon = torch.relu(layer(x_recon, edge_index, edge_attr))
        
        # Final reconstruction layer (no activation)
        return self.decoder[-1](x_recon, edge_index, edge_attr)