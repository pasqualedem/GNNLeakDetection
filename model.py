#%% Enhanced Model Definition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm


class FirstLayer(nn.Module):
    def __init__(self, node_in, window_size, hidden_size, lstm_layers=None, edge_in=None):
        super(FirstLayer, self).__init__()
        
        # LSTM layer or GATv2Conv
        self.window_size = window_size
        self.lstm_layers = lstm_layers
        if lstm_layers is not None:
            self.layer = nn.LSTM(input_size=node_in // window_size, 
                                hidden_size=hidden_size*4, 
                                num_layers=lstm_layers, 
                                batch_first=True)
        else:
            self.layer = GATv2Conv(node_in, hidden_size, heads=4, edge_dim=edge_in)
        
    
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


class AnomalyLeakDetector(torch.nn.Module):
    def __init__(self, node_in, hid_dim=None, num_layers=4, hidden_dims=None, edge_in=None, decoder_dims=None, lstm_layers=None, window_size=None, **kwargs):
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
            self.encoder.append(GATv2Conv(hidden_dims[i-1] * 4, hidden_dims[i], heads=4, edge_dim=edge_in))
            self.encoder.append(BatchNorm(hidden_dims[i] * 4))
        
        # Final layer (single head)
        self.encoder.append(GATv2Conv(hidden_dims[-1] * 4, hidden_dims[-1], heads=1, edge_dim=edge_in))
        
        # Decoder (reverse of encoder)
        self.decoder = torch.nn.ModuleList()
        for i in range(len(decoder_dims[:-1])):
            self.decoder.append(GATv2Conv(decoder_dims[i], decoder_dims[i+1], edge_dim=edge_in))
            self.decoder.append(BatchNorm(decoder_dims[i+1]))
        
        # Final decoder layer to reconstruct input
        self.decoder.append(GATv2Conv(decoder_dims[-1], node_in, edge_dim=edge_in))
        
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


class GATConvModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden_size=32, target_size=1, heads=1, dropout=0.0, num_layers=2, graph_classification=False, **kwargs):
        super().__init__()
        
        self.graph_classification = graph_classification
        self.node_encoder = nn.Linear(node_in, hidden_size)
        if edge_in:
            self.edge_encoder = nn.Linear(edge_in, hidden_size)
        
        self.hidden_size = hidden_size
        self.num_features = node_in
        self.num_edge_features = edge_in
        self.target_size = target_size
        self.num_layers = num_layers
        
        # Dynamically create GATConv layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(self.hidden_size, self.hidden_size, edge_dim=hidden_size, dropout=dropout, residual=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(self.hidden_size, self.hidden_size, edge_dim=hidden_size, heads=heads, dropout=dropout, residual=True))
        
        self.convs.append(GATConv(self.hidden_size, self.hidden_size, edge_dim=hidden_size, heads=heads, dropout=dropout, residual=True))

        if self.graph_classification:
            # Additional layers
            self.graph_norm = nn.LayerNorm(self.hidden_size*heads)
            self.graph_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.linear = nn.Linear(self.hidden_size*heads, self.target_size)
        
        for param in self.parameters():
            if param.dim() > 1:  # Apply to weights (not biases)
                nn.init.kaiming_normal_(param)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_encoder(x)
        if self.num_edge_features is not None:
            edge_attr = self.edge_encoder(edge_attr)
        

        for i, conv in enumerate(self.convs):
            if self.num_edge_features is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)  # Add edge features here
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Apply activation and dropout for all but the last layer
                x = F.leaky_relu(x)
                x = F.dropout(x, training=self.training)

        if self.graph_classification:
            x = global_mean_pool(x, data.batch)
            x = self.graph_norm(x)
            x = self.graph_act(x)
        x = self.linear(x)
        return x.squeeze(1)



MODELS = {
    "AnomalyLeakDetector": AnomalyLeakDetector,
    "GNNLeakDetector": GATConvModel,
}


def get_model(name, node_in, edge_in, **kwargs):
    class_name = MODELS[name]
    
    return class_name(
        node_in=node_in,
        edge_in=edge_in,
        **kwargs
    )