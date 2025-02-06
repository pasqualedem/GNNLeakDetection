import torch
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import einops
from scipy.fft import fft
from scipy.stats import entropy
import statsmodels.api as sm

import lovely_tensors as lt
lt.monkey_patch()


def make_undirected(graph_data):
    graph_data["edge_attr"] = torch.cat([graph_data["edge_attr"], graph_data["edge_attr"]], dim=1)
    edges_reversed = torch.stack([reversed(e) for e in graph_data["edge_index"].reshape(-1, 2)])
    edge_index = torch.cat([graph_data["edge_index"], edges_reversed], dim=0)
    graph_data["edge_index"] = edge_index
    return graph_data


def permutation_entropy(time_series, m=3, tau=1):
    """Compute permutation entropy for a time series."""
    results = []
    for series in time_series:
        permutations = []
        n = len(series)

        for i in range(n - (m - 1) * tau):
            vec = series[i:i + m * tau:tau]
            permutation = np.argsort(vec)
            permutations.append(tuple(permutation))

        unique_perms, counts = np.unique(permutations, axis=0, return_counts=True)
        probs = counts / len(permutations)
        results.append(entropy(probs, base=2))
    results = np.array(results)
    return results

def fourier_entropy(time_series):
    """Compute Fourier entropy for a time series."""
    fft_coeffs = fft(time_series)
    power_spectrum = np.abs(fft_coeffs) ** 2
    normalized_spectrum = power_spectrum / np.sum(power_spectrum)
    return entropy(normalized_spectrum, base=2, axis=-1)

def partial_autocorrelation(time_series, lag):
    """Compute partial autocorrelation for a time series at a given lag."""
    if time_series.ndim == 2:
        # If input is 2D (batch of time series), compute PACF for each time series
        pacf_values = np.array([sm.tsa.stattools.pacf(ts, nlags=lag)[lag] for ts in time_series])
        return pacf_values
    elif time_series.ndim == 1:
        # If input is 1D (single time series), compute PACF directly
        return sm.tsa.stattools.pacf(time_series, nlags=lag)[lag]
    else:
        raise ValueError("Input must be 1D or 2D.")

def extract_features_from_columns(features, window_size, stride, subsample):
    """Process features into [windows, nodes] format."""
    # Slice data into overlapping windows
    features = features.unfold(0, window_size, stride).float()
    
    # Compute mean, max, min, std for each window
    mean_values = features.mean(dim=2)  # Mean pooling
    max_values = features.max(dim=2).values  # Max pooling
    min_values = features.min(dim=2).values  # Min pooling
    std_values = features.std(dim=2)  # Standard deviation pooling
    
    # Compute additional features for each window
    perm_entropy_values = torch.tensor([permutation_entropy(window) for window in tqdm(features.numpy(), desc="permutation entropy")])
    fourier_entropy_values = torch.tensor([fourier_entropy(window) for window in tqdm(features.numpy(), desc="fourier entropy")])
    pacf_values = torch.tensor([partial_autocorrelation(window, lag=0) for window in tqdm(features.numpy(), desc="partial autocorrelation entropy")])
    
    if fourier_entropy_values.isnan().any():
        print("Fourier entropy contains NaN values, replacing with 0")
        fourier_entropy_values = fourier_entropy_values.nan_to_num()
    # Stack all features together
    features = torch.stack((mean_values, max_values, min_values, std_values, 
                            perm_entropy_values, fourier_entropy_values, pacf_values), dim=1)
    
    # Rearrange features into [windows, nodes] format
    features = einops.rearrange(features, "l f n -> l (f n)")
    
    # Subsample the windows
    return features[::subsample, :]


def sliding_window(tensor, window_size):
    """
    Apply a sliding window operation on the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [N, F].
        window_size (int): The size of the sliding window.

    Returns:
        torch.Tensor: Output tensor of shape [N, window_size * F].
    """
    N, F = tensor.shape
    # Create a padded tensor to handle the boundary conditions
    padded_tensor = torch.zeros((N + window_size - 1, F), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[window_size - 1:] = tensor  # Place the original tensor at the end of the padded tensor

    # Initialize the output tensor
    output_tensor = torch.zeros((N, window_size * F), dtype=tensor.dtype, device=tensor.device)

    # Fill the output tensor with the sliding window values
    for i in range(N):
        # Get the window for the current time step
        window = padded_tensor[i:i + window_size, :]
        # Flatten the window and store it in the output tensor
        output_tensor[i, :] = window.flatten()

    return output_tensor


def feature_extraction(graph_data, window_size, stride, subsample):
    # Process features and labels
    node_features = graph_data['node_features']
    edge_features = graph_data['edge_attr']
    print("Extracting Node Fetures")
    node_features = extract_features_from_columns(node_features, window_size, stride, subsample)
    print("Extracting Edge Fetures")
    # edge_features = einops.rearrange(graph_data['edge_attr'], "l e -> (l e)")
    edge_features = extract_features_from_columns(graph_data['edge_attr'], window_size, stride, subsample)
    window_labels = graph_data['y'].unfold(0, window_size, stride).max(dim=2).values[::subsample, :]
    window_scenarios = graph_data['scenario'].unfold(0, window_size, stride).mode(dim=2).values[::subsample, :]
    print("Processed node features:")
    print(node_features)
    print("Processed edge features:")
    print(edge_features)
    print("Processed labels:")
    print(window_labels)
    print("Processed scenarios:")
    print(window_scenarios)
    
    return {
        'node_features': node_features,
        'window_labels': window_labels,
        "edge_features": edge_features,
        'window_scenarios': window_scenarios,
        "edge_index": graph_data['edge_index'].long().t().contiguous(),
    }
    
@click.command()
@click.argument('window_size', type=int)
@click.argument('stride', type=int)
@click.option('--subsample', default=1, type=int, help="Subsample value (default: 1)")
@click.option('--doublewindow_size', default=None, type=int, help="Double window size (optional)")
@click.option('--subsize', default=None, type=int, help="Subsize (optional)")
def main(window_size, stride, subsample=1, doublewindow_size=None, subsize=None):
    print(f"Using window size {window_size}, stride {stride}, subsample {subsample} and double window size {doublewindow_size}")
    
    #%% Data Loading and Preprocessing
    print("Loading and preprocessing data...")
    graph_data = torch.load('data/graph_data_torch.pt')
    num_nodes = graph_data['node_features'].shape[1]
    num_edges = graph_data['edge_index'].shape[0]
    
    print("Graph_data:")
    for k, v in graph_data.items():
        print(f"{k}: {v}")
    
    graph_data = make_undirected(graph_data)
    num_nodes = graph_data['node_features'].shape[1]
    num_edges = graph_data['edge_index'].shape[0]
    print("Graph_data after making edges undirected:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(graph_data)
    
    if subsize is not None:
        print(f"\nSubsampling to {subsize}")
        graph_data['node_features'] = graph_data['node_features'][:subsize]
        graph_data['edge_attr'] = graph_data['edge_attr'][:subsize]
        graph_data['scenario'] = graph_data['scenario'][:subsize]
        graph_data['y'] = graph_data['y'][:subsize]
        print(f"Subsized node features: {graph_data['node_features']}")
        print(f"Subsized edge features: {graph_data['edge_attr']}")
        print(f"Subsized scenarios: {graph_data['scenario']}")
        print(f"Subsized labels: {graph_data['y']}")
    
    graph_data = feature_extraction(graph_data, window_size, stride, subsample)
    
    if doublewindow_size is not None:
        print(f"\nMaking double windows of size {doublewindow_size}")
        graph_data['node_features'] = sliding_window(graph_data['node_features'], doublewindow_size)
        graph_data['edge_features'] = sliding_window(graph_data['edge_features'], doublewindow_size)
        print("New node features:")
        print(graph_data['node_features'])
        print("New edge features:")
        print(graph_data['edge_features'])
    
    print("\nSaving processed data...")
    double_string = "" if doublewindow_size is None else f"_2W{doublewindow_size}"
    subsize_string = "" if subsize is None else f"_SUBSIZE{subsize}"
    torch.save(graph_data, f"data/f_extracted_data_W{window_size}{double_string}_S{subsample}_STRIDE{stride}{subsize_string}.pt" )
    print("Data saved successfully!")
    
    
if __name__ == "__main__":
    main()