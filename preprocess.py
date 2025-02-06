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
import yaml
lt.monkey_patch()


def make_undirected(graph_data):
    if "edge_attr" in graph_data:
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

def extract_features_from_columns(features, window_size, stride, subsample, features_list):
    """Process features into [windows, nodes] format."""
    # Slice data into overlapping windows
    features = features.unfold(0, window_size, stride).float()
    extracted_features = []
    
    # Compute mean, max, min, std for each window
    if "mean" in features_list:
        mean_values = features.mean(dim=2)  # Mean pooling
        extracted_features.append(mean_values)
    if "max" in features_list:
        max_values = features.max(dim=2).values  # Max pooling
        extracted_features.append(max_values)
    if "min" in features_list:
        min_values = features.min(dim=2).values  # Min pooling
        extracted_features.append(min_values)
    if "std" in features_list:
        std_values = features.std(dim=2)  # Standard deviation pooling
        extracted_features.append(std_values)
    
    # Compute additional features for each window
    if "perm_entropy" in features_list:
        perm_entropy_values = torch.tensor([permutation_entropy(window) for window in tqdm(features.numpy(), desc="permutation entropy")])
        extracted_features.append(perm_entropy_values)
    if "fourier_entropy" in features_list:
        fourier_entropy_values = torch.tensor([fourier_entropy(window) for window in tqdm(features.numpy(), desc="fourier entropy")])
        if fourier_entropy_values.isnan().any():
            print("Fourier entropy contains NaN values, replacing with 0")
            fourier_entropy_values = fourier_entropy_values.nan_to_num()
        extracted_features.append(fourier_entropy_values)
    if "pacf" in features_list:
        pacf_values = torch.tensor([partial_autocorrelation(window, lag=0) for window in tqdm(features.numpy(), desc="partial autocorrelation entropy")])
        extracted_features.append(pacf_values)
    
    # Stack all features together
    features = torch.cat(extracted_features, dim=1)
    
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


def feature_extraction(graph_data, window_size, stride, subsample, features):
    # Process features and labels
    node_features = graph_data['node_features']
    print("Extracting Node Fetures")
    node_features = extract_features_from_columns(node_features, window_size, stride, subsample, features)
    if "edge_attr" in graph_data:
        print("Extracting Edge Fetures")
        edge_features = graph_data['edge_attr']
        edge_features = extract_features_from_columns(graph_data['edge_attr'], window_size, stride, subsample, features)
    window_labels = graph_data['y'].unfold(0, window_size, stride).max(dim=2).values[::subsample, :]
    window_scenarios = graph_data['scenario'].unfold(0, window_size, stride).mode(dim=2).values[::subsample, :]
    print("Processed node features:")
    print(node_features)
    if "edge_attr" in graph_data:
        print("Processed edge features:")
        print(edge_features)
    print("Processed labels:")
    print(window_labels)
    print("Processed scenarios:")
    print(window_scenarios)
    
    new_graph = {
        'node_features': node_features,
        'window_labels': window_labels,
        'window_scenarios': window_scenarios,
        "edge_index": graph_data['edge_index'].long().t().contiguous(),
    }
    if "edge_attr" in graph_data:
        new_graph["edge_features"] = edge_features
    
    return new_graph
    
@click.command()
@click.option('--parameters', type=str, help="Parameters file")
def main(parameters):
    with open(parameters, "r") as f:
        hyperparams = yaml.safe_load(f)
    window_size = hyperparams['window_size']
    stride = hyperparams['stride']
    subsample = hyperparams['subsample']
    subsize = hyperparams['subsize']
    doublewindow_size = hyperparams['doublewindow_size']
    data_path = hyperparams['data_path']
    suffix = hyperparams.get('suffix', '')
    features = hyperparams['features']
    
    print(f"Using window size {window_size}, stride {stride}, subsample {subsample} and double window size {doublewindow_size}")
    
    #%% Data Loading and Preprocessing
    print("Loading and preprocessing data...")
    graph_data = torch.load(data_path)
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
        if "edge_attr" in graph_data:
            graph_data['edge_attr'] = graph_data['edge_attr'][:subsize]
        graph_data['scenario'] = graph_data['scenario'][:subsize]
        graph_data['y'] = graph_data['y'][:subsize]
        print(f"Subsized node features: {graph_data['node_features']}")
        if "edge_attr" in graph_data:
            print(f"Subsized edge features: {graph_data['edge_attr']}")
        print(f"Subsized scenarios: {graph_data['scenario']}")
        print(f"Subsized labels: {graph_data['y']}")
    
    graph_data = feature_extraction(graph_data, window_size, stride, subsample, features)
    
    if doublewindow_size is not None:
        print(f"\nMaking double windows of size {doublewindow_size}")
        graph_data['node_features'] = sliding_window(graph_data['node_features'], doublewindow_size)
        if "edge_features" in graph_data:
            graph_data['edge_features'] = sliding_window(graph_data['edge_features'], doublewindow_size)
        print("New node features:")
        print(graph_data['node_features'])
        if "edge_features" in graph_data:
            print("New edge features:")
            print(graph_data['edge_features'])
    
    print("\nSaving processed data...")
    double_string = "" if doublewindow_size is None else f"_2W{doublewindow_size}"
    subsize_string = "" if subsize is None else f"_SUBSIZE{subsize}"
    data_name = f"data/processed_data_W{window_size}{double_string}_S{subsample}_STRIDE{stride}{subsize_string}{suffix}.pt"
    torch.save(graph_data, data_name)
    print(f"Data saved successfully to {data_name}")
    
    
if __name__ == "__main__":
    main()