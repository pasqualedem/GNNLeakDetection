from enum import StrEnum
import os
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
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

from logger import get_logger
lt.monkey_patch()


def train_val_test_split(window_labels, window_scenarios, logger, anomaly=False):    
    scenarios = window_scenarios.unique()
    leak_scenarios = window_scenarios[window_labels.any(dim=1)].unique()
    non_leak_scenarios = set(scenarios.tolist()) - set(leak_scenarios.tolist())

    VAL_SCENARIOS, TEST_SCENARIOS = 100, 100

    # Set random seed
    np.random.seed(42)
    
    split_file = "data/split.txt"
    # Load data split if exists
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            train_scenarios = eval(f.readline().split(":")[1])
            val_scenarios = eval(f.readline().split(":")[1])
            test_scenarios = eval(f.readline().split(":")[1])
        logger.info(f"Loaded data split from {split_file}")
    else:
        test_scenarios = np.random.choice(
            list(leak_scenarios), TEST_SCENARIOS, replace=False
        )
        val_scenarios = np.random.choice(
            list(non_leak_scenarios - set(test_scenarios.tolist())),
            VAL_SCENARIOS,
            replace=False,
        )
        train_scenarios = list(
            set(scenarios.tolist())
            - set(test_scenarios.tolist())
            - set(val_scenarios.tolist())
        )
        # Save split to txt file
        logger.info(f"Saving data split to {split_file}")
        with open(split_file, "w") as f:
            f.write(f"train: {sorted(list(train_scenarios))}\n")
            f.write(f"val: {sorted(val_scenarios.tolist())}\n")
            f.write(f"test: {sorted(test_scenarios.tolist())}\n")

    # Identify normal windows (no leaks)
    normal_windows = (window_labels == 0).all(dim=1)
    logger.info(f"Normal windows: {normal_windows.sum()}/{len(normal_windows)}")

    train_mask = torch.tensor(np.isin(window_scenarios, train_scenarios))
    val_mask = torch.tensor(np.isin(window_scenarios, val_scenarios))
    test_mask = torch.tensor(np.isin(window_scenarios, test_scenarios))

    if anomaly:
        train_mask = train_mask[:, 0] * normal_windows
        val_mask = val_mask[:, 0] * normal_windows
    else:
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask[:, 0])[0]
    logger.info(f"train_idx: {train_idx}")
    logger.info(f"val_idx: {val_idx}")
    logger.info(f"test_idx: {test_idx}")
    
    return train_idx, val_idx, test_idx


import numpy as np

def check_standardization(data, tol=1e-4):
    """
    Checks if the given dataset is standardized (zero mean, unit variance).
    
    Parameters:
    - data: np.ndarray - The standardized data.
    - tol: float - The tolerance level for checking mean and std deviation.
    
    Returns:
    - bool: True if the data is properly standardized, False otherwise.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    mean_check = np.all(np.abs(mean) < tol)
    
    std_check = np.abs(std - 1) < tol
    zero_std_check = np.abs(std) < tol
    std_check = np.all(zero_std_check | std_check)

    if np.any(zero_std_check):
        print("Zero standard deviation detected!")

    return mean_check and std_check


def normalize_data(
    train_features,
    val_features,
    test_features
):
        # Normalize using training normal data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features.numpy())
    val_features = scaler.transform(val_features.numpy())
    test_features = scaler.transform(test_features.numpy())

    # Convert to tensors
    x_train = torch.tensor(train_features, dtype=torch.float32)
    x_val = torch.tensor(val_features, dtype=torch.float32)
    x_test = torch.tensor(test_features, dtype=torch.float32)
        
    return x_train, x_val, x_test


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


class Fuzzifier:
    def __init__(self, n_clusters, n_features, X_train):
        self.n_clusters = n_clusters
        self.n_features = n_features
        # Initialize cluster centers using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_)

        # Compute standard deviation for each cluster  
        labels = kmeans.labels_
        cluster_std_devs = []
        for i in range(n_clusters):
            cluster_points = X_train[labels == i]  # Select points in cluster i
            std_dev = torch.std(cluster_points, dim=0)  # Compute std deviation per feature
            cluster_std_devs.append(std_dev)
            
        self.std_devs = torch.cat(cluster_std_devs)

    def fuzzify(self, X):        
        u = torch.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            u[:, i] = torch.exp(-torch.sum((X - self.cluster_centers[i])*2, axis=1) / (2 * self.std_devs[i]*2))
        return u / torch.sum(u, axis=1, keepdims=True)
    
    
def fuzzify(graph_data, params, logger):
    n_clusters = params['n_clusters']
    logger.info(f"Fuzzifying with {n_clusters} clusters")
    
    num_nodes = graph_data['window_labels'].shape[1]
    node_features = graph_data['node_features']
    # Concat nodes
    node_features = einops.rearrange(node_features, 'b (f n) -> (b n) f', n=num_nodes)
    n_features = node_features.shape[1]
    
    train_idx = graph_data['train_idx']
    x_train = node_features[train_idx]
    
    fuzzifier = Fuzzifier(n_clusters, n_features, x_train)
    logger.info(f"Node Cluster centers: {fuzzifier.cluster_centers}")
    logger.info(f"Node Std Devs       : {fuzzifier.std_devs}")
    
    node_features = fuzzifier.fuzzify(node_features)
    # Reshape back to [time, features, nodes]
    node_features = einops.rearrange(node_features, '(b n) f -> b (f n)', n=num_nodes)
    
    graph_data['node_features'] = node_features
    
    if "edge_attr" in graph_data:
        num_edges = graph_data['edge_index'].shape[1]
        edge_features = graph_data['edge_features']
        edge_features = einops.rearrange(edge_features, 'b (f n) -> (b n) f', n=num_edges)
        
        n_features = edge_features.shape[1]
        
        x_edge_train = edge_features[train_idx]
        fuzzifier = Fuzzifier(n_clusters, n_features, x_edge_train)
        logger.info(f"Edge Cluster centers: {fuzzifier.cluster_centers}")
        logger.info(f"Edge Std Devs       : {fuzzifier.std_devs}")
        edge_features = fuzzifier.fuzzify(edge_features)
        # Reshape back to [time, features, edges]
        edge_features = einops.rearrange(edge_features, '(b n) f -> b (f n)', n=num_edges)
        
        graph_data['edge_features'] = edge_features
        
    return graph_data

    
@click.command()
@click.option('--parameters', type=str, help="Parameters file")
def main(parameters):
    with open(parameters, "r") as f:
        hyperparams = yaml.safe_load(f)
    window_size = hyperparams['window_size']
    stride = hyperparams['stride']
    subsample = hyperparams['subsample']
    subsize = hyperparams['subsize']
    doublewindow_size = hyperparams.get('doublewindow_size')
    data_path = hyperparams['data_path']
    suffix = hyperparams.get('suffix', '')
    features = hyperparams.get('features')
    fuzzy = hyperparams.get('fuzzy')
    
    prefix = "processed" if fuzzy is None else "fuzzified"
    
    double_string = "" if doublewindow_size is None else f"_2W{doublewindow_size}"
    subsize_string = "" if subsize is None else f"_SUBSIZE{subsize}"
    data_name = f"data/{prefix}_data_W{window_size}{double_string}_S{subsample}_STRIDE{stride}{subsize_string}{suffix}.pt"
    logger = get_logger("preprocess", log_file=f"{data_name}.log")
    
    logger.info(f"Using window size {window_size}, stride {stride}, subsample {subsample} and double window size {doublewindow_size}")
    
    #%% Data Loading and Preprocessing
    logger.info("Loading and preprocessing data...")
    graph_data = torch.load(data_path)
    num_nodes = graph_data['node_features'].shape[1]
    num_edges = graph_data['edge_index'].shape[0]
    
    logger.info("Graph_data:")
    for k, v in graph_data.items():
        logger.info(f"{k}: {v}")
    
    graph_data = make_undirected(graph_data)
    num_nodes = graph_data['node_features'].shape[1]
    num_edges = graph_data['edge_index'].shape[0]
    logger.info("Graph_data after making edges undirected:")
    logger.info(f"Number of nodes: {num_nodes}")
    logger.info(f"Number of edges: {num_edges}")
    logger.info(graph_data)
    
    if subsize is not None:
        logger.info(f"\nSubsampling to {subsize}")
        graph_data['node_features'] = graph_data['node_features'][:subsize]
        if "edge_attr" in graph_data:
            graph_data['edge_attr'] = graph_data['edge_attr'][:subsize]
        graph_data['scenario'] = graph_data['scenario'][:subsize]
        graph_data['y'] = graph_data['y'][:subsize]
        logger.info(f"Subsized node features: {graph_data['node_features']}")
        if "edge_attr" in graph_data:
            logger.info(f"Subsized edge features: {graph_data['edge_attr']}")
        logger.info(f"Subsized scenarios: {graph_data['scenario']}")
        logger.info(f"Subsized labels: {graph_data['y']}")

    graph_data = feature_extraction(graph_data, window_size, stride, subsample, features)
    train_idx, val_idx, test_idx = train_val_test_split(graph_data['window_labels'], graph_data['window_scenarios'], logger, anomaly=False)
    
    graph_data['train_idx'] = train_idx
    graph_data['val_idx'] = val_idx
    graph_data['test_idx'] = test_idx
    node_features = graph_data['node_features']
    
    train_node_features = node_features[train_idx]
    val_node_features = node_features[val_idx]
    test_node_features = node_features[test_idx]
    
    logger.info("Normalizing data...")
    x_train, x_val, x_test = normalize_data(train_node_features, val_node_features, test_node_features)
    node_features[train_idx] = x_train
    node_features[val_idx] = x_val
    node_features[test_idx] = x_test
    
    if "edge_attr" in graph_data:
        logger.info("Normalizing edge features...")
        edge_features = graph_data['edge_features']
        train_edge_features = edge_features[train_idx]
        val_edge_features = edge_features[val_idx]
        test_edge_features = edge_features[test_idx]
        
        x_train, x_val, x_test = normalize_data(train_edge_features, val_edge_features, test_edge_features)
        edge_features[train_idx] = x_train
        edge_features[val_idx] = x_val
        edge_features[test_idx] = x_test
        
        graph_data['edge_features'] = edge_features
    
    if fuzzy:
        graph_data = fuzzify(graph_data, fuzzy, logger)
    
    if doublewindow_size is not None:
        logger.info(f"\nMaking double windows of size {doublewindow_size}")
        graph_data['node_features'] = sliding_window(graph_data['node_features'], doublewindow_size)
        if "edge_features" in graph_data:
            graph_data['edge_features'] = sliding_window(graph_data['edge_features'], doublewindow_size)
        logger.info("New node features:")
        logger.info(graph_data['node_features'])
        if "edge_features" in graph_data:
            logger.info("New edge features:")
            logger.info(graph_data['edge_features'])
    
    logger.info("\nSaving processed data...")
    torch.save(graph_data, data_name)
    logger.info(f"Data saved successfully to {data_name}")
    
    
if __name__ == "__main__":
    main()