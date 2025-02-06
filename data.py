

import os
import einops
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm


def direct_edges(graphs):
    num_edges = graphs[0].edge_index.shape[1] // 2

    for i in tqdm(range(len(graphs))):
        graphs[i].edge_index = graphs[i].edge_index[:, :num_edges]
        graphs[i].edge_attr = graphs[i].edge_attr[:num_edges, :]


def get_data(data_path, edges_directed, logger):
    processed_data = torch.load(data_path, weights_only=True)
    node_features = processed_data["node_features"]
    edge_features = processed_data.get("edge_features", None)
    window_labels = processed_data["window_labels"]
    window_scenarios = processed_data["window_scenarios"]
    edge_index = processed_data["edge_index"]
    num_nodes = window_labels.shape[1]
    num_edges = edge_index.shape[1]

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

    train_idx = torch.where(train_mask[:, 0] * normal_windows)[0]
    val_idx = torch.where(val_mask[:, 0] * normal_windows)[0]
    test_idx = torch.where(test_mask[:, 0])[0]
    logger.info(f"train_idx: {train_idx}")
    logger.info(f"val_idx: {val_idx}")
    logger.info(f"test_idx: {test_idx}")

    # Normalize using training normal data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(node_features[train_idx].numpy())
    val_features = scaler.transform(node_features[val_idx].numpy())
    test_features = scaler.transform(node_features[test_idx].numpy())

    edge_scaler = StandardScaler()
    train_edge_features = edge_scaler.fit_transform(edge_features[train_idx].numpy()) if edge_features is not None else None
    val_edge_features = edge_scaler.transform(edge_features[val_idx].numpy()) if edge_features is not None else None
    test_edge_features = edge_scaler.transform(edge_features[test_idx].numpy()) if edge_features is not None else None

    # Convert to tensors
    x_train = torch.tensor(train_features, dtype=torch.float32)
    x_val = torch.tensor(val_features, dtype=torch.float32)
    x_test = torch.tensor(test_features, dtype=torch.float32)

    if edge_features is not None:
        x_edge_train = torch.tensor(train_edge_features, dtype=torch.float32)
        x_edge_val = torch.tensor(val_edge_features, dtype=torch.float32)
        x_edge_test = torch.tensor(test_edge_features, dtype=torch.float32)
    else:
        x_edge_train = None
        x_edge_val = None
        x_edge_test = None

    # Reshape back to [time, nodes, features]
    x_train = einops.rearrange(x_train, "w (f n) -> w n f", n=num_nodes)
    x_val = einops.rearrange(x_val, "w (f n) -> w n f", n=num_nodes)
    x_test = einops.rearrange(x_test, "w (f n) -> w n f", n=num_nodes)

    if edge_features is not None:
        # Reshape back to [time, edges, features]
        x_edge_train = einops.rearrange(x_edge_train, "w (f e) -> w e f", e=num_edges)
        x_edge_val = einops.rearrange(x_edge_val, "w (f e) -> w e f", e=num_edges)
        x_edge_test = einops.rearrange(x_edge_test, "w (f e) -> w e f", e=num_edges)

    logger.info(f"x_train: {x_train}")
    logger.info(f"x_val: {x_val}")
    logger.info(f"x_test: {x_test}")
    logger.info("----")
    if edge_features is not None:
        logger.info(f"x_edge_train: {x_edge_train}")
        logger.info(f"x_edge_val: {x_edge_val}")
        logger.info(f"x_edge_test: {x_edge_test}")
    else:
        logger.info("No edge features")

    # Create graph data
    train_data = [
        Data(
            x=x_train[i],  # Training uses only normal data,
            edge_attr=x_edge_train[i] if edge_features is not None else None,
            edge_index=edge_index,
            y=x_train[i],
        )
        for i in tqdm(range(len(x_train)))
    ]
    logger.info("Train data created")
    val_data = [
        Data(
            x=x_val[i],
            edge_attr=x_edge_val[i] if edge_features is not None else None,
            edge_index=edge_index,
            y=x_val[i],
        )
        for i in tqdm(range(len(x_val)))
    ]
    logger.info("Val data created")
    x_test_data = [
        Data(
            x=x_test[i],
            edge_index=edge_index,
            edge_attr=x_edge_test[i] if edge_features is not None else None,
            y=x_test[i],
        )
        for i in tqdm(range(len(x_test)))
    ]
    logger.info("Test data created")

    if edges_directed:
        logger.info("Directing edges")
        direct_edges(train_data)
        direct_edges(val_data)
        direct_edges(x_test_data)

    test_labels = window_labels[test_idx]

    return train_data, val_data, (x_test_data, test_labels), (num_nodes, num_edges)