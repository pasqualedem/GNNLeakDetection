

import os
import einops
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

from preprocess import check_standardization, normalize_data, train_val_test_split


def direct_edges(graphs):
    num_edges = graphs[0].edge_index.shape[1] // 2

    for i in tqdm(range(len(graphs))):
        graphs[i].edge_index = graphs[i].edge_index[:, :num_edges]
        graphs[i].edge_attr = graphs[i].edge_attr[:num_edges, :]


def get_data(data_path, edges_directed, logger, anomaly=False, graph_classification=False):
    processed_data = torch.load(data_path, weights_only=True)
    node_features = processed_data["node_features"]
    edge_features = processed_data.get("edge_features", None)
    window_labels = processed_data["window_labels"]
    window_scenarios = processed_data["window_scenarios"]
    edge_index = processed_data["edge_index"]
    num_nodes = window_labels.shape[1]
    num_edges = edge_index.shape[1]

    if "train_idx" in processed_data and "val_idx" in processed_data and "test_idx" in processed_data:
        train_idx = processed_data["train_idx"]
        val_idx = processed_data["val_idx"]
        test_idx = processed_data["test_idx"]
    else:
        train_idx, val_idx, test_idx = train_val_test_split(window_labels, window_scenarios, anomaly=anomaly)

    x_train = node_features[train_idx]
    x_val = node_features[val_idx]
    x_test = node_features[test_idx]
        
    logger.info(f"x_train: {x_train}")
    logger.info(f"x_val: {x_val}")
    logger.info(f"x_test: {x_test}")
    logger.info("----")

    # Reshape back to [time, nodes, features]
    x_train = einops.rearrange(x_train, "w (f n) -> w n f", n=num_nodes)
    x_val = einops.rearrange(x_val, "w (f n) -> w n f", n=num_nodes)
    x_test = einops.rearrange(x_test, "w (f n) -> w n f", n=num_nodes)
        
    if edge_features is not None:
        x_edge_train = edge_features[train_idx]
        x_edge_val = edge_features[val_idx]
        x_edge_test = edge_features[test_idx]

        # Reshape back to [time, edges, features]
        x_edge_train = einops.rearrange(x_edge_train, "w (f e) -> w e f", e=num_edges)
        x_edge_val = einops.rearrange(x_edge_val, "w (f e) -> w e f", e=num_edges)
        x_edge_test = einops.rearrange(x_edge_test, "w (f e) -> w e f", e=num_edges)

        logger.info(f"x_edge_train: {x_edge_train}")
        logger.info(f"x_edge_val: {x_edge_val}")
        logger.info(f"x_edge_test: {x_edge_test}")
    else:
        logger.info("No edge features")
        
    if graph_classification:
        window_labels = window_labels.any(dim=1).long()

    train_labels = window_labels[train_idx] if not anomaly else x_train
    val_labels = window_labels[val_idx] if not anomaly else x_val
    test_labels = window_labels[test_idx]
    inc_test_labels = test_labels if not anomaly else x_test

    # Create graph data
    train_data = [
        Data(
            x=x_train[i],  # Training uses only normal data,
            edge_attr=x_edge_train[i] if edge_features is not None else None,
            edge_index=edge_index,
            y=train_labels[i],
        )
        for i in tqdm(range(len(x_train)))
    ]
    logger.info("Train data created")
    val_data = [
        Data(
            x=x_val[i],
            edge_attr=x_edge_val[i] if edge_features is not None else None,
            edge_index=edge_index,
            y=val_labels[i],
        )
        for i in tqdm(range(len(x_val)))
    ]
    logger.info("Val data created")
    test_data = [
        Data(
            x=x_test[i],
            edge_index=edge_index,
            edge_attr=x_edge_test[i] if edge_features is not None else None,
            y=inc_test_labels[i],
        )
        for i in tqdm(range(len(x_test)))
    ]
    logger.info("Test data created")

    if edges_directed:
        logger.info("Directing edges")
        direct_edges(train_data)
        direct_edges(val_data)
        direct_edges(test_data)

    return train_data, val_data, (test_data, test_labels), (num_nodes, num_edges)