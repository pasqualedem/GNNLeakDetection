from datetime import datetime
import os
import csv
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from optuna.importance import get_param_importances
import seaborn as sns
import einops
import optuna
import yaml

from optuna.samplers import GridSampler

from logger import get_logger
from model import LeakDetector

import lovely_tensors as lt

lt.monkey_patch()


BATCH_SIZE = 512
MAX_EPOCHS = 500

STUDY_NAME = "DoubleWindow"
HYPERPARAMS = dict(
    patience=[10],
    sched_patience=[5],
    lr=[0.01, 0.001],
    hidden_dims=[[4, 4, 4]],
    decoder_dims=[[4, 4]],
    use_edges=[True],
    edges_directed=[False],
    # data_path = ["data/processed_data_W12_S5.pt", "data/processed_data_W12_S10.pt", "data/processed_data_W24_S2.pt"]
    data_path=["data/processed_doublewindowed_data_W24W4_S1_STRIDE18.pt"],
)


def get_model(node_in, edge_in, **kwargs):
    return LeakDetector(
        node_in=node_in,
        edge_in=edge_in,
        **kwargs
    )


def direct_edges(graphs):
    num_edges = graphs[0].edge_index.shape[1] // 2

    for i in tqdm(range(len(graphs))):
        graphs[i].edge_index = graphs[i].edge_index[:, :num_edges]
        graphs[i].edge_attr = graphs[i].edge_attr[:num_edges, :]


def get_data(data_path, edges_directed, logger):
    processed_data = torch.load(data_path, weights_only=True)
    node_features = processed_data["node_features"]
    edge_features = processed_data["edge_features"]
    window_labels = processed_data["window_labels"]
    window_scenarios = processed_data["window_scenarios"]
    edge_index = processed_data["edge_index"]
    num_nodes = window_labels.shape[1]
    num_edges = edge_index.shape[1]

    scenarios = window_scenarios.unique()
    leak_scenarios = window_scenarios[window_labels.any(dim=1)].unique()
    non_leak_scenarios = set(scenarios.tolist()) - set(leak_scenarios.tolist())

    VAL_SCENARIOS, TEST_SCENARIOS = 100, 100

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
    train_edge_features = edge_scaler.fit_transform(edge_features[train_idx].numpy())
    val_edge_features = edge_scaler.transform(edge_features[val_idx].numpy())
    test_edge_features = edge_scaler.transform(edge_features[test_idx].numpy())

    # Convert to tensors
    x_train = torch.tensor(train_features, dtype=torch.float32)
    x_val = torch.tensor(val_features, dtype=torch.float32)
    x_test = torch.tensor(test_features, dtype=torch.float32)

    x_edge_train = torch.tensor(train_edge_features, dtype=torch.float32)
    x_edge_val = torch.tensor(val_edge_features, dtype=torch.float32)
    x_edge_test = torch.tensor(test_edge_features, dtype=torch.float32)

    # Reshape back to [time, nodes, features]
    x_train = einops.rearrange(x_train, "w (f n) -> w n f", n=num_nodes)
    x_val = einops.rearrange(x_val, "w (f n) -> w n f", n=num_nodes)
    x_test = einops.rearrange(x_test, "w (f n) -> w n f", n=num_nodes)

    # Reshape back to [time, edges, features]
    x_edge_train = einops.rearrange(x_edge_train, "w (f e) -> w e f", e=num_edges)
    x_edge_val = einops.rearrange(x_edge_val, "w (f e) -> w e f", e=num_edges)
    x_edge_test = einops.rearrange(x_edge_test, "w (f e) -> w e f", e=num_edges)

    logger.info(f"x_train: {x_train}")
    logger.info(f"x_val: {x_val}")
    logger.info(f"x_test: {x_test}")
    logger.info("----")
    logger.info(f"x_edge_train: {x_edge_train}")
    logger.info(f"x_edge_val: {x_edge_val}")
    logger.info(f"x_edge_test: {x_edge_test}")

    # Create graph data
    train_data = [
        Data(
            x=x_train[i],  # Training uses only normal data,
            edge_attr=x_edge_train[i],
            edge_index=edge_index,
            y=x_train[i],
        )
        for i in tqdm(range(len(x_train)))
    ]
    logger.info("Train data created")
    val_data = [
        Data(
            x=x_val[i],
            edge_attr=x_edge_val[i],
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


def train(
    trial_dir, model, train_loader, val_batch, optimizer, scheduler, patience, logger
):
    cur_patience = 0
    best_val_loss = float("inf")
    logger.info("Training started")
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0

        # Batch training
        for data in train_loader:
            optimizer.zero_grad()

            x_recon = model(data)
            loss = torch.mean(torch.abs(data.y - x_recon))  # MAE loss

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(data)

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_recon = model(val_batch)
            val_loss = torch.mean(torch.abs(val_batch.y - val_recon))

        scheduler.step(val_loss)

        best = ""
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cur_patience = 0
            best = f"New best {best_val_loss:.4f}"
            torch.save(model.state_dict(), f"{trial_dir}/best_model.pt")
        else:
            cur_patience += 1
            best = f"Not best, patience: {cur_patience}"
            if cur_patience >= patience:
                logger.info("Early stopping")
                break

        logger.info(
            f"Epoch {epoch:04d}: Train Loss {avg_train_loss:.4f}, Val Loss {val_loss:.4f}. {best}"
        )


def test(trial_dir, model, x_test_data, val_batch, test_labels, num_nodes, logger):
    test_batch = next(
        iter(DataLoader(x_test_data, batch_size=len(x_test_data), shuffle=False))
    )
    # Load best model
    model.load_state_dict(torch.load(f"{trial_dir}/best_model.pt", weights_only=True))

    # %% Anomaly Detection
    model.eval()
    with torch.no_grad():
        # Get reconstruction for test data
        test_recon = model(test_batch)

        # Calculate node-level errors (MAE)
        node_errors = torch.abs(
            test_batch.x - test_recon
        )  # [num_test_windows, num_nodes]

        # Get max error per window
        window_scores = node_errors.max(dim=1).values.numpy()

    # Calculate threshold using validation normal data (80th percentile)
    with torch.no_grad():
        val_recon = model(val_batch)
        val_errors = torch.abs(val_batch.y - val_recon).max(dim=1).values.numpy()
    threshold = np.quantile(val_errors, 0.90)  # Lowered to 80th percentile

    # Detect anomalies in test set
    anomaly_pred = window_scores > threshold

    # %% Evaluation
    y_true_graph = test_labels.any(axis=1).numpy().astype(int)
    y_pred = anomaly_pred.astype(int)

    # If a node leaks -> the graph leaks
    y_pred_graph = einops.reduce(y_pred, "(b n) -> b", n=num_nodes, reduction="max")
    window_scores_graph = einops.reduce(
        window_scores, "(b n) -> b", n=num_nodes, reduction="max"
    )
    anomaly_pred_graph = einops.reduce(
        anomaly_pred, "(b n) -> b", n=num_nodes, reduction="max"
    )

    y_pred_random = torch.randint(0, 2, size=(len(y_true_graph),)).numpy().astype(int)

    logger.info("\n=== Detection Performance ===")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_graph, y_pred_graph, average="binary"
    )
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall:    {recall:.3f}")
    logger.info(f"F1-score:  {f1:.3f}")

    # Calculate precision, recall, and FPR for random predictions
    precision_random, recall_random, _ = precision_recall_curve(
        y_true_graph, y_pred_random
    )
    fpr_random, tpr_random, _ = roc_curve(y_true_graph, y_pred_random)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true_graph, window_scores_graph)
    logger.info(f"ROC-AUC:   {roc_auc:.3f}")

    cm = confusion_matrix(y_true_graph, y_pred_graph)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{trial_dir}/confusion_matrix.png")
    plt.show()

    # %% Additional Visualizations
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true_graph, window_scores_graph)
    plt.figure(figsize=(12, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot(fpr_random, tpr_random, label="Random Guess ROC", linestyle="--")
    plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{trial_dir}/roc_curve.png")
    plt.show()

    # 2. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true_graph, window_scores_graph
    )
    plt.plot(recall_curve, precision_curve, label="Precision-Recall Curve")
    plt.plot(
        recall_random,
        precision_random,
        label="Random Guess Precision-Recall",
        linestyle="--",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{trial_dir}/precision_recall_curve.png")
    plt.show()

    # 3. Distribution of Anomaly Scores
    plt.figure(figsize=(12, 5))
    plt.hist(
        window_scores_graph[y_true_graph == 0],
        bins=50,
        alpha=0.5,
        label="Normal Windows",
    )
    plt.hist(
        window_scores_graph[y_true_graph == 1],
        bins=50,
        alpha=0.5,
        label="Anomalous Windows",
    )
    plt.axvline(threshold, c="r", linestyle="--", label="Threshold")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Anomaly Scores")
    plt.legend()
    plt.savefig(f"{trial_dir}/anomaly_score_distribution.png")
    plt.show()

    print("\nTest complete!")
    return f1


def train_and_test(trial, i, study_dir):
    params = {k: trial.suggest_categorical(k, HYPERPARAMS[k]) for k in HYPERPARAMS}

    # Create unique dir for each trial
    trial_dir = os.path.join(study_dir, f"trial_{i}")
    os.makedirs(trial_dir, exist_ok=True)
    print(f"Trial directory created: {trial_dir}")

    logger = get_logger(f"trial_{i}", os.path.join(trial_dir, "train.log"))
    for k, v in params.items():
        logger.info(f"{k}: {v}")

    # Save hyperparameters to a YAML file
    hyperparams_path = os.path.join(trial_dir, "hyperparams.yaml")
    with open(hyperparams_path, "w") as yaml_file:
        yaml.dump(params, yaml_file, default_flow_style=False)

    logger.info(f"Hyperparameters saved to: {hyperparams_path}")

    data_path = params["data_path"]
    edges_directed = params["edges_directed"]

    train_data, val_data, (test_data, test_labels), (num_nodes, num_edges) = get_data(
        data_path, edges_directed, logger
    )

    # model params
    hid_dim = params.get("hid_dim")
    num_layers = params.get("num_layers")
    hidden_dims = params.get("hidden_dims")
    decoder_dims = params.get("decoder_dims")
    lstm_layers = params.get("lstm_layers")
    window_size = params.get("window_size")

    lr = params["lr"]
    sched_patience = params["sched_patience"]
    use_edges = params["use_edges"]

    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1]

    # %% Model Initialization
    model = get_model(
        node_in=node_dim,
        edge_in=edge_dim if use_edges else None,
        hid_dim=hid_dim,
        num_layers=num_layers,
        hidden_dims=hidden_dims,
        decoder_dims=decoder_dims,
        lstm_layers=lstm_layers,
        window_size=window_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=sched_patience
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_batch = next(
        iter(DataLoader(val_data, batch_size=len(val_data), shuffle=False))
    )

    patience = params["patience"]

    train(
        trial_dir,
        model,
        train_loader,
        val_batch,
        optimizer,
        scheduler,
        patience,
        logger,
    )
    score = test(trial_dir, model, test_data, val_batch, test_labels, num_nodes, logger)

    return score


def main():
    os.makedirs("results", exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique directory for this study
    study_dir = os.path.join("results", f"study_{current_time}_{STUDY_NAME}")
    os.makedirs(study_dir, exist_ok=True)

    print(f"Study directory created: {study_dir}")
    # You can save study-related information in this directory if needed

    # Save hyperparameters to a YAML file
    hyperparams_path = os.path.join(study_dir, "hyperparams.yaml")
    with open(hyperparams_path, "w") as yaml_file:
        yaml.dump(HYPERPARAMS, yaml_file, default_flow_style=False)

    print(f"Hyperparameters saved to: {hyperparams_path}")
    study_logger = get_logger("study", os.path.join(study_dir, "study.log"))

    study = optuna.create_study(direction="maximize", sampler=GridSampler(HYPERPARAMS))
    scores_df = pd.DataFrame(
        columns=["Trial"] + [key for key in HYPERPARAMS.keys()] + ["Score"]
    )

    MAX_TRIALS = 100

    for i in range(MAX_TRIALS):
        try:
            trial = study.ask()
            study_logger.info(f"Trial {i}")
            score = train_and_test(trial, i, study_dir)

            # Save to CSV
            trial_data = {key: value for key, value in trial.params.items()}
            trial_data.update({"Trial": i, "Score": score})
            scores_df = pd.concat(
                [scores_df, pd.DataFrame([trial_data])], ignore_index=True
            )
            scores_df.to_csv(os.path.join(study_dir, "scores.csv"), index=False)

            study.tell(trial, score)
            study_logger.info(f"Score for trial {i}: {score}")

        except RuntimeError as e:
            study_logger.warning(e)
            break

    best_trial = study.best_trial
    study_logger.info(f"Best trial: {best_trial.params}")
    study_logger.info(f"Best trial score: {best_trial.value}")
    importances = {k: v.item() for k, v in get_param_importances(study).items()}
    study_logger.info(f"Importances: {importances}")
    with open(os.path.join(study_dir, "importances.yaml"), "w") as f:
        yaml.dump(importances, f)


if __name__ == "__main__":
    main()
