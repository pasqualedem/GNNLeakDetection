import einops
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader
import yaml

from data import get_data
from logger import get_logger

import lovely_tensors as lt

from model import get_model
lt.monkey_patch()

TRIAL_DIR = "results/study_2025-01-30_09-57-24_DoubleWindow/trial_1"


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


if __name__ == "__main__":    
    # Load hyperparameters yaml
    with open(f"{TRIAL_DIR}/hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        
    data_path = hyperparams["data_path"]
    edges_directed = hyperparams["edges_directed"]
    
    logger = get_logger("test", f"{TRIAL_DIR}/reprise.log")
    
    train_data, val_data, (x_test_data, test_labels), (num_nodes, num_edges) = get_data(
        data_path, edges_directed=edges_directed, logger=logger
    )
    val_batch = next(
        iter(DataLoader(val_data, batch_size=len(val_data), shuffle=False))
    )
    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1]
    model = get_model(
        node_in=node_dim, edge_in=edge_dim,
        **hyperparams
    )

    logger.info("\n=== Test Performance ===")
    test(TRIAL_DIR, model, x_test_data, val_batch, test_labels, num_nodes, logger)