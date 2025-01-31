from datetime import datetime
import os
import csv
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from optuna.importance import get_param_importances
import optuna
import yaml

from optuna.samplers import GridSampler

from data import get_data
from logger import get_logger
from model import get_model

import lovely_tensors as lt

from test import test

lt.monkey_patch()


BATCH_SIZE = 512
MAX_EPOCHS = 500

STUDY_NAME = "GATv2ConvSearch"
HYPERPARAMS = dict(
    patience=[10],
    sched_patience=[5],
    lr=[0.1, 0.01, 0.001, 0.005, 0.0001],
    hidden_dims=[[8, 8, 4], [8, 8, 4, 4], [8, 8, 4, 2], [8, 4, 2, 1], [8, 16, 16, 8], [32, 32]],
    # decoder_dims=[[4, 8]],
    use_edges=[True],
    edges_directed=[False],
    # data_path = ["data/processed_data_W12_S5.pt", "data/processed_data_W12_S10.pt", "data/processed_data_W24_S2.pt"]
    data_path=["data/processed_doublewindowed_data_W24W4_S1_STRIDE18.pt"],
)


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
    decay = params.get("weight_decay", 1e-4)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
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
