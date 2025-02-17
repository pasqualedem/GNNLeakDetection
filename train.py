from datetime import datetime
import os
import click
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from optuna.importance import get_param_importances
import optuna
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
import yaml

from optuna.samplers import GridSampler

from data import get_data
from grid import delinearize_from_string, linearize, linearized_to_string
from logger import get_logger
from loss import get_loss
from model import get_model
from tracker import wandb_experiment

import lovely_tensors as lt

from test import test, test_anomaly

lt.monkey_patch()

class EarlyStopping:
    def __init__(self, patience):
        self.cur_patience = 0
        self.patience = patience
        self.best_val_loss = float("inf")
        
    def __call__(self, val_loss, model, trial_dir, logger):
        # Early stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.cur_patience = 0
            best = f"New best {self.best_val_loss:.4f}"
            torch.save(model.state_dict(), f"{trial_dir}/best_model.pt")
        else:
            self.cur_patience += 1
            best = f"Not best, patience: {self.cur_patience}"
            if self.cur_patience >= self.patience:
                logger.info("Early stopping")
                return True, ""
        return False, best

def train_anomaly(
    trial_dir, model, loss_fn, train_loader, val_batch, max_epochs, optimizer, scheduler, patience, logger
):
    early_stop = EarlyStopping(patience)
    logger.info("Training started")
    for epoch in range(max_epochs):
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
            val_loss = loss_fn(val_batch.y, val_recon)

        scheduler.step(val_loss)

        # Early stopping
        stop, best = early_stop(val_loss, model, trial_dir, logger)
        if stop:
            break
        logger.info(
            f"Epoch {epoch:04d}: Train Loss {avg_train_loss:.4f}, Val Loss {val_loss:.4f}. {best}"
        )


def train(trial_dir, model, loss_fn, train_loader, val_batch, max_epochs, optimizer, scheduler, patience, tracker, graph_classification):
    ''' 
    Train model 
    '''
    early_stop = EarlyStopping(patience)
    bar_update_interval = 10
    accuracy, f1score = Accuracy(task="binary"), F1Score(task="binary")
    for epoch in range(max_epochs):
        tracker.log_metrics({"epoch": epoch})
        print(f"Epoch {epoch}")
        epoch_loss = 0
        model.train()
        bar = tqdm(train_loader)
        
        # get lr
        lr = optimizer.param_groups[0]["lr"]
        tracker.log_metrics({"lr": lr})
        
        for idx, data in enumerate(bar):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.float())
            loss.backward()
                    
            optimizer.step()
            epoch_loss += loss.item()
            
            accuracy.update(out, data.y)
            f1score.update(out, data.y)
            
            if idx % bar_update_interval == 0:
                metrics = {
                    "loss": loss.item(),
                    "accuracy": accuracy.compute().item(),
                    "f1score": f1score.compute().item(),
                    }             
                bar.set_postfix(metrics)
                tracker.log_metrics(metrics)
                
        with tracker.validate():
            val_loss = test(model, loss_fn, val_batch, tracker, graph_classification=graph_classification)
        scheduler.step(val_loss)
        # Early stopping
        stop, best = early_stop(val_loss, model, trial_dir, tracker)
        if stop:
            break
        epoch_loss /= len(train_loader)
        tracker.log_metrics({"epoch_loss": epoch_loss})
        tracker.info(
            f"Epoch {epoch:04d}: Train Loss {epoch_loss:.4f}, Val Loss {val_loss:.4f}. {best}"
        )


def train_and_test(trial, i, study_dir, hyperparams):
    params = {k: trial.suggest_categorical(k, hyperparams[k]) for k in hyperparams}
    params = delinearize_from_string(params)
    
    anomaly = params["anomaly"]
    graph_classification = params["graph_classification"]

    # Create unique dir for each trial
    trial_dir = os.path.join(study_dir, f"trial_{i}")
    os.makedirs(trial_dir, exist_ok=True)
    print(f"Trial directory created: {trial_dir}")

    logger = get_logger(f"trial_{i}", os.path.join(trial_dir, "train.log"))
    tracker = wandb_experiment(params, logger)
    for k, v in params.items():
        tracker.info(f"{k}: {v}")

    # Save hyperparameters to a YAML file
    hyperparams_path = os.path.join(trial_dir, "hyperparams.yaml")
    with open(hyperparams_path, "w") as yaml_file:
        yaml.dump(params, yaml_file, default_flow_style=False)

    tracker.info(f"Hyperparameters saved to: {hyperparams_path}")

    data_path = params["data_path"]
    edges_directed = params["edges_directed"]

    train_data, val_data, (test_data, test_labels), (num_nodes, num_edges) = get_data(
        data_path, edges_directed, tracker, anomaly=anomaly, graph_classification=graph_classification
    )

    # model params
    model_params = params["model"]
    use_edges = model_params["use_edges"]
    
    decay = params.get("weight_decay", 1e-4)
    batch_size = params.get("batch_size")
    max_epochs = params.get("max_epochs")

    loss_fn = get_loss(params["loss"])
    lr = params["lr"]
    sched_patience = params["sched_patience"]

    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if use_edges else None

    # %% Model Initialization
    model = get_model(
        **model_params,
        node_in=node_dim,
        edge_in=edge_dim if use_edges else None,
        graph_classification=graph_classification
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=sched_patience
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_batch = next(
        iter(DataLoader(val_data, batch_size=len(val_data), shuffle=False))
    )
    test_batch = next(
        iter(DataLoader(test_data, batch_size=len(test_data), shuffle=False))
    )

    patience = params["patience"]

    train_fn = train_anomaly if anomaly else train
    with tracker.train():
        train_fn(
            trial_dir,
            model,
            loss_fn,
            train_loader,
            val_batch,
            max_epochs,
            optimizer,
            scheduler,
            patience,
            tracker,
            graph_classification=graph_classification
        )
    
    with tracker.test():
        tracker.info("\n---------------TESTING---------------")
        if anomaly:
            score = test_anomaly(trial_dir, model, loss_fn, test_batch, val_batch, test_labels, num_nodes, tracker)
        else:
            score = test(model, loss_fn, test_batch, tracker, metric="score", graph_classification=graph_classification)
            
    tracker.end()

    return score

@click.command()
@click.option("--parameters", type=str, default=None)
def cli(parameters):
    
    with open(parameters, "r") as f:
        hyperparams = yaml.safe_load(f)
    study_name = hyperparams.pop("study_name")
    
    os.makedirs("results", exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique directory for this study
    study_dir = os.path.join("results", f"study_{current_time}_{study_name}")
    os.makedirs(study_dir, exist_ok=True)

    print(f"Study directory created: {study_dir}")
    # You can save study-related information in this directory if needed

    # Save hyperparameters to a YAML file
    hyperparams_path = os.path.join(study_dir, "hyperparams.yaml")
    with open(hyperparams_path, "w") as yaml_file:
        yaml.dump(hyperparams, yaml_file, default_flow_style=False)

    print(f"Hyperparameters saved to: {hyperparams_path}")
    study_logger = get_logger("study", os.path.join(study_dir, "study.log"))

    linearized_hyperparams = dict(linearized_to_string(linearize(hyperparams)))
    study = optuna.create_study(direction="maximize", sampler=GridSampler(linearized_hyperparams))
    scores_df = pd.DataFrame(
        columns=["Trial"] + [key for key in hyperparams.keys()] + ["Score"]
    )

    MAX_TRIALS = 100

    for i in range(MAX_TRIALS):
        try:
            trial = study.ask()
            study_logger.info(f"Trial {i}")
            score = train_and_test(trial, i, study_dir, linearized_hyperparams) 

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
    cli()
