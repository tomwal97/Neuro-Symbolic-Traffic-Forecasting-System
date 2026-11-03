# File: train.py
# Description: Training loop script for the Spatio-Temporal Baseline traffic forecasting model.
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from dataset import PeMSDataset
from models import PureLSTMModel, PureGCNModel, LinearBaseline, SpatioTemporalBaseline
import time
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    # Optional: ensure deterministic behavior in CuDNN (might slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_losses(train_losses: list, val_losses: list, save_path: str = "./checkpoints/loss_curve.png"):
    """Generates and saves a clean loss convergence plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o", color="royalblue")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s", color="darkorange")

    plt.title("Spatio-Temporal GCN-LSTM Training Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    logger.info(f"Loss convergence plot saved successfully to {save_path}")
    plt.close()


def evaluate_metrics(model, data_loader, adj_matrix, device):
    """Computes comprehensive regression evaluation metrics (RMSE, MAE, MSE, R^2) over a dataloader."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x, adj_matrix)
            all_preds.append(preds.detach().cpu())
            all_targets.append(batch_y.detach().cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = F.mse_loss(preds_tensor, targets_tensor).item()
    rmse = torch.sqrt(F.mse_loss(preds_tensor, targets_tensor)).item()
    mae = F.l1_loss(preds_tensor, targets_tensor).item()

    # Flatten across nodes and time horizons for global R^2 score
    r2 = r2_score(targets_tensor.numpy().flatten(), preds_tensor.numpy().flatten())

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def train_model(
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        hidden_dim: int = 128,
        save_dir: str = "./checkpoints"
):
    """Executes the training and validation loop for the traffic forecasting model."""
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    logger.info("Initializing datasets...")
    train_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="train")
    val_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adj_matrix = train_dataset.get_adjacency_matrix().to(device)
    num_nodes = adj_matrix.shape[0]

    model = SpatioTemporalBaseline(
        num_nodes=num_nodes,
        in_features=1,
        hidden_dim=hidden_dim,
        horizon=12
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    history_train_losses = []
    history_val_losses = []

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x, adj_matrix)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                val_preds = model(val_x, adj_matrix)
                v_loss = criterion(val_preds, val_y)
                val_loss += v_loss.item() * val_x.size(0)

        epoch_val_loss = val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)

        history_train_losses.append(epoch_train_loss)
        history_val_losses.append(epoch_val_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(save_dir, "best_baseline_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"--> Saved new best model checkpoint to {ckpt_path}")

    logger.info("Training pipeline completed successfully!")
    plot_path = os.path.join(save_dir, "training_loss_curve.png")
    plot_losses(history_train_losses, history_val_losses, save_path=plot_path)


def run_model_comparison(epochs: int = 10, batch_size: int = 64, lr: float = 0.001, hidden_dim: int = 64):
    """Trains and evaluates multiple models, computing full regression metrics and printing a summary table."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"[*] Starting Multi-Model Comparison on device: {device}")

    train_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="train")
    val_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adj_matrix = train_dataset.get_adjacency_matrix().to(device)
    num_nodes = adj_matrix.shape[0]

    models_to_test = {
        "Spatio-Temporal GCN-LSTM": SpatioTemporalBaseline(num_nodes=num_nodes, in_features=1, hidden_dim=hidden_dim,horizon=12),
        "Pure LSTM (No Spatial)": PureLSTMModel(num_nodes=num_nodes, in_features=1, hidden_dim=hidden_dim, horizon=12),
        "Pure GCN (No Recurrence)": PureGCNModel(num_nodes=num_nodes, in_features=1, hidden_dim=hidden_dim, horizon=12),
        "Linear Regressor Baseline": LinearBaseline(num_nodes=num_nodes, in_features=1, history_window=12, horizon=12)
    }

    results_summary = []
    criterion = nn.L1Loss()

    for name, model in models_to_test.items():
        logger.info(f"\n================ Training Model: {name} ================")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x, adj_matrix)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = train_loss / len(train_dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    val_preds = model(val_x, adj_matrix)
                    v_loss = criterion(val_preds, val_y)
                    val_loss += v_loss.item() * val_x.size(0)

            epoch_val_loss = val_loss / len(val_dataset)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            logger.info(
                f"[{name}] Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        elapsed_time = time.time() - start_time

        # Compute comprehensive evaluation metrics on validation set
        metrics = evaluate_metrics(model, val_loader, adj_matrix, device)

        # Save individual loss curve plot
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_losses(train_losses, val_losses, save_path=f"./checkpoints/loss_curve_{safe_name}.png")

        results_summary.append({
            "Model": name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "R2": metrics["R2"],
            "Training Time (s)": elapsed_time
        })

    # Print Expanded Model Comparison Summary Table
    print("=" * 88)
    print("                 MODEL COMPARISON EVALUATION SUMMARY                  ")
    print("=" * 88)
    print(f"{'Model Architecture':<28} | {'RMSE':<8} | {'MAE':<8} | {'MSE':<8} | {'R^2':<6} | {'Time (s)':<8}")
    print("-" * 88)

    for res in results_summary:
        train_time = res.get('Training Time (s)', 0.0)
        print(
            f"{res['Model']:<28} | {res['RMSE']:<8.4f} | {res['MAE']:<8.4f} | {res['MSE']:<8.4f} | {res['R2']:<6.4f} | {train_time:<8.1f}")
    print("=" * 88)


if __name__ == "__main__":
    set_seed(42)
    #run_model_comparison(epochs=50, batch_size=32, lr=0.001, hidden_dim=64)
    train_model(epochs=50, batch_size=32, lr=0.001, hidden_dim=64)