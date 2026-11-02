# File: train.py
# Description: Training loop script for the Spatio-Temporal Baseline traffic forecasting model.
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import dropout
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt

from dataset import PeMSDataset
from models import PureLSTMModel, PureGCNModel, LinearBaseline, SpatioTemporalBaseline
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_losses(train_losses: list, val_losses: list, save_path: str = "./checkpoints/loss_curve.png"):
    """Generates and saves a clean loss convergence plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o", color="royalblue")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s", color="darkorange")

    plt.title("Spatio-Temporal GCN-LSTM Training Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    logger.info(f"Loss convergence plot saved successfully to {save_path}")
    plt.close()


def train_model(
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        hidden_dim: int = 32,
        save_dir: str = "./checkpoints"
):
    """Executes the training and validation loop for the traffic forecasting model."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Device Selection (Handles Mac MPS, CUDA, or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 2. Load Datasets and DataLoaders
    logger.info("Initializing datasets...")
    train_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="train")
    val_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Fetch Adjacency Matrix and move to device
    adj_matrix = train_dataset.get_adjacency_matrix().to(device)
    num_nodes = adj_matrix.shape[0]

    # 3. Initialize Model, Loss, and Optimizer
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

    # Trackers for plotting
    history_train_losses = []
    history_val_losses = []

    # 4. Training Loop
    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_x, adj_matrix)

            # Compute loss
            loss = criterion(predictions, batch_y)

            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        # 5. Validation Loop
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

        # Inside your training epoch loop (after computing epoch_val_loss):
        scheduler.step(epoch_val_loss)

        # Record history
        history_train_losses.append(epoch_train_loss)
        history_val_losses.append(epoch_val_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(save_dir, "best_baseline_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"--> Saved new best model checkpoint to {ckpt_path}")

    logger.info("Training pipeline completed successfully!")

    # 6. Generate and save loss curve plot
    plot_path = os.path.join(save_dir, "training_loss_curve.png")
    plot_losses(history_train_losses, history_val_losses, save_path=plot_path)

def run_model_comparison(epochs: int = 10, batch_size: int = 64, lr: float = 0.001, hidden_dim: int = 64):
    """Trains and evaluates multiple models, saving individual plots and printing a summary table."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"[*] Starting Multi-Model Comparison on device: {device}")

    # Load Data
    train_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="train")
    val_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adj_matrix = train_dataset.get_adjacency_matrix().to(device)
    num_nodes = adj_matrix.shape[0]

    # Define model registry to compare
    models_to_test = {
        #"Spatio-Temporal GCN-LSTM (Chosen Model)": SpatioTemporalBaseline(num_nodes=num_nodes, in_features=1, hidden_dim=hidden_dim, horizon=12),
        "Pure LSTM (No Spatial Graph)": PureLSTMModel(num_nodes=num_nodes, in_features=1, hidden_dim=hidden_dim, horizon=12),
        "Pure GCN (No Recurrence)": PureGCNModel(num_nodes=num_nodes, in_features=1, hidden_dim=hidden_dim, horizon=12),
        "Linear Regressor Baseline": LinearBaseline(num_nodes=num_nodes, in_features=1, history_window=12, horizon=12)
    }

    results_summary = []
    criterion = nn.L1Loss()

    os.path.dirname("./checkpoints/")

    # Python dict iteration syntax
    for name, model in models_to_test.items():
        logger.info(f"\n================ Training Model: {name} ================")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
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

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            logger.info(
                f"[{name}] Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        elapsed_time = time.time() - start_time

        # Save individual loss curve plot
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_losses(train_losses, val_losses, save_path=f"./checkpoints/loss_curve_{safe_name}.png")

        results_summary.append({
            "Model": name,
            "Best Val Loss": best_val_loss,
            "Final Train Loss": train_losses[-1],
            "Training Time (s)": elapsed_time
        })

    print("======================================================================")
    print("                 MODEL COMPARISON EVALUATION SUMMARY                  ")
    print("======================================================================")
    print(f"{'Model Architecture':<30} | {'Best Val Loss':<15} | {'Final Train Loss':<16} | {'Time (s)':<10}")
    print("-" * 75)

    for res in results_summary:
        # Gracefully handle varying key names for time
        train_time = res.get('Time (s)', res.get('Time', res.get('Train Time', 0.0)))
        print(
            f"{res['Model']:<30} | {res['Best Val Loss']:<15.4f} | {res['Final Train Loss']:<16.4f} | {train_time:<10.1f}")
    print("======================================================================")
    print("=" * 70)


if __name__ == "__main__":
    # train_model(epochs=50, batch_size=64, lr=0.001, hidden_dim=128)
    # comparisoon run
    run_model_comparison(epochs=50, batch_size=64, lr=0.001, hidden_dim=128)

