# File: evaluate.py
# Description: Test set evaluation script for Spatio-Temporal GCN-LSTM baseline.

import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json

# Import your actual model class from models.py
from models import SpatioTemporalBaseline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(preds, targets):
    """
    Computes MAE, RMSE, and MAPE between predictions and ground truth targets.
    """
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    # Avoid division by zero in MAPE by masking very small target values
    mask = np.abs(targets) > 1e-5
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100.0

    return mae, rmse, mape


def evaluate_model():
    # 1. Device configuration (matches your M4 Pro setup)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Load Adjacency Matrix
    adj_path = './processed_data/A_matrix.npy'
    if not os.path.exists(adj_path):
        adj_path = './data/processed_data/A_matrix.npy'  # Fallback path check

    if os.path.exists(adj_path):
        adj_np = np.load(adj_path)
        adj = torch.tensor(adj_np, dtype=torch.float32).to(device)
        logger.info(f"Loaded adjacency matrix from {adj_path}")
    else:
        logger.warning(f"Adjacency matrix not found at {adj_path}. Creating identity fallback.")
        adj = torch.eye(358, dtype=torch.float32).to(device)

    # 3. Load Processed Data and Construct Test Split Sliding Windows
    data_path = './processed_data/X_tensor.npy'
    if not os.path.exists(data_path):
        data_path = './data/processed_data/X_tensor.npy'  # Fallback path check

    if not os.path.exists(data_path):
        logger.error(f"Processed data tensor not found at {data_path}. Verify your preprocessing folder structure.")
        return

    logger.info(f"Loading raw data tensor from {data_path}")
    data_array = np.load(data_path)  # Expected shape: [Total_Timesteps, Nodes, Features] or [Total_Timesteps, Nodes]

    if data_array.ndim == 2:
        data_array = np.expand_dims(data_array, axis=-1)  # Ensure shape is [T, N, F]

    # Reconstruct sliding windows (12 history steps -> 12 horizon prediction steps)
    history_window = 12
    horizon = 12

    logger.info("Generating evaluation sliding windows...")
    X_list, y_list = [], []
    for i in range(len(data_array) - history_window - horizon + 1):
        x_window = data_array[i: i + history_window]
        y_window = data_array[i + history_window: i + history_window + horizon]
        X_list.append(x_window)
        y_list.append(y_window)

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)

    # Standard time-series split (using the final 20% as the held-out test set)
    num_samples = X_all.shape[0]
    test_start = int(num_samples * 0.8)

    X_test = torch.tensor(X_all[test_start:], dtype=torch.float32)
    y_test = torch.tensor(y_all[test_start:], dtype=torch.float32)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    logger.info(f"Initialized 'test' dataset split with {len(test_dataset)} samples.")

    # 4. Initialize Model (matching your 128 hidden dim training config)
    model = SpatioTemporalBaseline(
        num_nodes=358,
        in_features=1,
        hidden_dim=128,
        horizon=12
    ).to(device)

    # 5. Load Best Model Checkpoint
    checkpoint_path = './checkpoints/best_baseline_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Train the model first!")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Loaded model checkpoint from {checkpoint_path}")

    # 6. Evaluation Loop
    model.eval()
    all_preds = []
    all_targets = []

    logger.info("Starting test set evaluation...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass passing both input tensor and adjacency matrix
            preds = model(batch_x, adj)

            # Collect predictions and ground truths
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
#
    logger.info("=" * 40)
    logger.info("HORIZON-SPECIFIC MAE BREAKDOWN")
    logger.info("=" * 40)

    horizon_steps = all_preds.shape[1]
    for h in range(horizon_steps):
        # Extract predictions and targets for step h across all samples and nodes
        step_preds = all_preds[:, h, :, :]
        step_targets = all_targets[:, h, :, :]
        step_mae = np.mean(np.abs(step_preds - step_targets))
        logger.info(f"Step {h + 1:2d} (+{(h + 1) * 5:2d} mins) MAE: {step_mae:.4f}")

    logger.info("=" * 40)
 #
    # 7. Compute Metrics
    mae, rmse, mape = calculate_metrics(all_preds, all_targets)

    logger.info("=" * 40)
    logger.info("FINAL TEST SET EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Normalized MAE  : {mae:.4f}")
    logger.info(f"Normalized RMSE : {rmse:.4f}")
    logger.info(f"MAPE            : {mape:.2f}%")
    logger.info("=" * 40)

    # Save evaluation results artifact for the report
    os.makedirs("./checkpoints", exist_ok=True)
    results = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
    with open("./checkpoints/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Evaluation metrics saved to ./checkpoints/evaluation_metrics.json")


if __name__ == "__main__":
    evaluate_model()