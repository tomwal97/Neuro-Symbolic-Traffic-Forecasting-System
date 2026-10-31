# File: dataset.py
# Description: PyTorch Dataset class for loading preprocessed PeMS03 traffic tensors and adjacency matrices.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PeMSDataset(Dataset):
    """Custom PyTorch Dataset for PeMS03 spatial-temporal traffic forecasting."""

    def __init__(self, processed_dir: str = "./processed_data", history_window: int = 12, horizon: int = 12,
                 mode: str = "train"):
        """Initializes the dataset by loading preprocessed tensors and setting up time windows.

        Args:
            processed_dir: Path to directory containing X_tensor.npy and A_matrix.npy.
            history_window: Number of historical time steps to use as input (e.g., 1 hour = 12 steps).
            horizon: Number of future time steps to predict.
            mode: Dataset split ('train', 'val', or 'test').
        """
        self.history_window = history_window
        self.horizon = horizon
        self.mode = mode

        x_path = os.path.join(processed_dir, "X_tensor.npy")
        a_path = os.path.join(processed_dir, "A_matrix.npy")

        if not os.path.exists(x_path) or not os.path.exists(a_path):
            raise FileNotFoundError(f"Processed artifacts not found in {processed_dir}. Run data_ingestion.py first.")

        logger.info(f"Loading processed tensors from {processed_dir}...")
        self.x_data = np.load(x_path)  # Shape: [T, N, F]
        self.adj_matrix = np.load(a_path)  # Shape: [N, N]

        logger.info(f"Loaded X tensor shape: {self.x_data.shape}, Adjacency shape: {self.adj_matrix.shape}")

        # Temporal split ratios (Standard: 70% train, 10% validation, 20% test)
        total_steps = self.x_data.shape[0]
        train_end = int(total_steps * 0.7)
        val_end = int(total_steps * 0.8)

        if mode == "train":
            self.start_idx = 0
            self.end_idx = train_end
        elif mode == "val":
            self.start_idx = train_end - self.history_window
            self.end_idx = val_end
        elif mode == "test":
            self.start_idx = val_end - self.history_window
            self.end_idx = total_steps
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

        # Slice data for the specified split
        self.x_subset = self.x_data[self.start_idx:self.end_idx]

        # Calculate valid number of samples considering history and horizon
        self.num_samples = len(self.x_subset) - self.history_window - self.horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Dataset split '{mode}' is too small for history_window={history_window} and horizon={horizon}.")

        logger.info(f"Initialized '{mode}' dataset split with {self.num_samples} samples.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches a single sliding-window sample.

        Returns:
            x_sample: Historical input tensor of shape [history_window, N, F]
            y_sample: Target future tensor of shape [horizon, N, F] (or predicted feature slice)
        """
        start_t = idx
        end_t = idx + self.history_window
        target_end_t = end_t + self.horizon

        x_sample = self.x_subset[start_t:end_t]
        y_sample = self.x_subset[end_t:target_end_t]

        return torch.tensor(x_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.float32)

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Returns the spatial adjacency matrix as a PyTorch tensor."""
        return torch.tensor(self.adj_matrix, dtype=torch.float32)


# Quick verification script if run directly
if __name__ == "__main__":
    dataset = PeMSDataset(mode="train", history_window=12, horizon=12)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_x, sample_y = next(iter(dataloader))
    print(f"\nDataLoader Verification Successful!")
    print(f"Batch X shape (History): {sample_x.shape}")  # Expected: [32, 12, 358, 1]
    print(f"Batch Y shape (Target):  {sample_y.shape}")  # Expected: [32, 12, 358, 1]
    print(f"Adjacency matrix tensor shape: {dataset.get_adjacency_matrix().shape}")
