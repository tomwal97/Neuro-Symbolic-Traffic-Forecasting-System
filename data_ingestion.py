# Student Name: [Your Name]
# Student FAN: [YourFAN]
# File: [data_ingestion.py]
# Date: [DD-MM-YYYY]
# Description: [Brief one-line description of the script's purpose.]
# Usage: # Licence: [Optional: e.g., python filename.py --input data.csv]

# === Imports and Dependencies ===
import torch
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
# ================================

# === Device Selection (Handles different OS of team members) ===
# Dynamic device selection for Mac MPS, CUDA GPU, or CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# ================================
# === Extraction & Fetching ===

# ================================

# === Validation & Cleansing ===

# ================================

# === Transformation & Normalisation ===
# Description: Modular transformer for feature normalisation, graph adjacency construction, and 3D tensor shaping.

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrafficTransformer:
    """Handles feature normalization, spatial graph construction, and 3D tensor shaping."""

    def __init__(self, sigma: float = 10.0, epsilon: float = 0.1):
        """Initializes transformer parameters.

        Args:
            sigma: Standard deviation hyperparameter for the thresholded Gaussian kernel.
            epsilon: Threshold cutoff value for adjacency matrix sparsification.
        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.mean_val: Optional[float] = None
        self.std_val: Optional[float] = None

    def map_and_enrich_metadata(self, sensor_ids: np.ndarray, metadata_df: pd.DataFrame) \
            -> pd.DataFrame:
        """Maps sensor IDs to positional indices (0..N-1) and creates an enriched metadata lookup table.

        Args:
            sensor_ids: Array of N unique sensor IDs matching matrix columns.
            metadata_df: Validated metadata DataFrame containing station attributes.

        Returns:
            pd.DataFrame: Enriched node metadata table indexed from 0 to N-1.
        """
        logger.info("Enriching node metadata and mapping matrix indices (0..N-1)...")
        indexed_meta = []

        for idx, sid in enumerate(sensor_ids):
            match = metadata_df[metadata_df["ID"] == sid]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["node_idx"] = idx
                indexed_meta.append(row)
            else:
                logger.warning(f"Sensor ID {sid} missing in metadata DataFrame. Inserting default entries.")
                indexed_meta.append({
                    "node_idx": idx,
                    "ID": sid,
                    "Fwy": 0,
                    "Dir": "N/A",
                    "Latitude": 0.0,
                    "Longitude": 0.0,
                    "Type": "ML"
                })

        enriched_df = pd.DataFrame(indexed_meta)
        return enriched_df

    def compute_haversine_distance(self, lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) \
            -> np.ndarray:
        """Calculates pairwise Haversine distance matrix in kilometers between coordinates.

        Args:
            lat1, lon1: Vectors of latitude and longitude (N,).
            lat2, lon2: Vectors of latitude and longitude (N,).

        Returns:
            np.ndarray: Pairwise distance matrix of shape (N, N).
        """
        R = 6371.0  # earth's radius in kilometers
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        dlat = lat2_rad[:, None] - lat1_rad[None, :]
        dlon = lon2_rad[:, None] - lon1_rad[None, :]

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad[None, :]) * np.cos(lat2_rad[:, None]) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return R * c

    def construct_adjacency_matrix(self, enriched_metadata: pd.DataFrame) \
            -> np.ndarray:
        """Constructs spatial Graph Adjacency Matrix (A in R^{N x N}) using a thresholded Gaussian kernel.

        Args:
            enriched_metadata: DataFrame containing node spatial attributes (Latitude, Longitude).

        Returns:
            np.ndarray: Weighted adjacency matrix A of shape (N, N).
        """
        logger.info("Constructing spatial graph adjacency matrix (A ∈ R^{N x N})...")
        lats = enriched_metadata["Latitude"].values
        lons = enriched_metadata["Longitude"].values
        n_nodes = len(enriched_metadata)

        # Pairwise distance calculation
        dist_matrix = self.compute_haversine_distance(lats, lons, lats, lons)

        # Thresholded Gaussian Kernel
        distances_sq = dist_matrix ** 2
        adj_matrix = np.exp(-distances_sq / (self.sigma ** 2))

        # Sparsification mask based on threshold epsilon
        adj_matrix[adj_matrix < self.epsilon] = 0.0

        # Remove self-loops
        np.fill_diagonal(adj_matrix, 0.0)

        logger.info(
            f"Graph constructed successfully. Nodes: {n_nodes}, Edge Density: {np.count_nonzero(adj_matrix) / (n_nodes ** 2):.4f}")
        return adj_matrix.astype(np.float32)

    def normalize_features(self, data_array: np.ndarray, fit_stats: bool = True) \
            -> Tuple[np.ndarray, Dict[str, float]]:
        """Standardizes speed and volume matrices using Z-score normalization.

        Args:
            data_array: Raw numeric feature matrix of shape (T, N) or (T, N, F).
            fit_stats: Whether to calculate mean and std from the input data.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Normalized feature array and normalization statistics dictionary.
        """
        logger.info("Applying Z-score feature normalization...")
        if fit_stats:
            self.mean_val = float(np.mean(data_array))
            self.std_val = float(np.std(data_array))
            if self.std_val == 0:
                self.std_val = 1.0

        normalized_data = (data_array - self.mean_val) / self.std_val
        stats = {"mean": self.mean_val, "std": self.std_val}
        return normalized_data.astype(np.float32), stats

    def shape_to_3d_tensor(self, speed_data: np.ndarray, volume_data: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """Restructures flat 2D time series into a 3D feature tensor of shape [T, N, F].

        Args:
            speed_data: Normalized speed matrix of shape (T, N).
            volume_data: Optional normalized traffic volume matrix of shape (T, N).

        Returns:
            np.ndarray: Formatted 3D tensor of shape [T, N, F].
        """
        logger.info("Restructuring data matrices into 3D Tensor [T, N, F]...")
        T, N = speed_data.shape

        if volume_data is not None:
            tensor_3d = np.stack([speed_data, volume_data], axis=-1)
        else:
            tensor_3d = np.expand_dims(speed_data, axis=-1)

        logger.info(f"Tensor successfully shaped to [T={T}, N={N}, F={tensor_3d.shape[-1]}].")
        return tensor_3d.astype(np.float32)
# ================================

# === Loading & Persistence ===

# ================================