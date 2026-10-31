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
import h5py
import logging
import os
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

# === 1 (Extraction & Fetching) AI ===

# ================================
class PeMSExtractor:
    """Handles raw data ingestion from PeMS benchmark files (.npz, .h5) and Caltrans station metadata CSV files."""

    def __init__(self, data_path: str, metadata_path: str, sensor_id_path: Optional[str] = None):
        """Initializes the extractor with input file paths.

        Args:
            data_path: Path to the raw time-series data file (.npz or .h5).
            metadata_path: Path to the raw Caltrans station metadata CSV file.
        """
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.sensor_id_path = sensor_id_path

    def load_raw_benchmark_data(self) -> Dict[str, np.ndarray]:
        """Loads raw PeMS benchmark time-series speed/volume matrices and sensor IDs.

        Handles both NumPy archive (.npz) and HDF5 (.h5) binary formats.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'data': Raw speed/volume array (T, N) or (T, N, F).
                - 'sensor_ids': Array of raw sensor ID strings/integers.

        Raises:
            FileNotFoundError: If the benchmark file is missing.
            ValueError: If the file format is unsupported or necessary keys are absent.
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found at path: {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")

        logger.info(f"Loading raw PeMS benchmark file from: {self.data_path}")

        try:
            if self.data_path.endswith(".npz"):
                with np.load(self.data_path, allow_pickle=True) as raw_archive:
                    keys = list(raw_archive.keys())
                    logger.info(f"Successfully opened .npz archive. Found keys: {keys}")

                    # Standard PeMS keys are 'data'
                    if "data" in raw_archive:
                        data = raw_archive["data"]
                    else:
                        data = raw_archive[keys[0]]

                    # Load official sensor IDs from text file if provided, otherwise fallback
                    if self.sensor_id_path and os.path.exists(self.sensor_id_path):
                        logger.info(f"Loading sensor IDs from text file: {self.sensor_id_path}")
                        sensor_ids = np.loadtxt(self.sensor_id_path, dtype=int)
                    else:
                        sensor_ids = (
                            raw_archive["ids"]
                            if "ids" in raw_archive
                            else raw_archive["sensor_ids"] if "sensor_ids" in raw_archive
                            else np.arange(data.shape[1])
                        )

            elif self.data_path.endswith(".h5") or self.data_path.endswith(".hdf5"):
                with h5py.File(self.data_path, "r") as raw_h5:
                    keys = list(raw_h5.keys())
                    logger.info(f"Successfully opened .h5 file. Found keys: {keys}")

                    # Look up primary dataset inside HDF5 hierarchy
                    if "df" in raw_h5:
                        data = raw_h5["df"]["block0_values"][:]
                        sensor_ids = raw_h5["df"]["axis0"][:]
                    elif "data" in raw_h5:
                        data = raw_h5["data"][:]
                        sensor_ids = raw_h5["ids"][:] if "ids" in raw_h5 else np.arange(data.shape[1])
                    else:
                        data = raw_h5[keys[0]][:]
                        sensor_ids = np.arange(data.shape[1])

            else:
                raise ValueError("Unsupported file format. Only '.npz' and '.h5' / '.hdf5' files are supported.")

            logger.info(f"Raw data matrix loaded successfully. Shape: {data.shape}")
            logger.info(f"Extracted {len(sensor_ids)} sensor IDs.")

            return {
                "data": data,
                "sensor_ids": np.array(sensor_ids)
            }

        except Exception as e:
            logger.error(f"Failed to read benchmark file {self.data_path}: {str(e)}")
            raise

    def load_raw_station_metadata(self) -> pd.DataFrame:
        """Fetches and parses the raw Caltrans Station Metadata CSV file into a DataFrame.

        Returns:
            pd.DataFrame: Unmodified, uncleaned raw station metadata DataFrame.

        Raises:
            FileNotFoundError: If the metadata CSV file is missing.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        if not os.path.exists(self.metadata_path):
            logger.error(f"Metadata file not found at path: {self.metadata_path}")
            raise FileNotFoundError(f"File not found: {self.metadata_path}")

        logger.info(f"Loading raw Caltrans station metadata CSV from: {self.metadata_path}")

        try:
            # Load raw CSV without dropping missing values or modifying types
            metadata_df = pd.read_csv(self.metadata_path, sep=None, engine="python")
            logger.info(f"Metadata successfully loaded. Shape: {metadata_df.shape}")
            logger.info(f"Metadata Columns: {list(metadata_df.columns)}")

            return metadata_df

        except Exception as e:
            logger.error(f"Failed to parse metadata CSV at {self.metadata_path}: {str(e)}")
            raise

    def extract_all(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Convenience method to execute full raw data and metadata collection.

        Returns:
            Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
                - raw_data: Raw speed/volume matrix (T, N) or (T, N, F).
                - raw_sensor_ids: Raw sensor ID list.
                - raw_metadata_df: Raw Caltrans metadata DataFrame.
        """
        data_dict = self.load_raw_benchmark_data()
        raw_metadata_df = self.load_raw_station_metadata()

        return data_dict["data"], data_dict["sensor_ids"], raw_metadata_df


def extract_raw_pems_data(
    data_path: str, metadata_path: str, sensor_id_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Top-level functional interface for raw extraction step."""
    extractor = PeMSExtractor(
        data_path=data_path,
        metadata_path=metadata_path,
        sensor_id_path=sensor_id_path
    )
    return extractor.extract_all()

# ================================

# === 2 (Validation & Cleansing) ===

# ================================
# Configure module-level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrafficDataValidator:
    """Handles data sanity checks, missing value imputation, and anomaly filtering."""

    def __init__(self, max_speed_threshold: float = 120.0):
        """Initializes validation thresholds.

        Args:
            max_speed_threshold: Maximum plausible highway speed limit in mph.
        """
        self.max_speed_threshold = max_speed_threshold

    def clean_speed_matrix(self, speed_matrix: np.ndarray) -> np.ndarray:
        """Cleans anomalies, rogue zeros, and spikes from the raw speed matrix.

        Args:
            speed_matrix: Raw speed array of shape (T, N).

        Returns:
            np.ndarray: Cleaned speed matrix.
        """
        logger.info("Performing anomaly detection and cleaning on speed matrix...")
        cleaned = speed_matrix.copy()

        # 1. Flag impossible spikes (> max speed threshold)
        spike_mask = cleaned > self.max_speed_threshold
        if np.any(spike_mask):
            logger.warning(
                f"Detected {np.sum(spike_mask)} speed spikes > {self.max_speed_threshold} mph. Capping values.")
            cleaned[spike_mask] = self.max_speed_threshold

        # 2. Handle rogue zeros (sensor hardware dropouts during active traffic reporting 0 mph)
        # We treat isolated 0s as NaNs temporarily for imputation if they represent dropouts
        zero_mask = cleaned == 0.0
        if np.any(zero_mask):
            logger.info(
                f"Detected {np.sum(zero_mask)} zero-speed entries. Treating as missing telemetry for imputation.")
            cleaned[zero_mask] = np.nan

        return cleaned

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies forward-filling with column median fallback for missing time-series entries.

        Args:
            df: Pandas DataFrame of time-series telemetry (T, N).

        Returns:
            pd.DataFrame: Imputed DataFrame with zero NaNs.
        """
        logger.info("Applying temporal forward-fill and median fallback imputation...")
        # Forward fill along time axis, backfill any leading NaNs, and fallback to column median
        imputed_df = df.ffill().bfill().fillna(df.median())

        if imputed_df.isna().sum().sum() > 0:
            # Absolute fallback if an entire column was NaN
            imputed_df = imputed_df.fillna(0.0)

        return imputed_df

    def validate_and_clean_pipeline(self, raw_data: np.ndarray, sensor_ids: np.ndarray, metadata_df: pd.DataFrame) \
            -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Executes full validation, anomaly cleansing, and metadata alignment.

        Args:
            raw_data: Raw time-series array (T, N) or (T, N, F).
            sensor_ids: Raw sensor ID list.
            metadata_df: Raw Caltrans station metadata DataFrame.

        Returns:
            Tuple containing cleaned data matrix, filtered active sensor IDs, and valid metadata.
        """
        logger.info("Initiating validation and cleansing pipeline...")

        # Handle multi-feature vs single-feature shapes
        if raw_data.ndim == 3:
            # Shape is [T, N, F] (e.g., Feature 0 = Speed, Feature 1 = Flow)
            speed_raw = raw_data[:, :, 0]
            flow_raw = raw_data[:, :, 1] if raw_data.shape[-1] > 1 else None
        else:
            speed_raw = raw_data
            flow_raw = None

        # Clean speed anomalies
        cleaned_speed = self.clean_speed_matrix(speed_raw)

        # Convert to DataFrame for robust temporal imputation
        speed_df = pd.DataFrame(cleaned_speed, columns=sensor_ids)
        imputed_speed_df = self.impute_missing_values(speed_df)

        if flow_raw is not None:
            flow_df = pd.DataFrame(flow_raw, columns=sensor_ids)
            imputed_flow_df = self.impute_missing_values(flow_df)
            cleaned_data = np.stack([imputed_speed_df.values, imputed_flow_df.values], axis=-1)
        else:
            cleaned_data = imputed_speed_df.values

        # Filter metadata to drop orphan sensors not present in our data matrix
        valid_metadata = metadata_df[metadata_df["station_id"].isin(sensor_ids)].reset_index(drop=True)
        logger.info(f"Validation complete. Active sensors retained: {len(valid_metadata)}")

        return cleaned_data, sensor_ids, valid_metadata



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

    def map_and_enrich_metadata(self, sensor_ids: np.ndarray, metadata_df: Optional[pd.DataFrame] = None) \
            -> pd.DataFrame:
        """Maps sensor IDs to positional indices (0..N-1) and enriches with Caltrans metadata."""
        logger.info("Mapping sensor IDs to positional indices (0..N-1) and enriching metadata...")

        indexed_meta = []

        # Ensure metadata_df is available for lookup
        if metadata_df is not None and not metadata_df.empty:
            # Create a dictionary lookup from station_id to metadata row
            meta_dict = metadata_df.set_index("station_id").to_dict("index")
        else:
            meta_dict = {}

        for idx, sid in enumerate(sensor_ids):
            # Fetch real metadata if it exists for this station ID
            station_info = meta_dict.get(int(sid), {})

            indexed_meta.append({
                "node_idx": idx,
                "station_id": int(sid),
                "fwy": station_info.get("fwy", 0),
                "dir": station_info.get("dir", "N/A"),
                "type": station_info.get("type", "ML"),
                "latitude": station_info.get("latitude", 0.0),
                "longitude": station_info.get("longitude", 0.0)
            })

        enriched_df = pd.DataFrame(indexed_meta)
        logger.info(f"Enriched metadata created successfully with real attributes for {len(enriched_df)} sensors.")
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

    def construct_adjacency_matrix(self, sensor_ids: np.ndarray,
                                   dist_csv_path: str = "datasets/PEMS03.csv") -> np.ndarray:
        """Constructs spatial Graph Adjacency Matrix (A in R^{N x N}) directly from PEMS03.csv edge distances."""
        logger.info(f"Constructing spatial graph adjacency matrix from edge distance file: {dist_csv_path}")
        n_nodes = len(sensor_ids)
        node_to_idx = {node: idx for idx, node in enumerate(sensor_ids)}

        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        dist_df = pd.read_csv(dist_csv_path)

        for _, row in dist_df.iterrows():
            u, v, dist = int(row['from']), int(row['to']), float(row['distance'])
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                # Gaussian kernel with sigma=10.0
                weight = np.exp(-(dist ** 2) / (self.sigma ** 2))

                # Keep edge if it meets epsilon threshold (or use inverse distance if preferred)
                if weight >= self.epsilon:
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight  # Symmetric graph representation

        np.fill_diagonal(adj_matrix, 0.0)
        edge_count = np.count_nonzero(adj_matrix)
        logger.info(f"Graph constructed successfully. Nodes: {n_nodes}, Edges: {edge_count}")
        return adj_matrix

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
class TrafficDataPersistence:
    """Handles serialisation and persistence of processed tensors and adjacency matrices to disk."""

    def __init__(self, output_dir: str = "./processed_data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_processed_artifacts(self, x_tensor: np.ndarray, a_matrix: np.ndarray, metadata_df: pd.DataFrame) -> Dict[
        str, str]:
        """Saves processed 3D feature tensors, adjacency matrix, and enriched metadata to disk.

        Args:
            x_tensor: Processed 3D feature tensor of shape [T, N, F].
            a_matrix: Thresholded spatial adjacency matrix of shape [N, N].
            metadata_df: Enriched metadata DataFrame.

        Returns:
            Dict[str, str]: Filepaths of saved artifacts.
        """
        logger.info(f"Persisting processed artifacts to directory: {self.output_dir}")

        x_path = os.path.join(self.output_dir, "X_tensor.npy")
        a_path = os.path.join(self.output_dir, "A_matrix.npy")
        meta_path = os.path.join(self.output_dir, "enriched_metadata.csv")

        np.save(x_path, x_tensor)
        np.save(a_path, a_matrix)
        metadata_df.to_csv(meta_path, index=False)

        logger.info("Artifact persistence complete.")
        return {
            "x_tensor_path": x_path,
            "a_matrix_path": a_path,
            "metadata_path": meta_path
        }


def run_complete_ingestion_pipeline(
    data_path: str,
    metadata_path: str,
    sensor_id_path: Optional[str] = None,
    output_dir: str = "./processed_data"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """End-to-end execution function running Extraction -> Validation -> Transformation -> Persistence."""
    logger.info("=== STARTING END-TO-END DATA INGESTION PIPELINE ===")

    # 1. Extraction (Pass sensor_id_path here)
    raw_data, raw_sensor_ids, raw_meta = extract_raw_pems_data(
        data_path, metadata_path, sensor_id_path=sensor_id_path
    )

    # 2. Validation & Cleansing
    validator = TrafficDataValidator()
    cleaned_data, valid_ids, valid_meta = validator.validate_and_clean_pipeline(
        raw_data, raw_sensor_ids, raw_meta
    )

    # 3. Transformation & Normalisation
    transformer = TrafficTransformer(sigma=10.0, epsilon=0.01)  # Lower epsilon to 0.01 to ensure edges populate
    enriched_meta = transformer.map_and_enrich_metadata(valid_ids, valid_meta)

    # Construct adjacency matrix from spatial coordinates
    a_matrix = transformer.construct_adjacency_matrix(valid_ids, dist_csv_path="datasets/PEMS03.csv")

    # Normalize features
    if cleaned_data.ndim == 3:
        speed_norm, _ = transformer.normalize_features(cleaned_data[:, :, 0])
        flow_norm, _ = transformer.normalize_features(cleaned_data[:, :, 1])
        x_tensor = transformer.shape_to_3d_tensor(speed_norm, flow_norm)
    else:
        speed_norm, _ = transformer.normalize_features(cleaned_data)
        x_tensor = transformer.shape_to_3d_tensor(speed_norm)

    # 4. Persistence
    persistor = TrafficDataPersistence(output_dir=output_dir)
    persistor.save_processed_artifacts(x_tensor, a_matrix, enriched_meta)

    logger.info("=== DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY ===")
    return x_tensor, a_matrix, enriched_meta

# execution block
if __name__ == "__main__":
    DATA_PATH = "datasets/PEMS03.npz"
    METADATA_PATH = "datasets/pems_station_metadata_with_latlng.csv"
    SENSOR_ID_PATH = "datasets/PEMS03.txt"
    OUTPUT_DIR = "./processed_data"

    try:
        X_tensor, A_matrix, enriched_meta = run_complete_ingestion_pipeline(
            data_path=DATA_PATH,
            metadata_path=METADATA_PATH,
            sensor_id_path=SENSOR_ID_PATH,
            output_dir=OUTPUT_DIR
        )
        print("\nPipeline executed successfully!")
        print(f"X_tensor shape: {X_tensor.shape}")
        print(f"A_matrix shape: {A_matrix.shape}")
        print(f"Enriched metadata shape: {enriched_meta.shape}")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")