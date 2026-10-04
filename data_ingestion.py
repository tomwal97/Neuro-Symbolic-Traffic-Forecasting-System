# Student Name: [Your Name]
# Student FAN: [YourFAN]
# File: [data_ingestion.py]
# Date: [DD-MM-YYYY]
# Description: [Brief one-line description of the script's purpose.]
# Usage: # Licence: [Optional: e.g., python filename.py --input data.csv]

# === Imports and Dependencies ===
import torch
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

# ================================

# === Loading & Persistence ===

# ================================