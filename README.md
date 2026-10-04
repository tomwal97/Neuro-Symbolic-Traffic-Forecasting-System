# Neuro-Symbolic Traffic Forecasting

**Authors:** ADD EVERYONES NAMES HERE   
**Repository:** NeuroSymbolicTrafficForecasting  
**File:** `README.md`

---

## Overview
This repository contains the data ingestion pipeline and model implementations for the Neuro-Symbolic Traffic Prediction project. The project combines deep learning architectures (such as Graph Neural Networks) with symbolic rule engines to forecast traffic dynamics on spatiotemporal road networks.

---

## Prerequisites & Dependencies
Ensure **Python 3.10** or **3.11** (64-bit) is installed across your environment. Core dependencies include:
* `torch`, `torchvision`, `torchaudio`
* `numpy`
* `pandas`
* `scikit-learn`
* `requests`
* `pyyaml`

A complete list of pinned dependencies is maintained in `requirements.txt`.

---

## Environment & Device Compatibility

### 1. Dynamic Hardware Acceleration
The pipeline automatically detects and assigns the optimal compute device at runtime (CUDA GPU, Apple Silicon MPS, or fallback CPU):

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)