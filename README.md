# Neuro-Symbolic Traffic Forecasting

**Authors:** Thomas William Wallace & Project Team  
**Repository:** NeuroSymbolicTrafficForecasting  
**File:** `README.md`

---

## Overview
This repository contains the complete implementation for the **Neuro-Symbolic Traffic Forecasting** capstone project. The system bridges deep spatio-temporal graph neural networks with a symbolic logic reasoning engine to predict highway traffic dynamics, detect congestion tiers, model spatial spillbacks, and enforce unphysical neural prediction guardrails using the Caltrans PeMS03 dataset.

---

## Core Architecture
1. **Spatio-Temporal Graph Neural Network (GCN-LSTM):** Captures spatial network topology via spectral graph convolutions across road sensors combined with recurrent temporal LSTM layers.
2. **Symbolic Reasoning Engine (SWI-Prolog):** Translates neural predictions into deterministic knowledge-base facts, evaluating congestion tiers (moderate/severe), sudden shockwave drops, spatial spillback propagation across connected edges, and safety overrides.
3. **Arbitration & Export Bridge:** Aggregates symbolic rule violations into structured `recommendation/6` facts (`bottleneck_alert_output.pl`) for downstream team coordination.

---

## Repository Structure
```text
NeuroSymbolicTrafficForecasting/
│
├── models.py                  # Neural architectures (GCN-LSTM, Pure LSTM, Pure GCN, Linear)
├── train.py                   # Model comparison training loop and loss curve generation
├── symbolic_inference.py      # Neural-to-Prolog bridge and recommendation exporter
├── traffic_rules.pl           # SWI-Prolog symbolic reasoning rules and thresholds
├── checkpoints/               # Saved model weights (.pth) and training convergence plots
├── processed_data/            # Processed PeMS03 tensors (X_tensor.npy, A_matrix.npy)
├── requirements.txt           # Pinned python package dependencies
└── README.md                  # Project documentation