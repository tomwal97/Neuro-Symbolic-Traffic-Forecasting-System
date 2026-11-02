# File: interactive_demo.py
# Description: Interactive CLI wrapper for running Neuro-Symbolic traffic inference across custom indices and peak times.
# Student Name: Thomas William Wallace
# Student FAN: wall0496
# Date: 03-11-2026

import os
import torch
import logging
from dataset import PeMSDataset
from models import SpatioTemporalBaseline
from pyswip import Prolog
import json

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


def load_scaling_stats(processed_dir="./processed_data"):
    stats_path = os.path.join(processed_dir, "scaling_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]
    # Fallback defaults if stats file isn't found
    return 60.0, 12.0


def load_inference_engine():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Initializing device: {device}")

    # Load validation dataset to sample indices safely
    val_dataset = PeMSDataset(processed_dir="./processed_data", history_window=12, horizon=12, mode="val")
    adj_matrix = val_dataset.get_adjacency_matrix().to(device)
    num_nodes = adj_matrix.shape[0]

    # Load trained model checkpoint
    model = SpatioTemporalBaseline(num_nodes=num_nodes, in_features=1, hidden_dim=128, horizon=12).to(device)
    ckpt_path = "./checkpoints/best_baseline_model.pth"

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("[*] Loaded trained model weights successfully.")
    else:
        print(
            "[!] Warning: No checkpoint found at ./checkpoints/best_baseline_model.pth. Running with uninitialized weights.")

    model.eval()
    return model, val_dataset, adj_matrix, device


def run_inference_for_index(target_idx, model, dataset, adj_matrix, device):
    if target_idx < 0 or target_idx >= len(dataset):
        print(f"[!] Error: Index {target_idx} is out of bounds for the dataset (Max: {len(dataset) - 1}).")
        return

    # Fetch sample
    history_x, target_y = dataset[target_idx]
    # Add batch dimension: [1, 12, num_nodes, 1]
    batch_x = history_x.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(batch_x, adj_matrix).squeeze(0)  # Shape: [horizon, num_nodes]

    # Load precise scaling statistics
    mean_val, std_val = load_scaling_stats()

    # Initialize SWI-Prolog and load rules
    prolog = Prolog()
    prolog.consult("traffic_rules.pl")

    # Clear previous facts
    list(prolog.query("retractall(predicted_speed(_,_,_))"))

    # Assert predictions into Prolog with precise inverse Z-score transformation
    step_to_evaluate = 0  # 0-indexed for step 1 (+5m)
    for node_idx in range(preds.shape[1]):
        normalized_val = preds[step_to_evaluate, node_idx].item()

        # Precise Inverse Z-Score: x = (x_norm * std) + mean
        speed_mph = (normalized_val * std_val) + mean_val

        prolog.assertz(f"predicted_speed({node_idx}, {step_to_evaluate + 1}, {speed_mph:.2f})")

    print("\n" + "=" * 60)
    print(f"       SYMBOLIC REASONING REPORT FOR INDEX {target_idx}")
    print("=" * 60)

    # Query Prolog rules matching your exact traffic_rules.pl definitions
    free_flow = list(prolog.query("is_free_flow(Node, Step, Speed)"))
    warnings = list(prolog.query("warning_congestion(Node, Step, Speed)"))
    alerts = list(prolog.query("is_severely_congested(Node, Step, Speed)"))
    bottlenecks = list(prolog.query("sudden_bottleneck(Node, StepNext, Drop)"))
    spillbacks = list(prolog.query("spatial_spillback(Upstream, Downstream, StepNext)"))
    guardrails = list(prolog.query("unrealistic_speed_jump(Node, StepNext, Increase)"))

    print(f"[INFO] Free-Flow Traffic (>= 55 mph): {len(free_flow)} instances found.")
    for res in free_flow[:3]:
        print(f"  -> Node {res['Node']} | Step {res['Step']} (+ {res['Step'] * 5}m) | Speed: {res['Speed']:.1f} mph")

    print(f"[WARNING] Moderate Congestion (40 - 55 mph): {len(warnings)} instances found.")
    for res in warnings[:3]:
        print(f"  -> Node {res['Node']} | Step {res['Step']} (+ {res['Step'] * 5}m) | Speed: {res['Speed']:.1f} mph")

    print(f"[ALERT 1] Severe Congestion (< 40 mph): {len(alerts)} instances found.")
    for res in alerts[:3]:
        print(f"  -> Node {res['Node']} | Step {res['Step']} (+ {res['Step'] * 5}m) | Speed: {res['Speed']:.1f} mph")

    print(f"[ALERT 2] Sudden Bottlenecks (> 15 mph drop): {len(bottlenecks)} instances found.")
    for res in bottlenecks[:3]:
        print(f"  -> Node {res['Node']} | Step {res['StepNext']} | Drop: {res['Drop']:.1f} mph")

    print(f"[ALERT 3] Spatial Spillback Propagation: {len(spillbacks)} instances found.")
    for res in spillbacks[:3]:
        print(f"  -> Upstream Node {res['Upstream']} -> Downstream Node {res['Downstream']} at Step {res['StepNext']}")

    print(f"[GUARDRAIL] Unrealistic Speed Jumps (> 25 mph increase): {len(guardrails)} instances found.")
    for res in guardrails[:3]:
        print(f"  -> Node {res['Node']} | Step {res['StepNext']} | Jump: {res['Increase']:.1f} mph")

    print("=" * 60 + "\n")


def main_menu():
    model, dataset, adj_matrix, device = load_inference_engine()

    while True:
        print("\n=== NEURO-SYMBOLIC TRAFFIC INFERENCE CLI ===")
        print("1. Run Peak Morning Commute (Index 372 - 7:00 AM)")
        print("2. Run Peak Evening Commute (Index 492 - 5:00 PM)")
        print("3. Run Off-Peak Midnight (Index 288 - 12:00 AM)")
        print("4. Run Off-Peak Midday (Index 432 - 12:00 PM)")
        print("5. Enter Custom Dataset Index")
        print("6. Exit")

        choice = input("Select an option (1-6): ").strip()

        if choice == "1":
            run_inference_for_index(372, model, dataset, adj_matrix, device)
        elif choice == "2":
            run_inference_for_index(492, model, dataset, adj_matrix, device)
        elif choice == "3":
            run_inference_for_index(288, model, dataset, adj_matrix, device)
        elif choice == "4":
            run_inference_for_index(432, model, dataset, adj_matrix, device)
        elif choice == "5":
            try:
                custom_idx = int(input(f"Enter index (0 to {len(dataset) - 1}): "))
                run_inference_for_index(custom_idx, model, dataset, adj_matrix, device)
            except ValueError:
                print("[!] Please enter a valid integer.")
        elif choice == "6":
            print("[*] Exiting interactive inference tool. Goodbye!")
            break
        else:
            print("[!] Invalid selection. Please choose a number between 1 and 6.")


if __name__ == "__main__":
    main_menu()