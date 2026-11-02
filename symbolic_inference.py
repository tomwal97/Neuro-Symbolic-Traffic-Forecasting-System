# ==============================================================================
# File: symbolic_inference.py
# Description: Bridge connecting GCN-LSTM predictions to SWI-Prolog Engine
# ==============================================================================

import os
import torch
import numpy as np
from pyswip import Prolog
import sys

# Import your model definition
from models import SpatioTemporalBaseline


def denormalize_speed(normalized_speed, mean=60.0, std=12.0):
    """
    Inverse Z-score transform to map normalized predictions back to real-world mph.
    """
    return (normalized_speed * std) + mean


def run_neuro_symbolic_pipeline():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Running Neuro-Symbolic Pipeline on device: {device}")

    # 1. Load Model and Weights
    model = SpatioTemporalBaseline(num_nodes=358, in_features=1, hidden_dim=128, horizon=12).to(device)
    checkpoint_path = './checkpoints/best_baseline_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Load Adjacency and Sample Test Batch
    adj_path = './processed_data/A_matrix.npy'
    data_path = './processed_data/X_tensor.npy'

    if not os.path.exists(adj_path) or not os.path.exists(data_path):
        adj_path = './data/processed_data/A_matrix.npy'
        data_path = './data/processed_data/X_tensor.npy'

    adj = torch.tensor(np.load(adj_path), dtype=torch.float32).to(device)
    raw_data = np.load(data_path)

    # Prepare a single sample window (12 history steps)
    #sample_x = torch.tensor(raw_data[:12], dtype=torch.float32).unsqueeze(0).to(device) # uncomment for single 12 set time step

    # Instead of raw_data[:12], jump to a peak hour slice (e.g., index 5000)
    start_idx = 12000
    sample_x = torch.tensor(raw_data[start_idx: start_idx + 12], dtype=torch.float32).unsqueeze(0).to(device)

    if sample_x.ndim == 3:
        sample_x = sample_x.unsqueeze(-1)  # Shape: [1, 12, 358, 1]


    # 3. Generate Neural Predictions

    with torch.no_grad():
        preds = model(sample_x, adj)  # Output shape: [1, 12, 358, 1]

    # Extract prediction grid [12 steps, 358 nodes] and denormalize to mph
    preds_np = preds.squeeze(0).squeeze(-1).cpu().numpy()
    speeds_mph = denormalize_speed(preds_np)

    # --- TEMPORARY DEMO INJECTION: Simulate a sudden crash at Node 12 ---
    # This forces a severe drop so you can see Prolog catch congestions & spillbacks
    speeds_mph[5, 12] = 22.5  # Step 5 at Node 12 drops to severe congestion (22.5 mph)
    speeds_mph[6, 15] = 28.0  # Step 6 at Downstream Node 15 experiences spillback
    speeds_mph[1, 5] = 65.0
    speeds_mph[2, 5] = 35.0  # Sudden 30 mph drop (Shockwave test)

    print(f"[*] Generated 12-step predictions across 358 nodes.")
    print(f"    Speed Range: Min = {speeds_mph.min():.2f} mph | Max = {speeds_mph.max():.2f} mph\n")

    # 4. Initialize SWI-Prolog Engine & Load Rules
    prolog = Prolog()
    prolog.consult("traffic_rules.pl")

    # Retract any leftover dynamic facts
    prolog.query("retractall(predicted_speed(_, _, _))")
    prolog.query("retractall(connected(_, _))")

    # 5. Populate Prolog Database with Dynamic Facts (Speeds)
    horizon, num_nodes = speeds_mph.shape
    for step in range(horizon):
        for node in range(num_nodes):
            val = float(speeds_mph[step, node])
            prolog.assertz(f"predicted_speed({node}, {step + 1}, {val:.2f})")

    # --- TEMPORARY DEMO INJECTION: Force a connection between Node 12 and Node 15 ---
    prolog.assertz("connected(12, 15)")
    # 6. Populate Prolog Database with Spatial Graph Edges (Adjacency Matrix A)
    epsilon = 0.1
    edge_count = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj[i, j].item() >= epsilon:
                prolog.assertz(f"connected({i}, {j})")
                edge_count += 1

    print(f"[*] Asserted {edge_count} spatial graph edges into Prolog knowledge base.")

    # 7. Execute Symbolic Rule Queries
    print("=" * 60)
    print("        SYMBOLIC REASONING ENGINE OUTPUT REPORT        ")
    print("=" * 60)

    # Medium Congestion warning
    warning_query = "warning_congestion(Node, Step, Speed)."
    warning_results = list(prolog.query(warning_query))

    print(f"[WARNING] Moderate Congestion Notified (40 - 55 mph): {len(warning_results)} instances found.")
    for res in warning_results[:5]:
        print(f"  -> Node {res['Node']} at Step {res['Step']} slowed to {res['Speed']:.1f} mph")

    # Query 1: Severe Congestion Alerts
    query_congested = list(prolog.query("is_severely_congested(Node, Step, Speed)"))
    print(f"\n[ALERT 1] Severe Congestion Detected (< 30 mph): {len(query_congested)} instances found.")
    for res in query_congested[:5]:
        print(
            f"  -> Node {res['Node']:3d} | Step {res['Step']:2d} (+{res['Step'] * 5:2d}m) | Speed: {res['Speed']:.1f} mph")

    # Query 2: Sudden Bottlenecks / Shockwaves
    query_bottleneck = list(prolog.query("sudden_bottleneck(Node, Step, Drop)"))
    print(f"\n[ALERT 2] Sudden Bottleneck/Shockwaves (>15 mph drop): {len(query_bottleneck)} instances found.")
    for res in query_bottleneck[:5]:
        print(
            f"  -> Node {res['Node']:3d} | Step {res['Step']:2d} (+{res['Step'] * 5:2d}m) | Speed Drop: {res['Drop']:.1f} mph")

    # Query 3: Spatial Spillback Propagation
    query_spillback = list(prolog.query("spatial_spillback(Upstream, Downstream, Step)"))
    print(f"\n[ALERT 3] Spatial Spillback Propagation: {len(query_spillback)} instances found.")
    if not query_spillback:
        print("  -> No downstream congestion propagation detected in this window.")
    else:
        for res in query_spillback[:5]:
            print(
                f"  -> Spillback: Node {res['Upstream']} (Upstream) propagated to Node {res['Downstream']} at Step {res['Step']} (+{res['Step'] * 5:2d}m)")

    # Query 4: Unphysical Guardrail Violation (Neural Speed Spikes)
    query_spikes = list(prolog.query("unrealistic_speed_jump(Node, Step, Increase)"))
    print(f"\n[GUARDRAIL] Unrealistic Speed Jump (>25 mph increase): {len(query_spikes)} instances found.")
    if not query_spikes:
        print("  -> PASSED: No unphysical neural prediction jumps detected.")
    else:
        for res in query_spikes[:5]:
            print(f"  -> WARNING: Node {res['Node']:3d} at Step {res['Step']:2d} jumped by {res['Increase']:.1f} mph")

    # Query 5: Free-Flow Traffic Nodes (>= 55 mph)
    query_free_flow = list(prolog.query("is_free_flow(Node, Step, Speed)"))
    print(f"\n[INFO] Free-Flow Traffic (>= 55 mph): {len(query_free_flow)} instances found.")
    for res in query_free_flow[:5]:
        print(
            f"  -> Node {res['Node']:3d} | Step {res['Step']:2d} (+{res['Step'] * 5:2d}m) | Speed: {res['Speed']:.1f} mph")

    print("=" * 60)


    # --- EXPORT RECOMMENDATIONS FOR TEAM ARBITRATION (Aria's Arbiter Format) ---
    with open("bottleneck_alert_output.pl", "w") as f:
        # 0. Moderate Congestion Warnings -> Class: monitor
        for res in warning_results:
            node_id = f"node_{res['Node']}"
            step = res['Step']
            speed = f"{res['Speed']:.1f}"
            f.write(
                f"recommendation(bottleneck_alert, {node_id}, {start_idx + step}, moderate_congestion({speed}), monitor, [warning_congestion]).\n")

        # 1. Severe Congestion -> Class: alert
        for res in query_congested:
            node_id = f"node_{res['Node']}"
            step = res['Step']
            speed = f"{res['Speed']:.1f}"
            f.write(
                f"recommendation(bottleneck_alert, {node_id}, {start_idx + step}, speed_drop({speed}), alert, [is_severely_congested]).\n")

        # 2. Sudden Bottlenecks / Shockwaves -> Class: alert
        for res in query_bottleneck:
            node_id = f"node_{res['Node']}"
            step = res['Step']
            drop = f"{res['Drop']:.1f}"
            f.write(
                f"recommendation(bottleneck_alert, {node_id}, {start_idx + step}, shockwave_drop({drop}), alert, [sudden_bottleneck]).\n")

        # 3. Spatial Spillback -> Class: alert
        for res in query_spillback:
            upstream_id = f"node_{res['Upstream']}"
            downstream_id = f"node_{res['Downstream']}"
            step = res['Step']
            f.write(
                f"recommendation(bottleneck_alert, {downstream_id}, {start_idx + step}, spillback_from({upstream_id}), alert, [spatial_spillback]).\n")

        # 4. Guardrail Violations -> Class: monitor
        for res in query_spikes:
            node_id = f"node_{res['Node']}"
            step = res['Step']
            increase = f"{res['Increase']:.1f}"
            f.write(
                f"recommendation(bottleneck_alert, {node_id}, {start_idx + step}, neural_spike_jump({increase}), monitor, [unrealistic_speed_jump]).\n")

            # Free-Flow -> Class: monitor (Optimal flow tracking)
            for res in query_free_flow:
                node_id = f"node_{res['Node']}"
                step = res['Step']
                speed = f"{res['Speed']:.1f}"
                f.write(
                    f"recommendation(bottleneck_alert, {node_id}, {start_idx + step}, optimal_flow({speed}), monitor, [is_free_flow]).\n")

    print("[*] Exported structured recommendation/6 facts for team_arbitration.pl")


if __name__ == "__main__":
    run_neuro_symbolic_pipeline()