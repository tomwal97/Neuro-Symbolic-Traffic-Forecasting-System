# File: models.py
# Description: Neural network architectures for traffic forecasting (Baseline Spatio-Temporal GCN-LSTM).

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GraphConvolutionLayer(nn.Module):
    """Simple Spectral Graph Convolution Layer."""

    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass for Graph Convolution.

        Args:
            x: Node feature tensor of shape [Batch, Nodes, In_Features] or [Batch, Time, Nodes, In_Features]
            adj: Normalized Adjacency matrix of shape [Nodes, Nodes]
        """
        # Handle 4D tensors [B, T, N, F] by reshaping temporarily or batching matrix multiplication
        is_4d = x.dim() == 4
        if is_4d:
            B, T, N, F = x.shape
            x = x.reshape(B * T, N, F)

        # Graph convolution operation: Support = X * W, Out = A * Support + bias
        support = torch.matmul(x, self.weight)

        # Ensure adjacency matrix is broadcastable or matches dimensions
        if adj.dim() == 2:
            out = torch.matmul(adj, support) + self.bias
        else:
            # Batch adjacency support if varying per batch
            out = torch.matmul(adj, support) + self.bias

        if is_4d:
            out = out.reshape(B, T, N, -1)

        return torch.relu(out)


class SpatioTemporalBaseline(nn.Module):
    """Baseline Spatio-Temporal Model combining GCN for spatial dependency and LSTM for temporal modeling."""

    def __init__(self, num_nodes: int, in_features: int = 1, hidden_dim: int = 64, horizon: int = 12):
        super(SpatioTemporalBaseline, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        # Spatial Graph Convolution
        self.gcn = GraphConvolutionLayer(in_features=in_features, out_features=hidden_dim)

        # Temporal Recurrent Unit (LSTM processes temporal sequence per node)
        # Input dim per node after GCN is hidden_dim
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_nodes,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Fully connected projection layer to map back to future horizon & features
        self.fc = nn.Linear(hidden_dim, horizon * in_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [Batch, History_Window, Nodes, Features] e.g., [32, 12, 358, 1]
            adj: Adjacency matrix of shape [Nodes, Nodes] e.g., [358, 358]

        Returns:
            torch.Tensor: Predicted future tensor of shape [Batch, Horizon, Nodes, Features]
        """
        B, T, N, F = x.shape

        # 1. Spatial modeling via GCN applied across time steps
        gcn_out = self.gcn(x, adj)  # Shape: [B, T, N, hidden_dim]

        # 2. Flatten nodes and features for temporal LSTM processing
        # Reshape to [B, T, N * hidden_dim]
        gcn_flat = gcn_out.reshape(B, T, N * gcn_out.shape[-1])

        # 3. Temporal modeling via LSTM
        lstm_out, _ = self.lstm(gcn_flat)  # Output shape: [B, T, hidden_dim]

        # Take the final hidden state of the history window to forecast the horizon
        last_hidden_state = lstm_out[:, -1, :]  # Shape: [B, hidden_dim]

        # 4. Project to future horizon
        fc_out = self.fc(last_hidden_state)  # Shape: [B, horizon * F]

        # Reshape back to [Batch, Horizon, Nodes, Features]
        # Note: Mapping vector to spatial dimensions for multi-node prediction
        predictions = fc_out.reshape(B, self.horizon, 1, F)
        # Expand across all nodes for baseline output structure matching [B, Horizon, N, F]
        predictions = predictions.expand(B, self.horizon, N, F)

        return predictions


# Quick verification script if run directly
if __name__ == "__main__":
    model = SpatioTemporalBaseline(num_nodes=358, in_features=1, hidden_dim=32, horizon=12)

    # Dummy batch matching your DataLoader shapes
    dummy_x = torch.randn(4, 12, 358, 1)  # [Batch=4, History=12, Nodes=358, Features=1]
    dummy_adj = torch.randn(358, 358)  # [Nodes=358, Nodes=358]

    output = model(dummy_x, dummy_adj)
    logger.info(f"Model Architecture Verification Successful!")
    logger.info(f"Input shape:  {dummy_x.shape}")
    logger.info(f"Output shape: {output.shape}")  # Expected: [4, 12, 358, 1]