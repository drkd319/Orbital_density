"""
EGNN Encoder for Electron Density Model

Based on: "E(n) Equivariant Graph Neural Networks" (Satorras et al., 2021)
Adapted from Orbital_Energy implementation.

Key features:
- E(n) Equivariance: Equivariant to rotations, reflections, translations
- Outputs per-atom features for downstream cross-attention
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def build_radius_graph(
    pos: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor = None,
) -> torch.Tensor:
    """
    Build radius graph without torch-cluster dependency
    
    Args:
        pos: Node positions [N, 3]
        cutoff: Radius cutoff
        batch: Batch indices [N] (optional)
    
    Returns:
        edge_index: [2, E] edge indices
    """
    N = pos.size(0)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(pos, pos)  # [N, N]
    
    # Mask for edges within cutoff (excluding self-loops)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    
    # If batched, mask out cross-batch edges
    if batch is not None:
        batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)  # [N, N]
        mask = mask & batch_mask
    
    # Get edge indices
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
    
    return edge_index


class EGNNConv(nn.Module):
    """
    EGNN Convolutional Layer
    
    Message Passing Equations:
    1. m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)    # Edge message
    2. x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)  # Coordinate update
    3. m_i = Σ_j m_ij                            # Aggregate messages
    4. h_i' = φ_h(h_i, m_i)                      # Node update
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        act: nn.Module = nn.SiLU(),
        update_coords: bool = False,  # For density, we don't update coords
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_coords = update_coords
        
        # φ_e: Edge Message MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),
            act,
            nn.Linear(hidden_channels, hidden_channels),
            act,
        )
        
        # φ_x: Coordinate Update MLP (optional)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                act,
                nn.Linear(hidden_channels, 1),
            )
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        
        # φ_h: Node Update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            act,
            nn.Linear(hidden_channels, out_channels),
        )
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            h: Node features [N, in_channels]
            pos: Node coordinates [N, 3]
            edge_index: Edge indices [2, E]
        
        Returns:
            h_out: Updated node features [N, out_channels]
            pos_out: Updated coordinates [N, 3]
        """
        row, col = edge_index
        
        # Relative position and distance
        rel_pos = pos[row] - pos[col]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        # Edge messages
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        
        # Coordinate update (optional)
        if self.update_coords:
            coord_weight = self.coord_mlp(edge_msg)
            coord_update = rel_pos * coord_weight
            coord_agg = scatter(coord_update, row, dim=0, dim_size=h.size(0), reduce='add')
            pos_out = pos + coord_agg
        else:
            pos_out = pos
        
        # Aggregate messages
        msg_agg = scatter(edge_msg, row, dim=0, dim_size=h.size(0), reduce='add')
        
        # Update node features
        node_input = torch.cat([h, msg_agg], dim=-1)
        h_out = self.node_mlp(node_input)
        
        return h_out, pos_out


class EGNNEncoder(nn.Module):
    """
    EGNN Encoder for molecular representation
    
    Takes molecular structure and outputs per-atom features
    for downstream cross-attention decoder.
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        max_z: int = 100,
        cutoff: float = 5.0,
    ):
        """
        Args:
            hidden_channels: Hidden dimension
            num_layers: Number of EGNN layers
            max_z: Maximum atomic number
            cutoff: Radius cutoff for edge construction (Å)
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cutoff = cutoff
        
        # Atom type embedding
        self.embedding = nn.Embedding(max_z, hidden_channels)
        
        # Stack of EGNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGNNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    update_coords=False,  # Don't update coords for density
                )
            )
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
    
    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: Atomic numbers [N]
            pos: Atomic positions [N, 3]
            batch: Batch indices [N] (optional)
            edge_index: Pre-computed edges [2, E] (optional)
        
        Returns:
            h: Per-atom features [N, hidden_channels]
        """
        # Build edge index if not provided
        if edge_index is None:
            edge_index = build_radius_graph(
                pos, cutoff=self.cutoff, batch=batch
            )
        
        # Initial embedding
        h = self.embedding(z)
        
        # Message passing
        for conv, ln in zip(self.convs, self.layer_norms):
            h_new, pos = conv(h, pos, edge_index)
            h = ln(h + h_new)  # Residual + LayerNorm
        
        return h


if __name__ == "__main__":
    # Test encoder
    print("Testing EGNN Encoder...")
    
    # Mock data
    z = torch.tensor([6, 1, 1, 1, 1])  # CH4
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    
    encoder = EGNNEncoder(hidden_channels=64, num_layers=3)
    h = encoder(z, pos)
    
    print(f"Input: z={z.shape}, pos={pos.shape}")
    print(f"Output: h={h.shape}")
    print("✓ Encoder test passed!")
