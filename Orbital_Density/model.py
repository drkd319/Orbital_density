"""
Electron Density Prediction Model

Combines:
1. EGNN Encoder: Molecular structure -> Per-atom features
2. CrossAttention Decoder: Query positions + Atom features -> Density

Supports multiple orbital types (HOMO, LUMO, total) with separate heads.
"""

import torch
import torch.nn as nn
from typing import Literal

from encoder import EGNNEncoder
from decoder import FourierPositionEncoder, CrossAttentionDecoder, DensityHead
from config import Config


class DensityModel(nn.Module):
    """
    Complete Model for Electron Density Prediction
    
    Architecture:
    1. EGNN Encoder: {Z, R} -> per-atom features h_i
    2. Fourier Encoder: query positions r -> high-freq features
    3. CrossAttention Decoder: (queries, atoms) -> spatial features
    4. Density Heads: features -> ρ(r) for each orbital type
    """
    
    def __init__(self, config: Config = None, **kwargs):
        """
        Args:
            config: Configuration object
            **kwargs: Override config values
        """
        super().__init__()
        
        if config is None:
            config = Config()
        
        self.config = config
        enc_cfg = config.encoder
        dec_cfg = config.decoder
        
        # =====================================================================
        # Encoder: Molecular Structure -> Atom Features
        # =====================================================================
        self.encoder = EGNNEncoder(
            hidden_channels=enc_cfg.hidden_channels,
            num_layers=enc_cfg.num_layers,
            max_z=enc_cfg.max_z,
        )
        
        # =====================================================================
        # Position Encoder: 3D coordinates -> Fourier features
        # =====================================================================
        self.pos_encoder = FourierPositionEncoder(
            num_frequencies=dec_cfg.fourier_levels,
            include_input=True,
        )
        
        # =====================================================================
        # Cross-Attention Decoder: Queries attend to atoms
        # =====================================================================
        self.decoder = CrossAttentionDecoder(
            query_dim=self.pos_encoder.output_dim,
            kv_dim=enc_cfg.hidden_channels,
            hidden_dim=dec_cfg.hidden_dim,
            num_heads=dec_cfg.attention_heads,
            num_layers=dec_cfg.num_decoder_layers,
            use_distance_bias=True,
        )
        
        # =====================================================================
        # Density Heads: Separate head for each orbital type
        # =====================================================================
        self.orbital_types = config.data.orbital_types
        self.heads = nn.ModuleDict()
        
        for orbital_type in self.orbital_types:
            self.heads[orbital_type] = DensityHead(
                input_dim=dec_cfg.hidden_dim,
                hidden_dim=dec_cfg.hidden_dim // 2,
                num_layers=2,
            )
    
    def encode(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode molecular structure
        
        Args:
            z: Atomic numbers [N_atoms]
            pos: Atomic positions [N_atoms, 3]
            batch: Batch indices [N_atoms]
            edge_index: Pre-computed edges [2, E]
        
        Returns:
            h: Per-atom features [N_atoms, hidden]
        """
        return self.encoder(z, pos, batch, edge_index)
    
    def decode(
        self,
        query_pos: torch.Tensor,
        atom_features: torch.Tensor,
        atom_pos: torch.Tensor,
        orbital_type: str = "total",
        atom_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decode density at query positions
        
        Args:
            query_pos: 3D query positions [B, N_query, 3] or [N_query, 3]
            atom_features: Per-atom features [B, N_atoms, hidden]
            atom_pos: Atom positions [B, N_atoms, 3]
            orbital_type: Which orbital head to use
            atom_mask: Padding mask [B, N_atoms]
        
        Returns:
            density: Predicted density [B, N_query] or [N_query]
        """
        # Fourier encode query positions
        query_enc = self.pos_encoder(query_pos)
        
        # Cross-attention (pass both encoded and raw positions)
        features = self.decoder(
            query_enc=query_enc,
            atom_features=atom_features,
            atom_pos=atom_pos,
            query_pos_raw=query_pos,  # For distance bias
            atom_mask=atom_mask,
        )
        
        # Orbital-specific head
        if orbital_type not in self.heads:
            raise ValueError(f"Unknown orbital type: {orbital_type}. "
                           f"Available: {list(self.heads.keys())}")
        
        density = self.heads[orbital_type](features)
        
        return density
    
    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        query_pos: torch.Tensor,
        orbital_type: str = "total",
        batch: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full forward pass: molecule + query points -> density
        
        For batched data, use forward_batch instead.
        
        Args:
            z: Atomic numbers [N_atoms]
            pos: Atomic positions [N_atoms, 3]
            query_pos: Query 3D positions [N_query, 3]
            orbital_type: Which orbital to predict
            batch: Batch indices (for multi-molecule)
            edge_index: Pre-computed edges
        
        Returns:
            density: Predicted density at query points [N_query]
        """
        # Encode molecule
        h = self.encode(z, pos, batch, edge_index)
        
        # Reshape for decoder (add batch dim)
        h = h.unsqueeze(0)  # [1, N_atoms, hidden]
        pos = pos.unsqueeze(0)  # [1, N_atoms, 3]
        query_pos = query_pos.unsqueeze(0)  # [1, N_query, 3]
        
        # Decode
        density = self.decode(query_pos, h, pos, orbital_type)
        
        return density.squeeze(0)  # [N_query]
    
    def forward_grid(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        grid_spec,
        orbital_type: str = "total",
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        Predict density on full 3D grid
        
        Processes grid in batches to avoid OOM.
        
        Args:
            z: Atomic numbers [N_atoms]
            pos: Atomic positions [N_atoms, 3]
            grid_spec: GridSpec object defining the grid
            orbital_type: Which orbital to predict
            batch_size: Query points per forward pass
        
        Returns:
            density: 3D density grid [nx, ny, nz]
        """
        device = z.device
        
        # Encode molecule once
        h = self.encode(z, pos)
        h = h.unsqueeze(0)
        pos = pos.unsqueeze(0)
        
        # Get all grid coordinates
        grid_coords = grid_spec.get_coordinates(device)  # [N, 3]
        N = grid_coords.shape[0]
        
        # Process in batches
        density_list = []
        for i in range(0, N, batch_size):
            query_batch = grid_coords[i:i+batch_size].unsqueeze(0)
            density_batch = self.decode(query_batch, h, pos, orbital_type)
            density_list.append(density_batch.squeeze(0))
        
        # Concatenate and reshape
        density = torch.cat(density_list, dim=0)
        density = density.reshape(grid_spec.shape)
        
        return density


class DensityModelForTraining(DensityModel):
    """
    Wrapper for training that handles PyG Data objects
    """
    
    def forward_train(
        self,
        data,
        query_pos: torch.Tensor,
        orbital_type: str = "total",
    ) -> torch.Tensor:
        """
        Forward pass for training with PyG Data
        
        Args:
            data: PyG Data object (z, pos, edge_index, batch)
            query_pos: Query positions [N_query, 3]
            orbital_type: Which orbital to predict
        
        Returns:
            density: Predicted density [N_query]
        """
        return self.forward(
            z=data.z,
            pos=data.pos,
            query_pos=query_pos,
            orbital_type=orbital_type,
            batch=data.batch if hasattr(data, 'batch') else None,
            edge_index=data.edge_index if hasattr(data, 'edge_index') else None,
        )


if __name__ == "__main__":
    print("Testing DensityModel...")
    
    # Create model with default config
    config = Config()
    model = DensityModel(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Mock data
    z = torch.tensor([6, 1, 1, 1, 1])  # CH4
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    query_pos = torch.randn(100, 3)  # 100 random query points
    
    # Test forward pass
    density = model(z, pos, query_pos, orbital_type="total")
    print(f"\nInput: z={z.shape}, pos={pos.shape}, query={query_pos.shape}")
    print(f"Output: density={density.shape}")
    print(f"Density range: [{density.min():.4f}, {density.max():.4f}]")
    
    # Test grid prediction
    from utils import make_grid_spec
    grid_spec = make_grid_spec(pos, grid_size=32, spacing=0.2)
    
    density_grid = model.forward_grid(z, pos, grid_spec, orbital_type="total")
    print(f"\nGrid output: {density_grid.shape}")
    
    print("\n✓ DensityModel test passed!")
