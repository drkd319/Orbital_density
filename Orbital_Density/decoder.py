"""
Field Decoder for Electron Density Prediction

Components:
1. FourierPositionEncoder: High-frequency positional encoding
2. CrossAttentionDecoder: Query points attend to atom features
3. DensityHead: Final MLP to predict scalar density

Key idea: Query 3D positions attend to molecular representation
to predict electron density at those positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FourierPositionEncoder(nn.Module):
    """
    Fourier Feature Encoding for 3D positions
    
    γ(r) = [sin(2^0·π·r), cos(2^0·π·r), ..., sin(2^(L-1)·π·r), cos(2^(L-1)·π·r)]
    
    This allows the network to learn high-frequency spatial patterns,
    which is crucial for capturing the oscillatory nature of electron density.
    
    Reference: "Fourier Features Let Networks Learn High Frequency Functions"
    (Tancik et al., NeurIPS 2020)
    """
    
    def __init__(
        self,
        num_frequencies: int = 8,
        include_input: bool = True,
        log_scale: bool = True,
    ):
        """
        Args:
            num_frequencies: Number of frequency bands (L)
            include_input: Whether to concatenate original coordinates
            log_scale: Use log-spaced frequencies (2^0, 2^1, ..., 2^(L-1))
        """
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Frequency bands
        if log_scale:
            # 2^0, 2^1, ..., 2^(L-1)
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            # Linear spacing
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)
        
        # Register as buffer (not a parameter)
        self.register_buffer('freq_bands', freq_bands)
        
        # Output dimension
        # For each of 3 coords: 2 (sin+cos) * L frequencies
        self.output_dim = 3 * 2 * num_frequencies
        if include_input:
            self.output_dim += 3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D positions
        
        Args:
            x: 3D coordinates [*, 3]
        
        Returns:
            encoded: Fourier features [*, output_dim]
        """
        # x: [..., 3]
        # freq_bands: [L]
        
        # Scale coordinates by frequencies
        # [..., 3, 1] * [L] -> [..., 3, L]
        scaled = x.unsqueeze(-1) * self.freq_bands * math.pi
        
        # Apply sin and cos
        # [..., 3, L] -> [..., 3, 2L]
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        
        # Flatten last two dimensions
        # [..., 3, 2L] -> [..., 6L]
        encoded = encoded.flatten(-2)
        
        # Optionally include original coordinates
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        
        return encoded


class CrossAttentionDecoder(nn.Module):
    """
    Cross-Attention Decoder for Electron Density
    
    Query positions attend to atom features to aggregate
    molecular information relevant to each spatial location.
    
    Query: Fourier-encoded 3D positions
    Key/Value: Per-atom features from EGNN encoder
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_distance_bias: bool = True,
    ):
        """
        Args:
            query_dim: Dimension of query (Fourier-encoded positions)
            kv_dim: Dimension of key/value (atom features)
            hidden_dim: Internal hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer decoder layers
            dropout: Dropout rate
            use_distance_bias: Add distance-based attention bias
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_distance_bias = use_distance_bias
        
        # Project query and key/value to same dimension
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_dim)
        
        # Position encoding for atoms (optional, since we use distance bias)
        self.atom_pos_encoder = FourierPositionEncoder(num_frequencies=4)
        self.atom_pos_proj = nn.Linear(self.atom_pos_encoder.output_dim, hidden_dim)
        
        # Multi-head cross-attention layers
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.layer_norms1 = nn.ModuleList()
        self.layer_norms2 = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.layer_norms1.append(nn.LayerNorm(hidden_dim))
            self.layer_norms2.append(nn.LayerNorm(hidden_dim))
        
        # Distance bias projection (if used)
        if use_distance_bias:
            self.distance_proj = nn.Sequential(
                nn.Linear(1, num_heads),
                nn.Softplus(),
            )
    
    def forward(
        self,
        query_pos: torch.Tensor,
        atom_features: torch.Tensor,
        atom_pos: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query_pos: Query 3D positions [B, N_query, 3] or [N_query, 3]
            atom_features: Per-atom features [B, N_atoms, kv_dim] or [N_atoms, kv_dim]
            atom_pos: Atom positions [B, N_atoms, 3] or [N_atoms, 3]
            atom_mask: Optional mask for padding [B, N_atoms]
        
        Returns:
            output: Features at query positions [B, N_query, hidden_dim]
        """
        # Handle unbatched input
        if query_pos.dim() == 2:
            query_pos = query_pos.unsqueeze(0)
            atom_features = atom_features.unsqueeze(0)
            atom_pos = atom_pos.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N_query, _ = query_pos.shape
        _, N_atoms, _ = atom_features.shape
        
        # Encode query positions with Fourier features
        # This is handled externally - query_pos should already be Fourier-encoded
        # But we also project query positions
        query_pos_enc = self.query_proj(query_pos)  # [B, N_query, hidden]
        
        # Encode atom features + positions
        atom_pos_enc = self.atom_pos_encoder(atom_pos)
        atom_enc = self.kv_proj(atom_features) + self.atom_pos_proj(atom_pos_enc)
        
        # Compute distance-based attention bias
        attn_bias = None
        if self.use_distance_bias:
            # [B, N_query, N_atoms]
            dist = torch.cdist(query_pos, atom_pos)
            # Convert to attention bias (closer = higher attention)
            # [B, N_query, N_atoms, 1] -> [B, N_query, N_atoms, num_heads]
            dist_bias = self.distance_proj(-dist.unsqueeze(-1))
            # Reshape for attention: [B * num_heads, N_query, N_atoms]
            attn_bias = dist_bias.permute(0, 3, 1, 2).reshape(
                B * self.num_heads, N_query, N_atoms
            )
        
        # Create attention mask from atom_mask
        key_padding_mask = None
        if atom_mask is not None:
            key_padding_mask = ~atom_mask  # True = masked
        
        # Apply cross-attention layers
        x = query_pos_enc
        for attn, ffn, ln1, ln2 in zip(
            self.attention_layers, self.ffn_layers,
            self.layer_norms1, self.layer_norms2
        ):
            # Cross-attention with residual
            attn_out, _ = attn(
                query=x,
                key=atom_enc,
                value=atom_enc,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_bias,
            )
            x = ln1(x + attn_out)
            
            # FFN with residual
            x = ln2(x + ffn(x))
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x


class DensityHead(nn.Module):
    """
    Final MLP to predict scalar density from decoder features
    
    Uses Softplus activation to ensure ρ(r) > 0
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        
        # Final layer outputs scalar
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize last layer to small values
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [*, hidden_dim]
        
        Returns:
            density: Predicted density [*]
        """
        out = self.mlp(x)
        # Softplus ensures positive density
        return F.softplus(out).squeeze(-1)


if __name__ == "__main__":
    # Test all components
    print("Testing Decoder Components...")
    
    # Test Fourier encoder
    print("\n1. FourierPositionEncoder")
    pos_encoder = FourierPositionEncoder(num_frequencies=8)
    pos = torch.randn(100, 3)
    encoded = pos_encoder(pos)
    print(f"   Input: {pos.shape} -> Output: {encoded.shape}")
    print(f"   Output dim: {pos_encoder.output_dim}")
    
    # Test CrossAttentionDecoder
    print("\n2. CrossAttentionDecoder")
    decoder = CrossAttentionDecoder(
        query_dim=pos_encoder.output_dim,
        kv_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
    )
    
    query_pos = torch.randn(2, 100, 3)  # 2 molecules, 100 query points
    query_enc = pos_encoder(query_pos)
    atom_features = torch.randn(2, 10, 128)  # 10 atoms each
    atom_pos = torch.randn(2, 10, 3)
    
    output = decoder(query_enc, atom_features, atom_pos)
    print(f"   Query: {query_pos.shape}, Atoms: {atom_features.shape}")
    print(f"   Output: {output.shape}")
    
    # Test DensityHead
    print("\n3. DensityHead")
    head = DensityHead(input_dim=128, hidden_dim=64)
    density = head(output)
    print(f"   Input: {output.shape} -> Density: {density.shape}")
    print(f"   Density range: [{density.min():.4f}, {density.max():.4f}]")
    
    print("\n✓ All decoder tests passed!")
