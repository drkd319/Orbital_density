"""
Configuration dataclasses for Electron Density Model
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_dir: str = "data/orbital_density"
    grid_size: int = 64
    spacing: float = 0.2  # Å per voxel
    n_sample_points: int = 4096  # Points sampled per molecule during training
    train_size: int = 8000
    val_size: int = 1000
    test_size: int = 1000
    batch_size: int = 8
    orbital_types: list = field(default_factory=lambda: ["HOMO", "LUMO", "total"])


@dataclass
class EncoderConfig:
    """EGNN Encoder configuration"""
    hidden_channels: int = 128
    num_layers: int = 4
    max_z: int = 100


@dataclass
class DecoderConfig:
    """Field Decoder configuration"""
    fourier_levels: int = 8  # L in Fourier encoding
    attention_heads: int = 4
    attention_dim: int = 128
    hidden_dim: int = 256
    num_decoder_layers: int = 2


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    gradient_clip: float = 1.0


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file"""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**raw.get('data', {})),
            encoder=EncoderConfig(**raw.get('encoder', {})),
            decoder=DecoderConfig(**raw.get('decoder', {})),
            training=TrainingConfig(**raw.get('training', {})),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "data": {
                "grid_size": self.data.grid_size,
                "spacing": self.data.spacing,
                "n_sample_points": self.data.n_sample_points,
                "batch_size": self.data.batch_size,
            },
            "encoder": {
                "hidden_channels": self.encoder.hidden_channels,
                "num_layers": self.encoder.num_layers,
            },
            "decoder": {
                "fourier_levels": self.decoder.fourier_levels,
                "attention_heads": self.decoder.attention_heads,
                "attention_dim": self.decoder.attention_dim,
            },
            "training": {
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  data: grid={self.data.grid_size}³, spacing={self.data.spacing}Å, "
            f"sample_points={self.data.n_sample_points}\n"
            f"  encoder: hidden={self.encoder.hidden_channels}, layers={self.encoder.num_layers}\n"
            f"  decoder: fourier_L={self.decoder.fourier_levels}, heads={self.decoder.attention_heads}\n"
            f"  training: epochs={self.training.epochs}, lr={self.training.learning_rate}\n"
            f")"
        )
