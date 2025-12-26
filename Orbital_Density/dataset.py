"""
Dataset classes for Electron Density Prediction

Supports:
1. TotalDensityDataset: Caltech QM9 total electron density
2. OrbitalDensityDataset: HOMO/LUMO orbital densities

Data format:
- Each molecule folder contains:
  - centered.xyz or geometry file
  - rho.npy or density grid
  - grid info (origin, spacing, shape)
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Literal
from dataclasses import dataclass
from torch.utils.data import Dataset

from utils import GridSpec, parse_xyz, sample_grid_points


@dataclass
class DensitySample:
    """Single training sample"""
    z: torch.Tensor           # Atomic numbers [N_atoms]
    pos: torch.Tensor         # Atomic positions [N_atoms, 3]
    query_pos: torch.Tensor   # Query positions [N_query, 3]
    density: torch.Tensor     # Density values [N_query]
    grid_spec: GridSpec       # Full grid specification
    orbital_type: str         # 'HOMO', 'LUMO', or 'total'
    mol_id: str               # Molecule identifier


class TotalDensityDataset(Dataset):
    """
    Dataset for Caltech QM9 Total Electron Density
    
    Expected directory structure:
    data_dir/
    ├── 000001/
    │   ├── centered.xyz
    │   ├── rho_22.npy
    │   ├── grid_sizes_22.dat
    │   └── box.dat
    ├── 000002/
    │   └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        n_sample_points: int = 4096,
        importance_sampling: bool = True,
        transform = None,
        max_molecules: int = None,
    ):
        """
        Args:
            data_dir: Path to data directory
            n_sample_points: Points to sample per molecule
            importance_sampling: Sample more from high-density regions
            transform: Optional transform
            max_molecules: Limit dataset size (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.n_sample_points = n_sample_points
        self.importance_sampling = importance_sampling
        self.transform = transform
        
        # Find all molecule directories
        self.mol_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if max_molecules is not None:
            self.mol_dirs = self.mol_dirs[:max_molecules]
        
        print(f"Found {len(self.mol_dirs)} molecules in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.mol_dirs)
    
    def __getitem__(self, idx: int) -> DensitySample:
        mol_dir = self.mol_dirs[idx]
        mol_id = mol_dir.name
        
        # Load geometry
        xyz_path = mol_dir / 'centered.xyz'
        if xyz_path.exists():
            with open(xyz_path, 'r') as f:
                xyz_content = f.read()
            z, pos = parse_xyz(xyz_content)
        else:
            raise FileNotFoundError(f"No geometry file in {mol_dir}")
        
        # Load density grid
        density_path = mol_dir / 'rho_22.npy'
        if not density_path.exists():
            density_path = mol_dir / 'rho.npy'
        
        density_full = torch.from_numpy(np.load(density_path)).float()
        
        # Load grid info
        grid_sizes_path = mol_dir / 'grid_sizes_22.dat'
        box_path = mol_dir / 'box.dat'
        
        if grid_sizes_path.exists():
            grid_shape = tuple(map(int, open(grid_sizes_path).read().split()))
        else:
            grid_shape = tuple(density_full.shape)
        
        if box_path.exists():
            box_size = float(open(box_path).read().strip())
        else:
            box_size = 12.8  # Default: 64 * 0.2 Å
        
        spacing = box_size / grid_shape[0]
        
        # Center grid on molecule
        center = pos.mean(dim=0)
        origin = center - box_size / 2
        
        grid_spec = GridSpec(
            origin=origin,
            spacing=spacing,
            shape=grid_shape,
        )
        
        # Sample query points
        query_pos, density_values = sample_grid_points(
            grid_spec,
            density_full,
            self.n_sample_points,
            self.importance_sampling,
        )
        
        sample = DensitySample(
            z=z,
            pos=pos,
            query_pos=query_pos,
            density=density_values,
            grid_spec=grid_spec,
            orbital_type='total',
            mol_id=mol_id,
        )
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class OrbitalDensityDataset(Dataset):
    """
    Dataset for HOMO/LUMO orbital densities
    
    Expected directory structure:
    data_dir/
    ├── 000001/
    │   ├── geometry.xyz
    │   ├── homo_density.npy
    │   ├── lumo_density.npy
    │   └── grid_info.npz
    ├── 000002/
    │   └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        orbital_type: Literal['HOMO', 'LUMO', 'both'] = 'both',
        n_sample_points: int = 4096,
        importance_sampling: bool = True,
        transform = None,
        max_molecules: int = None,
    ):
        """
        Args:
            data_dir: Path to data directory
            orbital_type: Which orbital(s) to load
            n_sample_points: Points to sample per molecule
            importance_sampling: Sample more from high-density regions
            transform: Optional transform
            max_molecules: Limit dataset size
        """
        self.data_dir = Path(data_dir)
        self.orbital_type = orbital_type
        self.n_sample_points = n_sample_points
        self.importance_sampling = importance_sampling
        self.transform = transform
        
        # Find all molecule directories
        self.mol_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if max_molecules is not None:
            self.mol_dirs = self.mol_dirs[:max_molecules]
        
        # Build sample list
        self.samples = []
        for mol_dir in self.mol_dirs:
            mol_id = mol_dir.name
            if orbital_type == 'both':
                self.samples.append((mol_dir, 'HOMO', mol_id))
                self.samples.append((mol_dir, 'LUMO', mol_id))
            else:
                self.samples.append((mol_dir, orbital_type, mol_id))
        
        print(f"Found {len(self.mol_dirs)} molecules, {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DensitySample:
        mol_dir, orbital, mol_id = self.samples[idx]
        
        # Load geometry
        xyz_path = mol_dir / 'geometry.xyz'
        if not xyz_path.exists():
            xyz_path = mol_dir / 'centered.xyz'
        
        with open(xyz_path, 'r') as f:
            xyz_content = f.read()
        z, pos = parse_xyz(xyz_content)
        
        # Load density
        density_file = f"{orbital.lower()}_density.npy"
        density_path = mol_dir / density_file
        density_full = torch.from_numpy(np.load(density_path)).float()
        
        # Load grid info
        grid_info_path = mol_dir / 'grid_info.npz'
        if grid_info_path.exists():
            grid_info = np.load(grid_info_path)
            origin = torch.from_numpy(grid_info['origin']).float()
            spacing = float(grid_info['spacing'])
            grid_shape = tuple(grid_info['shape'])
        else:
            # Default grid specification
            grid_shape = tuple(density_full.shape)
            spacing = 0.2
            box_size = grid_shape[0] * spacing
            center = pos.mean(dim=0)
            origin = center - box_size / 2
        
        grid_spec = GridSpec(
            origin=origin,
            spacing=spacing,
            shape=grid_shape,
        )
        
        # Sample query points
        query_pos, density_values = sample_grid_points(
            grid_spec,
            density_full,
            self.n_sample_points,
            self.importance_sampling,
        )
        
        sample = DensitySample(
            z=z,
            pos=pos,
            query_pos=query_pos,
            density=density_values,
            grid_spec=grid_spec,
            orbital_type=orbital,
            mol_id=mol_id,
        )
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def collate_density_samples(samples: List[DensitySample]) -> dict:
    """
    Collate function for DataLoader
    
    Since molecules have different numbers of atoms,
    we process each molecule separately in training.
    """
    return {
        'samples': samples,
        'batch_size': len(samples),
    }


class MockDensityDataset(Dataset):
    """
    Mock dataset for testing (generates random data)
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        n_atoms_range: Tuple[int, int] = (5, 20),
        grid_size: int = 32,
        n_sample_points: int = 1024,
    ):
        self.num_samples = num_samples
        self.n_atoms_range = n_atoms_range
        self.grid_size = grid_size
        self.n_sample_points = n_sample_points
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> DensitySample:
        # Random molecule
        n_atoms = torch.randint(*self.n_atoms_range, (1,)).item()
        z = torch.randint(1, 10, (n_atoms,))  # H to F
        pos = torch.randn(n_atoms, 3) * 2  # Random positions
        
        # Center molecule
        pos = pos - pos.mean(dim=0)
        
        # Grid specification
        grid_spec = GridSpec(
            origin=torch.tensor([-3.2, -3.2, -3.2]),
            spacing=0.2,
            shape=(self.grid_size, self.grid_size, self.grid_size),
        )
        
        # Random query points and density
        query_pos = torch.randn(self.n_sample_points, 3) * 3
        density = torch.rand(self.n_sample_points) * 0.1
        
        return DensitySample(
            z=z,
            pos=pos,
            query_pos=query_pos,
            density=density,
            grid_spec=grid_spec,
            orbital_type='total',
            mol_id=f'mock_{idx:06d}',
        )


if __name__ == "__main__":
    print("Testing Dataset Classes...")
    
    # Test mock dataset
    print("\n1. MockDensityDataset")
    mock_dataset = MockDensityDataset(num_samples=10)
    sample = mock_dataset[0]
    print(f"   z: {sample.z.shape}")
    print(f"   pos: {sample.pos.shape}")
    print(f"   query_pos: {sample.query_pos.shape}")
    print(f"   density: {sample.density.shape}")
    print(f"   orbital_type: {sample.orbital_type}")
    
    # Test DataLoader with collate
    from torch.utils.data import DataLoader
    
    print("\n2. DataLoader with collate")
    loader = DataLoader(
        mock_dataset,
        batch_size=4,
        collate_fn=collate_density_samples,
    )
    batch = next(iter(loader))
    print(f"   batch_size: {batch['batch_size']}")
    print(f"   samples: list of {len(batch['samples'])} DensitySample objects")
    
    print("\n✓ Dataset tests passed!")
