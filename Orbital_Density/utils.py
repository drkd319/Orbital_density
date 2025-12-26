"""
Utility functions for Electron Density Model

Includes:
- Device detection (CUDA/MPS/CPU)
- Grid generation utilities
- Data loading helpers
"""

import torch
import platform
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Device Detection
# =============================================================================

def get_device() -> torch.device:
    """
    Automatically detect and return the best available device
    
    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name}")
        
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print(f"Using Apple Silicon MPS (Metal)")
        
    else:
        device = torch.device('cpu')
        cpu_info = platform.processor() or "Unknown CPU"
        print(f"Using CPU: {cpu_info}")
    
    return device


# =============================================================================
# Grid Utilities
# =============================================================================

@dataclass
class GridSpec:
    """Specification for 3D density grid"""
    origin: torch.Tensor  # [3] - grid origin (Å)
    spacing: float        # Å per voxel
    shape: Tuple[int, int, int]  # (nx, ny, nz)
    
    @property
    def size(self) -> torch.Tensor:
        """Physical size of the grid box (Å)"""
        return torch.tensor(self.shape) * self.spacing
    
    def get_coordinates(self, device: torch.device = None) -> torch.Tensor:
        """
        Generate 3D grid coordinates
        
        Returns:
            coords: [nx*ny*nz, 3] tensor of 3D coordinates
        """
        nx, ny, nz = self.shape
        
        # Create 1D coordinate arrays
        x = torch.linspace(0, (nx - 1) * self.spacing, nx) + self.origin[0]
        y = torch.linspace(0, (ny - 1) * self.spacing, ny) + self.origin[1]
        z = torch.linspace(0, (nz - 1) * self.spacing, nz) + self.origin[2]
        
        # Create 3D meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # Flatten and stack
        coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        
        if device is not None:
            coords = coords.to(device)
        
        return coords
    
    def to(self, device: torch.device) -> "GridSpec":
        """Move grid spec to device"""
        return GridSpec(
            origin=self.origin.to(device),
            spacing=self.spacing,
            shape=self.shape,
        )


def make_grid_spec(
    pos: torch.Tensor,
    grid_size: int = 64,
    spacing: float = 0.2,
    padding: float = 2.0,
) -> GridSpec:
    """
    Create grid specification centered on molecule
    
    Args:
        pos: Atomic positions [N, 3]
        grid_size: Number of voxels per dimension
        spacing: Voxel size in Å
        padding: Extra space around molecule (Å)
    
    Returns:
        GridSpec object
    """
    # Center of molecule
    center = pos.mean(dim=0)
    
    # Grid origin (bottom-left corner)
    box_size = grid_size * spacing
    origin = center - box_size / 2
    
    return GridSpec(
        origin=origin,
        spacing=spacing,
        shape=(grid_size, grid_size, grid_size),
    )


def sample_grid_points(
    grid_spec: GridSpec,
    density: torch.Tensor,
    n_points: int,
    importance_sampling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points from grid for training
    
    Args:
        grid_spec: Grid specification
        density: 3D density tensor [nx, ny, nz]
        n_points: Number of points to sample
        importance_sampling: If True, sample more from high-density regions
    
    Returns:
        coords: Sampled coordinates [n_points, 3]
        values: Density values at sampled points [n_points]
    """
    all_coords = grid_spec.get_coordinates()  # [N, 3]
    flat_density = density.flatten()          # [N]
    
    if importance_sampling and flat_density.sum() > 0:
        # Sample proportional to density (+ small epsilon for zero regions)
        probs = (flat_density + 1e-8) / (flat_density + 1e-8).sum()
        indices = torch.multinomial(probs, n_points, replacement=True)
    else:
        # Uniform random sampling
        indices = torch.randint(0, len(flat_density), (n_points,))
    
    return all_coords[indices], flat_density[indices]


# =============================================================================
# Data Parsing
# =============================================================================

def parse_xyz(xyz_content: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse XYZ format molecular structure
    
    Args:
        xyz_content: String content of XYZ file
    
    Returns:
        z: Atomic numbers [N]
        pos: Atomic positions [N, 3]
    """
    # Atomic number lookup
    ATOMIC_NUMBERS = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
        'Ar': 18, 'K': 19, 'Ca': 20,
    }
    
    lines = xyz_content.strip().split('\n')
    n_atoms = int(lines[0].strip())
    
    z_list = []
    pos_list = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        element = parts[0]
        coords = [float(x) for x in parts[1:4]]
        
        z_list.append(ATOMIC_NUMBERS.get(element, 0))
        pos_list.append(coords)
    
    z = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float32)
    
    return z, pos


def read_cube_file(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, GridSpec]:
    """
    Read Gaussian cube file
    
    Args:
        filepath: Path to .cube file
    
    Returns:
        z: Atomic numbers [N]
        pos: Atomic positions [N, 3]
        density: 3D density grid
        grid_spec: Grid specification
    """
    BOHR_TO_ANGSTROM = 0.529177

    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines (lines 0-1)
    # Line 2: N_atoms, origin
    parts = lines[2].split()
    n_atoms = int(parts[0])
    origin = torch.tensor([float(x) for x in parts[1:4]]) * BOHR_TO_ANGSTROM
    
    # Lines 3-5: Grid info (n_points, axis vector)
    grid_shape = []
    spacing_vec = []
    for i in range(3, 6):
        parts = lines[i].split()
        grid_shape.append(int(parts[0]))
        # Spacing is the magnitude of the axis vector
        vec = torch.tensor([float(x) for x in parts[1:4]]) * BOHR_TO_ANGSTROM
        spacing_vec.append(vec.norm().item())
    
    # Average spacing (assume cubic)
    spacing = sum(spacing_vec) / 3
    
    # Read atoms
    z_list = []
    pos_list = []
    for i in range(6, 6 + n_atoms):
        parts = lines[i].split()
        z_list.append(int(parts[0]))
        pos_list.append([float(x) * BOHR_TO_ANGSTROM for x in parts[2:5]])
    
    z = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float32)
    
    # Read density data
    density_values = []
    for line in lines[6 + n_atoms:]:
        density_values.extend([float(x) for x in line.split()])
    
    density = torch.tensor(density_values, dtype=torch.float32)
    density = density.reshape(grid_shape)
    
    grid_spec = GridSpec(
        origin=origin,
        spacing=spacing,
        shape=tuple(grid_shape),
    )
    
    return z, pos, density, grid_spec


if __name__ == "__main__":
    # Test grid utilities
    print("Testing Grid Utilities...")
    
    # Mock molecular positions
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.87, 0.0],
    ])
    
    grid_spec = make_grid_spec(pos, grid_size=32, spacing=0.2)
    print(f"Grid spec: shape={grid_spec.shape}, spacing={grid_spec.spacing}Å")
    
    coords = grid_spec.get_coordinates()
    print(f"Grid coordinates shape: {coords.shape}")
    
    # Test device detection
    device = get_device()
    print(f"Selected device: {device}")
