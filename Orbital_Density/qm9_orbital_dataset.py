"""
QM9 Orbital Density Dataset with On-the-fly PySCF Generation

Generates HOMO/LUMO orbital densities from QM9 molecules using PySCF.
Caches results to disk for efficiency.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Literal
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from pyscf import gto, dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("PySCF not installed. Run 'pip install pyscf'")

from utils import GridSpec


# Atomic symbols for QM9
ATOMIC_SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}


@dataclass
class OrbitalSample:
    """Single orbital density sample"""
    z: torch.Tensor           # Atomic numbers [N_atoms]
    pos: torch.Tensor         # Atomic positions [N_atoms, 3]
    query_pos: torch.Tensor   # Query positions [N_query, 3]
    density: torch.Tensor     # Density values [N_query]
    grid_spec: GridSpec       # Grid specification
    orbital_type: str         # 'HOMO' or 'LUMO'
    mol_id: int               # QM9 molecule index
    orbital_energy: float     # Orbital energy in eV


def compute_orbital_density(
    z: torch.Tensor,
    pos: torch.Tensor,
    orbital: Literal['HOMO', 'LUMO'] = 'HOMO',
    grid_size: int = 32,
    basis: str = 'sto-3g',
    xc: str = 'lda',
) -> Optional[dict]:
    """
    Compute orbital density for a molecule using PySCF
    
    Args:
        z: Atomic numbers [N]
        pos: Atomic positions [N, 3] in Angstrom
        orbital: 'HOMO' or 'LUMO'
        grid_size: Grid resolution
        basis: Basis set (sto-3g is fast, def2-svp is more accurate)
        xc: XC functional (lda is fast, pbe is better)
    
    Returns:
        Dictionary with density and metadata, or None if failed
    """
    if not PYSCF_AVAILABLE:
        raise ImportError("PySCF not available")
    
    # Build atom string
    atoms = []
    for i in range(len(z)):
        symbol = ATOMIC_SYMBOLS.get(z[i].item(), 'X')
        x, y, z_coord = pos[i].tolist()
        atoms.append(f'{symbol} {x} {y} {z_coord}')
    
    atom_str = '; '.join(atoms)
    
    try:
        # Build molecule
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)
        
        # Run DFT
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0
        mf.kernel()
        
        if not mf.converged:
            return None
        
        # Orbital indices
        n_occ = mol.nelectron // 2
        if orbital == 'HOMO':
            orb_idx = n_occ - 1
        else:  # LUMO
            orb_idx = n_occ
        
        orbital_energy = mf.mo_energy[orb_idx] * 27.2114  # Hartree to eV
        
        # Create grid
        atom_coords = mol.atom_coords() * 0.529177  # Bohr to Angstrom
        center = atom_coords.mean(axis=0)
        box_size = max(atom_coords.max(axis=0) - atom_coords.min(axis=0)) + 6.0
        
        spacing = box_size / grid_size
        origin = center - box_size / 2
        
        x = np.linspace(origin[0], origin[0] + box_size, grid_size)
        y = np.linspace(origin[1], origin[1] + box_size, grid_size)
        z = np.linspace(origin[2], origin[2] + box_size, grid_size)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
        coords_bohr = coords / 0.529177
        
        # Evaluate orbital
        ao_values = mol.eval_gto('GTOval', coords_bohr)
        psi = ao_values @ mf.mo_coeff[:, orb_idx]
        density = psi ** 2
        density = density.reshape(grid_size, grid_size, grid_size)
        
        grid_spec = GridSpec(
            origin=torch.tensor(origin, dtype=torch.float32),
            spacing=spacing,
            shape=(grid_size, grid_size, grid_size),
        )
        
        return {
            'density': torch.tensor(density, dtype=torch.float32),
            'grid_spec': grid_spec,
            'orbital_energy': orbital_energy,
            'total_energy': mf.e_tot,
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None


class QM9OrbitalDataset(Dataset):
    """
    QM9 Dataset with on-the-fly HOMO/LUMO generation
    
    Generates and caches orbital densities using PySCF.
    """
    
    def __init__(
        self,
        root: str = 'data/QM9',
        cache_dir: str = 'data/orbital_cache',
        orbital_type: Literal['HOMO', 'LUMO', 'both'] = 'both',
        n_sample_points: int = 2048,
        grid_size: int = 32,
        max_molecules: int = None,
        precompute: bool = False,
        basis: str = 'sto-3g',
        xc: str = 'lda',
    ):
        """
        Args:
            root: QM9 data root
            cache_dir: Where to cache computed densities
            orbital_type: Which orbital(s) to use
            n_sample_points: Query points to sample
            grid_size: Density grid resolution
            max_molecules: Limit number of molecules
            precompute: If True, precompute all densities
            basis: PySCF basis set
            xc: XC functional
        """
        from torch_geometric.datasets import QM9
        
        self.qm9 = QM9(root=root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.orbital_type = orbital_type
        self.n_sample_points = n_sample_points
        self.grid_size = grid_size
        self.basis = basis
        self.xc = xc
        
        # Limit molecules
        if max_molecules is not None:
            self.indices = list(range(min(max_molecules, len(self.qm9))))
        else:
            self.indices = list(range(len(self.qm9)))
        
        # Build sample list
        self.samples = []
        for idx in self.indices:
            if orbital_type == 'both':
                self.samples.append((idx, 'HOMO'))
                self.samples.append((idx, 'LUMO'))
            else:
                self.samples.append((idx, orbital_type))
        
        print(f"QM9OrbitalDataset: {len(self.indices)} molecules, {len(self.samples)} samples")
        
        # Precompute if requested
        if precompute:
            self._precompute_all()
    
    def _get_cache_path(self, mol_idx: int, orbital: str) -> Path:
        """Get cache file path for a molecule/orbital"""
        return self.cache_dir / f"mol_{mol_idx:06d}_{orbital.lower()}.npz"
    
    def _compute_and_cache(self, mol_idx: int, orbital: str) -> Optional[dict]:
        """Compute orbital density and cache to disk"""
        cache_path = self._get_cache_path(mol_idx, orbital)
        
        # Check cache
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return {
                'density': torch.tensor(data['density']),
                'grid_spec': GridSpec(
                    origin=torch.tensor(data['origin']),
                    spacing=float(data['spacing']),
                    shape=tuple(data['shape']),
                ),
                'orbital_energy': float(data['orbital_energy']),
            }
        
        # Get molecule
        mol = self.qm9[mol_idx]
        
        # Compute density
        result = compute_orbital_density(
            mol.z, mol.pos, orbital,
            grid_size=self.grid_size,
            basis=self.basis,
            xc=self.xc,
        )
        
        if result is None:
            return None
        
        # Cache to disk
        np.savez(
            cache_path,
            density=result['density'].numpy(),
            origin=result['grid_spec'].origin.numpy(),
            spacing=result['grid_spec'].spacing,
            shape=result['grid_spec'].shape,
            orbital_energy=result['orbital_energy'],
        )
        
        return result
    
    def _precompute_all(self):
        """Precompute all orbital densities"""
        print("Precomputing orbital densities...")
        failed = 0
        for mol_idx, orbital in tqdm(self.samples):
            result = self._compute_and_cache(mol_idx, orbital)
            if result is None:
                failed += 1
        print(f"Precomputed {len(self.samples) - failed}/{len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[OrbitalSample]:
        mol_idx, orbital = self.samples[idx]
        
        # Get molecule
        mol = self.qm9[mol_idx]
        
        # Get or compute density
        result = self._compute_and_cache(mol_idx, orbital)
        
        if result is None:
            # Return None for failed molecules
            return None
        
        # Sample query points
        grid_spec = result['grid_spec']
        all_coords = grid_spec.get_coordinates()
        flat_density = result['density'].flatten()
        
        # Importance sampling
        probs = (flat_density + 1e-8) / (flat_density + 1e-8).sum()
        indices = torch.multinomial(probs, self.n_sample_points, replacement=True)
        
        query_pos = all_coords[indices]
        density_values = flat_density[indices]
        
        return OrbitalSample(
            z=mol.z,
            pos=mol.pos,
            query_pos=query_pos,
            density=density_values,
            grid_spec=grid_spec,
            orbital_type=orbital,
            mol_id=mol_idx,
            orbital_energy=result['orbital_energy'],
        )


def collate_orbital_samples(samples):
    """Collate function that filters out None samples"""
    samples = [s for s in samples if s is not None]
    if len(samples) == 0:
        return None
    return {'samples': samples, 'batch_size': len(samples)}


if __name__ == "__main__":
    print("Testing QM9OrbitalDataset...")
    
    # Test with small subset
    dataset = QM9OrbitalDataset(
        max_molecules=10,
        orbital_type='both',
        grid_size=32,
        n_sample_points=512,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    if sample is not None:
        print(f"\nSample 0:")
        print(f"  z: {sample.z}")
        print(f"  pos: {sample.pos.shape}")
        print(f"  query_pos: {sample.query_pos.shape}")
        print(f"  density: {sample.density.shape}")
        print(f"  orbital_type: {sample.orbital_type}")
        print(f"  orbital_energy: {sample.orbital_energy:.4f} eV")
        print("\n✓ Dataset test passed!")
    else:
        print("Sample failed")
