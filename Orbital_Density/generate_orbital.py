"""
HOMO/LUMO Orbital Density Generation with PySCF

Generates orbital density grids for QM9 molecules.

Usage:
    python generate_orbital.py --n_molecules 1000
    python generate_orbital.py --test  # Test with 10 molecules
    python generate_orbital.py --start 0 --end 1000  # Index range
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Suppress PySCF warnings
warnings.filterwarnings('ignore')

try:
    from pyscf import gto, dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("PySCF not installed. Run 'pip install pyscf' for orbital generation.")


# Atomic symbols for QM9
ATOMIC_SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}


def parse_qm9_xyz(xyz_path: str) -> tuple:
    """
    Parse QM9 xyz file
    
    Returns:
        atoms: List of (symbol, x, y, z)
        properties: Dict of properties from comment line
    """
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    # QM9 comment line contains properties
    comment = lines[1].strip()
    
    atoms = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((symbol, x, y, z))
    
    return atoms, comment


def make_grid(mol, grid_size: int = 64, padding: float = 3.0):
    """
    Create 3D grid around molecule
    
    Args:
        mol: PySCF molecule object
        grid_size: Number of points per dimension
        padding: Extra space around molecule (Å)
    
    Returns:
        coords: Grid coordinates [N, 3]
        grid_info: Dict with origin, spacing, shape
    """
    # Get atom coordinates
    atom_coords = mol.atom_coords()  # Bohr
    BOHR_TO_ANG = 0.529177
    atom_coords_ang = atom_coords * BOHR_TO_ANG
    
    # Compute bounding box
    min_coords = atom_coords_ang.min(axis=0) - padding
    max_coords = atom_coords_ang.max(axis=0) + padding
    
    # Center the box
    center = (min_coords + max_coords) / 2
    box_size = (max_coords - min_coords).max()
    
    # Create grid
    origin = center - box_size / 2
    spacing = box_size / grid_size
    
    x = np.linspace(origin[0], origin[0] + box_size, grid_size)
    y = np.linspace(origin[1], origin[1] + box_size, grid_size)
    z = np.linspace(origin[2], origin[2] + box_size, grid_size)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    
    # Convert to Bohr for PySCF
    coords_bohr = coords / BOHR_TO_ANG
    
    grid_info = {
        'origin': origin,
        'spacing': spacing,
        'shape': np.array([grid_size, grid_size, grid_size]),
        'box_size': box_size,
    }
    
    return coords_bohr, grid_info


def compute_orbital_density(
    xyz_path: str,
    output_dir: str,
    grid_size: int = 64,
    basis: str = 'def2-svp',
    xc: str = 'pbe',
) -> dict:
    """
    Compute HOMO and LUMO orbital densities for a molecule
    
    Args:
        xyz_path: Path to XYZ file
        output_dir: Output directory
        grid_size: Grid resolution
        basis: Basis set
        xc: Exchange-correlation functional
    
    Returns:
        info: Dict with computation info
    """
    if not PYSCF_AVAILABLE:
        raise ImportError("PySCF not available")
    
    mol_id = Path(xyz_path).stem
    
    try:
        # Parse XYZ
        atoms, _ = parse_qm9_xyz(xyz_path)
        
        # Build PySCF molecule
        atom_str = '; '.join([f'{s} {x} {y} {z}' for s, x, y, z in atoms])
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom')
        
        # DFT calculation
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0
        energy = mf.kernel()
        
        if not mf.converged:
            return {'mol_id': mol_id, 'success': False, 'error': 'SCF not converged'}
        
        # Orbital indices
        n_occ = mol.nelectron // 2
        homo_idx = n_occ - 1
        lumo_idx = n_occ
        
        # Generate grid
        coords_bohr, grid_info = make_grid(mol, grid_size)
        
        # Evaluate atomic orbitals on grid
        ao_values = mol.eval_gto('GTOval', coords_bohr)  # [N, nao]
        
        # MO coefficients
        mo_coeff = mf.mo_coeff
        
        # HOMO density: |ψ_HOMO(r)|²
        homo_psi = ao_values @ mo_coeff[:, homo_idx]
        homo_density = homo_psi ** 2
        homo_density = homo_density.reshape(grid_size, grid_size, grid_size)
        
        # LUMO density: |ψ_LUMO(r)|²
        lumo_psi = ao_values @ mo_coeff[:, lumo_idx]
        lumo_density = lumo_psi ** 2
        lumo_density = lumo_density.reshape(grid_size, grid_size, grid_size)
        
        # Save results
        out_path = Path(output_dir) / mol_id
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save geometry (centered)
        atom_coords = mol.atom_coords() * 0.529177  # Bohr to Ang
        center = atom_coords.mean(axis=0)
        centered_coords = atom_coords - center
        
        with open(out_path / 'geometry.xyz', 'w') as f:
            f.write(f'{len(atoms)}\n')
            f.write(f'Generated from {mol_id}\n')
            for i, (s, _, _, _) in enumerate(atoms):
                x, y, z = centered_coords[i]
                f.write(f'{s} {x:.6f} {y:.6f} {z:.6f}\n')
        
        # Save densities
        np.save(out_path / 'homo_density.npy', homo_density.astype(np.float32))
        np.save(out_path / 'lumo_density.npy', lumo_density.astype(np.float32))
        
        # Save grid info
        np.savez(
            out_path / 'grid_info.npz',
            origin=grid_info['origin'] - center,  # Adjust for centering
            spacing=grid_info['spacing'],
            shape=grid_info['shape'],
        )
        
        # Save energies
        homo_energy = mf.mo_energy[homo_idx] * 27.2114  # Hartree to eV
        lumo_energy = mf.mo_energy[lumo_idx] * 27.2114
        
        np.savez(
            out_path / 'energies.npz',
            total_energy=energy,
            homo_energy=homo_energy,
            lumo_energy=lumo_energy,
            homo_idx=homo_idx,
            lumo_idx=lumo_idx,
        )
        
        return {
            'mol_id': mol_id,
            'success': True,
            'homo_energy': homo_energy,
            'lumo_energy': lumo_energy,
            'n_atoms': len(atoms),
        }
        
    except Exception as e:
        return {'mol_id': mol_id, 'success': False, 'error': str(e)}


def process_qm9_molecules(
    qm9_dir: str,
    output_dir: str,
    start_idx: int = 0,
    end_idx: int = None,
    grid_size: int = 64,
    n_workers: int = 1,
):
    """
    Process multiple QM9 molecules
    
    Args:
        qm9_dir: Directory containing QM9 XYZ files
        output_dir: Output directory
        start_idx: Starting index
        end_idx: Ending index (exclusive)
        grid_size: Grid resolution
        n_workers: Number of parallel workers
    """
    qm9_path = Path(qm9_dir)
    xyz_files = sorted(qm9_path.glob('*.xyz'))
    
    if end_idx is None:
        end_idx = len(xyz_files)
    
    xyz_files = xyz_files[start_idx:end_idx]
    print(f"Processing {len(xyz_files)} molecules (index {start_idx} to {end_idx})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    if n_workers == 1:
        # Sequential processing
        for xyz_file in tqdm(xyz_files, desc="Generating orbital densities"):
            result = compute_orbital_density(
                str(xyz_file),
                output_dir,
                grid_size=grid_size,
            )
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    compute_orbital_density,
                    str(xyz_file),
                    output_dir,
                    grid_size,
                ): xyz_file
                for xyz_file in xyz_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)
    
    # Summary
    n_success = sum(1 for r in results if r['success'])
    n_failed = len(results) - n_success
    
    print(f"\n=== Generation Complete ===")
    print(f"Success: {n_success}")
    print(f"Failed:  {n_failed}")
    
    if n_failed > 0:
        print("\nFailed molecules:")
        for r in results:
            if not r['success']:
                print(f"  {r['mol_id']}: {r.get('error', 'Unknown error')}")
    
    # Save results summary
    import json
    with open(Path(output_dir) / 'generation_summary.json', 'w') as f:
        json.dump({
            'n_total': len(results),
            'n_success': n_success,
            'n_failed': n_failed,
            'grid_size': grid_size,
            'results': results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate HOMO/LUMO orbital densities')
    parser.add_argument('--qm9_dir', type=str, default='data/QM9/raw',
                        help='Directory containing QM9 XYZ files')
    parser.add_argument('--output_dir', type=str, default='data/orbital_density',
                        help='Output directory')
    parser.add_argument('--n_molecules', type=int, default=None,
                        help='Number of molecules to process')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index')
    parser.add_argument('--end', type=int, default=None,
                        help='Ending index')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid resolution')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--test', action='store_true',
                        help='Test with 10 molecules')
    args = parser.parse_args()
    
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF not installed!")
        print("Install with: pip install pyscf")
        return
    
    end_idx = args.end
    if args.n_molecules is not None:
        end_idx = args.start + args.n_molecules
    
    if args.test:
        end_idx = args.start + 10
        print("=== TEST MODE: Processing 10 molecules ===\n")
    
    process_qm9_molecules(
        qm9_dir=args.qm9_dir,
        output_dir=args.output_dir,
        start_idx=args.start,
        end_idx=end_idx,
        grid_size=args.grid_size,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
