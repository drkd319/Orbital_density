"""
Quick Test: Generate HOMO/LUMO density for a single molecule

This script tests PySCF orbital density generation on one molecule.
"""

import numpy as np
from pathlib import Path
from pyscf import gto, dft


def generate_orbital_density_single(xyz_path: str, grid_size: int = 32):
    """Generate HOMO and LUMO density for a single molecule"""
    
    print(f"Processing: {xyz_path}")
    
    # Parse XYZ file
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    atoms = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(f'{symbol} {x} {y} {z}')
    
    atom_str = '; '.join(atoms)
    
    # Build molecule
    print("  Building molecule...")
    mol = gto.M(atom=atom_str, basis='sto-3g', unit='Angstrom')  # Use smaller basis for speed
    print(f"  Atoms: {mol.natm}, Electrons: {mol.nelectron}")
    
    # Run DFT
    print("  Running DFT calculation...")
    mf = dft.RKS(mol)
    mf.xc = 'lda'  # Use faster functional for testing
    mf.verbose = 0
    energy = mf.kernel()
    
    if not mf.converged:
        print("  WARNING: SCF not converged!")
        return None
    
    print(f"  Total energy: {energy:.6f} Hartree")
    
    # Orbital indices
    n_occ = mol.nelectron // 2
    homo_idx = n_occ - 1
    lumo_idx = n_occ
    
    homo_energy = mf.mo_energy[homo_idx] * 27.2114  # Hartree to eV
    lumo_energy = mf.mo_energy[lumo_idx] * 27.2114
    
    print(f"  HOMO energy: {homo_energy:.4f} eV")
    print(f"  LUMO energy: {lumo_energy:.4f} eV")
    print(f"  HOMO-LUMO gap: {lumo_energy - homo_energy:.4f} eV")
    
    # Create grid
    print(f"  Creating {grid_size}³ grid...")
    atom_coords = mol.atom_coords() * 0.529177  # Bohr to Angstrom
    center = atom_coords.mean(axis=0)
    box_size = max(atom_coords.max(axis=0) - atom_coords.min(axis=0)) + 6.0  # 3 Å padding
    
    spacing = box_size / grid_size
    origin = center - box_size / 2
    
    x = np.linspace(origin[0], origin[0] + box_size, grid_size)
    y = np.linspace(origin[1], origin[1] + box_size, grid_size)
    z = np.linspace(origin[2], origin[2] + box_size, grid_size)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    coords_bohr = coords / 0.529177  # Angstrom to Bohr
    
    # Evaluate orbitals on grid
    print("  Evaluating orbitals on grid...")
    ao_values = mol.eval_gto('GTOval', coords_bohr)
    
    homo_psi = ao_values @ mf.mo_coeff[:, homo_idx]
    lumo_psi = ao_values @ mf.mo_coeff[:, lumo_idx]
    
    homo_density = homo_psi ** 2
    lumo_density = lumo_psi ** 2
    
    homo_density = homo_density.reshape(grid_size, grid_size, grid_size)
    lumo_density = lumo_density.reshape(grid_size, grid_size, grid_size)
    
    print(f"  HOMO density: max={homo_density.max():.6f}, integral={homo_density.sum() * spacing**3:.4f}")
    print(f"  LUMO density: max={lumo_density.max():.6f}, integral={lumo_density.sum() * spacing**3:.4f}")
    
    return {
        'homo_density': homo_density,
        'lumo_density': lumo_density,
        'homo_energy': homo_energy,
        'lumo_energy': lumo_energy,
        'grid_info': {
            'origin': origin,
            'spacing': spacing,
            'shape': (grid_size, grid_size, grid_size),
        }
    }


if __name__ == "__main__":
    # Test with first molecule
    xyz_files = sorted(Path("data/caltech_qm9_density").glob("*/centered.xyz"))
    
    if not xyz_files:
        print("No xyz files found. Trying QM9 from PyTorch Geometric...")
        # Alternative: Load from PyTorch Geometric QM9
        from torch_geometric.datasets import QM9
        
        dataset = QM9(root='data/QM9')
        mol = dataset[0]
        
        # Convert to XYZ format
        atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        atoms = []
        for i in range(mol.z.size(0)):
            z = mol.z[i].item()
            x, y, z_coord = mol.pos[i].tolist()
            symbol = atomic_symbols.get(z, 'X')
            atoms.append(f'{symbol} {x} {y} {z_coord}')
        
        atom_str = '; '.join(atoms)
        print(f"Testing with QM9 molecule 0: {len(atoms)} atoms")
        
        # Generate directly
        mol_pyscf = gto.M(atom=atom_str, basis='sto-3g', unit='Angstrom')
        mf = dft.RKS(mol_pyscf)
        mf.xc = 'lda'
        mf.verbose = 0
        mf.kernel()
        print(f"  SCF converged: {mf.converged}")
        print(f"  Total energy: {mf.e_tot:.6f} Hartree")
    else:
        print(f"Found {len(xyz_files)} xyz files")
        result = generate_orbital_density_single(str(xyz_files[0]), grid_size=32)
        
        if result:
            print("\n✓ HOMO/LUMO generation successful!")
            print(f"  HOMO density shape: {result['homo_density'].shape}")
            print(f"  LUMO density shape: {result['lumo_density'].shape}")
