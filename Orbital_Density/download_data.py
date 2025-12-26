"""
Download Caltech QM9 DFT Electron Density Dataset

Dataset: QM9 DFT Electron Density (Caltech)
Source: https://data.caltech.edu/records/7vr2f-0r732

Each molecule folder contains:
- rho_22.npy: DFT density in atomic units
- grid_sizes_22.dat: Shape of density grid
- energy_22.dat: DFT ground state energy
- box.dat: Box size in angstroms
- centered.xyz: Molecular geometry

Usage:
    python download_data.py                    # Download first partition (1000 molecules)
    python download_data.py --parts 1 2 3      # Download specific partitions
    python download_data.py --all              # Download all (WARNING: ~200GB)
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


BASE_URL = "https://sdsc.osn.xsede.org/ini210004tommorrell/7vr2f-0r732"
DATA_DIR = Path("data/caltech_qm9_density")


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Download file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def extract_tarfile(tar_path: Path, extract_dir: Path):
    """Extract tar.gz file"""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def download_partition(part_num: int, extract: bool = True, keep_tar: bool = False):
    """Download and optionally extract a single partition"""
    part_name = f"part{part_num:03d}.tar.gz"
    url = f"{BASE_URL}/{part_name}"
    tar_path = DATA_DIR / part_name
    
    print(f"\n=== Downloading {part_name} ===")
    print(f"URL: {url}")
    
    if tar_path.exists():
        print(f"{part_name} already exists, skipping download")
    else:
        download_file(url, tar_path)
    
    if extract:
        extract_tarfile(tar_path, DATA_DIR)
        
        if not keep_tar:
            tar_path.unlink()
            print(f"Removed {part_name}")


def list_partitions():
    """List available partitions with sizes"""
    sizes = {
        1: 5.7, 2: 7.3, 3: 7.4, 4: 6.6, 5: 6.8,
        6: 7.4, 7: 8.8, 8: 8.4, 9: 8.2, 10: 8.7,
        11: 8.4, 12: 8.3, 13: 8.0, 14: 8.7, 15: 8.5,
        16: 8.6, 17: 7.8, 18: 8.1, 19: 7.8, 20: 7.8,
        21: 7.3, 22: 5.7, 23: 9.3, 24: 7.5, 25: 7.5,
        26: 6.9, 27: 8.4, 28: 6.9, 29: 7.2, 30: 7.4,
        31: 8.8, 32: 9.2, 33: 8.9,
    }
    
    print("Available partitions:")
    print("-" * 40)
    total = 0
    for i, size in sizes.items():
        print(f"  part{i:03d}.tar.gz: {size:.1f} GB (~1000 molecules)")
        total += size
    print("-" * 40)
    print(f"  Total: {total:.1f} GB (~133K molecules)")
    
    return sizes


def main():
    parser = argparse.ArgumentParser(description='Download Caltech QM9 Density Dataset')
    parser.add_argument('--parts', nargs='+', type=int, default=[1],
                        help='Partition numbers to download (default: 1)')
    parser.add_argument('--all', action='store_true',
                        help='Download all partitions (WARNING: ~200GB)')
    parser.add_argument('--list', action='store_true',
                        help='List available partitions')
    parser.add_argument('--no-extract', action='store_true',
                        help='Do not extract tar files')
    parser.add_argument('--keep-tar', action='store_true',
                        help='Keep tar files after extraction')
    args = parser.parse_args()
    
    if args.list:
        list_partitions()
        return
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        parts = list(range(1, 34))
        print("WARNING: Downloading ALL partitions (~200GB)")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            return
    else:
        parts = args.parts
    
    print(f"Will download {len(parts)} partition(s)")
    
    for part in parts:
        if part < 1 or part > 33:
            print(f"Invalid partition number: {part} (valid: 1-33)")
            continue
        download_partition(part, extract=not args.no_extract, keep_tar=args.keep_tar)
    
    # Count downloaded molecules
    mol_dirs = list(DATA_DIR.glob("*/centered.xyz"))
    print(f"\n=== Download Complete ===")
    print(f"Total molecules: {len(mol_dirs)}")
    print(f"Data directory: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
