"""
Training Script for HOMO/LUMO Orbital Density Model

Uses QM9OrbitalDataset with on-the-fly PySCF computation.

Usage:
    python train_orbital.py                    # Default: 100 molecules
    python train_orbital.py --n_molecules 1000 # More molecules
    python train_orbital.py --epochs 50        # More epochs
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import DensityModel
from qm9_orbital_dataset import QM9OrbitalDataset, collate_orbital_samples
from config import Config
from utils import get_device

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_epoch(model, loader, optimizer, device, gradient_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_samples = 0
    
    for batch in loader:
        if batch is None:
            continue
            
        samples = batch['samples']
        batch_loss = 0
        
        for sample in samples:
            z = sample.z.to(device)
            pos = sample.pos.to(device)
            query_pos = sample.query_pos.to(device)
            target = sample.density.to(device)
            orbital = sample.orbital_type
            
            optimizer.zero_grad()
            pred = model(z, pos, query_pos, orbital_type=orbital)
            loss = F.mse_loss(pred, target)
            loss.backward()
            batch_loss += loss.item()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += batch_loss
        n_samples += len(samples)
    
    return total_loss / max(n_samples, 1)


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    metrics = {'mse': 0, 'mae': 0, 'cos_sim': 0}
    n_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
                
            samples = batch['samples']
            
            for sample in samples:
                z = sample.z.to(device)
                pos = sample.pos.to(device)
                query_pos = sample.query_pos.to(device)
                target = sample.density.to(device)
                orbital = sample.orbital_type
                
                pred = model(z, pos, query_pos, orbital_type=orbital)
                
                metrics['mse'] += F.mse_loss(pred, target).item()
                metrics['mae'] += F.l1_loss(pred, target).item()
                metrics['cos_sim'] += F.cosine_similarity(
                    pred.unsqueeze(0), target.unsqueeze(0)
                ).item()
                
                n_samples += 1
    
    for k in metrics:
        metrics[k] /= max(n_samples, 1)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train HOMO/LUMO Density Model')
    parser.add_argument('--n_molecules', type=int, default=100,
                        help='Number of molecules to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--grid_size', type=int, default=32,
                        help='Density grid resolution')
    parser.add_argument('--n_sample_points', type=int, default=1024,
                        help='Query points per sample')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB')
    args = parser.parse_args()
    
    device = get_device()
    
    # WandB
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="EGNN-Orbital-density",
            config=vars(args),
            tags=["HOMO-LUMO", "QM9"],
        )
    
    # Dataset
    print(f"\n>>> Preparing dataset ({args.n_molecules} molecules)...")
    dataset = QM9OrbitalDataset(
        max_molecules=args.n_molecules,
        orbital_type='both',
        grid_size=args.grid_size,
        n_sample_points=args.n_sample_points,
    )
    
    # Split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_orbital_samples, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_orbital_samples,
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    config = Config()
    config.decoder.fourier_levels = 6
    config.decoder.hidden_dim = 128
    config.encoder.hidden_channels = 64
    config.encoder.num_layers = 3
    
    model = DensityModel(config)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Model Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print(f"\n>>> Training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_cos_sim': val_metrics['cos_sim'],
                'lr': lr,
            })
        
        is_best = val_metrics['mse'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['mse']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/orbital_model_best.pt')
        
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val MSE: {val_metrics['mse']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f} | "
            f"CosSim: {val_metrics['cos_sim']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
            + (' *' if is_best else '')
        )
    
    print(f"\n>>> Training Complete! Best Val MSE: {best_val_loss:.6f}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
