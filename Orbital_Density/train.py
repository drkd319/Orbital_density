"""
Training Script for Electron Density Model

Supports:
- Phase A: Total electron density (Caltech QM9)
- Phase B: HOMO/LUMO orbital density

Usage:
    python train.py                        # Default config
    python train.py --config my.yaml       # Custom config
    python train.py --debug                # Quick test with mock data
    python train.py --orbital HOMO         # Train specific orbital
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from model import DensityModel
from dataset import (
    TotalDensityDataset,
    OrbitalDensityDataset,
    MockDensityDataset,
    collate_density_samples,
)
from config import Config
from utils import get_device

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Run 'pip install wandb' for experiment tracking.")


def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute training loss
    
    Uses MSE loss with optional normalization
    """
    return F.mse_loss(pred, target)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics"""
    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        
        # Relative error
        rel_error = (torch.abs(pred - target) / (target.abs() + 1e-8)).mean().item()
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            pred.flatten().unsqueeze(0),
            target.flatten().unsqueeze(0)
        ).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_error': rel_error,
        'cos_sim': cos_sim,
    }


def train_epoch(
    model: DensityModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    orbital_type: str = 'total',
    gradient_clip: float = 1.0,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_samples = 0
    
    for batch in loader:
        samples = batch['samples']
        batch_loss = 0
        
        for sample in samples:
            # Move to device
            z = sample.z.to(device)
            pos = sample.pos.to(device)
            query_pos = sample.query_pos.to(device)
            target = sample.density.to(device)
            otype = sample.orbital_type
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(z, pos, query_pos, orbital_type=otype)
            
            # Compute loss
            loss = compute_loss(pred, target)
            
            # Backward pass
            loss.backward()
            
            batch_loss += loss.item()
        
        # Clip gradients
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += batch_loss
        n_samples += len(samples)
    
    return total_loss / n_samples


def evaluate(
    model: DensityModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model"""
    model.eval()
    all_metrics = {'mse': 0, 'mae': 0, 'rel_error': 0, 'cos_sim': 0}
    n_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            samples = batch['samples']
            
            for sample in samples:
                z = sample.z.to(device)
                pos = sample.pos.to(device)
                query_pos = sample.query_pos.to(device)
                target = sample.density.to(device)
                otype = sample.orbital_type
                
                pred = model(z, pos, query_pos, orbital_type=otype)
                
                metrics = compute_metrics(pred, target)
                for k, v in metrics.items():
                    all_metrics[k] += v
                
                n_samples += 1
    
    # Average
    for k in all_metrics:
        all_metrics[k] /= n_samples
    
    return all_metrics


def run_training(
    config: Config,
    use_wandb: bool = True,
    debug: bool = False,
    orbital_type: str = 'total',
):
    """Main training loop"""
    
    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="EGNN-Orbital-density",
            config=config.to_dict(),
            tags=[orbital_type],
        )
        print(">>> WandB initialized")
    
    print(f">>> Configuration:\n{config}\n")
    
    # Device
    device = get_device()
    
    # =========================================================================
    # Dataset
    # =========================================================================
    print(">>> Preparing data...")
    
    if debug:
        # Use mock dataset for testing
        dataset = MockDensityDataset(
            num_samples=100,
            n_sample_points=config.data.n_sample_points,
        )
    else:
        # Use real dataset
        if orbital_type == 'total':
            dataset = TotalDensityDataset(
                data_dir=config.data.data_dir,
                n_sample_points=config.data.n_sample_points,
            )
        else:
            dataset = OrbitalDensityDataset(
                data_dir=config.data.data_dir,
                orbital_type=orbital_type,
                n_sample_points=config.data.n_sample_points,
            )
    
    # Split dataset
    total_size = len(dataset)
    train_size = min(config.data.train_size, int(total_size * 0.8))
    val_size = min(config.data.val_size, int(total_size * 0.1))
    test_size = min(config.data.test_size, total_size - train_size - val_size)
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size + (total_size - train_size - val_size - test_size)],
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_density_samples,
        num_workers=0,  # Avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_density_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_density_samples,
    )
    
    # =========================================================================
    # Model
    # =========================================================================
    model = DensityModel(config)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n>>> Model Parameters: {n_params:,}")
    
    # =========================================================================
    # Optimizer & Scheduler
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    if config.training.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    elif config.training.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    else:
        scheduler = None
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\n>>> Start Training for {config.training.epochs} epochs!\n")
    
    for epoch in range(1, config.training.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            orbital_type=orbital_type,
            gradient_clip=config.training.gradient_clip,
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics['mse']
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        # Logging
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_cos_sim': val_metrics['cos_sim'],
                'learning_rate': current_lr,
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'val_metrics': val_metrics,
                'orbital_type': orbital_type,
            }, f'checkpoints/density_model_best_{orbital_type}.pt')
        
        # Print progress
        print(
            f'Epoch {epoch:3d}/{config.training.epochs} | '
            f'Train: {train_loss:.6f} | '
            f'Val MSE: {val_loss:.6f} | '
            f'Val MAE: {val_metrics["mae"]:.6f} | '
            f'CosSim: {val_metrics["cos_sim"]:.4f} | '
            f'LR: {current_lr:.2e} | '
            f'Time: {epoch_time:.1f}s'
            + (' *' if epoch == best_epoch else '')
        )
    
    print(f"\n>>> Training Complete!")
    print(f">>> Best model at epoch {best_epoch} with Val MSE: {best_val_loss:.6f}")
    
    # =========================================================================
    # Test Evaluation
    # =========================================================================
    print("\n>>> Evaluating on Test Set...")
    
    checkpoint = torch.load(f'checkpoints/density_model_best_{orbital_type}.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f">>> Test Results:")
    print(f"    MSE:        {test_metrics['mse']:.6f}")
    print(f"    MAE:        {test_metrics['mae']:.6f}")
    print(f"    Rel Error:  {test_metrics['rel_error']:.4f}")
    print(f"    Cos Sim:    {test_metrics['cos_sim']:.4f}")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_cos_sim': test_metrics['cos_sim'],
        })
        wandb.finish()
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Electron Density Model Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--debug', action='store_true',
                        help='Use mock dataset for testing')
    parser.add_argument('--orbital', type=str, default='total',
                        choices=['total', 'HOMO', 'LUMO'],
                        help='Which orbital to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = Config()
    
    # Override epochs if specified
    if args.epochs is not None:
        config.training.epochs = args.epochs
    
    # Quick debug settings
    if args.debug:
        config.training.epochs = min(config.training.epochs, 10)
        config.data.batch_size = 4
        config.data.n_sample_points = 512
    
    # Run training
    run_training(
        config=config,
        use_wandb=not args.no_wandb,
        debug=args.debug,
        orbital_type=args.orbital,
    )


if __name__ == "__main__":
    main()
