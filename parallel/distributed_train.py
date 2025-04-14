import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from utils.data_utils import load_texture_dataset
from utils.visualization import plot_learning_curves
from models.texture_pointnet import TexturePointNet
from parallel.utils import init_distributed_mode, is_main_process, get_rank, get_world_size, save_on_master


def distributed_train_model(args):
    """
    Train the model using distributed data parallel training.

    Args:
        args: Command-line arguments

    Returns:
        dict: Training results and metrics
    """
    # Initialize distributed environment
    init_distributed_mode(args)
    world_size = get_world_size()
    rank = get_rank()

    # Create output directory if main process
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Distributed training with {world_size} processes")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(args.device)

    # Load dataset
    dataset_dict = load_texture_dataset(args.data_path)

    # Create distributed samplers and data loaders
    train_dataset = TensorDataset(dataset_dict['X_train'], dataset_dict['y_train'])
    val_dataset = TensorDataset(dataset_dict['X_val'], dataset_dict['y_val'])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    # For validation, we don't need to use distributed sampler
    # but we'll use it to maintain consistent batch sizes
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    # Initialize model
    model = TexturePointNet(
        num_points=dataset_dict['num_points'],
        in_channels=dataset_dict['in_channels'],
        output_dim=dataset_dict['output_dim'],
        dropout_rate=args.dropout_rate
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank] if args.device.startswith('cuda') else None)

    # Scale learning rate according to world_size (number of processes)
    lr = args.learning_rate * world_size

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience, factor=args.lr_factor
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                print(f"Loading checkpoint {args.resume}")

            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']

            if is_main_process():
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            if is_main_process():
                print(f"No checkpoint found at {args.resume}")

    # Training loop
    training_losses = []
    validation_losses = []
    counter = 0

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler shuffling
        train_sampler.set_epoch(epoch)

        # Training
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args)

        # Validation
        model.eval()
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # LR scheduling (based on global validation loss)
        val_loss_global = synchronize_loss(val_loss, val_loader.dataset, world_size)
        scheduler.step(val_loss_global)

        # Save metrics on main process
        if is_main_process():
            training_losses.append(train_loss)
            validation_losses.append(val_loss_global)

            print(f"Epoch {epoch + 1}/{args.epochs}, "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss_global:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoint
            is_best = val_loss_global < best_val_loss
            best_val_loss = min(val_loss_global, best_val_loss)

            save_on_master({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args,
            }, is_best, args.output_dir)

            # Early stopping
            if is_best:
                counter = 0
            else:
                counter += 1

            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Plotting learning curves (on main process only)
    if is_main_process():
        plot_learning_curves(
            training_losses,
            validation_losses,
            os.path.join(args.output_dir, 'learning_curves.png')
        )

    # Load best model for returning
    if is_main_process():
        checkpoint = torch.load(os.path.join(args.output_dir, 'model_best.pth'),
                                map_location=device)
        model.load_state_dict(checkpoint['model'])

        return {
            'model': model,
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'best_val_loss': best_val_loss
        }

    return None


def train_epoch(model, data_loader, optimizer, criterion, device, epoch, args):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to use for training
        epoch: Current epoch number
        args: Command line arguments

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_X.shape[0]
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        if batch_idx % args.log_interval == 0 and get_rank() == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(data_loader)}, "
                  f"Loss: {loss.item():.6f}")

    # Calculate average loss across all processes
    return synchronize_loss(total_loss, num_samples, get_world_size())


def validate_epoch(model, data_loader, criterion, device):
    """
    Validate the model for one epoch.

    Args:
        model: The model to validate
        data_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use for validation

    Returns:
        float: Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            batch_size = batch_X.shape[0]
            total_loss += loss.item() * batch_size
            num_samples += batch_size

    return total_loss / num_samples


def synchronize_loss(local_loss, num_samples, world_size):
    """
    Synchronize loss across all processes.

    Args:
        local_loss: Local loss value
        num_samples: Number of samples processed locally
        world_size: Number of processes

    Returns:
        float: Synchronized global loss
    """
    if world_size > 1:
        # Create tensor of [loss, count]
        tensor = torch.tensor([local_loss, num_samples], device='cuda')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor[0].item() / tensor[1].item()
    return local_loss / num_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Training for PointNet++')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU/process')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Base learning rate (will be scaled by number of processes)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor for learning rate scheduler')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval for training batches')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Distributed training parameters
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl',
                        help='Distributed backend')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='Distributed process rank')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')

    # Checkpoint parameters
    parser.add_argument('--resume', default='', type=str,
                        help='Path to latest checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    train_results = distributed_train_model(args)