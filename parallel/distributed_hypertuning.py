import os
import argparse
import numpy as np
import torch
import json
from mpi4py import MPI
from sklearn.model_selection import ParameterGrid

from utils.data_utils import load_texture_dataset, create_data_loaders
from models.texture_pointnet import TexturePointNet
from parallel.utils import init_distributed_mode, get_rank, get_world_size


def distributed_hyperparameter_tuning(args):
    """
    Perform distributed hyperparameter tuning using MPI to distribute
    hyperparameter combinations across multiple nodes.

    Args:
        args: Command-line arguments

    Returns:
        dict: Results of hyperparameter tuning including best parameters
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Create output directory if it doesn't exist (only on rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Distributed hyperparameter tuning with {world_size} processes")

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0005],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.3, 0.4, 0.5],
        'sa1_npoint': [64, 128],
        'sa2_npoint': [32, 64]
    }

    # Convert to list of parameter combinations
    grid = list(ParameterGrid(param_grid))

    # Limit the number of combinations if needed
    if args.max_combinations > 0:
        grid = grid[:args.max_combinations]

    # Distribute hyperparameter combinations among processes
    local_combinations = []
    for i in range(rank, len(grid), world_size):
        local_combinations.append(grid[i])

    if rank == 0:
        print(f"Total combinations: {len(grid)}")
        print(f"Each process will evaluate ~{len(grid) // world_size} combinations")

    # Load dataset (each process loads its own copy)
    dataset_dict = load_texture_dataset(args.data_path)

    # Set device - use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")

    # Store results
    local_results = []

    # Process local hyperparameter combinations
    for i, params in enumerate(local_combinations):
        print(f"Rank {rank}, Combination {i + 1}/{len(local_combinations)}")
        print(f"Parameters: {params}")

        # Create data loaders with current batch size
        data_loaders = create_data_loaders(dataset_dict, params['batch_size'])

        # Initialize model with current hyperparameters
        model = TexturePointNet(
            num_points=dataset_dict['num_points'],
            in_channels=dataset_dict['in_channels'],
            output_dim=dataset_dict['output_dim'],
            dropout_rate=params['dropout_rate']
        ).to(device)

        # Train and evaluate model
        val_loss = train_with_params(
            model, params, data_loaders['train_loader'],
            data_loaders['val_loader'], device, epochs=args.tuning_epochs
        )

        # Store results
        params_copy = params.copy()
        params_copy['val_loss'] = val_loss
        local_results.append(params_copy)

        # Log results locally
        log_file = os.path.join(args.output_dir, f'hyperparameter_tuning_rank{rank}.log')
        with open(log_file, 'a') as f:
            f.write(f"Combination {i + 1}:\n")
            for key, value in params_copy.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"  Validation Loss: {val_loss:.6f}\n\n")

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)

    # Process and save results (only on rank 0)
    if rank == 0:
        # Flatten list of lists
        flat_results = [item for sublist in all_results for item in sublist]

        # Sort results by validation loss
        flat_results.sort(key=lambda x: x['val_loss'])
        best_params = flat_results[0]

        # Save results to file
        with open(os.path.join(args.output_dir, 'hyperparameter_tuning_results.json'), 'w') as f:
            json.dump(flat_results, f, indent=2)

        # Write final results
        log_file = os.path.join(args.output_dir, 'hyperparameter_tuning.log')
        with open(log_file, 'w') as f:
            f.write("Hyperparameter Tuning Results\n")
            f.write("-----------------------------\n\n")

            f.write("Best Hyperparameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nTop 5 Hyperparameter Combinations:\n")
            for i, params in enumerate(flat_results[:5]):
                f.write(f"Rank {i + 1}:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

        print("\nHyperparameter tuning completed!")
        print(f"Best validation loss: {best_params['val_loss']:.6f}")
        print("Best parameters:")
        for key, value in best_params.items():
            if key != 'val_loss':
                print(f"  {key}: {value}")

        return {
            'best_params': best_params,
            'all_results': flat_results
        }

    return None


def train_with_params(model, params, train_loader, val_loader, device, epochs=30):
    """
    Train a model with the given parameters for a fixed number of epochs.

    Args:
        model (nn.Module): The model to train
        params (dict): Hyperparameters dictionary
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for training
        epochs (int): Number of epochs to train

    Returns:
        float: Best validation loss achieved
    """
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Learning rate scheduling
        scheduler.step(val_loss)

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Hyperparameter Tuning for PointNet++')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Tuning parameters
    parser.add_argument('--tuning_epochs', type=int, default=30,
                        help='Number of epochs for each hyperparameter combination')
    parser.add_argument('--max_combinations', type=int, default=-1,
                        help='Maximum number of combinations to try (-1 for all)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/tuning',
                        help='Directory to save results')

    args = parser.parse_args()
    tuning_results = distributed_hyperparameter_tuning(args)