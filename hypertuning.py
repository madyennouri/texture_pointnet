import os
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid

from utils.data_utils import load_texture_dataset, create_data_loaders
from models.texture_pointnet import TexturePointNet


def train_with_params(model, optimizer, train_loader, val_loader, device, epochs=30):
    """
    Train a model with the given parameters for a fixed number of epochs.

    Args:
        model (nn.Module): The model to train
        optimizer (optim.Optimizer): The optimizer to use
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for training
        epochs (int): Number of epochs to train

    Returns:
        float: Best validation loss achieved
    """
    criterion = nn.MSELoss()
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


def hyperparameter_tuning(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess the dataset
    dataset_dict = load_texture_dataset(args.data_path)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Testing {len(grid)} hyperparameter combinations")

    # Store results
    results = []
    best_val_loss = float('inf')
    best_params = None

    # Log file for results
    log_file = os.path.join(args.output_dir, 'hyperparameter_tuning.log')
    with open(log_file, 'w') as f:
        f.write("Hyperparameter Tuning Results\n")
        f.write("-----------------------------\n\n")

    # Loop through hyperparameter combinations
    for i, params in enumerate(grid):
        print(f"\nCombination {i + 1}/{len(grid)}")
        print(f"Parameters: {params}")

        # Create data loaders with current batch size
        data_loaders = create_data_loaders(dataset_dict, params['batch_size'])

        # Initialize model with current hyperparameters
        model = TexturePointNet(
            num_points=dataset_dict['num_points'],
            output_dim=dataset_dict['output_dim'],
            dropout_rate=params['dropout_rate']
        ).to(device)

        # Initialize optimizer with current learning rate
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Train for a fixed number of epochs
        val_loss = train_with_params(
            model, optimizer, data_loaders['train_loader'],
            data_loaders['val_loader'], device, epochs=args.tuning_epochs
        )

        # Store results
        params['val_loss'] = val_loss
        results.append(params.copy())

        # Update best parameters
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()
            print(f"New best validation loss: {best_val_loss:.6f}")

        # Log results
        with open(log_file, 'a') as f:
            f.write(f"Combination {i + 1}:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"  Validation Loss: {val_loss:.6f}\n\n")

    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'])

    # Write final results
    with open(log_file, 'a') as f:
        f.write("\nBest Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"  Validation Loss: {best_val_loss:.6f}\n\n")

        f.write("\nTop 5 Hyperparameter Combinations:\n")
        for i, params in enumerate(results[:5]):
            f.write(f"Rank {i + 1}:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    print("\nHyperparameter tuning completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to {log_file}")

    return {
        'best_params': best_params,
        'best_val_loss': best_val_loss,
        'all_results': results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for PointNet++')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Tuning parameters
    parser.add_argument('--tuning_epochs', type=int, default=30,
                        help='Number of epochs for each hyperparameter combination')
    parser.add_argument('--max_combinations', type=int, default=-1,
                        help='Maximum number of hyperparameter combinations to try (-1 for all)')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda/cpu)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    tuning_results = hyperparameter_tuning(args)