import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_utils import load_texture_dataset, create_data_loaders
from utils.visualization import plot_learning_curves
from models.texture_pointnet import TexturePointNet


def train_model(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess the dataset
    dataset_dict = load_texture_dataset(args.data_path)
    data_loaders = create_data_loaders(dataset_dict, args.batch_size)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = TexturePointNet(
        num_points=dataset_dict['num_points'],
        in_channels=dataset_dict['in_channels'],  # This is now either 1 or 2 depending on complex data
        output_dim=dataset_dict['output_dim'],
        dropout_rate=args.dropout_rate
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience, factor=args.lr_factor
    )

    # Training loop
    best_val_loss = float('inf')
    counter = 0
    training_losses = []
    validation_losses = []

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in data_loaders['train_loader']:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss = train_loss / len(data_loaders['train_loader'].dataset)
        training_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in data_loaders['val_loader']:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss = val_loss / len(data_loaders['val_loader'].dataset)
        validation_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{args.epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Model saved at epoch {epoch + 1}")
            counter = 0
        else:
            counter += 1

        if counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    # Plot learning curves
    plot_learning_curves(
        training_losses,
        validation_losses,
        os.path.join(args.output_dir, 'learning_curves.png')
    )

    return {
        'model': model,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'best_val_loss': best_val_loss
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PointNet++ for Texture Analysis')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor for learning rate scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout rate')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda/cpu)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    train_results = train_model(args)