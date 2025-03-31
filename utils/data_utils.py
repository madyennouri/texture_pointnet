import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def load_texture_dataset(file_path='data/ml_dataset.mat'):
    """
    Load and preprocess the texture dataset with complex numbers.

    Args:
        file_path (str): Path to the .mat file

    Returns:
        dict: Dictionary containing data loaders and other dataset information
    """
    # Load the dataset
    print(f"Loading dataset from {file_path}")
    data = sio.loadmat(file_path)
    X = data['X']  # Fourier coefficients
    Y = data['Y']  # r-values (0°, 45°, 90°)

    # Handle complex coefficients by separating real and imaginary parts
    print(f"Original data shape: {X.shape}")
    print(f"Checking for complex values: {np.iscomplexobj(X)}")

    # Get dataset dimensions
    num_samples = X.shape[0]
    num_points = X.shape[1]

    # Handle complex numbers by separating real and imaginary parts
    if np.iscomplexobj(X):
        print("Converting complex coefficients to real values...")
        # Extract real and imaginary parts and concatenate as separate channels
        X_real = np.real(X)
        X_imag = np.imag(X)
        # Reshape to create a 2-channel point cloud
        X_points = np.stack([X_real, X_imag], axis=2)
        print(f"New shape after complex conversion: {X_points.shape}")
    else:
        # If data is already real, just reshape
        X_points = X.reshape(num_samples, num_points, 1)

    # Normalize each channel separately
    X_points_normalized = np.zeros_like(X_points)
    for i in range(X_points.shape[2]):
        channel = X_points[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        # Avoid division by zero
        if std > 0:
            X_points_normalized[:, :, i] = (channel - mean) / std
        else:
            X_points_normalized[:, :, i] = channel - mean

    print(f"Normalized data shape: {X_points_normalized.shape}")

    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_points_normalized, Y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15 / 0.85, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    y_test_tensor = torch.FloatTensor(y_test)

    # For baseline models, we'll need flattened data
    # Reshape data for baseline models (flatten the spatial dimensions)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Save original numpy arrays for baseline models
    original_data = {
        'X_train': X_train_flat,
        'X_val': X_val_flat,
        'X_test': X_test_flat,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    # Get feature dimensions for the model
    in_channels = X_points.shape[2]  # Number of channels (real + imaginary)

    print(
        f"Dataset loaded: {X_train_tensor.shape[0]} training, {X_val_tensor.shape[0]} validation, {X_test_tensor.shape[0]} test samples")

    return {
        'num_points': num_points,
        'in_channels': in_channels,
        'output_dim': Y.shape[1],
        'X_train': X_train_tensor,
        'X_val': X_val_tensor,
        'X_test': X_test_tensor,
        'y_train': y_train_tensor,
        'y_val': y_val_tensor,
        'y_test': y_test_tensor,
        'original_data': original_data
    }


def create_data_loaders(dataset_dict, batch_size=32):
    """
    Create PyTorch data loaders from the dataset dictionary.

    Args:
        dataset_dict (dict): Dictionary containing the dataset tensors
        batch_size (int): Batch size for the data loaders

    Returns:
        dict: Dictionary containing the data loaders
    """
    # Create datasets
    train_dataset = TensorDataset(dataset_dict['X_train'], dataset_dict['y_train'])
    val_dataset = TensorDataset(dataset_dict['X_val'], dataset_dict['y_val'])
    test_dataset = TensorDataset(dataset_dict['X_test'], dataset_dict['y_test'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }