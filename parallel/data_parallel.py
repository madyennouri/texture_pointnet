import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from utils.data_utils import load_texture_dataset


def parallel_data_loading(file_path, num_processes=4):
    """
    Load and preprocess the dataset using parallel processing.

    Args:
        file_path: Path to the dataset file
        num_processes: Number of parallel processes to use

    Returns:
        dict: Dictionary containing the dataset
    """
    # Load the raw data (this part cannot be easily parallelized)
    dataset_dict = load_texture_dataset(file_path)

    # Further preprocessing can be parallelized
    # For example, applying additional transformations to the data

    return dataset_dict


def process_batch(batch_data, transform_fn):
    """
    Process a batch of data with a transformation function.

    Args:
        batch_data: Batch of data to process
        transform_fn: Transformation function to apply

    Returns:
        ndarray: Processed batch
    """
    return transform_fn(batch_data)


def parallel_transform_dataset(dataset, transform_fn, num_processes=4, batch_size=1000):
    """
    Apply a transformation to a dataset in parallel.

    Args:
        dataset: Dataset to transform (numpy array)
        transform_fn: Function to apply to each batch
        num_processes: Number of parallel processes
        batch_size: Size of batches to process

    Returns:
        ndarray: Transformed dataset
    """
    # Split dataset into batches
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    batches = [dataset[i * batch_size:min((i + 1) * batch_size, num_samples)] for i in range(num_batches)]

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        processed_batches = list(executor.map(partial(process_batch, transform_fn=transform_fn), batches))

    # Concatenate results
    return np.concatenate(processed_batches, axis=0)


def parallel_preprocessing(dataset_dict, num_processes=4, add_features=False):
    """
    Preprocess a dataset dictionary in parallel.

    Args:
        dataset_dict: Dictionary containing dataset tensors
        num_processes: Number of parallel processes
        add_features: Whether to add engineered features

    Returns:
        dict: Preprocessed dataset dictionary
    """

    # Define transformation functions
    def normalize(data):
        """Normalize data to zero mean and unit variance."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        return (data - mean) / std

    def add_engineered_features(data):
        """Add engineered features to the data."""
        # Example: Add magnitude of complex coefficients
        if data.shape[-1] == 2:  # Complex data with real and imaginary parts
            real = data[..., 0]
            imag = data[..., 1]
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            phase = np.arctan2(imag, real)

            # Add as new channels
            new_data = np.concatenate([
                data,
                magnitude[..., np.newaxis],
                phase[..., np.newaxis]
            ], axis=-1)

            return new_data
        return data

    # Apply transformations in parallel
    result_dict = {}

    # Process training data
    X_train_processed = parallel_transform_dataset(
        dataset_dict['X_train'].numpy(), normalize, num_processes
    )

    if add_features:
        X_train_processed = parallel_transform_dataset(
            X_train_processed, add_engineered_features, num_processes
        )

    result_dict['X_train'] = torch.FloatTensor(X_train_processed)

    # Process validation data
    X_val_processed = parallel_transform_dataset(
        dataset_dict['X_val'].numpy(), normalize, num_processes
    )

    if add_features:
        X_val_processed = parallel_transform_dataset(
            X_val_processed, add_engineered_features, num_processes
        )

    result_dict['X_val'] = torch.FloatTensor(X_val_processed)

    # Process test data
    X_test_processed = parallel_transform_dataset(
        dataset_dict['X_test'].numpy(), normalize, num_processes
    )

    if add_features:
        X_test_processed = parallel_transform_dataset(
            X_test_processed, add_engineered_features, num_processes
        )

    result_dict['X_test'] = torch.FloatTensor(X_test_processed)

    # Copy other elements from the original dictionary
    for key in ['y_train', 'y_val', 'y_test', 'num_points', 'in_channels', 'output_dim']:
        if key in dataset_dict:
            result_dict[key] = dataset_dict[key]

    # Update in_channels if features were added
    if add_features and 'in_channels' in result_dict:
        result_dict['in_channels'] = result_dict['X_train'].shape[-1]

    return result_dict