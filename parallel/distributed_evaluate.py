import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_utils import load_texture_dataset
from utils.visualization import plot_predictions_vs_true
from models.texture_pointnet import TexturePointNet
from parallel.utils import init_distributed_mode, is_main_process, get_rank, get_world_size


def distributed_evaluate_model(args):
    """
    Evaluate model using distributed processing.

    Args:
        args: Command-line arguments

    Returns:
        dict: Evaluation metrics
    """
    # Initialize distributed environment
    init_distributed_mode(args)
    world_size = get_world_size()
    rank = get_rank()

    # Create output directory if main process
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Distributed evaluation with {world_size} processes")

    # Set device
    device = torch.device(args.device)

    # Load dataset
    dataset_dict = load_texture_dataset(args.data_path)

    # Create distributed test sampler and data loader
    test_dataset = TensorDataset(dataset_dict['X_test'], dataset_dict['y_test'])

    # Use DistributedSampler to partition the test dataset
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # Don't shuffle for evaluation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
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

    # Load model weights
    if is_main_process():
        print(f"Loading model from {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank] if args.device.startswith('cuda') else None)
    model.eval()

    # Evaluate on test set
    local_predictions = []
    local_true_values = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            local_predictions.append(outputs.cpu().numpy())
            local_true_values.append(batch_y.cpu().numpy())

    local_predictions = np.vstack(local_predictions)
    local_true_values = np.vstack(local_true_values)

    # Gather predictions and labels from all processes
    gathered_predictions = [None for _ in range(world_size)]
    gathered_true_values = [None for _ in range(world_size)]

    # Use all_gather to collect results from all processes
    dist.all_gather_object(gathered_predictions, local_predictions)
    dist.all_gather_object(gathered_true_values, local_true_values)

    # Process evaluation metrics (only on rank 0)
    if is_main_process():
        # Concatenate results from all processes
        all_predictions = np.vstack(gathered_predictions)
        all_true_values = np.vstack(gathered_true_values)

        # Calculate metrics
        mse = mean_squared_error(all_true_values, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_true_values, all_predictions)

        # Calculate per-angle metrics
        angle_labels = ['0°', '45°', '90°']
        angle_metrics = []

        for i in range(dataset_dict['output_dim']):
            angle_mse = mean_squared_error(all_true_values[:, i], all_predictions[:, i])
            angle_rmse = np.sqrt(angle_mse)
            angle_r2 = r2_score(all_true_values[:, i], all_predictions[:, i])
            angle_metrics.append({
                'angle': angle_labels[i],
                'mse': angle_mse,
                'rmse': angle_rmse,
                'r2': angle_r2
            })

        # Print evaluation results
        print(f"Overall Test MSE: {mse:.6f}")
        print(f"Overall Test RMSE: {rmse:.6f}")
        print(f"Overall Test R²: {r2:.6f}")
        print("\nPer-angle metrics:")

        for metric in angle_metrics:
            print(f"  {metric['angle']} - RMSE: {metric['rmse']:.6f}, R²: {metric['r2']:.6f}")

        # Save results to file
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Overall Test MSE: {mse:.6f}\n")
            f.write(f"Overall Test RMSE: {rmse:.6f}\n")
            f.write(f"Overall Test R²: {r2:.6f}\n\n")
            f.write("Per-angle metrics:\n")

            for metric in angle_metrics:
                f.write(f"  {metric['angle']} - MSE: {metric['mse']:.6f}, "
                        f"RMSE: {metric['rmse']:.6f}, R²: {metric['r2']:.6f}\n")

        # Plot predictions vs. true values
        plot_predictions_vs_true(
            all_true_values,
            all_predictions,
            os.path.join(args.output_dir, 'prediction_vs_true.png')
        )

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'angle_metrics': angle_metrics,
            'predictions': all_predictions,
            'true_values': all_true_values
        }

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Evaluation for PointNet++')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU/process')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout rate')

    # Distributed parameters
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl',
                        help='Distributed backend')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='Distributed process rank')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed evaluation')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    eval_results = distributed_evaluate_model(args)