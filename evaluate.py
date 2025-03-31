import os
import argparse
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_utils import load_texture_dataset, create_data_loaders
from utils.visualization import plot_predictions_vs_true
from models.texture_pointnet import TexturePointNet


def evaluate_model(args):
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

    # Load the saved model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Evaluate on test set
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_X, batch_y in data_loaders['test_loader']:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            true_values.append(batch_y.numpy())

    predictions = np.vstack(predictions)
    true_values = np.vstack(true_values)

    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)

    # Calculate per-angle metrics
    angle_labels = ['0°', '45°', '90°']
    angle_metrics = []

    for i in range(dataset_dict['output_dim']):
        angle_mse = mean_squared_error(true_values[:, i], predictions[:, i])
        angle_rmse = np.sqrt(angle_mse)
        angle_r2 = r2_score(true_values[:, i], predictions[:, i])
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
        true_values,
        predictions,
        os.path.join(args.output_dir, 'prediction_vs_true.png')
    )

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'angle_metrics': angle_metrics,
        'predictions': predictions,
        'true_values': true_values
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PointNet++ for Texture Analysis')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation (cuda/cpu)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    eval_results = evaluate_model(args)