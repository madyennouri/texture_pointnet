import os
import argparse
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_utils import load_texture_dataset, create_data_loaders
from utils.visualization import plot_model_comparison, plot_feature_importance
from models.texture_pointnet import TexturePointNet


def compare_with_baselines(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess the dataset
    dataset_dict = load_texture_dataset(args.data_path)
    data_loaders = create_data_loaders(dataset_dict, args.batch_size)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get original numpy arrays for scikit-learn models
    orig_data = dataset_dict['original_data']
    X_train = orig_data['X_train'].reshape(orig_data['X_train'].shape[0], -1)
    X_test = orig_data['X_test'].reshape(orig_data['X_test'].shape[0], -1)
    y_train = orig_data['y_train']
    y_test = orig_data['y_test']

    results = {
        'models': ['PointNet++', 'Linear Regression', 'Random Forest'],
        'rmse': [],
        'r2': []
    }

    # 1. Evaluate PointNet++ model
    print("Evaluating PointNet++ model...")
    model = TexturePointNet(
        num_points=dataset_dict['num_points'],
        in_channels=dataset_dict['in_channels'],  # This is now either 1 or 2 depending on complex data
        output_dim=dataset_dict['output_dim'],
        dropout_rate=args.dropout_rate
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Get predictions on test set
    pointnet_preds = []
    true_values = []

    with torch.no_grad():
        for batch_X, batch_y in data_loaders['test_loader']:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            pointnet_preds.append(outputs.cpu().numpy())
            true_values.append(batch_y.numpy())

    pointnet_preds = np.vstack(pointnet_preds)
    true_values = np.vstack(true_values)

    # Calculate metrics
    pointnet_rmse = np.sqrt(mean_squared_error(true_values, pointnet_preds))
    pointnet_r2 = r2_score(true_values, pointnet_preds)

    results['rmse'].append(pointnet_rmse)
    results['r2'].append(pointnet_r2)

    # 2. Linear Regression
    print("Training and evaluating Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    results['rmse'].append(lr_rmse)
    results['r2'].append(lr_r2)

    # 3. Random Forest
    print("Training and evaluating Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    results['rmse'].append(rf_rmse)
    results['r2'].append(rf_r2)

    # Print results
    print("\nModel Comparison Results:")
    for i, model_name in enumerate(results['models']):
        print(f"{model_name}:")
        print(f"  RMSE: {results['rmse'][i]:.6f}")
        print(f"  R²: {results['r2'][i]:.6f}")

    # Save results to file
    with open(os.path.join(args.output_dir, 'model_comparison.txt'), 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("----------------------\n\n")

        for i, model_name in enumerate(results['models']):
            f.write(f"{model_name}:\n")
            f.write(f"  RMSE: {results['rmse'][i]:.6f}\n")
            f.write(f"  R²: {results['r2'][i]:.6f}\n\n")

    # Plot model comparison
    plot_model_comparison(
        results['models'],
        results['rmse'],
        results['r2'],
        os.path.join(args.output_dir, 'model_comparison.png')
    )

    # Feature importance analysis for Random Forest
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plot_feature_importance(
            importances,
            indices,
            os.path.join(args.output_dir, 'feature_importance.png')
        )

        # Save top 20 feature importances to file
        with open(os.path.join(args.output_dir, 'feature_importance.txt'), 'w') as f:
            f.write("Feature Importance Analysis (Random Forest)\n")
            f.write("----------------------------------------\n\n")
            f.write("Top 20 most important Fourier coefficients:\n")

            for i in range(20):
                f.write(f"Coefficient {indices[i]}: {importances[indices[i]]:.6f}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare PointNet++ with baseline models')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.mat',
                        help='Path to the dataset file')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved PointNet++ model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation (cuda/cpu)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    comparison_results = compare_with_baselines(args)