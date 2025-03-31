import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def plot_learning_curves(training_losses, validation_losses, save_path='learning_curves.png'):
    """
    Plot training and validation loss curves.

    Args:
        training_losses (list): List of training losses
        validation_losses (list): List of validation losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves saved to {save_path}")


def plot_predictions_vs_true(true_values, predictions, save_path='prediction_vs_true.png'):
    """
    Plot predicted r-values against true r-values.

    Args:
        true_values (ndarray): True r-values
        predictions (ndarray): Predicted r-values
        save_path (str): Path to save the plot
    """
    angle_labels = ['0°', '45°', '90°']
    plt.figure(figsize=(15, 5))

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(true_values[:, i], predictions[:, i], alpha=0.5)
        plt.plot([min(true_values[:, i]), max(true_values[:, i])],
                 [min(true_values[:, i]), max(true_values[:, i])],
                 'r--')
        plt.xlabel(f'True r-value ({angle_labels[i]})')
        plt.ylabel(f'Predicted r-value ({angle_labels[i]})')
        plt.title(f'R² = {r2_score(true_values[:, i], predictions[:, i]):.3f}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction vs. true plot saved to {save_path}")


def plot_model_comparison(models, rmse_values, r2_values, save_path='model_comparison.png'):
    """
    Plot comparison between different models.

    Args:
        models (list): List of model names
        rmse_values (list): List of RMSE values for each model
        r2_values (list): List of R² values for each model
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(models, rmse_values)
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison (lower is better)')

    plt.subplot(1, 2, 2)
    plt.bar(models, r2_values)
    plt.ylabel('R²')
    plt.title('R² Comparison (higher is better)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Model comparison plot saved to {save_path}")


def plot_feature_importance(importances, indices, save_path='feature_importance.png', top_n=20):
    """
    Plot feature importance for random forest model.

    Args:
        importances (ndarray): Feature importance values
        indices (ndarray): Sorted indices of feature importances
        save_path (str): Path to save the plot
        top_n (int): Number of top features to plot
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xlabel('Fourier Coefficient Index')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Most Important Fourier Coefficients')
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")