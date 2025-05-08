import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json


def calculate_mae(pred_counts, gt_counts):
    """
    Calculate Mean Absolute Error

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts

    Returns:
        float: Mean Absolute Error value
    """
    assert len(pred_counts) == len(gt_counts), "Number of predictions and ground truths must match"

    mae = np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts)))
    return mae


def calculate_mse(pred_counts, gt_counts):
    """
    Calculate Mean Squared Error

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts

    Returns:
        float: Mean Squared Error value
    """
    assert len(pred_counts) == len(gt_counts), "Number of predictions and ground truths must match"

    mse = np.mean(np.square(np.array(pred_counts) - np.array(gt_counts)))
    return mse


def calculate_rmse(pred_counts, gt_counts):
    """
    Calculate Root Mean Squared Error

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts

    Returns:
        float: Root Mean Squared Error value
    """
    return np.sqrt(calculate_mse(pred_counts, gt_counts))


def calculate_nae(pred_counts, gt_counts):
    """
    Calculate Normalized Absolute Error

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts

    Returns:
        float: Normalized Absolute Error value
    """
    assert len(pred_counts) == len(gt_counts), "Number of predictions and ground truths must match"

    gt_counts = np.array(gt_counts)
    # Avoid division by zero
    gt_counts[gt_counts == 0] = 1

    nae = np.mean(np.abs(np.array(pred_counts) - gt_counts) / gt_counts)
    return nae


def calculate_all_metrics(pred_counts, gt_counts):
    """
    Calculate all metrics: MAE, MSE, RMSE, NAE

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts

    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'mae': calculate_mae(pred_counts, gt_counts),
        'mse': calculate_mse(pred_counts, gt_counts),
        'rmse': calculate_rmse(pred_counts, gt_counts),
        'nae': calculate_nae(pred_counts, gt_counts)
    }

    return metrics


def plot_error_distribution(pred_counts, gt_counts, output_path=None):
    """
    Plot distribution of absolute errors

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts
        output_path (str, optional): Path to save the plot. If None, plot will be shown.

    Returns:
        None
    """
    errors = np.abs(np.array(pred_counts) - np.array(gt_counts))

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20)
    plt.title('Distribution of Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_prediction_vs_ground_truth(pred_counts, gt_counts, output_path=None):
    """
    Plot scatter plot of predictions vs ground truth

    Args:
        pred_counts (list): List of predicted counts
        gt_counts (list): List of ground truth counts
        output_path (str, optional): Path to save the plot. If None, plot will be shown.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_counts, pred_counts, alpha=0.7)

    # Add perfect prediction line
    max_count = max(max(gt_counts), max(pred_counts))
    plt.plot([0, max_count], [0, max_count], 'r--')

    plt.title('Predicted vs Ground Truth Counts')
    plt.xlabel('Ground Truth Count')
    plt.ylabel('Predicted Count')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def save_metrics_to_file(metrics, output_path):
    """
    Save metrics to a JSON file

    Args:
        metrics (dict): Dictionary containing metrics
        output_path (str): Path to save the metrics

    Returns:
        None
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy values to Python native types
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)

    print(f"Metrics saved to {output_path}")


def load_metrics_from_file(file_path):
    """
    Load metrics from a JSON file

    Args:
        file_path (str): Path to the metrics file

    Returns:
        dict: Dictionary containing metrics
    """
    with open(file_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def evaluate_model_performance(model, dataloader, device):
    """
    Evaluate model performance on a dataset

    Args:
        model (torch.nn.Module): PyTorch model
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        device (str): Device to run the model on ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing metrics
    """
    model.eval()
    pred_counts = []
    gt_counts = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Calculate counts
            for i in range(output.shape[0]):
                pred_count = output[i].sum().item()
                gt_count = target[i].sum().item()

                pred_counts.append(pred_count)
                gt_counts.append(gt_count)

    # Calculate metrics
    metrics = calculate_all_metrics(pred_counts, gt_counts)

    return metrics, pred_counts, gt_counts


if __name__ == "__main__":
    # Example usage
    pred_counts = [100, 150, 200, 250, 300]
    gt_counts = [110, 160, 190, 270, 280]

    metrics = calculate_all_metrics(pred_counts, gt_counts)
    print("Metrics:", metrics)

    plot_error_distribution(pred_counts, gt_counts)
    plot_prediction_vs_ground_truth(pred_counts, gt_counts)
