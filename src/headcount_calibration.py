import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Import inference modules
from headcount_inference import get_device, load_model, predict_headcount


def load_ground_truth_data(ground_truth_file):
    """
    Load ground truth data from JSON or CSV file

    Expected format for JSON:
    {
        "images": [
            {
                "image_path": "path/to/image1.jpg",
                "ground_truth_count": 42
            },
            ...
        ]
    }

    Expected format for CSV:
    image_path,ground_truth_count
    path/to/image1.jpg,42
    ...
    """
    ext = os.path.splitext(ground_truth_file)[1].lower()

    if ext == '.json':
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'images' in data:
            return data['images']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Invalid JSON format. Expected 'images' key with list of objects")

    elif ext == '.csv':
        df = pd.read_csv(ground_truth_file)

        if 'image_path' not in df.columns or 'ground_truth_count' not in df.columns:
            raise ValueError("CSV must have 'image_path' and 'ground_truth_count' columns")

        return df.to_dict('records')

    else:
        raise ValueError(f"Unsupported file format: {ext}")

def calculate_calibration_factors(model, ground_truth_data, device, target_size=(384, 384)):
    """
    Calculate calibration factors based on ground truth data

    Args:
        model: CSRNet model
        ground_truth_data: List of dictionaries with image_path and ground_truth_count
        device: Device to run model on
        target_size: Input size for the model

    Returns:
        dict: Dictionary of calibration statistics
    """
    results = []
    ratios = []
    errors_before = []
    density_stats = defaultdict(list)

    print(f"Calculating calibration factors for {len(ground_truth_data)} images...")

    for item in tqdm(ground_truth_data):
        image_path = item['image_path']
        gt_count = item['ground_truth_count']

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue

        # Get raw prediction
        pred_count, density_map, _ = predict_headcount(
            model, image_path, device, target_size, calibration_factor=1.0
        )

        if pred_count is None or pred_count == 0:
            print(f"Warning: Failed to get prediction for {image_path}")
            continue

        # Calculate ratio
        ratio = gt_count / pred_count
        ratios.append(ratio)

        # Calculate error before calibration
        error_before = abs(pred_count - gt_count) / gt_count * 100
        errors_before.append(error_before)

        # Categorize by density
        density_category = "low" if gt_count < 50 else "medium" if gt_count < 150 else "high"
        density_stats[density_category].append(ratio)

        # Store result
        results.append({
            'image_path': image_path,
            'ground_truth': gt_count,
            'prediction': pred_count,
            'ratio': ratio,
            'error_before': error_before,
            'density_category': density_category
        })

    if not ratios:
        print("No valid calibration data found")
        return None

    # Calculate overall statistics
    overall_calibration = {
        'mean_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'min_ratio': float(np.min(ratios)),
        'max_ratio': float(np.max(ratios)),
        'std_ratio': float(np.std(ratios)),
        'mean_error_before': float(np.mean(errors_before)),
        'count_ranges': {
            'all': len(ratios),
            'low': len(density_stats['low']),
            'medium': len(density_stats['medium']),
            'high': len(density_stats['high'])
        }
    }

    # Calculate density-specific calibration factors
    density_calibration = {}
    for category, values in density_stats.items():
        if values:
            density_calibration[category] = {
                'mean_ratio': float(np.mean(values)),
                'median_ratio': float(np.median(values)),
                'count': len(values)
            }

    return {
        'overall': overall_calibration,
        'by_density': density_calibration,
        'recommended_factor': float(np.median(ratios)),
        'detailed_results': results
    }

def validate_calibration(model, ground_truth_data, calibration_factor, device, target_size=(384, 384)):
    """
    Validate calibration factor on ground truth data

    Args:
        model: CSRNet model
        ground_truth_data: List of dictionaries with image_path and ground_truth_count
        calibration_factor: Calibration factor to apply
        device: Device to run model on
        target_size: Input size for the model

    Returns:
        dict: Validation statistics
    """
    errors_before = []
    errors_after = []
    mae_before = []
    mae_after = []
    mse_before = []
    mse_after = []
    results = []

    print(f"Validating calibration factor {calibration_factor:.4f} on {len(ground_truth_data)} images...")

    for item in tqdm(ground_truth_data):
        image_path = item['image_path']
        gt_count = item['ground_truth_count']

        if not os.path.exists(image_path):
            continue

        # Get raw prediction
        raw_pred, _, _ = predict_headcount(
            model, image_path, device, target_size, calibration_factor=1.0
        )

        if raw_pred is None:
            continue

        # Apply calibration
        calibrated_pred = raw_pred * calibration_factor

        # Calculate percentage errors
        error_before = abs(raw_pred - gt_count) / gt_count * 100
        error_after = abs(calibrated_pred - gt_count) / gt_count * 100

        # Calculate MAE and MSE
        mae_before.append(abs(raw_pred - gt_count))
        mae_after.append(abs(calibrated_pred - gt_count))
        mse_before.append((raw_pred - gt_count) ** 2)
        mse_after.append((calibrated_pred - gt_count) ** 2)

        errors_before.append(error_before)
        errors_after.append(error_after)

        # Store result
        results.append({
            'image_path': image_path,
            'ground_truth': gt_count,
            'raw_prediction': raw_pred,
            'calibrated_prediction': calibrated_pred,
            'error_before': error_before,
            'error_after': error_after,
            'mae_before': abs(raw_pred - gt_count),
            'mae_after': abs(calibrated_pred - gt_count),
            'mse_before': (raw_pred - gt_count) ** 2,
            'mse_after': (calibrated_pred - gt_count) ** 2
        })

    if not errors_before:
        print("No valid validation data found")
        return None

    # Calculate statistics
    validation_stats = {
        'mean_error_before': float(np.mean(errors_before)),
        'mean_error_after': float(np.mean(errors_after)),
        'median_error_before': float(np.median(errors_before)),
        'median_error_after': float(np.median(errors_after)),
        'max_error_before': float(np.max(errors_before)),
        'max_error_after': float(np.max(errors_after)),
        'mae_before': float(np.mean(mae_before)),
        'mae_after': float(np.mean(mae_after)),
        'rmse_before': float(np.sqrt(np.mean(mse_before))),
        'rmse_after': float(np.sqrt(np.mean(mse_after))),
        'improvement': float(np.mean(errors_before) - np.mean(errors_after)),
        'improvement_percentage': float((np.mean(errors_before) - np.mean(errors_after)) / np.mean(errors_before) * 100),
        'calibration_factor': float(calibration_factor),
        'num_images': len(errors_before)
    }

    return {
        'statistics': validation_stats,
        'detailed_results': results
    }

def plot_calibration_results(calibration_results, output_dir):
    """
    Generate plots for calibration results

    Args:
        calibration_results: Dictionary from calculate_calibration_factors
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    results = calibration_results['detailed_results']
    ground_truth = [r['ground_truth'] for r in results]
    predictions = [r['prediction'] for r in results]
    ratios = [r['ratio'] for r in results]

    # Create scatter plot of predictions vs ground truth
    plt.figure(figsize=(10, 8))
    plt.scatter(ground_truth, predictions, alpha=0.7)

    # Add perfect prediction line
    max_val = max(max(ground_truth), max(predictions))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')

    # Add calibrated prediction line
    calib_factor = calibration_results['recommended_factor']
    plt.plot([0, max_val], [0, max_val/calib_factor], 'g--',
             label=f'Calibrated (factor={calib_factor:.2f})')

    plt.xlabel('Ground Truth Count')
    plt.ylabel('Predicted Count')
    plt.title('Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_vs_truth.png'), dpi=150)
    plt.close()

    # Create histogram of calibration ratios
    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=20, alpha=0.7)
    plt.axvline(np.median(ratios), color='r', linestyle='--',
                label=f'Median: {np.median(ratios):.2f}')
    plt.axvline(np.mean(ratios), color='g', linestyle='--',
                label=f'Mean: {np.mean(ratios):.2f}')
    plt.xlabel('Calibration Ratio (Ground Truth / Prediction)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Calibration Ratios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'calibration_ratios.png'), dpi=150)
    plt.close()

    # Create scatter plot of ratios vs ground truth count
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, ratios, alpha=0.7)
    plt.axhline(np.median(ratios), color='r', linestyle='--',
                label=f'Median Ratio: {np.median(ratios):.2f}')
    plt.xlabel('Ground Truth Count')
    plt.ylabel('Calibration Ratio')
    plt.title('Calibration Ratio vs Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ratio_vs_count.png'), dpi=150)
    plt.close()

def plot_validation_results(validation_results, output_dir):
    """
    Generate plots for validation results

    Args:
        validation_results: Dictionary from validate_calibration
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    results = validation_results['detailed_results']
    ground_truth = [r['ground_truth'] for r in results]
    raw_predictions = [r['raw_prediction'] for r in results]
    calibrated_predictions = [r['calibrated_prediction'] for r in results]
    errors_before = [r['error_before'] for r in results]
    errors_after = [r['error_after'] for r in results]

    # Create comparison scatter plot
    plt.figure(figsize=(12, 9))

    plt.scatter(ground_truth, raw_predictions, alpha=0.5, label='Raw Predictions')
    plt.scatter(ground_truth, calibrated_predictions, alpha=0.5, label='Calibrated Predictions')

    # Add perfect prediction line
    max_val = max(max(ground_truth), max(raw_predictions), max(calibrated_predictions))
    plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Prediction')

    plt.xlabel('Ground Truth Count')
    plt.ylabel('Predicted Count')
    plt.title('Raw vs Calibrated Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'raw_vs_calibrated.png'), dpi=150)
    plt.close()

    # Create error comparison histogram
    plt.figure(figsize=(12, 6))

    plt.hist(errors_before, bins=20, alpha=0.5, label='Before Calibration')
    plt.hist(errors_after, bins=20, alpha=0.5, label='After Calibration')

    plt.axvline(np.mean(errors_before), color='r', linestyle='--',
                label=f'Mean Before: {np.mean(errors_before):.2f}%')
    plt.axvline(np.mean(errors_after), color='g', linestyle='--',
                label=f'Mean After: {np.mean(errors_after):.2f}%')

    plt.xlabel('Error Percentage')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Before and After Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=150)
    plt.close()

    # Create paired error comparison
    plt.figure(figsize=(10, 6))

    # Pair errors for each image
    paired_data = list(zip(errors_before, errors_after))
    paired_data.sort(key=lambda x: x[0])  # Sort by error before

    indices = range(len(paired_data))
    errors_before_sorted = [x[0] for x in paired_data]
    errors_after_sorted = [x[1] for x in paired_data]

    plt.plot(indices, errors_before_sorted, 'r-', label='Before Calibration')
    plt.plot(indices, errors_after_sorted, 'g-', label='After Calibration')

    plt.xlabel('Image Index (sorted by error before calibration)')
    plt.ylabel('Error Percentage')
    plt.title('Error Comparison for Each Image')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'paired_error_comparison.png'), dpi=150)
    plt.close()

def create_calibration_file(calibration_results, output_path):
    """
    Create calibration file for use with inference script

    Args:
        calibration_results: Dictionary from calculate_calibration_factors
        output_path: Path to save calibration file
    """
    calibration_data = {
        'recommended_factor': calibration_results['recommended_factor'],
        'density_specific_factors': {
            k: v['median_ratio'] for k, v in calibration_results['by_density'].items()
        },
        'metadata': {
            'date_created': pd.Timestamp.now().isoformat(),
            'num_calibration_images': calibration_results['overall']['count_ranges']['all'],
            'mean_error_before': calibration_results['overall']['mean_error_before']
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print(f"Calibration file saved to {output_path}")
    return calibration_data

def main():
    parser = argparse.ArgumentParser(description='CSRNet Calibration Tool')

    # Input options
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='path to ground truth data (JSON or CSV)')
    parser.add_argument('--model', type=str, default='models/csrnet.pth',
                        help='path to CSRNet model')
    parser.add_argument('--target_size', type=int, nargs=2, default=[384, 384],
                        help='target input size (width, height)')

    # Output options
    parser.add_argument('--output_dir', type=str, default='calibration',
                        help='output directory for results')
    parser.add_argument('--calibration_file', type=str, default='calibration/csrnet_calibration.json',
                        help='output path for calibration file')

    # Processing options
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='portion of data to use for validation (0-1)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed for data splitting')
    parser.add_argument('--specific_factor', type=float,
                        help='test specific calibration factor instead of calculating')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    # Get device
    device = get_device()

    # Load model
    model = load_model(args.model, device)

    # Load ground truth data
    ground_truth_data = load_ground_truth_data(args.ground_truth)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Split data into calibration and validation sets if needed
    if args.validation_split > 0:
        # Shuffle data
        indices = np.arange(len(ground_truth_data))
        np.random.shuffle(indices)

        # Calculate split point
        split_idx = int(len(ground_truth_data) * (1 - args.validation_split))

        # Split data
        calibration_indices = indices[:split_idx]
        validation_indices = indices[split_idx:]

        calibration_data = [ground_truth_data[i] for i in calibration_indices]
        validation_data = [ground_truth_data[i] for i in validation_indices]

        print(f"Split data: {len(calibration_data)} for calibration, {len(validation_data)} for validation")
    else:
        # Use all data for calibration
        calibration_data = ground_truth_data
        validation_data = ground_truth_data
        print(f"Using all {len(calibration_data)} images for both calibration and validation")

    # Calculate calibration factors
    if args.specific_factor is not None:
        calibration_factor = args.specific_factor
        print(f"Using specified calibration factor: {calibration_factor}")
    else:
        print("Calculating calibration factors...")
        calibration_results = calculate_calibration_factors(
            model, calibration_data, device, target_size=tuple(args.target_size)
        )

        # Save full calibration results
        with open(os.path.join(args.output_dir, 'calibration_results.json'), 'w') as f:
            json.dump(calibration_results, f, indent=4)

        # Plot calibration results
        plot_calibration_results(calibration_results, args.output_dir)

        # Create calibration file
        create_calibration_file(calibration_results, args.calibration_file)

        # Use recommended factor
        calibration_factor = calibration_results['recommended_factor']
        print(f"Calculated calibration factor: {calibration_factor}")

    # Validate calibration
    print("\nValidating calibration factor...")
    validation_results = validate_calibration(
        model, validation_data, calibration_factor, device, target_size=tuple(args.target_size)
    )

    # Save validation results
    with open(os.path.join(args.output_dir, 'validation_results.json'), 'w') as f:
        json.dump(validation_results, f, indent=4)

    # Plot validation results
    plot_validation_results(validation_results, args.output_dir)

    # Print summary
    stats = validation_results['statistics']
    improvement = stats['improvement_percentage']

    print("\nCalibration Summary:")
    print(f"Calibration Factor: {calibration_factor:.4f}")
    print(f"Mean Error Before: {stats['mean_error_before']:.2f}%")
    print(f"Mean Error After: {stats['mean_error_after']:.2f}%")
    print(f"MAE Before: {stats['mae_before']:.2f}")
    print(f"MAE After: {stats['mae_after']:.2f}")
    print(f"RMSE Before: {stats['rmse_before']:.2f}")
    print(f"RMSE After: {stats['rmse_after']:.2f}")
    print(f"Improvement: {improvement:.2f}%")

    if improvement > 0:
        print("\nCalibration successful! Use this factor for more accurate predictions.")
    else:
        print("\nWarning: Calibration did not improve accuracy. Check your ground truth data.")

    print(f"\nAll results saved to {args.output_dir}")
    print(f"Calibration file saved to {args.calibration_file}")

if __name__ == '__main__':
    main()
