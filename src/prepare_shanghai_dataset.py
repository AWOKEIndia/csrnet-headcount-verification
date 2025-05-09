import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

def create_density_map(points, height, width, sigma=15):
    """
    Generate density map using basic Gaussian filter

    Args:
        points (numpy.ndarray): Array of head point coordinates [x, y]
        height (int): Height of the output density map
        width (int): Width of the output density map
        sigma (int): Sigma for Gaussian kernel

    Returns:
        numpy.ndarray: Generated density map
    """
    # Create binary map of points
    density_map = np.zeros((height, width), dtype=np.float32)

    # If no points, return empty density map
    if len(points) == 0:
        return density_map

    # Mark points in the density map
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            density_map[y, x] = 1

    # Apply Gaussian filter
    density_map = gaussian_filter(density_map, sigma)

    # Normalize to preserve count
    if np.sum(density_map) > 0:
        density_map = density_map * (len(points) / np.sum(density_map))

    return density_map

def prepare_shanghai_tech(dataset_path, output_path, part='A', visualize=False, target_size=None):
    """
    Prepare ShanghaiTech dataset for CSRNet with basic Gaussian filter density map generation

    Args:
        dataset_path (str): Path to ShanghaiTech dataset directory
        output_path (str): Path to output processed dataset
        part (str): Dataset part, 'A' or 'B'
        visualize (bool): Whether to visualize density maps
        target_size (tuple): Optional target size for resizing images (width, height)

    Returns:
        dict: Dataset statistics
    """
    print(f"Preparing ShanghaiTech Part {part} dataset...")

    # Convert to Path objects for better cross-platform compatibility
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Dataset path (absolute): {dataset_path.absolute()}")
    print(f"Output path (absolute): {output_path.absolute()}")

    # Paths
    part_path = dataset_path / f'ShanghaiTech_Part{part}'
    train_images_path = part_path / 'train_data' / 'images'
    train_gt_path = part_path / 'train_data' / 'ground-truth'
    test_images_path = part_path / 'test_data' / 'images'
    test_gt_path = part_path / 'test_data' / 'ground-truth'

    # Check if directories exist
    if not train_gt_path.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {train_gt_path}")

    # Create output directories
    train_output_img_path = output_path / 'train' / 'images'
    train_output_den_path = output_path / 'train' / 'density_maps'
    test_output_img_path = output_path / 'test' / 'images'
    test_output_den_path = output_path / 'test' / 'density_maps'
    val_output_img_path = output_path / 'val' / 'images'
    val_output_den_path = output_path / 'val' / 'density_maps'

    os.makedirs(train_output_img_path, exist_ok=True)
    os.makedirs(train_output_den_path, exist_ok=True)
    os.makedirs(test_output_img_path, exist_ok=True)
    os.makedirs(test_output_den_path, exist_ok=True)
    os.makedirs(val_output_img_path, exist_ok=True)
    os.makedirs(val_output_den_path, exist_ok=True)

    if visualize:
        vis_path = output_path / 'visualizations'
        os.makedirs(vis_path, exist_ok=True)

    # Initialize dataset statistics
    dataset_stats = {
        'train': {'count': 0, 'total_people': 0, 'min_count': float('inf'), 'max_count': 0},
        'val': {'count': 0, 'total_people': 0, 'min_count': float('inf'), 'max_count': 0},
        'test': {'count': 0, 'total_people': 0, 'min_count': float('inf'), 'max_count': 0},
    }

    # Track ground truth data for calibration
    calibration_data = {'images': []}

    # Process train data
    print("Processing train data...")
    train_images = list(train_images_path.glob('*.jpg'))

    if not train_images:
        raise FileNotFoundError(f"No training images found in {train_images_path}")

    # Shuffle the training images for consistent split
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(train_images)

    # Calculate split indices for 70/15/15 split
    n_train = len(train_images)
    n_train_split = int(0.7 * n_train)
    n_val_split = int(0.15 * n_train)

    for i, img_path in enumerate(tqdm(train_images)):
        img_name = img_path.name
        img_id = img_path.stem

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        height, width = img.shape[:2]

        # Load ground truth
        gt_path = train_gt_path / f'GT_{img_id}.mat'

        if not gt_path.exists():
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue

        try:
            mat_data = sio.loadmat(str(gt_path))
        except Exception as e:
            print(f"Error loading MAT file {gt_path}: {e}")
            continue

        # Extract points
        if 'image_info' in mat_data:
            # ShanghaiTech format
            points = mat_data['image_info'][0, 0]['location'][0, 0]
        else:
            raise ValueError(f"Unexpected MAT file format: {gt_path}")

        # Determine output paths based on split
        if i < n_train_split:
            output_img_path = train_output_img_path / img_name
            output_den_path = train_output_den_path / f'{img_id}.npy'
            subset = 'train'
        elif i < n_train_split + n_val_split:
            output_img_path = val_output_img_path / img_name
            output_den_path = val_output_den_path / f'{img_id}.npy'
            subset = 'val'
        else:
            output_img_path = test_output_img_path / img_name
            output_den_path = test_output_den_path / f'{img_id}.npy'
            subset = 'test'

        # Resize image if target size is specified
        if target_size is not None:
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError("target_size must be a tuple of two positive integers")
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.resize(target_size, Image.BILINEAR)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Scale points accordingly
            scale_x = target_size[0] / width
            scale_y = target_size[1] / height
            points[:, 0] = points[:, 0] * scale_x
            points[:, 1] = points[:, 1] * scale_y

            height, width = target_size[1], target_size[0]

        # Create density map using basic Gaussian filter
        density_map = create_density_map(points, height, width, sigma=15)

        # Save image and density map
        try:
            cv2.imwrite(str(output_img_path), img)
        except Exception as e:
            print(f"Error saving image {output_img_path}: {e}")
            continue
        np.save(str(output_den_path), density_map)

        # Update dataset statistics
        dataset_stats[subset]['count'] += 1
        dataset_stats[subset]['total_people'] += len(points)
        dataset_stats[subset]['min_count'] = min(dataset_stats[subset]['min_count'], len(points))
        dataset_stats[subset]['max_count'] = max(dataset_stats[subset]['max_count'], len(points))

        # Add to calibration data
        calibration_data['images'].append({
            'image_path': str(output_img_path),
            'ground_truth_count': float(len(points))
        })

        if visualize:
            # Visualize density map
            plt.figure(figsize=(15, 5))

            # Original image with points
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.scatter(points[:, 0], points[:, 1], c='red', s=2)
            plt.title(f'Image with {len(points)} people')
            plt.axis('off')

            # Density map
            plt.subplot(1, 3, 2)
            plt.imshow(density_map, cmap='jet')
            plt.title(f'Density Map (Count: {len(points)}, Sum: {np.sum(density_map):.2f})')
            plt.axis('off')
            plt.colorbar()

            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(density_map, cmap='jet', alpha=0.5)
            plt.title('Overlay')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(str(vis_path / f'{subset}_{img_id}_vis.jpg'), dpi=150)
            plt.close()

    # Process test data
    print("Processing test data...")
    test_images = list(test_images_path.glob('*.jpg'))

    if not test_images:
        print(f"Warning: No test images found in {test_images_path}")
        test_images = []

    for img_path in tqdm(test_images):
        img_name = img_path.name
        img_id = img_path.stem

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        height, width = img.shape[:2]

        # Load ground truth
        gt_path = test_gt_path / f'GT_{img_id}.mat'

        if not gt_path.exists():
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue

        try:
            mat_data = sio.loadmat(str(gt_path))
        except Exception as e:
            print(f"Error loading MAT file {gt_path}: {e}")
            continue

        # Extract points
        if 'image_info' in mat_data:
            # ShanghaiTech format
            points = mat_data['image_info'][0, 0]['location'][0, 0]
        else:
            raise ValueError(f"Unexpected MAT file format: {gt_path}")

        # Determine output paths
        output_img_path = test_output_img_path / img_name
        output_den_path = test_output_den_path / f'{img_id}.npy'

        # Resize image if target size is specified
        if target_size is not None:
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError("target_size must be a tuple of two positive integers")
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.resize(target_size, Image.BILINEAR)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Scale points accordingly
            scale_x = target_size[0] / width
            scale_y = target_size[1] / height
            points[:, 0] = points[:, 0] * scale_x
            points[:, 1] = points[:, 1] * scale_y

            height, width = target_size[1], target_size[0]

        # Create density map using basic Gaussian filter
        density_map = create_density_map(points, height, width, sigma=15)

        # Save image and density map
        try:
            cv2.imwrite(str(output_img_path), img)
        except Exception as e:
            print(f"Error saving image {output_img_path}: {e}")
            continue
        np.save(str(output_den_path), density_map)

        # Update dataset statistics
        dataset_stats['test']['count'] += 1
        dataset_stats['test']['total_people'] += len(points)
        dataset_stats['test']['min_count'] = min(dataset_stats['test']['min_count'], len(points))
        dataset_stats['test']['max_count'] = max(dataset_stats['test']['max_count'], len(points))

        # Add to calibration data
        calibration_data['images'].append({
            'image_path': str(output_img_path),
            'ground_truth_count': float(len(points))
        })

        if visualize:
            # Visualize density map
            plt.figure(figsize=(15, 5))

            # Original image with points
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.scatter(points[:, 0], points[:, 1], c='red', s=2)
            plt.title(f'Image with {len(points)} people')
            plt.axis('off')

            # Density map
            plt.subplot(1, 3, 2)
            plt.imshow(density_map, cmap='jet')
            plt.title(f'Density Map (Count: {len(points)}, Sum: {np.sum(density_map):.2f})')
            plt.axis('off')
            plt.colorbar()

            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(density_map, cmap='jet', alpha=0.5)
            plt.title('Overlay')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(str(vis_path / f'test_{img_id}_vis.jpg'), dpi=150)
            plt.close()

    # Calculate average statistics
    for subset in ['train', 'val', 'test']:
        if dataset_stats[subset]['count'] > 0:
            dataset_stats[subset]['avg_count'] = dataset_stats[subset]['total_people'] / dataset_stats[subset]['count']
        else:
            dataset_stats[subset]['avg_count'] = 0

    # Save dataset statistics
    dataset_stats_path = output_path / 'dataset_stats.json'
    with open(dataset_stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=4)

    # Save calibration data
    calibration_path = output_path / 'calibration_data.json'
    with open(calibration_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    # Print summary
    print(f"\nDataset statistics:")
    for subset in ['train', 'val', 'test']:
        if dataset_stats[subset]['count'] > 0:
            print(f"  {subset.capitalize()}: {dataset_stats[subset]['count']} images, " +
                  f"{dataset_stats[subset]['total_people']} people, " +
                  f"avg: {dataset_stats[subset]['avg_count']:.1f}, " +
                  f"min: {dataset_stats[subset]['min_count']}, " +
                  f"max: {dataset_stats[subset]['max_count']}")

    print(f"\nFiles saved to {output_path}")
    print(f"Dataset statistics saved to {dataset_stats_path}")
    print(f"Calibration data saved to {calibration_path}")

    return dataset_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare ShanghaiTech dataset for CSRNet')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to ShanghaiTech dataset directory')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to output processed dataset')
    parser.add_argument('--part', type=str, default='A', choices=['A', 'B'],
                        help='Dataset part, A or B')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                        help='Optional target size for resizing images (width height)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize density maps')
    args = parser.parse_args()

    prepare_shanghai_tech(
        args.dataset_path,
        args.output_path,
        args.part,
        args.visualize,
        args.target_size
    )
