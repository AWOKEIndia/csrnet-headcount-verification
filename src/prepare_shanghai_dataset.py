import os
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
import glob
import argparse
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import create_density_map_gaussian, create_density_map_adaptive


def prepare_shanghai_tech(dataset_path, output_path, part='A', use_adaptive=False, visualize=False):
    """
    Prepare ShanghaiTech dataset for CSRNet

    Args:
        dataset_path (str): Path to ShanghaiTech dataset directory
        output_path (str): Path to output processed dataset
        part (str): Dataset part, 'A' or 'B'
        use_adaptive (bool): Whether to use adaptive Gaussian kernels
        visualize (bool): Whether to visualize density maps

    Returns:
        None
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

        # Create density map
        if use_adaptive:
            density_map = create_density_map_adaptive(points, height, width, k=3)
        else:
            density_map = create_density_map_gaussian(points, height, width, sigma=4)

        # Save image and density map based on split
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

        shutil.copy(str(img_path), str(output_img_path))
        np.save(str(output_den_path), density_map)

        if visualize:
            # Visualize density map
            plt.figure(figsize=(12, 4))

            # Original image with points
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.scatter(points[:, 0], points[:, 1], c='red', s=1)
            plt.title(f'Image with {len(points)} points')
            plt.axis('off')

            # Density map
            plt.subplot(1, 3, 2)
            plt.imshow(density_map, cmap='jet')
            plt.title(f'Density Map (Sum: {np.sum(density_map):.2f})')
            plt.axis('off')
            plt.colorbar()

            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(density_map, cmap='jet', alpha=0.5)
            plt.title('Overlay')
            plt.axis('off')

            plt.suptitle(f'Part {part} - {subset} - {img_id}')
            plt.tight_layout()
            plt.savefig(str(vis_path / f'{subset}_{img_id}_vis.jpg'))
            plt.close()

    # Process test data
    print("Processing test data...")
    test_images = list(test_images_path.glob('*.jpg'))

    if not test_images:
        raise FileNotFoundError(f"No test images found in {test_images_path}")

    for img_path in tqdm(test_images):
        img_name = img_path.name
        img_id = img_path.stem

        # Load image
        img = cv2.imread(str(img_path))
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

        # Create density map
        if use_adaptive:
            density_map = create_density_map_adaptive(points, height, width, k=3)
        else:
            density_map = create_density_map_gaussian(points, height, width, sigma=4)

        # Save image and density map
        output_img_path = test_output_img_path / img_name
        output_den_path = test_output_den_path / f'{img_id}.npy'

        shutil.copy(str(img_path), str(output_img_path))
        np.save(str(output_den_path), density_map)

        if visualize:
            # Visualize density map
            plt.figure(figsize=(12, 4))

            # Original image with points
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.scatter(points[:, 0], points[:, 1], c='red', s=1)
            plt.title(f'Image with {len(points)} points')
            plt.axis('off')

            # Density map
            plt.subplot(1, 3, 2)
            plt.imshow(density_map, cmap='jet')
            plt.title(f'Density Map (Sum: {np.sum(density_map):.2f})')
            plt.axis('off')
            plt.colorbar()

            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(density_map, cmap='jet', alpha=0.5)
            plt.title('Overlay')
            plt.axis('off')

            plt.suptitle(f'Part {part} - test - {img_id}')
            plt.tight_layout()
            plt.savefig(str(vis_path / f'test_{img_id}_vis.jpg'))
            plt.close()

    print(f"Processed {len(train_images)} train images and {len(test_images)} test images.")

    # Count files in each directory
    train_count = len(list(train_output_img_path.glob('*.jpg')))
    val_count = len(list(val_output_img_path.glob('*.jpg')))
    test_count = len(list(test_output_img_path.glob('*.jpg')))

    print(f"Files saved to {output_path}:")
    print(f"  Train: {train_count} images (70%)")
    print(f"  Validation: {val_count} images (15%)")
    print(f"  Test: {test_count} images (15%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare ShanghaiTech dataset for CSRNet')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to ShanghaiTech dataset directory')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to output processed dataset')
    parser.add_argument('--part', type=str, default='A', choices=['A', 'B'],
                        help='Dataset part, A or B')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive Gaussian kernels')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize density maps')
    args = parser.parse_args()

    prepare_shanghai_tech(args.dataset_path, args.output_path, args.part, args.adaptive, args.visualize)
