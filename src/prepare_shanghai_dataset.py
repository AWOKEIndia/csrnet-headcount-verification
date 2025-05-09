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
import torch
import torch.nn.functional as F
from torchvision import transforms
import json

def get_device():
    """
    Get the appropriate device for processing
    Prioritizes Apple Silicon GPU (MPS) if available, then CUDA, then CPU
    """
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def create_density_map_gaussian(points, height, width, sigma=15):
    """
    Generate density map based on head point annotations
    using Gaussian kernels with proper normalization for headcount

    Args:
        points (numpy.ndarray): Array of head point coordinates [x, y]
        height (int): Height of the output density map
        width (int): Width of the output density map
        sigma (int): Sigma for Gaussian kernel

    Returns:
        numpy.ndarray: Generated density map that integrates to the point count
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    # If no points, return empty density map
    if len(points) == 0:
        return density_map

    # Generate density map with fixed sigma
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            # Create small Gaussian kernel centered at point
            # This is more efficient than creating full-size kernels
            kernel_size = max(1, int(sigma * 6)) // 2 * 2 + 1  # Ensure odd size
            kernel_radius = kernel_size // 2

            # Create coordinates for kernel
            x_left, x_right = max(0, x - kernel_radius), min(width, x + kernel_radius + 1)
            y_top, y_bottom = max(0, y - kernel_radius), min(height, y + kernel_radius + 1)

            # Get actual kernel dimensions
            kernel_width = x_right - x_left
            kernel_height = y_bottom - y_top

            # Create coordinate meshgrid for Gaussian
            mesh_x = np.arange(x_left, x_right)
            mesh_y = np.arange(y_top, y_bottom)
            xx, yy = np.meshgrid(mesh_x, mesh_y)

            # Generate Gaussian
            gaussian_kernel = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

            # Make sure kernel preserves person count (integrates to 1)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

            # Add to density map
            density_map[y_top:y_bottom, x_left:x_right] += gaussian_kernel

    # Scale density map to match head count
    density_map = density_map * len(points)

    return density_map

def create_density_map_adaptive(points, height, width, k=3):
    """
    Generate density map with adaptive Gaussian kernels
    based on distances between points

    Args:
        points (numpy.ndarray): Array of head point coordinates [x, y]
        height (int): Height of the output density map
        width (int): Width of the output density map
        k (int): Number of nearest neighbors to consider

    Returns:
        numpy.ndarray: Generated density map
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    # If no points or too few points, use fixed sigma
    if len(points) <= k + 1:
        return create_density_map_gaussian(points, height, width)

    # Calculate average distance to k nearest neighbors for each point
    from scipy.spatial import KDTree
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k+1)  # +1 because first neighbor is the point itself

    # Generate density map with adaptive sigma
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            # Calculate adaptive sigma based on average distance to neighbors
            # Skip first distance (distance to self is 0)
            avg_distance = np.mean(distances[i][1:])
            sigma = max(3, avg_distance * 0.3)  # Lower bound of sigma

            # Calculate kernel size based on sigma (must be odd)
            kernel_size = max(1, int(sigma * 6)) // 2 * 2 + 1  # Ensure odd size
            kernel_radius = kernel_size // 2

            # Create coordinates for kernel
            x_left, x_right = max(0, x - kernel_radius), min(width, x + kernel_radius + 1)
            y_top, y_bottom = max(0, y - kernel_radius), min(height, y + kernel_radius + 1)

            # Create coordinate meshgrid for Gaussian
            mesh_x = np.arange(x_left, x_right)
            mesh_y = np.arange(y_top, y_bottom)
            xx, yy = np.meshgrid(mesh_x, mesh_y)

            # Generate Gaussian
            gaussian_kernel = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

            # Make sure kernel preserves person count (integrates to 1)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

            # Add to density map
            density_map[y_top:y_bottom, x_left:x_right] += gaussian_kernel

    # Scale density map to match head count
    density_map = density_map * len(points)

    return density_map

def gpu_create_density_map(points, height, width, sigma=15, device=None):
    """
    Create density map using GPU acceleration with proper normalization for headcount

    Args:
        points (numpy.ndarray): Array of head point coordinates [x, y]
        height (int): Height of the output density map
        width (int): Width of the output density map
        sigma (int): Sigma for Gaussian kernel
        device (torch.device): Device to use for computation

    Returns:
        numpy.ndarray: Generated density map that integrates to point count
    """
    if device is None:
        device = get_device()

    # If no points, return empty density map
    if len(points) == 0:
        return np.zeros((height, width), dtype=np.float32)

    # Convert points to tensor
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

    # Initialize density map
    density_map = torch.zeros((height, width), device=device)

    # Process points in batches to avoid memory issues
    batch_size = min(100, len(points))

    for idx in range(0, len(points), batch_size):
        batch_points = points_tensor[idx:idx + batch_size]

        for point in batch_points:
            x, y = point[0], point[1]

            if 0 <= x < width and 0 <= y < height:
                # Calculate kernel region - only generate Gaussian in a local region around each point
                kernel_size = int(sigma * 6) // 2 * 2 + 1  # Ensure odd size
                kernel_radius = kernel_size // 2

                # Determine local region bounds
                x_left = max(0, int(x) - kernel_radius)
                x_right = min(width, int(x) + kernel_radius + 1)
                y_top = max(0, int(y) - kernel_radius)
                y_bottom = min(height, int(y) + kernel_radius + 1)

                # Skip if point is too close to the edge
                if x_right <= x_left or y_bottom <= y_top:
                    continue

                # Create local coordinate grid
                local_h = y_bottom - y_top
                local_w = x_right - x_left

                y_coords = torch.arange(y_top, y_bottom, device=device).view(-1, 1).expand(-1, local_w)
                x_coords = torch.arange(x_left, x_right, device=device).view(1, -1).expand(local_h, -1)

                # Generate local Gaussian kernel
                gaussian = torch.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))

                # Normalize kernel to preserve count
                if gaussian.sum() > 0:
                    gaussian = gaussian / gaussian.sum()

                    # Add to density map
                    density_map[y_top:y_bottom, x_left:x_right] += gaussian

    # Scale to match point count
    density_map = density_map * len(points)

    return density_map.cpu().numpy()

def prepare_shanghai_tech(dataset_path, output_path, part='A', use_adaptive=False, visualize=False, use_gpu=True, target_size=None):
    """
    Prepare ShanghaiTech dataset for CSRNet with enhanced density map generation

    Args:
        dataset_path (str): Path to ShanghaiTech dataset directory
        output_path (str): Path to output processed dataset
        part (str): Dataset part, 'A' or 'B'
        use_adaptive (bool): Whether to use adaptive Gaussian kernels
        visualize (bool): Whether to visualize density maps
        use_gpu (bool): Whether to use GPU acceleration if available
        target_size (tuple): Optional target size for resizing images (width, height)

    Returns:
        dict: Dataset statistics
    """
    print(f"Preparing ShanghaiTech Part {part} dataset...")

    # Set up device
    device = get_device() if use_gpu else torch.device('cpu')
    print(f"Using device: {device}")

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

        # Create density map using appropriate method
        if use_gpu and device.type != 'cpu':
            # Use adaptive sigma based on crowd density
            if use_adaptive:
                sigma = max(4, min(8, len(points) / 150 * 4)) if len(points) > 0 else 4
            else:
                sigma = 15  # Default sigma

            # The GPU density map generation could be memory intensive for large images
            # Consider adding a warning for large images:
            if height * width > 1000000:  # 1M pixels
                print("Warning: Large image size may cause memory issues")

            density_map = gpu_create_density_map(points, height, width, sigma=sigma, device=device)
        else:
            if use_adaptive:
                density_map = create_density_map_adaptive(points, height, width, k=3)
            else:
                density_map = create_density_map_gaussian(points, height, width, sigma=15)

        # Verify density map sum is close to people count
        dm_sum = np.sum(density_map)
        if abs(dm_sum - len(points)) > 1.0:
            print(f"Warning: Density map sum ({dm_sum:.2f}) does not match people count ({len(points)})")
            # Rescale density map to match count
            if dm_sum > 0:
                density_map = density_map * (len(points) / dm_sum)

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
            plt.title(f'Density Map (Sum: {np.sum(density_map):.2f})')
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

        # Create density map using appropriate method
        if use_gpu and device.type != 'cpu':
            # Use adaptive sigma based on crowd density
            if use_adaptive:
                sigma = max(4, min(8, len(points) / 150 * 4)) if len(points) > 0 else 4
            else:
                sigma = 15  # Default sigma

            # The GPU density map generation could be memory intensive for large images
            # Consider adding a warning for large images:
            if height * width > 1000000:  # 1M pixels
                print("Warning: Large image size may cause memory issues")

            density_map = gpu_create_density_map(points, height, width, sigma=sigma, device=device)
        else:
            if use_adaptive:
                density_map = create_density_map_adaptive(points, height, width, k=3)
            else:
                density_map = create_density_map_gaussian(points, height, width, sigma=15)

        # Verify density map sum is close to people count
        dm_sum = np.sum(density_map)
        if abs(dm_sum - len(points)) > 1.0:
            print(f"Warning: Density map sum ({dm_sum:.2f}) does not match people count ({len(points)})")
            # Rescale density map to match count
            if dm_sum > 0:
                density_map = density_map * (len(points) / dm_sum)

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
            plt.title(f'Density Map (Sum: {np.sum(density_map):.2f})')
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
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive Gaussian kernels')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                        help='Optional target size for resizing images (width height)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize density maps')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU acceleration if available')
    args = parser.parse_args()

    prepare_shanghai_tech(
        args.dataset_path,
        args.output_path,
        args.part,
        args.adaptive,
        args.visualize,
        args.use_gpu,
        args.target_size
    )
