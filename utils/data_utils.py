import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import shutil
import scipy.io as sio
import pandas as pd


def parse_annotations_json(annotation_path):
    """
    Parse annotations from JSON file

    Args:
        annotation_path (str): Path to annotation file

    Returns:
        numpy.ndarray: Array of head point coordinates [x, y]
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    head_points = []

    # Adapt this based on your JSON format
    if isinstance(annotations, list):
        # Format: list of annotations
        for annotation in annotations:
            if 'points' in annotation:
                for point in annotation['points']:
                    x, y = point['x'], point['y']
                    head_points.append([x, y])
    elif isinstance(annotations, dict):
        # Format: dictionary of annotations
        if 'points' in annotations:
            for point in annotations['points']:
                x, y = point['x'], point['y']
                head_points.append([x, y])
        elif 'annotations' in annotations:
            for annotation in annotations['annotations']:
                if 'point' in annotation:
                    x, y = annotation['point']['x'], annotation['point']['y']
                    head_points.append([x, y])

    return np.array(head_points)


def parse_annotations_mat(annotation_path):
    """
    Parse annotations from MATLAB .mat file

    Args:
        annotation_path (str): Path to annotation file

    Returns:
        numpy.ndarray: Array of head point coordinates [x, y]
    """
    mat_data = sio.loadmat(annotation_path)

    # Adapt this based on your .mat file structure
    if 'image_info' in mat_data:
        # ShanghaiTech format
        if 'location' in mat_data['image_info'][0, 0]:
            points = mat_data['image_info'][0, 0]['location'][0, 0]
            return points.astype(np.float32)
    elif 'annPoints' in mat_data:
        # UCF-QNRF format
        return mat_data['annPoints'].astype(np.float32)

    # Generic format - try common field names
    for field in ['points', 'annotation', 'coordinates', 'positions']:
        if field in mat_data:
            return mat_data[field].astype(np.float32)

    print(f"Warning: Could not parse annotations from {annotation_path}")
    return np.empty((0, 2), dtype=np.float32)


def create_density_map_gaussian(points, height, width, sigma=15):
    """
    Generate density map based on head point annotations
    using Gaussian kernels

    Args:
        points (numpy.ndarray): Array of head point coordinates [x, y]
        height (int): Height of the output density map
        width (int): Width of the output density map
        sigma (int): Sigma for Gaussian kernel

    Returns:
        numpy.ndarray: Generated density map
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    # If no points, return empty density map
    if len(points) == 0:
        return density_map

    # Make sure sigma is valid
    if sigma <= 0:
        sigma = 15

    # Calculate kernel size based on sigma (must be odd)
    kernel_size = max(1, int(sigma * 3)) # 3 sigma rule
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make it odd

    # Generate density map with fixed sigma
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            # Generate a Gaussian kernel for each point
            gaussian_kernel = np.zeros((height, width), dtype=np.float32)
            gaussian_kernel[y, x] = 1
            gaussian_kernel = cv2.GaussianBlur(gaussian_kernel, (kernel_size, kernel_size), sigma)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # Normalize

            density_map += gaussian_kernel

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
            kernel_size = max(1, int(sigma * 3))  # 3 sigma rule
            if kernel_size % 2 == 0:
                kernel_size += 1  # Make it odd

            # Generate a Gaussian kernel with adaptive sigma
            gaussian_kernel = np.zeros((height, width), dtype=np.float32)
            gaussian_kernel[y, x] = 1

            gaussian_kernel = cv2.GaussianBlur(gaussian_kernel, (kernel_size, kernel_size), sigma)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # Normalize

            density_map += gaussian_kernel

    return density_map


def load_image_and_density_map(image_path, density_map_path):
    """
    Load image and corresponding density map

    Args:
        image_path (str): Path to image file
        density_map_path (str): Path to density map file

    Returns:
        tuple: (image, density_map)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Load density map
    if density_map_path.endswith('.npy'):
        density_map = np.load(density_map_path)
    elif density_map_path.endswith('.mat'):
        mat_data = sio.loadmat(density_map_path)
        # Adapt this based on your .mat file structure
        if 'density' in mat_data:
            density_map = mat_data['density']
        else:
            # Try other common field names
            for field in ['density_map', 'densityMap', 'map']:
                if field in mat_data:
                    density_map = mat_data[field]
                    break
            else:
                raise ValueError(f"Could not find density map in {density_map_path}")
    else:
        raise ValueError(f"Unsupported density map format: {density_map_path}")

    return image, density_map


def split_dataset(data_root, output_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train, validation, and test sets

    Args:
        data_root (str): Path to raw dataset
        output_root (str): Path to output processed dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Set random seed
    random.seed(random_seed)

    # Get all image files
    image_files = sorted(glob.glob(os.path.join(data_root, 'images', '*.jpg')))
    image_files.extend(sorted(glob.glob(os.path.join(data_root, 'images', '*.png'))))

    # Shuffle files
    random.shuffle(image_files)

    # Split into train, val, test
    n_train = int(len(image_files) * train_ratio)
    n_val = int(len(image_files) * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]

    # Create output directories
    os.makedirs(os.path.join(output_root, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'train', 'density_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'val', 'density_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'density_maps'), exist_ok=True)

    return train_files, val_files, test_files


class CrowdDataset(Dataset):
    """
    Dataset class for crowd counting
    """
    def __init__(self, image_root, density_map_root, transform=None):
        """
        Initialize dataset

        Args:
            image_root (str): Path to images directory
            density_map_root (str): Path to density maps directory
            transform (callable, optional): Transform to apply to images
        """
        self.image_root = image_root
        self.density_map_root = density_map_root
        self.transform = transform

        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
        self.image_files.extend(sorted(glob.glob(os.path.join(image_root, '*.png'))))

        # Get corresponding density map files
        self.density_map_files = []
        for image_file in self.image_files:
            image_id = os.path.splitext(os.path.basename(image_file))[0]

            # Try both .npy and .mat formats
            density_map_file_npy = os.path.join(density_map_root, f"{image_id}.npy")
            density_map_file_mat = os.path.join(density_map_root, f"{image_id}.mat")

            if os.path.exists(density_map_file_npy):
                self.density_map_files.append(density_map_file_npy)
            elif os.path.exists(density_map_file_mat):
                self.density_map_files.append(density_map_file_mat)
            else:
                raise ValueError(f"Could not find density map for {image_id}")

        # Sanity check
        assert len(self.image_files) == len(self.density_map_files), "Number of images and density maps don't match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_files[idx]).convert('RGB')

        # Load density map
        if self.density_map_files[idx].endswith('.npy'):
            density_map = np.load(self.density_map_files[idx])
        elif self.density_map_files[idx].endswith('.mat'):
            mat_data = sio.loadmat(self.density_map_files[idx])
            if 'density' in mat_data:
                density_map = mat_data['density']
            else:
                # Try other common field names
                for field in ['density_map', 'densityMap', 'map']:
                    if field in mat_data:
                        density_map = mat_data[field]
                        break
                else:
                    raise ValueError(f"Could not find density map in {self.density_map_files[idx]}")
        else:
            raise ValueError(f"Unsupported density map format: {self.density_map_files[idx]}")

        # Apply transform to image if specified
        if self.transform is not None:
            image = self.transform(image)

        # Convert density map to tensor
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)

        return image, density_map


def analyze_dataset_stats(dataset_path):
    """
    Analyze dataset statistics

    Args:
        dataset_path (str): Path to processed dataset

    Returns:
        dict: Dictionary containing dataset statistics
    """
    stats = {}

    # Count samples
    train_images = glob.glob(os.path.join(dataset_path, 'train', 'images', '*.jpg'))
    train_images.extend(glob.glob(os.path.join(dataset_path, 'train', 'images', '*.png')))

    val_images = glob.glob(os.path.join(dataset_path, 'val', 'images', '*.jpg'))
    val_images.extend(glob.glob(os.path.join(dataset_path, 'val', 'images', '*.png')))

    test_images = glob.glob(os.path.join(dataset_path, 'test', 'images', '*.jpg'))
    test_images.extend(glob.glob(os.path.join(dataset_path, 'test', 'images', '*.png')))

    stats['num_train'] = len(train_images)
    stats['num_val'] = len(val_images)
    stats['num_test'] = len(test_images)
    stats['total_samples'] = stats['num_train'] + stats['num_val'] + stats['num_test']

    # Analyze counts
    train_counts = []
    val_counts = []
    test_counts = []

    for subset, counts_list in [('train', train_counts), ('val', val_counts), ('test', test_counts)]:
        density_maps = glob.glob(os.path.join(dataset_path, subset, 'density_maps', '*.npy'))
        density_maps.extend(glob.glob(os.path.join(dataset_path, subset, 'density_maps', '*.mat')))

        for density_map_file in density_maps:
            if density_map_file.endswith('.npy'):
                density_map = np.load(density_map_file)
            elif density_map_file.endswith('.mat'):
                mat_data = sio.loadmat(density_map_file)
                if 'density' in mat_data:
                    density_map = mat_data['density']
                else:
                    # Try other common field names
                    for field in ['density_map', 'densityMap', 'map']:
                        if field in mat_data:
                            density_map = mat_data[field]
                            break
                    else:
                        print(f"Warning: Could not find density map in {density_map_file}")
                        continue

            count = np.sum(density_map)
            counts_list.append(count)

    # Calculate statistics
    stats['train_count_min'] = min(train_counts) if train_counts else 0
    stats['train_count_max'] = max(train_counts) if train_counts else 0
    stats['train_count_mean'] = np.mean(train_counts) if train_counts else 0
    stats['train_count_median'] = np.median(train_counts) if train_counts else 0

    stats['val_count_min'] = min(val_counts) if val_counts else 0
    stats['val_count_max'] = max(val_counts) if val_counts else 0
    stats['val_count_mean'] = np.mean(val_counts) if val_counts else 0
    stats['val_count_median'] = np.median(val_counts) if val_counts else 0

    stats['test_count_min'] = min(test_counts) if test_counts else 0
    stats['test_count_max'] = max(test_counts) if test_counts else 0
    stats['test_count_mean'] = np.mean(test_counts) if test_counts else 0
    stats['test_count_median'] = np.median(test_counts) if test_counts else 0

    # Combine all counts
    all_counts = train_counts + val_counts + test_counts
    stats['total_count_min'] = min(all_counts) if all_counts else 0
    stats['total_count_max'] = max(all_counts) if all_counts else 0
    stats['total_count_mean'] = np.mean(all_counts) if all_counts else 0
    stats['total_count_median'] = np.median(all_counts) if all_counts else 0

    return stats


def export_dataset_stats(stats, output_path):
    """
    Export dataset statistics to CSV and JSON files

    Args:
        stats (dict): Dictionary containing dataset statistics
        output_path (str): Path to save the statistics

    Returns:
        None
    """
    # Save to JSON
    with open(os.path.join(output_path, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    # Save to CSV
    df = pd.DataFrame([stats])
    df.to_csv(os.path.join(output_path, 'dataset_stats.csv'), index=False)

    print(f"Dataset statistics saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data utilities for CSRNet Headcount Verification System")
