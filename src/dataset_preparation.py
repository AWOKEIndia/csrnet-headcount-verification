import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import shutil


def parse_annotations(annotation_path):
    """
    Parse annotation file containing head positions
    Format can be adapted based on your specific annotation format
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Adapt this according to your annotation format
    head_points = []
    for annotation in annotations:
        if 'points' in annotation:
            for point in annotation['points']:
                x, y = point['x'], point['y']
                head_points.append([x, y])

    return np.array(head_points)


def create_density_map_gaussian(points, height, width, sigma=15):
    """
    Generate density map based on head point annotations
    using Gaussian kernels
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            # Generate a Gaussian kernel for each point
            gaussian_kernel = np.zeros((height, width), dtype=np.float32)
            gaussian_kernel[y, x] = 1
            gaussian_kernel = cv2.GaussianBlur(gaussian_kernel, (sigma, sigma), 0)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # Normalize

            density_map += gaussian_kernel

    return density_map


def prepare_dataset(data_root, output_root, sigma=15, visualize=False):
    """
    Prepare dataset by creating density maps from annotations
    """
    # Create output directories
    os.makedirs(os.path.join(output_root, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'train', 'density_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'val', 'density_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'density_maps'), exist_ok=True)

    if visualize:
        os.makedirs(os.path.join(output_root, 'visualizations'), exist_ok=True)

    # Get all image files
    image_files = sorted(glob.glob(os.path.join(data_root, 'images', '*.jpg')))

    # Process each image
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        # Load image
        img = cv2.imread(image_file)
        height, width = img.shape[:2]

        # Get image ID from filename
        image_id = os.path.basename(image_file).split('.')[0]

        # Load annotation
        annotation_file = os.path.join(data_root, 'annotations', f'{image_id}.json')

        if os.path.exists(annotation_file):
            head_points = parse_annotations(annotation_file)

            # Create density map
            density_map = create_density_map_gaussian(head_points, height, width, sigma)

            # Check number of people
            num_people = len(head_points)
            total_density = np.sum(density_map)

            print(f"Image {image_id}: {num_people} people, Density sum: {total_density:.2f}")

            # Split into train/val/test (70/15/15 split)
            if i < int(len(image_files) * 0.7):
                subset = 'train'
            elif i < int(len(image_files) * 0.85):
                subset = 'val'
            else:
                subset = 'test'

            # Save image and density map
            image_output_path = os.path.join(output_root, subset, 'images', f'{image_id}.jpg')
            density_map_output_path = os.path.join(output_root, subset, 'density_maps', f'{image_id}.npy')

            # Copy image
            shutil.copy(image_file, image_output_path)

            # Save density map as numpy array
            np.save(density_map_output_path, density_map)

            # Visualize if needed
            if visualize:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax[0].set_title(f'Original Image ({num_people} people)')
                ax[0].axis('off')

                # Image with head points
                ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if len(head_points) > 0:
                    ax[1].scatter(head_points[:, 0], head_points[:, 1], c='red', s=3)
                ax[1].set_title('Head Points')
                ax[1].axis('off')

                # Density map
                ax[2].imshow(density_map, cmap='jet')
                ax[2].set_title(f'Density Map (Sum: {total_density:.2f})')
                ax[2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_root, 'visualizations', f'{image_id}_vis.jpg'))
                plt.close(fig)
        else:
            print(f"Warning: Annotation file {annotation_file} not found.")


def analyze_dataset(data_root):
    """
    Analyze dataset stats
    """
    train_images = glob.glob(os.path.join(data_root, 'train', 'images', '*.jpg'))
    val_images = glob.glob(os.path.join(data_root, 'val', 'images', '*.jpg'))
    test_images = glob.glob(os.path.join(data_root, 'test', 'images', '*.jpg'))

    print(f"Dataset Statistics:")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")

    # Analyze counts
    counts = []
    for subset in ['train', 'val', 'test']:
        density_maps = glob.glob(os.path.join(data_root, subset, 'density_maps', '*.npy'))

        subset_counts = []
        for density_map_file in tqdm(density_maps, desc=f"Analyzing {subset} set"):
            density_map = np.load(density_map_file)
            count = np.sum(density_map)
            subset_counts.append(count)

        if subset_counts:
            print(f"{subset} set stats:")
            print(f"  Min count: {min(subset_counts):.2f}")
            print(f"  Max count: {max(subset_counts):.2f}")
            print(f"  Mean count: {np.mean(subset_counts):.2f}")
            print(f"  Median count: {np.median(subset_counts):.2f}")

        counts.extend(subset_counts)

    # Plot count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50)
    plt.title('Distribution of Head Counts')
    plt.xlabel('Number of people')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(data_root, 'count_distribution.png'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for CSRNet')
    parser.add_argument('--data-root', type=str, required=True, help='Path to raw dataset')
    parser.add_argument('--output-root', type=str, required=True, help='Path to output processed dataset')
    parser.add_argument('--sigma', type=int, default=15, help='Sigma for Gaussian kernel (default: 15)')
    parser.add_argument('--visualize', action='store_true', help='Visualize annotations and density maps')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing processed dataset')
    args = parser.parse_args()

    if args.analyze_only:
        analyze_dataset(args.output_root)
    else:
        prepare_dataset(args.data_root, args.output_root, args.sigma, args.visualize)
        analyze_dataset(args.output_root)
