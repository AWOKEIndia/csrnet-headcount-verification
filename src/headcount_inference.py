import argparse
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import the model definition
from headcount_solution import CSRNet, get_device


def load_and_prepare_image(image_path, target_size=(384, 384)):
    """
    Load and preprocess an image for CSRNet inference

    Args:
        image_path (str): Path to image file
        target_size (tuple): Target size for model input

    Returns:
        tuple: (tensor, original_image, original_size)
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Store original size for scaling
        original_size = img.size

        # Resize image to target size
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)

        # Convert to tensor and normalize with ImageNet stats
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)

        return img_tensor, img, original_size
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None, None

def predict_headcount(model, image_path, device, target_size=(384, 384), calibration_factor=1.0):
    """
    Predict headcount from an image using CSRNet

    Args:
        model (torch.nn.Module): CSRNet model
        image_path (str): Path to image file
        device (torch.device): Device to run inference on
        target_size (tuple): Target size for model input
        calibration_factor (float): Calibration factor to adjust predictions

    Returns:
        tuple: (predicted_count, density_map, original_image)
    """
    # Load and preprocess image
    img_tensor, original_img, original_size = load_and_prepare_image(image_path, target_size)

    if img_tensor is None:
        return None, None, None

    # Move tensor to device
    img_tensor = img_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    # Get density map
    density_map = output.squeeze().cpu().numpy()

    # Clamp density map to non-negative values
    density_map = np.clip(density_map, 0, None)

    # Calculate predicted count (sum of density map)
    raw_count = float(np.sum(density_map))

    # Print sum for debugging
    print(f"[DEBUG] Density map sum (raw count): {raw_count:.2f}")

    # Apply calibration factor
    predicted_count = raw_count * calibration_factor

    # Resize density map back to original size if needed
    if original_size != target_size:
        # Calculate scale factor for preserving count during resize
        scale_factor = (original_size[0] * original_size[1]) / (target_size[0] * target_size[1])

        # Resize density map
        density_map = cv2.resize(density_map, original_size, interpolation=cv2.INTER_LINEAR)

        # Apply scale factor to preserve the count
        density_map = density_map * scale_factor

        # Verify density map sum is preserved
        if abs(np.sum(density_map) - raw_count) > 0.1:
            print(f"Warning: Density map sum changed after resize. Original: {raw_count:.2f}, New: {np.sum(density_map):.2f}")

    return predicted_count, density_map, original_img

def visualize_prediction(image, density_map, count, output_path=None, display=False):
    """
    Visualize prediction results with heatmap overlay and detailed metrics

    Args:
        image (PIL.Image): Original image
        density_map (numpy.ndarray): Predicted density map
        count (float): Predicted count
        output_path (str): Path to save visualization
        display (bool): Whether to display the visualization

    Returns:
        str: Path to saved visualization if output_path is provided
    """
    # Convert image to numpy array
    img_np = np.array(image)

    # Ensure density map matches image size for overlay
    if density_map.shape != img_np.shape[:2]:
        from cv2 import resize, INTER_LINEAR
        density_map_resized = resize(density_map, (img_np.shape[1], img_np.shape[0]), interpolation=INTER_LINEAR)
    else:
        density_map_resized = density_map

    # Create figure with 2x2 subplots
    plt.figure(figsize=(20, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')

    # Density map
    plt.subplot(2, 2, 2)
    density_plot = plt.imshow(density_map, cmap='jet')
    plt.title(f'Density Map\nPredicted Count: {count:.1f}', fontsize=12)
    plt.axis('off')
    plt.colorbar(density_plot, fraction=0.046, pad=0.04)

    # Overlay (density map on original image)
    plt.subplot(2, 2, 3)
    plt.imshow(img_np)
    plt.imshow(density_map_resized, cmap='jet', alpha=0.5)  # Use alpha=0.5 for better blending
    plt.title(f'Overlay\nPredicted Count: {count:.1f}', fontsize=12)
    plt.axis('off')

    # Density map statistics
    plt.subplot(2, 2, 4)
    plt.hist(density_map.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Density Distribution', fontsize=12)
    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Add overall title with metrics
    plt.suptitle(f'CSRNet Headcount Analysis\nPredicted Count: {count:.1f}', fontsize=14, y=0.95)

    # Save visualization
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")

    # Display if requested
    if display:
        plt.show()
    else:
        plt.close()

    return output_path if output_path else None

def process_single_image(model, image_path, device, target_size=(384, 384),
                        calibration_factor=1.0, output_dir='output/results', display=False):
    """Process a single image and visualize the results"""
    # Predict headcount
    count, density_map, original_img = predict_headcount(
        model, image_path, device, target_size, calibration_factor
    )

    if count is None:
        print(f"Failed to process {image_path}")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output visualization path
    output_path = os.path.join(output_dir, f'result_{os.path.basename(image_path)}')

    # Visualize prediction
    visualize_prediction(original_img, density_map, count, output_path, display)

    # Save results as JSON
    result = {
        'image_path': image_path,
        'predicted_count': float(count),
        'visualization_path': output_path
    }

    # Save JSON result
    json_path = os.path.splitext(output_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Image: {os.path.basename(image_path)}, Predicted Count: {count:.1f}")

    return result

def process_batch(model, image_dir, device, target_size=(384, 384),
                calibration_factor=1.0, output_dir='results', display=False):
    """Process all images in a directory and generate batch statistics"""
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*.{ext}')))

    if not image_files:
        print(f"No images found in {image_dir}")
        return []

    print(f"Processing {len(image_files)} images...")

    # Process each image
    results = []
    counts = []
    for image_path in tqdm(image_files):
        result = process_single_image(
            model, image_path, device, target_size, calibration_factor, output_dir, display
        )
        if result:
            results.append(result)
            counts.append(result['predicted_count'])

    if not results:
        print("No valid results generated")
        return []

    # Calculate batch statistics
    counts = np.array(counts)
    batch_stats = {
        'total_images': len(results),
        'average_count': float(np.mean(counts)),
        'median_count': float(np.median(counts)),
        'min_count': float(np.min(counts)),
        'max_count': float(np.max(counts)),
        'std_count': float(np.std(counts)),
        'calibration_factor': calibration_factor,
        'results': results
    }

    # Save batch results
    batch_json_path = os.path.join(output_dir, 'batch_results.json')
    with open(batch_json_path, 'w') as f:
        json.dump(batch_stats, f, indent=4)

    # Generate batch visualization
    plt.figure(figsize=(15, 10))

    # Count distribution
    plt.subplot(2, 2, 1)
    plt.hist(counts, bins=30, color='blue', alpha=0.7)
    plt.title('Count Distribution')
    plt.xlabel('Predicted Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(counts)
    plt.title('Count Statistics')
    plt.ylabel('Predicted Count')
    plt.grid(True, alpha=0.3)

    # Scatter plot of counts
    plt.subplot(2, 2, 3)
    plt.scatter(range(len(counts)), counts, alpha=0.6)
    plt.axhline(y=np.mean(counts), color='r', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    plt.title('Count Distribution Over Images')
    plt.xlabel('Image Index')
    plt.ylabel('Predicted Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = (
        f"Batch Statistics:\n\n"
        f"Total Images: {len(results)}\n"
        f"Average Count: {np.mean(counts):.1f}\n"
        f"Median Count: {np.median(counts):.1f}\n"
        f"Min Count: {np.min(counts):.1f}\n"
        f"Max Count: {np.max(counts):.1f}\n"
        f"Std Dev: {np.std(counts):.1f}\n"
        f"Calibration Factor: {calibration_factor:.4f}"
    )
    plt.text(0.1, 0.5, stats_text, fontsize=12, va='center')

    plt.suptitle('Batch Analysis Results', fontsize=14, y=0.95)

    # Save batch visualization
    batch_vis_path = os.path.join(output_dir, 'batch_analysis.png')
    plt.savefig(batch_vis_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nProcessed {len(results)} images")
    print(f"Average count: {batch_stats['average_count']:.1f}")
    print(f"Results saved to {batch_json_path}")
    print(f"Batch visualization saved to {batch_vis_path}")

    return results

def load_model(model_path, device=None):
    """Load CSRNet model from checkpoint"""
    if device is None:
        device = get_device()

    # Create model
    model = CSRNet().to(device)

    # Load weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model weights not found at {model_path}")
        print("Using ImageNet pre-trained weights only")

    return model

def calibrate_model(model, calibration_data_path, device):
    """
    Calibrate model predictions using known ground truth counts

    Args:
        model (torch.nn.Module): CSRNet model
        calibration_data_path (str): Path to calibration data JSON
        device (torch.device): Device to run inference on

    Returns:
        float: Calibration factor
    """
    if not os.path.exists(calibration_data_path):
        print(f"Calibration data not found at {calibration_data_path}")
        return 1.0

    with open(calibration_data_path, 'r') as f:
        calibration_data = json.load(f)

    if not calibration_data or 'images' not in calibration_data:
        print("Invalid calibration data format")
        return 1.0

    print(f"Calibrating model with {len(calibration_data['images'])} images...")

    ratios = []
    for item in calibration_data['images']:
        if 'image_path' in item and 'ground_truth_count' in item:
            image_path = item['image_path']
            gt_count = item['ground_truth_count']

            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue

            # Predict count with raw model
            pred_count, _, _ = predict_headcount(model, image_path, device, calibration_factor=1.0)

            if pred_count is None or pred_count == 0:
                continue

            # Calculate ratio of ground truth to prediction
            ratio = gt_count / pred_count
            ratios.append(ratio)

    if not ratios:
        print("No valid calibration data found")
        return 1.0

    # Calculate median ratio as calibration factor
    # Using median instead of mean to be robust to outliers
    calibration_factor = np.median(ratios)

    print(f"Calibration factor: {calibration_factor:.4f}")
    return calibration_factor

def main():
    parser = argparse.ArgumentParser(description='CSRNet Headcount Inference')

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='path to single image')
    group.add_argument('--dir', type=str, help='path to directory of images')

    # Model options
    parser.add_argument('--model', type=str, default='models/csrnet.pth', help='path to model')
    parser.add_argument('--target_size', type=int, nargs=2, default=[384, 384], help='target input size (width, height)')

    # Calibration options
    parser.add_argument('--calibration_data', type=str, help='path to calibration data JSON')
    parser.add_argument('--calibration_factor', type=float, default=1.0, help='manual calibration factor')

    # Output options
    parser.add_argument('--output_dir', type=str, default='output/results', help='output directory')
    parser.add_argument('--display', action='store_true', help='display results (requires GUI)')

    args = parser.parse_args()

    # Get device
    device = get_device()

    # Load model
    model = load_model(args.model, device)

    # Determine calibration factor
    calibration_factor = args.calibration_factor
    if args.calibration_data:
        calibration_factor = calibrate_model(model, args.calibration_data, device)

    # Process images
    if args.image:
        # Process single image
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return

        process_single_image(
            model, args.image, device,
            target_size=tuple(args.target_size),
            calibration_factor=calibration_factor,
            output_dir=args.output_dir,
            display=args.display
        )
    else:
        # Process directory of images
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return

        process_batch(
            model, args.dir, device,
            target_size=tuple(args.target_size),
            calibration_factor=calibration_factor,
            output_dir=args.output_dir,
            display=args.display
        )

if __name__ == '__main__':
    main()
