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

    # Calculate predicted count (sum of density map)
    raw_count = float(np.sum(density_map))

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
    Visualize prediction results with heatmap overlay

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

    # Create figure
    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

    # Density map
    plt.subplot(1, 3, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title(f'Density Map\nPredicted Count: {count:.1f}')
    plt.axis('off')
    plt.colorbar()

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(density_map, cmap='jet', alpha=0.6)
    plt.title(f'Overlay\nPredicted Count: {count:.1f}')
    plt.axis('off')

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
                        calibration_factor=1.0, output_dir='results', display=False):
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
    """Process all images in a directory"""
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
    for image_path in tqdm(image_files):
        result = process_single_image(
            model, image_path, device, target_size, calibration_factor, output_dir, display
        )
        if result:
            results.append(result)

    # Save batch results
    batch_results = {
        'total_images': len(results),
        'average_count': sum(r['predicted_count'] for r in results) / len(results) if results else 0,
        'calibration_factor': calibration_factor,
        'results': results
    }

    batch_json_path = os.path.join(output_dir, 'batch_results.json')
    with open(batch_json_path, 'w') as f:
        json.dump(batch_results, f, indent=4)

    print(f"\nProcessed {len(results)} images")
    print(f"Average count: {batch_results['average_count']:.1f}")
    print(f"Results saved to {batch_json_path}")

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
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
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
