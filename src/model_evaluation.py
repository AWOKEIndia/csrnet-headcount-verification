import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import argparse
from torchvision import transforms
from tqdm import tqdm
import json
from scipy.ndimage import gaussian_filter

# Import CSRNet model
# Make sure to have the CSRNet implementation file in the same directory
from csrnet_implementation import CSRNet


def evaluate_model(model_path, test_dir, output_dir, device='cuda'):
    """
    Evaluate CSRNet model on test images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get test images
    image_files = sorted(glob.glob(os.path.join(test_dir, 'images', '*.jpg')))
    density_map_files = sorted(glob.glob(os.path.join(test_dir, 'density_maps', '*.npy')))

    results = []
    mae = 0
    mse = 0

    # Process each test image
    for i, (image_file, density_map_file) in enumerate(tqdm(zip(image_files, density_map_files), total=len(image_files))):
        # Extract image ID
        image_id = os.path.basename(image_file).split('.')[0]

        # Load image
        img = Image.open(image_file).convert('RGB')
        width, height = img.size

        # Convert to tensor
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Load ground truth density map
        gt_density_map = np.load(density_map_file)
        gt_count = np.sum(gt_density_map)

        # Make prediction
        with torch.no_grad():
            predicted_density_map = model(img_tensor).squeeze().cpu().numpy()

        # Calculate predicted count
        pred_count = np.sum(predicted_density_map)

        # Calculate error
        abs_error = abs(gt_count - pred_count)
        squared_error = (gt_count - pred_count) ** 2

        # Accumulate errors
        mae += abs_error
        mse += squared_error

        # Store result
        results.append({
            'image_id': image_id,
            'ground_truth': float(gt_count),
            'prediction': float(pred_count),
            'absolute_error': float(abs_error)
        })

        # Visualize and save result
        plt.figure(figsize=(16, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(img))
        plt.title(f'Original Image\nImage ID: {image_id}')
        plt.axis('off')

        # Ground truth density map
        plt.subplot(1, 3, 2)
        plt.imshow(gt_density_map, cmap='jet')
        plt.title(f'Ground Truth Density Map\nCount: {gt_count:.2f}')
        plt.axis('off')
        plt.colorbar()

        # Predicted density map
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_density_map, cmap='jet')
        plt.title(f'Predicted Density Map\nCount: {pred_count:.2f}\nError: {abs_error:.2f}')
        plt.axis('off')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{image_id}_result.png'))
        plt.close()

    # Calculate average metrics
    num_samples = len(image_files)
    mae = mae / num_samples
    mse = mse / num_samples
    rmse = np.sqrt(mse)

    # Log metrics
    metrics = {
        'total_samples': num_samples,
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse)
    }

    print(f"Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Save metrics and results
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Generate error histogram
    errors = [result['absolute_error'] for result in results]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20)
    plt.title('Distribution of Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
    plt.close()

    # Generate scatter plot of predictions vs ground truth
    gt_counts = [result['ground_truth'] for result in results]
    pred_counts = [result['prediction'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(gt_counts, pred_counts, alpha=0.7)

    # Add perfect prediction line
    max_count = max(max(gt_counts), max(pred_counts))
    plt.plot([0, max_count], [0, max_count], 'r--')

    plt.title('Predicted vs Ground Truth Counts')
    plt.xlabel('Ground Truth Count')
    plt.ylabel('Predicted Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
    plt.close()

    return metrics


def process_new_image(model_path, image_path, output_dir, device='cuda'):
    """
    Process a new unseen image
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image
    img = Image.open(image_path).convert('RGB')
    img_name = os.path.basename(image_path).split('.')[0]

    # Convert to tensor
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        predicted_density_map = model(img_tensor).squeeze().cpu().numpy()

    # Calculate predicted count
    pred_count = np.sum(predicted_density_map)

    print(f"Predicted count: {pred_count:.2f}")

    # Visualize and save result
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title('Original Image')
    plt.axis('off')

    # Predicted density map
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_density_map, cmap='jet')
    plt.title(f'Predicted Density Map\nEstimated Count: {pred_count:.2f}')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{img_name}_prediction.png'))
    plt.close()

    # Save density map for further analysis
    np.save(os.path.join(output_dir, f'{img_name}_density.npy'), predicted_density_map)

    # Save count to text file
    with open(os.path.join(output_dir, f'{img_name}_count.txt'), 'w') as f:
        f.write(f"Predicted count: {pred_count:.2f}\n")

    return pred_count, predicted_density_map


def process_batch_images(model_path, image_dir, output_dir, device='cuda'):
    """
    Process a batch of images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get all images
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    image_files.extend(sorted(glob.glob(os.path.join(image_dir, '*.png'))))

    results = []

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        img_name = os.path.basename(image_file).split('.')[0]

        # Load image
        img = Image.open(image_file).convert('RGB')

        # Convert to tensor
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            predicted_density_map = model(img_tensor).squeeze().cpu().numpy()

        # Calculate predicted count
        pred_count = np.sum(predicted_density_map)

        results.append({
            'image': image_file,
            'predicted_count': float(pred_count)
        })

        # Visualize and save result
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(img))
        plt.title('Original Image')
        plt.axis('off')

        # Predicted density map
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_density_map, cmap='jet')
        plt.title(f'Predicted Density Map\nEstimated Count: {pred_count:.2f}')
        plt.axis('off')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{img_name}_prediction.png'))
        plt.close()

    # Save all results to a JSON file
    with open(os.path.join(output_dir, 'batch_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create summary CSV
    with open(os.path.join(output_dir, 'summary.csv'), 'w') as f:
        f.write("image,predicted_count\n")
        for result in results:
            f.write(f"{os.path.basename(result['image'])},{result['predicted_count']:.2f}\n")

    print(f"Processed {len(results)} images. Results saved to {output_dir}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CSRNet for Crowd Counting')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--mode', type=str, required=True, choices=['evaluate', 'single', 'batch'],
                        help='Evaluation mode: evaluate on test set, process single image, or batch processing')
    parser.add_argument('--test-dir', type=str, help='Directory containing test images and density maps (for evaluate mode)')
    parser.add_argument('--image', type=str, help='Path to input image (for single mode)')
    parser.add_argument('--image-dir', type=str, help='Directory containing images to process (for batch mode)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'

    if args.mode == 'evaluate':
        if not args.test_dir:
            parser.error("--test-dir is required for evaluate mode")
        evaluate_model(args.model, args.test_dir, args.output_dir, args.device)

    elif args.mode == 'single':
        if not args.image:
            parser.error("--image is required for single mode")
        process_new_image(args.model, args.image, args.output_dir, args.device)

    elif args.mode == 'batch':
        if not args.image_dir:
            parser.error("--image-dir is required for batch mode")
        process_batch_images(args.model, args.image_dir, args.output_dir, args.device)
