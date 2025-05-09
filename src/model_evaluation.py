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
import logging
from datetime import datetime
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.csrnet_implementation import CSRNet
from utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger('model_evaluation', 'logs/evaluation.log')

def get_best_device():
    """
    Automatically detect and return the best available device
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Enable memory efficient settings for CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Metal (MPS) device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

class ModelEvaluator:
    def __init__(self, model_path):
        """
        Initialize the model evaluator with automatic device detection
        """
        self.device = get_best_device()
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """
        Load the CSRNet model with proper device handling
        """
        try:
            model = CSRNet().to(self.device)

            # Load state dict with proper device mapping
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle potential device mismatch in state dict
            if self.device.type == 'mps':
                # Convert CUDA tensors to CPU first for MPS
                state_dict = {k: v.cpu() for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            model.eval()

            # Enable memory efficient inference if using CUDA
            if self.device.type == 'cuda':
                model = model.half()  # Use FP16 for better memory efficiency
                torch.cuda.empty_cache()

            logger.info(f"Successfully loaded model from {model_path} on {self.device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def evaluate_model(self, test_dir, output_dir):
        """
        Evaluate CSRNet model on test images
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_dir = os.path.join(output_dir, f'evaluation_{timestamp}')
        os.makedirs(eval_dir, exist_ok=True)

        # Get test images
        image_files = sorted(glob.glob(os.path.join(test_dir, 'images', '*.jpg')))
        density_map_files = sorted(glob.glob(os.path.join(test_dir, 'density_maps', '*.npy')))

        if not image_files or not density_map_files:
            logger.error(f"No test images or density maps found in {test_dir}")
            return None

        results = []
        mae = 0
        mse = 0

        # Process each test image
        for i, (image_file, density_map_file) in enumerate(tqdm(zip(image_files, density_map_files), total=len(image_files))):
            try:
                result = self._process_single_evaluation(image_file, density_map_file, eval_dir)
                if result:
                    results.append(result)
                    mae += result['absolute_error']
                    mse += result['absolute_error'] ** 2
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue

        # Calculate metrics
        metrics = self._calculate_metrics(results, mae, mse)
        self._save_evaluation_results(metrics, results, eval_dir)
        self._generate_visualizations(results, eval_dir)

        return metrics

    def _process_single_evaluation(self, image_file, density_map_file, output_dir):
        """
        Process a single evaluation image
        """
        image_id = os.path.basename(image_file).split('.')[0]

        # Load and process image
        img = Image.open(image_file).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Load ground truth
        gt_density_map = np.load(density_map_file)
        gt_count = np.sum(gt_density_map)

        # Make prediction
        with torch.no_grad():
            predicted_density_map = self.model(img_tensor).squeeze().cpu().numpy()

        pred_count = np.sum(predicted_density_map)
        abs_error = abs(gt_count - pred_count)

        # Save visualization
        self._save_prediction_visualization(img, gt_density_map, predicted_density_map,
                                         image_id, gt_count, pred_count, abs_error, output_dir)

        return {
            'image_id': image_id,
            'ground_truth': float(gt_count),
            'prediction': float(pred_count),
            'absolute_error': float(abs_error)
        }

    def _calculate_metrics(self, results, mae, mse):
        """
        Calculate evaluation metrics
        """
        num_samples = len(results)
        if num_samples == 0:
            return None

        metrics = {
            'total_samples': num_samples,
            'mae': float(mae / num_samples),
            'mse': float(mse / num_samples),
            'rmse': float(np.sqrt(mse / num_samples))
        }

        return metrics

    def _save_evaluation_results(self, metrics, results, output_dir):
        """
        Save evaluation results and metrics
        """
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save detailed results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Evaluation results saved to {output_dir}")

    def _generate_visualizations(self, results, output_dir):
        """
        Generate visualization plots
        """
        # Error histogram
        errors = [result['absolute_error'] for result in results]
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20)
        plt.title('Distribution of Absolute Errors')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
        plt.close()

        # Scatter plot
        gt_counts = [result['ground_truth'] for result in results]
        pred_counts = [result['prediction'] for result in results]

        plt.figure(figsize=(10, 6))
        plt.scatter(gt_counts, pred_counts, alpha=0.7)
        max_count = max(max(gt_counts), max(pred_counts))
        plt.plot([0, max_count], [0, max_count], 'r--')
        plt.title('Predicted vs Ground Truth Counts')
        plt.xlabel('Ground Truth Count')
        plt.ylabel('Predicted Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
        plt.close()

    def _save_prediction_visualization(self, img, gt_density_map, pred_density_map,
                                    image_id, gt_count, pred_count, abs_error, output_dir):
        """
        Save visualization of prediction results
        """
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
        plt.imshow(pred_density_map, cmap='jet')
        plt.title(f'Predicted Density Map\nCount: {pred_count:.2f}\nError: {abs_error:.2f}')
        plt.axis('off')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{image_id}_result.png'))
        plt.close()

    def process_single_image(self, image_path, output_dir):
        """
        Process a single new image
        """
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(image_path).split('.')[0]

        try:
            # Load and process image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                predicted_density_map = self.model(img_tensor).squeeze().cpu().numpy()

            pred_count = np.sum(predicted_density_map)

            # Save visualization
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(predicted_density_map, cmap='jet')
            plt.title(f'Predicted Density Map\nEstimated Count: {pred_count:.2f}')
            plt.axis('off')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{img_name}_prediction.png'))
            plt.close()

            # Save density map and count
            np.save(os.path.join(output_dir, f'{img_name}_density.npy'), predicted_density_map)
            with open(os.path.join(output_dir, f'{img_name}_count.txt'), 'w') as f:
                f.write(f"Predicted count: {pred_count:.2f}\n")

            logger.info(f"Processed image {image_path}, predicted count: {pred_count:.2f}")
            return pred_count, predicted_density_map

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None, None

    def process_batch_images(self, image_dir, output_dir):
        """
        Process a batch of images
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(output_dir, f'batch_{timestamp}')
        os.makedirs(batch_dir, exist_ok=True)

        # Get all images
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        image_files.extend(sorted(glob.glob(os.path.join(image_dir, '*.png'))))

        if not image_files:
            logger.error(f"No images found in {image_dir}")
            return None

        results = []

        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                pred_count, _ = self.process_single_image(image_file, batch_dir)
                if pred_count is not None:
                    results.append({
                        'image': image_file,
                        'predicted_count': float(pred_count)
                    })
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue

        # Save results
        with open(os.path.join(batch_dir, 'batch_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # Create summary CSV
        with open(os.path.join(batch_dir, 'summary.csv'), 'w') as f:
            f.write("image,predicted_count\n")
            for result in results:
                f.write(f"{os.path.basename(result['image'])},{result['predicted_count']:.2f}\n")

        logger.info(f"Processed {len(results)} images. Results saved to {batch_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSRNet for Crowd Counting')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--mode', type=str, required=True, choices=['evaluate', 'single', 'batch'],
                        help='Evaluation mode: evaluate on test set, process single image, or batch processing')
    parser.add_argument('--test-dir', type=str, help='Directory containing test images and density maps (for evaluate mode)')
    parser.add_argument('--image', type=str, help='Path to input image (for single mode)')
    parser.add_argument('--image-dir', type=str, help='Directory containing images to process (for batch mode)')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results')
    args = parser.parse_args()

    try:
        evaluator = ModelEvaluator(args.model)

        if args.mode == 'evaluate':
            if not args.test_dir:
                parser.error("--test-dir is required for evaluate mode")
            evaluator.evaluate_model(args.test_dir, args.output_dir)

        elif args.mode == 'single':
            if not args.image:
                parser.error("--image is required for single mode")
            evaluator.process_single_image(args.image, args.output_dir)

        elif args.mode == 'batch':
            if not args.image_dir:
                parser.error("--image-dir is required for batch mode")
            evaluator.process_batch_images(args.image_dir, args.output_dir)

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
