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

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with proper formatting"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

logger = setup_logger('model_evaluation', 'logs/evaluation.log')

def get_best_device():
    """Automatically detect and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
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
        """Initialize the model evaluator with automatic device detection"""
        self.device = get_best_device()
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """Load the CSRNet model with proper device handling"""
        try:
            model = CSRNet().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)

            if self.device.type == 'mps':
                state_dict = {k: v.cpu() for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            model.eval()

            if self.device.type == 'cuda':
                model = model.half()
                torch.cuda.empty_cache()

            logger.info(f"Successfully loaded model from {model_path} on {self.device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def evaluate_model(self, test_dir, output_dir):
        """Evaluate CSRNet model on test images with improved metrics"""
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
        metrics = {
            'mae': 0,
            'mse': 0,
            'rmse': 0,
            'mape': 0,  # Mean Absolute Percentage Error
            'r2': 0,    # R-squared score
            'total_samples': len(image_files)
        }

        # Process each test image
        for i, (image_file, density_map_file) in enumerate(tqdm(zip(image_files, density_map_files), total=len(image_files))):
            try:
                result = self._process_single_evaluation(image_file, density_map_file, eval_dir)
                if result:
                    results.append(result)
                    metrics['mae'] += result['absolute_error']
                    metrics['mse'] += result['absolute_error'] ** 2
                    metrics['mape'] += abs(result['percentage_error'])
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue

        # Calculate final metrics
        metrics = self._calculate_metrics(metrics, results)
        self._save_evaluation_results(metrics, results, eval_dir)
        self._generate_visualizations(results, eval_dir)

        return metrics

    def _process_single_evaluation(self, image_file, density_map_file, output_dir):
        """Process a single evaluation image with improved metrics"""
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
        percentage_error = (abs_error / gt_count) * 100 if gt_count > 0 else 0

        # Save visualization
        self._save_prediction_visualization(img, gt_density_map, predicted_density_map,
                                         image_id, gt_count, pred_count, abs_error, output_dir)

        return {
            'image_id': image_id,
            'ground_truth': float(gt_count),
            'prediction': float(pred_count),
            'absolute_error': float(abs_error),
            'percentage_error': float(percentage_error)
        }

    def _calculate_metrics(self, metrics, results):
        """Calculate comprehensive evaluation metrics"""
        num_samples = len(results)
        if num_samples == 0:
            return None

        # Calculate basic metrics
        metrics['mae'] = float(metrics['mae'] / num_samples)
        metrics['mse'] = float(metrics['mse'] / num_samples)
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mape'] = float(metrics['mape'] / num_samples)

        # Calculate R-squared score
        gt_counts = np.array([r['ground_truth'] for r in results])
        pred_counts = np.array([r['prediction'] for r in results])
        ss_tot = np.sum((gt_counts - np.mean(gt_counts)) ** 2)
        ss_res = np.sum((gt_counts - pred_counts) ** 2)
        metrics['r2'] = float(1 - (ss_res / ss_tot))

        return metrics

    def _save_evaluation_results(self, metrics, results, output_dir):
        """Save evaluation results and metrics with improved formatting"""
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save detailed results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Evaluation results saved to {output_dir}")

    def _generate_visualizations(self, results, output_dir):
        """Generate comprehensive visualization plots"""
        # Error histogram
        errors = [result['absolute_error'] for result in results]
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, alpha=0.7, color='blue')
        plt.title('Distribution of Absolute Errors')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
        plt.close()

        # Scatter plot with regression line
        gt_counts = [result['ground_truth'] for result in results]
        pred_counts = [result['prediction'] for result in results]

        plt.figure(figsize=(10, 6))
        plt.scatter(gt_counts, pred_counts, alpha=0.7, color='blue')

        # Add regression line
        z = np.polyfit(gt_counts, pred_counts, 1)
        p = np.poly1d(z)
        max_count = max(max(gt_counts), max(pred_counts))
        x_range = np.linspace(0, max_count, 100)
        plt.plot(x_range, p(x_range), 'r--', label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        plt.plot([0, max_count], [0, max_count], 'k--', label='Perfect Prediction')
        plt.title('Predicted vs Ground Truth Counts')
        plt.xlabel('Ground Truth Count')
        plt.ylabel('Predicted Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
        plt.close()

        # Percentage error distribution
        percentage_errors = [result['percentage_error'] for result in results]
        plt.figure(figsize=(10, 6))
        plt.hist(percentage_errors, bins=20, alpha=0.7, color='green')
        plt.title('Distribution of Percentage Errors')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'percentage_error_histogram.png'))
        plt.close()

    def _save_prediction_visualization(self, img, gt_density_map, pred_density_map,
                                    image_id, gt_count, pred_count, abs_error, output_dir):
        """Save enhanced visualization of prediction results"""
        plt.figure(figsize=(15, 5))

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
        """Process a single image with enhanced visualization"""
        os.makedirs(output_dir, exist_ok=True)

        # Load and process image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            predicted_density_map = self.model(img_tensor).squeeze().cpu().numpy()

        pred_count = np.sum(predicted_density_map)

        # Save visualization
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(img))
        plt.title('Original Image')
        plt.axis('off')

        # Predicted density map
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_density_map, cmap='jet')
        plt.title(f'Predicted Density Map\nCount: {pred_count:.2f}')
        plt.axis('off')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_result.png'))
        plt.close()

        return pred_count, predicted_density_map

def main():
    """Main function with improved argument parsing"""
    parser = argparse.ArgumentParser(description='CSRNet Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dir', type=str, help='Directory containing test images and density maps')
    parser.add_argument('--image_path', type=str, help='Path to single test image')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path)

    if args.test_dir:
        metrics = evaluator.evaluate_model(args.test_dir, args.output_dir)
        if metrics:
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    elif args.image_path:
        count, density_map = evaluator.process_single_image(args.image_path, args.output_dir)
        print(f"\nPredicted count: {count:.2f}")

if __name__ == '__main__':
    main()
