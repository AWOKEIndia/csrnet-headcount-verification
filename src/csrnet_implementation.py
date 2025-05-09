import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import glob
import argparse
import time
from tqdm import tqdm
import ssl
import urllib.request
import platform
import psutil
import gc
from datetime import datetime
import json
import logging
import pandas as pd
from pathlib import Path

# Try importing profiler, but don't fail if not available
try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("Warning: PyTorch profiler not available. Performance monitoring will be limited.")

# Disable SSL certificate verification for downloading pre-trained weights
ssl._create_default_https_context = ssl._create_unverified_context

# Setup logging
def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with proper formatting"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logger = setup_logger('csrnet', 'logs/csrnet.log')

def get_device():
    """
    Get the appropriate device for training/inference
    Prioritizes Apple Silicon GPU (MPS) if available, then CUDA, then CPU
    """
    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class CSRNet(nn.Module):
    """
    Enhanced CSRNet with improved attention mechanisms and feature extraction
    """
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        # Frontend - VGG16 features without the last pooling layer
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)

        # Enhanced attention modules
        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.spatial_attention3 = SpatialAttention()

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(1)

        # Output layer with improved initialization
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)

        # Upsampling layer to match target size
        self.upsample = nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)

        if load_weights:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        try:
            vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
            self._initialize_weights()

            # Copy VGG16 weights to frontend
            vgg_frontend_dict = dict([(name, param) for name, param in vgg16.named_parameters()])
            frontend_state_dict = self.frontend.state_dict()

            for name, param in frontend_state_dict.items():
                if name in vgg_frontend_dict:
                    frontend_state_dict[name].copy_(vgg_frontend_dict[name])

            self.frontend.load_state_dict(frontend_state_dict)
            print("Successfully loaded pre-trained VGG16 weights")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
            print("Initializing weights randomly")
            self._initialize_weights()

    def forward(self, x):
        # Frontend with attention
        x = self.frontend(x)
        x = self.bn1(x)
        x = self.spatial_attention1(x)

        # Backend with attention
        x = self.backend(x)
        x = self.bn2(x)
        x = self.spatial_attention2(x)

        # Output with attention
        x = self.output_layer(x)
        x = self.bn3(x)
        x = self.spatial_attention3(x)

        # Upsample to match target size
        x = self.upsample(x)

        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CrowdDataset(Dataset):
    """
    Enhanced dataset class for crowd counting with improved preprocessing
    """
    def __init__(self, image_root, density_map_root, transform=None, target_size=(384, 384), cache_size=1000):
        self.image_root = image_root
        self.density_map_root = density_map_root
        self.target_size = target_size
        self.cache_size = cache_size

        # Initialize cache
        self.cache = {}
        self.cache_keys = []

        # Get all image and density map files
        self.image_files = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
        self.density_map_files = sorted(glob.glob(os.path.join(density_map_root, '*.npy')))

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Training augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])

        assert len(self.image_files) == len(self.density_map_files), "Number of images and density maps don't match!"
        print(f"Dataset size: {len(self.image_files)} images")

    def _load_and_cache_item(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        # Load and preprocess image
        img = Image.open(self.image_files[idx]).convert('RGB')
        img = img.resize(self.target_size, Image.BILINEAR)

        # Load and preprocess density map
        density_map = np.load(self.density_map_files[idx])
        density_map = cv2.resize(density_map, self.target_size, interpolation=cv2.INTER_CUBIC)

        # Apply adaptive scaling based on crowd density
        scale_factor = (self.target_size[0] * self.target_size[1]) / (density_map.shape[0] * density_map.shape[1])
        density_map = density_map * scale_factor

        # Apply data augmentation
        img = self.augmentation(img)

        # Convert to tensor and normalize
        img = self.transform(img)
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)

        # Cache the result
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]

        self.cache[idx] = (img, density_map)
        self.cache_keys.append(idx)

        return img, density_map

    def __getitem__(self, idx):
        return self._load_and_cache_item(idx)

    def __len__(self):
        return len(self.image_files)

def create_density_map_gaussian(points, height, width, sigma=15):
    """
    Enhanced density map generation with adaptive sigma and improved accuracy
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    if len(points) == 0:
        return density_map

    # Calculate local density
    local_density = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            local_density[y, x] += 1

    # Apply Gaussian blur to get density estimate
    density_estimate = cv2.GaussianBlur(local_density, (15, 15), 0)

    # Generate density map with adaptive sigma
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            # Adaptive sigma based on local density
            local_sigma = sigma * (1 + 0.1 * density_estimate[y, x])

            # Create Gaussian kernel
            x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * local_sigma**2))
            gaussian = gaussian / (2 * np.pi * local_sigma**2)

            density_map += gaussian

    # Normalize density map
    if np.sum(density_map) > 0:
        density_map = density_map / np.sum(density_map) * len(points)

    return density_map

class MetricsLogger:
    """Class to handle metrics logging and visualization"""
    def __init__(self, log_dir='logs/metrics'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'mse': [],
            'rmse': [],
            'learning_rate': [],
            'batch_time': [],
            'memory_usage': [],
            'epoch_time': []
        }

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_file = os.path.join(log_dir, f'metrics_{self.timestamp}.csv')

        # Initialize DataFrame
        self.df = pd.DataFrame(columns=self.metrics.keys())

    def log_metrics(self, epoch, metrics_dict):
        """Log metrics for current epoch"""
        # Update metrics storage
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        # Update DataFrame
        self.df.loc[epoch] = metrics_dict

        # Save to CSV
        self.df.to_csv(self.metrics_file)

        # Log to console
        logger.info(f"Epoch {epoch} Metrics:")
        for key, value in metrics_dict.items():
            logger.info(f"{key}: {value:.4f}")

    def plot_metrics(self):
        """Generate and save metric plots"""
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'loss_plot_{self.timestamp}.png'))
        plt.close()

        # Plot MAE and RMSE
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['mae'], label='MAE')
        plt.plot(self.metrics['rmse'], label='RMSE')
        plt.title('MAE and RMSE over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'error_plot_{self.timestamp}.png'))
        plt.close()

        # Plot memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['memory_usage'])
        plt.title('Memory Usage over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.savefig(os.path.join(plots_dir, f'memory_plot_{self.timestamp}.png'))
        plt.close()

class PerformanceMonitor:
    """
    Enhanced monitor and log training performance metrics
    """
    def __init__(self, log_dir='logs/performance'):
        self.batch_times = []
        self.train_losses = []
        self.val_losses = []
        self.mae_scores = []
        self.start_time = time.time()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize profiler if available
        self.profiler = None
        self.profiler_active = False

        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(log_dir)

    def _get_profiler_activities(self):
        """Get appropriate profiler activities based on available devices"""
        if not PROFILER_AVAILABLE:
            return []

        activities = [ProfilerActivity.CPU]

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
            logger.info("CUDA profiling enabled")

        # Check for Metal (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal) device detected - using CPU profiling for MPS operations")

        return activities

    def start_profiling(self):
        """Start profiling"""
        if not PROFILER_AVAILABLE:
            logger.warning("Profiler not available - skipping profiling")
            return

        if not self.profiler_active:
            try:
                self.profiler = profile(
                    activities=self._get_profiler_activities(),
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=3,
                        repeat=2
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                )
                self.profiler.start()
                self.profiler_active = True
                logger.info(f"Started profiling with activities: {self._get_profiler_activities()}")

                # Log device information
                if torch.backends.mps.is_available():
                    logger.info("Using Apple Silicon GPU (MPS)")
                elif torch.cuda.is_available():
                    logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("Using CPU")

            except Exception as e:
                logger.warning(f"Failed to start profiler: {str(e)}")

    def stop_profiling(self):
        """Stop profiling"""
        if not PROFILER_AVAILABLE:
            return

        if self.profiler_active and self.profiler is not None:
            try:
                self.profiler.stop()
                self.profiler.step()
                self.profiler_active = False
                logger.info("Stopped profiling")
            except Exception as e:
                logger.warning(f"Failed to stop profiler: {str(e)}")

    def log_batch_time(self, batch_time):
        """Log the time taken for a batch"""
        self.batch_times.append(batch_time)

    def update_metrics(self, train_loss, val_loss, mae, learning_rate=None):
        """Update training metrics"""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if mae is not None:
            self.mae_scores.append(mae)

        # Log metrics
        metrics_dict = {
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else float('nan'),
            'mae': mae if mae is not None else float('nan'),
            'learning_rate': learning_rate if learning_rate is not None else float('nan'),
            'batch_time': self.get_average_batch_time(),
            'memory_usage': self.get_memory_usage(),
            'epoch_time': self.get_elapsed_time()
        }

        self.metrics_logger.log_metrics(len(self.train_losses) - 1, metrics_dict)

    def get_average_batch_time(self):
        """Get average batch processing time"""
        return np.mean(self.batch_times) if self.batch_times else 0

    def get_memory_usage(self):
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        elif torch.backends.mps.is_available():
            # For MPS, we can only get CPU memory usage
            return psutil.Process().memory_info().rss / 1024**2  # MB
        return psutil.Process().memory_info().rss / 1024**2  # MB

    def get_elapsed_time(self):
        """Get total elapsed time"""
        return time.time() - self.start_time

    def log_performance(self, epoch):
        """Log current performance metrics"""
        logger.info(f"\nPerformance Metrics (Epoch {epoch}):")
        logger.info(f"Average Batch Time: {self.get_average_batch_time():.3f}s")
        logger.info(f"Memory Usage: {self.get_memory_usage():.1f}MB")
        logger.info(f"Elapsed Time: {self.get_elapsed_time():.1f}s")
        if self.mae_scores:
            logger.info(f"Current MAE: {self.mae_scores[-1]:.4f}")

        # Generate plots
        self.metrics_logger.plot_metrics()

def train(model, train_loader, optimizer, epoch, device, monitor):
    """
    Enhanced training function with improved loss calculation and monitoring
    """
    model.train()
    total_loss = 0
    batch_time = 0
    data_time = 0

    end = time.time()

    try:
        monitor.start_profiling()

        for i, (img, target) in enumerate(train_loader):
            data_time = time.time() - end

            img = img.to(device)
            target = target.to(device)

            # Forward pass
            if PROFILER_AVAILABLE:
                with record_function("model_inference"):
                    output = model(img)
            else:
                output = model(img)

            # Calculate loss with L1 and L2 components
            l1_loss = F.l1_loss(output, target)
            l2_loss = F.mse_loss(output, target)
            loss = l1_loss + 0.1 * l2_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_time = time.time() - end
            end = time.time()

            # Update monitoring
            monitor.log_batch_time(batch_time)

            if i % 10 == 0:
                logger.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                           f'Loss {loss.item():.4f}\t'
                           f'Time {batch_time:.3f}s')
    finally:
        monitor.stop_profiling()

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """
    Enhanced validation function with improved metrics
    """
    model.eval()
    mae = 0
    mse = 0

    with torch.no_grad():
        for img, target in val_loader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            # Calculate metrics
            pred_count = torch.sum(output).item()
            gt_count = torch.sum(target).item()

            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2

    mae = mae / len(val_loader)
    mse = mse / len(val_loader)
    rmse = np.sqrt(mse)

    return mae, rmse

def predict(model, image_path, device):
    """
    Enhanced prediction function with improved visualization
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    # Process output
    density_map = output.squeeze().cpu().numpy()
    count = np.sum(density_map)

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title('Original Image')
    plt.axis('off')

    # Density map
    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title(f'Predicted Density Map\nCount: {count:.2f}')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()

    return density_map, count, plt.gcf()

def main():
    """
    Main function with improved argument parsing and training loop
    """
    parser = argparse.ArgumentParser(description='CSRNet Crowd Counting')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--image_path', type=str, help='path to test image')
    parser.add_argument('--model_path', type=str, default='models/csrnet.pth',
                       help='path to save/load the model (will be created if training)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    if args.train:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

        # Training setup
        model = CSRNet(load_weights=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        monitor = PerformanceMonitor()

        # Create data loaders
        train_dataset = CrowdDataset('data/processed/train/images', 'data/processed/train/density_maps')
        val_dataset = CrowdDataset('data/processed/val/images', 'data/processed/val/density_maps')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Training loop
        best_mae = float('inf')
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

            # Training phase
            train_loss = train(model, train_loader, optimizer, epoch, device, monitor)

            # Validation phase
            mae, rmse = validate(model, val_loader, device)

            # Update metrics
            monitor.update_metrics(train_loss, None, mae, optimizer.param_groups[0]['lr'])

            logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}')

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), args.model_path)
                logger.info(f'Model saved with MAE: {mae:.4f}')

            # Log performance metrics
            monitor.log_performance(epoch)

    elif args.test and args.image_path:
        # Load model and make prediction
        model = CSRNet().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        density_map, count, fig = predict(model, args.image_path, device)
        plt.show()
        logger.info(f'Predicted count: {count:.2f}')

if __name__ == '__main__':
    main()
