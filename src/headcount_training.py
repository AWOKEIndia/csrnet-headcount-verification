import argparse
import glob
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# Import CSRNet model definition
from headcount_solution import CSRNet, get_device

# Set up logging
def setup_logger(output_dir):
    """Set up logger with file and console handlers"""
    logger = logging.getLogger('CSRNet')
    logger.setLevel(logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # File handler
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class PerformanceMonitor:
    """Class to monitor and log training performance metrics"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics = {
            'train_loss': [],
            'val_mae': [],
            'val_mse': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_memory': []
        }
        self.best_metrics = {
            'mae': float('inf'),
            'mse': float('inf'),
            'loss': float('inf')
        }
        self.start_time = time.time()

        # Create metrics directory
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    def update(self, epoch, metrics):
        """Update metrics and log to TensorBoard"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                self.writer.add_scalar(f'metrics/{key}', value, epoch)

        # Update best metrics
        if 'val_mae' in metrics and metrics['val_mae'] < self.best_metrics['mae']:
            self.best_metrics['mae'] = metrics['val_mae']
        if 'val_mse' in metrics and metrics['val_mse'] < self.best_metrics['mse']:
            self.best_metrics['mse'] = metrics['val_mse']
        if 'train_loss' in metrics and metrics['train_loss'] < self.best_metrics['loss']:
            self.best_metrics['loss'] = metrics['train_loss']

    def plot_metrics(self):
        """Plot and save training metrics"""
        # Set matplotlib style
        plt.style.use('default')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)

        # Plot loss
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', color='#1f77b4')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        axes[0, 0].legend()

        # Plot MAE
        axes[0, 1].plot(self.metrics['val_mae'], label='Validation MAE', color='#ff7f0e')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].legend()

        # Plot MSE
        axes[1, 0].plot(self.metrics['val_mse'], label='Validation MSE', color='#2ca02c')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        axes[1, 0].legend()

        # Plot learning rate
        axes[1, 1].plot(self.metrics['learning_rate'], label='Learning Rate', color='#d62728')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1, 1].legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self):
        """Save metrics to CSV file"""
        # Find the minimum length among all metric arrays
        min_length = min(len(arr) for arr in self.metrics.values())

        # Create a new dictionary with truncated arrays
        truncated_metrics = {
            key: arr[:min_length] for key, arr in self.metrics.items()
        }

        # Create DataFrame with truncated arrays
        df = pd.DataFrame(truncated_metrics)
        df.to_csv(os.path.join(self.metrics_dir, 'metrics.csv'), index=False)

        # Save best metrics
        with open(os.path.join(self.metrics_dir, 'best_metrics.json'), 'w') as f:
            json.dump(self.best_metrics, f, indent=4)

    def log_epoch(self, epoch, metrics, logger):
        """Log epoch metrics"""
        elapsed_time = time.time() - self.start_time
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"Time elapsed: {elapsed_time/3600:.2f} hours")
        logger.info(f"Train Loss: {metrics['train_loss']:.4f}")
        logger.info(f"Val MAE: {metrics['val_mae']:.4f}")
        logger.info(f"Val MSE: {metrics['val_mse']:.4f}")
        logger.info(f"Learning Rate: {metrics['learning_rate']:.2e}")
        logger.info(f"Best MAE: {self.best_metrics['mae']:.4f}")
        logger.info(f"Best MSE: {self.best_metrics['mse']:.4f}")
        logger.info(f"Best Loss: {self.best_metrics['loss']:.4f}")

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

class CrowdDataset(Dataset):
    """
    Dataset class for crowd counting with proper scaling
    """
    def __init__(self, image_root, density_map_root, target_size=(384, 384), transform=None, is_train=True):
        self.image_root = image_root
        self.density_map_root = density_map_root
        self.target_size = target_size
        self.is_train = is_train

        # Define transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Define augmentation for training
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
        self.image_files.extend(sorted(glob.glob(os.path.join(image_root, '*.png'))))

        # Get corresponding density map files
        self.density_map_files = []
        for image_file in self.image_files:
            image_id = os.path.splitext(os.path.basename(image_file))[0]
            density_map_file = os.path.join(self.density_map_root, f"{image_id}.npy")
            if os.path.exists(density_map_file):
                self.density_map_files.append(density_map_file)
            else:
                print(f"Warning: Density map not found for {image_id}")

        # Keep only images with corresponding density maps
        self.image_files = [self.image_files[i] for i in range(len(self.image_files))
                           if i < len(self.density_map_files)]

        print(f"Loaded {len(self.image_files)} images with density maps")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_files[idx]).convert('RGB')

        # Get original dimensions
        orig_width, orig_height = img.size

        # Apply augmentation for training
        if self.is_train:
            img = self.augmentation(img)

        # Resize to target size
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.BILINEAR)

        # Apply transforms
        img_tensor = self.transform(img)

        # Load density map
        density_map = np.load(self.density_map_files[idx])

        # Resize density map to target size and adjust for scale
        if density_map.shape[:2] != self.target_size[::-1]:  # Note the inversion (h,w) vs (w,h)
            # Calculate scale factor for preserving count during resize
            scale_factor = (self.target_size[0] * self.target_size[1]) / (density_map.shape[1] * density_map.shape[0])

            # Resize density map
            density_map = cv2.resize(density_map, self.target_size, interpolation=cv2.INTER_LINEAR)

            # Apply scale factor to preserve the count
            density_map = density_map * scale_factor

            # Apply Gaussian smoothing to the resized density map
            density_map = gaussian_filter(density_map, sigma=1.0)

            # Verify density map sum is preserved
            original_sum = np.sum(density_map)
            if abs(original_sum - np.sum(density_map)) > 0.1:
                print(f"Warning: Density map sum changed after resize. Original: {original_sum:.2f}, New: {np.sum(density_map):.2f}")

        # Convert to tensor
        density_map_tensor = torch.from_numpy(density_map).float().unsqueeze(0)

        return img_tensor, density_map_tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr

    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def train(model, train_loader, optimizer, criterion, device, epoch, args, logger):
    """Training function with enhanced logging and monitoring"""
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    # Create progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')

    for i, (imgs, targets) in enumerate(pbar):
        data_time.update(time.time() - end)

        # Move data to device
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(imgs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Update metrics
        losses.update(loss.item(), imgs.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update time metrics
        batch_time.update(time.time() - end)
        end = time.time()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'batch_time': f'{batch_time.avg:.3f}s',
            'data_time': f'{data_time.avg:.3f}s'
        })

        if i % args.print_freq == 0:
            logger.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                       f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                       f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                       f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)')

    return losses.avg

def validate(model, val_loader, criterion, device, logger):
    """Validation function with enhanced metrics"""
    model.eval()
    mae = AverageMeter()
    mse = AverageMeter()

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc='Validation'):
            # Move data to device
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(imgs)

            # Calculate metrics
            pred_count = outputs.sum().item()
            gt_count = targets.sum().item()

            # Update metrics
            mae.update(abs(pred_count - gt_count))
            mse.update((pred_count - gt_count) ** 2)

    logger.info(f' * MAE {mae.avg:.3f}')
    logger.info(f' * MSE {mse.avg:.3f}')

    return mae.avg, mse.avg

def main():
    parser = argparse.ArgumentParser(description='Train CSRNet for crowd counting')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-7, help='initial learning rate')
    parser.add_argument('--data_root', type=str, default='data/processed', help='data root')
    parser.add_argument('--model_path', type=str, default='models/csrnet_improved.pth', help='model save path')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--print_freq', type=int, default=30, help='print frequency')
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', f'training_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Set up logger
    logger = setup_logger(output_dir)
    logger.info(f"Starting training at {timestamp}")
    logger.info(f"Arguments: {args}")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Additional training parameters
    args.original_lr = args.lr
    args.momentum = 0.95
    args.decay = 5e-4
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4

    # Initialize performance monitor
    monitor = PerformanceMonitor(output_dir)

    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create model
    model = CSRNet(load_weights=True).to(device)

    # Resume from checkpoint if needed
    start_epoch = 0
    best_mae = float('inf')
    if args.resume and os.path.exists(args.model_path):
        logger.info(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mae = checkpoint.get('best_mae', float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}")

    # Create data loaders
    train_dataset = CrowdDataset(
        os.path.join(args.data_root, 'train', 'images'),
        os.path.join(args.data_root, 'train', 'density_maps'),
        is_train=True
    )

    val_dataset = CrowdDataset(
        os.path.join(args.data_root, 'val', 'images'),
        os.path.join(args.data_root, 'val', 'density_maps'),
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Define loss function
    criterion = nn.MSELoss(size_average=False).to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), args.lr,
                         momentum=args.momentum,
                         weight_decay=args.decay)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.10f}")

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args, logger)

        # Validate
        mae, mse = validate(model, val_loader, criterion, device, logger)

        # Update metrics
        metrics = {
            'train_loss': train_loss,
            'val_mae': mae,
            'val_mse': mse,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': time.time() - monitor.start_time
        }
        monitor.update(epoch, metrics)

        # Log epoch summary
        monitor.log_epoch(epoch, metrics, logger)

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae': best_mae,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"New best model saved with MAE: {mae:.3f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae': best_mae,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(output_dir, 'checkpoint.pth'))

        # Plot metrics
        monitor.plot_metrics()

    # Save final metrics
    monitor.save_metrics()
    monitor.close()

    logger.info("Training completed!")
    logger.info(f"Best MAE: {best_mae:.3f}")
    logger.info(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()
