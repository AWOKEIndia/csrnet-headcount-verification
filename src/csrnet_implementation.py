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
from torch.profiler import profile, record_function, ProfilerActivity

# Disable SSL certificate verification for downloading pre-trained weights
ssl._create_default_https_context = ssl._create_unverified_context

def get_device():
    """
    Get the appropriate device for training/inference
    Prioritizes Apple Silicon GPU (MPS) if available, then CUDA, then CPU
    """
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        if in_channels == 1:
            # For single channel, use a simpler attention mechanism
            self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()
        else:
            # For multi-channel inputs
            self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.size(1) == 1:
            # Simple attention for single channel
            attention = self.conv1(x)
            attention = self.sigmoid(attention)
            return x * attention
        else:
            # Multi-channel attention
            attention = self.conv1(x)
            attention = F.relu(attention)
            attention = self.conv2(attention)
            attention = self.sigmoid(attention)
            return x * attention

class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
    """
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        # Frontend - VGG16 features without the last pooling layer
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)

        # Add attention modules with correct channel dimensions
        self.attention1 = AttentionModule(512)  # After frontend
        self.attention2 = AttentionModule(64)   # After backend
        self.attention3 = AttentionModule(1)    # After output layer

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Add batch normalization with correct channel dimensions
        self.bn1 = nn.BatchNorm2d(512)  # After frontend
        self.bn2 = nn.BatchNorm2d(64)   # After backend
        self.bn3 = nn.BatchNorm2d(1)    # After output layer

        if load_weights:
            try:
                # Load pretrained VGG16 weights
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
        # Frontend
        x = self.frontend(x)
        x = self.bn1(x)
        x = self.attention1(x)

        # Backend
        x = self.backend(x)
        x = self.bn2(x)
        x = self.attention2(x)

        # Output
        x = self.output_layer(x)
        x = self.bn3(x)
        x = self.attention3(x)

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
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CrowdDataset(Dataset):
    """
    Dataset class for crowd counting
    """
    def __init__(self, image_root, density_map_root, transform=None, target_size=(384, 384), cache_size=1000):
        self.image_root = image_root
        self.density_map_root = density_map_root
        self.transform = transform
        self.target_size = target_size
        self.cache_size = cache_size

        # Initialize cache
        self.cache = {}
        self.cache_keys = []

        # Get all image and density map files
        self.image_files = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
        self.density_map_files = sorted(glob.glob(os.path.join(density_map_root, '*.npy')))

        # Enhanced data augmentation with more variations
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        assert len(self.image_files) == len(self.density_map_files), "Number of images and density maps don't match!"
        print(f"Dataset size: {len(self.image_files)} images")

    def _load_and_cache_item(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img = Image.open(self.image_files[idx]).convert('RGB')
        img = img.resize(self.target_size, Image.BILINEAR)

        density_map = np.load(self.density_map_files[idx])
        density_map = cv2.resize(density_map, self.target_size, interpolation=cv2.INTER_CUBIC)

        # Apply adaptive scaling based on crowd density
        scale_factor = (self.target_size[0] * self.target_size[1]) / (density_map.shape[0] * density_map.shape[1])
        density_map = density_map * scale_factor

        if self.transform is not None:
            img = self.augmentation(img)

        density_map = torch.from_numpy(density_map).float().unsqueeze(0)

        # Cache the result
        if len(self.cache) >= self.cache_size:
            # Remove oldest item from cache
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
    Enhanced density map generation with adaptive sigma based on crowd density
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    # Calculate local density
    local_density = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x < width and y < height:
            local_density[y, x] += 1

    # Apply Gaussian blur to get density estimate
    density_estimate = cv2.GaussianBlur(local_density, (15, 15), 0)

    # Generate density map with adaptive sigma
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x < width and y < height:
            # Adaptive sigma based on local density
            local_sigma = max(5, min(20, sigma * (1 + density_estimate[y, x])))

            gaussian_kernel = np.zeros((height, width), dtype=np.float32)
            gaussian_kernel[y, x] = 1
            gaussian_kernel = cv2.GaussianBlur(gaussian_kernel, (int(local_sigma), int(local_sigma)), 0)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

            density_map += gaussian_kernel

    return density_map


class PerformanceMonitor:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'epoch_time': [],
            'memory_usage': [],
            'gpu_memory': [],
            'batch_times': []
        }
        self.start_time = None
        self.batch_times = []

    def start_epoch(self):
        self.start_time = time.time()
        self.batch_times = []

    def end_epoch(self):
        epoch_time = time.time() - self.start_time
        self.metrics['epoch_time'].append(epoch_time)

        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.metrics['memory_usage'].append(memory_info.rss / 1024 / 1024)  # MB

        # Get GPU memory if available
        if torch.backends.mps.is_available():
            try:
                gpu_memory = torch.mps.current_allocated_memory() / 1024 / 1024  # MB
                self.metrics['gpu_memory'].append(gpu_memory)
            except:
                self.metrics['gpu_memory'].append(0)
        elif torch.cuda.is_available():
            self.metrics['gpu_memory'].append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
        else:
            self.metrics['gpu_memory'].append(0)

        # Calculate average batch time
        if self.batch_times:
            self.metrics['batch_times'].append(np.mean(self.batch_times))

    def update_metrics(self, train_loss=None, val_loss=None, mae=None):
        if train_loss is not None:
            self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if mae is not None:
            self.metrics['mae'].append(mae)

    def log_batch_time(self, batch_time):
        self.batch_times.append(batch_time)

    def save_metrics(self, epoch):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(self.log_dir, f'metrics_epoch_{epoch}_{timestamp}.json')

        # Convert numpy values to Python native types
        metrics_dict = {k: [float(x) if isinstance(x, np.float32) else x for x in v]
                       for k, v in self.metrics.items()}

        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    def plot_metrics(self, epoch):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot training metrics
        plt.figure(figsize=(15, 10))

        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # MAE plot
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['mae'])
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')

        # Memory usage plot
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['memory_usage'], label='RAM')
        plt.plot(self.metrics['gpu_memory'], label='GPU Memory')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.title('Memory Usage')

        # Batch time plot
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['batch_times'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Average Batch Processing Time')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'metrics_epoch_{epoch}_{timestamp}.png'))
        plt.close()

def train(model, train_loader, optimizer, epoch, device, monitor):
    """
    Enhanced training function with improved loss calculation
    """
    model.train()
    train_loss = 0
    monitor.start_epoch()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        batch_start = time.time()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with record_function("model_forward"):
            output = model(data)

        if output.size() != target.size():
            output = F.interpolate(output, size=target.size()[2:], mode='bilinear', align_corners=False)

        # Combined loss function
        mse_loss = F.mse_loss(output, target)

        # Add L1 loss for better count accuracy
        l1_loss = F.l1_loss(output, target)

        # Add gradient loss for better density map quality
        gradient_loss = torch.mean(torch.abs(
            torch.diff(output, dim=2) - torch.diff(target, dim=2)
        )) + torch.mean(torch.abs(
            torch.diff(output, dim=3) - torch.diff(target, dim=3)
        ))

        # Combine losses with weights
        loss = mse_loss + 0.5 * l1_loss + 0.1 * gradient_loss

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

        batch_time = time.time() - batch_start
        monitor.log_batch_time(batch_time)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss /= len(train_loader)
    print(f'Train Epoch: {epoch}, Average Loss: {train_loss:.6f}')

    monitor.end_epoch()
    return train_loss


def validate(model, val_loader, device):
    """
    Validation function
    """
    model.eval()
    val_loss = 0
    mae = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Ensure output and target have the same size
            if output.size() != target.size():
                output = F.interpolate(output, size=target.size()[2:], mode='bilinear', align_corners=False)

            # MSE Loss
            val_loss += F.mse_loss(output, target).item()

            # MAE (Mean Absolute Error)
            pred_count = output.sum().item()
            true_count = target.sum().item()
            mae += abs(pred_count - true_count)

    val_loss /= len(val_loader)
    mae /= len(val_loader)

    print(f'Validation Loss: {val_loss:.6f}, MAE: {mae:.2f}')
    return val_loss, mae


def predict(model, image_path, device):
    """
    Enhanced prediction function with post-processing
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    # Post-processing
    output_np = output.squeeze().cpu().numpy()

    # Apply Gaussian smoothing to reduce noise
    output_np = cv2.GaussianBlur(output_np, (5, 5), 0)

    # Apply adaptive thresholding
    threshold = np.mean(output_np) + 2 * np.std(output_np)
    output_np[output_np < threshold] = 0

    # Get the predicted count with confidence
    count = np.sum(output_np)
    confidence = np.mean(output_np[output_np > 0]) if np.any(output_np > 0) else 0

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.array(img))
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(output_np, cmap='jet')
    plt.title(f'Density Map (Count: {count:.2f})')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(output_np > threshold, cmap='gray')
    plt.title(f'Detected Heads (Confidence: {confidence:.2f})')

    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.close()

    return count, output_np


def main():
    parser = argparse.ArgumentParser(description='CSRNet for Crowd Counting')
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='save the current model')
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test mode')
    parser.add_argument('--test-image', type=str, default=None, help='path to test image')
    parser.add_argument('--load-model', type=str, default=None, help='path to saved model')
    parser.add_argument('--image-size', type=int, default=384, help='input image size (default: 384)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading (default: 4)')
    parser.add_argument('--profile', action='store_true', help='enable PyTorch profiler')
    parser.add_argument('--cache-size', type=int, default=1000, help='number of images to cache in memory')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='number of batches to prefetch')
    parser.add_argument('--pin-memory', action='store_true', default=True, help='pin memory in data loader')
    parser.add_argument('--persistent-workers', action='store_true', default=True, help='use persistent workers')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    args = parser.parse_args()

    # Set device
    device = get_device()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    elif device.type == 'mps':
        torch.mps.manual_seed(args.seed)

    # Create model
    model = CSRNet(load_weights=True).to(device)

    # Load model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")

    if args.mode == 'train':
        # Initialize performance monitor
        monitor = PerformanceMonitor()

        # Data transforms with optimizations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets with consistent image size and caching
        target_size = (args.image_size, args.image_size)
        train_dataset = CrowdDataset(
            os.path.join(args.data_path, 'train', 'images'),
            os.path.join(args.data_path, 'train', 'density_maps'),
            transform=transform,
            target_size=target_size,
            cache_size=args.cache_size
        )

        val_dataset = CrowdDataset(
            os.path.join(args.data_path, 'val', 'images'),
            os.path.join(args.data_path, 'val', 'density_maps'),
            transform=transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            target_size=target_size,
            cache_size=args.cache_size
        )

        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=True  # Drop last incomplete batch
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor
        )

        # Optimizer with gradient clipping and learning rate scheduling
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Initialize early stopping variables
        best_mae = float('inf')
        patience_counter = 0
        patience = args.patience

        # Training loop with improved memory management
        for epoch in range(1, args.epochs + 1):
            # Clear cache periodically
            if epoch % 5 == 0:
                train_dataset.cache.clear()
                train_dataset.cache_keys.clear()
                val_dataset.cache.clear()
                val_dataset.cache_keys.clear()
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()

            if args.profile and epoch == 1:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                           on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
                           record_shapes=True,
                           profile_memory=True,
                           with_stack=True) as prof:
                    train_loss = train(model, train_loader, optimizer, epoch, device, monitor)
                    val_loss, mae = validate(model, val_loader, device)
                    prof.step()
            else:
                train_loss = train(model, train_loader, optimizer, epoch, device, monitor)
                val_loss, mae = validate(model, val_loader, device)

            # Update learning rate
            scheduler.step()

            # Update metrics
            monitor.update_metrics(train_loss=train_loss, val_loss=val_loss, mae=mae)

            # Save metrics and plots
            monitor.save_metrics(epoch)
            monitor.plot_metrics(epoch)

            # Early stopping check
            if mae < best_mae:
                best_mae = mae
                if args.save_model:
                    torch.save(model.state_dict(), 'csrnet_best.pth')
                    print(f"Saved best model with MAE: {best_mae:.2f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs. Best MAE: {best_mae:.2f}")

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

            # Save checkpoint
            if args.save_model and epoch % 5 == 0:
                torch.save(model.state_dict(), f'csrnet_epoch_{epoch}.pth')

            # Clear memory
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # Save final model
        if args.save_model:
            torch.save(model.state_dict(), 'csrnet_final.pth')

    elif args.mode == 'test' and args.test_image:
        count, density_map = predict(model, args.test_image, device)
        print(f"Predicted count: {count:.2f}")


if __name__ == '__main__':
    main()
