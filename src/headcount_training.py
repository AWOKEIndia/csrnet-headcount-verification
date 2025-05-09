import argparse
import glob
import json
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Import CSRNet model definition
from headcount_solution import CSRNet, get_device


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
            density_map_file = os.path.join(density_map_root, f"{image_id}.npy")
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

def train(model, train_loader, optimizer, criterion, device, epoch, args):
    """Training function with proper loss calculation"""
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, (imgs, targets) in enumerate(train_loader):
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

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg

def validate(model, val_loader, criterion, device):
    """Validation function with MAE metric"""
    model.eval()
    mae = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            # Move data to device
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(imgs)

            # Calculate MAE
            mae += abs(outputs.sum() - targets.sum())

    mae = mae / len(val_loader)
    print(' * MAE {mae:.3f}'.format(mae=mae))

    return mae

def plot_metrics(metrics_history, save_path):
    """Plot and save training metrics"""
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['mae'], label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

    # Create model directory
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(args.model_path), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Initialize metrics history
    metrics_history = {
        'train_loss': [],
        'mae': []
    }

    # Set device
    device = get_device()

    # Create model
    model = CSRNet(load_weights=True).to(device)

    # Resume from checkpoint if needed
    start_epoch = 0
    best_mae = float('inf')
    if args.resume and os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mae = checkpoint.get('best_mae', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

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

    # Define optimizer with SGD
    optimizer = optim.SGD(model.parameters(), args.lr,
                         momentum=args.momentum,
                         weight_decay=args.decay)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.10f}")

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args)

        # Validate
        mae = validate(model, val_loader, criterion, device)

        # Update metrics history
        metrics_history['train_loss'].append(train_loss)
        metrics_history['mae'].append(mae)

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae': best_mae,
                'optimizer': optimizer.state_dict(),
            }, args.model_path)
            print(f"New best model saved with MAE: {mae:.3f}")

            # Save additional info
            info = {
                'epoch': epoch,
                'mae': float(mae),
                'train_loss': float(train_loss),
                'lr': float(optimizer.param_groups[0]['lr'])
            }

            with open(os.path.splitext(args.model_path)[0] + '_info.json', 'w') as f:
                json.dump(info, f, indent=4)

    # Plot final metrics
    plot_metrics(metrics_history, os.path.join(vis_dir, 'final_metrics.png'))
    print("Training completed!")

if __name__ == '__main__':
    main()
