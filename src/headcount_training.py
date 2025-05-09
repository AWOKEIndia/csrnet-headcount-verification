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

def train(model, train_loader, optimizer, criterion, device, epoch, print_freq=10):
    """Training function with proper loss calculation"""
    model.train()
    running_loss = 0.0

    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, (imgs, targets) in enumerate(pbar):
        # Move data to device
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(imgs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Update progress bar
        if i % print_freq == 0:
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validation function with MAE and MSE metrics"""
    model.eval()
    mae = 0
    mse = 0
    val_loss = 0

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Validation"):
            # Move data to device
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(imgs)

            # Calculate loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Calculate metrics
            for idx in range(outputs.size(0)):
                pred_count = outputs[idx].sum().item()
                gt_count = targets[idx].sum().item()

                mae += abs(pred_count - gt_count)
                mse += (pred_count - gt_count) ** 2

    # Calculate average metrics
    mae = mae / len(val_loader.dataset)
    mse = mse / len(val_loader.dataset)
    rmse = np.sqrt(mse)
    val_loss = val_loss / len(val_loader)

    return val_loss, mae, rmse

def main():
    parser = argparse.ArgumentParser(description='Train CSRNet for crowd counting')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--data_root', type=str, default='data/processed', help='data root')
    parser.add_argument('--model_path', type=str, default='models/csrnet_improved.pth', help='model save path')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create model directory
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Set device
    device = get_device()

    # Create model
    model = CSRNet(load_weights=True).to(device)

    # Resume from checkpoint if needed
    start_epoch = 0
    if args.resume and os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        # Try to load epoch info
        checkpoint_info_path = os.path.splitext(args.model_path)[0] + '_info.json'
        if os.path.exists(checkpoint_info_path):
            with open(checkpoint_info_path, 'r') as f:
                info = json.load(f)
                start_epoch = info.get('epoch', 0) + 1
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Define loss function and optimizer
    # Using MSE loss for regression as primary loss
    criterion = nn.MSELoss()

    # Optionally, you can use a custom loss function with L1 and L2 components
    # This often gives better results for density map prediction
    class CombinedLoss(nn.Module):
        def __init__(self, mse_weight=1.0, l1_weight=0.1):
            super(CombinedLoss, self).__init__()
            self.mse_weight = mse_weight
            self.l1_weight = l1_weight

        def forward(self, pred, target):
            mse_loss = F.mse_loss(pred, target)
            l1_loss = F.l1_loss(pred, target)
            return self.mse_weight * mse_loss + self.l1_weight * l1_loss

    # Use the combined loss
    criterion = CombinedLoss(mse_weight=1.0, l1_weight=0.1)

    # Define optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    best_mae = float('inf')
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss, mae, rmse = validate(model, val_loader, criterion, device)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), args.model_path)
            print(f"New best model saved with MAE: {mae:.2f}")

            # Save additional info
            info = {
                'epoch': epoch,
                'mae': float(mae),
                'rmse': float(rmse),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'lr': float(optimizer.param_groups[0]['lr'])
            }

            with open(os.path.splitext(args.model_path)[0] + '_info.json', 'w') as f:
                json.dump(info, f, indent=4)

    print("Training completed!")

if __name__ == '__main__':
    main()
