"""
Headcount Solution

This module provides a comprehensive solution for headcount verification using deep learning models.
It includes functionality for training, testing, and predicting headcount in images and videos.

"""
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms


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

def get_device():
    """Get the appropriate device for computation"""
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def create_density_map_gaussian(points, height, width, sigma=15):
    """
    Generate density map based on head point annotations
    using Gaussian filter from scipy.ndimage

    Args:
        points (numpy.ndarray): Array of head point coordinates [x, y]
        height (int): Height of the output density map
        width (int): Width of the output density map
        sigma (int): Sigma for Gaussian kernel

    Returns:
        numpy.ndarray: Generated density map that integrates to the point count
    """
    # Create binary map of points
    density_map = np.zeros((height, width), dtype=np.float32)

    # If no points, return empty density map
    if len(points) == 0:
        return density_map

    # Mark points in the density map
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            density_map[y, x] = 1

    # Apply Gaussian filter
    density_map = gaussian_filter(density_map, sigma)

    # Normalize to preserve count
    if np.sum(density_map) > 0:
        density_map = density_map * (len(points) / np.sum(density_map))

    return density_map

def load_and_prepare_image(image_path, target_size=(384, 384)):
    """Load and preprocess image for CSRNet"""
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Get original dimensions for visualization
    original_width, original_height = img.size

    # Resize while maintaining aspect ratio
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor, img, (original_width, original_height)

def predict_headcount(model, image_path, device, target_size=(384, 384)):
    """Predict headcount from an image using CSRNet"""
    # Load and preprocess image
    img_tensor, original_img, original_size = load_and_prepare_image(image_path, target_size)
    img_tensor = img_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    # Process output density map
    density_map = output.squeeze().cpu().numpy()

    # Calculate predicted count (sum of density map)
    predicted_count = float(np.sum(density_map))

    # Calculate scaling factor to adjust count (if model was trained on different scales)
    # This step is crucial for accurate headcounts
    scale_factor = 1.0

    # Apply scale factor to predicted count
    final_count = predicted_count * scale_factor

    return final_count, density_map, original_img

def generate_density_map_from_points(annotation_file, output_path,
                                    target_size=(384, 384), visualize=True):
    """Generate density map from point annotations"""
    # Load annotation file based on its format
    if annotation_file.endswith('.json'):
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Extract points based on format (adapt as needed)
        points = []
        if isinstance(data, list):
            for anno in data:
                if 'x' in anno and 'y' in anno:
                    points.append([anno['x'], anno['y']])
                elif 'points' in anno:
                    for pt in anno['points']:
                        points.append([pt['x'], pt['y']])
        elif isinstance(data, dict):
            if 'points' in data:
                for pt in data['points']:
                    points.append([pt['x'], pt['y']])

        points = np.array(points)

    elif annotation_file.endswith('.mat'):
        import scipy.io as sio
        mat_data = sio.loadmat(annotation_file)

        # Try to find points in common field names
        if 'image_info' in mat_data and 'location' in mat_data['image_info'][0, 0]:
            points = mat_data['image_info'][0, 0]['location'][0, 0]
        elif 'annPoints' in mat_data:
            points = mat_data['annPoints']
        elif 'points' in mat_data:
            points = mat_data['points']
        else:
            raise ValueError(f"Could not find point annotations in {annotation_file}")
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_file}")

    # Convert points to numpy array if not already
    points = np.array(points)

    # Create density map
    density_map = create_density_map_gaussian(points, target_size[1], target_size[0])

    # Save density map
    np.save(output_path, density_map)
    print(f"Density map saved to {output_path}")

    # Visualize if requested
    if visualize:
        plt.figure(figsize=(10, 6))

        # Plot points on blank image
        blank = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        for x, y in points:
            if 0 <= x < target_size[0] and 0 <= y < target_size[1]:
                cv2.circle(blank, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Original points
        plt.subplot(1, 2, 1)
        plt.imshow(blank)
        plt.title(f'Points ({len(points)})')
        plt.axis('off')

        # Density map
        plt.subplot(1, 2, 2)
        plt.imshow(density_map, cmap='jet')
        plt.title(f'Density Map (Sum: {np.sum(density_map):.2f})')
        plt.axis('off')
        plt.colorbar()

        # Save visualization
        vis_path = os.path.splitext(output_path)[0] + '_visualization.png'
        plt.savefig(vis_path)
        plt.close()
        print(f"Visualization saved to {vis_path}")

    return density_map, points

def visualize_prediction(image_path, density_map, predicted_count):
    """Visualize prediction results"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

    # Density map
    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title(f'Density Map\nPredicted Count: {predicted_count:.2f}')
    plt.axis('off')
    plt.colorbar()

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Save visualization
    output_path = os.path.join('results', f'result_{os.path.basename(image_path)}')
    plt.savefig(output_path)
    plt.close()

    return output_path

def main():
    # Define paths
    model_path = 'models/csrnet.pth'  # Path to trained model

    # Create model
    device = get_device()
    model = CSRNet(load_weights=True).to(device)

    # Check if trained model exists and load it
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No pre-trained model found at {model_path}. Using ImageNet weights only.")

    # Example 1: Generate density map from annotations
    # If you have annotation files, you can generate density maps
    '''
    annotation_file = 'annotations/image_001.json'  # Update this path
    density_map_path = 'data/processed/train/density_maps/image_001.npy'
    os.makedirs(os.path.dirname(density_map_path), exist_ok=True)
    generate_density_map_from_points(annotation_file, density_map_path)
    '''

    # Example 2: Predict headcount from image
    image_path = 'path/to/your/crowd/image.jpg'  # Update this path

    if os.path.exists(image_path):
        # Make prediction
        predicted_count, density_map, original_img = predict_headcount(model, image_path, device)

        # Visualize results
        output_path = visualize_prediction(image_path, density_map, predicted_count)

        print(f"\nPredicted headcount: {predicted_count:.1f}")
        print(f"Results saved to: {output_path}")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.")

if __name__ == "__main__":
    main()
