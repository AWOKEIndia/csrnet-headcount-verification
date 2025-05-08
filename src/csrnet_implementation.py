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
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            # Load pretrained VGG16 weights
            vgg16 = models.vgg16(pretrained=True)
            self._initialize_weights()

            # Copy VGG16 weights to frontend
            vgg_frontend_dict = dict([(name, param) for name, param in vgg16.named_parameters()])
            frontend_state_dict = self.frontend.state_dict()

            for name, param in frontend_state_dict.items():
                if name in vgg_frontend_dict:
                    frontend_state_dict[name].copy_(vgg_frontend_dict[name])

            self.frontend.load_state_dict(frontend_state_dict)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
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
    def __init__(self, image_root, density_map_root, transform=None):
        self.image_root = image_root
        self.density_map_root = density_map_root
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
        self.density_map_files = sorted(glob.glob(os.path.join(density_map_root, '*.npy')))

        # Sanity check
        assert len(self.image_files) == len(self.density_map_files), "Number of images and density maps don't match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        density_map = np.load(self.density_map_files[idx])

        if self.transform is not None:
            img = self.transform(img)

        density_map = torch.from_numpy(density_map).float().unsqueeze(0)

        return img, density_map


def create_density_map_gaussian(points, height, width, sigma=15):
    """
    Generate density map based on head point annotations
    using Gaussian kernels
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])
        if x < width and y < height:
            # Generate a Gaussian kernel for each point
            gaussian_kernel = np.zeros((height, width), dtype=np.float32)
            gaussian_kernel[y, x] = 1
            gaussian_kernel = cv2.GaussianBlur(gaussian_kernel, (sigma, sigma), 0)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # Normalize

            density_map += gaussian_kernel

    return density_map


def train(model, train_loader, optimizer, epoch, device):
    """
    Training function
    """
    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # MSE Loss
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss /= len(train_loader)
    print(f'Train Epoch: {epoch}, Average Loss: {train_loss:.6f}')
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
    Make prediction on a single image
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

    # Get the predicted count
    count = output.sum().item()

    # Visualize the result
    output_np = output.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(output_np, cmap='jet')
    plt.title(f'Predicted Density Map (Count: {count:.2f})')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.close()

    return count, output_np


def main():
    parser = argparse.ArgumentParser(description='CSRNet for Crowd Counting')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='save the current model')
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test mode')
    parser.add_argument('--test-image', type=str, default=None, help='path to test image')
    parser.add_argument('--load-model', type=str, default=None, help='path to saved model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create model
    model = CSRNet(load_weights=True).to(device)

    # Load model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print(f"Loaded model from {args.load_model}")

    if args.mode == 'train':
        # Data transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = CrowdDataset(
            os.path.join(args.data_path, 'train', 'images'),
            os.path.join(args.data_path, 'train', 'density_maps'),
            transform=transform
        )

        val_dataset = CrowdDataset(
            os.path.join(args.data_path, 'val', 'images'),
            os.path.join(args.data_path, 'val', 'density_maps'),
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_mae = float('inf')

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, epoch, device)
            val_loss, mae = validate(model, val_loader, device)

            # Save the best model
            if mae < best_mae and args.save_model:
                best_mae = mae
                torch.save(model.state_dict(), 'csrnet_best.pth')
                print(f"Saved best model with MAE: {best_mae:.2f}")

            # Save checkpoint
            if args.save_model and epoch % 5 == 0:
                torch.save(model.state_dict(), f'csrnet_epoch_{epoch}.pth')

        # Save final model
        if args.save_model:
            torch.save(model.state_dict(), 'csrnet_final.pth')

    elif args.mode == 'test' and args.test_image:
        count, density_map = predict(model, args.test_image, device)
        print(f"Predicted count: {count:.2f}")


if __name__ == '__main__':
    main()
