# CSRNet Headcount Verification System

This project implements a headcount verification system based on CSRNet (Congested Scene Recognition Network), a deep learning model specifically designed for crowd counting in images and videos.

## Overview

The CSRNet Headcount Verification System allows you to:

1. Count people in images with high accuracy
2. Generate density maps that visualize crowd distribution
3. Process individual images or batches of images
4. Automatically utilize the best available hardware (CUDA GPU, Apple Metal, or CPU)
5. Monitor training progress with detailed logging and metrics
6. Calibrate model predictions for improved accuracy

## Key Features

- **Enhanced Model Architecture**:
  - Spatial attention mechanism for better head detection
  - Multi-scale feature processing
  - Improved batch normalization
  - Adaptive density map generation
  - Enhanced weight initialization

- **Advanced Training Features**:
  - Combined L1 and L2 loss for better accuracy
  - Adaptive learning rate scheduling
  - Memory-efficient training pipeline
  - In-memory dataset caching
  - Comprehensive performance monitoring

- **Optimized Data Processing**:
  - Efficient data loading with prefetching
  - Enhanced data augmentation
  - Adaptive density map scaling
  - Improved preprocessing pipeline

- **Comprehensive Evaluation**:
  - Multiple evaluation metrics (MAE, MSE, RMSE)
  - Detailed visualization of results
  - Performance monitoring and logging
  - Memory usage tracking

## Requirements

- Python 3.6+
- PyTorch 2.0+
- Hardware Support:
  - NVIDIA GPU with CUDA support (recommended for best performance)
  - Apple Silicon (M1/M2/M3) with Metal support
  - CPU (fallback option)
- Additional dependencies (see Installation section)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/csrnet-headcount-verification.git
cd csrnet-headcount-verification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## File Structure

```
csrnet-headcount-verification/
├── data/
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── density_maps/
│   │   └── val/
│   │       ├── images/
│   │       └── density_maps/
├── models/                       # Saved models directory
├── logs/                         # Log files directory
├── src/
│   ├── headcount_solution.py     # Core CSRNet model implementation
│   ├── headcount_training.py     # Training pipeline
│   ├── headcount_inference.py    # Inference and visualization
│   ├── headcount_calibration.py  # Model calibration
│   ├── prepare_shanghai_dataset.py # Dataset preparation
│   └── dateset_path_checker.py   # Dataset validation
└── requirements.txt              # Dependencies file
```

## Usage

### Training the Model

```bash
python src/headcount_training.py --train
```

Optional training parameters:
- `--model_path`: Path to save the model (default: 'models/csrnet_improved.pth')
- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-5)
- `--data_root`: Root directory for training data (default: 'data/processed')
- `--resume`: Resume training from checkpoint
- `--seed`: Random seed for reproducibility (default: 42)

Example with custom parameters:
```bash
python src/headcount_training.py \
    --train \
    --model_path models/custom_model.pth \
    --batch_size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --data_root data/custom_dataset
```

### Running Inference

```bash
python src/headcount_inference.py --image_path path/to/image.jpg
```

Optional inference parameters:
- `--model_path`: Path to trained model
- `--output_dir`: Directory to save results (default: 'results')
- `--calibration_factor`: Calibration factor for predictions (default: 1.0)
- `--display`: Display results interactively
- `--batch`: Process all images in a directory

Example with custom parameters:
```bash
python src/headcount_inference.py \
    --image_path path/to/image.jpg \
    --model_path models/custom_model.pth \
    --output_dir results/custom \
    --calibration_factor 1.2 \
    --display
```

### Model Calibration

```bash
python src/headcount_calibration.py --calibration_data path/to/calibration_data
```

### Dataset Preparation

```bash
python src/prepare_shanghai_dataset.py --data_root path/to/raw_data
```

## Model Architecture

The enhanced CSRNet implementation includes:

1. **Spatial Attention Mechanism**:
   - Multi-scale attention for better feature extraction
   - Channel-wise attention for feature refinement
   - Adaptive attention based on crowd density

2. **Enhanced Feature Processing**:
   - Improved VGG16 frontend
   - Dilated convolutions in backend
   - Better batch normalization
   - Adaptive density map generation

3. **Loss Function**:
   - Combined L1 and L2 loss
   - Adaptive weighting of loss components
   - Improved gradient handling

4. **Data Augmentation**:
   - Random horizontal flips
   - Random rotations
   - Color jittering
   - Random affine transformations
   - Random perspective changes
   - Gaussian blur

## Performance Monitoring

The system includes comprehensive performance monitoring:

1. **Training Metrics**:
   - Loss values (training and validation)
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - Learning rate tracking

2. **System Metrics**:
   - Memory usage
   - Batch processing time
   - Epoch completion time
   - Device utilization

3. **Logging**:
   - Detailed console output
   - Log file generation
   - Performance metrics visualization
   - Error tracking and debugging

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   - Reduce batch size
   - Decrease cache size
   - Use memory-efficient settings

2. **Training Stability**:
   - Adjust learning rate
   - Modify loss weights
   - Check batch normalization

3. **Performance Issues**:
   - Enable hardware acceleration
   - Optimize data loading
   - Use appropriate batch size

4. **Logging Issues**:
   - Check logs directory permissions
   - Verify console output settings
   - Monitor log file size

## References

- [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062)
- [PyTorch](https://pytorch.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please open an issue in this repository.
