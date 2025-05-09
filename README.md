# CSRNet Headcount Verification System

This project implements a headcount verification system based on CSRNet (Congested Scene Recognition Network), a deep learning model specifically designed for crowd counting in images and videos.

## Overview

The CSRNet Headcount Verification System allows you to:

1. Count people in images and videos with high accuracy
2. Generate density maps that visualize crowd distribution
3. Process individual images, video files, webcam feeds, or RTSP streams
4. Batch process multiple images
5. Use a convenient GUI interface for all functionality
6. Automatically utilize the best available hardware (CUDA GPU, Apple Metal, or CPU)

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
  - Early stopping with configurable patience
  - Memory-efficient training pipeline
  - In-memory dataset caching

- **Optimized Data Processing**:
  - Efficient data loading with prefetching
  - Enhanced data augmentation
  - Adaptive density map scaling
  - Improved preprocessing pipeline

- **Comprehensive Evaluation**:
  - Multiple evaluation metrics (MAE, MSE, RMSE, MAPE, R²)
  - Detailed visualization of results
  - Error analysis and distribution plots
  - Performance monitoring

## Requirements

- Python 3.6+
- PyTorch 1.7+
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

4. Download the Shanghai CSRNet Dataset:
   - Visit the [ShanghaiTech Dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
   - Download both Part A and Part B datasets
   - Extract the datasets to your project directory

## File Structure

```
├── src/
│   ├── csrnet_implementation.py   # Enhanced CSRNet model implementation
│   ├── model_evaluation.py        # Model evaluation and metrics
│   ├── video_processing.py        # Video and stream processing
│   ├── gui_application.py         # Graphical user interface
│   └── prepare_shanghai_dataset.py # Dataset preparation script
├── data/                         # Dataset directory
│   ├── train/                    # Training data
│   │   ├── images/              # Training images
│   │   └── density_maps/        # Training density maps
│   └── val/                      # Validation data
│       ├── images/              # Validation images
│       └── density_maps/        # Validation density maps
├── models/                       # Saved models directory
├── logs/                         # Log files directory
├── evaluation_results/          # Evaluation results directory
└── README.md                    # This file
```

## Usage

### Training the Model

```bash
python src/csrnet_implementation.py \
    --train \
    --model_path models/csrnet.pth \
    --batch_size 8 \
    --epochs 200 \
    --lr 1e-5
```

Key training parameters:
- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Learning rate (default: 1e-5)

### Evaluating the Model

```bash
python src/model_evaluation.py \
    --model_path models/csrnet.pth \
    --test_dir data/test \
    --output_dir evaluation_results
```

The evaluation will generate:
- Detailed metrics (MAE, MSE, RMSE, MAPE, R²)
- Error distribution plots
- Prediction vs ground truth scatter plots
- Individual image results

### Processing a Single Image

```bash
python src/model_evaluation.py \
    --model_path models/csrnet.pth \
    --image_path path/to/image.jpg \
    --output_dir evaluation_results
```

The output will include:
- Original image
- Predicted density map
- Head count estimate

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

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

1. **MAE (Mean Absolute Error)**:
   - Measures average absolute difference between predicted and actual counts
   - Lower values indicate better accuracy

2. **MSE (Mean Squared Error)**:
   - Measures average squared difference
   - More sensitive to larger errors

3. **RMSE (Root Mean Squared Error)**:
   - Square root of MSE
   - In same units as original counts

4. **MAPE (Mean Absolute Percentage Error)**:
   - Measures relative accuracy
   - Expressed as percentage

5. **R² Score**:
   - Measures proportion of variance explained
   - Range: 0 to 1 (higher is better)

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

## References

- [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062)
- [ShanghaiTech Dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
- [PyTorch](https://pytorch.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please open an issue in this repository.
