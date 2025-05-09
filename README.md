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
  - Attention mechanisms for better head detection
  - Batch normalization for improved training stability
  - Adaptive density map generation
  - Multi-scale feature processing

- **Advanced Training Features**:
  - In-memory dataset caching for faster training
  - Adaptive learning rate scheduling
  - Early stopping with configurable patience
  - Combined loss function (MSE, L1, and gradient loss)
  - Memory-efficient training pipeline

- **Optimized Data Processing**:
  - Efficient data loading with prefetching
  - Persistent workers for faster training
  - Pinned memory for faster GPU transfer
  - Advanced data augmentation techniques

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
├── csrnet_implementation.py   # Enhanced CSRNet model implementation
├── prepare_shanghai_dataset.py # Script for preparing ShanghaiTech dataset
├── model_evaluation.py        # Tools for evaluating model performance
├── video_processing.py        # Video and stream processing capabilities
├── gui_application.py         # Graphical user interface
├── requirements.txt           # Required Python packages
├── data/                      # Dataset directory
│   ├── ShanghaiTech/
│   │   ├── part_A/           # Part A dataset
│   │   └── part_B/           # Part B dataset
└── README.md                  # This file
```

## Usage

### Command Line Interface

#### Training the Model

```bash
python src/csrnet_implementation.py \
    --mode train \
    --data-path data/processed \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 8 \
    --cache-size 2000 \
    --prefetch-factor 3 \
    --patience 15 \
    --pin-memory \
    --persistent-workers
```

Key training parameters:
- `--cache-size`: Number of images to cache in memory (default: 1000)
- `--prefetch-factor`: Number of batches to prefetch (default: 2)
- `--patience`: Early stopping patience (default: 10)
- `--pin-memory`: Enable pinned memory for faster GPU transfer
- `--persistent-workers`: Use persistent workers for data loading

#### Evaluating the Model

```bash
python model_evaluation.py --model path/to/model.pth --mode evaluate --test-dir path/to/test/data --output-dir path/to/results
```

The system will automatically detect and use the best available hardware:
- NVIDIA GPU with CUDA support (if available)
- Apple Silicon GPU with Metal support (if available)
- CPU (as fallback)

#### Processing a Single Image

```bash
python model_evaluation.py --model path/to/model.pth --mode single --image path/to/image.jpg --output-dir path/to/results
```

The output will include:
- Original image
- Density map visualization
- Detected heads with confidence scores

#### Batch Processing Images

```bash
python model_evaluation.py --model path/to/model.pth --mode batch --image-dir path/to/images --output-dir path/to/results
```

#### Video Processing

```bash
python video_processing.py --model path/to/model.pth --mode video --input path/to/video.mp4 --output path/to/output.mp4
```

For webcam:
```bash
python video_processing.py --model path/to/model.pth --mode webcam
```

For RTSP stream:
```bash
python video_processing.py --model path/to/model.pth --mode rtsp --input "rtsp://your-stream-url"
```

### Training Your Own Model

#### Data Preparation

1. Download and extract the ShanghaiTech dataset
2. Organize the dataset structure as shown in the File Structure section
3. Run the dataset preparation script:
```bash
python prepare_shanghai_dataset.py --dataset-path data/ShanghaiTech --output-path data/processed --part A --visualize
```

#### Training Process

1. Train the model with optimized settings:
```bash
python csrnet_implementation.py \
    --mode train \
    --data-path data/processed \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 8 \
    --cache-size 2000 \
    --prefetch-factor 3 \
    --patience 15
```

2. Monitor training progress:
   - Training and validation loss
   - MAE (Mean Absolute Error)
   - Memory usage
   - Batch processing time

3. The model will automatically:
   - Save the best model based on MAE
   - Apply early stopping if no improvement
   - Clear cache periodically to manage memory
   - Use the best available hardware

## Model Architecture

The enhanced CSRNet implementation includes:

1. **Attention Mechanism**:
   - Multi-scale attention modules
   - Channel-wise attention for feature refinement
   - Spatial attention for head localization

2. **Batch Normalization**:
   - Applied at multiple network stages
   - Improves training stability
   - Better feature normalization

3. **Loss Function**:
   - Combined MSE and L1 loss
   - Gradient loss for better density map quality
   - Adaptive weighting of loss components

4. **Data Augmentation**:
   - Random horizontal flips
   - Random rotations
   - Color jittering
   - Random affine transformations
   - Random perspective changes
   - Gaussian blur

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   - Reduce cache size
   - Decrease batch size
   - Use fewer workers
   - Enable periodic cache clearing

2. **Training Stability**:
   - Adjust learning rate
   - Modify patience for early stopping
   - Tune loss weights
   - Check batch normalization layers

3. **Performance Issues**:
   - Increase prefetch factor
   - Enable pinned memory
   - Use persistent workers
   - Optimize data loading pipeline

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
