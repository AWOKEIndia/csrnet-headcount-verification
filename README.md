# CSRNet Headcount Verification System

This project implements a headcount verification system based on CSRNet (Congested Scene Recognition Network), a deep learning model specifically designed for crowd counting in images and videos.

## Overview

The CSRNet Headcount Verification System allows you to:

1. Count people in images and videos with high accuracy
2. Generate density maps that visualize crowd distribution
3. Process individual images, video files, webcam feeds, or RTSP streams
4. Batch process multiple images
5. Use a convenient GUI interface for all functionality

## Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA-enabled GPU (recommended for better performance)
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
├── csrnet_implementation.py   # CSRNet model implementation
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

#### Preparing a Dataset

To prepare a dataset for training the model:

```bash
python prepare_shanghai_dataset.py --dataset-path data/ShanghaiTech --output-path data/processed --part A --visualize
```

The raw data should contain:
- `images/` folder with image files
- `annotations/` folder with corresponding annotation files (JSON format)

#### Training the Model

```bash
python csrnet_implementation.py --mode train --data-path data/processed --epochs 50 --batch-size 8 --lr 1e-5
```

#### Evaluating the Model

```bash
python model_evaluation.py --model path/to/model.pth --mode evaluate --test-dir path/to/test/data --output-dir path/to/results
```

#### Processing a Single Image

```bash
python model_evaluation.py --model path/to/model.pth --mode single --image path/to/image.jpg --output-dir path/to/results
```

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

### Graphical User Interface

Launch the GUI application:

```bash
python gui_application.py --model path/to/model.pth
```

The GUI provides four tabs:
1. **Image Processing** - Process individual images
2. **Video Processing** - Process video files, webcam feeds, or RTSP streams
3. **Batch Processing** - Process multiple images in a directory
4. **Settings** - Configure model and application settings

## Training Your Own Model

### Data Preparation

1. Download and extract the ShanghaiTech dataset
2. Organize the dataset structure as shown in the File Structure section
3. Run the dataset preparation script to generate density maps:
```bash
python prepare_shanghai_dataset.py --dataset-path data/ShanghaiTech --output-path data/processed --part A --visualize
```
   - For Part B dataset, use `--part B`
   - Add `--adaptive` flag to use adaptive Gaussian kernels
   - Add `--visualize` flag to generate visualization of density maps

4. Verify the generated density maps match the annotations

### Training Process

1. Train the model using the training script:
```bash
python csrnet_implementation.py --mode train --data-path data/processed --epochs 50 --batch-size 8 --lr 1e-5
```
2. Monitor the loss and MAE (Mean Absolute Error) values
3. Save checkpoints during training
4. Select the best model based on validation performance

## Pre-trained Model

A pre-trained model is available for download:
- [CSRNet Pre-trained Model](https://drive.google.com/file/d/your-model-file/view)
- Trained on ShanghaiTech dataset Part A and Part B
- Handles diverse crowd scenes from sparse to dense crowds

## Extending the System

### Supporting New Annotation Formats

Edit the `parse_annotations` function in `prepare_shanghai_dataset.py` to support your annotation format.

### Customizing the Model Architecture

Modify the `CSRNet` class in `csrnet_implementation.py` to adjust the network architecture.

### Adding New Features

The modular design allows for easy extension:
- Add new processing modes to `video_processing.py`
- Enhance the GUI by modifying `gui_application.py`
- Implement additional analysis tools in `model_evaluation.py`

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use input images with lower resolution
   - Process video at lower frame rates

2. **Poor counting accuracy**
   - Ensure proper data preparation with accurate annotations
   - Train for more epochs
   - Adjust learning rate
   - Try using a pre-trained VGG16 backbone

3. **Low performance on video**
   - Reduce processing resolution
   - Skip frames (process every Nth frame)
   - Use a more powerful GPU

## References

- [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) - Original paper by Y. Li et al.
- [ShanghaiTech Dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf) - Dataset used for training and evaluation
- [PyTorch](https://pytorch.org/) - Deep learning framework used in this implementation

## Acknowledgements

- The CSRNet implementation is based on the original paper by Y. Li, X. Zhang, and D. Chen
- The density map generation approach follows established methods in crowd counting literature
- Special thanks to the crowd counting research community for advancing this field

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original CSRNet paper:

```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1091--1100},
  year={2018}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

If you have any questions or feedback, please open an issue in this repository.
