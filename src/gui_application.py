import os
import sys
import torch
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QTabWidget, QComboBox, QSpinBox, QCheckBox,
                            QGroupBox, QFormLayout, QLineEdit, QMessageBox,
                            QProgressBar, QTextEdit, QScrollArea, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
from torchvision import transforms
import threading
import time
import argparse

# Import CSRNet model
# Make sure to have the CSRNet implementation file in the same directory
from csrnet_implementation import CSRNet


class ProcessThread(QThread):
    """
    Thread for processing images without freezing GUI
    """
    update_signal = pyqtSignal(object, object, float)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)

    def __init__(self, model, image_path, device):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.device = device

    def run(self):
        try:
            # Image transform
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Load image
            img = Image.open(self.image_path).convert('RGB')

            # Progress update
            self.progress_signal.emit(20)

            # Convert to tensor
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            # Progress update
            self.progress_signal.emit(40)

            # Make prediction
            with torch.no_grad():
                predicted_density_map = self.model(img_tensor).squeeze().cpu().numpy()

            # Progress update
            self.progress_signal.emit(80)

            # Calculate predicted count
            pred_count = np.sum(predicted_density_map)

            # Progress update
            self.progress_signal.emit(100)

            # Emit signal with results
            self.update_signal.emit(np.array(img), predicted_density_map, pred_count)

        except Exception as e:
            print(f"Error in thread: {e}")
        finally:
            self.finished_signal.emit()


class VideoProcessThread(QThread):
    """
    Thread for processing video streams
    """
    frame_signal = pyqtSignal(object, object, float)
    finished_signal = pyqtSignal()

    def __init__(self, model, video_source, device):
        super().__init__()
        self.model = model
        self.video_source = video_source
        self.device = device
        self.running = True

    def run(self):
        try:
            # Image transform
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Open video capture
            if isinstance(self.video_source, int) or self.video_source.isdigit():
                # If input is a number, treat as webcam
                cap = cv2.VideoCapture(int(self.video_source))
            else:
                # Otherwise, treat as video file path
                cap = cv2.VideoCapture(self.video_source)

            if not cap.isOpened():
                print(f"Error: Could not open video source {self.video_source}")
                self.finished_signal.emit()
                return

            frame_count = 0

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame (every 3rd frame to improve performance)
                if frame_count % 3 == 0:
                    # Convert frame to PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Process image
                    img_tensor = transform(pil_image).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        predicted_density_map = self.model(img_tensor).squeeze().cpu().numpy()

                    # Calculate predicted count
                    pred_count = np.sum(predicted_density_map)

                    # Convert frame to RGB for Qt
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Emit signal with results
                    self.frame_signal.emit(rgb_frame, predicted_density_map, pred_count)

                frame_count += 1

                # Small delay to prevent GUI from being overloaded
                time.sleep(0.01)

            # Release video capture
            cap.release()

        except Exception as e:
            print(f"Error in video thread: {e}")
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.running = False


class BatchProcessThread(QThread):
    """
    Thread for batch processing images
    """
    progress_signal = pyqtSignal(int, str)
    result_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()

    def __init__(self, model, image_dir, output_dir, device):
        super().__init__()
        self.model = model
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.device = device

    def run(self):
        try:
            # Image transform
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Get all image files
            image_files = []
            for ext in ['jpg', 'jpeg', 'png', 'bmp']:
                image_files.extend([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
                                    if f.lower().endswith(f'.{ext}')])

            total_files = len(image_files)
            if total_files == 0:
                self.progress_signal.emit(100, "No image files found!")
                self.finished_signal.emit()
                return

            results = []

            # Process each image
            for i, image_file in enumerate(image_files):
                try:
                    img_name = os.path.basename(image_file)
                    self.progress_signal.emit(int(100 * i / total_files), f"Processing {img_name}...")

                    # Load image
                    img = Image.open(image_file).convert('RGB')

                    # Convert to tensor
                    img_tensor = transform(img).unsqueeze(0).to(self.device)

                    # Make prediction
                    with torch.no_grad():
                        predicted_density_map = self.model(img_tensor).squeeze().cpu().numpy()

                    # Calculate predicted count
                    pred_count = np.sum(predicted_density_map)

                    # Save result
                    results.append({
                        'image': image_file,
                        'count': float(pred_count)
                    })

                    # Save visualization
                    plt.figure(figsize=(12, 5))

                    # Original image
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.array(img))
                    plt.title('Original Image')
                    plt.axis('off')

                    # Density map
                    plt.subplot(1, 2, 2)
                    plt.imshow(predicted_density_map, cmap='jet')
                    plt.title(f'Density Map (Count: {pred_count:.2f})')
                    plt.colorbar()
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f"{os.path.splitext(img_name)[0]}_result.png"))
                    plt.close()

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

            # Save results to CSV
            with open(os.path.join(self.output_dir, 'results.csv'), 'w') as f:
                f.write("image,count\n")
                for result in results:
                    f.write(f"{os.path.basename(result['image'])},{result['count']:.2f}\n")

            # Signal completion
            self.result_signal.emit(results)
            self.progress_signal.emit(100, f"Completed processing {total_files} images")

        except Exception as e:
            print(f"Error in batch thread: {e}")
            self.progress_signal.emit(100, f"Error: {str(e)}")
        finally:
            self.finished_signal.emit()


class HeadcountApp(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_path = None
        self.video_thread = None

        self.setWindowTitle("CSRNet Headcount Verification")
        self.setGeometry(100, 100, 1200, 800)

        self.init_ui()

        # Load model if provided
        if model_path:
            self.load_model(model_path)

    def init_ui(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tabs for different functionalities
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.create_image_tab()
        self.create_video_tab()
        self.create_batch_tab()
        self.create_settings_tab()

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_image_tab(self):
        # Image processing tab
        image_tab = QWidget()
        layout = QVBoxLayout(image_tab)

        # Controls
        controls_layout = QHBoxLayout()

        # Image selection
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("Select an image...")

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_image)

        controls_layout.addWidget(QLabel("Image:"))
        controls_layout.addWidget(self.image_path_edit, 1)
        controls_layout.addWidget(browse_btn)

        # Process button
        process_btn = QPushButton("Process Image")
        process_btn.clicked.connect(self.process_image)
        controls_layout.addWidget(process_btn)

        layout.addLayout(controls_layout)

        # Progress bar
        self.image_progress = QProgressBar()
        self.image_progress.setVisible(False)
        layout.addWidget(self.image_progress)

        # Display area
        self.image_display = QWidget()
        self.image_layout = QHBoxLayout(self.image_display)

        # Original image
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setText("Original Image")
        self.original_image_label.setStyleSheet("border: 1px solid #cccccc;")
        self.original_image_label.setMinimumSize(400, 300)

        # Density map
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)

        # Result label
        self.count_label = QLabel("Count: --")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("font-size: 16pt; font-weight: bold;")

        # Layout for original image
        orig_layout = QVBoxLayout()
        orig_layout.addWidget(QLabel("Original Image"))
        orig_layout.addWidget(self.original_image_label, 1)

        # Layout for density map
        density_layout = QVBoxLayout()
        density_layout.addWidget(QLabel("Density Map"))
        density_layout.addWidget(self.canvas, 1)

        self.image_layout.addLayout(orig_layout)
        self.image_layout.addLayout(density_layout)

        # Create a scroll area for the display
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_display)

        layout.addWidget(scroll_area, 1)
        layout.addWidget(self.count_label)

        # Add tab
        self.tabs.addTab(image_tab, "Image Processing")

    def create_video_tab(self):
        # Video processing tab
        video_tab = QWidget()
        layout = QVBoxLayout(video_tab)

        # Controls
        controls_group = QGroupBox("Video Controls")
        controls_layout = QHBoxLayout(controls_group)

        # Source selection
        self.video_source_combo = QComboBox()
        self.video_source_combo.addItem("Webcam", 0)
        self.video_source_combo.addItem("Video File", 1)
        self.video_source_combo.addItem("RTSP Stream", 2)
        self.video_source_combo.currentIndexChanged.connect(self.update_video_source_ui)

        # Source path/URL
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setEnabled(False)

        # Browse button
        self.video_browse_btn = QPushButton("Browse")
        self.video_browse_btn.clicked.connect(self.browse_video)
        self.video_browse_btn.setEnabled(False)

        # Start/Stop button
        self.video_start_btn = QPushButton("Start")
        self.video_start_btn.clicked.connect(self.toggle_video_processing)

        controls_layout.addWidget(QLabel("Source:"))
        controls_layout.addWidget(self.video_source_combo)
        controls_layout.addWidget(self.video_path_edit, 1)
        controls_layout.addWidget(self.video_browse_btn)
        controls_layout.addWidget(self.video_start_btn)

        layout.addWidget(controls_group)

        # Display area
        self.video_display = QWidget()
        self.video_layout = QHBoxLayout(self.video_display)

        # Video frame
        self.video_frame_label = QLabel()
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setText("Video Feed")
        self.video_frame_label.setStyleSheet("border: 1px solid #cccccc;")
        self.video_frame_label.setMinimumSize(500, 400)

        # Density map for video
        self.video_figure = Figure(figsize=(5, 4), dpi=100)
        self.video_canvas = FigureCanvas(self.video_figure)
        self.video_canvas.setMinimumSize(500, 400)

        # Video count label
        self.video_count_label = QLabel("Count: --")
        self.video_count_label.setAlignment(Qt.AlignCenter)
        self.video_count_label.setStyleSheet("font-size: 16pt; font-weight: bold;")

        # Layout for video frame
        vid_layout = QVBoxLayout()
        vid_layout.addWidget(QLabel("Video Feed"))
        vid_layout.addWidget(self.video_frame_label, 1)

        # Layout for density map
        vid_density_layout = QVBoxLayout()
        vid_density_layout.addWidget(QLabel("Real-time Density Map"))
        vid_density_layout.addWidget(self.video_canvas, 1)

        self.video_layout.addLayout(vid_layout)
        self.video_layout.addLayout(vid_density_layout)

        # Create a scroll area for the display
        video_scroll_area = QScrollArea()
        video_scroll_area.setWidgetResizable(True)
        video_scroll_area.setWidget(self.video_display)

        layout.addWidget(video_scroll_area, 1)
        layout.addWidget(self.video_count_label)

        # Add tab
        self.tabs.addTab(video_tab, "Video Processing")

    def create_batch_tab(self):
        # Batch processing tab
        batch_tab = QWidget()
        layout = QVBoxLayout(batch_tab)

        # Controls
        controls_group = QGroupBox("Batch Processing")
        form_layout = QFormLayout(controls_group)

        # Input folder
        self.batch_input_edit = QLineEdit()
        self.batch_input_edit.setReadOnly(True)
        browse_input_btn = QPushButton("Browse")
        browse_input_btn.clicked.connect(self.browse_batch_input)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.batch_input_edit, 1)
        input_layout.addWidget(browse_input_btn)
        form_layout.addRow("Input Folder:", input_layout)

        # Output folder
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setReadOnly(True)
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_batch_output)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.batch_output_edit, 1)
        output_layout.addWidget(browse_output_btn)
        form_layout.addRow("Output Folder:", output_layout)

        # Start button
        self.batch_start_btn = QPushButton("Start Batch Processing")
        self.batch_start_btn.clicked.connect(self.start_batch_processing)

        layout.addWidget(controls_group)
        layout.addWidget(self.batch_start_btn)

        # Progress indicator
        self.batch_progress = QProgressBar()
        self.batch_status_label = QLabel("Ready")

        layout.addWidget(self.batch_progress)
        layout.addWidget(self.batch_status_label)

        # Results area
        self.batch_results = QTextEdit()
        self.batch_results.setReadOnly(True)

        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.batch_results, 1)

        # Add tab
        self.tabs.addTab(batch_tab, "Batch Processing")

    def create_settings_tab(self):
        # Settings tab
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)

        # Model controls
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout(model_group)

        # Model path
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)

        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self.browse_model)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model_from_ui)

        model_path_layout.addWidget(QLabel("Model Path:"))
        model_path_layout.addWidget(self.model_path_edit, 1)
        model_path_layout.addWidget(browse_model_btn)
        model_path_layout.addWidget(load_model_btn)

        model_layout.addLayout(model_path_layout)

        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Computation Device:")

        self.device_combo = QComboBox()
        self.device_combo.addItem("CUDA (GPU)", "cuda")
        self.device_combo.addItem("CPU", "cpu")

        # Set default based on availability
        default_index = 0 if torch.cuda.is_available() else 1
        self.device_combo.setCurrentIndex(default_index)
        self.device_combo.currentIndexChanged.connect(self.update_device)

        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch(1)

        model_layout.addLayout(device_layout)

        # Model status
        self.model_status_label = QLabel("Model Status: Not Loaded")
        model_layout.addWidget(self.model_status_label)

        layout.addWidget(model_group)

        # Application settings
        app_group = QGroupBox("Application Settings")
        app_layout = QFormLayout(app_group)

        # Video processing frame rate limit
        self.fps_limit_spin = QSpinBox()
        self.fps_limit_spin.setRange(1, 60)
        self.fps_limit_spin.setValue(15)
        app_layout.addRow("Target FPS:", self.fps_limit_spin)

        # Save results option
        self.save_results_check = QCheckBox("Save Results Automatically")
        self.save_results_check.setChecked(True)
        app_layout.addRow("", self.save_results_check)

        layout.addWidget(app_group)
        layout.addStretch(1)

        # Add tab
        self.tabs.addTab(settings_tab, "Settings")

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.image_path_edit.setText(file_path)

    def browse_video(self):
        if self.video_source_combo.currentData() == 1:  # Video file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                self.video_path_edit.setText(file_path)

    def browse_batch_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.batch_input_edit.setText(folder)

    def browse_batch_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.batch_output_edit.setText(folder)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Model (*.pth *.pt)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def update_video_source_ui(self):
        source_type = self.video_source_combo.currentData()

        if source_type == 0:  # Webcam
            self.video_path_edit.clear()
            self.video_path_edit.setPlaceholderText("Using default webcam")
            self.video_path_edit.setEnabled(False)
            self.video_browse_btn.setEnabled(False)

        elif source_type == 1:  # Video file
            self.video_path_edit.clear()
            self.video_path_edit.setPlaceholderText("Select video file...")
            self.video_path_edit.setEnabled(True)
            self.video_browse_btn.setEnabled(True)

        elif source_type == 2:  # RTSP
            self.video_path_edit.clear()
            self.video_path_edit.setPlaceholderText("Enter RTSP URL...")
            self.video_path_edit.setEnabled(True)
            self.video_browse_btn.setEnabled(False)

    def update_device(self):
        self.device = self.device_combo.currentData()
        self.statusBar().showMessage(f"Device set to {self.device}")

        # If model is loaded, move it to the new device
        if self.model:
            try:
                self.model = self.model.to(self.device)
                self.model_status_label.setText(f"Model Status: Loaded (on {self.device})")
            except Exception as e:
                QMessageBox.warning(self, "Device Error", f"Error moving model to {self.device}: {str(e)}")

    def load_model(self, model_path):
        try:
            self.statusBar().showMessage(f"Loading model from {model_path}...")

            # Load model
            self.model = CSRNet()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()

            # Update UI
            self.model_path_edit.setText(model_path)
            self.model_status_label.setText(f"Model Status: Loaded (on {self.device})")
            self.statusBar().showMessage(f"Model loaded successfully!")

            return True

        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Model Error", f"Failed to load model: {str(e)}")
            return False

    def load_model_from_ui(self):
        model_path = self.model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "Model Error", "Please select a model file first.")
            return

        self.load_model(model_path)

    def process_image(self):
        # Check if model is loaded
        if not self.model:
            QMessageBox.warning(self, "Model Error", "Please load a model first.")
            return

        # Check if image is selected
        if not self.image_path:
            QMessageBox.warning(self, "Input Error", "Please select an image first.")
            return

        try:
            # Show progress bar
            self.image_progress.setVisible(True)
            self.image_progress.setValue(0)

            # Create and start processing thread
            self.process_thread = ProcessThread(self.model, self.image_path, self.device)
            self.process_thread.update_signal.connect(self.update_image_results)
            self.process_thread.finished_signal.connect(lambda: self.image_progress.setVisible(False))
            self.process_thread.progress_signal.connect(self.image_progress.setValue)
            self.process_thread.start()

        except Exception as e:
            self.statusBar().showMessage(f"Error processing image: {str(e)}")
            self.image_progress.setVisible(False)
            QMessageBox.critical(self, "Processing Error", f"Failed to process image: {str(e)}")

    def update_image_results(self, img, density_map, count):
        try:
            # Display original image
            h, w, c = img.shape
            bytes_per_line = c * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale image to fit the label while preserving aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            self.original_image_label.setPixmap(pixmap.scaled(
                self.original_image_label.width(),
                self.original_image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            # Display density map
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            im = ax.imshow(density_map, cmap='jet')
            ax.set_title(f'Density Map')
            ax.axis('off')
            self.figure.colorbar(im)
            self.canvas.draw()

            # Update count label
            self.count_label.setText(f"Count: {count:.2f}")

            # Update status
            self.statusBar().showMessage(f"Image processed successfully. Estimated count: {count:.2f}")

        except Exception as e:
            self.statusBar().showMessage(f"Error updating results: {str(e)}")

    def toggle_video_processing(self):
        # Check if model is loaded
        if not self.model:
            QMessageBox.warning(self, "Model Error", "Please load a model first.")
            return

        # If video thread is running, stop it
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_start_btn.setText("Start")
            self.video_source_combo.setEnabled(True)
            self.video_path_edit.setEnabled(True if self.video_source_combo.currentData() != 0 else False)
            self.video_browse_btn.setEnabled(True if self.video_source_combo.currentData() == 1 else False)
            self.statusBar().showMessage("Video processing stopped")

        else:
            # Get video source
            source_type = self.video_source_combo.currentData()

            if source_type == 0:  # Webcam
                video_source = 0
            elif source_type == 1:  # Video file
                video_source = self.video_path_edit.text()
                if not video_source:
                    QMessageBox.warning(self, "Input Error", "Please select a video file.")
                    return
            elif source_type == 2:  # RTSP
                video_source = self.video_path_edit.text()
                if not video_source:
                    QMessageBox.warning(self, "Input Error", "Please enter an RTSP URL.")
                    return

            try:
                # Create and start video thread
                self.video_thread = VideoProcessThread(self.model, video_source, self.device)
                self.video_thread.frame_signal.connect(self.update_video_frame)
                self.video_thread.finished_signal.connect(self.on_video_finished)
                self.video_thread.start()

                # Update UI
                self.video_start_btn.setText("Stop")
                self.video_source_combo.setEnabled(False)
                self.video_path_edit.setEnabled(False)
                self.video_browse_btn.setEnabled(False)
                self.statusBar().showMessage("Video processing started")

            except Exception as e:
                self.statusBar().showMessage(f"Error starting video processing: {str(e)}")
                QMessageBox.critical(self, "Video Error", f"Failed to start video processing: {str(e)}")

    def update_video_frame(self, frame, density_map, count):
        try:
            # Display video frame
            h, w, c = frame.shape
            bytes_per_line = c * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            self.video_frame_label.setPixmap(pixmap.scaled(
                self.video_frame_label.width(),
                self.video_frame_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            # Display density map
            self.video_figure.clear()
            ax = self.video_figure.add_subplot(111)
            im = ax.imshow(density_map, cmap='jet')
            ax.set_title(f'Density Map')
            ax.axis('off')
            self.video_figure.colorbar(im)
            self.video_canvas.draw()

            # Update count label
            self.video_count_label.setText(f"Count: {count:.2f}")

        except Exception as e:
            print(f"Error updating video frame: {e}")

    def on_video_finished(self):
        self.video_start_btn.setText("Start")
        self.video_source_combo.setEnabled(True)
        self.video_path_edit.setEnabled(True if self.video_source_combo.currentData() != 0 else False)
        self.video_browse_btn.setEnabled(True if self.video_source_combo.currentData() == 1 else False)
        self.statusBar().showMessage("Video processing finished")

    def start_batch_processing(self):
        # Check if model is loaded
        if not self.model:
            QMessageBox.warning(self, "Model Error", "Please load a model first.")
            return

        # Check if input and output folders are selected
        input_dir = self.batch_input_edit.text()
        output_dir = self.batch_output_edit.text()

        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Input Error", "Please select both input and output folders.")
            return

        try:
            # Clear previous results
            self.batch_results.clear()

            # Start batch processing thread
            self.batch_thread = BatchProcessThread(self.model, input_dir, output_dir, self.device)
            self.batch_thread.progress_signal.connect(self.update_batch_progress)
            self.batch_thread.result_signal.connect(self.display_batch_results)
            self.batch_thread.finished_signal.connect(self.on_batch_finished)

            # Update UI
            self.batch_progress.setValue(0)
            self.batch_status_label.setText("Processing...")
            self.batch_start_btn.setEnabled(False)

            # Start processing
            self.batch_thread.start()

        except Exception as e:
            self.statusBar().showMessage(f"Error starting batch processing: {str(e)}")
            QMessageBox.critical(self, "Batch Error", f"Failed to start batch processing: {str(e)}")

    def update_batch_progress(self, progress, status):
        self.batch_progress.setValue(progress)
        self.batch_status_label.setText(status)

    def display_batch_results(self, results):
        try:
            self.batch_results.clear()

            # Add headers
            self.batch_results.append("<b>Batch Processing Results:</b><br>")
            self.batch_results.append("<table border='1' cellspacing='0' cellpadding='5'>")
            self.batch_results.append("<tr><th>Image</th><th>Estimated Count</th></tr>")

            # Add results
            for result in results:
                img_name = os.path.basename(result['image'])
                count = result['count']
                self.batch_results.append(f"<tr><td>{img_name}</td><td>{count:.2f}</td></tr>")

            self.batch_results.append("</table>")

            # Add summary
            total_count = sum(result['count'] for result in results)
            avg_count = total_count / len(results) if results else 0

            self.batch_results.append("<br><b>Summary:</b>")
            self.batch_results.append(f"<br>Total images: {len(results)}")
            self.batch_results.append(f"<br>Total estimated count: {total_count:.2f}")
            self.batch_results.append(f"<br>Average count per image: {avg_count:.2f}")

            # Scroll to top
            self.batch_results.verticalScrollBar().setValue(0)

        except Exception as e:
            self.batch_results.append(f"<br><font color='red'>Error displaying results: {str(e)}</font>")

    def on_batch_finished(self):
        self.batch_start_btn.setEnabled(True)
        QMessageBox.information(self, "Batch Processing", "Batch processing complete!")


def main():
    parser = argparse.ArgumentParser(description='CSRNet Headcount GUI')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = HeadcountApp(args.model)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
