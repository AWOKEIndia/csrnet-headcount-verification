import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
from PIL import Image
import torch


def visualize_density_map(image, density_map, count=None, alpha=0.5, colormap='jet', save_path=None):
    """
    Visualize image and its corresponding density map side by side

    Args:
        image (numpy.ndarray): Input image (H, W, 3) in RGB format
        density_map (numpy.ndarray): Density map (H, W)
        count (float, optional): Total count from density map
        alpha (float): Transparency for density map overlay
        colormap (str): Colormap for density map
        save_path (str, optional): Path to save the visualization

    Returns:
        numpy.ndarray: Visualization image
    """
    # Create figure
    plt.figure(figsize=(16, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Density map
    plt.subplot(1, 3, 2)
    plt.imshow(density_map, cmap=colormap)
    if count is not None:
        plt.title(f'Density Map (Count: {count:.2f})')
    else:
        plt.title('Density Map')
    plt.axis('off')
    plt.colorbar()

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    # Resize density map to match image size if necessary
    if image.shape[:2] != density_map.shape:
        resized_density = cv2.resize(density_map, (image.shape[1], image.shape[0]))
    else:
        resized_density = density_map

    # Normalize density map for visualization
    if np.max(resized_density) > 0:
        norm_density = resized_density / np.max(resized_density)
    else:
        norm_density = resized_density

    # Apply colormap and overlay
    colored_density = plt.cm.get_cmap(colormap)(norm_density)
    colored_density = (colored_density[:, :, :3] * 255).astype(np.uint8)

    overlay = cv2.addWeighted(
        image.astype(np.uint8),
        1 - alpha,
        colored_density,
        alpha,
        0
    )

    plt.imshow(overlay)
    if count is not None:
        plt.title(f'Overlay (Count: {count:.2f})')
    else:
        plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return cv2.imread(save_path)
    else:
        plt.show()
        return overlay


def create_comparison_visualization(image, gt_density, pred_density, save_path=None):
    """
    Create visualization comparing ground truth and predicted density maps

    Args:
        image (numpy.ndarray): Input image (H, W, 3) in RGB format
        gt_density (numpy.ndarray): Ground truth density map (H, W)
        pred_density (numpy.ndarray): Predicted density map (H, W)
        save_path (str, optional): Path to save the visualization

    Returns:
        numpy.ndarray: Visualization image
    """
    # Calculate counts
    gt_count = np.sum(gt_density)
    pred_count = np.sum(pred_density)

    # Create figure
    plt.figure(figsize=(16, 12))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Ground truth density map
    plt.subplot(2, 2, 2)
    plt.imshow(gt_density, cmap='jet')
    plt.title(f'Ground Truth (Count: {gt_count:.2f})')
    plt.axis('off')
    plt.colorbar()

    # Predicted density map
    plt.subplot(2, 2, 3)
    plt.imshow(pred_density, cmap='jet')
    plt.title(f'Prediction (Count: {pred_count:.2f})')
    plt.axis('off')
    plt.colorbar()

    # Difference map
    plt.subplot(2, 2, 4)
    diff = pred_density - gt_density
    plt.imshow(diff, cmap='bwr')
    plt.title(f'Difference (Error: {abs(pred_count - gt_count):.2f})')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return cv2.imread(save_path)
    else:
        plt.show()
        return None


def visualize_training_progress(train_losses, val_losses, val_maes, save_path=None):
    """
    Visualize training progress by plotting losses and MAE

    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        val_maes (list): List of validation MAEs
        save_path (str, optional): Path to save the visualization

    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot MAE
    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_maes, 'g-', label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_video_frame_visualization(frame, density_map, count, fps=None):
    """
    Create visualization for video frame with density map and count

    Args:
        frame (numpy.ndarray): Input video frame (H, W, 3) in BGR format
        density_map (numpy.ndarray): Density map (H, W)
        count (float): Total count from density map
        fps (float, optional): Frames per second for display

    Returns:
        numpy.ndarray: Visualization frame
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize density map to match frame dimensions
    h, w = frame.shape[:2]
    density_map_resized = cv2.resize(density_map, (w, h))

    # Normalize density map for visualization
    if np.max(density_map_resized) > 0:
        density_norm = density_map_resized / np.max(density_map_resized)
    else:
        density_norm = density_map_resized

    # Apply colormap to density map
    colormap = cm.get_cmap('jet')
    density_colored = (colormap(density_norm) * 255).astype(np.uint8)
    density_colored = density_colored[:, :, :3]  # Remove alpha channel

    # Create alpha blend with original frame
    alpha = 0.5
    overlay = cv2.addWeighted(
        frame_rgb,
        1 - alpha,
        density_colored,
        alpha,
        0
    )

    # Add text with count and FPS
    text = f"Count: {count:.1f}"
    if fps is not None:
        text += f" | FPS: {fps:.1f}"

    # Convert back to BGR for OpenCV
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Add text
    cv2.putText(
        overlay_bgr,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return overlay_bgr


def create_batch_visualization_grid(images, density_maps, counts, grid_size=(3, 3), save_path=None):
    """
    Create a grid visualization for batch processing results

    Args:
        images (list): List of images
        density_maps (list): List of density maps
        counts (list): List of counts
        grid_size (tuple): Grid size (rows, cols)
        save_path (str, optional): Path to save the visualization

    Returns:
        numpy.ndarray: Grid visualization
    """
    rows, cols = grid_size
    num_samples = min(len(images), rows * cols)

    plt.figure(figsize=(cols * 6, rows * 4))

    for i in range(num_samples):
        # Original image
        plt.subplot(rows, cols * 2, i * 2 + 1)
        plt.imshow(images[i])
        plt.title(f'Image {i+1}')
        plt.axis('off')

        # Density map
        plt.subplot(rows, cols * 2, i * 2 + 2)
        plt.imshow(density_maps[i], cmap='jet')
        plt.title(f'Count: {counts[i]:.2f}')
        plt.axis('off')
        plt.colorbar()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return cv2.imread(save_path)
    else:
        plt.show()
        return None


def create_model_summary_visualization(model_name, dataset_name, metrics, example_results, save_path=None):
    """
    Create a comprehensive visualization summarizing model performance

    Args:
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        metrics (dict): Dictionary containing performance metrics
        example_results (list): List of tuples (image, gt_density, pred_density)
        save_path (str, optional): Path to save the visualization

    Returns:
        None
    """
    plt.figure(figsize=(16, 12))

    # Title
    plt.suptitle(f"Model: {model_name} | Dataset: {dataset_name}", fontsize=16)

    # Plot metrics as text
    plt.subplot(2, 2, 1)
    plt.axis('off')
    metrics_text = "\n".join([f"{key.upper()}: {value:.4f}" for key, value in metrics.items()])
    plt.text(0.5, 0.5, f"Performance Metrics:\n\n{metrics_text}",
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12)

    # Plot one example result
    if example_results:
        image, gt_density, pred_density = example_results[0]

        gt_count = np.sum(gt_density)
        pred_count = np.sum(pred_density)

        # Original image
        plt.subplot(2, 2, 2)
        plt.imshow(image)
        plt.title('Example Image')
        plt.axis('off')

        # Ground truth density map
        plt.subplot(2, 2, 3)
        plt.imshow(gt_density, cmap='jet')
        plt.title(f'Ground Truth (Count: {gt_count:.2f})')
        plt.axis('off')
        plt.colorbar()

        # Predicted density map
        plt.subplot(2, 2, 4)
        plt.imshow(pred_density, cmap='jet')
        plt.title(f'Prediction (Count: {pred_count:.2f})')
        plt.axis('off')
        plt.colorbar()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    # Create a sample image and density map
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    density_map = np.zeros((120, 160))

    # Add some Gaussian blobs to represent people
    for _ in range(20):
        x = np.random.randint(10, 150)
        y = np.random.randint(10, 110)
        sigma = np.random.uniform(1.5, 3.0)

        for i in range(density_map.shape[0]):
            for j in range(density_map.shape[1]):
                density_map[i, j] += np.exp(-((i - y) ** 2 + (j - x) ** 2) / (2 * sigma ** 2))

    # Test visualization functions
    count = np.sum(density_map)
    print(f"Sample density map count: {count:.2f}")

    visualize_density_map(image, density_map, count)

    # Create a predicted density map with some errors
    pred_density = density_map.copy()
    pred_density += np.random.normal(0, 0.1, pred_density.shape)
    pred_density = np.maximum(0, pred_density)

    create_comparison_visualization(image, density_map, pred_density)

    # Test training visualization
    train_losses = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35]
    val_losses = [0.9, 0.8, 0.7, 0.65, 0.6, 0.58, 0.57, 0.56]
    val_maes = [25, 22, 20, 18, 16, 15, 14.5, 14]

    visualize_training_progress(train_losses, val_losses, val_maes)
