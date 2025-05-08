import os
import cv2
import torch
import numpy as np
import argparse
import time
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import CSRNet model
# Make sure to have the CSRNet implementation file in the same directory
from csrnet_implementation import CSRNet


def process_video(model_path, video_path, output_path=None, device='cuda', display=True, save_frames=False):
    """
    Process video for crowd counting
    """
    # Load model
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open video capture
    if video_path.isdigit():
        # If input is a number, treat as webcam
        cap = cv2.VideoCapture(int(video_path))
    else:
        # Otherwise, treat as video file path
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create directory for saving frames if needed
    if save_frames:
        frames_dir = 'output_frames'
        os.makedirs(frames_dir, exist_ok=True)

    # Frame counter and FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0

    # Colormap for density map visualization
    colormap = cm.get_cmap('jet')

    print("Processing video... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame (every nth frame to improve performance)
        process_this_frame = (frame_count % 2 == 0)

        if process_this_frame:
            # Convert frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Process image
            img_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_density_map = model(img_tensor).squeeze().cpu().numpy()

            # Calculate predicted count
            pred_count = np.sum(predicted_density_map)

            # Update FPS calculation
            current_time = time.time()
            if (current_time - start_time) > 0:
                fps_display = frame_count / (current_time - start_time)

        # Resize density map to match frame dimensions
        if process_this_frame:
            density_map = cv2.resize(predicted_density_map, (width, height))

            # Normalize density map for visualization (0-1)
            if np.max(density_map) > 0:
                density_map = density_map / np.max(density_map)

            # Convert to colormap
            density_map_colored = (colormap(density_map) * 255).astype(np.uint8)

            # Convert to BGR for display with OpenCV
            density_map_bgr = cv2.cvtColor(density_map_colored, cv2.COLOR_RGBA2BGR)

            # Create alpha blend with original frame
            alpha = 0.5
            overlay = cv2.addWeighted(frame, 1-alpha, density_map_bgr, alpha, 0)

        # Add text with count and FPS
        text = f"Count: {pred_count:.1f} | FPS: {fps_display:.1f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        if display:
            cv2.imshow('CSRNet Crowd Counting', overlay)

        # Save frame if needed
        if save_frames and process_this_frame:
            cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg'), overlay)

        # Write to output video if needed
        if output_path:
            out.write(overlay)

        # Increment frame counter
        frame_count += 1

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    # Final stats
    print(f"Processed {frame_count} frames")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")


def process_rtsp_stream(model_path, rtsp_url, output_path=None, device='cuda', display=True):
    """
    Process RTSP stream for crowd counting
    """
    print(f"Connecting to RTSP stream: {rtsp_url}")
    process_video(model_path, rtsp_url, output_path, device, display)


def batch_process_videos(model_path, video_dir, output_dir, device='cuda'):
    """
    Process multiple video files in a directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all video files
    video_files = []
    for ext in ['mp4', 'avi', 'mov', 'mkv']:
        video_files.extend(list(os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(f'.{ext}')))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video files to process")

    # Process each video
    for video_file in video_files:
        video_name = os.path.basename(video_file)
        print(f"Processing {video_name}...")

        output_path = os.path.join(output_dir, f"processed_{video_name}")
        process_video(model_path, video_file, output_path, device, display=False)

        print(f"Saved processed video to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSRNet Video Processing')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'rtsp', 'batch', 'webcam'],
                        help='Processing mode: video file, RTSP stream, batch directory, or webcam')
    parser.add_argument('--input', type=str, help='Input video file path, RTSP URL, or directory with videos')
    parser.add_argument('--output', type=str, help='Output video file path or directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--save-frames', action='store_true', help='Save individual frames')
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'

    if args.mode == 'video':
        if not args.input:
            parser.error("--input is required for video mode")
        process_video(args.model, args.input, args.output, args.device,
                      display=not args.no_display, save_frames=args.save_frames)

    elif args.mode == 'rtsp':
        if not args.input:
            parser.error("--input (RTSP URL) is required for rtsp mode")
        process_rtsp_stream(args.model, args.input, args.output, args.device,
                          display=not args.no_display)

    elif args.mode == 'batch':
        if not args.input:
            parser.error("--input (directory) is required for batch mode")
        if not args.output:
            parser.error("--output (directory) is required for batch mode")
        batch_process_videos(args.model, args.input, args.output, args.device)

    elif args.mode == 'webcam':
        # For webcam, input should be the camera index (usually 0 for built-in webcam)
        webcam_id = args.input if args.input else "0"
        process_video(args.model, webcam_id, args.output, args.device,
                    display=not args.no_display, save_frames=args.save_frames)
