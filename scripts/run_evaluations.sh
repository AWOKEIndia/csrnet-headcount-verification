#!/bin/bash

# Script to run model evaluations on test data

# Default parameters
MODEL_PATH="models/pretrained/csrnet_pretrained.pth"
TEST_DIR="data/processed/test"
OUTPUT_DIR="output/results"
DEVICE="cuda"

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model PATH      Path to the trained model (default: models/pretrained/csrnet_pretrained.pth)"
    echo "  -t, --test-dir PATH   Directory containing test images and density maps (default: data/processed/test)"
    echo "  -o, --output-dir PATH Directory to save results (default: output/results)"
    echo "  -d, --device DEVICE   Device to use (cuda or cpu, default: cuda)"
    echo "  -h, --help            Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        -t|--test-dir)
            TEST_DIR="$2"
            shift
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -d|--device)
            DEVICE="$2"
            shift
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if the model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file does not exist: $MODEL_PATH"
    exit 1
fi

# Check if the test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory does not exist: $TEST_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting model evaluation:"
echo "  Model: $MODEL_PATH"
echo "  Test directory: $TEST_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"

# Run the evaluation
CMD="python -m src.model_evaluation --model $MODEL_PATH --mode evaluate --test-dir $TEST_DIR --output-dir $OUTPUT_DIR --device $DEVICE"

echo "Running command: $CMD"
eval $CMD

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to $OUTPUT_DIR"
else
    echo "Error: Evaluation failed."
    exit 1
fi

echo "All done!"
