#!/bin/bash

# Script to prepare a dataset for CSRNet

# Default parameters
DATA_ROOT="data/raw"
OUTPUT_ROOT="data/processed"
SIGMA=15
VISUALIZE=false

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --data-root PATH     Path to raw dataset (default: data/raw)"
    echo "  -o, --output-root PATH   Path to output processed dataset (default: data/processed)"
    echo "  -s, --sigma VALUE        Sigma for Gaussian kernel (default: 15)"
    echo "  -v, --visualize          Enable visualization of annotations and density maps"
    echo "  -h, --help               Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--data-root)
            DATA_ROOT="$2"
            shift
            shift
            ;;
        -o|--output-root)
            OUTPUT_ROOT="$2"
            shift
            shift
            ;;
        -s|--sigma)
            SIGMA="$2"
            shift
            shift
            ;;
        -v|--visualize)
            VISUALIZE=true
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

# Check if the data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# Check if images directory exists
if [ ! -d "$DATA_ROOT/images" ]; then
    echo "Error: Images directory does not exist: $DATA_ROOT/images"
    exit 1
fi

# Check if annotations directory exists
if [ ! -d "$DATA_ROOT/annotations" ]; then
    echo "Error: Annotations directory does not exist: $DATA_ROOT/annotations"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_ROOT"

echo "Starting dataset preparation:"
echo "  Data root: $DATA_ROOT"
echo "  Output root: $OUTPUT_ROOT"
echo "  Sigma: $SIGMA"
echo "  Visualize: $VISUALIZE"

# Construct the Python command
CMD="python -m src.dataset_preparation --data-root $DATA_ROOT --output-root $OUTPUT_ROOT --sigma $SIGMA"

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

# Run the command
echo "Running command: $CMD"
eval $CMD

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Dataset preparation completed successfully!"
else
    echo "Error: Dataset preparation failed."
    exit 1
fi

# Analyze the dataset
echo "Analyzing the processed dataset..."
python -m src.dataset_preparation --output-root $OUTPUT_ROOT --analyze-only

echo "All done!"
