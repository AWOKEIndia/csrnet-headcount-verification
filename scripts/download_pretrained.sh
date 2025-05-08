#!/bin/bash

# Script to download pre-trained CSRNet models

# Create models directory if it doesn't exist
mkdir -p models/pretrained

echo "Downloading pre-trained CSRNet model..."

# You would replace this URL with your actual model hosting location
MODEL_URL="https://your-model-hosting-url.com/csrnet_pretrained.pth"
OUTPUT_PATH="models/pretrained/csrnet_pretrained.pth"

# Download the model using curl or wget
if command -v curl &>/dev/null; then
    curl -L $MODEL_URL -o $OUTPUT_PATH
elif command -v wget &>/dev/null; then
    wget $MODEL_URL -O $OUTPUT_PATH
else
    echo "Error: Neither curl nor wget is installed. Please install one of them and try again."
    exit 1
fi

# Check if download was successful
if [ $? -eq 0 ] && [ -f $OUTPUT_PATH ]; then
    echo "Pre-trained model downloaded successfully to $OUTPUT_PATH"
else
    echo "Error: Failed to download the pre-trained model."
    exit 1
fi

echo "Download complete!"
