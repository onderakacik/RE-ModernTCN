#!/bin/bash

# Set the target directory
BASE_DIR="../all_datasets"
SPEECH_DIR="$BASE_DIR/speech"
TEMP_DIR="$SPEECH_DIR/temp"

# Create directories if they don't exist
mkdir -p "$SPEECH_DIR"
mkdir -p "$TEMP_DIR"

# Download the dataset
echo "Downloading Speech Commands dataset..."
wget --no-check-certificate -P "$TEMP_DIR" "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
if [ $? -ne 0 ]; then
    echo "Error downloading Speech Commands dataset"
    exit 1
fi

# Check if file exists before extracting
if [ -f "$TEMP_DIR/speech_commands_v0.02.tar.gz" ]; then
    echo "Extracting Speech Commands dataset..."
    tar -xzf "$TEMP_DIR/speech_commands_v0.02.tar.gz" -C "$SPEECH_DIR"
else
    echo "Speech Commands dataset file not found"
    rm -rf "$TEMP_DIR"  # Clean up before exit
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
if [ -d "$TEMP_DIR" ]; then
    echo "Warning: Failed to remove temporary directory"
    echo "Please manually remove: $TEMP_DIR"
fi

# Verify extraction was successful
dir_count=$(ls -1 "$SPEECH_DIR" | grep -E '^(yes|no|up|down|left|right|on|off|stop|go)$' | wc -l)
if [ "$dir_count" -eq 10 ]; then
    echo "Download and extraction complete. Found all 10 command directories in $SPEECH_DIR"
else
    echo "Error: Not all command directories were found"
    exit 1
fi 