#!/bin/bash

# Set the target directory
BASE_DIR="../all_datasets"
SEPSIS_DIR="$BASE_DIR/sepsis"
TEMP_DIR="$SEPSIS_DIR/temp"

# Create directories if they don't exist
mkdir -p "$SEPSIS_DIR"
mkdir -p "$TEMP_DIR"

# Download the dataset files with --no-check-certificate
echo "Downloading training set A..."
wget --no-check-certificate -P "$TEMP_DIR" "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip"
if [ $? -ne 0 ]; then
    echo "Error downloading training set A"
    exit 1
fi

echo "Downloading training set B..."
wget --no-check-certificate -P "$TEMP_DIR" "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip"
if [ $? -ne 0 ]; then
    echo "Error downloading training set B"
    exit 1
fi

# Check if files exist before unzipping
if [ -f "$TEMP_DIR/training_setA.zip" ]; then
    echo "Extracting training set A..."
    unzip "$TEMP_DIR/training_setA.zip" -d "$TEMP_DIR"
else
    echo "Training set A zip file not found"
    rm -rf "$TEMP_DIR"  # Clean up before exit
    exit 1
fi

if [ -f "$TEMP_DIR/training_setB.zip" ]; then
    echo "Extracting training set B..."
    unzip "$TEMP_DIR/training_setB.zip" -d "$TEMP_DIR"
else
    echo "Training set B zip file not found"
    rm -rf "$TEMP_DIR"  # Clean up before exit
    exit 1
fi

# Move all .psv files to the main sepsis directory
echo "Moving files to main directory..."
if [ -d "$TEMP_DIR/training" ]; then
    mv "$TEMP_DIR/training"/*.psv "$SEPSIS_DIR"/ 2>/dev/null || echo "No .psv files found in training set A"
fi

if [ -d "$TEMP_DIR/training_setB" ]; then
    mv "$TEMP_DIR/training_setB"/*.psv "$SEPSIS_DIR"/ 2>/dev/null || echo "No .psv files found in training set B"
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
if [ -d "$TEMP_DIR" ]; then
    echo "Warning: Failed to remove temporary directory"
    echo "Please manually remove: $TEMP_DIR"
fi

# Verify files were moved successfully
psv_count=$(ls -1 "$SEPSIS_DIR"/*.psv 2>/dev/null | wc -l)
if [ "$psv_count" -gt 0 ]; then
    echo "Download and extraction complete. Found $psv_count .psv files in $SEPSIS_DIR"
else
    echo "Error: No .psv files found in the final directory"
    exit 1
fi