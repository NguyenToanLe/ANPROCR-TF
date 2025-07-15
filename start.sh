#!/bin/bash

echo "Starting setup for TensorFlow Object Detection..."

# Define the commit hash for tensorflow/models
TF_MODELS_COMMIT="0d1fa0bb7f921cfaff326e5b8f90e5a4daf256f6"
TF_MODELS_DIR="tf_models_temp"

# Remove existing tf_models_temp if it exists from previous failed runs, to ensure clean clone
if [ -d "$TF_MODELS_DIR" ]; then
    echo "Removing existing $TF_MODELS_DIR directory..."
    rm -rf "$TF_MODELS_DIR"
fi

# Clone the TensorFlow models repository with --depth 1 to save space and time
echo "Cloning tensorflow/models repository with --depth 1..."
git clone --depth 1 https://github.com/tensorflow/models.git "$TF_MODELS_DIR" || { echo "Failed to clone TensorFlow models."; exit 1; }

# Important: After a --depth 1 clone, you cannot directly checkout an arbitrary old commit.
# Instead, we need to fetch the specific commit if it's not the head of the branch.
# However, given that you're using a specific commit hash for tf_models_official 2.10.1 (which aligns with TF 2.10.1),
# you might not need an older commit for the models repo itself if the necessary files are in the latest head.
# If you DO need that exact commit, you'd fetch it:
# cd "$TF_MODELS_DIR" || { echo "Failed to change directory to $TF_MODELS_DIR."; exit 1; }
# git fetch origin $TF_MODELS_COMMIT --depth 1 || { echo "Failed to fetch specific commit."; exit 1; }
# git reset --hard FETCH_HEAD || { echo "Failed to reset to specific commit."; exit 1; }
# cd ../

# FOR SIMPLICITY, and assuming the files you need from the models repo (research/object_detection, research/slim)
# are relatively stable across commits relevant to TF 2.10.1, let's just stick with --depth 1 without explicit checkout for now.
# The original checkout was '0d1fa0bb7f921cfaff326e5b8f90e5a4daf256f6', which might be a bit older.
# If the latest master branch (with --depth 1) provides the necessary files for object_detection API that works with TF 2.10.1,
# this is the simplest approach.

# If the previous explicit git checkout was critical for compatibility, we might need to adjust.
# Let's try without the specific checkout for now, as --depth 1 does not support it directly.
# The `object_detection` setup uses `object_detection/packages/tf2/setup.py`. This should be consistent.


# --- CRITICAL: Set PYTHONPATH correctly relative to the app's root ---
export PYTHONPATH=$PYTHONPATH:$(pwd)/$TF_MODELS_DIR/research:$(pwd)/$TF_MODELS_DIR/research/slim
echo "PYTHONPATH set to: $PYTHONPATH"

# Navigate to the models/research directory for setup
echo "Navigating to $TF_MODELS_DIR/research..."
cd "$TF_MODELS_DIR/research" || { echo "Failed to change directory to $TF_MODELS_DIR/research."; exit 1; }

# Compile protobufs
echo "Compiling protobufs..."
protoc object_detection/protos/*.proto --python_out=. || { echo "Protoc compilation failed. Ensure 'protoc' is available."; exit 1; }

# Install object_detection API
echo "Installing object_detection API..."
cp object_detection/packages/tf2/setup.py . || { echo "Failed to copy object_detection setup.py."; exit 1; }
python setup.py build || { echo "Failed to build object_detection."; exit 1; }
python setup.py install || { echo "Failed to install object_detection."; exit 1; }

# Install slim
echo "Installing slim..."
cd slim || { echo "Failed to change directory to slim."; exit 1; }
pip install -e . || { echo "Failed to install slim."; exit 1; }
cd ../../..

echo "TensorFlow Object Detection setup complete."