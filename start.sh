#!/bin/bash

echo "Starting setup for TensorFlow Object Detection..."

# Define the commit hash for tensorflow/models
TF_MODELS_COMMIT="0d1fa0bb7f921cfaff326e5b8f90e5a4daf256f6"
TF_MODELS_DIR="tf_models_temp" # Temporary directory to clone models into

# Remove existing tf_models_temp if it exists from previous failed runs, to ensure clean clone
if [ -d "$TF_MODELS_DIR" ]; then
    echo "Removing existing $TF_MODELS_DIR directory..."
    rm -rf "$TF_MODELS_DIR"
fi

# Clone the TensorFlow models repository
echo "Cloning tensorflow/models repository..."
git clone https://github.com/tensorflow/models.git "$TF_MODELS_DIR" || { echo "Failed to clone TensorFlow models."; exit 1; } # Exit if clone fails
cd "$TF_MODELS_DIR" || { echo "Failed to change directory to $TF_MODELS_DIR."; exit 1; }
git checkout "$TF_MODELS_COMMIT" || { echo "Failed to checkout commit $TF_MODELS_COMMIT."; exit 1; }
cd ../ # Go back to the root of your app repository

# --- CRITICAL: Set PYTHONPATH correctly relative to the app's root ---
# $(pwd) here refers to the root directory where your app.py and start.sh reside
export PYTHONPATH=$PYTHONPATH:$(pwd)/$TF_MODELS_DIR/research:$(pwd)/$TF_MODELS_DIR/research/slim
echo "PYTHONPATH set to: $PYTHONPATH"

# Navigate to the models/research directory for setup
echo "Navigating to $TF_MODELS_DIR/research..."
cd "$TF_MODELS_DIR/research" || { echo "Failed to change directory to $TF_MODELS_DIR/research."; exit 1; }

# Compile protobufs
echo "Compiling protobufs..."
# Ensure protoc is available. If it's not, this command will fail.
# Streamlit Cloud generally has it, but if not, this is a blocker.
protoc object_detection/protos/*.proto --python_out=. || { echo "Protoc compilation failed. Ensure 'protoc' is available."; exit 1; }

# Install object_detection API
echo "Installing object_detection API..."
# Copy setup.py to the current directory (models/research) as per TensorFlow's instructions
cp object_detection/packages/tf2/setup.py . || { echo "Failed to copy object_detection setup.py."; exit 1; }
python setup.py build || { echo "Failed to build object_detection."; exit 1; }
python setup.py install || { echo "Failed to install object_detection."; exit 1; }

# Install slim
echo "Installing slim..."
cd slim || { echo "Failed to change directory to slim."; exit 1; }
pip install -e . || { echo "Failed to install slim."; exit 1; }
cd ../../.. # Go back to the root of your app repository (where app.py is)

echo "TensorFlow Object Detection setup complete."

# Streamlit will now automatically run app.py because it's set as the main file