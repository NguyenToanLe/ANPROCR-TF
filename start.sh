#!/bin/bash

echo "Starting setup for TensorFlow Object Detection..."

# Define the commit hash for tensorflow/models
TF_MODELS_COMMIT="0d1fa0bb7f921cfaff326e5b8f90e5a4daf256f6"
TF_MODELS_DIR="tf_models_temp"

# --- Install protoc (Protocol Buffer Compiler) ---
# Check if protoc is already in PATH
if ! command -v protoc &> /dev/null; then
    echo "protoc not found, downloading and installing..."
    PROTOC_VERSION="21.12" # A common stable version, compatible with TF 2.10.1
    PROTOC_ZIP="protoc-$PROTOC_VERSION-linux-x86_64.zip"
    PROTOC_INSTALL_DIR="/usr/local" # Good location for binaries, or a local bin dir
    LOCAL_BIN_DIR="$(pwd)/bin" # Create a local bin directory if /usr/local is not writable

    mkdir -p "$LOCAL_BIN_DIR" # Ensure local bin exists

    wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP -O /tmp/$PROTOC_ZIP || { echo "Failed to download protoc."; exit 1; }
    unzip -o /tmp/$PROTOC_ZIP -d $LOCAL_BIN_DIR || { echo "Failed to unzip protoc."; exit 1; } # Unzip to local bin

    # Add the local bin directory to PATH for the current session
    export PATH="$LOCAL_BIN_DIR/bin:$PATH"
    echo "protoc installed to $LOCAL_BIN_DIR and added to PATH."
else
    echo "protoc already found in PATH, skipping download."
fi

# Remove existing tf_models_temp if it exists from previous failed runs, to ensure clean clone
if [ -d "$TF_MODELS_DIR" ]; then
    echo "Removing existing $TF_MODELS_DIR directory..."
    rm -rf "$TF_MODELS_DIR"
fi

# Clone the TensorFlow models repository with --depth 1 to save space and time
echo "Cloning tensorflow/models repository with --depth 1..."
git clone --depth 1 https://github.com/tensorflow/models.git "$TF_MODELS_DIR" || { echo "Failed to clone TensorFlow models."; exit 1; }

# --- CRITICAL: Set PYTHONPATH correctly relative to the app's root ---
# $(pwd) at the start of start.sh will be /mount/src/anprocr-tf/
export PYTHONPATH=$PYTHONPATH:$(pwd)/$TF_MODELS_DIR/research:$(pwd)/$TF_MODELS_DIR/research/slim
echo "PYTHONPATH set to: $PYTHONPATH"

# Navigate to the models/research directory for setup
echo "Navigating to $TF_MODELS_DIR/research..."
cd "$TF_MODELS_DIR/research" || { echo "Failed to change directory to $TF_MODELS_DIR/research."; exit 1; }

# Compile protobufs
echo "Compiling protobufs..."
# Now protoc should be found because we added its local directory to PATH
protoc object_detection/protos/*.proto --python_out=. || { echo "Protoc compilation failed."; exit 1; }

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