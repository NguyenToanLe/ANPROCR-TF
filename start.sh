#!/bin/bash

echo "Starting setup for TensorFlow Object Detection..."

# Define the commit hash for tensorflow/models
TF_MODELS_COMMIT="0d1fa0bb7f921cfaff326e5b8f90e5a4daf256f6"
TF_MODELS_DIR="tf_models_temp" # Temporary directory to clone models into

# Clone the TensorFlow models repository
echo "Cloning tensorflow/models repository..."
git clone https://github.com/tensorflow/models.git "$TF_MODELS_DIR"
cd "$TF_MODELS_DIR"
git checkout "$TF_MODELS_COMMIT"
cd ../ # Go back to the root of your app repository

# Set PYTHONPATH to include the research directory and slim directory
# This ensures Python can find the object_detection and slim modules
export PYTHONPATH=$PYTHONPATH:$(pwd)/$TF_MODELS_DIR/research:$(pwd)/$TF_MODELS_DIR/research/slim

# Navigate to the models/research directory for setup
echo "Navigating to $TF_MODELS_DIR/research..."
cd "$TF_MODELS_DIR/research"

# Compile protobufs
# Streamlit Cloud environments usually have 'protoc' installed.
echo "Compiling protobufs..."
protoc object_detection/protos/*.proto --python_out=. || { echo "Protoc compilation failed, but continuing..."; } # Add || true to make it non-fatal if protoc not found in some cases, or handle error properly

# Install object_detection API
echo "Installing object_detection API..."
# Create a dummy setup.py in the current directory, if needed by object_detection setup.
# Or, if the existing setup.py from tf2 is sufficient, copy it.
# Based on your local steps, 'copy object_detection\packages\tf2\setup.py .' is used.
cp object_detection/packages/tf2/setup.py .
python setup.py build
python setup.py install

# Install slim
echo "Installing slim..."
cd slim
pip install -e . # -e ensures it's installed in editable mode from this cloned repo
cd ../../.. # Go back to the root of your app repository (where app.py is)

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py