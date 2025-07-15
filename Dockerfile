# Start from a Streamlit base image (recommended for Streamlit Cloud)
# This image already has Python, Streamlit, and some common tools.
# Python 3.10 is used here, matching your requirements.
FROM python:3.10-slim-buster

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by OpenCV
# libgl1-mesa-glx and libgthread-2.0-0 are common for headless OpenCV
# libglib2.0-0 is often a dependency too.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    libglib2.0-0 \
    protobuf-compiler \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
# Use --no-cache-dir for smaller image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy start.sh and make it executable
COPY start.sh .
RUN chmod +x start.sh

# Run the start.sh script to clone TF models and install object_detection/slim
RUN ./start.sh

# Copy your Streamlit app
COPY . .

# Set the PYTHONPATH environment variable for the app runtime
# This is crucial so app.py can find object_detection and slim
# The paths here must match what start.sh creates: tf_models_temp/research and tf_models_temp/research/slim
ENV PYTHONPATH="/app/tf_models_temp/research:/app/tf_models_temp/research/slim:${PYTHONPATH}"

# Command to run your Streamlit app when the container starts
CMD ["streamlit", "run", "app.py"]