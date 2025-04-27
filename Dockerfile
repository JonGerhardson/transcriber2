# Use the official PyTorch image with CUDA 12.1 and cuDNN 8 support
# This matches the base image used in your example docker run command
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by the project
# - ffmpeg: For audio/video processing
# - git: Might be needed by some Python packages for installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git && \
    # Clean up apt cache to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install ruamel.yaml separately as specified in the docker run command
# (Pinning this version might be necessary for compatibility with other packages)
RUN pip install --no-cache-dir 'ruamel.yaml<0.18'

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from the requirements file
# --no-cache-dir: Reduces image size by not storing the pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main Python script into the container
COPY whispxpyan_step4a_punct.py .

# Define the default command to run when the container starts
# This executes your Python script
ENTRYPOINT ["python", "/app/whispxpyan_step4a_punct.py"]

# --- Notes on Running ---
# Build the image:
# docker build -t whispxpyan-image .
#
# Run the container (similar to your example command):
# Make sure the paths on your host machine are correct.
# docker run --gpus all --rm \
#   -v "/path/on/host/to/audio_input:/app/audio_input" \
#   -v "/path/on/host/to/transcripts_output:/app/transcripts_output" \
#   -v "/path/on/host/to/hf-token.txt:/app/hf-token.txt" \
#   whispxpyan-image
#
# Explanation of runtime flags used in the example:
# --gpus all: Makes NVIDIA GPUs available inside the container (requires nvidia-docker)
# --rm: Automatically removes the container when it exits
# -v: Mounts a directory or file from the host into the container
#     - /app/audio_input: Mount your audio files here
#     - /app/transcripts_output: Output transcripts will be saved here
#     - /app/hf-token.txt: Mount your Hugging Face token file here
