# Use NVIDIA CUDA base image to ensure GPU compatibility (matches cu118 requirement)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies (OpenCV requires libgl1 and libglib2.0)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install pip for python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set the working directory
WORKDIR /app

# 1. Install PyTorch matching the lab's bundle requirement (2.0.1+cu118)
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. Install OpenMIM and the extremely strict MMCV / MMDet versions
RUN pip install -U openmim setuptools numpy==1.26.4
RUN mim install mmcv-full==1.7.2 mmdet==2.28.2

# 3. Copy the requirements file and install the rest (FastAPI, boto3, YOLO, timm, etc.)
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install timm

# 4. Copy the rest of the application files
COPY . .

# Set YOLO settings to avoid the hub prompt we fixed earlier
RUN yolo settings reset

# Expose FastAPI port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
