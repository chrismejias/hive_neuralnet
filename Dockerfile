# Hive Neural Network — GPU training container
# CUDA 12.8 + Python 3.14 + PyTorch 2.10.0
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + Python 3.14 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.14 \
    python3.14-dev \
    python3.14-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.14
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14

# Make python3.14 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1

WORKDIR /workspace

# Install PyTorch 2.10.0 with CUDA 12.8
# (adjust index URL if a different nightly/release channel is needed)
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install remaining Python dependencies
RUN python -m pip install \
    numpy==2.4.2 \
    pytest==9.0.2 \
    ninja==1.13.0

# Copy project source
COPY . /workspace

# Compile the CUDA extension (inplace inside the container)
RUN cd /workspace && python -m pip install -e hive_gpu/

# Verify the build
RUN python -c "import hive_gpu; ext = hive_gpu.load_extension(); print('hive_gpu_ext loaded, BOARD_SIZE =', ext.BOARD_SIZE)"

# Default command: show training help
CMD ["python", "-m", "hive_gpu", "--help"]
