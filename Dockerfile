# Use an official CUDA image as the base
FROM nvidia/cuda:11.8.0-base-ubuntu18.04

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.8 \
    python3.8-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    wget \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -afy

# Create a working directory
WORKDIR /workspace

# Install MuJoCo dependencies
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Copy environment.yml and install dependencies using conda
COPY environment.yml /workspace/
RUN conda env create -f environment.yml && \
    conda clean -afy

# Set up CUDA paths for PyTorch
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda

RUN conda init bash && \
    echo "conda activate pgmorl" >> ~/.bashrc

RUN conda init && . ~/.bashrc && conda activate pgmorl && conda remove --force ffmpeg -y
RUN conda init && . ~/.bashrc && conda activate pgmorl && pip install opencv-python-headless opencv-contrib-python
RUN apt update && apt install -y ffmpeg \
    libosmesa6-dev libgl1-mesa-glx libglfw3 \
    xvfb
RUN export MUJOCO_GL=egl
RUN conda init && . ~/.bashrc && conda activate pgmorl && python -c "import gym; gym.make('HalfCheetah-v2')"
# Default command to run in the container
CMD ["bash"]