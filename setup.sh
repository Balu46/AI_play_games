#!/bin/bash

# Configuration
VENV_DIR=".venv"
PYTHON="python3"

echo "--- AI Games Setup ---"

# 0. Install System Dependencies (Debian/Ubuntu)
if [ -x "$(command -v apt-get)" ]; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        swig \
        cmake \
        libgl1 \
        python3-dev \
        python3-venv \
        ffmpeg
else
    echo "Warning: apt-get not found. Please install system dependencies manually."
fi

# 1. Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv $VENV_DIR
    created=1
else
    echo "Virtual environment already exists."
    created=0
fi

# 2. Activate Virtual Environment
source $VENV_DIR/bin/activate

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install Dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "--- Setup Complete ---"
echo "To activate manually run: source $VENV_DIR/bin/activate"
