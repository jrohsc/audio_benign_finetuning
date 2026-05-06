#!/bin/bash
# Setup script for Qwen2.5-Omni LoRA finetuning environment
# Run this script to install all dependencies

set -e

CONDA_PATH="/work/anon/miniconda3"
ENV_NAME="qwen-omni"

echo "============================================"
echo "Setting up Qwen2.5-Omni Finetuning Environment"
echo "============================================"

# Initialize conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate the environment
conda activate $ENV_NAME

echo "Activated environment: $ENV_NAME"

# Navigate to LlamaFactory directory
cd "$(dirname "$0")/../LlamaFactory"

# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install LlamaFactory with all extras
echo "Installing LlamaFactory..."
pip install -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes]"

# Install additional dependencies for Qwen2.5-Omni
echo "Installing additional dependencies..."
pip install transformers>=4.49.0
pip install accelerate>=0.31.0
pip install peft>=0.12.0
pip install datasets>=2.16.0
pip install trl>=0.8.6
pip install flash-attn --no-build-isolation
pip install soundfile
pip install librosa
pip install fire

echo ""
echo "============================================"
echo "Environment setup complete!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "  source $CONDA_PATH/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "To train, run:"
echo "  ./2_finetune_qwen_omni.sh --model 7b --percentage 50"
echo ""
