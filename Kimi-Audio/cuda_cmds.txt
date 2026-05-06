# Look for CUDA modules
module avail cuda

# Load CUDA (replace with actual version available)
module load cuda/12.6  # or whatever version is available

# Verify
nvcc --version
echo $CUDA_HOME

# Now try installing again
uv pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases


