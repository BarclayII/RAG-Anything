#!/bin/bash
# Install VideoRAG dependencies for RAG-Anything
# This script installs all necessary dependencies for video processing

set -e

echo "Installing video processing dependencies..."

# Install standard dependencies via uv
echo "Step 1: Installing standard dependencies..."
uv pip install -e ".[video]"

# Install pytorchvideo from git
echo "Step 2: Installing pytorchvideo from git..."
uv pip install "git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d"

# Install ImageBind from git (without dependencies to avoid conflicts)
echo "Step 3: Installing ImageBind from git..."
uv pip install --no-deps "git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3"

# Add VideoRAG to PYTHONPATH in a .envrc or similar
echo "Step 4: Setting up VideoRAG path..."
VIDEORAG_PATH="/Users/pinggan/RAGAll/VideoRAG/VideoRAG-algorithm"
if [ -d "$VIDEORAG_PATH" ]; then
    echo "VideoRAG found at $VIDEORAG_PATH"
    echo "Add this to your environment:"
    echo "  export PYTHONPATH=\"$VIDEORAG_PATH:\$PYTHONPATH\""
else
    echo "Warning: VideoRAG not found at $VIDEORAG_PATH"
    echo "Please update the path in this script or install VideoRAG"
fi

echo ""
echo "Installation complete!"
echo ""
echo "To use video processing, make sure to set PYTHONPATH:"
echo "  export PYTHONPATH=\"$VIDEORAG_PATH:\$PYTHONPATH\""
echo ""
echo "Or run with:"
echo "  PYTHONPATH=\"$VIDEORAG_PATH:\$PYTHONPATH\" uv run python your_script.py"
