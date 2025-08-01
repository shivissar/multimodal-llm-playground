#!/bin/bash
set -e

# Activate venv
source venv/bin/activate

# Clean old builds
rm -rf build dist __pycache__ *.spec

# Detect metadata flag based on PyInstaller version
PYI_VERSION=$(pyinstaller --version | cut -d'.' -f1)
if (( PYI_VERSION >= 6 )); then
    META_FLAG="--copy-metadata"
else
    META_FLAG="--collect-metadata"
fi

# Build binary
pyinstaller --onefile --windowed \
  --hidden-import streamlit \
  $META_FLAG streamlit \
  $META_FLAG importlib_metadata \
  --name "MultimodalLLMPlayground" \
  --add-data ".env.example:." \
  app.py

echo "✅ Build complete: dist/MultimodalLLMPlayground"

