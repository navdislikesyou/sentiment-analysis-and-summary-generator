#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting build script..."

# 1. Install all Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 2. Download NLTK stopwords data
echo "Downloading NLTK stopwords data..."
python -c "import nltk; nltk.download('stopwords')"

echo "Build script finished successfully."