#!/bin/bash

# Local Transcription Platform Setup Script

set -e

echo "🚀 Setting up Local Transcription Platform..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements-local.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads models logs examples/sample_audio

# Initialize database
echo "🗄️  Initializing database..."
python -c "from src.database.init_db import init_database; init_database()"

# Download Whisper models
echo "🤖 Downloading Whisper models..."
python scripts/download_models.py

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the application: python src/main.py"
echo "3. Open http://localhost:8000 in your browser"
echo ""
echo "📖 For more information, see README.md" 