#!/bin/bash

# CVD Risk Predictor - Setup Script for Unix/Linux/Mac

echo "====================================="
echo "❤️  CVD Risk Predictor Setup"
echo "====================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .streamlit directory
echo "📁 Creating Streamlit configuration..."
mkdir -p .streamlit

# Create secrets file if it doesn't exist
if [ ! -f .streamlit/secrets.toml ]; then
    echo "🔐 Creating secrets.toml template..."
    cp .streamlit/secrets.example.toml .streamlit/secrets.toml
    echo ""
    echo "⚠️  IMPORTANT: Edit .streamlit/secrets.toml and add your GEMINI_API_KEY"
    echo "   Get your key at: https://makersuite.google.com/app/apikey"
    echo ""
fi

echo ""
echo "====================================="
echo "✅ Setup Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Edit .streamlit/secrets.toml and add your GEMINI_API_KEY"
echo "2. Run: streamlit run app.py"
echo "3. Visit: http://localhost:8501"
echo ""
