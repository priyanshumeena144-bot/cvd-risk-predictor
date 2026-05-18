#!/bin/bash

# CVD Predictor Setup Script for Mac/Linux

echo "===================================="
echo "CVD Risk Predictor - Setup"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/5] Installing backend dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

echo "[3/5] Installing TensorFlow and ML dependencies..."
pip install tensorflow scikit-learn numpy joblib

echo "[4/5] Creating necessary directories..."
mkdir -p backend/uploads

echo "[5/5] Setting up environment..."
if [ ! -f .env ]; then
    cat > .env << EOF
FLASK_ENV=development
JWT_SECRET_KEY=your-secret-key-change-in-production
GEMINI_API_KEY=
EOF
fi

echo ""
echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""
echo "To start the backend:"
echo "  1. Run: source venv/bin/activate"
echo "  2. Run: python -m backend.app"
echo ""
echo "To start the frontend:"
echo "  Open frontend/index.html in your browser"
echo "  OR run: cd frontend && python -m http.server 8000"
echo ""
echo "API will be available at: http://localhost:5000"
echo "Frontend will be available at: http://localhost:8000 (if using Python server)"
echo ""
