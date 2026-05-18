#!/bin/bash

# =====================================
# CVD PREDICTOR 3D - Quick Setup Script
# =====================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     CVD PREDICTOR 3D - SETUP SCRIPT (Linux/Mac)           ║"
echo "║     Modern 3D Website with No Glitches                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python found"
echo ""

# Install dependencies
echo "Installing dependencies..."
cd "$(dirname "$0")"
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"
echo ""

# Start backend
echo "Starting CVD Predictor Backend..."
echo ""
echo "🚀 Backend server starting on http://localhost:5000"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "   NEXT: Open your browser and go to:"
echo "   📂 $(pwd)/frontend/index.html"
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 backend/app.py
