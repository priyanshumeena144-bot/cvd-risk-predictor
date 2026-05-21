# CVD Risk Predictor - Deployment Guide

Welcome! This guide will help you deploy your **CVD Risk Predictor** web application to the cloud.

## 📋 Table of Contents
1. [Local Setup](#local-setup)
2. [Docker Deployment](#docker-deployment)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Environment Variables](#environment-variables)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Local Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone or download the project:**
```bash
cd CCVD_Project
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create Streamlit secrets file:**
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Get your API key: [Google AI Studio](https://makersuite.google.com/app/apikey)

5. **Run the app:**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Quick Start

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

2. **Access the app:**
Open `http://localhost:8501`

### Manual Docker Commands

1. **Build the image:**
```bash
docker build -t cvd-predictor .
```

2. **Run the container:**
```bash
docker run -p 8501:8501 \
  -e GEMINI_API_KEY="your-api-key" \
  cvd-predictor
```

3. **View logs:**
```bash
docker logs cvd-predictor
```

4. **Stop the container:**
```bash
docker stop cvd-predictor
```

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

1. **Create a GitHub repository:**
   - Go to [github.com/new](https://github.com/new)
   - Create a public repo named `cvd-risk-predictor`

2. **Push your code to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cvd-risk-predictor.git
git push -u origin main
```

3. **Ensure these files are included:**
   - `app.py`
   - `requirements.txt`
   - `my_scaler.joblib`
   - `my_cnn_lstm_model.keras`
   - `.streamlit/config.toml` (optional)

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign up/Login with GitHub** (if not already done)

3. **Click "New app"**

4. **Fill in deployment details:**
   - Repository: `YOUR_USERNAME/cvd-risk-predictor`
   - Branch: `main`
   - Main file path: `app.py`

5. **Advanced settings (if needed):**
   - Python version: 3.11
   - Custom subdomain: `cvd-predictor`

6. **Click "Deploy"**

### Step 3: Add Secrets

1. **After deployment, click "Manage secrets"** (in the app menu)

2. **Add your Gemini API key:**
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

3. **Your app will automatically restart with the secrets**

---

## 🌍 Alternative Cloud Platforms

### Hugging Face Spaces

1. Create a Space: [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Docker" as the SDK
3. Upload your files
4. Hugging Face will automatically build and deploy

### AWS (EC2)

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum install -y docker
sudo systemctl start docker

# Clone repository
git clone https://github.com/YOUR_USERNAME/cvd-risk-predictor.git
cd cvd-risk-predictor

# Run with Docker
docker-compose up -d
```

### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/cvd-predictor

# Deploy
gcloud run deploy cvd-predictor \
  --image gcr.io/YOUR_PROJECT/cvd-predictor \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY="your-key"
```

---

## 🔐 Environment Variables

### Required
- `GEMINI_API_KEY` — Google Gemini API key for voice processing

### Optional
- `STREAMLIT_SERVER_PORT` — Default: 8501
- `STREAMLIT_SERVER_ADDRESS` — Default: localhost

---

## 🐛 Troubleshooting

### Issue: "Model file not found"
**Solution:** Ensure these files are in your project directory:
- `my_scaler.joblib`
- `my_cnn_lstm_model.keras`

### Issue: "Gemini API key missing"
**Solution:** Add `GEMINI_API_KEY` to:
- Local: `.streamlit/secrets.toml`
- Streamlit Cloud: App settings → Secrets
- Docker: Environment variables in `docker-compose.yml`

### Issue: "Out of memory" errors
**Solution:** 
- Reduce model complexity or use quantized version
- Increase available RAM (Docker container or VM specs)

### Issue: "Port 8501 already in use"
**Solution:**
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID YOUR_PID /F

# Linux/Mac
lsof -i :8501
kill -9 PID
```

---

## 📊 Performance Tips

1. **Model Optimization:**
   - Use TensorFlow Lite quantization
   - Consider model pruning for deployment

2. **Caching:**
   - Streamlit caches models with `@st.cache_resource`
   - Proper caching reduces memory usage

3. **Scaling:**
   - For high traffic, use cloud load balancers
   - Consider multi-instance deployments

---

## 🆘 Support

- **Streamlit Issues:** [docs.streamlit.io](https://docs.streamlit.io)
- **Docker Help:** [docker.com/resources](https://www.docker.com/resources)
- **Report Issues:** Create an issue on your GitHub repository

---

**Happy Deploying!** 🚀❤️
