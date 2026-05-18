# 🚀 Quick Start Reference

## 📋 Table of Contents
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [Cloud Deployment](#cloud-deployment)
- [Common Commands](#common-commands)

---

## 🖥️ Local Setup (Windows/Mac/Linux)

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

Then edit `.streamlit/secrets.toml` with your API key.

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Windows: Create .streamlit\secrets.toml
# Mac/Linux: Create .streamlit/secrets.toml
# Content: GEMINI_API_KEY = "your-key-here"

# 4. Run app
streamlit run app.py
```

✅ **Access:** `http://localhost:8501`

---

## 🐳 Docker Setup

### Quick Start
```bash
docker-compose up --build
```

✅ **Access:** `http://localhost:8501`

### Manual Docker
```bash
docker build -t cvd-app .
docker run -p 8501:8501 \
  -e GEMINI_API_KEY="your-key" \
  cvd-app
```

### Cleanup
```bash
docker-compose down
docker system prune
```

---

## ☁️ Cloud Deployment

### **Streamlit Cloud (Easiest)**
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select your repo
4. Add `GEMINI_API_KEY` in Secrets
5. Done! 🎉

### **Hugging Face Spaces**
1. Create Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select Docker
3. Upload files
4. Auto-deploys

### **AWS/GCP/Azure**
See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed steps.

---

## 📌 Common Commands

### Development
```bash
# Run app
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8080

# Clear Streamlit cache
streamlit cache clear
```

### Docker
```bash
# View logs
docker logs cvd-risk-predictor

# Stop container
docker stop cvd-risk-predictor

# Remove image
docker rmi cvd-predictor

# Join container shell
docker exec -it cvd-risk-predictor bash
```

### Dependencies
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Freeze current versions
pip freeze > requirements-locked.txt
```

---

## 🔑 API Keys

### Get Gemini API Key
1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy key
4. Add to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `API Key missing` | Add to `.streamlit/secrets.toml` |
| `Port 8501 in use` | Change port: `streamlit run app.py --server.port 8080` |
| `Model not found` | Ensure `.keras` and `.joblib` files exist |
| `Memory error` | Increase Docker memory or use lighter model |

---

## 📚 Useful Links

- **Streamlit Docs:** https://docs.streamlit.io
- **TensorFlow Docs:** https://www.tensorflow.org/api
- **Docker Docs:** https://docs.docker.com
- **Gemini API:** https://ai.google.dev

---

## ✨ Features Checklist

- ✅ Voice input with AI transcription
- ✅ Manual form entry
- ✅ CNN-LSTM prediction model
- ✅ Personalized health recommendations
- ✅ Risk factor analysis
- ✅ Health dashboard & resources
- ✅ Beautiful UI with custom styling
- ✅ Docker support
- ✅ Cloud deployment ready

---

## 📞 Need Help?

- Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- Review [README.md](./README.md)
- Check GitHub Issues (if available)
- Review Streamlit/Docker documentation

---

**Happy Predicting!** ❤️
