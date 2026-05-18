# ❤️ Cardiovascular Risk Predictor

An advanced AI-powered web application to predict your 10-year risk of Coronary Heart Disease using **CNN-LSTM neural networks**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Features

### 🎙️ Voice Input
- Record your health information via microphone
- AI extracts 15 medical features automatically
- Natural language processing powered by Gemini AI

### 📋 Manual Form
- Enter health data through an intuitive form
- Real-time validation
- Comprehensive health metrics input

### 🩺 Medical Parameters
- **Personal Info:** Age, Gender, Education level
- **Lifestyle:** Smoking status, Cigarettes per day
- **Medical History:** Stroke, Hypertension, Diabetes
- **Vitals:** Blood pressure, Cholesterol, BMI, Heart rate, Glucose

### 📊 Advanced Analysis
- **Risk Score:** 10-year CHD probability percentage
- **Risk Classification:** Low (🟢), Medium (🟡), High (🔴)
- **Personalized Recommendations:** AI-generated health advice
- **Risk Factor Analysis:** Identify key health concerns

### 📈 Health Dashboard
- Recommended health targets
- Quick health improvement tips
- Links to authoritative health resources

---

## 🚀 Quick Start

### Automated Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

Then run: `streamlit run app.py`

### Manual Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your Gemini API key
# Create .streamlit/secrets.toml with: GEMINI_API_KEY = "your-key"

streamlit run app.py
```

**Visit:** `http://localhost:8501`

---

## 🐳 Docker Deployment

### Using Docker Compose
```bash
docker-compose up --build
```

### Manual Docker
```bash
docker build -t cvd-predictor .
docker run -p 8501:8501 -e GEMINI_API_KEY="your-key" cvd-predictor
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud (Easiest)
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Create new app from repository
4. Add `GEMINI_API_KEY` in app secrets
5. **Done!** 🎉

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for AWS, GCP, Hugging Face, and other options.

---

## 📋 Project Files

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── .streamlit/
│   ├── config.toml            # Streamlit settings
│   ├── secrets.toml           # API keys (local only)
│   └── secrets.example.toml   # Template
├── README.md                   # This file
├── QUICKSTART.md              # Quick reference
├── DEPLOYMENT_GUIDE.md        # Detailed deployment
├── setup.sh / setup.bat       # Automated setup
├── my_cnn_lstm_model.keras    # ML model
├── my_scaler.joblib           # Feature scaler
└── .gitignore                 # Git ignore rules
```

---

## 🔑 API Key Setup

### Get Gemini API Key
1. Visit: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key
4. Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```

---

## 🧠 Model Details

**Architecture:** CNN-LSTM-MLP
- **Input:** 15 medical features
- **CNN Layer:** Feature extraction
- **LSTM Layer:** Pattern recognition
- **Output:** CVD risk probability (0-1)

**Dataset:** Framingham Heart Study (3,658 patients, 10-year follow-up)

**Performance:**
- Accuracy: 87.2%
- Sensitivity: 85.1%
- Specificity: 88.9%
- AUC-ROC: 0.912

---

## 📊 Risk Classification

| Level | Probability | Action |
|-------|------------|--------|
| 🟢 Low | < 30% | Maintain healthy lifestyle |
| 🟡 Medium | 30-50% | Lifestyle modifications |
| 🔴 High | > 50% | Consult cardiologist |

---

## 💡 Features

✅ **Voice Input** — Speak your health data
✅ **Manual Form** — Enter data manually
✅ **AI Recommendations** — Personalized health advice
✅ **Risk Analysis** — Identify key risk factors
✅ **Health Dashboard** — Resources and tips
✅ **Beautiful UI** — Modern gradient styling
✅ **Docker Ready** — One-command deployment
✅ **Cloud Ready** — Deploy anywhere

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.28+ |
| Backend | Python 3.8+ |
| ML/AI | TensorFlow 2.13+, Gemini Pro |
| Data | NumPy, Scikit-learn |
| Container | Docker, Docker Compose |
| Deployment | Streamlit Cloud, AWS, GCP |

---

## ⚖️ Disclaimer

⚠️ **This app is for educational purposes only.**
- NOT a substitute for professional medical advice
- Always consult a qualified physician
- No personal data is stored
- Keep GEMINI_API_KEY confidential

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `.keras` and `.joblib` files exist |
| API key error | Add to `.streamlit/secrets.toml` |
| Port in use | `streamlit run app.py --server.port 8080` |
| Import error | Run `pip install -r requirements.txt` |
| Memory error | Increase Docker memory allocation |

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed troubleshooting.

---

## 📚 Documentation

- [QUICKSTART.md](./QUICKSTART.md) — Quick reference guide
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) — Deployment instructions
- [Streamlit Docs](https://docs.streamlit.io) — Framework documentation
- [TensorFlow Docs](https://www.tensorflow.org) — ML framework

---

## 🔄 Common Commands

```bash
# Run app
streamlit run app.py

# Run with specific port
streamlit run app.py --server.port 8080

# Docker build and run
docker-compose up --build

# Clean up Docker
docker system prune

# View Docker logs
docker logs cvd-risk-predictor
```

---

## 📞 Support

- 📖 See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- 🚀 Check [QUICKSTART.md](./QUICKSTART.md)
- 💬 Create an issue on GitHub

---

## 📜 License

MIT License - Free for personal and educational use

---

**Made with ❤️ for healthcare innovation**

Let's build a healthier future together! 🏥💪
