# CVD Risk Predictor - Full Stack Application

## 🚀 Quick Start

### Option 1: Local Development (Windows)
```bash
# 1. Run setup
setup-new.bat

# 2. Activate virtual environment
venv\Scripts\activate.bat

# 3. Start backend
python -m backend.app

# 4. In another terminal, start frontend server
cd frontend
python -m http.server 8000

# 5. Open browser
http://localhost:8000
```

### Option 2: Local Development (Mac/Linux)
```bash
# 1. Run setup
chmod +x setup-new.sh
./setup-new.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start backend
python -m backend.app

# 4. In another terminal, start frontend server
cd frontend
python -m http.server 8000

# 5. Open browser
http://localhost:8000
```

### Option 3: Docker (Recommended for Deployment)
```bash
# Build and run with Docker Compose
docker-compose -f docker-compose-new.yml up --build

# Access
Frontend: http://localhost:8080
Backend API: http://localhost:5000
```

## 📋 Features Implemented

✅ **User Authentication**
- User registration and login
- JWT token-based authentication
- Password management
- Session handling

✅ **Health Risk Prediction**
- AI-powered CVD risk assessment using CNN-LSTM models
- Real-time health metrics input
- Personalized recommendations

✅ **Health History Tracking**
- Store prediction history
- Compare multiple assessments
- Track health trends over time

✅ **Multi-Language Support**
- Language preference selection
- Support for: English, Spanish, French, German, Chinese

✅ **Export Functionality**
- PDF reports with detailed health metrics
- CSV export of prediction history
- Professional report formatting

✅ **Comparison Tool**
- Compare multiple predictions
- Visualize risk trends
- Track improvements/changes

✅ **Responsive Design**
- Mobile-friendly interface
- Works on desktop, tablet, and smartphone
- Progressive Web App ready

## 🏗️ Project Structure

```
CCVD_Project/
├── backend/
│   ├── app.py              # Flask application factory
│   ├── config.py           # Configuration management
│   ├── requirements.txt    # Python dependencies
│   ├── models/
│   │   └── user.py         # User & HealthPrediction models
│   ├── routes/
│   │   ├── auth.py         # Authentication endpoints
│   │   ├── predictions.py  # Prediction endpoints
│   │   └── export.py       # Export endpoints
│   └── utils/
│       ├── auth.py         # JWT authentication
│       └── prediction.py   # ML prediction engine
├── frontend/
│   ├── index.html          # Main HTML
│   ├── styles.css          # Styling
│   └── app.js              # JavaScript application
├── docker-compose-new.yml  # Docker Compose configuration
├── Dockerfile.backend      # Backend Docker image
├── Dockerfile.frontend     # Frontend Docker image
├── nginx.conf             # Nginx reverse proxy config
├── setup-new.bat          # Windows setup script
└── setup-new.sh           # Mac/Linux setup script
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
FLASK_ENV=production
DATABASE_URL=sqlite:///cvd_app.db
JWT_SECRET_KEY=your-long-random-secret-key
GEMINI_API_KEY=your-gemini-api-key
```

### Backend Configuration
- **Port**: 5000 (default)
- **Database**: SQLite (default) or PostgreSQL
- **CORS**: Enabled for all origins (configure in production)

## 📚 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update profile
- `POST /api/auth/change-password` - Change password

### Predictions
- `POST /api/predictions/predict` - Make prediction
- `GET /api/predictions/history` - Get history
- `GET /api/predictions/<id>` - Get specific prediction
- `DELETE /api/predictions/<id>` - Delete prediction
- `POST /api/predictions/compare` - Compare predictions

### Export
- `GET /api/export/pdf/<id>` - Export PDF report
- `GET /api/export/csv/history` - Export CSV history

## 🌐 Deployment

### Heroku
1. Create `Procfile`:
```
web: python -m backend.app
```

2. Deploy:
```bash
git push heroku main
```

### AWS
1. Create EC2 instance (Ubuntu 20.04)
2. Install Docker and Docker Compose
3. Clone repository and run:
```bash
docker-compose -f docker-compose-new.yml up -d
```

### Google Cloud Run
1. Build Docker image
2. Push to Container Registry
3. Deploy from Cloud Run console

### DigitalOcean
1. Create Droplet (Ubuntu 20.04)
2. SSH into droplet
3. Follow Docker deployment steps

## 🔐 Security Considerations

⚠️ **Before Production Deployment:**

1. Change JWT_SECRET_KEY to a strong random string
2. Enable HTTPS/SSL certificates
3. Set FLASK_ENV=production
4. Configure CORS properly
5. Use environment variables for sensitive data
6. Enable database backups
7. Set up proper logging
8. Configure firewall rules

## 📊 Model Files

The application uses pre-trained CNN-LSTM models:
- `my_cnn_lstm_model_fixed.keras` - Neural network model
- `my_scaler.joblib` - Feature scaler for normalization

These files should be in the project root directory.

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r backend/requirements.txt

# Check port 5000 is available
netstat -ano | findstr :5000  # Windows
lsof -i :5000  # Mac/Linux
```

### Frontend not connecting to backend
- Check CORS settings in backend
- Verify API_URL in app.js matches your backend
- Check firewall rules

### Database errors
```bash
# Reset database (WARNING: loses all data)
rm backend/cvd_app.db
python -c "from backend.app import create_app; app = create_app(); app.app_context().push()"
```

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review API responses and error messages
3. Check browser console for JavaScript errors
4. Review backend logs

## 📄 License

This project is provided as-is for educational and commercial use.

---

**Version**: 1.0.0  
**Last Updated**: 2024
