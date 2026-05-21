# 🏥 CVD Risk Predictor - Professional Website

**A complete, production-ready web application for Cardiovascular Disease risk assessment.**

## ✨ What's Included

This is a **full-stack web application** with:

### 🎯 Features
- ✅ User authentication & registration
- ✅ AI-powered health risk prediction
- ✅ Health history tracking & trends
- ✅ Multi-language support (EN, ES, FR, DE, ZH)
- ✅ PDF & CSV export reports
- ✅ Prediction comparison tool
- ✅ Responsive mobile design
- ✅ Docker deployment ready

### 🏗️ Tech Stack
- **Backend**: Flask + SQLAlchemy + TensorFlow
- **Frontend**: HTML5 + CSS3 + Vanilla JavaScript
- **Database**: SQLite (expandable to PostgreSQL)
- **Authentication**: JWT tokens
- **Deployment**: Docker & Docker Compose
- **Reverse Proxy**: Nginx

## 🚀 Quick Start (5 minutes)

### Windows
```bash
setup-new.bat
venv\Scripts\activate.bat
python -m backend.app
```

### Mac/Linux
```bash
chmod +x setup-new.sh
./setup-new.sh
source venv/bin/activate
python -m backend.app
```

### Docker (Recommended)
```bash
docker-compose -f docker-compose-new.yml up --build
```

Then open:
- **Frontend**: http://localhost:8080 (or http://localhost:8000)
- **API**: http://localhost:5000
- **API Docs**: http://localhost:5000/api/health

## 📁 Project Structure

```
✓ backend/          → Flask API
✓ frontend/         → Website (HTML/CSS/JS)
✓ Docker files      → Container configuration
✓ Setup scripts     → Installation helpers
✓ Configuration     → nginx, env templates
```

## 📖 Documentation

📚 **Full deployment guide**: [DEPLOYMENT_GUIDE_FULL.md](DEPLOYMENT_GUIDE_FULL.md)

## 🔐 Configuration

1. Copy `.env.example` → `.env`
2. Update `JWT_SECRET_KEY` with a random string
3. Add `GEMINI_API_KEY` if you want voice input
4. Adjust other settings as needed

## 🌐 Deployment Options

- **Local Development**: Run `setup-new.bat` or `setup-new.sh`
- **Docker**: `docker-compose -f docker-compose-new.yml up`
- **Heroku**: Push with Procfile
- **AWS/GCP/Azure**: Use Docker images
- **DigitalOcean**: Docker + reverse proxy

## ✅ All Features Implemented

| Feature | Status | Location |
|---------|--------|----------|
| User Registration | ✅ | `backend/routes/auth.py` |
| User Login | ✅ | `backend/routes/auth.py` |
| Health Prediction | ✅ | `backend/routes/predictions.py` |
| History Tracking | ✅ | `backend/routes/predictions.py` |
| Trend Comparison | ✅ | `backend/routes/predictions.py` |
| PDF Export | ✅ | `backend/routes/export.py` |
| CSV Export | ✅ | `backend/routes/export.py` |
| Language Support | ✅ | `frontend/app.js` |
| Responsive UI | ✅ | `frontend/styles.css` |
| Mobile Design | ✅ | Media queries in CSS |

## 🎨 UI Features

- Modern gradient design with smooth animations
- Mobile-responsive layout
- Dark/light theme compatible
- Professional color scheme
- Interactive forms with validation
- Real-time notifications
- Chart and trend visualization

## 📊 User Workflow

1. **Register** → Create account with email
2. **Login** → Access dashboard
3. **Enter Health Data** → Fill comprehensive form
4. **Get Assessment** → See CVD risk with personalized recommendations
5. **Track History** → View all past assessments
6. **Compare** → Track changes over time
7. **Export** → Download PDF or CSV reports
8. **Settings** → Manage profile & preferences

## 🛠️ API Endpoints

### Authentication
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/profile`
- `PUT /api/auth/profile`
- `POST /api/auth/change-password`

### Predictions
- `POST /api/predictions/predict`
- `GET /api/predictions/history`
- `GET /api/predictions/<id>`
- `DELETE /api/predictions/<id>`
- `POST /api/predictions/compare`

### Export
- `GET /api/export/pdf/<id>`
- `GET /api/export/csv/history`

## ⚙️ Next Steps

1. **Test Locally**: Run setup and test the application
2. **Customize**: Add your branding/logo
3. **Add Features**: Extend with SMS notifications, email reminders, etc.
4. **Deploy**: Use Docker Compose on your server
5. **Monitor**: Set up logging and error tracking

## 🔒 Security Checklist

- [ ] Change JWT_SECRET_KEY
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Set FLASK_ENV=production
- [ ] Use environment variables for secrets
- [ ] Enable database backups
- [ ] Set up proper logging
- [ ] Configure firewall rules
- [ ] Enable rate limiting

## 📞 Troubleshooting

**Backend won't start?**
```bash
pip install -r backend/requirements.txt
python -m backend.app
```

**Frontend not loading?**
```bash
# Check API_URL in app.js matches your backend
# Try: python -m http.server 8000 in frontend folder
```

**Docker issues?**
```bash
docker-compose -f docker-compose-new.yml down
docker-compose -f docker-compose-new.yml up --build
```

---

**Ready to deploy your website!** 🚀

For more details, see [DEPLOYMENT_GUIDE_FULL.md](DEPLOYMENT_GUIDE_FULL.md)
