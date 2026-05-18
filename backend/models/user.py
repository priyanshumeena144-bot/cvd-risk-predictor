from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import jwt
from functools import wraps

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    language = db.Column(db.String(10), default='en')  # Language preference
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship
    predictions = db.relationship('HealthPrediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'age': self.age,
            'gender': self.gender,
            'language': self.language,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }

class HealthPrediction(db.Model):
    """Health prediction history model"""
    __tablename__ = 'health_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Health metrics
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    cigarettes_per_day = db.Column(db.Float)
    systolic_bp = db.Column(db.Float)
    diastolic_bp = db.Column(db.Float)
    cholesterol = db.Column(db.Float)
    bmi = db.Column(db.Float)
    glucose = db.Column(db.Float)
    heart_rate = db.Column(db.Integer)
    stroke = db.Column(db.Boolean)
    hypertension = db.Column(db.Boolean)
    diabetes = db.Column(db.Boolean)
    current_smoker = db.Column(db.Boolean)
    prevalent_stroke = db.Column(db.Boolean)
    prevalent_hyp = db.Column(db.Boolean)
    diabetes_status = db.Column(db.String(50))
    education = db.Column(db.String(50))
    
    # Prediction results
    risk_score = db.Column(db.Float)  # 0-1 probability
    risk_percentage = db.Column(db.Float)  # 0-100
    risk_category = db.Column(db.String(20))  # 'low', 'medium', 'high'
    recommendations = db.Column(db.Text)  # AI-generated recommendations
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender,
            'risk_score': self.risk_score,
            'risk_percentage': self.risk_percentage,
            'risk_category': self.risk_category,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat(),
            'notes': self.notes,
            'health_metrics': {
                'cigarettes_per_day': self.cigarettes_per_day,
                'systolic_bp': self.systolic_bp,
                'diastolic_bp': self.diastolic_bp,
                'cholesterol': self.cholesterol,
                'bmi': self.bmi,
                'glucose': self.glucose,
                'heart_rate': self.heart_rate,
                'stroke': self.stroke,
                'hypertension': self.hypertension,
                'diabetes': self.diabetes,
                'current_smoker': self.current_smoker
            }
        }
