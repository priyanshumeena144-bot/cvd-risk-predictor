import numpy as np
import joblib
from tensorflow import keras
import os
from pathlib import Path

class PredictionEngine:
    """Handle CVD risk predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model_and_scaler()
    
    def load_model_and_scaler(self):
        """Load pre-trained model and scaler"""
        try:
            base_path = Path(__file__).parent.parent.parent
            
            # Try to load the model
            model_path = base_path / 'my_cnn_lstm_model_fixed.keras'
            if model_path.exists():
                self.model = keras.models.load_model(str(model_path))
            else:
                # Try alternative model paths
                for model_file in ['my_cnn_lstm_model_v4.h5', 'my_cnn_lstm_model_v3.h5', 'my_cnn_lstm_model.keras']:
                    alt_path = base_path / model_file
                    if alt_path.exists():
                        self.model = keras.models.load_model(str(alt_path))
                        break
            
            # Load scaler
            scaler_path = base_path / 'my_scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
        
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            self.model = None
            self.scaler = None
    
    def prepare_features(self, health_data):
        """Prepare features for prediction"""
        features = np.array([
            health_data.get('age', 0),
            health_data.get('cigarettes_per_day', 0),
            health_data.get('systolic_bp', 0),
            health_data.get('diastolic_bp', 0),
            health_data.get('cholesterol', 0),
            health_data.get('bmi', 0),
            health_data.get('glucose', 0),
            health_data.get('heart_rate', 0),
            int(health_data.get('stroke', False)),
            int(health_data.get('hypertension', False)),
            int(health_data.get('diabetes', False)),
            int(health_data.get('current_smoker', False)),
            int(health_data.get('prevalent_stroke', False)),
            int(health_data.get('prevalent_hyp', False)),
            int(health_data.get('gender') == 'male'),
        ]).reshape(1, -1)
        
        return features
    
    def predict(self, health_data):
        """Make CVD risk prediction"""
        if self.model is None or self.scaler is None:
            return {
                'error': 'Model not loaded',
                'risk_score': None,
                'risk_percentage': None,
                'risk_category': 'unknown'
            }
        
        try:
            # Prepare features
            features = self.prepare_features(health_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Reshape for CNN-LSTM (add time dimension)
            features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
            
            # Make prediction
            prediction = self.model.predict(features_scaled, verbose=0)
            risk_score = float(prediction[0][0])
            
            # Convert to percentage
            risk_percentage = risk_score * 100
            
            # Categorize risk
            if risk_percentage < 10:
                risk_category = 'low'
            elif risk_percentage < 20:
                risk_category = 'medium'
            else:
                risk_category = 'high'
            
            return {
                'risk_score': risk_score,
                'risk_percentage': round(risk_percentage, 2),
                'risk_category': risk_category,
                'error': None
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'risk_score': None,
                'risk_percentage': None,
                'risk_category': 'unknown'
            }
    
    def get_recommendations(self, health_data, risk_category):
        """Generate health recommendations based on risk"""
        recommendations = []
        
        # Age-based recommendations
        if health_data.get('age', 0) > 50:
            recommendations.append("Regular cardiovascular check-ups are recommended given your age.")
        
        # Smoking recommendations
        if health_data.get('current_smoker') or health_data.get('cigarettes_per_day', 0) > 0:
            recommendations.append("Quit smoking to significantly reduce your cardiovascular risk.")
        
        # Blood pressure recommendations
        systolic = health_data.get('systolic_bp', 0)
        if systolic > 140:
            recommendations.append("Your blood pressure is elevated. Consult your doctor for management.")
        elif systolic > 130:
            recommendations.append("Monitor your blood pressure regularly and maintain a healthy lifestyle.")
        
        # Cholesterol recommendations
        cholesterol = health_data.get('cholesterol', 0)
        if cholesterol > 240:
            recommendations.append("High cholesterol detected. Consider dietary changes and possible medication.")
        elif cholesterol > 200:
            recommendations.append("Maintain healthy cholesterol through diet and exercise.")
        
        # BMI recommendations
        bmi = health_data.get('bmi', 0)
        if bmi > 30:
            recommendations.append("Overweight/Obese: Aim for healthy weight through exercise and nutrition.")
        elif bmi > 25:
            recommendations.append("Maintain a healthy weight through regular exercise.")
        
        # Glucose recommendations
        glucose = health_data.get('glucose', 0)
        if glucose > 126:
            recommendations.append("High glucose levels detected. Get tested for diabetes.")
        
        # Exercise recommendations
        recommendations.append("Engage in 150 minutes of moderate aerobic exercise per week.")
        
        # General lifestyle
        if risk_category == 'high':
            recommendations.append("Urgent: Consult a cardiologist for detailed risk assessment and management plan.")
        
        recommendations.append("Maintain a heart-healthy diet rich in fruits, vegetables, and whole grains.")
        
        return recommendations
