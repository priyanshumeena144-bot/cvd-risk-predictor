import streamlit as st
import numpy as np
import joblib
from tensorflow import keras
import requests # Yeh abhi bhi hai, lekin hum isse use nahi kar rahe

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# --- 2. Model Loading (From app.py) ---
# Model aur scaler ko app ke start hote hi load karein
# @st.cache_resource decorator ka matlab hai ki model sirf ek baar load hoga
@st.cache_resource
def load_model_and_scaler():
    print("--- Loading model and scaler ---")
    try:
        scaler = joblib.load('my_scaler.joblib')
        model = keras.models.load_model('my_cnn_lstm_model.keras')
        print("✅ Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None

model, scaler = load_model_and_scaler()

# --- 3. Title and Header ---
st.title("❤️ Cardiovascular Risk Predictor")
st.write(
    "Enter your health details below to predict your 10-year risk of CHD. "
    "This app uses an AI model (CNN+LSTM+MLP) to provide a risk score."
)

# --- 4. Input Form ---
st.header("Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    male = st.selectbox("Gender (male=1, female=0)", options=[0, 1])
    education = st.number_input("Education (e.g., 1-4)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

with col2:
    currentSmoker = st.selectbox("Are you a current smoker? (yes=1, no=0)", options=[0, 1])
    cigsPerDay = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=0)
    BPMeds = st.selectbox("On Blood Pressure Medication? (yes=1, no=0)", options=[0, 1])

with col3:
    prevalentStroke = st.selectbox("History of Stroke? (yes=1, no=0)", options=[0, 1])
    prevalentHyp = st.selectbox("History of Hypertension? (yes=1, no=0)", options=[0, 1])
    diabetes = st.selectbox("History of Diabetes? (yes=1, no=0)", options=[0, 1])

st.divider()
st.header("Medical Vitals")
col4, col5, col6 = st.columns(3)

with col4:
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("Systolic Blood Pressure (sysBP)", min_value=80.0, max_value=300.0, value=120.0, step=0.1)
    diaBP = st.number_input("Diastolic Blood Pressure (diaBP)", min_value=50.0, max_value=200.0, value=80.0, step=0.1)
    
with col5:
    BMI = st.number_input("BMI (e.g., 25.4)", min_value=15.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=400, value=80)

# --- 5. Prediction Button and Logic ---
if st.button("Click Here to Predict Risk", type="primary"):
    
    if model is None or scaler is None:
        st.error("Model Error: The model or scaler failed to load. Please check the server logs.")
    else:
        # 1. Collect all 15 features
        features_list = [
            cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, 
            totChol, sysBP, diaBP, BMI, heartRate, glucose,
            male, age, education, currentSmoker
        ]

        try:
            # 2. Scale and reshape data (logic from app.py)
            features_np = np.array(features_list).reshape(1, -1)
            scaled_features = scaler.transform(features_np)
            reshaped_features = np.expand_dims(scaled_features, axis=2)
            
            # 3. Make the prediction directly
            prediction = model.predict(reshaped_features)
            probability = float(prediction[0][0])
            
            # 4. Display the result
            st.subheader(f"Risk Prediction Result:")
            st.metric(
                label="10-Year CHD Risk Probability",
                value=f"{probability * 100:.2f} %"
            )
            
            if probability > 0.5:
                st.error("The model predicts a HIGH risk of cardiovascular disease.")
            else:
                st.success("The model predicts a LOW risk of cardiovascular disease.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")