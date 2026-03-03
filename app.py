import streamlit as st
import numpy as np
import joblib
from tensorflow import keras
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai
import json

# --- 1. Load Model & Scaler ---
@st.cache_resource
def load_assets():
    scaler = joblib.load('my_scaler.joblib')
    model = keras.models.load_model('my_cnn_lstm_model.keras')
    return scaler, model

scaler, model = load_assets()

# --- 2. AI Setup (Gemini) ---
# Secrets se API Key uthayega
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-pro')
else:
    st.error("Please add GEMINI_API_KEY in Streamlit Secrets!")

def extract_features_with_ai(user_text):
    prompt = f"""
    Analyze this medical data: "{user_text}". 
    Extract 15 numerical features for CVD prediction. 
    Format: A simple Python list of 15 numbers only.
    Example: [50, 1, 165, 75, 120, 80, 1, 1, 0, 0, 1, 25.4, 0, 1, 0]
    """
    response = ai_model.generate_content(prompt)
    return json.loads(response.text)

# --- 3. UI Interface ---
st.title("🎙️ Voice-Powered CVD Predictor")

# Voice Button
text = speech_to_text(language='en', start_prompt="Boliye: 'Age 45, BMI 28, BP 130/80...'", key='voice_input')

if text:
    st.info(f"Aapne kaha: {text}")
    try:
        with st.spinner("AI processing..."):
            features = extract_features_with_ai(text)
            
        if len(features) == 15:
            # --- 4. Prediction ---
            features_np = np.array(features).reshape(1, -1)
            scaled = scaler.transform(features_np)
            reshaped = np.expand_dims(scaled, axis=2)
            
            prob = model.predict(reshaped)[0][0]
            
            st.metric("Risk Probability", f"{round(prob*100, 2)}%")
            if prob > 0.5:
                st.error("High Risk!")
            else:
                st.success("Low Risk!")
        else:
            st.warning("Could not extract all 15 features. Try again.")
    except Exception as e:
        st.error(f"Error: {e}")
    
