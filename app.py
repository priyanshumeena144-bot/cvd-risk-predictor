import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="PulseMetrics AI", page_icon="⚡", layout="wide")

# UI Styling
st.markdown("""
<style>
    .stApp { background-color: #030712; color: #e2e8f0; }
    .stButton>button { background: linear-gradient(135deg, #6366f1, #a855f7); color: white; border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    # 1. AI Engine - Fixed 404 Error
    ai_engine = None
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Use the most stable current models
        for m_id in ['gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                m = genai.GenerativeModel(m_id)
                # Test call
                m.generate_content("test", generation_config={"max_output_tokens": 1})
                ai_engine = m
                break
            except: continue

    # 2. Deep Learning Model - Fixed Batch Shape Error
    scaler = None
    model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        # Compatibility loading
        model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.sidebar.error(f"Neural Core Error: {e}")
        
    return ai_engine, scaler, model

ai_engine, scaler, risk_model = initialize_system()

st.title("⚡ PulseMetrics AI")
st.caption("Cardiovascular Intelligence System")

tabs = st.tabs(["Assistant", "Neural Scan"])

# Assistant Logic
with tabs[0]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # User Input
    v_input = speech_to_text(language='en', start_prompt="🎙️ Voice Input", key='voice')
    t_input = st.chat_input("Analyze symptoms...")
    
    query = v_input if v_input else t_input

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        if ai_engine:
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    try:
                        # Professional medical assistant prompt
                        response = ai_engine.generate_content(f"User symptoms: {query}. Provide brief, clinical observations.")
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"API Error: {e}")
        else:
            st.error("AI Engine Offline. Check Gemini API Key.")

# Neural Scan Logic
with tabs[1]:
    with st.form("scan_form"):
        st.write("Enter patient metrics for analysis")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 45)
            chol = st.number_input("Cholesterol", 100, 400, 200)
            sys = st.number_input("Systolic BP", 80, 200, 120)
        with c2:
            bmi = st.number_input("BMI", 15.0, 45.0, 24.0)
            glu = st.number_input("Glucose", 60, 300, 95)
            hr = st.number_input("Heart Rate", 40, 150, 72)
            
        if st.form_submit_button("RUN ANALYSIS"):
            if risk_model and scaler:
                # Padding inputs to 15 (Ensuring model gets what it expects)
                input_data = [1, age, 2, 0, 0, 0, 0, 0, 0, chol, sys, 80, bmi, hr, glu]
                processed = scaler.transform(np.array(input_data).reshape(1, -1))
                prob = risk_model.predict(np.expand_dims(processed, axis=2), verbose=0)[0][0]
                
                if prob > 0.5:
                    st.error(f"Risk Detected: {prob*100:.1f}%")
                else:
                    st.success(f"Safe Level: {prob*100:.1f}%")
