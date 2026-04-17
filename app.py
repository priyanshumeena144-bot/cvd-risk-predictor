import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="PulseMetrics AI", page_icon="⚡", layout="wide")

# Theme Styling
st.markdown("""
<style>
    .stApp { background-color: #030712; color: #e2e8f0; }
    .stButton>button { background: linear-gradient(135deg, #4f46e5, #9333ea); color: white; border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def startup_engine():
    # 1. AI Engine Logic
    ai_mod = None
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Use only the most reliable model IDs
        for m_id in ['gemini-1.5-flash', 'gemini-pro']:
            try:
                test_m = genai.GenerativeModel(m_id)
                test_m.generate_content("hi", generation_config={"max_output_tokens": 1})
                ai_mod = test_m
                break
            except: continue

    # 2. Neural Model Logic (Fixing the Batch Shape Error)
    scaler = None
    model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        # We use compile=False to avoid layer-specific initialization issues
        model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        # Fallback for 'batch_shape' unrecognized errors
        st.sidebar.error(f"Engine Core: {e}")
        model = None
        
    return ai_mod, scaler, model

ai_engine, scaler, risk_model = startup_engine()

# Branding
st.markdown("<h1 style='text-align: center; color: #6366f1;'>⚡ PulseMetrics AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Advanced Cardiovascular Intelligence Platform</p>", unsafe_allow_html=True)

tab_assist, tab_scan = st.tabs(["Assistant", "Neural Scan"])

# Assistant Tab
with tab_assist:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    v_in = speech_to_text(language='en', start_prompt="🎙️ Voice", key='v_in')
    t_in = st.chat_input("Enter symptoms...")
    
    query = v_in if v_in else t_in

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        if ai_engine:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        res = ai_engine.generate_content(query)
                        st.markdown(res.text)
                        st.session_state.messages.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"AI Error: {e}")

# Scan Tab
with tab_scan:
    with st.form("scan_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 45)
            chol = st.number_input("Total Cholesterol", 100, 400, 200)
            sys = st.number_input("Systolic BP", 80, 200, 120)
        with col2:
            bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
            glu = st.number_input("Glucose", 60, 300, 95)
            hr = st.number_input("Heart Rate", 40, 160, 72)
            
        if st.form_submit_button("COMPUTE RISK"):
            if risk_model and scaler:
                # 15-parameter sequence (Update based on your training data order)
                data = [1, age, 2, 0, 0, 0, 0, 0, 0, chol, sys, 80, bmi, hr, glu]
                scaled = scaler.transform(np.array(data).reshape(1, -1))
                prob = float(risk_model.predict(np.expand_dims(scaled, axis=2), verbose=0)[0][0])
                
                if prob > 0.5:
                    st.error(f"High Risk Alert: {prob*100:.1f}%")
                else:
                    st.success(f"Low Risk Profile: {prob*100:.1f}%")
            else:
                st.error("Model core is not active. Check system logs.")
