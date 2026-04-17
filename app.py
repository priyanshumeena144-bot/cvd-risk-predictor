import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# 1. Dashboard Basics
st.set_page_config(page_title="PulseMetrics AI", page_icon="⚡", layout="wide")

# 2. Advanced Initialization (Fixed Batch_Shape & 404 Errors)
@st.cache_resource
def system_init():
    # AI Engine Setup - Fixed Model IDs
    ai_eng = None
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Loop handles 404 by trying multiple stable versions
        for m_id in ['gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                m = genai.GenerativeModel(m_id)
                m.generate_content("ping", generation_config={"max_output_tokens": 1})
                ai_eng = m
                break
            except: continue

    # Neural Core Setup - Forced Legacy Loading
    scaler = None
    model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        # Using compile=False prevents Keras from trying to 'rebuild' old layers
        model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.sidebar.error(f"Neural Core Offline: {str(e)}")
        
    return ai_eng, scaler, model

ai_engine, scaler, risk_model = system_init()

# 3. Professional UI
st.markdown("<h1 style='text-align: center;'>⚡ PulseMetrics AI</h1>", unsafe_allow_html=True)

tab_1, tab_2 = st.tabs(["Clinical Assistant", "Neural Diagnostic"])

# Assistant Tab Logic
with tab_1:
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    for msg in st.session_state.chat_log:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Dual Input Support
    v_query = speech_to_text(language='en', start_prompt="🎙️ Voice Scan", key='v_input')
    t_query = st.chat_input("Describe symptoms or ask health questions...")
    
    final_q = v_query if v_query else t_query

    if final_q:
        st.session_state.chat_log.append({"role": "user", "content": final_q})
        with st.chat_message("user"):
            st.markdown(final_q)
            
        if ai_engine:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        res = ai_engine.generate_content(final_q)
                        st.markdown(res.text)
                        st.session_state.chat_log.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"Engine Error: {e}")

# Neural Diagnostic Tab Logic
with tab_2:
    st.subheader("Compute CVD Risk Score")
    with st.form("diag_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 45)
            chol = st.number_input("Cholesterol", 100, 400, 200)
            sys_bp = st.number_input("Systolic BP", 80, 200, 120)
        with c2:
            bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
            glucose = st.number_input("Glucose", 60, 300, 95)
            hr = st.number_input("Heart Rate", 40, 160, 72)

        if st.form_submit_button("PROCESS NEURAL SCAN"):
            if risk_model and scaler:
                # Aligning 15 inputs for your V4 model
                raw = [1, age, 2, 0, 0, 0, 0, 0, 0, chol, sys_bp, 80, bmi, hr, glucose]
                transformed = scaler.transform(np.array(raw).reshape(1, -1))
                # Predict with batch dimension for LSTM/CNN
                score = risk_model.predict(np.expand_dims(transformed, axis=2), verbose=0)[0][0]
                
                st.divider()
                if score > 0.5:
                    st.error(f"High Risk Detected: {score*100:.1f}% - Consult a physician.")
                else:
                    st.success(f"Low Risk Profile: {score*100:.1f}% - Maintain healthy lifestyle.")
