import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="PulseMetrics AI", page_icon="⚡", layout="wide")

# Professional Custom UI
st.markdown("""
<style>
    .stApp { background-color: #030712; color: #e2e8f0; }
    .stButton>button { 
        background: linear-gradient(135deg, #4f46e5, #9333ea); 
        color: white; border: none; font-weight: 600; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def system_init():
    # 1. AI Engine - Multiple Model Fallback
    ai_eng = None
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        for m_id in ['gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                m = genai.GenerativeModel(m_id)
                m.generate_content("hi", generation_config={"max_output_tokens": 1})
                ai_eng = m
                break
            except: continue

    # 2. Neural Model - Fixed Batch Shape Error
    scaler = None
    model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        # compile=False prevents Keras from trying to rebuild old layers
        model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.sidebar.error(f"Neural Core Error: {str(e)}")
        
    return ai_eng, scaler, model

ai_engine, scaler, risk_model = system_init()

st.title("⚡ PulseMetrics AI")
st.caption("Advanced Cardiovascular Intelligence Platform")

t1, t2 = st.tabs(["AI Assistant", "Neural Scan"])

# --- TAB 1: AI ASSISTANT ---
with t1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    v_query = speech_to_text(language='en', start_prompt="🎙️ Voice", key='v_in')
    t_query = st.chat_input("Enter symptoms...")
    
    query = v_query if v_query else t_query

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        if ai_engine:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        res = ai_engine.generate_content(query)
                        st.markdown(res.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"Engine Error: {e}")

# --- TAB 2: NEURAL SCAN ---
with t2:
    with st.form("scan_form"):
        st.write("Enter patient metrics:")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 45)
            chol = st.number_input("Cholesterol", 100, 400, 200)
            sys = st.number_input("Systolic BP", 80, 200, 120)
        with c2:
            bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
            glu = st.number_input("Glucose", 60, 300, 95)
            hr = st.number_input("Heart Rate", 40, 160, 72)
            
        if st.form_submit_button("PROCESS RISK SCAN"):
            if risk_model and scaler:
                # 15-parameter sequence check
                data = [1, age, 2, 0, 0, 0, 0, 0, 0, chol, sys, 80, bmi, hr, glu]
                scaled = scaler.transform(np.array(data).reshape(1, -1))
                # Predict with batch dimension
                prob = float(risk_model.predict(np.expand_dims(scaled, axis=2), verbose=0)[0][0])
                
                if prob > 0.5:
                    st.error(f"High Risk Alert: {prob*100:.1f}%")
                else:
                    st.success(f"Low Risk Profile: {prob*100:.1f}%")
