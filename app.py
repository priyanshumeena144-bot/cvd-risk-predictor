import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="PulseMetrics AI", page_icon="⚡", layout="wide")

# Industrial Stealth Theme (Pro UI)
st.markdown("""
<style>
    .stApp { background-color: #030712; color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton>button { 
        background: linear-gradient(135deg, #4f46e5, #9333ea); 
        color: white; border: none; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def system_startup():
    # 1. Gemini Engine Initialization
    ai_engine = None
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Stable model identification
        available_models = ['gemini-1.5-flash', 'gemini-1.5-pro']
        for m_id in available_models:
            try:
                m = genai.GenerativeModel(m_id)
                m.generate_content("ping", generation_config={"max_output_tokens": 1})
                ai_engine = m
                break
            except Exception: continue

    # 2. Deep Learning Core Initialization
    scaler = None
    model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        # Compatibility Mode: Loading via tf.keras with manual config override
        model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.sidebar.warning(f"Engine Core: {str(e)}")
        
    return ai_engine, scaler, model

ai_engine, scaler, risk_model = system_startup()

# UI Layout
st.title("⚡ PulseMetrics AI")
st.caption("Research-Grade Cardiovascular Intelligence Platform")

tab_assistant, tab_neural = st.tabs(["AI Assistant", "Neural Scan"])

# --- AI ASSISTANT LOGIC ---
with tab_assistant:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input handlers
    col_v, col_t = st.columns([1, 4])
    with col_v:
        v_input = speech_to_text(language='en', start_prompt="🎙️ Voice", key='v_main')
    with col_t:
        t_input = st.chat_input("Enter clinical symptoms...")

    query = v_input if v_input else t_input

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        if ai_engine:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        res = ai_engine.generate_content(f"Analyze symptoms: {query}. Keep advice concise.")
                        st.markdown(res.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"Engine Error: {e}")
        else:
            st.error("AI Engine Offline. Check Secrets.")

# --- NEURAL SCAN LOGIC ---
with tab_neural:
    st.subheader("Clinical Metric Analysis")
    with st.form("scan_input"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 100, 40)
            chol = st.number_input("Cholesterol", 100, 400, 200)
        with c2:
            sys = st.number_input("Systolic BP", 80, 200, 120)
            hr = st.number_input("Heart Rate", 40, 150, 72)
        with c3:
            bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
            glu = st.number_input("Glucose", 60, 300, 90)
            
        if st.form_submit_button("PROCESS SCAN"):
            if risk_model and scaler:
                # 15-feature alignment for model V4
                features = [1, age, 2, 0, 0, 0, 0, 0, 0, chol, sys, 80, bmi, hr, glu]
                scaled = scaler.transform(np.array(features).reshape(1, -1))
                prob = float(risk_model.predict(np.expand_dims(scaled, axis=2), verbose=0)[0][0])
                
                if prob > 0.5:
                    st.error(f"CVD Risk Detected: {prob*100:.1f}%")
                else:
                    st.success(f"Low Clinical Risk: {prob*100:.1f}%")
