import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM INITIALIZATION (Optimized for Streamlit Cloud)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PulseMetrics AI", page_icon="⚡", layout="wide")

@st.cache_resource
def load_core():
    # AI Engine Setup
    ai_mod = None
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Try-catch for model selection
        try:
            ai_mod = genai.GenerativeModel('gemini-1.5-flash')
        except:
            try: ai_mod = genai.GenerativeModel('gemini-pro')
            except: ai_mod = None
            
    # Model Loading (Compatibility Mode)
    scaler = None
    model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
    
    return ai_mod, scaler, model

ai_engine, scaler, risk_model = load_core()

# ──────────────────────────────────────────────────────────────────────────────
# UI DESIGN (Industrial Stealth Theme)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #030712; color: #e2e8f0; }
    .stButton>button { 
        width: 100%; border-radius: 8px; background: linear-gradient(135deg, #6366f1, #a855f7); 
        color: white; font-weight: bold; border: none; height: 3rem;
    }
    .stTextInput>div>div>input { background-color: #1f2937; color: white; border: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ PulseMetrics AI")
st.caption("Advanced Cardiovascular Risk Analysis System")

tabs = st.tabs(["🤖 AI Assistant", "📋 Manual Scan"])

# --- TAB 1: ASSISTANT ---
with tabs[0]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display History
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input Logic (Direct - No Forms)
    prompt = st.chat_input("Ask about symptoms or reports...")
    
    # Voice handling
    v_input = speech_to_text(language='en', start_prompt="🎙️ Speak", key='voice_btn')
    
    final_q = v_input if v_input else prompt

    if final_q:
        st.session_state.chat_history.append({"role": "user", "content": final_q})
        with st.chat_message("user"):
            st.markdown(final_q)
            
        if ai_engine:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        res = ai_engine.generate_content(final_q)
                        st.markdown(res.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"AI Error: {e}")
        else:
            st.error("AI Key is missing. Please check your Streamlit Secrets.")

# --- TAB 2: MANUAL SCAN ---
with tabs[1]:
    st.markdown("### Clinical Parameters")
    c1, c2, c3 = st.columns(3)
    
    # Inputs (No Form for instant response)
    with c1:
        age = st.number_input("Age", 18, 100, 45)
        chol = st.number_input("Cholesterol", 100, 500, 200)
    with c2:
        sys_bp = st.number_input("Systolic BP", 80, 200, 120)
        dia_bp = st.number_input("Diastolic BP", 40, 150, 80)
    with c3:
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        smoke = st.selectbox("Smoker?", [0, 1])

    if st.button("RUN NEURAL ANALYSIS"):
        if risk_model and scaler:
            with st.spinner("Computing..."):
                # Padding inputs to 15 (Model expectation)
                # Note: Adjust the sequence based on your model's exact feature order
                data = [0, age, 2, smoke, 0, 0, 0, 0, 0, chol, sys_bp, dia_bp, bmi, 72, 90]
                
                scaled = scaler.transform(np.array(data).reshape(1, -1))
                prob = float(risk_model.predict(np.expand_dims(scaled, axis=2), verbose=0)[0][0])
                
                st.divider()
                if prob > 0.5:
                    st.error(f"High Risk Detected: {prob*100:.1f}%")
                else:
                    st.success(f"Low Risk Profile: {prob*100:.1f}%")
        else:
            st.error("Model not loaded. Check logs for Resource Loading Error.")
