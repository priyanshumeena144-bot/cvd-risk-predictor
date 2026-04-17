import streamlit as st
import joblib
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
# CORE CONFIGURATION & UI THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseMetrics AI | Cardiovascular Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Professional UI (Glassmorphism + Cyberpunk Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@500&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --accent-color: #00d4ff;
    }

    .stApp {
        background-color: #030712;
        font-family: 'Inter', sans-serif;
    }

    /* Main Branding */
    .brand-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(to bottom, rgba(99, 102, 241, 0.05), transparent);
    }

    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Professional Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND: SYSTEM INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_system_core():
    """Initializes AI Engine and Deep Learning Models with Error Redundancy."""
    # 1. AI Engine Setup (Handles 404 & Beta version issues)
    ai_engine = None
    if "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            # Fallback chain for different API environments
            for model_id in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']:
                try:
                    m = genai.GenerativeModel(model_id)
                    # Simple ping to verify model availability
                    m.generate_content("init", generation_config={"max_output_tokens": 1})
                    ai_engine = m
                    break
                except: continue
        except Exception: pass

    # 2. Deep Learning Assets (Handles Keras 3 Compatibility Errors)
    scaler = None
    risk_model = None
    try:
        scaler = joblib.load('my_scaler.joblib')
        # Using tf.keras load for legacy .h5 support
        risk_model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.sidebar.error(f"System Load Error: {str(e)}")

    return ai_engine, scaler, risk_model

ai_engine, scaler, risk_model = load_system_core()

# ──────────────────────────────────────────────────────────────────────────────
# UI: HEADER SECTION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="brand-header">
        <div class="main-title">PULSE METRICS AI</div>
        <p style="color: #64748b; font-weight: 500;">Clinical-Grade Cardiovascular Risk Analysis Platform</p>
    </div>
""", unsafe_allow_html=True)

# System Status Bar
c1, c2, c3 = st.columns([1, 1, 4])
with c1:
    status = "Online" if ai_engine else "Offline"
    color = "#10b981" if ai_engine else "#ef4444"
    st.markdown(f"● <span style='color:{color}; font-size:0.8rem;'>AI Engine: {status}</span>", unsafe_allow_html=True)
with c2:
    status = "Ready" if risk_model else "Error"
    color = "#10b981" if risk_model else "#ef4444"
    st.markdown(f"● <span style='color:{color}; font-size:0.8rem;'>Neural Core: {status}</span>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# UI: FUNCTIONAL TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Assistant", "Neural Scan", "Manual Entry"])

# --- TAB 1: COGNITIVE ASSISTANT ---
with tabs[0]:
    if "chat_stack" not in st.session_state:
        st.session_state.chat_stack = []

    # Chat Container
    for msg in st.session_state.chat_stack:
        align = "flex-end" if msg["role"] == "user" else "flex-start"
        bg = "rgba(99, 102, 241, 0.1)" if msg["role"] == "user" else "rgba(255, 255, 255, 0.03)"
        st.markdown(f"""
            <div style="display: flex; justify-content: {align}; margin-bottom: 1rem;">
                <div style="max-width: 80%; padding: 1rem; background: {bg}; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <small style="color: #64748b; text-transform: uppercase; font-size: 0.7rem;">{msg['role']}</small><br>
                    <div style="color: #e2e8f0; font-size: 0.95rem;">{msg['content']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Inputs
    st.markdown("<br>", unsafe_allow_html=True)
    voice_query = speech_to_text(language='en', start_prompt="🎙️ Stream Voice", key='main_voice')
    text_query = st.chat_input("Query clinical symptoms or health guidance...")

    final_query = voice_query if voice_query else text_query

    if final_query:
        st.session_state.chat_stack.append({"role": "user", "content": final_query})
        if ai_engine:
            with st.spinner("Processing Clinical Data..."):
                try:
                    # Professional Prompting
                    prompt = f"As a medical assistant, analyze: '{final_query}'. Provide concise potential observations and necessary precautions. No boilerplate."
                    response = ai_engine.generate_content(prompt)
                    st.session_state.chat_stack.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Engine Timeout: {str(e)}")
        st.rerun()

# --- TAB 2: NEURAL SCAN (VOICE DATA EXTRACTION) ---
with tabs[1]:
    st.markdown("### Neural Data Extraction")
    st.write("Dictate your clinical metrics (Age, BP, Cholesterol, etc.) for automated risk assessment.")
    
    v_data = speech_to_text(language='en', start_prompt="🎤 Initiate Health Stream", key='scan_voice')
    
    if v_data:
        st.info(f"Stream Captured: {v_data}")
        if ai_engine and risk_model:
            with st.spinner("Extracting Neural Features..."):
                try:
                    p = f"Convert this to a 15-float JSON list: male(0/1),age,edu(1-4),smoker(0/1),cigs,bpmeds(0/1),stroke(0/1),hyp(0/1),diab(0/1),chol,sysBP,diaBP,bmi,hr,gluc. Input: {v_data}. Output ONLY the list."
                    res = ai_engine.generate_content(p)
                    # Robust extraction logic
                    clean_res = res.text.replace('[','').replace(']','').split(',')
                    features = [float(val.strip()) for val in clean_res if val.strip()]
                    
                    if len(features) == 15:
                        processed_data = scaler.transform(np.array(features).reshape(1, -1))
                        prediction = risk_model.predict(np.expand_dims(processed_data, axis=2))[0][0]
                        
                        # Visualization
                        st.divider()
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.metric("Risk Probability", f"{float(prediction)*100:.1f}%")
                        with col_r:
                            risk_status = "CRITICAL" if prediction > 0.5 else "LOW"
                            st.markdown(f"#### Status: <span style='color:{'#ef4444' if risk_status == 'CRITICAL' else '#10b981'}'>{risk_status}</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Neural extraction failed. Please ensure all 15 metrics are mentioned.")

# --- TAB 3: MANUAL ENTRY ---
with tabs[2]:
    with st.form("clinical_form"):
        st.markdown("#### Clinical Input Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            in_sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            in_age = st.number_input("Patient Age", 18, 110, 45)
            in_edu = st.slider("Education Level", 1, 4, 2)
            in_smoke = st.toggle("Active Smoker")
            in_cigs = st.number_input("Cigarettes / Day", 0, 100, 0)
        with c2:
            in_bpm = st.toggle("On BP Medication")
            in_str = st.toggle("Stroke History")
            in_hyp = st.toggle("Hypertension History")
            in_diab = st.toggle("Diabetes Diagnosis")
            in_chol = st.number_input("Total Cholesterol (mg/dL)", 100, 500, 200)
        with c3:
            in_sys = st.number_input("Systolic BP", 80, 250, 120)
            in_dia = st.number_input("Diastolic BP", 40, 150, 80)
            in_bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
            in_hr = st.number_input("Resting Heart Rate", 40, 200, 72)
            in_glu = st.number_input("Glucose Level", 50, 400, 90)

        if st.form_submit_button("COMPUTE RISK SCORE", use_container_width=True):
            if risk_model:
                raw_data = [in_sex, in_age, in_edu, int(in_smoke), in_cigs, int(in_bpm), int(in_str), int(in_hyp), int(in_diab), in_chol, in_sys, in_dia, in_bmi, in_hr, in_glu]
                scaled_data = scaler.transform(np.array(raw_data).reshape(1, -1))
                prob = float(risk_model.predict(np.expand_dims(scaled_data, axis=2))[0][0])
                
                # Result Display
                if prob > 0.5:
                    st.error(f"High Risk Alert: {prob*100:.1f}%. Immediate clinical consultation recommended.")
                else:
                    st.success(f"Low Risk Profile: {prob*100:.1f}%. Continue monitoring.")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #475569; font-size: 0.8rem;'>
        © 2026 PulseMetrics AI Research. Non-Diagnostic Educational Tool.<br>
        Built with TensorFlow & Gemini Neural Engines.
    </div>
""", unsafe_allow_html=True)
