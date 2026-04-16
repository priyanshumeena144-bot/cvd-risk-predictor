import streamlit as st
import joblib
import numpy as np
import json
from tensorflow import keras
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HealthAI — Your Smart Health Assistant",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Glassmorphism Dark Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #020818;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #020818 0%, #0a1628 40%, #0d1f3c 70%, #020818 100%);
    min-height: 100vh;
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff, #7c3aed, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    color: #64748b;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
}

.chat-user {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.05));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    margin-left: 15%;
}

.chat-ai {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(124, 58, 237, 0.05));
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    margin-right: 15%;
}

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
}

.result-high { background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 20px; padding: 2rem; text-align: center; }
.result-low { background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 20px; padding: 2rem; text-align: center; }
.result-percentage { font-family: 'Orbitron'; font-size: 3rem; font-weight: 900; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# GEMINI AI SETUP (FIXED 404 ERROR)
# ─────────────────────────────────────────
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Changed from 'gemini-pro' to 'gemini-1.5-flash' to fix the 404 error
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    ai_model = None

# ─────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler = joblib.load('my_scaler.joblib')
    # Make sure this file exists in your repository
    model = keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    return scaler, model

try:
    scaler, model = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">🫀 HEALTH AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Advanced Cardiovascular Risk Intelligence System</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 AI Health Assistant",
    "🎙️ Voice CVD Scan",
    "📋 Manual CVD Scan",
    "📷 Camera"
])

# ═══════════════════════════════════════
# TAB 1: AI HEALTH ASSISTANT
# ═══════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🤖 AI Health Assistant</div>', unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display Chat
    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{role_class}">{icon} <b>{msg["role"].capitalize()}:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_voice, col_text = st.columns([1, 2])
    with col_voice:
        st.markdown("**🎙️ Speak:**")
        voice_input = speech_to_text(language='en', start_prompt="🎤 Start Speaking", stop_prompt="⏹️ Stop", key='health_voice')
    with col_text:
        st.markdown("**⌨️ Type:**")
        typed_input = st.text_input("Input", placeholder="Describe your symptoms...", label_visibility="collapsed")

    final_input = voice_input if voice_input else typed_input

    if final_input and st.button("🔍 ANALYZE SYMPTOMS", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": final_input})
        
        if ai_model is None:
            response = "⚠️ AI key missing in Secrets."
        else:
            with st.spinner("🧠 Analyzing..."):
                try:
                    prompt = f"Patient says: '{final_input}'. Give structured health guidance: 1. Possible Conditions, 2. Warning Signs, 3. Immediate Steps, 4. When to see a Doctor. Be concise."
                    result = ai_model.generate_content(prompt)
                    response = result.text
                except Exception as e:
                    response = f"API Error: {str(e)}"
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ═══════════════════════════════════════
# TAB 2: VOICE CVD SCAN
# ═══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎙️ Voice CVD Risk Scan</div>', unsafe_allow_html=True)
    voice_cvd = speech_to_text(language='en', start_prompt="🎤 Record Health Details", stop_prompt="⏹️ Stop", key='cvd_voice')

    if voice_cvd:
        st.info(f"Detected: {voice_cvd}")
        if ai_model and model_loaded:
            try:
                with st.spinner("Extracting features..."):
                    prompt = f"Extract 15 numbers for CVD features from: '{voice_cvd}'. Male(0/1), age, education(1-4), currentSmoker(0/1), cigsPerDay, BPMeds(0/1), prevalentStroke(0/1), prevalentHyp(0/1), diabetes(0/1), totChol, sysBP, diaBP, BMI, heartRate, glucose. Return ONLY a JSON list."
                    result = ai_model.generate_content(prompt)
                    # Cleaning response
                    clean_text = result.text.strip().replace("```json", "").replace("```", "").strip()
                    features = json.loads(clean_text)

                if len(features) == 15:
                    feat_np = np.array(features).reshape(1, -1)
                    scaled = scaler.transform(feat_np)
                    reshaped = np.expand_dims(scaled, axis=2)
                    prob = float(model.predict(reshaped)[0][0])

                    if prob > 0.5:
                        st.markdown(f'<div class="result-high"><div class="result-percentage">{prob*100:.1f}%</div><p>HIGH RISK</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-low"><div class="result-percentage">{prob*100:.1f}%</div><p>LOW RISK</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing voice data: {e}")

# ═══════════════════════════════════════
# TAB 3: MANUAL CVD SCAN
# ═══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📋 Manual CVD Risk Scan</div>', unsafe_allow_html=True)
    with st.form("cvd_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_male = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            f_age = st.number_input("Age", 18, 100, 45)
            f_edu = st.slider("Education", 1, 4, 2)
        with c2:
            f_smoke = st.selectbox("Smoker?", [0, 1])
            f_cigs = st.number_input("Cigs/Day", 0, 100, 0)
            f_bpmeds = st.selectbox("BP Meds?", [0, 1])
        with c3:
            f_stroke = st.selectbox("Stroke History?", [0, 1])
            f_hyp = st.selectbox("Hypertension?", [0, 1])
            f_diab = st.selectbox("Diabetes?", [0, 1])
        
        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            f_chol = st.number_input("Cholesterol", 100, 500, 200)
            f_sys = st.number_input("Systolic BP", 80, 250, 120)
        with c5:
            f_dia = st.number_input("Diastolic BP", 40, 150, 80)
            f_bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        with c6:
            f_hr = st.number_input("Heart Rate", 40, 200, 72)
            f_gluc = st.number_input("Glucose", 50, 400, 90)

        submit = st.form_submit_button("🔬 ANALYZE RISK")

    if submit and model_loaded:
        feats = [f_male, f_age, f_edu, f_smoke, f_cigs, f_bpmeds, f_stroke, f_hyp, f_diab, f_chol, f_sys, f_dia, f_bmi, f_hr, f_gluc]
        scaled = scaler.transform(np.array(feats).reshape(1, -1))
        prob = float(model.predict(np.expand_dims(scaled, axis=2))[0][0])
        
        if prob > 0.5:
            st.markdown(f'<div class="result-high"><div class="result-percentage">{prob*100:.1f}%</div><p>HIGH RISK</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-low"><div class="result-percentage">{prob*100:.1f}%</div><p>LOW RISK</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 4: CAMERA
# ═══════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📷 Camera Analysis</div>', unsafe_allow_html=True)
    img = st.camera_input("Capture report or face")
    if img:
        st.image(img)
        st.success("Feature coming soon: Auto-extraction from reports!")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#475569;'>FOR RESEARCH PURPOSES ONLY • HealthAI © 2026</div>", unsafe_allow_html=True)
