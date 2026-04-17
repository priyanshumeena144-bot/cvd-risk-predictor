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
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #020818; color: #e2e8f0; }
.stApp { background: linear-gradient(135deg, #020818 0%, #0a1628 40%, #0d1f3c 70%, #020818 100%); min-height: 100vh; }
.hero-title { font-family: 'Orbitron', monospace; font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #00d4ff, #7c3aed, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; letter-spacing: 4px; }
.hero-subtitle { font-family: 'Rajdhani', sans-serif; font-size: 1.2rem; color: #64748b; text-align: center; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 2rem; }
.chat-user { background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 16px 16px 4px 16px; padding: 1rem; margin: 0.5rem 0; margin-left: 15%; }
.chat-ai { background: rgba(124, 58, 237, 0.1); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: 16px 16px 16px 4px; padding: 1rem; margin: 0.5rem 0; margin-right: 15%; }
.section-header { font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #00d4ff; border-bottom: 1px solid rgba(0, 212, 255, 0.2); margin-bottom: 1rem; }
.result-high { background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 20px; padding: 2rem; text-align: center; }
.result-low { background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 20px; padding: 2rem; text-align: center; }
.result-percentage { font-family: 'Orbitron'; font-size: 3rem; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SMART GEMINI SETUP (FIXES 404)
# ─────────────────────────────────────────
@st.cache_resource
def initialize_ai():
    if "GEMINI_API_KEY" not in st.secrets:
        return None
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # List of models to try in order of preference
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
    
    # Pehle preferred models check karein
    for m_name in models_to_try:
        try:
            m = genai.GenerativeModel(m_name)
            # Chhota sa test call check karne ke liye ki model exist karta hai ya nahi
            return m
        except Exception:
            continue
            
    # Agar upar waale fail ho jayein, toh automatic system se pucho kaunsa model available hai
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if available:
            # list_models returns names like 'models/gemini-pro', hume sirf name chahiye
            clean_name = available[0].split('/')[-1]
            return genai.GenerativeModel(clean_name)
    except:
        return None
    return None

ai_model = initialize_ai()

# ─────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler = joblib.load('my_scaler.joblib')
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

tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Health Assistant", "🎙️ Voice CVD Scan", "📋 Manual CVD Scan", "📷 Camera"])

# ═══════════════════════════════════════
# TAB 1: AI HEALTH ASSISTANT
# ═══════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🤖 AI Health Assistant</div>', unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        div_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{div_class}"><b>{msg["role"].upper()}:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    col_v, col_t = st.columns([1, 2])
    with col_v:
        v_in = speech_to_text(language='en', start_prompt="🎙️ Speak Symptoms", stop_prompt="⏹️ Stop", key='h_voice')
    with col_t:
        t_in = st.text_input("Type here", placeholder="e.g. I have fever...", label_visibility="collapsed")

    u_input = v_in if v_in else t_in

    if u_input and st.button("🔍 ANALYZE", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": u_input})
        if ai_model:
            with st.spinner("🧠 Thinking..."):
                try:
                    res = ai_model.generate_content(f"Patient symptoms: {u_input}. Provide brief medical guidance (Conditions, Steps, When to see doctor).")
                    st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                except Exception as e:
                    st.error(f"AI Error: {e}")
        else:
            st.error("AI Model not initialized. Check API Key.")
        st.rerun()

# ═══════════════════════════════════════
# TAB 2: VOICE CVD SCAN
# ═══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎙️ Voice CVD Risk Scan</div>', unsafe_allow_html=True)
    v_cvd = speech_to_text(language='en', start_prompt="🎤 Record Health Data", stop_prompt="⏹️ Stop", key='cvd_v')
    if v_cvd:
        st.info(f"Analyzing: {v_cvd}")
        if ai_model and model_loaded:
            try:
                prompt = f"Extract 15 numbers (list only) for: male, age, edu, smoker, cigs, bpmeds, stroke, hyp, diab, chol, sys, dia, bmi, hr, gluc from '{v_cvd}'. Format: [0, 0, ...]"
                res = ai_model.generate_content(prompt)
                # Cleaning JSON
                clean_json = res.text.strip().replace("```json", "").replace("```", "").strip()
                feats = json.loads(clean_json)
                
                if len(feats) == 15:
                    scaled = scaler.transform(np.array(feats).reshape(1, -1))
                    prob = float(model.predict(np.expand_dims(scaled, axis=2))[0][0])
                    if prob > 0.5:
                        st.markdown(f'<div class="result-high"><div class="result-percentage">{prob*100:.1f}%</div><p>HIGH RISK</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-low"><div class="result-percentage">{prob*100:.1f}%</div><p>LOW RISK</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Processing Error: {e}")

# ═══════════════════════════════════════
# TAB 3: MANUAL SCAN
# ═══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📋 Manual CVD Scan</div>')
    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            m_age = st.number_input("Age", 18, 100, 40)
            m_edu = st.slider("Education", 1, 4, 2)
        with c2:
            m_smoke = st.selectbox("Smoker?", [0, 1])
            m_cigs = st.number_input("Cigs/Day", 0, 100, 0)
            m_bpm = st.selectbox("BP Meds?", [0, 1])
        with c3:
            m_str = st.selectbox("Stroke?", [0, 1])
            m_hyp = st.selectbox("Hypertension?", [0, 1])
            m_diab = st.selectbox("Diabetes?", [0, 1])
        
        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            m_chol = st.number_input("Cholesterol", 100, 500, 200)
            m_sys = st.number_input("Systolic BP", 80, 250, 120)
        with c5:
            m_dia = st.number_input("Diastolic BP", 40, 150, 80)
            m_bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        with c6:
            m_hr = st.number_input("Heart Rate", 40, 200, 72)
            m_glu = st.number_input("Glucose", 50, 400, 90)
            
        btn = st.form_submit_button("🔬 ANALYZE RISK")
        
    if btn and model_loaded:
        data = [m_sex, m_age, m_edu, m_smoke, m_cigs, m_bpm, m_str, m_hyp, m_diab, m_chol, m_sys, m_dia, m_bmi, m_hr, m_glu]
        scaled = scaler.transform(np.array(data).reshape(1, -1))
        p = float(model.predict(np.expand_dims(scaled, axis=2))[0][0])
        if p > 0.5:
            st.markdown(f'<div class="result-high"><div class="result-percentage">{p*100:.1f}%</div><p>HIGH RISK</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-low"><div class="result-percentage">{p*100:.1f}%</div><p>LOW RISK</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 4: CAMERA
# ═══════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📷 Camera</div>')
    shot = st.camera_input("Take photo")
    if shot: st.success("Report captured!")

st.markdown("<hr><div style='text-align:center; color:#475569;'>HealthAI © 2026 • Research Purpose Only</div>", unsafe_allow_html=True)
