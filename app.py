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
    animation: glow 3s ease-in-out infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3)); }
    to { filter: drop-shadow(0 0 25px rgba(124, 58, 237, 0.5)); }
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
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.stat-card {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.08), rgba(124, 58, 237, 0.08));
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 212, 255, 0.4);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.1);
}

.stat-number {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}

.stat-label {
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s ease-in-out infinite;
}

.result-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s ease-in-out infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.1); }
    50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.2); }
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.1); }
    50% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.2); }
}

.result-percentage {
    font-family: 'Orbitron', monospace;
    font-size: 4rem;
    font-weight: 900;
    margin: 0.5rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.06);
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    color: #64748b;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 10px 24px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(124, 58, 237, 0.15)) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    padding: 12px 32px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.3) !important;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent) !important;
    margin: 2rem 0 !important;
}

.chat-user {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.05));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    margin-left: 20%;
}

.chat-ai {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(124, 58, 237, 0.05));
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    margin-right: 20%;
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

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020818; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00d4ff, #7c3aed);
    border-radius: 3px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# GEMINI AI SETUP
# ─────────────────────────────────────────
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-pro')
else:
    ai_model = None

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

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><div class="stat-number">CNN</div><div class="stat-label">Deep Learning</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><div class="stat-number">LSTM</div><div class="stat-label">Neural Network</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stat-number">15</div><div class="stat-label">Health Markers</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-card"><div class="stat-number">AI</div><div class="stat-label">Powered</div></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

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
    st.markdown("""
    <div class="glass-card">
        <p style="color:#94a3b8; font-size:1.1rem;">
        Tell me your symptoms — fever, chest pain, headache, anything.
        Our AI will analyze and give you personalized health guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 <b>HealthAI:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_voice, col_text = st.columns([1, 2])
    with col_voice:
        st.markdown("**🎙️ Speak your symptoms:**")
        voice_text = speech_to_text(language='en', start_prompt="🎤 Start Speaking", stop_prompt="⏹️ Stop", key='health_voice')
    with col_text:
        st.markdown("**⌨️ Or type here:**")
        typed_text = st.text_input("", placeholder="e.g. I have fever, headache and body pain since 2 days...", label_visibility="collapsed")

    user_input = voice_text if voice_text else typed_text

    if user_input and st.button("🔍 ANALYZE SYMPTOMS", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        if ai_model is None:
            response = "⚠️ AI service unavailable. Please add GEMINI_API_KEY in Streamlit Secrets."
        else:
            with st.spinner("🧠 AI analyzing your symptoms..."):
                try:
                    prompt = f"""You are HealthAI, an advanced medical AI assistant.
                    A patient says: "{user_input}"
                    Provide a helpful, structured response with:
                    1. 🔍 Possible Conditions (2-3 likely causes)
                    2. ⚠️ Warning Signs to watch for
                    3. 💊 Immediate Steps to take
                    4. 🏥 When to see a Doctor
                    5. 🌿 Home Remedies (if applicable)
                    Be empathetic, clear and concise. Always recommend consulting a doctor for serious symptoms."""
                    result = ai_model.generate_content(prompt)
                    response = result.text
                except Exception as e:
                    response = f"Error: {str(e)}"
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
    st.markdown("""
    <div class="glass-card">
        <p style="color:#94a3b8;">Speak your health details and our CNN+LSTM model will predict your 10-year cardiovascular risk.</p>
        <p style="color:#00d4ff;">💬 Example: <i>"Male, age 45, BMI 28, BP 130 over 80, cholesterol 210, glucose 90, heart rate 75"</i></p>
    </div>
    """, unsafe_allow_html=True)

    voice_cvd = speech_to_text(language='en', start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", key='cvd_voice')

    if voice_cvd:
        st.markdown(f'<div class="chat-user">🎙️ You said: <b>{voice_cvd}</b></div>', unsafe_allow_html=True)
        if ai_model and model_loaded:
            try:
                with st.spinner("🤖 Extracting health features..."):
                    prompt = f"""Extract exactly 15 numerical features from: "{voice_cvd}"
                    Order: male(0/1), age, education(1-4), currentSmoker(0/1), cigsPerDay, BPMeds(0/1), prevalentStroke(0/1), prevalentHyp(0/1), diabetes(0/1), totChol, sysBP, diaBP, BMI, heartRate, glucose
                    Return ONLY a Python list of 15 numbers."""
                    result = ai_model.generate_content(prompt)
                    text = result.text.strip().replace("```", "").replace("python", "").strip()
                    features = json.loads(text)

                if len(features) == 15:
                    with st.spinner("🧠 Running CNN+LSTM prediction..."):
                        features_np = np.array(features).reshape(1, -1)
                        scaled = scaler.transform(features_np)
                        reshaped = np.expand_dims(scaled, axis=2)
                        prob = float(model.predict(reshaped)[0][0])

                    if prob > 0.5:
                        st.markdown(f'<div class="result-high"><div style="font-size:1.2rem;color:#ef4444;letter-spacing:3px;">⚠️ HIGH RISK DETECTED</div><div class="result-percentage" style="color:#ef4444;">{prob*100:.1f}%</div><div style="color:#94a3b8;">10-Year CVD Risk</div></div>', unsafe_allow_html=True)
                        st.error("Please consult a cardiologist immediately.")
                    else:
                        st.markdown(f'<div class="result-low"><div style="font-size:1.2rem;color:#10b981;letter-spacing:3px;">✅ LOW RISK</div><div class="result-percentage" style="color:#10b981;">{prob*100:.1f}%</div><div style="color:#94a3b8;">10-Year CVD Risk</div></div>', unsafe_allow_html=True)
                        st.success("Keep maintaining a healthy lifestyle!")
            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════
# TAB 3: MANUAL CVD SCAN
# ═══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📋 Manual CVD Risk Scan</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**👤 Patient Profile**")
        age = st.number_input("Age", 18, 100, 50)
        male = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.number_input("Education (1–4)", 1.0, 4.0, 2.0, 0.1)
    with col_b:
        st.markdown("**🚬 Lifestyle**")
        currentSmoker = st.selectbox("Current Smoker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cigsPerDay = st.number_input("Cigarettes per Day", 0, 100, 0)
        BPMeds = st.selectbox("BP Medication?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col_c:
        st.markdown("**🏥 Medical History**")
        prevalentStroke = st.selectbox("History of Stroke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prevalentHyp = st.selectbox("History of Hypertension?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("History of Diabetes?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🩺 Medical Vitals**")
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        totChol = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 200)
        sysBP = st.number_input("Systolic BP", 80.0, 300.0, 120.0, 0.1)
    with col_e:
        diaBP = st.number_input("Diastolic BP", 50.0, 200.0, 80.0, 0.1)
        BMI = st.number_input("BMI", 15.0, 60.0, 25.0, 0.1)
    with col_f:
        heartRate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
        glucose = st.number_input("Glucose (mg/dL)", 50, 400, 80)

    if st.button("🔬 RUN CVD ANALYSIS", type="primary", use_container_width=True):
        if not model_loaded:
            st.error(f"Model Error: {model_error}")
        else:
            features_list = [male, age, education, currentSmoker, cigsPerDay, BPMeds,
                           prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]
            try:
                with st.spinner("🧠 CNN+LSTM analyzing..."):
                    features_np = np.array(features_list).reshape(1, -1)
                    scaled = scaler.transform(features_np)
                    reshaped = np.expand_dims(scaled, axis=2)
                    prob = float(model.predict(reshaped)[0][0])

                if prob > 0.5:
                    st.markdown(f'<div class="result-high"><div style="font-size:1.2rem;color:#ef4444;letter-spacing:3px;">⚠️ HIGH RISK DETECTED</div><div class="result-percentage" style="color:#ef4444;">{prob*100:.1f}%</div><div style="color:#94a3b8;">10-Year CVD Risk</div></div>', unsafe_allow_html=True)
                    st.error("Please consult a cardiologist immediately.")
                else:
                    st.markdown(f'<div class="result-low"><div style="font-size:1.2rem;color:#10b981;letter-spacing:3px;">✅ LOW RISK</div><div class="result-percentage" style="color:#10b981;">{prob*100:.1f}%</div><div style="color:#94a3b8;">10-Year CVD Risk</div></div>', unsafe_allow_html=True)
                    st.success("Keep maintaining a healthy lifestyle!")
            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════
# TAB 4: CAMERA
# ═══════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📷 Camera Analysis</div>', unsafe_allow_html=True)
    col_cam, col_info = st.columns([1, 1])
    with col_cam:
        img = st.camera_input("Take a photo for future AI analysis")
    with col_info:
        if img:
            st.image(img, caption="Photo captured!", use_container_width=True)
            st.success("✅ Photo saved!")
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">🔮 Coming Soon</div>
            <p style="color:#94a3b8;">📄 <b>Medical Report Scan</b> — Auto-fill from photo</p>
            <p style="color:#94a3b8;">🧠 <b>AI Image Analysis</b> — Detect health indicators</p>
            <p style="color:#94a3b8;">👤 <b>Face Age Estimation</b> — AI age prediction</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-family:'Rajdhani',sans-serif; letter-spacing:2px; font-size:0.85rem;">
    ⚠️ FOR EDUCATIONAL & RESEARCH PURPOSES ONLY — CONSULT A QUALIFIED DOCTOR FOR MEDICAL ADVICE
</div>
""", unsafe_allow_html=True)
