import streamlit as st
import joblib
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from streamlit_mic_recorder import speech_to_text
import google.generativeai as genai

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="HealthAI", page_icon="🫀", layout="wide")

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #020818; color: #e2e8f0; }
.stApp { background: linear-gradient(135deg, #020818 0%, #0d1f3c 100%); min-height: 100vh; }
.hero-title { font-family: 'Orbitron'; font-size: 3rem; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-weight: 900; }
.chat-box { padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.1); }
.user-msg { background: rgba(0, 212, 255, 0.1); border-left: 5px solid #00d4ff; }
.ai-msg { background: rgba(124, 58, 237, 0.1); border-left: 5px solid #7c3aed; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# AI & ASSETS INITIALIZATION
# ─────────────────────────────────────────
@st.cache_resource
def setup_resources():
    ai_mod = None
    scaler_mod = None
    deep_model = None
    
    # 1. AI Setup
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        for m_name in ['gemini-1.5-flash', 'gemini-1.0-pro']:
            try:
                test_model = genai.GenerativeModel(m_name)
                ai_mod = test_model
                break
            except:
                continue

    # 2. Asset Loading (Fixed Indentation & Error Handling)
    try:
        scaler_mod = joblib.load('my_scaler.joblib')
        # Using tf.keras for better compatibility with .h5
        deep_model = tf.keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.error(f"Resource Loading Error: {e}")
        
    return ai_mod, scaler_mod, deep_model

ai_model, scaler, model = setup_resources()

# ─────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">🫀 HEALTH AI</div>', unsafe_allow_html=True)

tabs = st.tabs(["🤖 Assistant", "🎙️ Voice Scan", "📋 Manual Scan"])

# --- TAB 1: ASSISTANT ---
with tabs[0]:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        msg_type = "user-msg" if m["role"] == "user" else "ai-msg"
        st.markdown(f'<div class="chat-box {msg_type}"><b>{m["role"].upper()}:</b><br>{m["content"]}</div>', unsafe_allow_html=True)

    v_input = speech_to_text(language='en', start_prompt="🎙️ Speak Symptoms", key='voice_main')
    t_input = st.text_input("Or type here...", key="text_main")

    final_query = v_input if v_input else t_input

    if st.button("🔍 ANALYZE") and final_query:
        st.session_state.messages.append({"role": "user", "content": final_query})
        if ai_model:
            with st.spinner("Analyzing..."):
                try:
                    res = ai_model.generate_content(final_query)
                    st.session_state.messages.append({"role": "assistant", "content": res.text})
                except Exception as e:
                    st.error(f"AI Error: {e}")
        st.rerun()

# --- TAB 2 & 3: Scan Logic ---
with tabs[1]:
    st.info("Speak metrics like 'Age 45, Cholesterol 200...'")
    # Add your voice parsing logic here as needed

with tabs[2]:
    st.info("Enter metrics manually for deep analysis.")
