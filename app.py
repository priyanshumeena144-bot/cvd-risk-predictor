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
# app.py ki line 48 ke aas-paas ise change karein:
@st.cache_resource
def setup_resources():
    # ... baki code ...
    try:
        # Purane .h5 files ke liye compile=False zaroori hai
        model = keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        # Ek dummy model load karein ya error handle karein
        model = None
    return ai_model, scaler, model
                break
            except: continue
            
    scaler = joblib.load('my_scaler.joblib')
    model = keras.models.load_model('my_cnn_lstm_model_v4.h5', compile=False)
    return ai_mod, scaler, model

ai_model, scaler, model = setup_resources()

# ─────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">🫀 HEALTH AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b;'>Cardiovascular Risk Intelligence System</p>", unsafe_allow_html=True)

tabs = st.tabs(["🤖 Assistant", "🎙️ Voice Scan", "📋 Manual Scan"])

# --- TAB 1: ASSISTANT ---
with tabs[0]:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for m in st.session_state.messages:
        msg_type = "user-msg" if m["role"] == "user" else "ai-msg"
        st.markdown(f'<div class="chat-box {msg_type}"><b>{m["role"].upper()}:</b><br>{m["content"]}</div>', unsafe_allow_html=True)

    # Input Section
    with st.container():
        v_col, t_col = st.columns([1, 3])
        with v_col:
            v_input = speech_to_text(language='en', start_prompt="🎙️ Speak", key='voice_main')
        with t_col:
            t_input = st.text_input("Describe symptoms...", key="text_main", label_visibility="collapsed")

        final_query = v_input if v_input else t_input

        if st.button("🔍 ANALYZE SYMPTOMS", use_container_width=True) and final_query:
            # 1. Add User Msg
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            # 2. Get AI Response
            if ai_model:
                with st.spinner("AI is thinking..."):
                    try:
                        res = ai_model.generate_content(f"Analyze symptoms: {final_query}. Short medical advice only.")
                        st.session_state.messages.append({"role": "assistant", "content": res.text})
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")
            else:
                st.warning("AI Key not detected in Secrets.")
            
            st.rerun() # Refresh to show new messages

# --- TAB 2: VOICE SCAN ---
with tabs[1]:
    st.subheader("Speak your health metrics")
    v_data = speech_to_text(language='en', start_prompt="🎤 Record Data", key='voice_cvd')
    if v_data:
        st.info(f"Captured: {v_data}")
        if ai_model:
            with st.spinner("Extracting features..."):
                try:
                    p = f"Extract exactly 15 numbers as a list from: {v_data}. List only."
                    res = ai_model.generate_content(p)
                    # Force extract list from potential text
                    nums = [float(s) for s in res.text.replace('[','').replace(']','').split(',') if s.strip()]
                    if len(nums) == 15:
                        s_data = scaler.transform(np.array(nums).reshape(1, -1))
                        prob = model.predict(np.expand_dims(s_data, axis=2))[0][0]
                        color = "red" if prob > 0.5 else "green"
                        st.markdown(f"<h1 style='color:{color}; text-align:center;'>Risk: {prob*100:.1f}%</h1>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Could not parse voice data. Try manual input.")

# --- TAB 3: MANUAL SCAN ---
with tabs[2]:
    with st.form("manual"):
        cols = st.columns(3)
        # Simplified form for testing
        f1 = cols[0].number_input("Age", 18, 100, 40)
        f2 = cols[1].number_input("Cholesterol", 100, 400, 200)
        f3 = cols[2].number_input("Systolic BP", 80, 200, 120)
        # [Note: Add all 15 inputs here similar to your original code]
        sub = st.form_submit_button("Run Analysis")
        if sub:
            st.success("Analysis complete (Add all 15 fields for full prediction)")

st.markdown("<br><hr><center>HealthAI 2026</center>", unsafe_allow_html=True)
