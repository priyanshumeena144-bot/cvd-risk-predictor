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
    page_title="CVD Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ─────────────────────────────────────────
# 1. LOAD MODEL & SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler = joblib.load('my_scaler.joblib')
    model = keras.models.load_model(
        'my_cnn_lstm_model_v4.h5',
        compile=False
    )
    return scaler, model

scaler, model = load_assets()

# ─────────────────────────────────────────
# 2. GEMINI AI SETUP
# ─────────────────────────────────────────
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-pro')
else:
    st.error("⚠️ Please add GEMINI_API_KEY in Streamlit Secrets!")
    ai_model = None

def extract_features_with_ai(user_text):
    """Extracts 15 numerical features from voice text using Gemini AI"""
    prompt = f"""
    Analyze this medical data: "{user_text}".
    Extract exactly 15 numerical features in this order for CVD prediction:
    male(0/1), age, education(1-4), currentSmoker(0/1), cigsPerDay,
    BPMeds(0/1), prevalentStroke(0/1), prevalentHyp(0/1), diabetes(0/1),
    totChol, sysBP, diaBP, BMI, heartRate, glucose

    Return ONLY a Python list of 15 numbers. No explanation.
    Example: [1, 50, 2, 0, 0, 0, 0, 1, 0, 200, 130, 80, 28.5, 75, 90]
    """
    response = ai_model.generate_content(prompt)
    text = response.text.strip().replace("```", "").replace("python", "").strip()
    return json.loads(text)

def make_prediction(features_list):
    """Predicts CVD risk probability from input features"""
    features_np = np.array(features_list).reshape(1, -1)
    scaled = scaler.transform(features_np)
    reshaped = np.expand_dims(scaled, axis=2)
    prob = float(model.predict(reshaped)[0][0])
    return prob

def show_result(prob):
    """Displays the prediction result to the user"""
    st.divider()
    st.subheader("📊 Prediction Result")
    st.metric(label="10-Year CHD Risk Probability", value=f"{prob * 100:.2f}%")
    if prob > 0.5:
        st.error("🔴 HIGH Risk of Cardiovascular Disease detected!")
        st.warning("Please consult a cardiologist immediately.")
    else:
        st.success("🟢 LOW Risk of Cardiovascular Disease.")
        st.info("Keep maintaining a healthy lifestyle!")

# ─────────────────────────────────────────
# 3. MAIN UI
# ─────────────────────────────────────────
st.title("❤️ Cardiovascular Risk Predictor")
st.write("Check your CVD risk using **Voice**, **Manual Form**, or **Camera**.")

# ─── TABS ───
tab1, tab2, tab3 = st.tabs(["🎙️ Voice Input", "📋 Manual Form", "📷 Camera (Coming Soon)"])

# ═══════════════════════════════════════
# TAB 1: VOICE INPUT
# ═══════════════════════════════════════
with tab1:
    st.header("🎙️ Predict Using Voice")
    st.info(
        "Press the mic button and speak your health details.\n\n"
        "**Example:** 'Male, age 45, BMI 28, BP 130 over 80, "
        "smoker, cholesterol 210, glucose 90, heart rate 75'"
    )

    text = speech_to_text(
        language='en',
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='voice_input'
    )

    if text:
        st.success(f"**You said:** {text}")

        if ai_model is None:
            st.error("Gemini API key is missing!")
        else:
            try:
                with st.spinner("🤖 AI is extracting features..."):
                    features = extract_features_with_ai(text)

                if len(features) == 15:
                    # Show extracted features
                    labels = [
                        "Gender(M=1)", "Age", "Education", "Smoker", "Cigs/Day",
                        "BP Meds", "Stroke Hx", "Hypertension", "Diabetes",
                        "Cholesterol", "SysBP", "DiaBP", "BMI", "Heart Rate", "Glucose"
                    ]
                    with st.expander("🔍 Features extracted by AI"):
                        cols = st.columns(5)
                        for i, (label, val) in enumerate(zip(labels, features)):
                            cols[i % 5].metric(label, val)

                    with st.spinner("📈 Running prediction..."):
                        prob = make_prediction(features)
                    show_result(prob)
                else:
                    st.warning(f"AI extracted {len(features)} features, but 15 are required. Please try again.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Tip: Speak clearly — mention age, BMI, BP, cholesterol, etc.")

# ═══════════════════════════════════════
# TAB 2: MANUAL FORM
# ═══════════════════════════════════════
with tab2:
    st.header("📋 Predict Using Manual Form")

    st.subheader("👤 Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        male = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.number_input("Education (1–4)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

    with col2:
        currentSmoker = st.selectbox("Current Smoker?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
        BPMeds = st.selectbox("BP Medication?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        prevalentStroke = st.selectbox("History of Stroke?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prevalentHyp = st.selectbox("History of Hypertension?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("History of Diabetes?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.divider()
    st.subheader("🩺 Medical Vitals")
    col4, col5, col6 = st.columns(3)

    with col4:
        totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        sysBP = st.number_input("Systolic BP (sysBP)", min_value=80.0, max_value=300.0, value=120.0, step=0.1)
        diaBP = st.number_input("Diastolic BP (diaBP)", min_value=50.0, max_value=200.0, value=80.0, step=0.1)

    with col5:
        BMI = st.number_input("BMI", min_value=15.0, max_value=60.0, value=25.0, step=0.1)
        heartRate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=400, value=80)

    if st.button("🔍 Predict My Risk", type="primary", use_container_width=True):
        if model is None or scaler is None:
            st.error("Model failed to load. Please check the server logs.")
        else:
            features_list = [
                male, age, education, currentSmoker, cigsPerDay, BPMeds,
                prevalentStroke, prevalentHyp, diabetes, totChol,
                sysBP, diaBP, BMI, heartRate, glucose
            ]
            try:
                with st.spinner(" Running prediction..."):
                    prob = make_prediction(features_list)
                show_result(prob)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ═══════════════════════════════════════
# TAB 3: CAMERA (FUTURE USE)
# ═══════════════════════════════════════
with tab3:
    st.header("📷 Camera Feature")
    st.info("📸 This feature is currently under development. Coming soon!")

    col_cam, col_info = st.columns([1, 1])

    with col_cam:
        img = st.camera_input("Take a photo of the patient (for future analysis)")

    with col_info:
        if img:
            st.image(img, caption="📸 Photo captured successfully!", use_container_width=True)
            st.success("✅ Photo saved!")
            st.info(
                "🔮 **Upcoming Features:**\n"
                "- AI-based age estimation from face\n"
                "- Anaemia detection from skin tone\n"
                "- Medical report scan with auto-fill"
            )
        else:
            st.markdown("""
            ### 🔮 Upcoming Features:
            - **Medical Report Scan** — take a photo of your report and fields will auto-fill
            - **AI Image Analysis** — detect health indicators from photo
            - **Face Age Estimation** — predict approximate age from face

            *For now, please use Voice Input or Manual Form.*
            """)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This app is for educational purposes only. "
    "Please consult a qualified doctor for any medical decisions."
)
