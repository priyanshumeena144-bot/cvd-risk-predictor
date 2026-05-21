import streamlit as st
import joblib
import numpy as np
import json
import os
from datetime import datetime
from tensorflow import keras
try:
    from streamlit_mic_recorder import speech_to_text
except ModuleNotFoundError:
    speech_to_text = None
import google.generativeai as genai

# ─────────────────────────────────────────
# PAGE CONFIG & CUSTOM STYLING
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #e74c3c;
        --secondary-color: #3498db;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #c0392b;
    }
    
    /* Enhanced styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #e74c3c;
    }
    
    .recommendation-card {
        background: #ecf0f1;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    .feature-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 1. LOAD MODEL & SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler = joblib.load('my_scaler.joblib')
    model = keras.models.load_model('my_cnn_lstm_model.keras')
    return scaler, model

scaler, model = load_assets()

# ─────────────────────────────────────────
# 2. GEMINI AI SETUP
# ─────────────────────────────────────────
gemini_api_key = os.getenv("GEMINI_API_KEY")
try:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", gemini_api_key)
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    pass

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    ai_model = genai.GenerativeModel('gemini-pro')
else:
    st.warning("Gemini API key is missing. Voice AI extraction is disabled, but the Symptom Checker and Manual Form still work.")
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

def show_result(prob, age, male, currentSmoker, BMI, sysBP, totChol, diabetes, prevalentHyp):
    """Displays the prediction result with health recommendations"""
    st.divider()
    
    # Risk Assessment
    if prob > 0.5:
        risk_level = "HIGH"
        risk_class = "risk-high"
        risk_emoji = "🔴"
    elif prob > 0.3:
        risk_level = "MEDIUM"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_class = "risk-low"
        risk_emoji = "🟢"
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h2>{risk_emoji} {risk_level} Risk of Cardiovascular Disease</h2>
        <h3 style="margin: 1rem 0;">Probability: {prob * 100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Health Recommendations
    st.subheader("💡 Personalized Health Recommendations")
    
    recommendations = get_recommendations(prob, age, male, currentSmoker, BMI, sysBP, totChol, diabetes, prevalentHyp)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-card">
            <b>{i}. {rec['title']}</b><br>
            {rec['description']}
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Factors Summary
    st.subheader("📊 Risk Factors Analysis")
    risk_factors = analyze_risk_factors(age, male, currentSmoker, BMI, sysBP, totChol, diabetes, prevalentHyp)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Primary Risk Factors:**")
        for factor in risk_factors["high"]:
            st.write(f"🔴 {factor}")
    
    with col2:
        st.write("**Moderate Risk Factors:**")
        for factor in risk_factors["moderate"]:
            st.write(f"🟡 {factor}")

def get_recommendations(prob, age, male, currentSmoker, BMI, sysBP, totChol, diabetes, prevalentHyp):
    """Generate personalized health recommendations"""
    recommendations = []
    
    if prob > 0.5:
        recommendations.append({
            "title": "Immediate Medical Consultation",
            "description": "Schedule an appointment with your cardiologist or primary care physician as soon as possible for a comprehensive cardiovascular evaluation."
        })
    
    if currentSmoker == 1:
        recommendations.append({
            "title": "Quit Smoking",
            "description": "Smoking is a major risk factor. Consider smoking cessation programs or consult with your doctor about nicotine replacement therapy."
        })
    
    if BMI > 30:
        recommendations.append({
            "title": "Weight Management",
            "description": f"Your BMI is {BMI:.1f} (overweight). Aim to reduce weight through balanced diet and regular exercise. Target BMI: 18.5-24.9"
        })
    elif BMI > 25:
        recommendations.append({
            "title": "Maintain Healthy Weight",
            "description": f"Your BMI is {BMI:.1f} (overweight). Reducing weight by 5-10% can significantly improve cardiovascular health."
        })
    
    if sysBP > 140:
        recommendations.append({
            "title": "Blood Pressure Management",
            "description": f"Your systolic BP is {sysBP:.1f} mmHg (high). Start or adjust blood pressure medication. Target: < 130/80 mmHg"
        })
    elif sysBP > 130:
        recommendations.append({
            "title": "Monitor Blood Pressure",
            "description": f"Your systolic BP is {sysBP:.1f} mmHg (elevated). Reduce sodium intake and manage stress."
        })
    
    if totChol > 240:
        recommendations.append({
            "title": "Cholesterol Control",
            "description": f"Your total cholesterol is {totChol} mg/dL (high). Consider statin therapy and dietary changes (reduce saturated fats)."
        })
    elif totChol > 200:
        recommendations.append({
            "title": "Improve Cholesterol Levels",
            "description": f"Your total cholesterol is {totChol} mg/dL. Increase fiber intake and reduce processed foods."
        })
    
    if diabetes == 1:
        recommendations.append({
            "title": "Diabetes Management",
            "description": "Maintain strict glycemic control. Regular exercise and medication adherence are critical for reducing CVD risk."
        })
    
    if prevalentHyp == 1:
        recommendations.append({
            "title": "Hypertension Control",
            "description": "Continue antihypertensive medication as prescribed. Monitor BP regularly at home."
        })
    
    recommendations.append({
        "title": "Regular Exercise",
        "description": "Aim for 150 minutes of moderate-intensity cardiovascular exercise per week (e.g., brisk walking, cycling, swimming)."
    })
    
    recommendations.append({
        "title": "Heart-Healthy Diet",
        "description": "Follow a Mediterranean or DASH diet. Increase fruits, vegetables, whole grains, and lean proteins. Limit salt and sugar."
    })
    
    recommendations.append({
        "title": "Regular Check-ups",
        "description": "Schedule annual health check-ups and blood pressure monitoring to track your cardiovascular health."
    })
    
    return recommendations[:5]  # Return top 5 recommendations

def analyze_risk_factors(age, male, currentSmoker, BMI, sysBP, totChol, diabetes, prevalentHyp):
    """Analyze and categorize risk factors"""
    high_risk = []
    moderate_risk = []
    
    if currentSmoker == 1:
        high_risk.append("Current Smoker")
    
    if diabetes == 1:
        high_risk.append("Diabetes")
    
    if BMI > 30:
        high_risk.append("Obesity (BMI > 30)")
    
    if sysBP > 160:
        high_risk.append("Stage 2 Hypertension (SysBP > 160)")
    
    if totChol > 240:
        high_risk.append("High Cholesterol (> 240)")
    
    if age > 60 and male == 1:
        moderate_risk.append("Age > 60 (Male)")
    elif age > 65 and male == 0:
        moderate_risk.append("Age > 65 (Female)")
    
    if BMI > 25:
        moderate_risk.append("Overweight (BMI > 25)")
    
    if sysBP > 130:
        moderate_risk.append("Elevated Blood Pressure")
    
    if totChol > 200:
        moderate_risk.append("Elevated Cholesterol")
    
    if prevalentHyp == 1:
        moderate_risk.append("History of Hypertension")
    
    return {"high": high_risk if high_risk else ["None identified"], "moderate": moderate_risk if moderate_risk else ["None identified"]}

SYMPTOM_RULES = [
    {
        "disease": "Common cold or upper respiratory infection",
        "symptoms": {"runny nose", "sneezing", "sore throat", "cough", "mild fever", "headache"},
        "doctor": "Consult a doctor if symptoms last more than 3 days, fever is high, breathing becomes difficult, or you have existing health problems.",
        "care": "Rest, drink fluids, use steam inhalation, and avoid close contact with others."
    },
    {
        "disease": "Influenza or viral fever",
        "symptoms": {"fever", "body pain", "fatigue", "cough", "headache", "chills"},
        "doctor": "Consult a doctor, especially if fever is high, symptoms are severe, or the patient is elderly, pregnant, diabetic, or has heart/lung disease.",
        "care": "Rest, hydrate well, monitor temperature, and avoid self-medicating with antibiotics."
    },
    {
        "disease": "Possible dengue or mosquito-borne fever",
        "symptoms": {"high fever", "severe headache", "joint pain", "rash", "vomiting", "bleeding gums"},
        "doctor": "Visit a doctor urgently for blood tests and proper monitoring.",
        "care": "Drink fluids and avoid aspirin or ibuprofen unless a doctor advises it."
    },
    {
        "disease": "Possible stomach infection or food poisoning",
        "symptoms": {"vomiting", "diarrhea", "stomach pain", "nausea", "fever", "weakness"},
        "doctor": "Consult a doctor if there is dehydration, blood in stool, persistent vomiting, severe pain, or symptoms last more than 24 hours.",
        "care": "Take oral rehydration solution, eat light food, and avoid oily or spicy meals."
    },
    {
        "disease": "Possible migraine or tension headache",
        "symptoms": {"headache", "nausea", "sensitivity to light", "dizziness", "blurred vision"},
        "doctor": "Consult a doctor if headache is sudden, severe, repeated, or comes with weakness, fainting, confusion, or vision changes.",
        "care": "Rest in a quiet room, hydrate, reduce screen time, and track headache triggers."
    },
    {
        "disease": "Possible diabetes-related symptoms",
        "symptoms": {"excessive thirst", "frequent urination", "fatigue", "blurred vision", "weight loss", "slow wound healing"},
        "doctor": "Book a doctor consultation and ask about blood sugar testing.",
        "care": "Avoid sugary drinks and monitor symptoms until evaluated."
    },
    {
        "disease": "Possible urinary tract infection",
        "symptoms": {"burning urination", "frequent urination", "lower abdominal pain", "fever", "back pain"},
        "doctor": "Consult a doctor because UTIs may need urine testing and medicines.",
        "care": "Drink water and do not delay care if fever or back pain is present."
    },
    {
        "disease": "Possible heart-related emergency",
        "symptoms": {"chest pain", "shortness of breath", "left arm pain", "sweating", "dizziness", "palpitations"},
        "doctor": "Seek emergency medical help immediately. Do not wait for an online prediction.",
        "care": "Stop physical activity, sit down, and call local emergency services or go to the nearest emergency department."
    },
]

EMERGENCY_SYMPTOMS = {
    "chest pain",
    "shortness of breath",
    "left arm pain",
    "fainting",
    "confusion",
    "severe bleeding",
    "severe allergic reaction",
    "difficulty speaking",
    "face drooping",
}

SYMPTOM_OPTIONS = sorted({
    symptom
    for rule in SYMPTOM_RULES
    for symptom in rule["symptoms"]
}.union(EMERGENCY_SYMPTOMS))

def predict_from_symptoms(selected_symptoms, free_text):
    """Return possible diseases using simple symptom matching."""
    normalized_text = free_text.lower()
    symptoms = set(selected_symptoms)
    symptoms.update(symptom for symptom in SYMPTOM_OPTIONS if symptom in normalized_text)

    matches = []
    for rule in SYMPTOM_RULES:
        matched = symptoms.intersection(rule["symptoms"])
        if matched:
            score = len(matched) / len(rule["symptoms"])
            matches.append({
                "disease": rule["disease"],
                "matched": sorted(matched),
                "score": score,
                "doctor": rule["doctor"],
                "care": rule["care"],
            })

    matches.sort(key=lambda item: (item["score"], len(item["matched"])), reverse=True)
    emergency = bool(symptoms.intersection(EMERGENCY_SYMPTOMS))
    return symptoms, matches[:3], emergency

# ─────────────────────────────────────────
# 3. MAIN UI
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>❤️ Cardiovascular Risk Predictor</h1>
    <p>Advanced AI-powered CVD risk assessment using CNN-LSTM neural networks</p>
</div>
""", unsafe_allow_html=True)

st.write("Check symptoms and predict your 10-year risk of Coronary Heart Disease using **Voice**, **Manual Form**, or **Camera**.")

# Sidebar Info
with st.sidebar:
    st.header("📋 About This App")
    st.info("""
    **Cardiovascular Risk Predictor** uses advanced machine learning to assess your CHD risk based on:
    - Personal health history
    - Medical measurements
    - Lifestyle factors
    
    **Disclaimer:** This app is for educational purposes only. Always consult a qualified physician for medical decisions.
    """)
    
    st.divider()
    st.subheader("📊 How It Works")
    st.markdown("""
    1. **Input Data** — Enter your health information
    2. **AI Processing** — CNN-LSTM model analyzes patterns
    3. **Risk Score** — Get your 10-year CHD probability
    4. **Recommendations** — Receive personalized health advice
    """)

# ─── TABS ───
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Symptom Checker", "🎙️ Voice Input", "📋 Manual Form", "📷 Camera", "📈 Health Dashboard"])

# ===========================================================================
# TAB 0: SYMPTOM CHECKER
# ===========================================================================
with tab0:
    st.header("Symptom Checker")
    st.warning(
        "This tool gives possible health conditions based on symptoms. "
        "It is not a confirmed diagnosis. Please consult a qualified doctor for proper medical advice."
    )

    col_symptoms, col_notes = st.columns([1, 1])
    with col_symptoms:
        selected_symptoms = st.multiselect(
            "Select your symptoms",
            options=SYMPTOM_OPTIONS,
            placeholder="Choose symptoms such as fever, cough, chest pain..."
        )

    with col_notes:
        symptom_text = st.text_area(
            "Or describe symptoms in your own words",
            placeholder="Example: I have fever, cough, body pain and headache since yesterday.",
            height=150
        )

    if st.button("Check Possible Disease", type="primary", use_container_width=True):
        detected_symptoms, disease_matches, emergency = predict_from_symptoms(selected_symptoms, symptom_text)

        if not detected_symptoms:
            st.info("Please select or type at least one symptom.")
        else:
            st.subheader("Result")

            if emergency:
                st.error(
                    "Emergency warning: some symptoms can be serious. "
                    "Please go to the nearest hospital or contact emergency medical services immediately."
                )

            if disease_matches:
                for index, match in enumerate(disease_matches, 1):
                    confidence = min(round(match["score"] * 100), 100)
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <b>{index}. Possible condition: {match['disease']}</b><br>
                        Matched symptoms: {", ".join(match['matched'])}<br>
                        Match strength: {confidence}%<br><br>
                        <b>Doctor advice:</b> {match['doctor']}<br>
                        <b>Basic care:</b> {match['care']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(
                    "The symptoms do not strongly match the current rule set. "
                    "Please consult a doctor for a proper check-up."
                )

            st.success("Recommendation: consult a qualified doctor before taking any medicine or making medical decisions.")

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

    if speech_to_text is None:
        st.warning(
            "Voice input is unavailable because the streamlit-mic-recorder package is not installed. "
            "Use the Symptom Checker or Manual Form tabs, or install it with: pip install streamlit-mic-recorder"
        )
        text = None
    else:
        text = speech_to_text(
            language='en',
            start_prompt=" Start Recording",
            stop_prompt=" Stop Recording",
            key='voice_input'
        )

    if text:
        st.success(f"**You said:** {text}")

        if ai_model is None:
            st.error("Gemini API key is missing!")
        else:
            try:
                with st.spinner(" AI is extracting features..."):
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
                    
                    # Extract values for recommendations
                    male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose = features
                    show_result(prob, int(age), int(male), int(currentSmoker), BMI, sysBP, int(totChol), int(diabetes), int(prevalentHyp))
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
                show_result(prob, int(age), int(male), int(currentSmoker), BMI, sysBP, int(totChol), int(diabetes), int(prevalentHyp))
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ═══════════════════════════════════════
# TAB 3: CAMERA
# ═══════════════════════════════════════
with tab3:
    st.header("📷 Camera Feature")
    col_cam, col_info = st.columns([1, 1])

    with col_cam:
        img = st.camera_input("Take a photo for future analysis")

    with col_info:
        if img:
            st.image(img, caption="📸 Photo captured successfully!", use_container_width=True)
            st.success(" Photo saved!")
        
        st.markdown("""
        ### 🔮 Planned Features:
        - **Medical Report Scan** — AI-powered OCR to extract data from medical reports
        - **Age Estimation** — Predict age from facial analysis
        - **Health Insights** — Analyze skin tone for potential health indicators
        
        *Coming in next update!*
        """)

# ═══════════════════════════════════════
# TAB 4: HEALTH DASHBOARD
# ═══════════════════════════════════════
with tab4:
    st.header("Health Dashboard & Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Systolic BP", "< 130 mmHg", "-10 mmHg")
        st.metric("Target Cholesterol", "< 200 mg/dL", "-20 mg/dL")
    
    with col2:
        st.metric("Ideal BMI", "18.5 - 24.9", "Current")
        st.metric("Target Heart Rate", "60-100 bpm", "Normal")
    
    with col3:
        st.metric("Exercise Target", "150 min/week", "Cardiovascular")
        st.metric("Sleep Target", "7-9 hours", "Per night")
    
    st.divider()
    
    st.subheader(" Health Resources")
    
    resources = {
        " American Heart Association": "https://www.heart.org/",
        " Mayo Clinic - Heart Health": "https://www.mayoclinic.org/diseases-conditions/heart-disease/",
        " CDC - Heart Disease Prevention": "https://www.cdc.gov/heartdisease/",
        " WHO - Cardiovascular Diseases": "https://www.who.int/health-topics/cardiovascular-diseases/",
    }
    
    for title, url in resources.items():
        st.write(f"[{title}]({url})")
    
    st.divider()
    
    st.subheader(" Quick Risk Reduction Tips")
    tips = [
        " **Quit Smoking** — Improves heart health within months",
        " **Stay Active** — 30 mins of moderate exercise daily",
        " **Eat Smart** — Mediterranean or DASH diet recommended",
        " **Sleep Well** — 7-9 hours improves cardiovascular health",
        " **Manage Stress** — Meditation and yoga reduce CVD risk",
        " **Control Weight** — Reduces strain on the heart",
        " **Limit Alcohol** — Moderate consumption only",
        " **Reduce Sodium** — Less than 2,300 mg per day"
    ]
    
    cols = st.columns(2)
    for i, tip in enumerate(tips):
        cols[i % 2].info(tip)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption(
    " **Disclaimer:** This app is for educational purposes only. "
    "Please consult a qualified doctor for any medical decisions."
)
