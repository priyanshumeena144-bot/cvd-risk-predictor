# Cardiovascular Risk Predictor and Symptom Checker

A Streamlit web app for educational health screening. Users can enter symptoms to get possible condition suggestions and doctor consultation advice. The app also includes a cardiovascular disease risk predictor using a CNN-LSTM model.

## Features

- Symptom checker with possible disease suggestions
- Doctor consultation guidance for every result
- Emergency warning for serious symptoms such as chest pain or shortness of breath
- Manual cardiovascular risk prediction form
- Voice input support when optional packages/API keys are configured
- Camera input placeholder for future medical report/photo features
- Health dashboard with targets, tips, and trusted health resources

## Important Disclaimer

This project is for educational purposes only. It does not provide a confirmed medical diagnosis and must not replace professional medical advice. Users should consult a qualified doctor before making health decisions or taking medicine.

## Tech Stack

- Python
- Streamlit
- TensorFlow/Keras
- NumPy
- Scikit-learn
- Joblib
- Google Gemini API, optional

## Project Files

```text
app.py                         Main Streamlit app
requirements.txt               Python dependencies
my_cnn_lstm_model.keras        Trained CVD prediction model
my_scaler.joblib               Feature scaler
.streamlit/config.toml         Streamlit theme/config
run_app.bat                    Windows local launcher
PUBLIC_DEPLOYMENT_STEPS.md     Public deployment guide
```

## Run Locally

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

On Windows, you can also double-click:

```text
run_app.bat
```

Keep the terminal window open while using the website.

## Public Deployment

To make the website available for everyone, deploy it on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io
3. Click New app.
4. Select this repository.
5. Set the main file path to:

```text
app.py
```

6. Click Deploy.

Streamlit will create a public link like:

```text
https://your-app-name.streamlit.app
```

Share that link with your teacher or anyone else.

## Optional Gemini Setup

The app works without Gemini. If you want voice AI extraction, add this secret in Streamlit Cloud or `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Without this key, the Symptom Checker and Manual Form still work normally.

## Symptom Checker Examples

Try symptoms like:

```text
fever, cough, body pain, headache
```

or:

```text
chest pain, shortness of breath, sweating
```

The app will show possible conditions and advise whether the user should consult a doctor or seek urgent care.

## GitHub Repository

```text
https://github.com/priyanshumeena144-bot/cvd-risk-predictor
```
