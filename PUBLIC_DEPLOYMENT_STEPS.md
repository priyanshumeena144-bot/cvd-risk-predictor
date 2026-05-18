# Public Deployment Steps

Use Streamlit Community Cloud to make this app public for your teacher.

## 1. Upload Project To GitHub

1. Open <https://github.com>
2. Create a new repository, for example `cvd-symptom-checker`
3. Upload these project files:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `my_scaler.joblib`
   - `my_cnn_lstm_model.keras`

Do not upload `.env` or any private API keys.

## 2. Deploy On Streamlit Cloud

1. Open <https://share.streamlit.io>
2. Sign in with GitHub
3. Click `New app`
4. Select your repository
5. Set the main file path to:

```text
app.py
```

6. Click `Deploy`

After deployment, Streamlit will give you a public link like:

```text
https://your-app-name.streamlit.app
```

That link can be opened by anyone.

## 3. If Voice/Gemini Is Needed

The app works without Gemini. If you want Gemini voice extraction, add this secret in Streamlit Cloud:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

For your current project submission, the Symptom Checker and Manual Form can run without this key.
