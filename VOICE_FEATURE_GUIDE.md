# 🎤 Voice-Based CVD Risk Predictor

## Overview

This enhanced website allows **uneducated or illiterate users** to interact completely via **voice**:

- 🎤 **Voice Input** - Users speak their health information
- 🔊 **Voice Output** - System responds in voice
- 🌍 **Multi-Language** - English & Hindi support
- ♿ **Accessible** - No reading/typing required!

---

## 🚀 How It Works

### **User Flow:**

1. User visits website
2. Clicks **"🎤 Voice Input"** button
3. System says: *"Welcome to CVD Risk Predictor. Please speak your health information."*
4. System asks: *"What is your age?"*
5. **User speaks**: "Forty five"
6. System listens & understands
7. System asks next question
8. After all questions, system calculates risk
9. System **speaks the result**: *"Your 10-year cardiovascular risk is 15 percent. Your cardiovascular risk is low."*

### **No typing needed!** ✅

---

## 📋 Questions Asked (in Voice)

The system asks these questions in sequence:

1. **Age** - "What is your age?"
2. **Gender** - "Are you male or female?"
3. **Blood Pressure** - "What is your blood pressure?"
4. **Cholesterol** - "What is your cholesterol level?"
5. **BMI** - "What is your BMI?"
6. **Glucose** - "What is your glucose level?"
7. **Heart Rate** - "What is your heart rate?"
8. **Cigarettes** - "How many cigarettes per day?"
9. **Medical History** - Questions about stroke, hypertension, diabetes

Total time: **3-5 minutes**

---

## 🌐 Supported Languages

### **English (en-US)**
- Full voice recognition & synthesis
- Clear British English accent

### **Hindi (hi-IN)**
- Full voice recognition in Hindi
- Messages in Devanagari script
- Native Hindi speakers can use completely in Hindi

```
Example:
System: "आपकी उम्र क्या है?" (What is your age?)
User: "पैंतालीस" (Forty five)
System: "धन्यवाद, अगला प्रश्न..." (Thank you, next question...)
```

---

## 🎯 Use Cases

### **Target Users:**

1. **Illiterate farmers** - Can't read, but can speak
2. **Elderly people** - Prefer voice over typing
3. **Busy professionals** - Hands-free health check
4. **Rural areas** - No internet literacy required
5. **Accessibility** - Blind or visually impaired users

---

## 🔧 Technical Details

### **Browser Support**

Works on:
- ✅ Chrome/Chromium (best support)
- ✅ Edge
- ✅ Safari (macOS & iOS)
- ✅ Firefox (limited)

⚠️ **Requirement**: Modern browser with Web Speech API support

### **Web APIs Used**

1. **Web Speech API** (Recognition)
   - `SpeechRecognition` - Converts speech to text
   - Real-time listening
   - Auto-stop after 5 seconds of silence

2. **Web Speech API** (Synthesis)
   - `SpeechSynthesis` - Converts text to speech
   - Multiple voices & languages
   - Adjustable speed (0.9x for clarity)

3. **Noise Cancellation**
   - Browser handles background noise
   - Works in moderately noisy environments

---

## 📝 Code Structure

### **Files**

```
frontend/
├── voice.js              # Voice functionality
├── app.js               # Main application
├── index.html           # UI with voice buttons
└── styles.css           # Voice UI styling
```

### **Key Functions in voice.js**

```javascript
// Start listening to user
startListening(callback, language, timeout)

// Speak message to user
speak(text, language)

// Main voice-based prediction
voiceBasedPrediction()

// Capture inputs
captureNumberInput(lang)
captureGenderInput(lang)
captureBPInput(lang)
captureYesNoInput(lang)
```

### **Voice Messages**

Defined in `voiceMessages` object:
- English messages
- Hindi messages
- Customizable prompts

---

## 🎮 User Interface

### **Voice Mode Button**

```
[📝 Type Form] [🎤 Voice Input] ← Switch between modes
```

### **Language Selection**

```
Select Language:
- English 🇬🇧
- हिंदी (Hindi) 🇮🇳
```

### **Voice Assessment Button**

```
🎤 Start Voice Assessment
```

Transforms to: `🎤 Listening...` (when listening)

### **Voice Status Indicator**

Shows:
- "Listening..." with pulsing effect
- Real-time feedback

---

## 🔊 Audio Feedback

### **System Speaks:**

1. **Welcome Message** - Greeting
2. **Each Question** - Clear question about health
3. **Confirmation** - Acknowledges input
4. **Results** - Risk level and recommendations
5. **Goodbye** - Thank you message

### **Speech Settings**

- **Rate**: 0.9x (slower for clarity)
- **Pitch**: Normal (1.0)
- **Volume**: Maximum (1.0)
- **Language**: Auto-detect from user choice

---

## ✅ Testing the Feature

### **Step 1: Start Both Servers**

```bash
# Terminal 1 - Backend
python -m backend.app

# Terminal 2 - Frontend
cd frontend
python -m http.server 8000
```

### **Step 2: Open Website**

```
http://localhost:8000
```

### **Step 3: Register Account**

Create account with any credentials

### **Step 4: Go to Prediction Page**

Click **"Predict"** in menu

### **Step 5: Choose Voice Mode**

Click **"🎤 Voice Input"** button

### **Step 6: Select Language**

Choose English or Hindi

### **Step 7: Start Assessment**

Click **"🎤 Start Voice Assessment"**

### **Step 8: Follow Instructions**

- Listen to question
- Speak your answer clearly
- System continues

### **Step 9: View Results**

- Risk percentage displayed
- Recommendations in voice
- Can export as PDF/CSV

---

## 🎓 Example Conversation

### **English Example:**

```
System: "Welcome to the CVD Risk Predictor. Please speak your health information."
[Pause 1 second]

System: "What is your age?"
User: [Speaking] "I am forty five years old"
System: [Processing] → Next question

System: "Are you male or female?"
User: [Speaking] "I am male"
System: [Processing] → Next question

System: "What is your blood pressure? Say the number. For example, one thirty over eighty."
User: [Speaking] "One twenty over eighty"
System: [Processing] → Next question

[... continues for all questions ...]

System: "Your 10-year cardiovascular risk is 12 percent. Your cardiovascular risk is low. Keep maintaining your healthy lifestyle. Thank you for using CVD Risk Predictor. Stay healthy!"
```

### **Hindi Example:**

```
System: "CVD रिस्क प्रेडिक्टर में आपका स्वागत है। कृपया अपनी स्वास्थ्य जानकारी बताएं।"

System: "आपकी उम्र क्या है?"
User: [Speaking] "मेरी उम्र पैंतालीस साल है"
System: [Processing]

System: "क्या आप पुरुष या महिला हैं?"
User: [Speaking] "मैं पुरुष हूँ"
System: [Processing]

[... continues in Hindi ...]
```

---

## 🐛 Troubleshooting

### **"Speech recognition not supported"**
- ✅ Use Chrome/Edge browser
- ✅ Update your browser
- ✅ Allow microphone access

### **"No speech detected"**
- ✅ Speak clearly and slowly
- ✅ Check microphone is working
- ✅ Reduce background noise
- ✅ Speak into microphone

### **System not understanding**
- ✅ Speak number clearly (e.g., "one thirty" not "130")
- ✅ Use yes/no clearly
- ✅ Reduce accent if possible
- ✅ Try again with clearer speech

### **No sound output**
- ✅ Check system volume
- ✅ Check browser volume
- ✅ Enable speakers/headphones
- ✅ Refresh page

### **Microphone permission denied**
- ✅ Check browser permissions
- ✅ Allow microphone for this site
- ✅ Refresh page

---

## 🔒 Privacy & Security

### **No Data Stored**
- Speech input is processed locally
- Not recorded or stored
- Only health metrics sent to server
- Encrypted HTTPS connection

### **GDPR Compliant**
- No audio files created
- No tracking cookies
- User data protected

---

## 📱 Mobile Support

### **Works on:**
- ✅ Android phones (Chrome)
- ✅ iPhones (Safari)
- ✅ Tablets
- ✅ Desktops

### **Microphone Permission:**

iOS:
```
Settings → Safari → Microphone → Allow
```

Android:
```
Settings → Apps → Chrome → Permissions → Microphone
```

---

## 🚀 Future Enhancements

Possible improvements:

1. **Doctor Integration** - Share results with doctor
2. **SMS Results** - Send results via SMS
3. **WhatsApp Integration** - Get results on WhatsApp
4. **Appointment Booking** - Book doctor appointment
5. **More Languages** - Add more regional languages
6. **Offline Mode** - Work without internet
7. **Custom Voice** - Different voice accents
8. **Video Guidance** - Video instructions

---

## 📞 Support

For issues:

1. Check browser console (F12) for errors
2. Make sure microphone is working
3. Try different browser
4. Clear cache and refresh
5. Check internet connection

---

## 📄 License

This voice feature is part of the CVD Risk Predictor project.

---

**Help uneducated people get healthcare! 🏥❤️**
