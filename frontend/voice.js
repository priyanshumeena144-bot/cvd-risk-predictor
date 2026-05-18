// ===== VOICE RECOGNITION & SYNTHESIS =====

// Check browser support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const speechSynthesis = window.speechSynthesis;

let isListening = false;
let recognitionInstance = null;
let currentLanguage = 'en-US';

// Voice response messages
const voiceMessages = {
    en: {
        welcome: "Welcome to the CVD Risk Predictor. Please speak your health information.",
        askAge: "What is your age?",
        askGender: "Are you male or female?",
        askBP: "What is your blood pressure? Say the number. For example, one thirty over eighty.",
        askCholesterol: "What is your cholesterol level?",
        askBMI: "What is your BMI or body mass index?",
        askGlucose: "What is your glucose level?",
        askHeartRate: "What is your heart rate in beats per minute?",
        askCigarettes: "How many cigarettes do you smoke per day?",
        askStroke: "Do you have a history of stroke? Say yes or no.",
        askHypertension: "Do you have hypertension or high blood pressure? Say yes or no.",
        askDiabetes: "Do you have diabetes? Say yes or no.",
        processingRisk: "Processing your health information. Please wait.",
        riskLow: "Your cardiovascular risk is low. Keep maintaining your healthy lifestyle.",
        riskMedium: "Your cardiovascular risk is medium. Please consult a doctor for advice.",
        riskHigh: "Your cardiovascular risk is high. Please see a doctor as soon as possible.",
        thankYou: "Thank you for using CVD Risk Predictor. Stay healthy!"
    },
    hi: {
        welcome: "CVD रिस्क प्रेडिक्टर में आपका स्वागत है। कृपया अपनी स्वास्थ्य जानकारी बताएं।",
        askAge: "आपकी उम्र क्या है?",
        askGender: "क्या आप पुरुष या महिला हैं?",
        askBP: "आपका रक्तचाप क्या है? नंबर बताएं। उदाहरण के लिए, एक सौ तीस बाई अस्सी।",
        askCholesterol: "आपका कोलेस्ट्रॉल स्तर क्या है?",
        askBMI: "आपका बीएमआई क्या है?",
        askGlucose: "आपका ग्लूकोज स्तर क्या है?",
        askHeartRate: "आपकी हृदय गति क्या है?",
        askCigarettes: "आप प्रतिदिन कितनी सिगरेट पीते हैं?",
        askStroke: "क्या आपको पहले स्ट्रोक हुआ है? हाँ या नहीं कहें।",
        askHypertension: "क्या आपको उच्च रक्तचाप है? हाँ या नहीं कहें।",
        askDiabetes: "क्या आपको मधुमेह है? हाँ या नहीं कहें।",
        processingRisk: "आपकी स्वास्थ्य जानकारी प्रोसेस की जा रही है। कृपया प्रतीक्षा करें।",
        riskLow: "आपका हृदय रोग जोखिम कम है। अपनी स्वस्थ जीवनशैली बनाए रखें।",
        riskMedium: "आपका हृदय रोग जोखिम मध्यम है। कृपया डॉक्टर से सलाह लें।",
        riskHigh: "आपका हृदय रोग जोखिम अधिक है। कृपया जल्द से जल्द डॉक्टर से मिलें।",
        thankYou: "CVD रिस्क प्रेडिक्टर का उपयोग करने के लिए धन्यवाद। स्वस्थ रहें!"
    }
};

// ===== TEXT TO SPEECH =====
function speak(text, language = 'en') {
    return new Promise((resolve) => {
        // Cancel any ongoing speech
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        
        // Set language
        if (language === 'hi') {
            utterance.lang = 'hi-IN';
        } else {
            utterance.lang = 'en-US';
        }

        utterance.rate = 0.9; // Slower speech for clarity
        utterance.pitch = 1;
        utterance.volume = 1;

        utterance.onend = () => {
            resolve();
        };

        speechSynthesis.speak(utterance);
    });
}

// ===== SPEECH RECOGNITION =====
function startListening(callback, language = 'en-US', timeout = 5000) {
    return new Promise((resolve, reject) => {
        if (!SpeechRecognition) {
            showNotification('Speech recognition not supported in your browser', 'error');
            reject('Speech Recognition not supported');
            return;
        }

        recognitionInstance = new SpeechRecognition();
        recognitionInstance.lang = language;
        recognitionInstance.continuous = false;
        recognitionInstance.interimResults = true;

        let finalTranscript = '';
        let isListeningNow = true;

        recognitionInstance.onstart = () => {
            isListeningNow = true;
            updateListeningUI(true);
        };

        recognitionInstance.onresult = (event) => {
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;

                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }

            // Update UI with interim results
            if (interimTranscript) {
                console.log('Interim:', interimTranscript);
            }
        };

        recognitionInstance.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            updateListeningUI(false);
            reject(event.error);
        };

        recognitionInstance.onend = () => {
            isListeningNow = false;
            updateListeningUI(false);
            
            if (finalTranscript.trim()) {
                callback(finalTranscript.trim());
                resolve(finalTranscript.trim());
            } else {
                reject('No speech detected');
            }
        };

        recognitionInstance.start();

        // Auto-stop after timeout
        setTimeout(() => {
            if (isListeningNow) {
                recognitionInstance.stop();
            }
        }, timeout);
    });
}

function stopListening() {
    if (recognitionInstance) {
        recognitionInstance.stop();
        updateListeningUI(false);
    }
}

function updateListeningUI(listening) {
    const micButton = document.getElementById('micButton');
    if (micButton) {
        if (listening) {
            micButton.classList.add('listening');
            micButton.textContent = '🎤 Listening...';
        } else {
            micButton.classList.remove('listening');
            micButton.textContent = '🎤 Speak';
        }
    }
}

// ===== VOICE-BASED FORM =====
async function voiceBasedPrediction() {
    const selectedLanguage = document.getElementById('voiceLanguage')?.value || 'en';
    const lang = selectedLanguage === 'hi' ? 'hi-IN' : 'en-US';
    const msgLang = selectedLanguage;

    const healthData = {};
    let continueProcessing = true;

    try {
        // Welcome message
        await speak(voiceMessages[msgLang].welcome, msgLang);
        await new Promise(r => setTimeout(r, 1000));

        // Age
        await speak(voiceMessages[msgLang].askAge, msgLang);
        const ageResult = await captureNumberInput(lang);
        healthData.age = parseInt(ageResult);

        // Gender
        await speak(voiceMessages[msgLang].askGender, msgLang);
        const genderResult = await captureGenderInput(lang);
        healthData.gender = genderResult;

        // Blood Pressure
        await speak(voiceMessages[msgLang].askBP, msgLang);
        const bpResult = await captureBPInput(lang);
        healthData.systolic_bp = bpResult.systolic;
        healthData.diastolic_bp = bpResult.diastolic;

        // Cholesterol
        await speak(voiceMessages[msgLang].askCholesterol, msgLang);
        const cholesterolResult = await captureNumberInput(lang);
        healthData.cholesterol = parseInt(cholesterolResult);

        // BMI
        await speak(voiceMessages[msgLang].askBMI, msgLang);
        const bmiResult = await captureNumberInput(lang);
        healthData.bmi = parseFloat(bmiResult);

        // Glucose
        await speak(voiceMessages[msgLang].askGlucose, msgLang);
        const glucoseResult = await captureNumberInput(lang);
        healthData.glucose = parseInt(glucoseResult);

        // Heart Rate
        await speak(voiceMessages[msgLang].askHeartRate, msgLang);
        const hrResult = await captureNumberInput(lang);
        healthData.heart_rate = parseInt(hrResult);

        // Cigarettes per day
        await speak(voiceMessages[msgLang].askCigarettes, msgLang);
        const smokeResult = await captureNumberInput(lang);
        healthData.cigarettes_per_day = parseInt(smokeResult) || 0;

        // Medical history
        await speak(voiceMessages[msgLang].askStroke, msgLang);
        healthData.stroke = await captureYesNoInput(lang);

        await speak(voiceMessages[msgLang].askHypertension, msgLang);
        healthData.hypertension = await captureYesNoInput(lang);

        await speak(voiceMessages[msgLang].askDiabetes, msgLang);
        healthData.diabetes = await captureYesNoInput(lang);

        healthData.current_smoker = healthData.cigarettes_per_day > 0;

        // Process prediction
        await speak(voiceMessages[msgLang].processingRisk, msgLang);

        // Make API call
        const response = await fetch(`${API_URL}/predictions/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${currentToken}`
            },
            body: JSON.stringify(healthData)
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        const prediction = data.prediction;

        // Voice response based on risk
        let riskMessage = '';
        if (prediction.risk_category === 'low') {
            riskMessage = voiceMessages[msgLang].riskLow;
        } else if (prediction.risk_category === 'medium') {
            riskMessage = voiceMessages[msgLang].riskMedium;
        } else {
            riskMessage = voiceMessages[msgLang].riskHigh;
        }

        const riskPercentage = prediction.risk_percentage.toFixed(1);
        const fullMessage = `Your 10-year cardiovascular risk is ${riskPercentage} percent. ${riskMessage}`;

        await speak(fullMessage, msgLang);
        await speak(voiceMessages[msgLang].thankYou, msgLang);

        // Display results
        displayResults(prediction);
        showNotification('Prediction completed! Check results below.', 'success');

    } catch (error) {
        console.error('Voice prediction error:', error);
        showNotification('Error: ' + error.message, 'error');
        await speak('Sorry, an error occurred. Please try again.', msgLang);
    }
}

// ===== INPUT CAPTURE FUNCTIONS =====
async function captureNumberInput(lang) {
    return new Promise(async (resolve) => {
        try {
            const result = await startListening((text) => {
                // Extract numbers from speech
                const numbers = text.match(/\d+/);
                if (numbers) {
                    resolve(numbers[0]);
                } else {
                    resolve('0');
                }
            }, lang, 4000);
        } catch (error) {
            resolve('0');
        }
    });
}

async function captureGenderInput(lang) {
    return new Promise(async (resolve) => {
        try {
            const result = await startListening((text) => {
                const lower = text.toLowerCase();
                if (lower.includes('male') || lower.includes('m')) {
                    resolve('male');
                } else if (lower.includes('female') || lower.includes('f') || lower.includes('woman')) {
                    resolve('female');
                } else {
                    resolve('male');
                }
            }, lang, 4000);
        } catch (error) {
            resolve('male');
        }
    });
}

async function captureBPInput(lang) {
    return new Promise(async (resolve) => {
        try {
            const result = await startListening((text) => {
                const numbers = text.match(/\d+/g);
                if (numbers && numbers.length >= 2) {
                    resolve({
                        systolic: parseInt(numbers[0]),
                        diastolic: parseInt(numbers[1])
                    });
                } else if (numbers && numbers.length === 1) {
                    resolve({
                        systolic: parseInt(numbers[0]),
                        diastolic: parseInt(numbers[0]) - 40
                    });
                } else {
                    resolve({ systolic: 120, diastolic: 80 });
                }
            }, lang, 4000);
        } catch (error) {
            resolve({ systolic: 120, diastolic: 80 });
        }
    });
}

async function captureYesNoInput(lang) {
    return new Promise(async (resolve) => {
        try {
            const result = await startListening((text) => {
                const lower = text.toLowerCase();
                resolve(lower.includes('yes') || lower.includes('ha') || lower.includes('haan'));
            }, lang, 4000);
        } catch (error) {
            resolve(false);
        }
    });
}

// ===== VOICE BUTTON CLICK HANDLER =====
document.addEventListener('DOMContentLoaded', () => {
    const voiceButton = document.getElementById('voiceButton');
    if (voiceButton) {
        voiceButton.addEventListener('click', voiceBasedPrediction);
    }
});
