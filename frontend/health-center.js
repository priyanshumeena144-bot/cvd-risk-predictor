// ===== SIMPLIFIED VOICE FOR HEALTH CENTERS =====

// Simple helper to capture number input
async function captureSimpleNumber(lang) {
    return new Promise((resolve) => {
        const langCode = lang === 'hi-IN' ? 'hi-IN' : 'en-US';
        startListening(
            (transcript) => {
                const numbers = transcript.match(/\d+/g);
                resolve(numbers ? numbers[0] : '0');
            },
            langCode,
            5000
        ).catch(() => resolve('0'));
    });
}

// Simple helper for BP input (two numbers)
async function captureSimpleBP(lang) {
    return new Promise((resolve) => {
        const langCode = lang === 'hi-IN' ? 'hi-IN' : 'en-US';
        startListening(
            (transcript) => {
                const numbers = transcript.match(/\d+/g);
                if (numbers && numbers.length >= 2) {
                    resolve({ systolic: parseInt(numbers[0]), diastolic: parseInt(numbers[1]) });
                } else if (numbers && numbers.length === 1) {
                    resolve({ systolic: parseInt(numbers[0]), diastolic: 85 });
                } else {
                    resolve({ systolic: 120, diastolic: 80 });
                }
            },
            langCode,
            5000
        ).catch(() => resolve({ systolic: 120, diastolic: 80 }));
    });
}

// Simple helper for yes/no
async function captureSimpleYesNo(lang) {
    return new Promise((resolve) => {
        const langCode = lang === 'hi-IN' ? 'hi-IN' : 'en-US';
        startListening(
            (transcript) => {
                const lower = transcript.toLowerCase();
                if (lower.includes('yes') || lower.includes('haan') || lower.includes('ha') || lower.includes('हाँ')) {
                    resolve(true);
                } else {
                    resolve(false);
                }
            },
            langCode,
            5000
        ).catch(() => resolve(false));
    });
}

// Health worker mode - helps patient input measured values
async function healthCenterMode() {
    const selectedLanguage = document.getElementById('voiceLanguage')?.value || 'en';
    const lang = selectedLanguage === 'hi' ? 'hi-IN' : 'en-US';
    const msgLang = selectedLanguage;

    const healthData = {};

    try {
        // Welcome message
        const welcomeMsg = msgLang === 'hi' 
            ? "स्वास्थ्य केंद्र मोड में आपका स्वागत है। रोगी की जानकारी दर्ज करें।"
            : "Welcome to Health Center Mode. Let's enter the patient's information.";
        
        if (typeof speak === 'function') {
            await speak(welcomeMsg, msgLang);
            await new Promise(r => setTimeout(r, 1000));
        }

        // Age
        const ageMsg = msgLang === 'hi' 
            ? "रोगी की उम्र बताएं।"
            : "Tell the patient's age.";
        if (typeof speak === 'function') await speak(ageMsg, msgLang);
        const ageResult = await captureSimpleNumber(lang);
        healthData.age = parseInt(ageResult) || 50;

        // Gender
        const genderMsg = msgLang === 'hi' 
            ? "रोगी पुरुष है या महिला?"
            : "Is the patient male or female?";
        if (typeof speak === 'function') await speak(genderMsg, msgLang);
        const genderResult = await captureSimpleYesNo(lang);
        healthData.gender = genderResult ? 'male' : 'female';

        // Blood Pressure (already measured by health worker)
        const bpMsg = msgLang === 'hi' 
            ? "रोगी का रक्तचाप क्या है? डिवाइस से पढ़ी गई संख्या बताएं।"
            : "What is the patient's blood pressure from the device?";
        if (typeof speak === 'function') await speak(bpMsg, msgLang);
        const bpResult = await captureSimpleBP(lang);
        healthData.systolic_bp = bpResult.systolic || 120;
        healthData.diastolic_bp = bpResult.diastolic || 80;

        // Cholesterol (from test report)
        const cholMsg = msgLang === 'hi' 
            ? "कोलेस्ट्रॉल की रिपोर्ट से संख्या बताएं।"
            : "Tell the cholesterol number from the test report.";
        if (typeof speak === 'function') await speak(cholMsg, msgLang);
        const cholesterolResult = await captureSimpleNumber(lang);
        healthData.cholesterol = parseInt(cholesterolResult) || 200;

        // BMI (calculated from height/weight)
        const bmiMsg = msgLang === 'hi' 
            ? "रोगी का बीएमआई क्या है? वजन और ऊंचाई से गणना करें।"
            : "What is the patient's BMI? Calculate from weight and height.";
        if (typeof speak === 'function') await speak(bmiMsg, msgLang);
        const bmiResult = await captureSimpleNumber(lang);
        healthData.bmi = parseFloat(bmiResult) || 25;

        // Glucose (from test)
        const glucoseMsg = msgLang === 'hi' 
            ? "रोगी की ग्लूकोज रिपोर्ट से संख्या बताएं।"
            : "Tell the glucose number from the test report.";
        if (typeof speak === 'function') await speak(glucoseMsg, msgLang);
        const glucoseResult = await captureSimpleNumber(lang);
        healthData.glucose = parseInt(glucoseResult) || 100;

        // Heart Rate (from device)
        const hrMsg = msgLang === 'hi' 
            ? "रोगी की हृदय गति क्या है?"
            : "What is the patient's heart rate?";
        if (typeof speak === 'function') await speak(hrMsg, msgLang);
        const hrResult = await captureSimpleNumber(lang);
        healthData.heart_rate = parseInt(hrResult) || 70;

        // Cigarettes per day
        const smokeMsg = msgLang === 'hi' 
            ? "रोगी प्रतिदिन कितनी सिगरेट पीता है?"
            : "How many cigarettes does the patient smoke per day?";
        if (typeof speak === 'function') await speak(smokeMsg, msgLang);
        const smokeResult = await captureSimpleNumber(lang);
        healthData.cigarettes_per_day = parseInt(smokeResult) || 0;

        // Medical history
        const strokeMsg = msgLang === 'hi' 
            ? "क्या रोगी को पहले स्ट्रोक हुआ है?"
            : "Has the patient had a stroke?";
        if (typeof speak === 'function') await speak(strokeMsg, msgLang);
        healthData.stroke = await captureSimpleYesNo(lang);

        const hypMsg = msgLang === 'hi' 
            ? "क्या रोगी को उच्च रक्तचाप है?"
            : "Does the patient have high blood pressure?";
        if (typeof speak === 'function') await speak(hypMsg, msgLang);
        healthData.hypertension = await captureSimpleYesNo(lang);

        const diabMsg = msgLang === 'hi' 
            ? "क्या रोगी को मधुमेह है?"
            : "Does the patient have diabetes?";
        if (typeof speak === 'function') await speak(diabMsg, msgLang);
        healthData.diabetes = await captureSimpleYesNo(lang);

        healthData.current_smoker = healthData.cigarettes_per_day > 0;

        // Process prediction
        const processMsg = msgLang === 'hi' 
            ? "रोगी की स्वास्थ्य जानकारी प्रोसेस की जा रही है।"
            : "Processing the patient's health information.";
        if (typeof speak === 'function') await speak(processMsg, msgLang);

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

        // Results message
        const riskPercentage = prediction.risk_percentage.toFixed(1);
        
        let resultMsg = '';
        if (msgLang === 'hi') {
            if (prediction.risk_category === 'low') {
                resultMsg = `रोगी का 10 साल का हृदय रोग जोखिम ${riskPercentage} प्रतिशत है। यह कम जोखिम है। रोगी को स्वस्थ जीवनशैली बनाए रखनी चाहिए।`;
            } else if (prediction.risk_category === 'medium') {
                resultMsg = `रोगी का 10 साल का हृदय रोग जोखिम ${riskPercentage} प्रतिशत है। यह मध्यम जोखिम है। रोगी को डॉक्टर से सलाह लेनी चाहिए।`;
            } else {
                resultMsg = `रोगी का 10 साल का हृदय रोग जोखिम ${riskPercentage} प्रतिशत है। यह अधिक जोखिम है। रोगी को तुरंत डॉक्टर से मिलना चाहिए।`;
            }
        } else {
            if (prediction.risk_category === 'low') {
                resultMsg = `The patient's 10-year heart disease risk is ${riskPercentage} percent. This is low risk. The patient should maintain a healthy lifestyle.`;
            } else if (prediction.risk_category === 'medium') {
                resultMsg = `The patient's 10-year heart disease risk is ${riskPercentage} percent. This is medium risk. The patient should consult a doctor.`;
            } else {
                resultMsg = `The patient's 10-year heart disease risk is ${riskPercentage} percent. This is high risk. The patient should see a doctor immediately.`;
            }
        }

        await speak(resultMsg, msgLang);

        // Display results
        displayResults(prediction);
        
        // Print button for clinic
        const printBtn = document.createElement('button');
        printBtn.className = 'btn-primary';
        printBtn.textContent = '🖨️ Print Report for Patient';
        printBtn.onclick = () => window.print();
        document.getElementById('resultsContainer').appendChild(printBtn);

        showNotification('Health Center Assessment Complete!', 'success');

    } catch (error) {
        console.error('Health center mode error:', error);
        showNotification('Error: ' + error.message, 'error');
        const errorMsg = msgLang === 'hi' 
            ? 'खेद है, एक त्रुटि हुई। कृपया फिर से प्रयास करें।'
            : 'Sorry, an error occurred. Please try again.';
        await speak(errorMsg, msgLang);
    }
}

// ===== MANUAL INPUT MODE FOR HEALTH WORKERS =====
function showHealthWorkerInputForm() {
    const formCard = document.querySelector('.form-card');
    if (formCard) {
        formCard.style.display = 'block';
    }
    
    // Hide voice interface
    document.getElementById('voiceModeInterface').style.display = 'none';
    
    showNotification('Enter measured values from the patient', 'info');
}
