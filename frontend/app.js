// =====================================
// Main Application Logic
// =====================================

const API_BASE = 'http://localhost:5000/api';

// UI Helper Functions (Define First)
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = show ? 'flex' : 'none';
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;

    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAuth();
});

// Setup all event listeners
function setupEventListeners() {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Hamburger menu
    const hamburger = document.getElementById('hamburger');
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            const menu = document.getElementById('navMenu');
            menu.classList.toggle('active');
        });
    }

    // Prediction form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredict);
    }

    // Form tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(btn.dataset.tab);
        });
    });
}

// Navigate between pages
function navigate(page) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active'));

    // Show selected page
    const pageElement = document.getElementById(page + 'Page');
    if (pageElement) {
        pageElement.classList.add('active');
    }

    // Update navbar
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    event.target.classList.add('active');

    // Load page content
    switch (page) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'analytics':
            loadAnalytics();
            break;
        case 'profile':
            loadProfile();
            break;
    }
}

// Load Dashboard
async function loadDashboard() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE}/predictions/latest`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();
            updateDashboard(data);
        }
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

// Update Dashboard
function updateDashboard(data) {
    // Update risk score with 3D visualization
    if (data.risk_score !== undefined) {
        document.getElementById('riskPercentage').textContent = data.risk_score.toFixed(1) + '%';
        document.getElementById('riskCategory').textContent = getRiskCategory(data.risk_score);
        updateRiskMeterColor(data.risk_score);
    }

    // Update health metrics
    if (data.health_metrics) {
        document.getElementById('bpValue').textContent = data.health_metrics.blood_pressure || '-';
        document.getElementById('chValue').textContent = data.health_metrics.cholesterol || '-';
        document.getElementById('bmiValue').textContent = data.health_metrics.bmi || '-';
        document.getElementById('hrValue').textContent = data.health_metrics.heart_rate || '-';
    }

    // Update recommendations
    if (data.recommendations) {
        const recommendationsList = document.getElementById('recommendationsList');
        recommendationsList.innerHTML = data.recommendations
            .map(rec => `<div class="recommendation-item">✓ ${rec}</div>`)
            .join('');
    }
}

// Switch Form Tabs
function switchTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.form-section').forEach(s => s.classList.remove('active'));
    
    // Show selected tab
    const tabElement = document.getElementById(tab + 'Tab');
    if (tabElement) {
        tabElement.classList.add('active');
    }

    // Update button styles
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

// Handle Prediction
async function handlePredict(event) {
    event.preventDefault();

    const formData = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        systolic_bp: document.getElementById('sysBP').value,
        diastolic_bp: document.getElementById('diaBP').value,
        cholesterol: document.getElementById('cholesterol').value,
        bmi: document.getElementById('bmi').value,
        heart_rate: document.getElementById('heartRate').value,
        glucose: document.getElementById('glucose').value,
        smoking: document.getElementById('smoking').value === '2',
        stroke: document.getElementById('stroke').checked,
        hypertension: document.getElementById('hypertension').checked,
        diabetes: document.getElementById('diabetes').checked
    };

    showLoading(true);

    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE}/predictions/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (response.ok) {
            showPredictionResult(data);
            showToast('Prediction completed successfully!', 'success');
        } else {
            showToast(data.message || 'Prediction failed', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Error processing prediction', 'error');
    }

    showLoading(false);
}

// Show Prediction Result
function showPredictionResult(data) {
    document.getElementById('resultSection').style.display = 'block';
    
    const percentage = data.risk_score || 0;
    const category = getRiskCategory(percentage);

    document.getElementById('resultPercentage').textContent = percentage.toFixed(1) + '%';
    document.getElementById('resultCategory').textContent = category;

    // Create description
    const description = `
        <h4>Risk Assessment: ${category}</h4>
        <p>Your 10-year cardiovascular disease risk is <strong>${percentage.toFixed(1)}%</strong>.</p>
        ${getRecommendations(data)}
    `;
    document.getElementById('resultDescription').innerHTML = description;

    // Scroll to result
    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
}

// Get Risk Category
function getRiskCategory(percentage) {
    if (percentage < 5) return 'Very Low Risk 🟢';
    if (percentage < 10) return 'Low Risk 🟢';
    if (percentage < 20) return 'Medium Risk 🟡';
    if (percentage < 30) return 'High Risk 🔴';
    return 'Very High Risk 🔴';
}

// Get Recommendations
function getRecommendations(data) {
    const recommendations = data.recommendations || [];
    return `
        <h4>Recommendations:</h4>
        <ul>
            ${recommendations.map(rec => `<li>✓ ${rec}</li>`).join('')}
        </ul>
    `;
}

// Load Analytics
function loadAnalytics() {
    try {
        // Placeholder for analytics
        console.log('Loading analytics...');
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Load Profile
function loadProfile() {
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    if (user.email) {
        document.getElementById('profileName').textContent = user.name || 'User';
        document.getElementById('profileEmail').textContent = user.email;
    }
}

// Export Data
async function exportData() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE}/export/csv`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'cvd_predictions.csv';
            a.click();
            showToast('Data exported successfully!', 'success');
        }
    } catch (error) {
        console.error('Export error:', error);
        showToast('Export failed', 'error');
    }
}


