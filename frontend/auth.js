// =====================================
// Authentication Logic
// =====================================

// Toggle between Login and Register
function toggleAuth() {
    const loginPage = document.getElementById('loginPage');
    const registerPage = document.getElementById('registerPage');

    loginPage.classList.toggle('active');
    registerPage.classList.toggle('active');
}

// Handle Login
async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (response.ok) {
            // Save token
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));

            // Show success message
            showToast('Login successful!', 'success');

            // Transition to app
            setTimeout(() => {
                document.getElementById('authContainer').style.display = 'none';
                document.getElementById('appContainer').style.display = 'block';
                loadDashboard();
            }, 500);
        } else {
            showToast(data.message || 'Login failed', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showToast('Connection error. Please try again.', 'error');
    }

    showLoading(false);
}

// Handle Register
async function handleRegister(event) {
    event.preventDefault();

    const firstName = document.getElementById('regFirstName').value;
    const lastName = document.getElementById('regLastName').value;
    const email = document.getElementById('regEmail').value;
    const password = document.getElementById('regPassword').value;
    const confirmPassword = document.getElementById('regConfirmPassword').value;

    if (password !== confirmPassword) {
        showToast('Passwords do not match', 'error');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                first_name: firstName,
                last_name: lastName,
                email,
                password
            })
        });

        const data = await response.json();

        if (response.ok) {
            showToast('Registration successful! Please login.', 'success');
            toggleAuth();
            // Clear form
            document.getElementById('regFirstName').value = '';
            document.getElementById('regLastName').value = '';
            document.getElementById('regEmail').value = '';
            document.getElementById('regPassword').value = '';
            document.getElementById('regConfirmPassword').value = '';
        } else {
            showToast(data.message || 'Registration failed', 'error');
        }
    } catch (error) {
        console.error('Register error:', error);
        showToast('Connection error. Please try again.', 'error');
    }

    showLoading(false);
}

// Logout
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        document.getElementById('authContainer').style.display = 'block';
        document.getElementById('appContainer').style.display = 'none';
        showToast('Logged out successfully', 'success');
        cleanup3DScene();
    }
}

// Check if user is logged in
function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        document.getElementById('authContainer').style.display = 'block';
        document.getElementById('appContainer').style.display = 'none';
    } else {
        document.getElementById('authContainer').style.display = 'none';
        document.getElementById('appContainer').style.display = 'block';
    }
}

// Check auth on page load
document.addEventListener('DOMContentLoaded', checkAuth);
