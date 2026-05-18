// =====================================
// 3D Scene Management with Three.js
// =====================================

let scene, camera, renderer;
let riskMeter = null;
let animate3D = false;

// Initialize 3D Risk Meter
function init3DScene() {
    const container = document.getElementById('riskContainer');
    if (!container || !container.offsetParent) return; // Skip if not visible

    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f7fa);

    // Camera setup
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Create 3D Risk Meter (Sphere with animated color)
    createRiskMeter();
    
    // Lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 5, 5);
    scene.add(light);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Start animation
    animate3D = true;
    animateScene();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function createRiskMeter() {
    // Create a sphere for the risk meter
    const geometry = new THREE.IcosahedronGeometry(2, 4);
    const material = new THREE.MeshStandardMaterial({
        color: 0x2ecc71, // Green (Low Risk)
        metalness: 0.7,
        roughness: 0.2,
        emissive: 0x2ecc71,
        emissiveIntensity: 0.3
    });

    riskMeter = new THREE.Mesh(geometry, material);
    riskMeter.rotation.x = Math.random() * Math.PI;
    riskMeter.rotation.y = Math.random() * Math.PI;
    scene.add(riskMeter);
}

function updateRiskMeterColor(riskPercentage) {
    if (!riskMeter) return;

    let color, emissiveColor;
    if (riskPercentage < 10) {
        color = 0x2ecc71; // Green
        emissiveColor = 0x2ecc71;
    } else if (riskPercentage < 20) {
        color = 0xf39c12; // Orange
        emissiveColor = 0xf39c12;
    } else {
        color = 0xe74c3c; // Red
        emissiveColor = 0xe74c3c;
    }

    riskMeter.material.color.setHex(color);
    riskMeter.material.emissive.setHex(emissiveColor);
}

function animateScene() {
    if (!animate3D) return;

    requestAnimationFrame(animateScene);

    // Rotate the risk meter
    if (riskMeter) {
        riskMeter.rotation.x += 0.005;
        riskMeter.rotation.y += 0.008;
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    if (!camera || !renderer) return;

    const container = document.getElementById('riskContainer');
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Create 3D Charts
function create3DChart(elementId, data) {
    const container = document.getElementById(elementId);
    if (!container) return;

    // Using Chart.js for now (can be upgraded to Three.js later)
    const ctx = container.getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['BP', 'Cholesterol', 'BMI', 'Heart Rate'],
            datasets: [{
                label: 'Health Metrics',
                data: data,
                backgroundColor: [
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(241, 196, 15, 0.8)',
                    'rgba(231, 76, 60, 0.8)'
                ],
                borderColor: [
                    'rgba(46, 204, 113, 1)',
                    'rgba(52, 152, 219, 1)',
                    'rgba(241, 196, 15, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Cleanup on page change
function cleanup3DScene() {
    animate3D = false;
    if (renderer) {
        renderer.dispose();
    }
    if (scene) {
        scene.clear();
    }
}

// Initialize 3D animations on load
document.addEventListener('DOMContentLoaded', () => {
    // Delay initialization until after page loads
    setTimeout(init3DScene, 500);
});
