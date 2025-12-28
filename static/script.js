// API base URL
const API_URL = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeGenerateTab();
    initializeClassifyTab();
    loadStatus();
    loadResults();
});

// Tab functionality
function initializeTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        });
    });
}

// Generate tab
function initializeGenerateTab() {
    const numImagesSlider = document.getElementById('numImages');
    const numImagesValue = document.getElementById('numImagesValue');
    const generateBtn = document.getElementById('generateBtn');

    numImagesSlider.addEventListener('input', (e) => {
        numImagesValue.textContent = e.target.value;
    });

    generateBtn.addEventListener('click', generateImages);
}

async function generateImages() {
    const classSelect = document.getElementById('classSelect');
    const numImages = document.getElementById('numImages').value;
    const generateBtn = document.getElementById('generateBtn');
    const imagesContainer = document.getElementById('generatedImages');

    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';
    imagesContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Generating images...</p>';

    try {
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                class: parseInt(classSelect.value),
                num_images: parseInt(numImages)
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayGeneratedImages(data.images, data.class_name);
        } else {
            imagesContainer.innerHTML = `<p style="color: var(--danger);">Error: ${data.error}</p>`;
        }
    } catch (error) {
        imagesContainer.innerHTML = `<p style="color: var(--danger);">Error: ${error.message}</p>`;
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Images';
    }
}

function displayGeneratedImages(images, className) {
    const container = document.getElementById('generatedImages');
    container.innerHTML = '';

    images.forEach(imgB64 => {
        const div = document.createElement('div');
        div.className = 'image-item';
        div.innerHTML = `<img src="data:image/png;base64,${imgB64}" alt="${className}">`;
        container.appendChild(div);
    });
}

// Classify tab
function initializeClassifyTab() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border)';
        const file = e.dataTransfer.files[0];
        if (file) classifyImage(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) classifyImage(file);
    });
}

async function classifyImage(file) {
    const resultsContainer = document.getElementById('classificationResults');
    resultsContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Classifying...</p>';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch(`${API_URL}/api/classify`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayClassificationResults(data);
        } else {
            resultsContainer.innerHTML = `<p style="color: var(--danger);">Error: ${data.error}</p>`;
        }
    } catch (error) {
        resultsContainer.innerHTML = `<p style="color: var(--danger);">Error: ${error.message}</p>`;
    }
}

function displayClassificationResults(results) {
    const container = document.getElementById('classificationResults');
    container.innerHTML = '';

    if (results.baseline) {
        container.innerHTML += createResultCard('Baseline Classifier', results.baseline);
    }

    if (results.augmented) {
        container.innerHTML += createResultCard('Augmented Classifier', results.augmented);
    }
}

function createResultCard(title, result) {
    const topProbs = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    return `
        <div class="result-card">
            <h3 class="result-title">${title}</h3>
            <div class="prediction">
                <div class="prediction-class">${result.predicted_class}</div>
                <div class="prediction-confidence">Confidence: ${(result.confidence * 100).toFixed(2)}%</div>
            </div>
            <div class="probabilities">
                ${topProbs.map(([cls, prob]) => `
                    <div class="prob-item">
                        <span>${cls}</span>
                        <span>${(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar" style="width: ${prob * 100}%"></div>
                `).join('')}
            </div>
        </div>
    `;
}

// Load status
async function loadStatus() {
    try {
        const response = await fetch(`${API_URL}/api/status`);
        const data = await response.json();

        updateStatus('generatorStatus', data.generator_loaded);
        updateStatus('baselineStatus', data.baseline_loaded);
        updateStatus('augmentedStatus', data.augmented_loaded);
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

function updateStatus(elementId, isLoaded) {
    const element = document.getElementById(elementId);
    element.textContent = isLoaded ? 'Loaded' : 'Not Loaded';
    element.className = `status-value ${isLoaded ? 'loaded' : 'not-loaded'}`;
}

// Load results
async function loadResults() {
    try {
        const response = await fetch(`${API_URL}/api/results`);
        const data = await response.json();

        if (data.baseline && data.augmented) {
            displayMetrics(data);
            displayPerClassChart(data);
        }
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

function displayMetrics(data) {
    document.getElementById('baselineAcc').textContent = `${data.baseline.accuracy.toFixed(2)}%`;
    document.getElementById('augmentedAcc').textContent = `${data.augmented.accuracy.toFixed(2)}%`;
    document.getElementById('accImprovement').textContent = `+${data.comparison.accuracy_improvement.toFixed(2)}%`;

    document.getElementById('baselineF1').textContent = data.baseline.f1_score.toFixed(4);
    document.getElementById('augmentedF1').textContent = data.augmented.f1_score.toFixed(4);
    document.getElementById('f1Improvement').textContent = `+${data.comparison.f1_improvement.toFixed(4)}`;
}

function displayPerClassChart(data) {
    const ctx = document.getElementById('perClassChart').getContext('2d');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.baseline.class_names,
            datasets: [
                {
                    label: 'Baseline',
                    data: data.baseline.per_class_accuracy,
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Augmented',
                    data: data.augmented.per_class_accuracy,
                    backgroundColor: 'rgba(139, 92, 246, 0.6)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#cbd5e1' },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#cbd5e1' },
                    grid: { color: '#334155' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1' }
                }
            }
        }
    });
}
