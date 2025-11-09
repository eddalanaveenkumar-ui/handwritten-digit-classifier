// Complete Handwritten Digit Classifier JavaScript
console.log("🚀 Loading Handwritten Digit Classifier...");

class DigitDrawer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error("❌ Canvas not found:", canvasId);
            return;
        }

        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.brushSize = 15;

        console.log("✅ Canvas found, initializing...");
        this.init();
    }

    init() {
        // Set black background with white drawing (MNIST style)
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = this.brushSize;
        this.ctx.lineJoin = 'round';
        this.ctx.lineCap = 'round';

        this.setupEventListeners();
        this.setupBrushControls();
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));

        console.log("✅ Event listeners setup");
    }

    setupBrushControls() {
        const brushSize = document.getElementById('brushSize');
        const brushSizeValue = document.getElementById('brushSizeValue');

        if (brushSize && brushSizeValue) {
            brushSize.addEventListener('input', (e) => {
                this.brushSize = parseInt(e.target.value);
                brushSizeValue.textContent = this.brushSize;
                this.ctx.lineWidth = this.brushSize;
            });
        }
    }

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        [this.lastX, this.lastY] = [pos.x, pos.y];

        // Draw initial point
        this.ctx.beginPath();
        this.ctx.arc(this.lastX, this.lastY, this.brushSize/2, 0, Math.PI * 2);
        this.ctx.fill();

        // Hide overlay when drawing starts
        const overlay = document.querySelector('.canvas-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    draw(e) {
        if (!this.isDrawing) return;

        e.preventDefault();
        const pos = this.getMousePos(e);

        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();

        [this.lastX, this.lastY] = [pos.x, pos.y];
    }

    stopDrawing() {
        this.isDrawing = false;
        this.ctx.beginPath(); // Reset path for next drawing
    }

    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.canvas.dispatchEvent(mouseEvent);
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        let clientX, clientY;

        if (e.type.includes('touch')) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    clear() {
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        const overlay = document.querySelector('.canvas-overlay');
        if (overlay) {
            overlay.style.display = 'block';
        }
    }

    getImageData() {
        return this.canvas.toDataURL('image/png');
    }

    getProcessedImageData() {
        // Create a temporary canvas for processing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw and scale down to 28x28
        tempCtx.imageSmoothingEnabled = false;
        tempCtx.drawImage(this.canvas, 0, 0, 28, 28);

        // Update preview
        const previewCanvas = document.getElementById('previewCanvas');
        if (previewCanvas) {
            const previewCtx = previewCanvas.getContext('2d');
            previewCtx.drawImage(tempCanvas, 0, 0);
        }

        return tempCanvas.toDataURL('image/png');
    }
}

class PredictionManager {
    constructor() {
        this.digitDrawer = null;
        this.init();
    }

    async init() {
        console.log("🔄 Initializing Prediction Manager...");

        // Wait for DOM to be fully loaded
        await this.waitForElement('#drawingCanvas');
        this.digitDrawer = new DigitDrawer('drawingCanvas');
        this.setupEventListeners();
        this.setupCorrectionButtons();
        this.loadModelStats();

        console.log("✅ Prediction Manager initialized successfully!");
    }

    waitForElement(selector) {
        return new Promise(resolve => {
            if (document.querySelector(selector)) {
                return resolve();
            }

            const observer = new MutationObserver(mutations => {
                if (document.querySelector(selector)) {
                    observer.disconnect();
                    resolve();
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    }

    setupEventListeners() {
        const clearBtn = document.getElementById('clearBtn');
        const predictBtn = document.getElementById('predictBtn');

        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.digitDrawer.clear();
                this.hideResults();
                this.hidePreview();
                this.hideCorrectionOptions();
            });
        }

        if (predictBtn) {
            predictBtn.addEventListener('click', () => {
                this.predictDigit();
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'c' || e.key === 'C') {
                if (clearBtn) clearBtn.click();
            } else if (e.key === 'Enter') {
                if (predictBtn) predictBtn.click();
            }
        });
    }

    setupCorrectionButtons() {
        // Create correction section if it doesn't exist
        if (!document.getElementById('correctionSection')) {
            const resultsSection = document.querySelector('.results-section');
            if (resultsSection) {
                const correctionHTML = `
                    <div id="correctionSection" class="correction-section hidden">
                        <div class="correction-card">
                            <h4>❌ Wrong Prediction?</h4>
                            <p>Help improve the AI by correcting it:</p>
                            <div class="correction-buttons">
                                <p>What digit did you actually draw?</p>
                                <div class="digit-buttons">
                                    ${Array.from({length: 10}, (_, i) =>
                                        `<button class="digit-btn" data-digit="${i}">${i}</button>`
                                    ).join('')}
                                </div>
                            </div>
                            <button id="cancelCorrection" class="btn btn-secondary">Cancel</button>
                        </div>
                    </div>
                `;
                resultsSection.insertAdjacentHTML('beforeend', correctionHTML);

                // Add event listeners to digit buttons
                document.querySelectorAll('.digit-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const correctDigit = parseInt(e.target.dataset.digit);
                        this.submitCorrection(correctDigit);
                    });
                });

                document.getElementById('cancelCorrection').addEventListener('click', () => {
                    this.hideCorrectionOptions();
                });
            }
        }
    }

    async predictDigit() {
        if (!this.digitDrawer) {
            this.showError('Canvas not initialized. Please refresh the page.');
            return;
        }

        const canvas = this.digitDrawer.canvas;

        // Check if canvas has drawing
        const imageData = this.digitDrawer.ctx.getImageData(0, 0, canvas.width, canvas.height);
        const hasDrawing = this.hasDrawing(imageData);

        if (!hasDrawing) {
            this.showError('Please draw a digit first!');
            return;
        }

        this.showLoading();
        this.hideError();
        this.showPreview();

        try {
            const modelType = document.getElementById('modelSelect').value;
            const processedImage = this.digitDrawer.getProcessedImageData();

            // Convert data URL to blob
            const blob = await this.dataURLtoBlob(processedImage);

            const formData = new FormData();
            formData.append('image', blob, 'digit.png');
            formData.append('model_type', modelType);

            console.log("🔄 Sending prediction request...");
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log("📊 Prediction result:", result);

            if (result.success) {
                this.showPrediction(result);
            } else {
                this.showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Network error. Please try again.');
        }
    }

    showPrediction(result) {
        this.hideLoading();

        const predictionResult = document.getElementById('predictionResult');
        const predictedNumber = document.getElementById('predictedNumber');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceBars = document.getElementById('confidenceBars');

        if (predictionResult && predictedNumber && confidenceValue) {
            // Update main prediction
            predictedNumber.textContent = result.prediction;
            confidenceValue.textContent = `${(result.confidence * 100).toFixed(1)}%`;

            // Add model used indicator
            const modelIndicator = document.getElementById('modelUsed');
            if (!modelIndicator) {
                const newIndicator = document.createElement('p');
                newIndicator.id = 'modelUsed';
                newIndicator.className = 'model-indicator';
                predictionResult.appendChild(newIndicator);
            }
            document.getElementById('modelUsed').textContent = `Model: ${result.model_used}`;

            // Add success animation
            predictedNumber.classList.add('success');
            setTimeout(() => predictedNumber.classList.remove('success'), 500);

            // Update confidence bars if available
            if (result.all_predictions && confidenceBars) {
                this.updateConfidenceBars(result.all_predictions);
                confidenceBars.classList.remove('hidden');
            }

            predictionResult.classList.remove('hidden');

            // Show correction button if confidence is low
            if (result.confidence < 0.7) {
                this.showCorrectionOptions();
            }

            console.log(`✅ Prediction displayed: ${result.prediction} with ${(result.confidence * 100).toFixed(1)}% confidence`);
        }
    }

    updateConfidenceBars(predictions) {
        const barsContainer = document.getElementById('barsContainer');
        if (!barsContainer) return;

        barsContainer.innerHTML = '';

        predictions.forEach((confidence, digit) => {
            const percentage = (confidence * 100).toFixed(1);
            const barContainer = document.createElement('div');
            barContainer.className = 'bar-container';

            barContainer.innerHTML = `
                <div class="bar-label">
                    <span class="digit-label">${digit}</span>
                    <span class="confidence-value">${percentage}%</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" style="width: ${percentage}%"></div>
                </div>
            `;

            barsContainer.appendChild(barContainer);
        });
    }

    showCorrectionOptions() {
        const correctionSection = document.getElementById('correctionSection');
        if (correctionSection) {
            correctionSection.classList.remove('hidden');
        }
    }

    hideCorrectionOptions() {
        const correctionSection = document.getElementById('correctionSection');
        if (correctionSection) {
            correctionSection.classList.add('hidden');
        }
    }

    async submitCorrection(correctDigit) {
        const currentImage = this.digitDrawer.getImageData();

        try {
            const response = await fetch('/correct_prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: currentImage,
                    correct_label: correctDigit
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showMessage('✅ Thanks! The AI will learn from your correction.', 'success');
                console.log(`Correction recorded: ${correctDigit}. Total samples: ${result.samples_count}`);

                // Update stats display
                this.updateFineTuningStats(result.samples_count);
            } else {
                this.showMessage('❌ Failed to save correction', 'error');
            }

            this.hideCorrectionOptions();

        } catch (error) {
            console.error('Correction error:', error);
            this.showMessage('❌ Network error saving correction', 'error');
        }
    }

    updateFineTuningStats(samplesCount) {
        // Update UI to show fine-tuning progress
        const statsElement = document.getElementById('fineTuningStats');
        if (!statsElement) {
            // Create stats element if it doesn't exist
            const statsSection = document.querySelector('.stats-section');
            if (statsSection) {
                const statsHTML = `
                    <div class="stat-card">
                        <h4>🔄 Fine-Tuning Progress</h4>
                        <div id="fineTuningStats">
                            <p>User samples: <span id="samplesCount">${samplesCount}</span></p>
                            <p>Status: <span id="fineTuneStatus">Collecting samples...</span></p>
                        </div>
                    </div>
                `;
                statsSection.insertAdjacentHTML('beforeend', statsHTML);
            }
        } else {
            // Update existing stats
            document.getElementById('samplesCount').textContent = samplesCount;
            const statusElement = document.getElementById('fineTuneStatus');
            if (statusElement) {
                if (samplesCount >= 5) {
                    statusElement.textContent = 'Fine-tuned! 🎯';
                    statusElement.style.color = '#27ae60';
                } else {
                    statusElement.textContent = `Need ${5 - samplesCount} more samples`;
                }
            }
        }
    }

    showLoading() {
        const loadingElement = document.getElementById('loadingSpinner');
        const predictionResult = document.getElementById('predictionResult');
        const confidenceBars = document.getElementById('confidenceBars');

        if (loadingElement) loadingElement.classList.remove('hidden');
        if (predictionResult) predictionResult.classList.add('hidden');
        if (confidenceBars) confidenceBars.classList.add('hidden');
    }

    hideLoading() {
        const loadingElement = document.getElementById('loadingSpinner');
        if (loadingElement) loadingElement.classList.add('hidden');
    }

    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.querySelector('p').textContent = `⚠️ ${message}`;
            errorElement.classList.remove('hidden');
        }
        this.hideLoading();
    }

    hideError() {
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) errorElement.classList.add('hidden');
    }

    hideResults() {
        const predictionResult = document.getElementById('predictionResult');
        const confidenceBars = document.getElementById('confidenceBars');

        if (predictionResult) predictionResult.classList.add('hidden');
        if (confidenceBars) confidenceBars.classList.add('hidden');
    }

    showPreview() {
        const previewSection = document.getElementById('previewSection');
        if (previewSection) previewSection.classList.remove('hidden');
    }

    hidePreview() {
        const previewSection = document.getElementById('previewSection');
        if (previewSection) previewSection.classList.add('hidden');
    }

    showMessage(message, type = 'info') {
        // Create or update message element
        let messageElement = document.getElementById('userMessage');
        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.id = 'userMessage';
            messageElement.className = `user-message ${type}`;
            document.querySelector('.container').appendChild(messageElement);
        }

        messageElement.textContent = message;
        messageElement.className = `user-message ${type} visible`;

        // Auto-hide after 3 seconds
        setTimeout(() => {
            messageElement.classList.remove('visible');
        }, 3000);
    }

    hasDrawing(imageData) {
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            // Check if pixel is not black (has drawing)
            if (data[i] > 10 || data[i + 1] > 10 || data[i + 2] > 10) {
                return true;
            }
        }
        return false;
    }

    async dataURLtoBlob(dataURL) {
        const response = await fetch(dataURL);
        return await response.blob();
    }

    async loadModelStats() {
        try {
            const response = await fetch('/model_stats');
            if (!response.ok) return;

            const stats = await response.json();
            const rfAccuracy = document.getElementById('rfAccuracy');
            const nnAccuracy = document.getElementById('nnAccuracy');

            if (stats.rf_accuracy && rfAccuracy) {
                rfAccuracy.textContent = `${(stats.rf_accuracy * 100).toFixed(1)}%`;
            }
            if (stats.nn_accuracy && nnAccuracy) {
                nnAccuracy.textContent = `${(stats.nn_accuracy * 100).toFixed(1)}%`;
            }

            // Update fine-tuning stats if available
            if (stats.fine_tuned_samples !== undefined) {
                this.updateFineTuningStats(stats.fine_tuned_samples);
            }

        } catch (error) {
            console.error('Failed to load model stats:', error);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 DOM loaded, initializing Handwritten Digit Classifier...');
    new PredictionManager();

    console.log(`
    🎯 Handwritten Digit Classifier Ready!
    Shortcuts:
    - Press 'C' to clear canvas
    - Press 'Enter' to predict
    - Draw digits 0-9 and see AI magic!
    - Use correction feature to improve the AI
    `);
});

// Add some utility functions for debugging
window.debugCanvas = function() {
    const canvas = document.getElementById('drawingCanvas');
    if (canvas) {
        console.log('Canvas debug info:');
        console.log('- Size:', canvas.width, 'x', canvas.height);
        console.log('- Context:', canvas.getContext('2d'));

        // Test drawing
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'red';
        ctx.fillRect(10, 10, 50, 50);
        console.log('✅ Test drawing completed');
    } else {
        console.log('❌ Canvas not found');
    }
};

window.testPrediction = async function() {
    console.log('🧪 Testing prediction endpoint...');
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Health check:', data);
        return data;
    } catch (error) {
        console.error('Health check failed:', error);
        return null;
    }
};