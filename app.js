// ============================================================
// MNIST Handwritten Digit Recognition - Client-side with ONNXRuntime-Web
// Loads a pre-trained PyTorch CNN model (exported to ONNX)
// ============================================================

(function () {
    'use strict';

    // ---- DOM Elements ----
    const drawCanvas = document.getElementById('drawCanvas');
    const drawCtx = drawCanvas.getContext('2d');
    const previewCanvas = document.getElementById('previewCanvas');
    const previewCtx = previewCanvas.getContext('2d');
    const canvasOverlay = document.getElementById('canvasOverlay');
    const canvasWrapper = document.getElementById('canvasWrapper');
    const btnClear = document.getElementById('btnClear');
    const btnPredict = document.getElementById('btnPredict');
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeValue = document.getElementById('brushSizeValue');
    const predictionNumber = document.getElementById('predictionNumber');
    const predictionHero = document.getElementById('predictionHero');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const chartBars = document.getElementById('chartBars');
    const statusBadge = document.getElementById('statusBadge');

    // ---- State ----
    let isDrawing = false;
    let hasDrawn = false;
    let model = null;
    let brushSize = 18;

    // ---- Initialize Chart ----
    function initChart() {
        let html = '';
        for (let i = 0; i <= 9; i++) {
            html += `
                <div class="chart-row" id="chartRow${i}">
                    <span class="chart-label">${i}</span>
                    <div class="chart-bar-bg">
                        <div class="chart-bar-fill normal" id="chartFill${i}"></div>
                    </div>
                    <span class="chart-percent" id="chartPercent${i}">0%</span>
                </div>
            `;
        }
        chartBars.innerHTML = html;
    }

    // ---- Load ONNX Model ----
    async function loadModel() {
        updateStatus('Đang tải mô hình ONNX...', 'loading');
        try {
            // Setup ORT to use WASM backend
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
            
            model = await ort.InferenceSession.create('./models/mnist_improved_cnn.onnx');
            updateStatus('Sẵn sàng!', 'ready');
            btnPredict.disabled = false;
        } catch (e) {
            console.error('Lỗi tải mô hình:', e);
            updateStatus('Lỗi tải mô hình: ' + e.message, 'error');
        }
    }

    // ---- Status Badge ----
    function updateStatus(text, type) {
        const statusText = statusBadge.querySelector('.status-text');
        statusText.textContent = text;
        statusBadge.className = 'status-badge';
        if (type === 'ready') statusBadge.classList.add('ready');
        else if (type === 'error') statusBadge.classList.add('error');
    }

    // ---- Drawing Functions ----
    function initCanvas() {
        drawCtx.fillStyle = '#ffffff';
        drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.strokeStyle = '#000000';
        drawCtx.lineWidth = brushSize;
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        if (!hasDrawn) {
            hasDrawn = true;
            canvasOverlay.classList.add('hidden');
            canvasWrapper.classList.add('active');
        }
        const pos = getPos(e);
        drawCtx.beginPath();
        drawCtx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        drawCtx.lineTo(pos.x, pos.y);
        drawCtx.stroke();
    }

    function endDraw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        isDrawing = false;
        drawCtx.closePath();
        // Auto-predict after drawing
        if (model) {
            predict();
        }
    }

    function getPos(e) {
        const rect = drawCanvas.getBoundingClientRect();
        const scaleX = drawCanvas.width / rect.width;
        const scaleY = drawCanvas.height / rect.height;

        if (e.touches && e.touches.length > 0) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    // ---- Clear Canvas ----
    function clearCanvas() {
        drawCtx.fillStyle = '#ffffff';
        drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
        hasDrawn = false;
        canvasOverlay.classList.remove('hidden');
        canvasWrapper.classList.remove('active');
        predictionNumber.textContent = '?';
        confidenceBar.style.width = '0%';
        confidenceValue.textContent = '--%';
        predictionHero.classList.remove('active');
        previewCtx.clearRect(0, 0, 28, 28);

        for (let i = 0; i <= 9; i++) {
            const fill = document.getElementById(`chartFill${i}`);
            const pct = document.getElementById(`chartPercent${i}`);
            const row = document.getElementById(`chartRow${i}`);
            if (fill) {
                fill.style.width = '0%';
                fill.className = 'chart-bar-fill normal';
            }
            if (pct) pct.textContent = '0%';
            if (row) row.classList.remove('highlight');
        }
    }

    // ---- Preprocess Canvas -> 28x28 ----
    function preprocessCanvas() {
        const imageData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
        const { data, width, height } = imageData;
        let minX = width, minY = height, maxX = 0, maxY = 0;
        let found = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                if (r < 200 || g < 200 || b < 200) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }

        if (!found) return null;

        const padding = 20;
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(width - 1, maxX + padding);
        maxY = Math.min(height - 1, maxY + padding);

        const cropW = maxX - minX + 1;
        const cropH = maxY - minY + 1;
        const size = Math.max(cropW, cropH);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = size;
        tempCanvas.height = size;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.fillStyle = '#ffffff';
        tempCtx.fillRect(0, 0, size, size);

        const offsetX = Math.floor((size - cropW) / 2);
        const offsetY = Math.floor((size - cropH) / 2);
        tempCtx.drawImage(drawCanvas, minX, minY, cropW, cropH, offsetX, offsetY, cropW, cropH);

        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = 28;
        finalCanvas.height = 28;
        const finalCtx = finalCanvas.getContext('2d');
        finalCtx.fillStyle = '#ffffff';
        finalCtx.fillRect(0, 0, 28, 28);

        const innerSize = 20;
        const innerOffset = 4;
        finalCtx.drawImage(tempCanvas, 0, 0, size, size, innerOffset, innerOffset, innerSize, innerSize);

        const finalImageData = finalCtx.getImageData(0, 0, 28, 28);
        const pixels = new Float32Array(28 * 28);

        for (let i = 0; i < 28 * 28; i++) {
            const r = finalImageData.data[i * 4];
            pixels[i] = (255 - r) / 255.0;
        }

        // Update preview canvas
        const previewImageData = previewCtx.createImageData(28, 28);
        for (let i = 0; i < 28 * 28; i++) {
            const v = Math.round(pixels[i] * 255);
            previewImageData.data[i * 4] = v;
            previewImageData.data[i * 4 + 1] = v;
            previewImageData.data[i * 4 + 2] = v;
            previewImageData.data[i * 4 + 3] = 255;
        }
        previewCtx.putImageData(previewImageData, 0, 0);

        return pixels;
    }

    // ---- Prediction ----
    async function predict() {
        if (!model || !hasDrawn) return;

        const pixels = preprocessCanvas();
        if (!pixels) return;

        try {
            const inputTensor = new ort.Tensor('float32', pixels, [1, 1, 28, 28]);
            const feeds = { input: inputTensor };
            const results = await model.run(feeds);
            const logits = results.output.data;

            // Apply softmax (PyTorch models usually output logits)
            const maxLogit = Math.max(...logits);
            const exps = Array.from(logits).map(x => Math.exp(x - maxLogit));
            const sumExps = exps.reduce((a, b) => a + b, 0);
            const probabilities = exps.map(x => x / sumExps);

        let maxIdx = 0;
        let maxProb = probabilities[0];
        for (let i = 1; i < 10; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIdx = i;
            }
        }

        predictionNumber.textContent = maxIdx;
        const confPct = (maxProb * 100).toFixed(1);
        confidenceBar.style.width = confPct + '%';
        confidenceValue.textContent = confPct + '%';
        predictionHero.classList.add('active');

        for (let i = 0; i <= 9; i++) {
            const fill = document.getElementById(`chartFill${i}`);
            const pct = document.getElementById(`chartPercent${i}`);
            const row = document.getElementById(`chartRow${i}`);
            const prob = (probabilities[i] * 100).toFixed(1);

            fill.style.width = prob + '%';
            pct.textContent = prob + '%';

            if (i === maxIdx) {
                fill.className = 'chart-bar-fill top';
                row.classList.add('highlight');
            } else {
                fill.className = 'chart-bar-fill normal';
                row.classList.remove('highlight');
            }
        }
        } catch (e) {
            console.error("Prediction error:", e);
            updateStatus('Lỗi dự đoán', 'error');
        }
    }

    // ---- Event Listeners ----
    drawCanvas.addEventListener('mousedown', startDraw);
    drawCanvas.addEventListener('mousemove', draw);
    drawCanvas.addEventListener('mouseup', endDraw);
    drawCanvas.addEventListener('mouseleave', endDraw);

    drawCanvas.addEventListener('touchstart', startDraw, { passive: false });
    drawCanvas.addEventListener('touchmove', draw, { passive: false });
    drawCanvas.addEventListener('touchend', endDraw, { passive: false });
    drawCanvas.addEventListener('touchcancel', endDraw, { passive: false });

    btnClear.addEventListener('click', clearCanvas);
    btnPredict.addEventListener('click', predict);

    brushSizeInput.addEventListener('input', () => {
        brushSize = parseInt(brushSizeInput.value);
        brushSizeValue.textContent = brushSize + 'px';
        drawCtx.lineWidth = brushSize;
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'c' || e.key === 'C') clearCanvas();
        if (e.key === 'Enter') predict();
    });

    // ---- Init ----
    initChart();
    initCanvas();
    btnPredict.disabled = true;

    loadModel();

})();
