const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loadingState = document.getElementById('loading-state');
const resultsEmpty = document.getElementById('results-empty');
const resultsPopulated = document.getElementById('results-populated');
const routingBadge = document.getElementById('routing-badge');
const progressBar = document.getElementById('pipeline-progress');
const stepText = document.getElementById('step-text');

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

async function uploadFile(file) {
    // UI state transitions
    dropZone.classList.add('hidden');
    resultsEmpty.classList.add('hidden');
    resultsPopulated.classList.add('hidden');
    routingBadge.classList.add('hidden');
    loadingState.classList.remove('hidden');

    // Fake progress animation for UX
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = `${progress}%`;
        
        if (progress > 30 && progress < 60) stepText.innerText = "Extracting Named Entities (NER)...";
        if (progress > 60) stepText.innerText = "Calculating Confidence Bounds...";
    }, 500);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/dev/process-document', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        stepText.innerText = "Routing complete.";

        setTimeout(() => {
            renderResults(data);
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        alert('Error communicating with ML API. Check console.');
        console.error(error);
        dropZone.classList.remove('hidden');
        loadingState.classList.add('hidden');
    }
}

function renderResults(data) {
    loadingState.classList.add('hidden');
    resultsPopulated.classList.remove('hidden');
    dropZone.classList.remove('hidden'); // allow new uploads

    // Rendering Badge
    routingBadge.classList.remove('hidden', 'badge-success', 'badge-warning');
    if (data.routing === 'AUTO_SUCCESS') {
        routingBadge.innerText = '✓ VERIFIED';
        routingBadge.classList.add('badge-success');
    } else {
        routingBadge.innerText = '⚠ MANUAL REVIEW';
        routingBadge.classList.add('badge-warning');
    }

    // Rendering Overall Confidence
    const confScore = Math.round(data.overall_confidence * 100);
    const confFill = document.getElementById('confidence-fill');
    document.getElementById('overall-confidence').innerText = `${confScore}%`;
    
    confFill.style.width = `${confScore}%`;
    confFill.className = 'meter-fill';
    if (confScore >= (data.threshold_used * 100)) confFill.classList.add('fill-green');
    else if (confScore > 50) confFill.classList.add('fill-yellow');
    else confFill.classList.add('fill-red');

    // Render Extracted Entities
    const entitiesContainer = document.getElementById('entities-container');
    entitiesContainer.innerHTML = '';
    
    // Fallback if no entities extracted
    if (!data.entities || Object.keys(data.entities).length === 0) {
        entitiesContainer.innerHTML = '<p style="color:var(--text-muted); font-size:0.875rem;">No explicit entities mapped.</p>';
        return;
    }

    for (const [key, value] of Object.entries(data.entities)) {
        const confText = data.confidence_scores && data.confidence_scores[key] 
            ? `${Math.round(data.confidence_scores[key] * 100)}% conf`
            : '';

        const html = `
            <div class="entity-row">
                <div>
                    <div class="entity-label">${key.replace('_', ' ')}</div>
                    <div class="entity-val">${value}</div>
                </div>
                <div class="entity-conf" style="color: ${confText.startsWith('9') || confText.startsWith('100') ? 'var(--success)' : 'var(--warning)'}">
                    ${confText}
                </div>
            </div>
        `;
        entitiesContainer.insertAdjacentHTML('beforeend', html);
    }
}
