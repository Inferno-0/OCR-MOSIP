let originalOCRData = {};

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const placeholderText = document.getElementById('placeholderText');
    const extractBtn = document.getElementById('extractBtn');
    const verifyBtn = document.getElementById('verifyBtn');
    const dataForm = document.getElementById('dataForm');
    
    const formFields = ['name', 'age', 'gender', 'address', 'email', 'phone'];

    // --- File Upload Logic ---
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#764ba2';
        uploadArea.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (file) {
            const objectURL = URL.createObjectURL(file);
            imagePreview.src = objectURL;
            imagePreview.classList.remove('hidden');
            placeholderText.style.display = 'none';
            extractBtn.disabled = false;
        }
    }

    // --- Extract Data Logic ---
    extractBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        extractBtn.disabled = true;
        const originalText = extractBtn.textContent;
        extractBtn.textContent = '⏳ Extracting...';
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/extract', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Extraction failed');

            const data = await response.json();
            originalOCRData = { ...data };

            // Auto-fill form fields
            formFields.forEach(field => {
                const fieldEl = document.getElementById(field);
                if (fieldEl) {
                    fieldEl.value = data[field] || '';
                    fieldEl.classList.remove('low-confidence');
                    const badge = document.querySelector(`.confidence-badge[data-field="${field}"]`);
                    if (badge) badge.textContent = '';
                }
            });

            verifyBtn.disabled = false;
            extractBtn.disabled = false;
            extractBtn.textContent = originalText;

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to extract data. Please try again.');
            extractBtn.disabled = false;
            extractBtn.textContent = originalText;
        }
    });

    // --- Verify Data Logic ---
    verifyBtn.addEventListener('click', async () => {
        const currentData = {};
        formFields.forEach(field => {
            const el = document.getElementById(field);
            if (el) currentData[field] = el.value;
        });

        const payload = {
            extracted: originalOCRData,
            user_input: currentData
        };

        verifyBtn.disabled = true;
        const originalText = verifyBtn.textContent;
        verifyBtn.textContent = '⏳ Verifying...';

        try {
            const response = await fetch('http://localhost:8000/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Verification failed');

            const result = await response.json();
            
            formFields.forEach(field => {
                const inputElement = document.getElementById(field);
                if (!inputElement) return;

                const badge = document.querySelector(`.confidence-badge[data-field="${field}"]`);
                const score = result[field] !== undefined ? result[field] : 100;

                if (badge) {
                    badge.textContent = `Confidence: ${score}%`;
                    badge.classList.toggle('low', score < 80);
                }

                if (score < 80) {
                    inputElement.classList.add('low-confidence');
                } else {
                    inputElement.classList.remove('low-confidence');
                }
            });

            verifyBtn.disabled = false;
            verifyBtn.textContent = originalText;

        } catch (error) {
            console.error('Error:', error);
            alert('Verification failed. Please try again.');
            verifyBtn.disabled = false;
            verifyBtn.textContent = originalText;
        }
    });
});
