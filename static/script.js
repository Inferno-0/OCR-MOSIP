let originalOCRText = '';
let currentMode = 'handwritten';

// --- Notification System ---
function showNotification(message, type = 'error') {
    const container = document.getElementById('notificationContainer');
    if (!container) return;

    // Remove any existing notification
    container.innerHTML = '';

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `error-notification ${type}`;

    // Create message text
    const messageText = document.createElement('span');
    messageText.textContent = message;

    // Create close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'close-btn';
    closeBtn.innerHTML = '×';
    closeBtn.setAttribute('aria-label', 'Close notification');

    // Assemble notification
    notification.appendChild(messageText);
    notification.appendChild(closeBtn);
    container.appendChild(notification);

    // Trigger animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    // Auto-dismiss after 8 seconds
    const autoDismissTimer = setTimeout(() => {
        dismissNotification(notification);
    }, 8000);

    // Close button handler
    closeBtn.addEventListener('click', () => {
        clearTimeout(autoDismissTimer);
        dismissNotification(notification);
    });
}

function dismissNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 300); // Wait for animation to complete
}

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const placeholderText = document.getElementById('placeholderText');
    const extractBtn = document.getElementById('extractBtn');
    const verifyBtn = document.getElementById('verifyBtn');
    const dataForm = document.getElementById('dataForm');
    const modeRadios = document.getElementsByName('mode');

    const formFields = ['name', 'age', 'gender', 'address', 'email', 'phone'];

    // --- Mode Selection Logic ---
    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentMode = e.target.value;
            console.log(`Mode changed to: ${currentMode}`);
        });
    });

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
        formData.append('mode', currentMode);  // Add mode to request

        try {
            // FIXED: Changed port from 8000 to 5000
            console.log('[DEBUG] Starting extraction request...');
            const response = await fetch('http://127.0.0.1:5000/extract', {
                method: 'POST',
                body: formData
            });

            console.log('[DEBUG] Response received:', response.status, response.statusText);

            if (!response.ok) {
                console.log('[DEBUG] Response not OK, parsing error...');
                let errorMessage = 'Extraction failed';
                try {
                    const error = await response.json();
                    console.log('[DEBUG] Error response:', error);
                    errorMessage = error.error || errorMessage;
                } catch (parseError) {
                    console.error('[DEBUG] Failed to parse error JSON:', parseError);
                    errorMessage = `Server error (${response.status})`;
                }
                console.log('[DEBUG] Throwing error:', errorMessage);
                throw new Error(errorMessage);
            }

            const data = await response.json();
            console.log('[DEBUG] Success response:', data);
            // Backend returns: {filename: "...", extracted_text: "..."}
            originalOCRText = data.extracted_text || '';

            // Parse the extracted text and auto-fill form fields
            parseAndFillForm(originalOCRText);

            verifyBtn.disabled = false;
            extractBtn.disabled = false;
            extractBtn.textContent = originalText;

            console.log('Extraction successful:', data);

        } catch (error) {
            console.error('[DEBUG] Caught error:', error);
            console.log('[DEBUG] Showing notification...');
            showNotification(error.message, 'error');
            console.log('[DEBUG] Notification shown');
            extractBtn.disabled = false;
            extractBtn.textContent = originalText;
        }
    });

    // --- Parse OCR Text and Fill Form (IMPROVED) ---
    function parseAndFillForm(text) {
        console.log('Parsing OCR text:', text);

        // Field name variations (handles OCR typos)
        const fieldPatterns = {
            name: ['name', 'nam', 'nane', 'full name', 'fullname'],
            age: ['age', 'agi', 'agei', 'ag'],
            gender: ['gender', 'gendor', 'gendar', 'sex'],
            email: ['email', 'emaili', 'e-mail', 'mail', 'e mail'],
            phone: ['phone', 'phonei', 'phon', 'mobile', 'contact', 'tel'],
            address: ['address', 'addres', 'addr', 'location']
        };

        formFields.forEach(field => {
            const fieldEl = document.getElementById(field);
            if (!fieldEl) return;

            let foundValue = '';
            const patterns = fieldPatterns[field] || [field];

            // Try each pattern variation
            for (const pattern of patterns) {
                // Try different regex patterns
                const regexes = [
                    new RegExp(`${pattern}\\s*[:\\s]+\\s*([^\\n]+)`, 'i'),  // "Name: Value"
                    new RegExp(`${pattern}\\s+([^\\n:]+)`, 'i'),             // "Name Value"
                    new RegExp(`${pattern}[:\\s]*([^\\n]+)`, 'i')            // "Name:Value" or "NameValue"
                ];

                for (const regex of regexes) {
                    const match = text.match(regex);
                    if (match && match[1]) {
                        let value = match[1].trim();

                        // Clean up common OCR errors
                        value = value
                            .replace(/\s+\./g, '')          // Remove trailing " ."
                            .replace(/^\./g, '')            // Remove leading "."
                            .replace(/\s+/g, ' ')           // Normalize spaces
                            .replace(/[|]/g, '1')           // Fix | to 1
                            .trim();

                        // Field-specific cleaning
                        if (field === 'name') {
                            // Fix OCR misreading ":" as "'s" (e.g., "Name's Kanchan" -> "Kanchan")
                            // Safe fix: no legitimate form value starts with "'s "
                            value = value.replace(/^['']s\s+/i, '').trim();
                        }
                        if (field === 'phone') {
                            value = value.replace(/[^0-9]/g, ''); // Keep only digits
                        }
                        if (field === 'email') {
                            value = value.replace(/\s/g, '');     // Remove spaces
                        }


                        if (value && value.length > 0) {
                            foundValue = value;
                            break;
                        }
                    }
                }
                if (foundValue) break;
            }

            fieldEl.value = foundValue;

            // Reset confidence styling
            fieldEl.classList.remove('low-confidence');
            const badge = document.querySelector(`.confidence-badge[data-field="${field}"]`);
            if (badge) badge.textContent = '';
        });

        console.log('Form populated');
    }

    // --- Verify Data Logic ---
    verifyBtn.addEventListener('click', async () => {
        const currentData = {};
        formFields.forEach(field => {
            const el = document.getElementById(field);
            if (el) currentData[field] = el.value;
        });

        // FIXED: Changed payload format to match backend
        const payload = {
            ocr_text: originalOCRText,      // Backend expects "ocr_text"
            form_data: currentData           // Backend expects "form_data"
        };

        verifyBtn.disabled = true;
        const originalText = verifyBtn.textContent;
        verifyBtn.textContent = '⏳ Verifying...';

        try {
            // FIXED: Changed port from 8000 to 5000
            const response = await fetch('http://127.0.0.1:5000/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Verification failed');
            }

            const result = await response.json();
            // Backend returns: {total_score: 95, field_matches: {...}, status: "verified"}

            console.log('Verification result:', result);

            // FIXED: Read from result.field_matches instead of result directly
            formFields.forEach(field => {
                const inputElement = document.getElementById(field);
                if (!inputElement) return;

                const badge = document.querySelector(`.confidence-badge[data-field="${field}"]`);
                const score = result.field_matches && result.field_matches[field] !== undefined
                    ? result.field_matches[field]
                    : 0;

                if (badge) {
                    badge.textContent = `Match: ${score}%`;
                    badge.classList.toggle('low', score < 80);
                }

                if (score < 80) {
                    inputElement.classList.add('low-confidence');
                } else {
                    inputElement.classList.remove('low-confidence');
                }
            });

            // Show overall status
            const statusMessage = result.status === 'verified'
                ? `✅ Verified (${result.total_score}% match)`
                : `⚠️ Review Required (${result.total_score}% match)`;

            const notificationType = result.status === 'verified' ? 'success' : 'info';
            showNotification(statusMessage, notificationType);

            verifyBtn.disabled = false;
            verifyBtn.textContent = originalText;

        } catch (error) {
            console.error('Error:', error);
            showNotification(`Verification failed: ${error.message}`, 'error');
            verifyBtn.disabled = false;
            verifyBtn.textContent = originalText;
        }
    });

    // --- Clear Form Logic ---
    dataForm.addEventListener('reset', () => {
        // Clear all confidence badges
        document.querySelectorAll('.confidence-badge').forEach(badge => {
            badge.textContent = '';
        });

        // Remove low-confidence styling
        formFields.forEach(field => {
            const el = document.getElementById(field);
            if (el) el.classList.remove('low-confidence');
        });

        // Reset OCR data
        originalOCRText = '';

        // Disable verify button
        verifyBtn.disabled = true;

        console.log('Form cleared');
    });
});
