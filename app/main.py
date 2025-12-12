from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
from werkzeug.utils import secure_filename
from ocr_engine import extract_text
from utils import verify_form_data
import tempfile

app = Flask(__name__)

# CORS Configuration - Allow Live Server and localhost
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"],
    }
})

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pdf'}
import fitz # PyMuPDF

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'OCR Backend is running',
        'version': '6.3'
    }), 200

@app.route('/api/ocr/upload', methods=['POST'])
def upload_and_process():
    """
    Upload and process image for OCR
    Accepts: multipart/form-data with 'file' and optional 'mode' (handwritten/printed)
    Returns: JSON with extracted text and metadata
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Get OCR mode (default: handwritten)
        mode = request.form.get('mode', 'handwritten')
        if mode not in ['handwritten', 'printed']:
            return jsonify({'error': 'Invalid mode. Use "handwritten" or "printed"'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Process with OCR engine
            print(f"Processing {filename} in {mode} mode...")
            
            # PDF Handling
            actual_image_path = temp_path
            is_pdf = filename.lower().endswith('.pdf')
            temp_png_path = None

            if is_pdf:
                try:
                    print(f"   📄 Converting PDF to Image...")
                    doc = fitz.open(temp_path)
                    page = doc.load_page(0) # Get first page
                    pix = page.get_pixmap(dpi=300) # High quality
                    temp_png_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_page0.png")
                    pix.save(temp_png_path)
                    doc.close()
                    actual_image_path = temp_png_path
                    print(f"   ✅ Converted to {temp_png_path}")
                except Exception as pdf_err:
                    print(f"   ❌ PDF Conversion Failed: {pdf_err}")
                    raise pdf_err

            extracted_text = extract_text(actual_image_path, mode=mode)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_png_path and os.path.exists(temp_png_path):
                os.remove(temp_png_path)
            
            return jsonify({
                'success': True,
                'text': extracted_text,
                'mode': mode,
                'filename': filename
            }), 200
            
        except Exception as ocr_error:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_png_path and os.path.exists(temp_png_path):
                os.remove(temp_png_path)
            raise ocr_error
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ocr/process-base64', methods=['POST'])
def process_base64():
    """
    Process base64 encoded image
    Accepts: JSON with 'image' (base64 string) and optional 'mode'
    Returns: JSON with extracted text
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:  # Remove data URL prefix if present
            image_data = image_data.split(',')[1]
        
        mode = data.get('mode', 'handwritten')
        if mode not in ['handwritten', 'printed']:
            return jsonify({'error': 'Invalid mode'}), 400
        
        # Save to temp file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_ocr_image.jpg')
        
        with open(temp_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        try:
            # Process with OCR
            extracted_text = extract_text(temp_path, mode=mode)
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'text': extracted_text,
                'mode': mode
            }), 200
            
        except Exception as ocr_error:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ocr_error
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/extract', methods=['POST'])
def extract():
    """
    Text extraction endpoint (spec-compliant)
    Route: POST /extract
    Content-Type: multipart/form-data
    Parameters: file (image), mode (handwritten|printed)
    Returns: {"filename": "...", "extracted_text": "..."}
    """
    try:
        # Validate file presence
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Get mode (default: handwritten)
        mode = request.form.get('mode', 'handwritten')
        if mode not in ['handwritten', 'printed']:
            return jsonify({'error': 'Invalid mode. Use "handwritten" or "printed"'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Process with OCR engine
            print(f"[/extract] Processing {filename} in {mode} mode...")
            
            # PDF Handling
            actual_image_path = temp_path
            is_pdf = filename.lower().endswith('.pdf')
            temp_png_path = None

            if is_pdf:
                try:
                    print(f"   📄 Converting PDF to Image...")
                    doc = fitz.open(temp_path)
                    page = doc.load_page(0) # Get first page
                    pix = page.get_pixmap(dpi=300) # High quality
                    temp_png_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_page0.png")
                    pix.save(temp_png_path)
                    doc.close()
                    actual_image_path = temp_png_path
                    print(f"   ✅ Converted to {temp_png_path}")
                except Exception as pdf_err:
                    print(f"   ❌ PDF Conversion Failed: {pdf_err}")
                    raise pdf_err

            extracted_text = extract_text(actual_image_path, mode=mode)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_png_path and os.path.exists(temp_png_path):
                os.remove(temp_png_path)
            
            # SPEC-COMPLIANT RESPONSE FORMAT
            return jsonify({
                'filename': filename,
                'extracted_text': extracted_text
            }), 200
            
        except Exception as ocr_error:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_png_path and os.path.exists(temp_png_path):
                os.remove(temp_png_path)
            raise ocr_error
            
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/verify', methods=['POST'])
def verify():
    """
    Data verification endpoint using fuzzy matching
    Route: POST /verify
    Content-Type: application/json
    Payload: {"ocr_text": "...", "form_data": {...}}
    Returns: {"total_score": int, "field_matches": {...}, "status": "..."}
    """
    try:
        # Get JSON payload
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'ocr_text' not in data:
            return jsonify({'error': 'Missing ocr_text field'}), 400
        
        if 'form_data' not in data:
            return jsonify({'error': 'Missing form_data field'}), 400
        
        ocr_text = data['ocr_text']
        form_data = data['form_data']
        
        # Validate types
        if not isinstance(ocr_text, str):
            return jsonify({'error': 'ocr_text must be a string'}), 400
        
        if not isinstance(form_data, dict):
            return jsonify({'error': 'form_data must be an object'}), 400
        
        # Perform verification
        print(f"[/verify] Verifying {len(form_data)} fields against OCR text")
        result = verify_form_data(ocr_text, form_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the frontend application"""
    return send_from_directory('../static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (script.js, style.css)"""
    return send_from_directory('../static', filename)

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'OCR-MOSIP Backend',
        'version': '6.3',
        'endpoints': {
            'health': '/health',
            'extract': '/extract (POST) - Spec-compliant',
            'verify': '/verify (POST) - Spec-compliant',
            'upload': '/api/ocr/upload (POST) - Legacy',
            'base64': '/api/ocr/process-base64 (POST)'
        },
        'features': [
            'Adaptive Beam Search (GPU: 5 beams, CPU: 2 beams)',
            'GPU Support with CPU Fallback',
            'Adaptive Preprocessing (CLAHE, Denoising, Sharpening)',
            'Handwritten & Printed Text Support',
            'PDF Support (First Page Conversion)'
        ]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting OCR-MOSIP Backend Server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("📖 API Documentation: http://localhost:5000/")
    print("💚 Health Check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
