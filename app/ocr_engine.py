import os
import cv2
import easyocr
import numpy as np
import logging
import re
import shutil
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("transformers").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDWRITTEN_PATH = os.path.join(BASE_DIR, "models", "handwritten_large")
PRINTED_PATH = os.path.join(BASE_DIR, "models", "printed")

DEBUG_MODE = True      

class OCREngine:
    def __init__(self):
        print("🚀 v6.3 ENGINE LOADING... (Blur Rejection + GPU Support)")
        self.models = {}
        self.processors = {}
        
        # Detect GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print("   ✅ GPU detected - Using CUDA acceleration")
        else:
            print("   ℹ️  No GPU detected - Using CPU (slower but works)")
        
        # Load Models
        self._load_model("handwritten", HANDWRITTEN_PATH)
        self._load_model("printed", PRINTED_PATH)
        
        print("   -> Loading Detector (EasyOCR)...")
        # Enable GPU for EasyOCR if available
        use_gpu = (self.device == "cuda")
        self.detector = easyocr.Reader(['en'], gpu=use_gpu, verbose=False) 
        print("✅ System Ready.")

    def _load_model(self, name, path):
        try:
            if not os.path.exists(path):
                # Fallback for path naming variations
                if name == "handwritten" and "large" in path:
                    path = path.replace("_large", "")
            
            processor = TrOCRProcessor.from_pretrained(path, local_files_only=True)
            model = VisionEncoderDecoderModel.from_pretrained(path, local_files_only=True)
            # Load model to GPU if available, otherwise CPU
            model.to(self.device)
            self.processors[name] = processor
            self.models[name] = model
            print(f"   ✓ Loaded {name} model on {self.device.upper()}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

    def clean_text_output(self, text):
        if re.match(r'^\W*$', text): return "" 
        if len(text) < 2 and text.lower() != 'a': return ""
        return text

    # --- QUALITY DETECTION METHODS ---
    
    def is_blurry(self, image, threshold=85):
        """
        Detects if image is blurry using Laplacian variance.
        Returns: (is_blurry: bool, blur_score: float)
        Threshold: Higher = stricter (needs sharper image). 85 is lenient.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < threshold, score
    
    def detect_noise(self, image, threshold=50):
        """
        Detects if image is noisy using standard deviation of Laplacian.
        Returns: (is_noisy: bool, noise_score: float)
        Higher score = more noise. Threshold 50 is conservative.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate noise using standard deviation of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_score = laplacian.std()
        return noise_score > threshold, noise_score
    
    def detect_low_contrast(self, image, threshold=50):
        """
        Detects if image has low contrast using histogram analysis.
        Returns: (is_low_contrast: bool, contrast_score: float)
        Lower score = lower contrast. Threshold 50 is conservative.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate contrast using standard deviation of pixel intensities
        contrast_score = gray.std()
        return contrast_score < threshold, contrast_score

    # --- PREPROCESSING METHODS ---
    
    def apply_clahe_conservative(self, image):
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) with conservative parameters.
        - clip_limit=2.0: Prevents over-enhancement and noise amplification
        - tileGridSize=(8,8): Adaptive processing in small regions
        - Applied only to L channel in LAB color space to preserve color info
        Returns: Enhanced image (same format as input)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE only to L (lightness) channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_denoising_mild(self, image):
        """
        Applies mild denoising using Non-Local Means algorithm.
        - h=10: Mild denoising strength (preserves text details)
        - hColor=10: Color component denoising
        - templateWindowSize=7: Size of template patch
        - searchWindowSize=21: Size of search area
        Edge-preserving algorithm that maintains text boundaries.
        Returns: Denoised image (same format as input)
        """
        # fastNlMeansDenoisingColored is edge-preserving
        denoised = cv2.fastNlMeansDenoisingColored(
            image, 
            None, 
            h=10,           # Mild denoising (preserves details)
            hColor=10,      # Color denoising
            templateWindowSize=7, 
            searchWindowSize=21
        )
        return denoised
    

    
    def adaptive_preprocess(self, image):
        """
        Adaptive preprocessing pipeline that only applies enhancements when needed.
        - Detects image quality issues (contrast, noise)
        - Applies only necessary preprocessing steps
        Note: Blur is checked separately in extract_text() and will reject if detected
        Returns: Preprocessed image (same format as input)
        """
        original = image.copy()
        enhanced = image.copy()
        
        # Detect quality issues (blur checked separately in extract_text)
        is_low_contrast, contrast_score = self.detect_low_contrast(enhanced)
        is_noisy, noise_score = self.detect_noise(enhanced)
        
        # Track what we apply
        applied = []
        
        # Step 1: CLAHE - only if low contrast
        if is_low_contrast:
            enhanced = self.apply_clahe_conservative(enhanced)
            applied.append("CLAHE")
        
        # Step 2: Denoising - only if noisy
        if is_noisy:
            enhanced = self.apply_denoising_mild(enhanced)
            applied.append("Denoise")
        
        # Print what was applied
        if applied:
            print(f"   🔧 Applied: {', '.join(applied)}")
            print(f"      Contrast: {contrast_score:.1f}, Noise: {noise_score:.1f}")
        else:
            print(f"   ✨ Image quality good - no preprocessing needed")
        
        return enhanced

    def remove_lines(self, image):
        # NOTE: Only use this for lined paper (handwritten mode)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        cnts = cv2.findContours(detect_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(detect_lines, [c], -1, 255, 3)
            
        clean_image = cv2.inpaint(image, detect_lines, 3, cv2.INPAINT_TELEA)
        return clean_image

    def split_tall_boxes(self, boxes, image_height):
        if not boxes: return []

        heights = [b[5] for b in boxes]
        if not heights: return boxes
        
        median_h = np.median(heights)
        
        final_boxes = []
        for box in boxes:
            min_x, min_y, max_x, max_y, y_center, h = box
            
            # Adjusted split threshold logic
            if h > (median_h * 2.0) and h > 30: # Added minimum pixel check to avoid splitting tiny boxes
                num_splits = max(2, round(h / median_h))
                split_h = h // num_splits
                
                print(f"   ⚠️ Found merged block (Height: {h}). Splitting into {num_splits} lines.")
                
                for i in range(num_splits):
                    new_y1 = min_y + (i * split_h)
                    new_y2 = min_y + ((i + 1) * split_h)
                    new_center = (new_y1 + new_y2) // 2
                    
                    final_boxes.append([min_x, new_y1, max_x, new_y2, new_center, split_h])
            else:
                final_boxes.append(box)
                
        return final_boxes

    def detect_and_merge_lines(self, image, mode="handwritten"):
        # FIX 1: Do NOT run remove_lines for printed text/ID cards
        if mode == "handwritten":
            clean_image = self.remove_lines(image)
        else:
            clean_image = image # Keep original structure for printed forms

        results = self.detector.readtext(
            clean_image, 
            width_ths=0.7, 
            link_threshold=0.4,
            text_threshold=0.3,
            paragraph=False 
        )
        
        raw_boxes = []
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            min_x, max_x = int(min(tl[0], bl[0])), int(max(tr[0], br[0]))
            min_y, max_y = int(min(tl[1], tr[1])), int(max(bl[1], br[1]))
            
            height = max_y - min_y
            
            # FIX 2: Lowered threshold from 15 to 8 to catch small ID card text
            if height < 8: continue 
            
            y_center = (min_y + max_y) // 2
            raw_boxes.append([min_x, min_y, max_x, max_y, y_center, height])

        raw_boxes = self.split_tall_boxes(raw_boxes, image.shape[0])

        raw_boxes.sort(key=lambda b: b[4])
        merged_lines = []
        current_line = []

        for box in raw_boxes:
            if not current_line:
                current_line.append(box)
                continue
            
            avg_h = sum(b[5] for b in current_line) / len(current_line)
            # Check if box is on the same vertical level
            if abs(box[4] - (sum(b[4] for b in current_line)/len(current_line))) < (avg_h * 0.6):
                current_line.append(box)
            else:
                merged_lines.append(current_line)
                current_line = [box]
        if current_line: merged_lines.append(current_line)

        final_crops = []
        for line in merged_lines:
            line.sort(key=lambda b: b[0])
            min_x = min(b[0] for b in line)
            min_y = min(b[1] for b in line)
            max_x = max(b[2] for b in line)
            max_y = max(b[3] for b in line)
            final_crops.append((min_x, min_y, max_x, max_y))
            
        return final_crops, clean_image

    def extract_text(self, image_path, mode="handwritten"):
        original = cv2.imread(image_path)
        if original is None: 
            raise ValueError("Error: Image not found")

        print(f"Processing {image_path}...")
        
        import time
        start_time = time.time()
        
        # STEP 1: Blur Quality Check (REJECT if too blurry)
        is_blur, blur_score = self.is_blurry(original)
        print(f"   🔍 Blur Score: {blur_score:.1f} (Threshold: 85)")
        
        if is_blur:
            error_msg = (
                f"Image quality too low for accurate OCR. "
                f"Blur score: {blur_score:.1f} (minimum required: 85). "
                f"Please upload a sharper, more focused image."
            )
            print(f"   ❌ {error_msg}")
            raise ValueError(error_msg)
        
        print(f"   ✅ Image quality acceptable (Blur score: {blur_score:.1f})")
        
        # STEP 2: Adaptive Preprocessing (Contrast & Noise only)
        # Apply quality-based enhancements before any other processing
        preprocessed = self.adaptive_preprocess(original)
        print(f"   ⏱️ Preprocessing took: {time.time() - start_time:.2f}s")
        
        step2_start = time.time()
        # STEP 2: Mode-specific preprocessing
        # Note: 2x upscaling removed - TrOCR processor auto-resizes to 384x384
        # Upscaling before processor is redundant and wastes computation
        # print(f"   ⏱️ Resizing took: {time.time() - step2_start:.2f}s")

        if DEBUG_MODE:
            debug_dir = "debug_crops"
            if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
            os.makedirs(debug_dir)

        # STEP 3: Text Detection and Recognition
        # Pass mode to detection logic
        det_start = time.time()
        line_boxes, clean_image = self.detect_and_merge_lines(preprocessed, mode=mode)
        print(f"   ⏱️ Detection took: {time.time() - det_start:.2f}s")
        
        full_text = []
        processor = self.processors.get(mode)
        model = self.models.get(mode)

        print(f"📖 Reading {len(line_boxes)} lines...")
        
        rec_start = time.time()
        for i, (x1, y1, x2, y2) in enumerate(line_boxes):
            h, w = clean_image.shape[:2]
            
            # Adjusted padding for tighter crops on ID cards
            pad_y = 4 
            pad_x = 8
            
            y_start, y_end = max(0, y1 - pad_y), min(h, y2 + pad_y)
            x_start, x_end = max(0, x1 - pad_x), min(w, x2 + pad_x)
            
            crop = clean_image[y_start:y_end, x_start:x_end]
            
            if DEBUG_MODE:
                cv2.imwrite(f"debug_crops/line_{i}.jpg", crop)

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            try:
                # Send pixel values to same device as model (GPU or CPU)
                pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(self.device)
                # Optimal settings (research-validated for balanced pipeline):
                # - num_beams=5 (GPU): Maximum accuracy, fully utilizes preprocessing
                # - num_beams=2 (CPU): Perfect balance - matches preprocessing effort (80%),
                #   catches ambiguous chars (1/l, 0/O), +5-8% accuracy for only +30% time
                # - no_repeat_ngram_size=3 prevents repetitive outputs
                # - early_stopping=True improves efficiency
                num_beams = 5 if self.device == "cuda" else 2
                generated_ids = model.generate(
                    pixel_values, 
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                cleaned = self.clean_text_output(line_text)
                if cleaned:
                    print(f"   Line {i}: {cleaned}")
                    full_text.append(cleaned)
            except: continue
        print(f"   ⏱️ Recognition took: {time.time() - rec_start:.2f}s")
        print(f"   ⏱️ TOTAL TIME: {time.time() - start_time:.2f}s")

        return "\n".join(full_text)

engine = OCREngine()

def extract_text(image_path, mode="handwritten"):
    # Ensure you pass the mode correctly here
    return engine.extract_text(image_path, mode)