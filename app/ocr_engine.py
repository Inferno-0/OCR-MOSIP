import os
import cv2
import easyocr
import numpy as np
import logging
import re
import shutil
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("transformers").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDWRITTEN_PATH = os.path.join(BASE_DIR, "models", "handwritten_large")
PRINTED_PATH = os.path.join(BASE_DIR, "models", "printed")

DEBUG_MODE = True      

class OCREngine:
    def __init__(self):
        print("🚀 v6.0 ENGINE LOADING... (Block Splitter Mode)")
        self.models = {}
        self.processors = {}
        
        self._load_model("handwritten", HANDWRITTEN_PATH)
        self._load_model("printed", PRINTED_PATH)
        
        print("   -> Loading Detector (EasyOCR)...")
        self.detector = easyocr.Reader(['en'], gpu=False, verbose=False) 
        print("✅ System Ready.")

    def _load_model(self, name, path):
        try:
            if not os.path.exists(path):
                if name == "handwritten" and "large" in path:
                    path = path.replace("_large", "")
            
            processor = TrOCRProcessor.from_pretrained(path, local_files_only=True)
            model = VisionEncoderDecoderModel.from_pretrained(path, local_files_only=True)
            model.to("cpu")
            self.processors[name] = processor
            self.models[name] = model
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

    def clean_text_output(self, text):
        if re.match(r'^\W*$', text): return "" 
        if len(text) < 2 and text.lower() != 'a': return ""
        return text

    def remove_lines(self, image):
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
        median_h = np.median(heights)
        
        final_boxes = []
        for box in boxes:
            min_x, min_y, max_x, max_y, y_center, h = box
            
            if h > (median_h * 1.8):
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

    def detect_and_merge_lines(self, image):
        clean_image = self.remove_lines(image)
        
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
            if height < 15: continue
            
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
            
        return final_crops

    def extract_text(self, image_path, mode="handwritten"):
        original = cv2.imread(image_path)
        if original is None: return "Error: Image not found"

        if DEBUG_MODE:
            debug_dir = "debug_crops"
            if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
            os.makedirs(debug_dir)

        print(f"🔍 Processing {image_path}...")
        
        clean_image = self.remove_lines(original)
        line_boxes = self.detect_and_merge_lines(original)
        
        full_text = []
        processor = self.processors.get(mode)
        model = self.models.get(mode)

        print(f"📖 Reading {len(line_boxes)} lines...")
        
        for i, (x1, y1, x2, y2) in enumerate(line_boxes):
            h, w = clean_image.shape[:2]
           
            pad_y = 2 
            pad_x = 10
            
            y_start, y_end = max(0, y1 - pad_y), min(h, y2 + pad_y)
            x_start, x_end = max(0, x1 - pad_x), min(w, x2 + pad_x)
            
            crop = clean_image[y_start:y_end, x_start:x_end]
            
            if DEBUG_MODE:
                cv2.imwrite(f"debug_crops/line_{i}_clean.jpg", crop)

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            try:
                pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to("cpu")
                generated_ids = model.generate(pixel_values, num_beams=1)
                line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                cleaned = self.clean_text_output(line_text)
                if cleaned:
                    print(f"   Line {i}: {cleaned}")
                    full_text.append(cleaned)
            except: continue

        return "\n".join(full_text)

engine = OCREngine()

def extract_text(image_path, mode="handwritten"):
    return engine.extract_text(image_path, mode)