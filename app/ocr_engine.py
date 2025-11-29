import os
import io
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "handwritten")

class OCREngine:
    def __init__(self):
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model is None:
            print(f"Loading model from {MODEL_PATH}...")
            self.processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
            self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
            self.model.to("cpu")
            print("Model loaded successfully.")

    def extract_text(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            return "Error: The image file is missing or corrupt."
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        pixel_values = pixel_values.to("cpu")

        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

# Global instance
engine = OCREngine()

def extract_text(image_path):
    return engine.extract_text(image_path)