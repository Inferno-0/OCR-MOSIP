import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

models_to_download = {
    "handwritten": "microsoft/trocr-base-handwritten",
    "printed": "microsoft/trocr-base-printed"
}

base_path = "./models"

if not os.path.exists(base_path):
    os.makedirs(base_path)

print(f"Starting download to {base_path}...")

for model_type, model_name in models_to_download.items():
    print(f"\nProcessing {model_type} model: {model_name}...")
    
    save_path = os.path.join(base_path, model_type)
    
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        print(f"  - Folder {save_path} is not empty. Skipping download.")
        continue

    print(f"  - Downloading from Hugging Face...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    print(f"  - Saving to {save_path}...")
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

print("\n------------------------------------------------")
print("SUCCESS! Your 'models/' folder is ready.")
print("------------------------------------------------")