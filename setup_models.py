import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Define the models to download
# CHANGED: Now using the LARGE handwritten model
models_to_download = {
    "handwritten_large": "microsoft/trocr-large-handwritten",
    "printed": "microsoft/trocr-base-printed"
}

# Target the 'models' folder
base_path = "./models"

if not os.path.exists(base_path):
    os.makedirs(base_path)

print(f"Starting download to {base_path}...")

for folder_name, hub_name in models_to_download.items():
    print(f"\nProcessing {folder_name} model: {hub_name}...")
    
    # Define subfolder (e.g., models/handwritten_large)
    save_path = os.path.join(base_path, folder_name)
    
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        print(f"  - Folder {save_path} is not empty. Skipping download.")
        continue

    print(f"  - Downloading from Hugging Face...")
    processor = TrOCRProcessor.from_pretrained(hub_name)
    model = VisionEncoderDecoderModel.from_pretrained(hub_name)
    
    print(f"  - Saving to {save_path}...")
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

print("\n------------------------------------------------")
print("SUCCESS! Large models are ready.")
print("------------------------------------------------")