import os
import sys

# --- SETUP: Allow importing from the 'app' folder ---
# This line tells Python to look inside 'app/' for the engine
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.ocr_engine import extract_text

# --- CONFIGURATION: Select your images here ---
# You can list one file, or multiple files.
# Make sure these images are in your project root folder (ocr-project/)
selected_images = [
    "test_image8.jpg",
    # "another_image.png",  <-- You can uncomment these to run more
    # "form_sample.jpg"
]

# --- EXECUTION ---
for image_filename in selected_images:
    print(f"\n========================================")
    
    if not os.path.exists(image_filename):
        print(f"❌ Error: I cannot find '{image_filename}' in the project root.")
        continue

    print(f"📸 Testing OCR on: {image_filename}")
    print("⏳ Sending image to the Engine...")
    
    try:
        # Note: You can change mode="printed" if testing printed forms
        test_result = extract_text(image_filename, mode="handwritten")
        
        print("\n--- RESULT ---")
        print(test_result)
        print("--------------")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")