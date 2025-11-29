OCR-MOSIP Project

This project implements an offline OCR system using Microsoft's TrOCR models (Handwritten and Printed).

🚀 Quick Start Guide

1. Clone the Repository

Open your terminal and run:

git clone [https://github.com/Inferno-0/OCR-MOSIP.git](https://github.com/Inferno-0/OCR-MOSIP.git)
cd OCR-MOSIP


2. Set Up Virtual Environment

For Windows:

python -m venv venv
.\venv\Scripts\activate


For Mac/Linux:

python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


🧠 Model Setup (Critical Step)

⚠️ The AI models (~3GB) are NOT included in this repository to keep it light.
You must download them using one of the methods below before the code will run.

Option A: Automatic Setup (Recommended)

If the setup_models.py script is included in the root folder, simply run:

python setup_models.py


This will attempt to download the necessary weights automatically.

Option B: Manual Download

If the script fails or you prefer manual setup, download the model weights from the official Hugging Face repositories:

1. Handwritten Model:

Source: microsoft/trocr-base-handwritten

File to Download: pytorch_model.bin (~1.33 GB)

Action: Download the file and place it in models/handwritten/.

2. Printed Model:

Source: microsoft/trocr-base-printed

File to Download: pytorch_model.bin (~1.33 GB)

Action: Download the file and place it in models/printed/.

📂 Project Structure

IMPORTANT: After downloading, ensure your folder structure looks exactly like this, or the code won't find the models:

OCR-MOSIP/
├── app/
├── static/
├── venv/                 <-- Your local virtual environment
├── requirements.txt
└── models/               <-- CREATE THIS FOLDER
    ├── handwritten/      <-- CREATE THIS SUBFOLDER
    │   ├── config.json
    │   └── pytorch_model.bin  <-- Place 1.33GB file here
    └── printed/          <-- CREATE THIS SUBFOLDER
        ├── config.json
        └── pytorch_model.bin  <-- Place 1.33GB file here


🤝 Collaboration & Git Workflow

To keep our repository clean and avoid conflicts, please follow these rules:

Never push directly to main. The main branch is for production-ready code only.

Work on the dev branch.

Before starting work: git checkout dev then git pull origin dev.

Create Feature Branches.

For every new task, create a branch: git checkout -b feature/my-task-name.

Pushing Code:

git add .

git commit -m "Description of work"

git push origin feature/my-task-name

Merging: Create a Pull Request (PR) on GitHub to merge your feature into dev.

🛠 Troubleshooting

"Module not found" error?
Make sure you activated the virtual environment (venv) before running the code.

"Model not found" error?
Check the models/ folder structure. It must match the diagram above exactly.