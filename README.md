Welcome! This is our Optical Character Recognition Application that runs locally using Microsoft TrOCR models.


OCR-MOSIP — Quick Setup Guidelines

This file contains minimal, essential setup instructions: how to install Python dependencies and obtain the model files required to run the project.

Prerequisites
- Python 3.8+ installed.

Create and activate a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies

```bash
pip install -r [requirements.txt](requirements.txt)
```

Download models

Option A — Automatic (if available):

If `setup_models.py` is present, it can attempt to download the required weights automatically:

```bash
python setup_models.py
```

Option B — Manual download:

- Handwritten model: `microsoft/trocr-base-handwritten` — download the model file (pytorch_model.bin) and place it at `models/handwritten/pytorch_model.bin`.
- Printed model: `microsoft/trocr-base-printed` — download the model file (pytorch_model.bin) and place it at `models/printed/pytorch_model.bin`.

Create the `models/handwritten` and `models/printed` directories if they do not exist before placing the files.

Notes
- The model files are large (~1–1.5 GB each). Download from Hugging Face or your chosen model host.
- After installing dependencies and placing the model files, the application should be able to run locally.