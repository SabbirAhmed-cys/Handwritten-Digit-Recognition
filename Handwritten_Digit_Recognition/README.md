# DigitLab — Handwritten Digit Recognition Studio

A redesigned Streamlit application for your Class XII Artificial Intelligence project. It uses your trained TensorFlow/Keras CNN to recognise handwritten digits from 0 to 9.

## Features

- Light, responsive dashboard with a clean blue visual theme
- Upload PNG, JPG, JPEG and WEBP images
- Draw a digit directly on an interactive canvas
- CNN prediction with confidence score
- Probability chart for all ten digit classes
- Original/processed image workflow
- Session-based prediction history
- Analytics dashboard with digit frequency and confidence trends
- CSV export for history and analytics
- Custom cursor effect on desktop devices

## Project structure

```text
Handwritten_Digit_Recognition/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── digit_model.keras
```

Keep your existing `digit_model.keras` in the project root beside `app.py`.

## Setup (Windows)

1. Install Python 3.10 or 3.11 (recommended for TensorFlow compatibility).
2. Open Command Prompt or PowerShell in this folder.
3. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the website:

```bash
python -m streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

## Model

- Architecture: Convolutional Neural Network (CNN)
- Input: 128 × 128 × 1 grayscale image
- Output: 10 classes (0–9)
- Dataset: MNIST
- Framework: TensorFlow / Keras

The upgraded app keeps the existing `digit_model.keras`. You do not need to retrain it to use the new interface.

## Retrain (optional)

If you want to train a fresh model using the included training script:

```bash
python train.py
```

The script saves the trained model as `digit_model.keras`.

## Important notes

- Draw one digit at a time for the best results.
- Prediction history is kept in Streamlit session state and is not a permanent database.
- Confidence is the model's output score, not a guarantee of correctness.
- The custom cursor is a decorative desktop effect; touch devices use their normal interaction behavior.

## Developed by

**Sabbir Ahmed**  
Class XII · Artificial Intelligence · Academic Session 2026–27
