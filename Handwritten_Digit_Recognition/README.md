
# Handwritten Digit Recognition Using AI

## Project Overview
This project is an AI-based Handwritten Digit Recognition System developed using Python, TensorFlow, CNN and Streamlit.

It recognises handwritten digits from 0 to 9 using image processing and a Convolutional Neural Network.

## Objectives
- Recognise handwritten digits from 0 to 9.
- Apply image preprocessing techniques.
- Train a CNN model using the MNIST dataset.
- Display digit predictions and confidence scores.
- Develop a simple interactive web application.

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- MNIST Dataset

## Model Information
- Algorithm: Convolutional Neural Network (CNN)
- Input Image Size: 128 × 128 pixels
- Colour Mode: Grayscale
- Output Classes: 10 (Digits 0–9)
- Dataset: MNIST

## Project Structure

Handwritten_Digit_Recognition/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── digit_model.keras

## Installation

Install the required libraries:

pip install -r requirements.txt

## Training the Model

Run:

python train.py

The trained model will be saved as:

digit_model.keras

## Run the Application

Run:

python -m streamlit run app.py

Open the local URL displayed in your terminal.

## How It Works

1. User uploads a handwritten digit image.
2. The image is converted into grayscale.
3. Image preprocessing removes noise and centres the digit.
4. The image is resized to 128 × 128 pixels.
5. The CNN model predicts the digit.
6. The application displays the predicted digit and confidence.

## Expected Output

The application predicts a digit between 0 and 9 and displays its prediction confidence.

## Conclusion

This project demonstrates how Artificial Intelligence and Deep Learning can be used to recognise handwritten digits through image processing and Convolutional Neural Networks.

## Developed By

Student Name: [Sabbir Ahmed]

Class: XII

Subject: Artificial Intelligence

Academic Session: 2026–27

School: [Kendriya Vidyalaya]
