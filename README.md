# Synthetic Identity Detector

A web-based deep learning tool that detects whether a face is real or AI-generated. Built using a custom-trained convolutional neural network and an interactive Streamlit dashboard.

##  Overview

This project analyzes facial images to distinguish between authentic human faces and synthetically generated ones. Users can upload single or multiple images and receive real-time predictions with confidence scores. The app also displays results in a summary table with color-coded indicators.

##  Model Architecture

- **Input**: 100x100 RGB face images  
- **Model**: CNN with 3 convolutional layers and dropout  
- **Output**: Binary classification (Real = 1, Fake = 0)  
- **Training Accuracy**: ~63%  
- **Validation Accuracy**: ~73%

##  Folder Structure

synthetic_identity_detector/
├── app.py # Main Streamlit app
├── model/ # Trained .h5 model file
├── assets/ # Background image
├── scripts/ # Training and data loader scripts
├── test_images/ # Example images for testing
├── .gitignore
└── README.md


##  Tech Stack

- Python · TensorFlow · NumPy · PIL  
- Streamlit (frontend + UX)  
- Git/GitHub for version control

##  How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

 Features

    Upload multiple facial images

    Real-time classification and confidence scores

    Batch summary table of all predictions

    Color-coded results for quick review

    Custom background and clean visual layout

✍️ Author

Built by Blake Murray
GitHub: @woodfloor11