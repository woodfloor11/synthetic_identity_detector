import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64
import pandas as pd

# ====== Set Background Image ======
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ====== Load Model ======
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/synthetic_face_classifier.h5")

model = load_model()

# ====== Title and Styling ======
st.set_page_config(page_title="Synthetic Identity Detector", layout="wide")
set_background("assets/background.png")

st.markdown("""
    <h1 style='text-align: center; color: black; font-size: 3.5em;'>Synthetic Identity Detector</h1>
    <p style='text-align: center; color: #222; font-size: 1.1em;'>Upload facial images to determine if they are real or AI-generated</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ====== Image Upload ======
uploaded_files = st.file_uploader("Upload face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ====== Predict Function ======
def preprocess_image(img):
    img = img.resize((100, 100)).convert("RGB")
    return np.array(img) / 255.0

def predict_image(img):
    processed = preprocess_image(img)
    pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, round(float(confidence) * 100, 2)

# ====== Display Results ======
results = []
if uploaded_files:
    st.subheader("Predictions")

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        label, confidence = predict_image(img)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption=uploaded_file.name, use_container_width=True)
        with col2:
            color = "#2ca02c" if label == "Real" else "#d62728"
            st.markdown(f"<h3 style='color:{color}; font-size: 1.8em;'>Prediction: {label} ({confidence}% confidence)</h3>", unsafe_allow_html=True)

        results.append({
            "Filename": uploaded_file.name,
            "Prediction": label,
            "Confidence (%)": confidence
        })

    # ====== Summary Table ======
    st.markdown("---")
    st.subheader("Prediction Summary Table")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

# ====== Footer ======
st.markdown("<p style='text-align: center; color: gray;'>Built by Blake Murray · Powered by TensorFlow · Synthetic Face Classifier</p>", unsafe_allow_html=True)
