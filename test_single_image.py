import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model("model/synthetic_face_classifier.h5")

# Load a test image (replace path if needed)
image_path = "real_vs_fake/real-vs-fake/test/fake/0A266M95TD.jpg"
img = Image.open(image_path).resize((100, 100)).convert("RGB")
img_array = np.array(img) / 255.0

# Predict
pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
label = "Real" if pred > 0.5 else "Fake"
confidence = pred if pred > 0.5 else 1 - pred

print(f"Prediction: {label}")
print(f"Confidence: {round(confidence * 100, 2)}%")
print(f"Raw Score: {pred}")
