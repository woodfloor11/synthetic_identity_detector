import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import os

# ====== Load Data ======
print("🔁 Loading data...")
X = np.load("data/X.npy") / 255.0
y = np.load("data/y.npy")

print("✅ Data loaded:", X.shape, y.shape)

# ====== Train/Val Split ======
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ====== Data Augmentation ======
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# ====== Build Model ======
model = models.Sequential([
    layers.Input(shape=(100, 100, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ====== Callbacks ======
early_stop = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint("model/synthetic_face_classifier.h5", save_best_only=True)

# ====== Train ======
print("🚀 Training model...")
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

print("✅ Training complete — best model saved to model/synthetic_face_classifier.h5")
