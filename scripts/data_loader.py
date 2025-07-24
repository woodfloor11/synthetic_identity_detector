import os
import numpy as np
from PIL import Image

base_path = "real_vs_fake/real-vs-fake"
subsets = ['train', 'valid', 'test']
X = []
y = []

for subset in subsets:
    for label in ['real', 'fake']:
        folder = os.path.join(base_path, subset, label)
        print(f"Processing folder: {folder}")
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = Image.open(img_path).resize((100, 100)).convert("RGB")
                X.append(np.array(img))
                y.append(1 if label == "real" else 0)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

X = np.array(X)
y = np.array(y)

os.makedirs("data", exist_ok=True)
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("✅ Finished loading")
print("🖼️ Total images:", len(X))
print("📐 Image shape:", X.shape[1:])
print("🏷️ Labels shape:", y.shape)
