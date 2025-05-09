import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from utils.preprocessing import preprocess
from utils.features import extract_features

# Load mô hình segmentation và classifier
seg_model = load_model("./models/unet_model.h5")
clf_model = joblib.load("./models/classifier.pkl")

# Đọc ảnh đầu vào
img_path = "./data/imageinput/Neutrophil2.jpg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"❌ Không tìm thấy ảnh: {img_path}")

img_input = preprocess(img)
input_tensor = np.expand_dims(img_input, axis=0)

# Dự đoán mask
print("🧠 Đang dự đoán mask...")
mask_pred = seg_model.predict(input_tensor)[0, :, :, 0]
mask_bin = (mask_pred > 0.5).astype(np.uint8)

# Trích xuất đặc trưng
features = extract_features((img_input * 255).astype(np.uint8), mask_bin)

# Dự đoán nhãn
print("🤖 Đang chẩn đoán loại tế bào...")
pred_label = clf_model.predict([features])[0]
print("🩺 Kết quả chẩn đoán:", pred_label)
