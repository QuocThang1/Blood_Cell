import os
import cv2
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from utils.features import extract_features  # Đảm bảo bạn đã có file này

# Đọc file class labels
df = pd.read_csv("FinalProjectTest4/data/class labels.csv")

X_feat = []
y = []

print("📂 Bắt đầu xử lý ảnh...")

for idx, row in df.iterrows():
    raw_id = row["Image ID"]
    label = row["Categoriy"]

    image_id = raw_id.lower()
    mask_id = "mask" + raw_id[5:]

    image_path = os.path.join("FinalProjectTest4/data/image", image_id + ".jpg")
    mask_path = os.path.join("FinalProjectTest4/data/mask", mask_id + ".jpg")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"⚠️ Thiếu file: {image_path} hoặc {mask_path}")
        continue

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    if image is None or mask is None:
        print(f"⚠️ Lỗi khi đọc ảnh/mask: {image_id}")
        continue

    print(f"🔍 Trích xuất đặc trưng từ: {image_id}, {mask_id}")

    try:
        feature_vector = extract_features(image, mask)  # Trích xuất đặc trưng thật sự
        X_feat.append(feature_vector)
        y.append(label)
    except Exception as e:
        print(f"❌ Lỗi extract_features: {e}")
        continue

if len(X_feat) == 0:
    raise ValueError("❌ Không có đặc trưng nào được trích xuất!")

# Chia tập train/test
print("📊 Chia dữ liệu train/test...")
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
print("🧠 Huấn luyện mô hình Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Huấn luyện xong!")

# Dự đoán và đánh giá
print("🧪 Đánh giá mô hình trên tập test...")
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Lưu model
save_path = "FinalProjectTest4/models/classifier.pkl"
joblib.dump(model, save_path)
print(f"💾 Mô hình đã được lưu tại: {save_path}")
