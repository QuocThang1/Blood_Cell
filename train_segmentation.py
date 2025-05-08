import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.preprocessing import preprocess  # đảm bảo bạn có file này

# U-Net đơn giản
def build_unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    b = Conv2D(64, 3, activation='relu', padding='same')(p2)
    u1 = UpSampling2D()(b)
    m1 = concatenate([u1, c2])
    c3 = Conv2D(32, 3, activation='relu', padding='same')(m1)
    u2 = UpSampling2D()(c3)
    m2 = concatenate([u2, c1])
    c4 = Conv2D(16, 3, activation='relu', padding='same')(m2)
    output = Conv2D(1, 1, activation='sigmoid')(c4)
    model = Model(inputs, output)
    return model

# Load ảnh & mask
X, Y = [], []
img_dir = "./data/image"
mask_dir = "./data/mask"

for filename in os.listdir(img_dir):
    if filename.lower().endswith(".jpg") and filename.startswith("image"):
        image_path = os.path.join(img_dir, filename)
        
        # Ghép tên mask tương ứng: image1.jpg -> mask1.jpg
        mask_filename = filename.replace("image", "mask")
        mask_path = os.path.join(mask_dir, mask_filename)

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)

        if img is not None and mask is not None:
            img = preprocess(img)  # resize, normalize, etc. (nếu bạn có)
            mask = cv2.resize(mask, (256, 256)) / 255.0
            X.append(img)
            Y.append(mask)
        else:
            print(f"⚠️ Bỏ qua: {filename} (ảnh hoặc mask không tồn tại)")

X = np.array(X)
Y = np.expand_dims(np.array(Y), axis=-1)

print(f"Tổng ảnh huấn luyện: {X.shape[0]}")

# Train model
model = build_unet()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size=8, epochs=20, validation_split=0.1,
          callbacks=[ModelCheckpoint("./models/unet_model.h5", save_best_only=True)])

