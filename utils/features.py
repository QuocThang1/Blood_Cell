import cv2
import numpy as np

def extract_features(image, mask):
    # Đảm bảo mask là nhị phân
    mask = (mask > 0).astype(np.uint8)

    # Tính toán các đặc trưng hình học đơn giản từ vùng mask
    area = cv2.countNonZero(mask)
    mean_val = np.mean(mask)
    std_val = np.std(mask)

    # Hu Moments: đặc trưng hình dạng
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Đặc trưng màu từ ảnh gốc bên trong vùng mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    mean_color = cv2.mean(masked_img, mask=mask)[:3]  # BGR

    # Ghép tất cả đặc trưng thành 1 vector
    features = [area, mean_val, std_val] + list(hu_moments) + list(mean_color)
    return features
