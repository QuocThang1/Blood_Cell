import cv2
import numpy as np

def preprocess(img, target_size=(256, 256)):
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized / 255.0
    return img_norm