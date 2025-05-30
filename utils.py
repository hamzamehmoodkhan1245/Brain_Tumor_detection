import cv2
import numpy as np
import os


def draw_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    """Draws a semi-transparent mask on the image."""
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_box(image, box, color=(255, 0, 0), thickness=2):
    """Draws a bounding box on the image."""
    x1, y1, x2, y2 = box
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)


def save_result(image, output_dir, filename):
    """Saves the processed image to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    print(f"[INFO] Result saved: {path}")
