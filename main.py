import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from utils import draw_mask_on_image, draw_box, save_result

# ==== Configuration ====
IMAGE_DIR = "dataset/images"
YOLO_MODEL_PATH = "yolov11/yolov8n.pt"              # Change if your YOLO model name differs
SAM_CHECKPOINT_PATH = "sam2/sam_vit_h_4b8939.pth"
RESULTS_DIR = "results/"
DEVICE = "cpu"  # <-- Force CPU usage

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== Load YOLOv11 (or v8) model ====
print("[INFO] Loading YOLOv8 model (placeholder for YOLOv11)...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# ==== Load SAM2 Model ====
print("[INFO] Loading SAM2 (Segment Anything) model on CPU...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# ==== Process Each Image ====
for image_name in os.listdir(IMAGE_DIR):
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"[INFO] Processing: {image_name}")
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)

    # YOLO Tumor Detection
    yolo_results = yolo_model(image, device=DEVICE)[0]  # Force YOLO to use CPU
    boxes = yolo_results.boxes

    # Set SAM predictor image
    predictor.set_image(image)

    for i, box_tensor in enumerate(boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = box_tensor.astype(int)
        input_box = np.array([x1, y1, x2, y2])

        # SAM Segmentation
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        best_mask = masks[np.argmax(scores)]

        # Overlay mask and draw box
        output = draw_mask_on_image(image, best_mask)
        output = draw_box(output, (x1, y1, x2, y2))

        # Save result
        output_filename = f"{image_name.split('.')[0]}_seg{i}.png"
        save_result(output, RESULTS_DIR, output_filename)

print("[DONE] All images processed using CPU. Check the results folder.")
