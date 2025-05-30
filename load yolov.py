from ultralytics import YOLO

# Download the model once
model = YOLO("yolov11/yolov8n.pt")  # Automatically downloads to cache
