import os
from ultralytics import YOLO

# Load a model
model = YOLO("drone.pt")  # or "yolov8n.pt" for a pretrained model

# Full path to image
image_path = os.path.join("images", "drone4.jpg")

# Run inference
results = model([image_path])

# Process results
for result in results:
    result.show()
    result.save(filename="result.jpg")
