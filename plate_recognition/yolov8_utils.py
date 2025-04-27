from ultralytics import YOLO
import cv2
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path='yolov8s.pt', device='cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image):
        # image: numpy array BGR
        results = self.model(image)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                boxes.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': cls
                })
        return boxes
