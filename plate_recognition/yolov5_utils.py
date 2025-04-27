import torch
import cv2
import numpy as np

class YOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt', device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(self.device)

    def detect(self, image):
        # image: numpy array BGR
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        boxes = []
        for *box, conf, cls in detections:
            boxes.append({
                'box': [int(coord) for coord in box],
                'confidence': float(conf),
                'class': int(cls)
            })
        return boxes
