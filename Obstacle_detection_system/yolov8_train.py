import warnings
import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    model.load('yolov8s.pt')
    model.train(data = 'data.yaml',
                imgsz=160,
                epochs=2,
                batch=16,
                workers=4,
                device='cuda:0',
                optimizer='SGD',
                amp = False,
                cache = False)
