# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")
# source = 'img'
# model.predict(source, save=True)

from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO('runs/detect/train3/weights/best.pt')
# 修改为自己的图像或者文件夹的路径
source = 'img' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)