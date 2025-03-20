from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO("./ultralytics/cfg/models/v8/mtyolov8_myCBAM.yaml", verbose=True)  添加了自定义的注意力机制后
    model = YOLO("./ultralytics/cfg/models/v8/mtyolov8_CBAM.yaml", verbose=True) # 添加了官方的注意力机制
    model.train(data="./data.yaml",epochs=5)