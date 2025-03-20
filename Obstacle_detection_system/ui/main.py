import sys
import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, \
    QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
from ultralytics import YOLO


class Worker:
    def __init__(self):
        self.model = None
        self.current_annotated_image = None
        self.current_annotated_video_frame = None
        self.detection_type = None
        self.video_capture = None
        self.timer = QTimer()
        self.is_camera_running = False

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(None, "选择模型文件", "", "模型文件 (*.pt)")
        if model_path:
            self.model = YOLO(model_path)
            if self.model:
                return True
            else:
                return False

    def save_image(self, image):
        if image is not None:
            file_name, _ = QFileDialog.getSaveFileName(None, "保存图片", "", "JPEG (*.jpg);;PNG (*.png);;All Files (*)")
            if file_name:
                cv2.imwrite(file_name, image)

    def start_camera(self, camera_url):
        self.video_capture = cv2.VideoCapture(camera_url)
        if not self.video_capture.isOpened():
            QMessageBox.warning(None, "警告", "无法连接到摄像头，请检查连接！")
            return False
        self.is_camera_running = True
        return True

    def stop_camera(self):
        if self.video_capture is not None:
            self.video_capture.release()
        self.is_camera_running = False

    def process_frame(self, frame):
        results = self.model.predict(frame)
        annotated_frame = results[0].plot()
        return annotated_frame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 目标检测系统")
        self.setGeometry(300, 150, 1200, 600)

        # 创建两个 QLabel 分别显示左右图像
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumSize(580, 450)
        self.label1.setStyleSheet('border:3px solid #6950a1; background-color: black;')

        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumSize(580, 450)
        self.label2.setStyleSheet('border:3px solid #6950a1; background-color: black;')

        # 水平布局，用于放置左右两个 QLabel
        layout = QVBoxLayout()
        hbox_video = QHBoxLayout()
        hbox_video.addWidget(self.label1)
        hbox_video.addWidget(self.label2)
        layout.addLayout(hbox_video)

        self.worker = Worker()

        # 创建按钮布局
        hbox_buttons = QHBoxLayout()

        # 添加模型选择按钮
        self.load_model_button = QPushButton("👆模型选择")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.load_model_button)

        # 添加图片检测按钮
        self.image_detect_button = QPushButton("🖼️️图片检测")
        self.image_detect_button.clicked.connect(self.select_image)
        self.image_detect_button.setEnabled(False)
        self.image_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.image_detect_button)

        # 添加视频检测按钮
        self.video_detect_button = QPushButton("📹视频检测")
        self.video_detect_button.clicked.connect(self.select_video)
        self.video_detect_button.setEnabled(False)
        self.video_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.video_detect_button)

        # 添加摄像头检测按钮
        self.camera_detect_button = QPushButton("📷摄像头检测")
        self.camera_detect_button.clicked.connect(self.start_camera_detection)
        self.camera_detect_button.setEnabled(False)
        self.camera_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.camera_detect_button)

        # 添加停止摄像头按钮
        self.stop_camera_button = QPushButton("⏹️停止摄像头")
        self.stop_camera_button.clicked.connect(self.stop_camera_detection)
        self.stop_camera_button.setEnabled(False)
        self.stop_camera_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.stop_camera_button)

        # 添加保存检测结果按钮
        self.save_button = QPushButton("💾保存检测结果")
        self.save_button.clicked.connect(self.save_detection)
        self.save_button.setEnabled(False)
        self.save_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.save_button)

        # 添加退出按钮
        self.exit_button = QPushButton("❌退出")
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.exit_button)

        layout.addLayout(hbox_buttons)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 定时器用于更新摄像头画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)

    def save_detection(self):
        detection_type = self.worker.detection_type
        if detection_type == "image":
            self.save_detection_results()

    def select_image(self):
        image_path, _ = QFileDialog.getOpenFileName(None, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png)")
        self.flag = 0
        if image_path:
            self.detect_image(image_path)

    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(None, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if video_path:
            self.detect_video(video_path)

    def detect_video(self, video_path):
        self.worker.video_capture = cv2.VideoCapture(video_path)
        if not self.worker.video_capture.isOpened():
            QMessageBox.warning(self, "警告", "无法打开视频文件！")
            return

        self.timer.start(30)  # 30ms更新一帧

    def start_camera_detection(self):
        camera_url = "http://admin:admin@192.168.167.36:8081/"  # 摄像头URL  注意更改IP地址
        if self.worker.start_camera(camera_url):
            self.timer.start(30)  # 30ms更新一帧
            self.camera_detect_button.setEnabled(False)
            self.stop_camera_button.setEnabled(True)

    def stop_camera_detection(self):
        self.timer.stop()
        self.worker.stop_camera()
        self.camera_detect_button.setEnabled(True)
        self.stop_camera_button.setEnabled(False)

    def update_camera_frame(self):
        if self.worker.is_camera_running or self.worker.video_capture is not None:
            ret, frame = self.worker.video_capture.read()
            if ret:
                annotated_frame = self.worker.process_frame(frame)
                self.display_frames(frame, annotated_frame)

    def display_frames(self, original_frame, annotated_frame):
        original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        height1, width1, channel1 = original_frame_rgb.shape
        bytesPerLine1 = 3 * width1
        qimage1 = QImage(original_frame_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
        pixmap1 = QPixmap.fromImage(qimage1)
        self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        height2, width2, channel2 = annotated_frame_rgb.shape
        bytesPerLine2 = 3 * width2
        qimage2 = QImage(annotated_frame_rgb.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
        pixmap2 = QPixmap.fromImage(qimage2)
        self.label2.setPixmap(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))

    def detect_image(self, image_path):
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                results = self.worker.model.predict(image)
                self.worker.detection_type = "image"
                if results:
                    self.current_results = results
                    self.worker.current_annotated_image = results[0].plot()
                    self.display_frames(image, self.worker.current_annotated_image)
                    self.save_button.setEnabled(True)

    def save_detection_results(self):
        if self.worker.current_annotated_image is not None:
            self.worker.save_image(self.worker.current_annotated_image)

    def load_model(self):
        if self.worker.load_model():
            self.image_detect_button.setEnabled(True)
            self.video_detect_button.setEnabled(True)
            self.camera_detect_button.setEnabled(True)

    def exit_application(self):
        self.worker.stop_camera()
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())