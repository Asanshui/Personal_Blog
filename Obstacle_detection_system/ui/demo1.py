import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QHBoxLayout, QFileDialog,
                             QMessageBox, QStatusBar)
from ultralytics import YOLO


class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray, np.ndarray)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, source, model, mode):
        super().__init__()
        self.source = source
        self.model = model
        self.mode = mode
        self.mutex = QMutex()
        self.running = True
        self.frame_count = 0

    def run(self):
        cap = cv2.VideoCapture(self.source) if self.mode == "video" else self.source
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            results = self.model.predict(frame, verbose=False)
            annotated = results[0].plot()

            # 转换颜色空间
            orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            anno_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # 发射信号
            self.update_frame.emit(orig_rgb, anno_rgb)
            self.frame_count += 1
            self.progress_signal.emit(int((self.frame_count / total_frames) * 100)) if total_frames > 0 else None

        cap.release() if self.mode == "video" else None
        self.finished_signal.emit()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = None
        self.current_results = None
        self.video_thread = None

    def initUI(self):
        self.setWindowTitle("YOLOv8 目标检测系统")
        self.setGeometry(100, 100, 1200, 700)

        # 主布局
        main_widget = QWidget()
        layout = QVBoxLayout()

        # 图像显示区域
        self.img_layout = QHBoxLayout()
        self.orig_label = self.create_label("原始图像")
        self.result_label = self.create_label("检测结果")
        self.img_layout.addWidget(self.orig_label)
        self.img_layout.addWidget(self.result_label)

        # 按钮区域
        self.btn_layout = QHBoxLayout()
        self.create_button("选择模型", self.load_model, "#4CAF50")
        self.create_button("图片检测", self.detect_image, "#2196F3")
        self.create_button("文件夹检测", self.detect_folder, "#FF9800")
        self.create_button("视频检测", self.detect_video, "#9C27B0")
        self.create_button("显示物体", self.show_objects, "#607D8B")
        self.create_button("保存结果", self.save_result, "#009688")
        self.create_button("退出系统", self.close, "#F44336")

        # 组装界面
        layout.addLayout(self.img_layout)
        layout.addLayout(self.btn_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # 初始化按钮状态
        self.toggle_buttons(False)

    def create_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(580, 480)
        label.setStyleSheet("border: 2px solid #BDBDBD; background: #212121;")
        return label

    def create_button(self, text, callback, color):
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        btn.setFixedSize(120, 40)
        btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {color}; 
                color: white; 
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {color}80; }}
        """)
        self.btn_layout.addWidget(btn)
        return btn

    def toggle_buttons(self, enabled):
        for i in range(1, 6):  # 跳过第一个模型选择按钮
            self.btn_layout.itemAt(i).widget().setEnabled(enabled)

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "YOLO模型 (*.pt)"
        )
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.toggle_buttons(True)
                self.statusBar.showMessage(f"模型加载成功: {os.path.basename(model_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def detect_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)"
        )
        if path:
            self.process_frame(path, "image")

    def detect_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            for file in os.listdir(folder):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.process_frame(os.path.join(folder, file), "image")
                    QApplication.processEvents()



    def detect_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)"
        )
        if path:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()

            self.video_thread = VideoThread(path, self.model, "video")
            self.video_thread.update_frame.connect(self.update_display)
            self.video_thread.progress_signal.connect(
                lambda p: self.statusBar.showMessage(f"处理进度: {p}%")
            )
            self.video_thread.finished_signal.connect(
                lambda: self.statusBar.showMessage("视频处理完成")
            )
            self.video_thread.start()

    def process_frame(self, path, mode):
        if mode == "image":
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 保持原始质量
        else:
            return

        if img is not None:
            # 处理4通道图像
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 动态调整分辨率
            h, w = img.shape[:2]
            target_size = max(h, w)
            if target_size > 1280:
                scale = 1280 / target_size
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # 模型推理
            results = self.model.predict(img, verbose=False)

            # 转换颜色空间
            orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

            # 更新显示
            self.update_display(orig, annotated)
            self.current_results = results

    def update_display(self, orig, result, fps=None):
        # 显示原始图像
        h, w, c = orig.shape
        q_img = QImage(orig.data, w, h, QImage.Format_RGB888)
        self.orig_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.orig_label.width(),
                self.orig_label.height(),
                Qt.KeepAspectRatio,  # 保持宽高比
                Qt.SmoothTransformation  # 使用平滑缩放
            )
        )

        # 显示结果图像
        h, w, c = result.shape
        q_img = QImage(result.data, w, h, QImage.Format_RGB888)
        self.result_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.result_label.width(),
                self.result_label.height(),
                Qt.KeepAspectRatio,  # 保持宽高比
                Qt.SmoothTransformation  # 使用平滑缩放
            )
        )

    def show_objects(self):
        if self.current_results:
            detections = self.current_results[0].boxes.cls.cpu().numpy()
            names = self.current_results[0].names
            counts = {names[int(cls)]: np.count_nonzero(detections == cls)
                      for cls in np.unique(detections)}

            msg = "检测到的物体：\n"
            for obj, count in counts.items():
                msg += f"{obj}: {count}个\n"

            QMessageBox.information(self, "检测结果", msg)
        else:
            QMessageBox.warning(self, "提示", "请先进行检测操作")

    def save_result(self):
        if hasattr(self.current_results[0], "save_dir"):
            default_path = self.current_results[0].save_dir
        else:
            default_path = os.path.expanduser("~")

        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", default_path, "图片文件 (*.jpg *.png)"
        )
        if path:
            cv2.imwrite(path, self.current_results[0].plot())

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())