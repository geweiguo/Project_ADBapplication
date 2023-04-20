import sys
import concurrent.futures
from UI.Demo import Ui_MainWindow
from UI.resourceforQt.background import qInitResources
from Slot.VIS.settings import Settings
from Slot.VIS.visual_perception_opencv import process_frame, select_device, preprocess_frame, load_model
import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMenu


qInitResources()  # 初始化资源


class MainWindow(QMainWindow, Ui_MainWindow, Settings):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.settings = Settings(parent=self)

        self.device = select_device(0)
        self.model_path = None
        self.model = None

        self.input_video = None
        self.input_video_check = None
        self.cap = None
        self.scale_ratio = 0.8  # 默认缩放比例为 1.0（原始尺寸）

        self.buffer_size = self.settings.Buffer_size  # 添加缓冲区大小设置
        self.skip_frames = self.settings.Skip_frames  # 添加跳帧设置
        self.waitkey = self.settings.waitkey
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.settings.executor)  # 添加线程池设置

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.waitkey // 30)  # 设置刷新率，例如 30 FPS

        self.detect_on = False  # 添加这个布尔变量

        self.radioButton_camera.clicked.connect(self.settings.selected_video_source)
        self.radioButton_local.clicked.connect(self.settings.selected_video_source)
        self.radioButton_default_model.clicked.connect(self.settings.selected_model_source)
        self.radioButton_custom_model.clicked.connect(self.settings.selected_model_source)

        self.lineEdit_Message.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lineEdit_Message.customContextMenuRequested.connect(self.contextMenuEvent)

        self.horizontalSlider_videoProgress.setMinimum(0)
        self.horizontalSlider_videoProgress.setMaximum(100)
        self.horizontalSlider_videoProgress.setTracking(False)
        self.horizontalSlider_videoProgress.valueChanged.connect(self.on_slider_value_changed)
        self.lineEdit_currentTime.textChanged.connect(self.on_current_time_changed)

        self.pushButton_startend_detect.clicked.connect(self.on_button_click)
        self.pushButton_closedprocess.clicked.connect(self.stop_and_close)

    def set_scale_ratio(self, scale_ratio):
        self.scale_ratio = scale_ratio

    def on_slider_value_changed(self, value):
        if self.cap:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = int(total_frames * value / 100)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.update_current_time()

    def on_current_time_changed(self):
        if self.cap:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            input_time = self.lineEdit_currentTime.text()
            try:
                input_seconds = int(input_time)
                target_frame = input_seconds * fps
                if 0 <= target_frame <= total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    self.update_current_time()
            except ValueError:
                pass

    def update_current_time(self):
        if self.cap:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            current_seconds = int(current_frame / fps)
            minutes, seconds = divmod(current_seconds, 60)
            formatted_time = f"{minutes:02d}:{seconds:02d}"
            self.lineEdit_currentTime.setText(formatted_time)

    def update_frame(self):
        if not self.detect_on:
            return
        if self.detect_on:  # 根据布尔变量值决定是否执行检测和显示处理后的帧

            if not self.cap:
                self.cap = cv2.VideoCapture(self.input_video)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)  # 设置缓冲区大小

            if not self.model and self.model_path:
                self.model = load_model(self.model_path, self.device)

            for _ in range(self.skip_frames):  # 跳过指定帧数
                self.cap.read()

            ret, frame = self.cap.read()

            if not ret:
                return

            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame = preprocess_frame(frame, height, self.settings)
            frame = self.executor.submit(process_frame, frame, self.model).result()  # 使用线程池处理帧

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)

            # 设置 QPixmap 到 QLabel
            self.label_showvideo.setPixmap(pixmap)

            # 将 QLabel 设置为居中对齐
            self.label_showvideo.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if self.cap:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                slider_value = int(current_frame / total_frames * 100)
                self.horizontalSlider_videoProgress.setValue(slider_value)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        clear_action = context_menu.addAction("清除消息")
        clear_action.triggered.connect(self.clear_message)
        context_menu.exec(self.lineEdit_Message.mapToGlobal(event))

    def clear_message(self):
        self.lineEdit_Message.clear()

    def show_message(self, message):
        self.lineEdit_Message.setText(message)

    def on_button_click(self):
        if not self.input_video_check or not self.model_path:
            self.show_message("警告：请先定义视频源和模型类型！")
            QMessageBox.warning(self, "警告", "请先定义视频源和模型类型。")
            return

        if self.pushButton_startend_detect.text() == '启动视觉识别':
            self.pushButton_startend_detect.setText('暂停视觉识别')
            self.pushButton_startend_detect.setStyleSheet('background-color: lightblue')
            self.detect_on = True  # 修改布尔变量的值

        else:
            self.pushButton_startend_detect.setText('启动视觉识别')
            self.pushButton_startend_detect.setStyleSheet('')
            self.detect_on = False  # 修改布尔变量的值

            # 添加下方代码，程序从新回到视频开头执行；
            # 若删除下方代码，程序实现暂停功能，当再次点击按钮，程序接着暂停时的画面继续执行。

            # if self.cap:
            #     self.cap.release()  # 释放资源
            #     self.cap = None

    def stop_and_close(self):
        self.detect_on = False  # 停止视觉检测
        if self.cap:
            self.cap.release()  # 释放资源
            self.cap = None
        self.label_showvideo.clear()  # 清除label_showvideo控件上的视频显示


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()  # 添加这行代码以显示主窗口
    sys.exit(app.exec())
