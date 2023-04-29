import sys
import concurrent.futures
import threading
import random
from UI.Demo import Ui_MainWindow
from UI.resourceforQt.background import qInitResources
from Slot.VIS.settings import Settings
from Slot.VIS.visual_perception_opencv import process_frame, select_device, preprocess_frame, load_model
import cv2
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QMenu
from Slot.CAN.can_communication import CANCommunication
import sys
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QImage, QPixmap, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem


qInitResources()  # 初始化资源


class MainWindow(QMainWindow, Ui_MainWindow, Settings, CANCommunication):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.settings = Settings(parent=self)
        self.waitkey = self.settings.waitkey

        # 更新ADB画面部分 开始
        # 设置 QGraphicsView 和 QGraphicsScene 的背景颜色为透明
        self.graphicsView.setBackgroundBrush(QColor(0, 0, 0, 0))
        scene = QGraphicsScene()
        scene.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))
        self.graphicsView.setScene(scene)

        self.image_width = 1000
        self.image_height = 350

        # 创建 1000x350 像素的 QImage，并填充半透明白色（透明度 50%）
        self.image = QImage(self.image_width, self.image_height, QImage.Format.Format_ARGB32)
        self.image.fill(QColor(255, 255, 255, 128))

        # 将 QImage 显示在 QGraphicsView 中
        pixmap = QPixmap.fromImage(self.image)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphicsView.scene().addItem(self.pixmap_item)

        # 设置 QGraphicsView 的 viewport 透明
        self.graphicsView.viewport().setAutoFillBackground(False)
        self.graphicsView.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置定时器以更新 QImage
        self.timer_image = QTimer()
        self.timer_image.timeout.connect(self.update_image)
        self.timer_image.start(self.waitkey // 30)   # 与视频显示帧率保持一致

        self.ADBshow = False
        # 更新ADB画面部分 结束

        self.can_communication = CANCommunication(parent=self)
        self.can_communication.set_text_edit_can_message(self.textEdit_CANmessage)
        self.can_messages = self.can_communication.can_messages

        self.device = select_device(0)
        self.model_path = None
        self.model = None

        self.input_video = None
        self.input_video_check = None
        self.cap = None
        self.scale_ratio = 1  # 默认缩放比例为 1.0（原始尺寸）

        self.buffer_size = self.settings.Buffer_size  # 添加缓冲区大小设置
        self.skip_frames = self.settings.Skip_frames  # 添加跳帧设置

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.settings.executor)  # 添加线程池设置

        self.object_info_list = []  # 初始化物体信息列表

        # 更新帧
        self.timer_frame = QTimer()
        self.timer_frame.timeout.connect(self.update_frame)
        self.timer_frame.start(self.waitkey // 30)  # 设置刷新率，例如 30 FPS

        self.detect_on = False  # 添加这个布尔变量
        self.original_video = False

        self.horizontalSlider_videoProgress.setMinimum(0)
        self.horizontalSlider_videoProgress.setMaximum(100)
        self.horizontalSlider_videoProgress.setSingleStep(1)  # 添加这一行，将步长设为1
        self.horizontalSlider_videoProgress.setTracking(True)  # 将setTracking设置为True，使得拖动更加顺滑

        self.can_communication.set_text_edit_can_message(self.textEdit_CANmessage)
        self.can_communication.set_text_edit_can_message_receive(self.textEdit_CANmessage_receive)

        self.signal_slots_function()

    def signal_slots_function(self):
        self.radioButton_camera.clicked.connect(self.settings.selected_video_source)
        self.radioButton_local.clicked.connect(self.settings.selected_video_source)
        self.radioButton_default_model.clicked.connect(self.settings.selected_model_source)
        self.radioButton_custom_model.clicked.connect(self.settings.selected_model_source)

        self.lineEdit_Message.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lineEdit_Message.customContextMenuRequested.connect(self.contextMenuEvent)
        self.horizontalSlider_videoProgress.valueChanged.connect(self.on_slider_value_changed)
        self.lineEdit_currentTime.textChanged.connect(self.on_current_time_changed)

        self.pushButton_startend_detect.clicked.connect(self.on_button_click)
        self.pushButton_closedprocess.clicked.connect(self.stop_and_close)
        self.pushButton_original_video.clicked.connect(self.open_original_video)
        self.pushButton_ADBshow.clicked.connect(self.ADBshow_state)

    def ADBshow_state(self):
        if self.pushButton_ADBshow.text() == 'ADB ON':
            self.pushButton_ADBshow.setText('ADB OFF')
            self.pushButton_ADBshow.setStyleSheet('background-color: lightblue')
            self.ADBshow = True  # 修改布尔变量的值

        else:
            self.pushButton_ADBshow.setText('ADB ON')
            self.pushButton_ADBshow.setStyleSheet('')
            self.ADBshow = False  # 修改布尔变量的值

    def update_image(self):
        self.image.fill(QColor(255, 255, 255, 128))
        if not self.ADBshow:
            return

        painter = QPainter(self.image)  # 将 QPainter 的实例化移到循环外
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        for message in self.can_messages:
            x, y, width, height = message

            # 在 QImage 上绘制矩形
            painter.fillRect(x, y, width, height, QColor(255, 255, 255, 128))

        painter.end()  # 将 QPainter 的结束操作移到循环外

        # 更新 QGraphicsView 显示
        pixmap = QPixmap.fromImage(self.image)
        self.pixmap_item.setPixmap(pixmap)

        # 清空消息列表
        self.can_messages.clear()

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
            preprocessed_frame = preprocess_frame(frame, height, self.settings)
            detection_frame = preprocessed_frame.copy()

            # detection_frame = self.executor.submit(process_frame, detection_frame, self.model).result()  # 使用线程池处理帧
            detection_frame, object_info_list = self.executor.submit(process_frame, detection_frame,
                                                                     self.model).result()
            self.object_info_list = object_info_list  # 保存物体信息列表

            detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

            h, w, ch = detection_frame.shape
            bytes_per_line = ch * w

            image = QImage(detection_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
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

        if self.original_video:  # 仅当 self.ADB_show_on 为 True 时显示原始视频
            # 显示预处理后的帧（不带检测框）到 ADB 交互显示区
            preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)
            h_raw, w_raw, ch_raw = preprocessed_frame.shape
            bytes_per_line_raw = ch_raw * w_raw

            image_raw = QImage(preprocessed_frame.data, w_raw, h_raw, bytes_per_line_raw,
                               QImage.Format.Format_RGB888)
            pixmap_raw = QPixmap.fromImage(image_raw)

            self.label_ADBexchange.setPixmap(pixmap_raw)
            self.label_ADBexchange.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def pack_object_info_to_can_frame(self, object_info_list):
        can_frames = []

        for object_info in object_info_list:
            can_id = self.can_communication.device_id  # 从 spinBox 控件获取 CAN ID
            data = [
                min(max(object_info['category'], 0), 255),
                min(max(int(object_info['x'] * 255 / self.image_width), 0), 255),
                min(max(int(object_info['y'] * 255 / self.image_height), 0), 255),
                min(max(int(object_info['width'] * 255 / self.image_width), 0), 255),
                min(max(int(object_info['height'] * 255 / self.image_height), 0), 255)
            ]

            can_frame = (can_id, data)
            can_frames.append(can_frame)

        return can_frames

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
            self.can_communication.set_detect_on(True)  # 调用CANCommunication类中的函数

        else:
            self.pushButton_startend_detect.setText('启动视觉识别')
            self.pushButton_startend_detect.setStyleSheet('')
            self.detect_on = False  # 修改布尔变量的值
            self.can_communication.set_detect_on(False)  # 调用CANCommunication类中的函数

            # 添加下方代码，程序从新回到视频开头执行；
            # 若删除下方代码，程序实现暂停功能，当再次点击按钮，程序接着暂停时的画面继续执行。

            # if self.cap:
            #     self.cap.release()  # 释放资源
            #     self.cap = None

    def open_original_video(self):
        self.original_video = not self.original_video

    def stop_and_close(self):
        self.detect_on = False  # 停止视觉检测
        self.original_video = False  # 停止显示原始视频
        self.pushButton_startend_detect.setText('启动视觉识别')
        self.pushButton_startend_detect.setStyleSheet('')

        if self.cap:
            self.cap.release()  # 释放资源
            self.cap = None
        self.label_showvideo.clear()  # 清除label_showvideo控件上的视频显示
        self.label_ADBexchange.clear()  # 清除label_ADBexchange控件上的视频显示


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()  # 添加这行代码以显示主窗口
    sys.exit(app.exec())
