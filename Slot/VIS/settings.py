from PyQt6.QtWidgets import QFileDialog
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess as sp


class Settings:
    def __init__(self, parent=None):
        self.parent = parent

        self.crop_ratio = 1
        self.crop_top_ratio = 0

        # 设置默认值
        self.Buffer_size = 10
        self.Skip_frames = 0
        self.executor = 8  # 设置默认值
        self.waitkey = 1000

        self.scale_ratio = 1
        self.parent.doubleSpinBox_scaled_ratio.valueChanged.connect(self.set_scale_ratio)

        self.parent.doubleSpinBox_crop_ratio.valueChanged.connect(self.set_crop_ratio)
        self.parent.doubleSpinBox_crop_top_ratio.valueChanged.connect(self.set_crop_top_ratio)
        self.parent.spinBox_Buffer_size.valueChanged.connect(self.set_buffer_size)
        self.parent.spinBox_Skip_frames.valueChanged.connect(self.set_skip_frames)
        self.parent.spinBox_executor.valueChanged.connect(self.set_executor)
        self.parent.spinBox_WaitKey.valueChanged.connect(self.set_waitkey)

    # 选择视频输入源
    def selected_video_source(self):
        if self.parent.radioButton_camera.isChecked():
            self.parent.input_video = 0
            self.parent.input_video_check = -1
            self.parent.lineEdit_inputvideo.setText(str(self.parent.input_video ))

        elif self.parent.radioButton_local.isChecked():
            # 使用QFileDialog类直接创建文件对话框
            file_dialog = QFileDialog(self.parent, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                file_name = file_dialog.selectedFiles()[0]
                if file_name:
                    self.parent.input_video = file_name
                    self.parent.input_video_check = -1
                    self.parent.lineEdit_inputvideo.setText(file_name)
        else:
            print('没有选择视频输入源')

    # 选择检测模型
    def selected_model_source(self):
        if self.parent.radioButton_default_model.isChecked():
            self.parent.model_path = 'F:/12-Python/yolov5/pre_modelsfile_yolov5/yolov5s.pt'
            self.parent.lineEdit_showmodel.setText('Default_model')

        if self.parent.radioButton_custom_model.isChecked():
            # 使用QFileDialog类直接创建文件对话框
            file_dialog = QFileDialog(self.parent, "Select custom_model", "", "model Files (*.pt)")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                file_name = file_dialog.selectedFiles()[0]
                if file_name:
                    self.parent.lineEdit_showmodel.setText(file_name)
                    self.parent.model_path = file_name

    # 设置视频帧处理参数 视频垂直方向修剪比例 和 起始位置
    # Slot function for doubleSpinBox_crop_ratio
    def set_crop_ratio(self, value):
        self.crop_ratio = value
        self.parent.crop_ratio = value
        print("crop_ratio",self.crop_ratio)

    # Slot function for doubleSpinBox_crop_top_ratio
    def set_crop_top_ratio(self, value):
        self.crop_top_ratio = value
        self.parent.crop_top_ratio = value
        print("crop_top_ratio", self.crop_top_ratio)

    def set_scale_ratio(self, value):
        self.scale_ratio = value
        self.parent.scale_ratio = value
        print("scale_ratio", self.scale_ratio)

    # 其它设置 Buffer_size缓冲区大小、 Skip_frames划过帧数、 ExecutorCPU线程数量、 WaitKey等待时间：

    def set_buffer_size(self, value):
        self.Buffer_size = value
        self.parent.buffer_size = value
        print("Buffer_size", self.Buffer_size)

    def set_skip_frames(self, value):
        self.Skip_frames = value
        self.parent.skip_frames = value
        print("Skip_frames", self.Skip_frames)

    def set_executor(self, value):
        self.executor = value
        self.parent.executor._max_workers = self.executor
        print("executor", self.executor)

    def set_waitkey(self, value):
        self.waitkey = value
        self.parent.waitkey = value
        print("waitkey", self.waitkey)










