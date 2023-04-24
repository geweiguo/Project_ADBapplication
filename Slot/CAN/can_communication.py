import time
from PyQt6.QtCore import QObject, pyqtSignal
import can
from PyQt6.QtWidgets import QTextEdit
import threading
from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import QTimer


class CANCommunication:
    def __init__(self, parent=None):
        self.parent = parent

        self.detect_on = False
        self.channel = None
        self.textEdit_CANmessage = None
        self.connected = False

        self.can_type = "虚拟CAN"
        self.bitrate = 500000
        self.device_id = 1
        self.bus = None
        self.signal_slots_function()
        self.receive_thread = QThread()
        self.receive_data = False
        self.sending_data = False
        self.send_can_timer = QTimer()  # 添加这一行
        self.send_can_timer.timeout.connect(self.send_can_data)  # 添加这一行

    def set_detect_on(self, value):
        self.detect_on = value
        self.parent.detect_on = value

    def signal_slots_function(self):
        self.parent.comboBox_CANtype.currentTextChanged.connect(self.on_combobox_can_type_changed)
        self.parent.comboBox_baud_rate.currentTextChanged.connect(self.on_combobox_baud_rate_changed)
        self.parent.spinBox_recivedevice_ID.valueChanged.connect(self.on_spinbox_device_id_changed)

        self.parent.pushButton_Connect_Disconnect_CAN.clicked.connect(self.connect_disconnect_can)
        self.parent.pushButton_start_send_CAN.clicked.connect(self.start_send_can)
        self.parent.pushButton_start_receive_CAN.clicked.connect(self.start_receive_thread)

    def init_can_communication(self):
        can_type = self.can_type
        bitrate = self.bitrate
        self.textEdit_CANmessage.append(f"正在初始化 CAN 通信, 通信类型: {self.channel}, 波特率: {bitrate}")
        if self.can_type == "虚拟CAN":
            self.channel = "virtual"
            self.bus = can.interface.Bus(bustype='virtual', channel='vcan0', bitrate=bitrate)
        elif self.can_type == "物理CAN":
            # 根据具体的物理CAN类型进行初始化
            # 例如：
            # self.channel = "CANalyst-II"
            # self.bus = can.interface.Bus(bustype='pcan', channel=self.channel, bitrate=bitrate)
            pass
        else:
            raise ValueError(f"Unsupported CAN type: {can_type}")

    def on_combobox_can_type_changed(self, value):
        self.can_type = value
        self.parent.can_type = value

    def on_combobox_baud_rate_changed(self, value):
        self.bitrate = value
        self.parent.bitrate = value

    def on_spinbox_device_id_changed(self, value):
        self.device_id = value
        self.parent.device_id = value

    def connect_disconnect_can(self):
        if not self.detect_on:
            self.parent.textEdit_CANmessage.append("请开启视觉识别")
            return

        if not self.connected:  # 当前未连接
            self.textEdit_CANmessage.append("建立CAN 通信")
            self.connected = True
            self.parent.pushButton_Connect_Disconnect_CAN.setText("断开CAN 通信")
            try:
                self.init_can_communication()
                self.bus = self.bus
                self.parent.textEdit_CANmessage.append(f"连接成功: {self.can_type}, {self.bitrate}")
            except Exception as e:
                self.parent.textEdit_CANmessage.append(f"连接失败: {str(e)}")
                self.connected = False
                self.parent.pushButton_Connect_Disconnect_CAN.setText("建立CAN 通信")
        else:  # 当前已连接
            self.parent.textEdit_CANmessage.append("断开CAN通信")
            self.connected = False
            self.parent.pushButton_Connect_Disconnect_CAN.setText("建立CAN通信")
            if self.bus:
                self.bus.shutdown()
            self.parent.textEdit_CANmessage.append("已断开连接")

    def start_send_can(self):
        if not self.sending_data:
            self.sending_data = True
            self.parent.sending_data = True
            self.parent.pushButton_start_send_CAN.setText("停止发送CAN")
            self.send_can_timer.start(30)  # 启动定时器，每 100 毫秒调用一次 send_can_data
        else:
            self.stop_send_can()

    def stop_send_can(self):
        self.sending_data = False
        self.parent.sending_data = False
        self.parent.pushButton_start_send_CAN.setText("开始发送CAN")
        self.send_can_timer.stop()  # 停止定时器

    def send_can_data(self):
        if self.sending_data and self.detect_on:
            can_frames = self.parent.pack_object_info_to_can_frame(self.parent.object_info_list)
            for can_frame in can_frames:
                self.send_can_frame(can_frame)

    def start_receive_can(self):
        if not self.receive_data:
            self.receive_data = True
            self.parent.pushButton_start_receive_CAN.setText("停止接收CAN")
            self.start_receive_thread()
        else:
            self.stop_receive_can()

    def start_receive_thread(self):
        self.receive_thread.started.connect(self.receive_can_frames)
        self.receive_thread.start()

    def stop_receive_can(self):
        self.receive_data = False
        self.receive_thread.quit()
        self.receive_thread.wait()
        self.parent.pushButton_start_receive_CAN.setText("开始接收CAN")

    def send_can_frame(self, can_frame):
        if self.bus is not None:
            try:
                self.bus.send(can_frame)
                self.textEdit_CANmessage.append(f"发送 CAN 帧: {str(can_frame)}")
            except can.CanError as e:
                self.textEdit_CANmessage.append(f"发送失败: {str(e)}")

    def receive_can_frames(self):
        while self.receive_data and self.bus:
            msg = self.bus.recv(1)  # 设置1秒的超时时间
            if msg is not None:
                self.textEdit_CANmessage.append(f"接收到 CAN 帧: {str(msg)}")
                object_info_list = self.parent.unpack_can_frame_to_object_info(msg)
                # 在这里处理解析出的物体信息列表

    def set_text_edit_can_message(self, text_edit):
        self.textEdit_CANmessage = text_edit

