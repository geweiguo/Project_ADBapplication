import time
from PyQt6.QtCore import QObject, pyqtSignal
import can
from PyQt6.QtWidgets import QTextEdit
import threading
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from can.interfaces.virtual import VirtualBus
from queue import Queue



class CustomListener(can.Listener):
    def __init__(self, parent):
        self.parent = parent
        self.enabled = False  # 添加这一行

    def on_message_received(self, msg):
        if self.enabled:  # 添加这一行
            self.parent.on_message_received(msg)



class CANCommunication:
    def __init__(self, parent=None):
        super(CANCommunication, self).__init__()
        self.parent = parent

        self.detect_on = False
        self.channel = 'virtual'
        self.textEdit_CANmessage = None
        self.textEdit_CANmessage_receive = None
        self.connected = False

        self.can_type = "虚拟CAN"
        self.bitrate = 500000
        self.device_id = 1
        self.bus1 = None
        self.bus2 = None
        self.signal_slots_function()
        self.receive_thread = QThread()
        self.receive_data = False
        self.sending_data = False
        self.send_can_timer = QTimer()  # 添加这一行
        self.send_can_timer.timeout.connect(self.send_can_data)  # 添加这一行

        self.listener1 = CustomListener(self.parent)  # 初始化 listener1
        self.listener2 = CustomListener(self.parent)  # 初始化 listener2
        self.notifier1 = None  # 初始化 notifier1
        self.notifier2 = None  # 初始化 notifier2

    def set_detect_on(self, value):
        self.detect_on = value
        self.parent.detect_on = value

    def signal_slots_function(self):
        self.parent.comboBox_CANtype.currentTextChanged.connect(self.on_combobox_can_type_changed)
        self.parent.comboBox_baud_rate.currentTextChanged.connect(self.on_combobox_baud_rate_changed)
        self.parent.spinBox_recivedevice_ID.valueChanged.connect(self.on_spinbox_device_id_changed)

        self.parent.pushButton_Connect_Disconnect_CAN.clicked.connect(self.connect_disconnect_can)
        self.parent.pushButton_start_send_CAN.clicked.connect(self.start_send_can)
        self.parent.pushButton_start_receive_CAN.clicked.connect(self.start_receive_can)

    def init_can_communication(self):
        can_type = self.can_type
        bitrate = self.bitrate
        self.textEdit_CANmessage.append(f"正在初始化 CAN 通信, 通信类型: {self.channel}, 波特率: {bitrate}")
        if self.can_type == "虚拟CAN":
            self.channel = "virtual"
            queue1 = Queue()
            queue2 = Queue()
            self.bus1 = VirtualBus(channel='vcan0', queue=queue1, receive_own_messages=True)
            self.bus2 = VirtualBus(channel='vcan0', queue=queue2, receive_own_messages=True)

            # 添加过滤器
            can_filters = [{"can_id": self.device_id, "can_mask": 0x7FF}]
            print('device_id in can_filters', self.device_id)
            self.bus1.set_filters(can_filters)  # 修改这一行
            self.bus2.set_filters(can_filters)  # 添加这一行

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
                self.notifier1 = can.Notifier(self.bus1, [self.listener1])  # 创建 Notifier1 实例
                self.notifier2 = can.Notifier(self.bus2, [self.listener2])  # 创建 Notifier2 实例
                self.parent.textEdit_CANmessage.append(f"连接成功: {self.can_type}, {self.bitrate}")
            except Exception as e:
                self.parent.textEdit_CANmessage.append(f"连接失败: {str(e)}")
                self.connected = False
                self.parent.pushButton_Connect_Disconnect_CAN.setText("建立CAN 通信")
        else:  # 当前已连接
            self.parent.textEdit_CANmessage.append("断开CAN通信")
            self.connected = False
            self.parent.pushButton_Connect_Disconnect_CAN.setText("建立CAN通信")
            if self.bus1:
                self.bus1.shutdown()
            if self.bus2:
                self.bus2.shutdown()
            if self.notifier1:  # 关闭 Notifier1 实例
                self.notifier1.stop()
            if self.notifier2:  # 关闭 Notifier2 实例
                self.notifier2.stop()
            self.parent.textEdit_CANmessage.append("已断开连接")

    # 发送部分

    def start_send_can(self):
        if not self.connected:
            self.parent.textEdit_CANmessage.append("请先建立 CAN 通信")
            return

        if not self.sending_data:
            self.sending_data = True
            self.parent.sending_data = True
            self.parent.pushButton_start_send_CAN.setText("停止发送CAN")
            self.send_can_timer.start(30)  # 启动定时器，每 100 毫秒调用一次 send_can_data
            """
            self.send_can_timer.timeout.connect(self.send_can_data) 
            """
        else:
            self.stop_send_can()

    # 修改部分
    def send_can_data(self):
        if self.sending_data and self.detect_on:
            try:
                can_frames = self.parent.pack_object_info_to_can_frame(self.parent.object_info_list)
                for can_frame in can_frames:
                    self.send_can_frame(can_frame)
            except Exception as e:
                self.textEdit_CANmessage.append(f"发送过程中出现错误: {str(e)}")
                self.stop_send_can()

    # 新增部分
    def send_can_frame(self, can_frame):
        if self.bus1 is not None:
            can_id, data = can_frame  # 新增：从 can_frame 中提取 can_id 和 data
            message = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)  # 新增：创建一个新的 can.Message
            try:
                self.bus1.send(message)  # 修改这一行：使用新创建的 message 变量发送 CAN 消息
                self.textEdit_CANmessage.append(f"发送 CAN 帧: {str(message)}")

            except can.CanError as e:
                self.textEdit_CANmessage.append(f"发送失败: {str(e)}")

    def stop_send_can(self):
        self.sending_data = False
        self.parent.sending_data = False
        self.parent.pushButton_start_send_CAN.setText("开始发送CAN")
        self.send_can_timer.stop()  # 停止定时器

    # 接收部分

    def start_receive_can(self):
        if not self.connected:
            self.parent.textEdit_CANmessage_receive.append("请先建立 CAN 通信")
            return

        if not self.receive_data:
            self.receive_data = True
            self.listener2.enabled = True  # 添加这一行
            self.parent.pushButton_start_receive_CAN.setText("停止接收CAN")
            self.start_receive_thread()  # 启动接收线程

        else:
            self.stop_receive_can()

    def start_receive_thread(self):
        self.receive_thread = threading.Thread(target=self.receive_can_data)
        self.receive_thread.setDaemon(True)
        self.receive_thread.start()

    def receive_can_data(self):
        while self.receive_data:
            # 等待接收到的消息
            msg = self.bus2.recv(5)
            print('msg', msg)
            if msg is not None:
                self.listener2.on_message_received(msg)
            time.sleep(0.1)  # 添加这一行

    def stop_receive_can(self):
        self.receive_data = False
        self.listener2.enabled = False  # 添加这一行
        self.parent.pushButton_start_receive_CAN.setText("开始接收CAN")

    def on_message_received(self, msg):
        self.textEdit_CANmessage_receive.append(f"接收到 CAN 帧: {str(msg)}")

    def set_text_edit_can_message(self, text_edit):
        self.textEdit_CANmessage = text_edit

    def set_text_edit_can_message_receive(self, text_edit):
        self.textEdit_CANmessage_receive = text_edit
