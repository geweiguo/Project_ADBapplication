# Form implementation generated from reading ui file 'F:\12-Python\03-Python3.9_Pytorch\03-Project\01-ADB_Test\UI\Demo.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1396, 933)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/logoico/logo.PNG"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_startend_detect = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_startend_detect.sizePolicy().hasHeightForWidth())
        self.pushButton_startend_detect.setSizePolicy(sizePolicy)
        self.pushButton_startend_detect.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_startend_detect.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton_startend_detect.setObjectName("pushButton_startend_detect")
        self.horizontalLayout_4.addWidget(self.pushButton_startend_detect)
        self.pushButton_closedprocess = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_closedprocess.sizePolicy().hasHeightForWidth())
        self.pushButton_closedprocess.setSizePolicy(sizePolicy)
        self.pushButton_closedprocess.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_closedprocess.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton_closedprocess.setObjectName("pushButton_closedprocess")
        self.horizontalLayout_4.addWidget(self.pushButton_closedprocess)
        self.gridLayout_3.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(350, 0))
        self.tabWidget.setMaximumSize(QtCore.QSize(350, 16777215))
        self.tabWidget.setObjectName("tabWidget")
        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName("widget")
        self.groupBox_choosevideo = QtWidgets.QGroupBox(parent=self.widget)
        self.groupBox_choosevideo.setEnabled(True)
        self.groupBox_choosevideo.setGeometry(QtCore.QRect(9, 9, 321, 150))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_choosevideo.sizePolicy().hasHeightForWidth())
        self.groupBox_choosevideo.setSizePolicy(sizePolicy)
        self.groupBox_choosevideo.setMinimumSize(QtCore.QSize(0, 150))
        self.groupBox_choosevideo.setMaximumSize(QtCore.QSize(16777215, 180))
        self.groupBox_choosevideo.setObjectName("groupBox_choosevideo")
        self.radioButton_camera = QtWidgets.QRadioButton(parent=self.groupBox_choosevideo)
        self.radioButton_camera.setGeometry(QtCore.QRect(20, 40, 95, 19))
        self.radioButton_camera.setObjectName("radioButton_camera")
        self.radioButton_local = QtWidgets.QRadioButton(parent=self.groupBox_choosevideo)
        self.radioButton_local.setGeometry(QtCore.QRect(20, 70, 95, 19))
        self.radioButton_local.setObjectName("radioButton_local")
        self.horizontalLayoutWidget = QtWidgets.QWidget(parent=self.groupBox_choosevideo)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 90, 291, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_inputvideo = QtWidgets.QLineEdit(parent=self.horizontalLayoutWidget)
        self.lineEdit_inputvideo.setEnabled(True)
        self.lineEdit_inputvideo.setObjectName("lineEdit_inputvideo")
        self.horizontalLayout_2.addWidget(self.lineEdit_inputvideo)
        self.groupBox_setshowmode = QtWidgets.QGroupBox(parent=self.widget)
        self.groupBox_setshowmode.setGeometry(QtCore.QRect(9, 176, 321, 691))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_setshowmode.sizePolicy().hasHeightForWidth())
        self.groupBox_setshowmode.setSizePolicy(sizePolicy)
        self.groupBox_setshowmode.setObjectName("groupBox_setshowmode")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.groupBox_setshowmode)
        self.groupBox_3.setEnabled(True)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 30, 301, 141))
        self.groupBox_3.setObjectName("groupBox_3")
        self.radioButton_default_model = QtWidgets.QRadioButton(parent=self.groupBox_3)
        self.radioButton_default_model.setEnabled(True)
        self.radioButton_default_model.setGeometry(QtCore.QRect(10, 20, 95, 19))
        self.radioButton_default_model.setObjectName("radioButton_default_model")
        self.radioButton_custom_model = QtWidgets.QRadioButton(parent=self.groupBox_3)
        self.radioButton_custom_model.setGeometry(QtCore.QRect(10, 50, 95, 19))
        self.radioButton_custom_model.setObjectName("radioButton_custom_model")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(parent=self.groupBox_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 80, 281, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_showmodel = QtWidgets.QLineEdit(parent=self.horizontalLayoutWidget_2)
        self.lineEdit_showmodel.setEnabled(True)
        self.lineEdit_showmodel.setObjectName("lineEdit_showmodel")
        self.horizontalLayout_3.addWidget(self.lineEdit_showmodel)
        self.groupBox = QtWidgets.QGroupBox(parent=self.groupBox_setshowmode)
        self.groupBox.setGeometry(QtCore.QRect(10, 180, 301, 111))
        self.groupBox.setObjectName("groupBox")
        self.doubleSpinBox_crop_top_ratio = QtWidgets.QDoubleSpinBox(parent=self.groupBox)
        self.doubleSpinBox_crop_top_ratio.setGeometry(QtCore.QRect(200, 50, 62, 22))
        self.doubleSpinBox_crop_top_ratio.setDecimals(2)
        self.doubleSpinBox_crop_top_ratio.setMaximum(0.3)
        self.doubleSpinBox_crop_top_ratio.setObjectName("doubleSpinBox_crop_top_ratio")
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.groupBox)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_3.setGeometry(QtCore.QRect(20, 50, 141, 21))
        self.lineEdit_3.setMouseTracking(False)
        self.lineEdit_3.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_3.setFrame(False)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.groupBox)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_2.setGeometry(QtCore.QRect(20, 20, 141, 21))
        self.lineEdit_2.setMouseTracking(False)
        self.lineEdit_2.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_2.setFrame(False)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.doubleSpinBox_crop_ratio = QtWidgets.QDoubleSpinBox(parent=self.groupBox)
        self.doubleSpinBox_crop_ratio.setGeometry(QtCore.QRect(200, 20, 62, 22))
        self.doubleSpinBox_crop_ratio.setDecimals(2)
        self.doubleSpinBox_crop_ratio.setMaximum(1.0)
        self.doubleSpinBox_crop_ratio.setSingleStep(0.1)
        self.doubleSpinBox_crop_ratio.setProperty("value", 1.0)
        self.doubleSpinBox_crop_ratio.setObjectName("doubleSpinBox_crop_ratio")
        self.lineEdit_ = QtWidgets.QLineEdit(parent=self.groupBox)
        self.lineEdit_.setEnabled(False)
        self.lineEdit_.setGeometry(QtCore.QRect(20, 80, 141, 21))
        self.lineEdit_.setMouseTracking(False)
        self.lineEdit_.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_.setFrame(False)
        self.lineEdit_.setObjectName("lineEdit_")
        self.doubleSpinBox_scaled_ratio = QtWidgets.QDoubleSpinBox(parent=self.groupBox)
        self.doubleSpinBox_scaled_ratio.setGeometry(QtCore.QRect(200, 80, 62, 22))
        self.doubleSpinBox_scaled_ratio.setDecimals(2)
        self.doubleSpinBox_scaled_ratio.setMaximum(5.0)
        self.doubleSpinBox_scaled_ratio.setSingleStep(0.1)
        self.doubleSpinBox_scaled_ratio.setProperty("value", 1.56)
        self.doubleSpinBox_scaled_ratio.setObjectName("doubleSpinBox_scaled_ratio")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.groupBox_setshowmode)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 300, 301, 141))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        self.spinBox_Buffer_size = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.spinBox_Buffer_size.setGeometry(QtCore.QRect(200, 20, 61, 22))
        self.spinBox_Buffer_size.setMaximum(200)
        self.spinBox_Buffer_size.setProperty("value", 20)
        self.spinBox_Buffer_size.setObjectName("spinBox_Buffer_size")
        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_4.setEnabled(False)
        self.lineEdit_4.setGeometry(QtCore.QRect(20, 20, 141, 21))
        self.lineEdit_4.setMouseTracking(False)
        self.lineEdit_4.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_4.setFrame(False)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_5.setEnabled(False)
        self.lineEdit_5.setGeometry(QtCore.QRect(20, 50, 141, 21))
        self.lineEdit_5.setMouseTracking(False)
        self.lineEdit_5.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_5.setFrame(False)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.spinBox_Skip_frames = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.spinBox_Skip_frames.setGeometry(QtCore.QRect(200, 50, 61, 22))
        self.spinBox_Skip_frames.setMaximum(10)
        self.spinBox_Skip_frames.setObjectName("spinBox_Skip_frames")
        self.spinBox_Skip_frames.setProperty("value", 1)
        self.lineEdit_Executor = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_Executor.setEnabled(False)
        self.lineEdit_Executor.setGeometry(QtCore.QRect(20, 80, 141, 21))
        self.lineEdit_Executor.setMouseTracking(False)
        self.lineEdit_Executor.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_Executor.setFrame(False)
        self.lineEdit_Executor.setObjectName("lineEdit_Executor")
        self.spinBox_executor = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.spinBox_executor.setGeometry(QtCore.QRect(200, 80, 61, 22))
        self.spinBox_executor.setProperty("value", 8)
        self.spinBox_executor.setObjectName("spinBox_executor")
        self.lineEdit_WaitKey = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_WaitKey.setEnabled(False)
        self.lineEdit_WaitKey.setGeometry(QtCore.QRect(20, 110, 141, 21))
        self.lineEdit_WaitKey.setMouseTracking(False)
        self.lineEdit_WaitKey.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_WaitKey.setFrame(False)
        self.lineEdit_WaitKey.setObjectName("lineEdit_WaitKey")
        self.spinBox_WaitKey = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.spinBox_WaitKey.setGeometry(QtCore.QRect(200, 110, 61, 22))
        self.spinBox_WaitKey.setMaximum(1500)
        self.spinBox_WaitKey.setProperty("value", 800)
        self.spinBox_WaitKey.setObjectName("spinBox_WaitKey")
        self.groupBox_11 = QtWidgets.QGroupBox(parent=self.groupBox_setshowmode)
        self.groupBox_11.setGeometry(QtCore.QRect(0, 450, 321, 151))
        self.groupBox_11.setObjectName("groupBox_11")
        self.lineEdit_Message = QtWidgets.QLineEdit(parent=self.groupBox_11)
        self.lineEdit_Message.setGeometry(QtCore.QRect(10, 30, 301, 100))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_Message.sizePolicy().hasHeightForWidth())
        self.lineEdit_Message.setSizePolicy(sizePolicy)
        self.lineEdit_Message.setMinimumSize(QtCore.QSize(0, 100))
        self.lineEdit_Message.setMaximumSize(QtCore.QSize(16777215, 100))
        self.lineEdit_Message.setObjectName("lineEdit_Message")
        self.tabWidget.addTab(self.widget, "")
        self.widget_2 = QtWidgets.QWidget()
        self.widget_2.setObjectName("widget_2")
        self.pushButton_Connect_Disconnect_CAN = QtWidgets.QPushButton(parent=self.widget_2)
        self.pushButton_Connect_Disconnect_CAN.setGeometry(QtCore.QRect(10, 240, 321, 35))
        self.pushButton_Connect_Disconnect_CAN.setObjectName("pushButton_Connect_Disconnect_CAN")
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.widget_2)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 40, 321, 61))
        self.groupBox_4.setObjectName("groupBox_4")
        self.comboBox_CANtype = QtWidgets.QComboBox(parent=self.groupBox_4)
        self.comboBox_CANtype.setGeometry(QtCore.QRect(0, 20, 311, 30))
        self.comboBox_CANtype.setObjectName("comboBox_CANtype")
        self.comboBox_CANtype.addItem("")
        self.comboBox_CANtype.addItem("")
        self.groupBox_5 = QtWidgets.QGroupBox(parent=self.widget_2)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 120, 151, 81))
        self.groupBox_5.setObjectName("groupBox_5")
        self.comboBox_baud_rate = QtWidgets.QComboBox(parent=self.groupBox_5)
        self.comboBox_baud_rate.setGeometry(QtCore.QRect(10, 30, 121, 35))
        self.comboBox_baud_rate.setMinimumSize(QtCore.QSize(0, 35))
        self.comboBox_baud_rate.setMaximumSize(QtCore.QSize(16777215, 35))
        self.comboBox_baud_rate.setObjectName("comboBox_baud_rate")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.comboBox_baud_rate.addItem("")
        self.groupBox_6 = QtWidgets.QGroupBox(parent=self.widget_2)
        self.groupBox_6.setGeometry(QtCore.QRect(10, 470, 321, 151))
        self.groupBox_6.setObjectName("groupBox_6")
        self.textEdit_CANmessage = QtWidgets.QTextEdit(parent=self.groupBox_6)
        self.textEdit_CANmessage.setGeometry(QtCore.QRect(10, 30, 301, 101))
        self.textEdit_CANmessage.setObjectName("textEdit_CANmessage")
        self.groupBox_8 = QtWidgets.QGroupBox(parent=self.widget_2)
        self.groupBox_8.setGeometry(QtCore.QRect(180, 120, 151, 81))
        self.groupBox_8.setObjectName("groupBox_8")
        self.spinBox_recivedevice_ID = QtWidgets.QSpinBox(parent=self.groupBox_8)
        self.spinBox_recivedevice_ID.setGeometry(QtCore.QRect(20, 30, 101, 35))
        self.spinBox_recivedevice_ID.setMaximum(10)
        self.spinBox_recivedevice_ID.setProperty("value", 1)
        self.spinBox_recivedevice_ID.setObjectName("spinBox_recivedevice_ID")
        self.pushButton_start_send_CAN = QtWidgets.QPushButton(parent=self.widget_2)
        self.pushButton_start_send_CAN.setGeometry(QtCore.QRect(10, 340, 141, 41))
        self.pushButton_start_send_CAN.setObjectName("pushButton_start_send_CAN")
        self.pushButton_start_receive_CAN = QtWidgets.QPushButton(parent=self.widget_2)
        self.pushButton_start_receive_CAN.setGeometry(QtCore.QRect(180, 340, 151, 41))
        self.pushButton_start_receive_CAN.setObjectName("pushButton_start_receive_CAN")
        self.groupBox_9 = QtWidgets.QGroupBox(parent=self.widget_2)
        self.groupBox_9.setGeometry(QtCore.QRect(10, 620, 321, 151))
        self.groupBox_9.setObjectName("groupBox_9")
        self.textEdit_CANmessage_receive = QtWidgets.QTextEdit(parent=self.groupBox_9)
        self.textEdit_CANmessage_receive.setGeometry(QtCore.QRect(10, 30, 301, 101))
        self.textEdit_CANmessage_receive.setObjectName("textEdit_CANmessage_receive")
        self.tabWidget.addTab(self.widget_2, "")
        self.widget1 = QtWidgets.QWidget()
        self.widget1.setObjectName("widget1")
        self.groupBox_7 = QtWidgets.QGroupBox(parent=self.widget1)
        self.groupBox_7.setGeometry(QtCore.QRect(20, 70, 311, 101))
        self.groupBox_7.setObjectName("groupBox_7")
        self.doubleSpinBox_crop_top_ratio_2 = QtWidgets.QDoubleSpinBox(parent=self.groupBox_7)
        self.doubleSpinBox_crop_top_ratio_2.setGeometry(QtCore.QRect(70, 60, 62, 22))
        self.doubleSpinBox_crop_top_ratio_2.setObjectName("doubleSpinBox_crop_top_ratio_2")
        self.spinBox_Buffer_size_2 = QtWidgets.QSpinBox(parent=self.groupBox_7)
        self.spinBox_Buffer_size_2.setGeometry(QtCore.QRect(70, 30, 61, 22))
        self.spinBox_Buffer_size_2.setObjectName("spinBox_Buffer_size_2")
        self.spinBox_Skip_frames_2 = QtWidgets.QSpinBox(parent=self.groupBox_7)
        self.spinBox_Skip_frames_2.setGeometry(QtCore.QRect(230, 30, 61, 22))
        self.spinBox_Skip_frames_2.setObjectName("spinBox_Skip_frames_2")
        self.doubleSpinBox_crop_top_ratio_3 = QtWidgets.QDoubleSpinBox(parent=self.groupBox_7)
        self.doubleSpinBox_crop_top_ratio_3.setGeometry(QtCore.QRect(230, 60, 62, 22))
        self.doubleSpinBox_crop_top_ratio_3.setObjectName("doubleSpinBox_crop_top_ratio_3")
        self.lineEdit_7 = QtWidgets.QLineEdit(parent=self.groupBox_7)
        self.lineEdit_7.setEnabled(False)
        self.lineEdit_7.setGeometry(QtCore.QRect(20, 30, 41, 21))
        self.lineEdit_7.setMouseTracking(False)
        self.lineEdit_7.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_7.setFrame(False)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_9 = QtWidgets.QLineEdit(parent=self.groupBox_7)
        self.lineEdit_9.setEnabled(False)
        self.lineEdit_9.setGeometry(QtCore.QRect(180, 30, 41, 21))
        self.lineEdit_9.setMouseTracking(False)
        self.lineEdit_9.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_9.setFrame(False)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_10 = QtWidgets.QLineEdit(parent=self.groupBox_7)
        self.lineEdit_10.setEnabled(False)
        self.lineEdit_10.setGeometry(QtCore.QRect(20, 60, 41, 21))
        self.lineEdit_10.setMouseTracking(False)
        self.lineEdit_10.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_10.setFrame(False)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_11 = QtWidgets.QLineEdit(parent=self.groupBox_7)
        self.lineEdit_11.setEnabled(False)
        self.lineEdit_11.setGeometry(QtCore.QRect(180, 60, 41, 21))
        self.lineEdit_11.setMouseTracking(False)
        self.lineEdit_11.setStyleSheet("font: 9pt \"Microsoft YaHei UI\";")
        self.lineEdit_11.setFrame(False)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.pushButton_original_video = QtWidgets.QPushButton(parent=self.widget1)
        self.pushButton_original_video.setGeometry(QtCore.QRect(60, 330, 231, 51))
        self.pushButton_original_video.setObjectName("pushButton_original_video")
        self.pushButton_ADBshow = QtWidgets.QPushButton(parent=self.widget1)
        self.pushButton_ADBshow.setGeometry(QtCore.QRect(60, 450, 231, 51))
        self.pushButton_ADBshow.setObjectName("pushButton_ADBshow")
        self.groupBox_10 = QtWidgets.QGroupBox(parent=self.widget1)
        self.groupBox_10.setGeometry(QtCore.QRect(10, 560, 321, 151))
        self.groupBox_10.setObjectName("groupBox_10")
        self.textEdit_ADBshow_message = QtWidgets.QTextEdit(parent=self.groupBox_10)
        self.textEdit_ADBshow_message.setGeometry(QtCore.QRect(10, 30, 301, 101))
        self.textEdit_ADBshow_message.setObjectName("textEdit_ADBshow_message")
        self.pushButton_lighting_on = QtWidgets.QPushButton(parent=self.widget1)
        self.pushButton_lighting_on.setGeometry(QtCore.QRect(60, 390, 231, 51))
        self.pushButton_lighting_on.setObjectName("pushButton_lighting_on")
        self.radioButton = QtWidgets.QRadioButton(parent=self.widget1)
        self.radioButton.setGeometry(QtCore.QRect(10, 210, 251, 19))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(parent=self.widget1)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 30, 251, 19))
        self.radioButton_2.setObjectName("radioButton_2")
        self.tabWidget.addTab(self.widget1, "")
        self.gridLayout_3.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 0, 1, 1, 1)
        self.widget_3 = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(parent=self.widget_3)
        self.stackedWidget.setMinimumSize(QtCore.QSize(1000, 820))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.label_ADBexchange = QtWidgets.QLabel(parent=self.page)
        self.label_ADBexchange.setGeometry(QtCore.QRect(0, 0, 1000, 400))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_ADBexchange.sizePolicy().hasHeightForWidth())
        self.label_ADBexchange.setSizePolicy(sizePolicy)
        self.label_ADBexchange.setMinimumSize(QtCore.QSize(1000, 400))
        self.label_ADBexchange.setMaximumSize(QtCore.QSize(1000, 400))
        self.label_ADBexchange.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.label_ADBexchange.setAutoFillBackground(False)
        self.label_ADBexchange.setStyleSheet("background-image: url(:/back/background_ADB.JPG);\n"
                                           "background-position: center center;\n"
                                           "background-repeat: no-repeat;\n"
                                             "")
        self.label_ADBexchange.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.label_ADBexchange.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.label_ADBexchange.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_ADBexchange.setWordWrap(False)
        self.label_ADBexchange.setObjectName("label_ADBexchange")
        self.label_showvideo = QtWidgets.QLabel(parent=self.page)
        self.label_showvideo.setGeometry(QtCore.QRect(0, 420, 1000, 400))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_showvideo.sizePolicy().hasHeightForWidth())
        self.label_showvideo.setSizePolicy(sizePolicy)
        self.label_showvideo.setMinimumSize(QtCore.QSize(1000, 400))
        self.label_showvideo.setMaximumSize(QtCore.QSize(1000, 400))
        self.label_showvideo.setAutoFillBackground(False)
        self.label_showvideo.setStyleSheet("background-image: url(:/back/background.JPG);\n"
"background-position: center center;\n"
"background-repeat: no-repeat;\n"
"")
        self.label_showvideo.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_showvideo.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_showvideo.setObjectName("label_showvideo")
        self.graphicsView = QtWidgets.QGraphicsView(parent=self.page)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 1000, 400))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setMinimumSize(QtCore.QSize(1000, 400))
        self.graphicsView.setMaximumSize(QtCore.QSize(1000, 400))
        self.graphicsView.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.graphicsView.setAutoFillBackground(False)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
        self.graphicsView.setBackgroundBrush(brush)
        self.graphicsView.setSceneRect(QtCore.QRectF(0.0, 0.0, 0.0, 0.0))
        self.graphicsView.setObjectName("graphicsView")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.stackedWidget.addWidget(self.page_2)
        self.verticalLayout.addWidget(self.stackedWidget)
        self.horizontalSlider_videoProgress = QtWidgets.QSlider(parent=self.widget_3)
        self.horizontalSlider_videoProgress.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider_videoProgress.setObjectName("horizontalSlider_videoProgress")
        self.verticalLayout.addWidget(self.horizontalSlider_videoProgress)
        self.lineEdit_currentTime = QtWidgets.QLineEdit(parent=self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_currentTime.sizePolicy().hasHeightForWidth())
        self.lineEdit_currentTime.setSizePolicy(sizePolicy)
        self.lineEdit_currentTime.setFrame(False)
        self.lineEdit_currentTime.setObjectName("lineEdit_currentTime")
        self.verticalLayout.addWidget(self.lineEdit_currentTime)
        self.gridLayout.addWidget(self.widget_3, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ADB实时交互系统   "))
        self.pushButton_startend_detect.setText(_translate("MainWindow", "启动视觉识别"))
        self.pushButton_closedprocess.setText(_translate("MainWindow", "结束程序"))
        self.groupBox_choosevideo.setTitle(_translate("MainWindow", "视频源选择"))
        self.radioButton_camera.setText(_translate("MainWindow", "摄像头"))
        self.radioButton_local.setText(_translate("MainWindow", "本地文件"))
        self.groupBox_setshowmode.setTitle(_translate("MainWindow", "视觉识别参数设置"))
        self.groupBox_3.setTitle(_translate("MainWindow", "模型选择"))
        self.radioButton_default_model.setText(_translate("MainWindow", "默认模型"))
        self.radioButton_custom_model.setText(_translate("MainWindow", "自定义"))
        self.groupBox.setTitle(_translate("MainWindow", "帧预处理设置"))
        self.lineEdit_3.setText(_translate("MainWindow", "           切割起始位置："))
        self.lineEdit_2.setText(_translate("MainWindow", "                 切割比例："))
        self.lineEdit_.setText(_translate("MainWindow", "                 缩放比例："))
        self.groupBox_2.setTitle(_translate("MainWindow", "其它设置"))
        self.lineEdit_4.setText(_translate("MainWindow", "            Buffer_size："))
        self.lineEdit_5.setText(_translate("MainWindow", "          Skip_frames："))
        self.lineEdit_Executor.setText(_translate("MainWindow", "               线程数量："))
        self.lineEdit_WaitKey.setText(_translate("MainWindow", "                WaitKey："))
        self.groupBox_11.setTitle(_translate("MainWindow", "消息"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget), _translate("MainWindow", "视觉识别模块"))
        self.pushButton_Connect_Disconnect_CAN.setText(_translate("MainWindow", "建立CAN通信"))
        self.groupBox_4.setTitle(_translate("MainWindow", "CAN通讯类型"))
        self.comboBox_CANtype.setItemText(0, _translate("MainWindow", "虚拟CAN"))
        self.comboBox_CANtype.setItemText(1, _translate("MainWindow", "物理CAN"))
        self.groupBox_5.setTitle(_translate("MainWindow", "波特率设置"))
        self.comboBox_baud_rate.setCurrentText(_translate("MainWindow", "10000"))
        self.comboBox_baud_rate.setItemText(0, _translate("MainWindow", "10000"))
        self.comboBox_baud_rate.setItemText(1, _translate("MainWindow", "20000"))
        self.comboBox_baud_rate.setItemText(2, _translate("MainWindow", "50000"))
        self.comboBox_baud_rate.setItemText(3, _translate("MainWindow", "100000"))
        self.comboBox_baud_rate.setItemText(4, _translate("MainWindow", "125000"))
        self.comboBox_baud_rate.setItemText(5, _translate("MainWindow", "250000"))
        self.comboBox_baud_rate.setItemText(6, _translate("MainWindow", "500000"))
        self.comboBox_baud_rate.setItemText(7, _translate("MainWindow", "800000"))
        self.comboBox_baud_rate.setItemText(8, _translate("MainWindow", "1000000"))
        self.groupBox_6.setTitle(_translate("MainWindow", "发送"))
        self.groupBox_8.setTitle(_translate("MainWindow", "接收设备ID"))
        self.pushButton_start_send_CAN.setText(_translate("MainWindow", "开始 发送"))
        self.pushButton_start_receive_CAN.setText(_translate("MainWindow", "开始 接收"))
        self.groupBox_9.setTitle(_translate("MainWindow", "接收"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget_2), _translate("MainWindow", "CAN通信模块"))
        self.groupBox_7.setTitle(_translate("MainWindow", "光学分区设置"))
        self.lineEdit_7.setText(_translate("MainWindow", "行"))
        self.lineEdit_9.setText(_translate("MainWindow", "列"))
        self.lineEdit_10.setText(_translate("MainWindow", "间距"))
        self.lineEdit_11.setText(_translate("MainWindow", "间距"))
        self.pushButton_original_video.setText(_translate("MainWindow", "打开原始视频画面"))
        self.pushButton_ADBshow.setText(_translate("MainWindow", "ADB ON"))
        self.groupBox_10.setTitle(_translate("MainWindow", "消息"))
        self.pushButton_lighting_on.setText(_translate("MainWindow", "Lighting ON"))
        self.radioButton.setText(_translate("MainWindow", "Micro LED"))
        self.radioButton_2.setText(_translate("MainWindow", "Matrix LED"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget1), _translate("MainWindow", "ADB交互模块"))
        self.label_ADBexchange.setText(_translate("MainWindow", ""))
        self.label_showvideo.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:700; color:#aa007f;\"></span></p></body></html>"))
