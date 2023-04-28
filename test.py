import sys
import random
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QBrush, QImage, QPixmap, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget, \
    QGraphicsPixmapItem


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QGraphicsView Demo")
        self.setGeometry(100, 100, 1200, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.graphicsView = QGraphicsView()
        layout.addWidget(self.graphicsView)

        # 设置 QGraphicsView 和 QGraphicsScene 的背景颜色为透明
        self.graphicsView.setBackgroundBrush(QColor(25, 7, 158, 255))
        scene = QGraphicsScene()
        scene.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))
        self.graphicsView.setScene(scene)

        # 创建 1000x350 像素的 QImage，并填充半透明白色（透明度 50%）
        self.image = QImage(1000, 350, QImage.Format.Format_ARGB32)
        self.image.fill(QColor(255, 255, 255, 128))

        # 将 QImage 显示在 QGraphicsView 中
        self.pixmap_item = QGraphicsPixmapItem()
        self.graphicsView.scene().addItem(self.pixmap_item)

        # 设置 QGraphicsView 的 viewport 透明
        self.graphicsView.viewport().setAutoFillBackground(False)
        self.graphicsView.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置定时器以更新 QImage
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)  # 每1000毫秒（1秒）更新一次

    def get_virtual_can_data(self):
        # 模拟从虚拟 CAN 总线获取数据
        x = random.randint(0, 1000)
        y = random.randint(0, 350)
        width = random.randint(10, 200)
        height = random.randint(10, 100)
        return x, y, width, height

    def update_image(self):
        # 清除之前的内容
        self.image.fill(QColor(255, 255, 255, 128))

        # 获取虚拟 CAN 数据
        x, y, width, height = self.get_virtual_can_data()

        # 在 QImage 上绘制矩形
        painter = QPainter(self.image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(x, y, width, height, QColor(255, 255, 255, 128))
        painter.end()

        # 更新 QGraphicsView 显示
        pixmap = QPixmap.fromImage(self.image)
        self.pixmap_item.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    mainWin = MainWindow()
    mainWin.show()

    sys.exit(app.exec())

