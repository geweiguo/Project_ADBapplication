import torch
from yolov5 import YOLOv5
import os
import cv2

#1. 加载模型
def load_model(model_path, device):
    """
    加载预训练的YOLOv5模型。
    :param model_path: 模型文件路径
    :param device: 计算设备（CPU或GPU）
    :return: 加载的模型实例
    """
    return YOLOv5(model_path, device=device)

#2. 设置GPU


def select_device(device='', batch_size=0, newline=True):
    """
    根据用户输入选择使用的设备（CPU或GPU）。
    :param device: 设备名称
    :param batch_size: 批处理大小
    :param newline: 是否换行
    :return: torch.device实例
    """
    s = f'YOLOv5 🚀 {torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = 'cuda:0'
    else:
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    print(s)
    print(f"Selected device: {arg}")  # 添加调试输出
    return torch.device(arg)

#3. 获取视频源
def get_video_info(input_video):
    """
    使用OpenCV获取输入视频的相关信息（宽度、高度、帧率）。
    :param input_video: 输入视频文件路径
    :return: 视频宽度、高度和帧率
    """
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return width, height, fps

# 4. 预处理视频帧
def preprocess_frame(frame, height, settings):

    """
    对输入的视频帧进行裁剪。
    :param frame: 输入的视频帧
    :param height: 视频帧高度
    :param crop_ratio: 裁剪比例
    :param crop_top_ratio: 顶部裁剪比例
    :return: 裁剪后的视频帧
    """
    crop_ratio = settings.crop_ratio
    crop_top_ratio = settings.crop_top_ratio
    scale_ratio = settings.scale_ratio

    height_crop = int(height * crop_ratio)
    crop_top = int(height * crop_top_ratio)
    frame = frame[crop_top:crop_top + height_crop, :, :]

    # 缩放操作
    new_width = int(frame.shape[1] * scale_ratio)
    new_height = int(frame.shape[0] * scale_ratio)
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return frame

# 5. 运行目标检测
def process_frame(frame, model):
    """
    使用YOLOv5模型对输入的视频帧进行目标检测。
    :param frame: 输入的视频帧
    :param model: YOLOv5模型实例
    :return: 绘制边界框后的视频帧
    """
    results = model.predict(frame)
    return draw_bboxes(frame, results)

# 6. 后处理
def draw_bboxes(img, results):
    """
    在输入的图像上绘制检测到的边界框。
    :param img: 输入的图像
    :param results: 检测结果
    :return: 绘制边界框后的图像
    """
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) in [0, 1]:  # 只绘制人和车两个类别的检测结果
            label = f"{results.names[int(cls)]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img
