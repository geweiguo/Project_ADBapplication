import cv2
import subprocess as sp
import numpy as np
import torch
from pathlib import Path
from yolov5 import YOLOv5
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import os

frame_counter = 0
start_time = time.time()

# 选择CPU还是GPU
def select_device(device='', batch_size=0, newline=True):
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
    return torch.device(arg)

# 此为增加了人与车的标签，在画框时只有人和车
def draw_bboxes(img, results):
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) in [0, 1]:  # 只绘制人和车两个类别的检测结果
            label = f"{results.names[int(cls)]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

# 将处理每个视频帧的任务包装为一个函数
def process_frame(frame):
    global model
    results = model.predict(frame)
    return draw_bboxes(frame, results)


# 输入视频文件名
input_video = 'C:/1602646269313.mp4'

# 使用ffprobe获取视频信息
ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,r_frame_rate',
               '-of', 'csv=p=0', input_video]
info = sp.check_output(ffprobe_cmd).decode('utf-8').split(',')
width, height = map(int, info[:2])

# 设置缩放后的宽度和高度
# new_width, new_height = width // 2, height // 2

fps = round(eval(info[2]))

# 设置FFmpeg命令
command = ['ffmpeg',
           '-i', input_video,
           # '-vf', 'scale={}:{}'.format(new_width, new_height), # 设置缩放后的宽度和高度
           '-vf', 'scale={}:{}'.format(width, height),
           '-c:v', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-threads', '12',
           '-f', 'rawvideo',
           'pipe:']

device = select_device(0)

# 初始化YOLOv5模型
model_path = 'F:/12-Python/yolov5/pre_modelsfile_yolov5/yolov5s.pt'
model = YOLOv5(model_path, device=device)

# 启动FFmpeg子进程
pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

# 创建cv2.VideoCapture对象
cap = cv2.VideoCapture(input_video)

# 设置缓冲区大小
buffer_size = 10
cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

# 创建线程池
executor = ThreadPoolExecutor(max_workers=12)

skip_frames = 1  # 跳过的帧数
frame_counter = 0

while True:
    ret, frame = cap.read()


    if not ret:
        break

    if not ret or frame_counter % skip_frames != 0:
        frame_counter += 1
        continue


    # 裁切画面为原高度的60%
    height_crop = int(height * 0.8)
    crop_top = int(height * 0.1)
    frame = frame[crop_top:crop_top + height_crop, :, :]

    # 计时
    frame_counter += 1
    if frame_counter % 10 == 0:  # 每10帧输出一次信息
        elapsed_time = time.time() - start_time
        print(f"Processed frames: {frame_counter}, elapsed time: {elapsed_time:.2f}s")

    # 使用线程池处理帧
    future = executor.submit(process_frame, frame)
    processed_frame = future.result()

    # 通过threading模块来查看程序运行时的实际线程数量
    active_threads = threading.active_count()
    print(f"Active threads: {active_threads}")

    # # 使用YOLOv5模型进行目标检测
    # results = model.predict(frame)
    #
    # # 在帧上绘制检测框
    # frame = draw_bboxes(frame, results)

    # 显示解码后的帧
    cv2.imshow('Video', frame)
    if cv2.waitKey(int(1000/ fps)) & 0xFF == ord('q'):
        break

    pipe.stdout.flush()

cv2.destroyAllWindows()

# 确保子进程已经结束并释放资源
pipe.communicate()