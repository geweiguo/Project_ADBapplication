# 程序实现的功能：
# 这个程序实现了一个实时的视频目标检测应用。它使用了 YOLOv5 模型来检测输入视频中的人和车辆。程序读取输入视频，将视频帧批量发送到 YOLOv5 模型进行检测，然后在检测到的目标上绘制边界框。最后，处理后的视频帧将显示在屏幕上。
#
# 采用了哪些方案和技术路线：
#
# 方案2：使用线程池（ThreadPoolExecutor）来加速模型处理过程。通过将模型处理任务并行处理，可以提高程序执行的速度。
# 方案5：使用批处理。将多个帧组合成一个批次，然后在单次模型调用中对整个批次进行处理。这可以提高模型的吞吐量，因为一次处理多个帧比逐个处理更加高效。
# 这个程序的优缺点和可以改进的地方：
# 优点：
# 程序使用了线程池，可以充分利用多核处理器的性能，提高处理速度。
# 程序使用了批处理，提高了模型的吞吐量。
# 缺点：
#
# 对于不同大小的输入帧，需要确保在批处理之前对它们进行调整，以便它们具有相同的尺寸。
# 当处理的帧数较大时，可能会遇到内存问题。程序可能需要进行内存优化。
# 可以改进的地方：
#
# 添加一个适当的缩放和预处理步骤，以确保输入帧具有相同的尺寸。
# 可以通过限制缓冲区大小或实现更有效的内存管理策略来解决内存问题。
# 为了更好地支持实时处理，可以考虑使用多线程或异步技术来处理不同的任务，例如视频帧的获取、处理和显示。这样可以进一步提高程序的性能。


import cv2
import subprocess as sp
import numpy as np
import torch
from pathlib import Path

import yolov5
from yolov5 import YOLOv5
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from queue import Queue

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

device = select_device(0)

# 初始化YOLOv5模型
model_path = 'F:/12-Python/yolov5/pre_modelsfile_yolov5/yolov5s.pt'
model = yolov5.load(model_path, device=device)

frame_queue = Queue(maxsize=10)


# 更改原来的 process_frame 函数，以便它接受一个批量的帧
def process_frames(frames):
    results = model(frames, size=640) # 一次处理整个批次的帧
    processed_frames = []
    for i in range(len(frames)):
        processed_frame = draw_bboxes(frames[i], results, i)
        processed_frames.append(processed_frame)
    return processed_frames

# 此为增加了人与车的标签，在画框时只有人和车
def draw_bboxes(img, results, index):
    if index < len(results.xyxy):
        for *box, conf, cls in results.xyxy[index]:
            if int(cls) in [0, 1]:  # 只绘制人和车两个类别的检测结果
                label = f"{results.names[int(cls)]} {conf:.2f}"
                x1, y1, x2, y2 = map(int, box)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img



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

frames_to_process = []  # 保存要处理的帧

while True:
    ret, frame = cap.read()


    if not ret:
        break

    if frame_counter % skip_frames != 0:
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

    # 将帧添加到队列中
    frame_queue.put(frame)

    # 通过threading模块来查看程序运行时的实际线程数量
    active_threads = threading.active_count()
    print(f"Active threads: {active_threads}")

    # 如果队列已满，就将所有帧发送到线程池进行处理
    if frame_queue.qsize() == 10:
        # 从队列中取出帧，并将它们转换为 numpy 数组
        frames_to_process = [frame_queue.get() for _ in range(10)]
        frames_np = np.stack(frames_to_process, axis=0)

        # 提交任务到线程池，并等待结果
        future = executor.submit(process_frames, frames_np)
        processed_frames = future.result()

        # 显示处理后的帧
        for processed_frame in processed_frames:
            cv2.imshow('Video', processed_frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    pipe.stdout.flush()

cv2.destroyAllWindows()

# 确保子进程已经结束并释放资源
pipe.communicate()

