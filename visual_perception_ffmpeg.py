"""
1.1 本程序实现的功能：使用YOLOv5对输入的视频文件进行目标检测，并实时显示检测结果
1.2 图像帧处理和视觉识别推理分别采用的是什么计算方案：图像帧处理使用CPU，视觉识别推理使用GPU
1.3 版本：1.0，日期：2023-04-11


代码已按照您的要求进行了整理和添加备注。每个函数的功能如下：

1. 加载模型：`load_model` - 加载预训练的YOLOv5模型。
2. 设置GPU：`select_device` - 根据用户输入选择使用的设备（CPU或GPU）。
3. 获取视频源：`get_video_info` - 使用`ffprobe`获取输入视频的相关信息（宽度、高度、帧率）。
4. 预处理视频帧：`preprocess_frame` - 对输入的视频帧进行裁剪。
5. 运行目标检测：`process_frame` - 使用YOLOv5模型对输入的视频帧进行目标检测。
6. 后处理：`draw_bboxes` - 在输入的图像上绘制检测到的边界框。
7. 显示结果：已集成在主程序流程中，使用`cv2.imshow`实时显示处理后的视频帧。
8. 释放资源：已集成在主程序流程中，使用`cv2.destroyAllWindows`和`pipe.communicate`释放资源。

"""

import cv2
import subprocess as sp
import torch
from yolov5 import YOLOv5
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import os

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
    return torch.device(arg)

#3. 获取视频源
def get_video_info(input_video):
    """
    使用`ffprobe`获取输入视频的相关信息（宽度、高度、帧率）。
    :param input_video: 输入视频文件路径
    :return: 视频宽度、高度和帧率
    """
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,r_frame_rate',
                   '-of', 'csv=p=0', input_video]
    info = sp.check_output(ffprobe_cmd).decode('utf-8').split(',')
    width, height = map(int, info[:2])
    fps = round(eval(info[2]))
    return width, height, fps

# 4. 预处理视频帧
def preprocess_frame(frame, height, crop_ratio=1, crop_top_ratio=0):
    """
    对输入的视频帧进行裁剪。
    :param frame: 输入的视频帧
    :param height: 视频帧高度
    :param crop_ratio: 裁剪比例
    :param crop_top_ratio: 顶部裁剪比例
    :return: 裁剪后的视频帧
    """
    height_crop = int(height * crop_ratio)
    crop_top = int(height * crop_top_ratio)
    frame = frame[crop_top:crop_top + height_crop, :, :]
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

# 7. 后处理
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

# 主程序流程
if __name__ == '__main__':
    # 1.获取视频源信息
    input_video = 'C:/1602646269313.mp4'
    width, height, fps = get_video_info(input_video)

    # 2.设置模拟设备
    device = select_device(0)

    # 3.加载模型
    model_path = 'F:/12-Python/yolov5/pre_modelsfile_yolov5/yolov5s.pt'
    model = load_model(model_path, device)

    # 4.设置FFmpeg命令
    """
    从输入的视频文件中读取原始视频帧数据
    设置FFmpeg命令的目的是为了构建一个命令行命令，这个命令用于启动一个子进程（pipe）。
    子进程的作用是在后台运行FFmpeg，以便实时从输入的视频文件中读取并处理原始视频帧数据。
    """
    command = ['ffmpeg',
               '-i', input_video,
                # '-vf', 'scale={}:{}'.format(new_width, new_height), # 设置缩放后的宽度和高度
               '-vf', 'scale={}:{}'.format(width, height),
               '-c:v', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-threads', '6',
               '-f', 'rawvideo',
               'pipe:']

    # 5.pipe子进程的作用是在后台运行FFmpeg，以便实时从输入的视频文件中读取并处理原始视频帧数据
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

    """
    6.cap:
    创建一个VideoCapture对象cap，并以input_video为参数，即输入视频文件的路径。
    VideoCapture对象可以从输入视频中读取帧数据，这对于实时处理视频帧非常有用。
    """
    cap = cv2.VideoCapture(input_video)

    """
    buffer_size
    用于设置VideoCapture对象的内部缓冲区大小。
    较大的缓冲区大小可以减少从视频文件中读取帧数据的延迟，从而提高视频处理的效率。
    """
    buffer_size = 30

    """
    7.cap.set()方法:
    设置VideoCapture对象的缓冲区大小。通过调用cap.set()方法并传入两个参数：
    cv2.CAP_PROP_BUFFERSIZE（表示要设置的属性是缓冲区大小）
    buffer_size（要设置的缓冲区大小值）。
    """
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)


    """
    8.executor
    创建一个线程池，最大线程数为6。线程池用于并发地执行多个任务，这里用于并发处理视频帧的目标检测任务。
    """
    executor = ThreadPoolExecutor(max_workers=10)

    """
    9.skip_frames
    设置跳过的帧数。如果希望处理输入视频的每一帧，则将此值设置为1。
    如果希望跳过部分帧，可以将此值增大。这样可以减少处理的帧数，提高处理速度，但可能会导致检测结果不那么准确。
    """
    skip_frames = 1

    """
    初始化帧计数器
    """
    frame_counter = 0

    """
    10.frame_futures = [] 的作用是创建一个空列表，用于存储处理视频帧任务的future对象。
    future对象代表着一个尚未完成的计算任务，当任务完成后，可以通过future.result()获取任务的结果。

    将frame_futures = []放在这个位置的原因是，在开始处理视频帧的循环之前，需要初始化这个列表。
    在后续的循环中，每处理一个视频帧，都会将该帧对应的future对象添加到frame_futures列表中。
    当列表中的future对象数量超过缓冲区大小（buffer_size）时，便从列表中取出第一个future对象，
    等待任务完成，并获取处理结果。这种方式可以实现并发处理多个视频帧，提高程序的执行效率。
    """
    frame_futures = []

    processed_frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if not ret or frame_counter % skip_frames != 0:
            frame_counter += 1 # 是用于跳过一些帧以加速处理。当检测到该帧不是要处理的帧时，就直接跳过，不进行处理。
            continue

         # 11.首帧显示时记录时间
        if frame_counter == 1:
            start_timeProcessed = time.time()

        frame = preprocess_frame(frame, height)

        frame_counter += 1

        if frame_counter % 10 == 0:
            end_time = time.time()
            elapsed_Processedtime = end_time - start_timeProcessed
            print(f"Processed frames: {frame_counter}, Processed time_: {elapsed_Processedtime:.2f}s")

        future = executor.submit(process_frame, frame, model)
        frame_futures.append(future)

        # 12.当有足够的帧缓冲时，收集第一个任务的结果
        if len(frame_futures) > buffer_size:
            processed_frame = frame_futures.pop(0).result()
            processed_frame_counter += 1

            # 13.首帧显示时记录时间
            if processed_frame_counter == 1:
                start_time = time.time()

            current_time = time.time()
            elapsed_time = current_time - start_time

            #14.显示线程数量
            active_threads = threading.active_count()
            print(f"Active threads: {active_threads}")

            # 15.在屏幕上实时显示已经处理的帧数和花费的时间
            cv2.putText(frame, f"frame: {frame_counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2)
            cv2.putText(frame, f"video.time: {elapsed_time:.2f}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            # 16.显示处理后的帧
            cv2.imshow('Video', processed_frame)

            # time.sleep(1 / fps)

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

            pipe.stdout.flush()

        # 最后一帧显示时计算经过的时间
        if frame_counter == 470:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time by frame_counter 470: {elapsed_time:.2f}s")
            break



    # 释放资源
    # out.release()
    cap.release()

    cv2.destroyAllWindows()

    pipe.stdout.flush()

    pipe.communicate()


