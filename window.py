# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-jungong
File Name: window.py.py
Author: chenming
Create Date: 2021/11/8
Description：图形化界面，可以检测摄像头、视频和图片文件
-------------------------------------------------
"""
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QSound
import threading
import glob
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import traceback
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # script directory
# ensure script dir is on sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try to find a directory that contains a 'models' package and add it to sys.path.
# Search immediate subdirectories first (common for nested yolov5 copies), then search upward.
models_parent = None
try:
    for sub in ROOT.iterdir():
        if sub.is_dir() and (sub / 'models').exists():
            models_parent = sub
            break
except Exception:
    models_parent = None

if models_parent is None:
    p = ROOT
    for _ in range(6):
        if (p / 'models').exists():
            models_parent = p
            break
        if p.parent == p:
            break
        p = p.parent

if models_parent:
    sp = str(models_parent)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# use relative ROOT for UI paths
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync


# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('火灾检测系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/xf.jpg"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        # 音频播放冷却机制
        self.last_alert_time = 0
        self.alert_cooldown = 20  # 20秒冷却期
        # 连续检测机制
        self.fire_detections = []  # 存储最近的火灾检测时间
        self.required_detections = 2  # 需要连续检测的次数
        self.detection_window = 1.0  # 检测时间窗口（秒）
        # ensure tmp dir exists for logs/results
        os.makedirs(os.path.join('images', 'tmp'), exist_ok=True)
        try:
            self.model = self.model_load(weights="runs/train/exp_yolov5s/weights/best.pt",
                                         device=self.device)  # todo 指明模型加载的位置的设备
        except Exception as e:
            self.model = None
            print(f"模型加载失败: {e}")
        self.initUI()
        self.reset_vid()

    '''
    ***模型初始化***
    '''

    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}" 
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        det_img_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}" 
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}" \
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}" \
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}" 
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用目标检测系统\n\n')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>/a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        # self.addTab(about_widget, '联系我')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        # self.setTabIcon(2, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            if im0 is None:
                QMessageBox.warning(self, "读取失败", "无法读取刚上传的图片，请检查文件是否有效。")
                return
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            # use the local temp copy for prediction to avoid remote/original path issues
            self.img2predict = save_path
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    '''
    ***检测图片***
    '''

    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        # print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            # If the provided source path doesn't exist, try common tmp fallbacks in images/tmp
            if not os.path.exists(source):
                # look for tmp_upload.* first
                tmp_candidates = glob.glob(os.path.join('images', 'tmp', 'tmp_upload.*'))
                if tmp_candidates:
                    source = tmp_candidates[-1]
                else:
                    upload_show = os.path.join('images', 'tmp', 'upload_show_result.jpg')
                    single_result = os.path.join('images', 'tmp', 'single_result.jpg')
                    if os.path.exists(upload_show):
                        source = upload_show
                    elif os.path.exists(single_result):
                        source = single_result
                    else:
                        QMessageBox.warning(self, "文件未找到", f"未找到要检测的图片：{source}。请重新上传后再检测。")
                        return
            # final sanity check: ensure image is readable
            img_check = cv2.imread(source)
            if img_check is None:
                QMessageBox.warning(self, "读取失败", f"无法读取图片：{source}，请确认文件有效。")
                return
            device = select_device(self.device)
            webcam = False
            if model is None:
                QMessageBox.warning(self, "未加载模型", "模型未正确加载，无法进行检测。")
                return
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images

            # Special-case: single local image — run inference directly to avoid loader read issues
            ext = source.split('.')[-1].lower()
            if os.path.isfile(source) and ext in IMG_FORMATS:
                try:
                    im0 = img_check  # BGR
                    # Letterbox and prepare
                    img = letterbox(im0, imgsz, stride=stride, auto=pt and not jit)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
                    img = np.ascontiguousarray(img)

                    im = torch.from_numpy(img).to(device)
                    im = im.half() if half and device.type != 'cpu' else im.float()
                    im /= 255.0
                    if len(im.shape) == 3:
                        im = im.unsqueeze(0)

                    # Warmup
                    if pt and device.type != 'cpu':
                        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))

                    # Inference
                    pred = model(im, augment=augment, visualize=visualize)
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Process predictions (single image)
                    for i, det in enumerate(pred):
                        p = source
                        im0 = im0.copy()
                        s = ''
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                        if len(det):
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                            print(f"检测到 {len(det)} 个目标")
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()
                                class_name = names[int(c)]
                                s += f"{n} {class_name}{'s' * (n > 1)}, "
                                print(f"检测到 {n} 个 {class_name}")
                                # 检测到火灾时播放警报
                                if class_name == 'fire':
                                    import time
                                    current_time = time.time()
                                    # 检查是否在冷却期内
                                    if current_time - self.last_alert_time > self.alert_cooldown:
                                        # 图片检测：只需要一次检测
                                        if not webcam:
                                            print("检测到火灾（图片），准备播放警报")
                                            # 更新最后播放时间
                                            self.last_alert_time = current_time
                                            # 清空检测记录
                                            self.fire_detections = []
                                            # 异步播放音频，避免阻塞主线程
                                            def play_alert():
                                                try:
                                                    # 获取当前脚本所在目录
                                                    script_dir = os.path.dirname(os.path.abspath(__file__))
                                                    # 音频文件路径
                                                    alert_path = os.path.join(script_dir, 'fire_alert.wav.wav')
                                                    print(f"尝试播放音频: {alert_path}")
                                                    if os.path.exists(alert_path):
                                                        print(f"音频文件存在，大小: {os.path.getsize(alert_path)} 字节")
                                                        # 优先使用 winsound
                                                        try:
                                                            import winsound
                                                            print("使用 winsound 播放音频")
                                                            # 使用异步播放，避免阻塞
                                                            winsound.PlaySound(alert_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                                                            print("使用 winsound 播放成功")
                                                        except Exception as e:
                                                            print(f"winsound 失败: {e}")
                                                            # 尝试使用 QSound
                                                            try:
                                                                from PyQt5.QtMultimedia import QSound
                                                                alert = QSound(alert_path)
                                                                print(f"QSound 创建成功")
                                                                alert.play()
                                                                print("音频播放命令已发送")
                                                            except Exception as e:
                                                                print(f"QSound 失败: {e}")
                                                    else:
                                                        print(f"音频文件不存在: {alert_path}")
                                                except Exception as e:
                                                    print(f"播放音频失败: {e}")
                                            threading.Thread(target=play_alert).start()
                                        # 视频检测：需要连续检测
                                        else:
                                            # 添加当前检测时间到列表
                                            self.fire_detections.append(current_time)
                                            # 过滤掉时间窗口外的检测
                                            self.fire_detections = [t for t in self.fire_detections if current_time - t <= self.detection_window]
                                            print(f"火灾检测记录: {len(self.fire_detections)} 次在 {self.detection_window} 秒内")
                                            # 检查是否满足连续检测条件
                                            if len(self.fire_detections) >= self.required_detections:
                                                print("检测到火灾（视频），准备播放警报")
                                                # 更新最后播放时间
                                                self.last_alert_time = current_time
                                                # 清空检测记录
                                                self.fire_detections = []
                                                # 异步播放音频，避免阻塞主线程
                                                def play_alert():
                                                    try:
                                                        # 获取当前脚本所在目录
                                                        script_dir = os.path.dirname(os.path.abspath(__file__))
                                                        # 音频文件路径
                                                        alert_path = os.path.join(script_dir, 'fire_alert.wav.wav')
                                                        print(f"尝试播放音频: {alert_path}")
                                                        if os.path.exists(alert_path):
                                                            print(f"音频文件存在，大小: {os.path.getsize(alert_path)} 字节")
                                                            # 优先使用 winsound
                                                            try:
                                                                import winsound
                                                                print("使用 winsound 播放音频")
                                                                # 使用异步播放，避免阻塞
                                                                winsound.PlaySound(alert_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                                                                print("使用 winsound 播放成功")
                                                            except Exception as e:
                                                                print(f"winsound 失败: {e}")
                                                                # 尝试使用 QSound
                                                                try:
                                                                    from PyQt5.QtMultimedia import QSound
                                                                    alert = QSound(alert_path)
                                                                    print(f"QSound 创建成功")
                                                                    alert.play()
                                                                    print("音频播放命令已发送")
                                                                except Exception as e:
                                                                    print(f"QSound 失败: {e}")
                                                        else:
                                                            print(f"音频文件不存在: {alert_path}")
                                                    except Exception as e:
                                                        print(f"播放音频失败: {e}")
                                                threading.Thread(target=play_alert).start()
                                            else:
                                                print("检测到火灾，但未达到连续检测次数要求")
                                    else:
                                        print(f"在冷却期内，跳过音频播放 (剩余 {self.alert_cooldown - (current_time - self.last_alert_time):.1f} 秒)")

                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)

                                if save_img or save_crop or view_img:
                                    c = int(cls)
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))

                        LOGGER.info(f'{s}Done.')
                        im0 = annotator.result()
                        resize_scale = output_size / im0.shape[0]
                        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                        cv2.imwrite(os.path.join('images', 'tmp', 'single_result.jpg'), im0)
                        self.right_img.setPixmap(QPixmap(os.path.join('images', 'tmp', 'single_result.jpg')))
                    return
                except Exception:
                    tb = traceback.format_exc()
                    log_path = os.path.join('images', 'tmp', 'detect_error.log')
                    try:
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, 'w', encoding='utf-8') as f:
                            f.write(tb)
                    except Exception:
                        pass
                    QMessageBox.critical(self, '检测出错', f'检测过程中发生错误，已记录到 {log_path}。请将该文件内容粘贴给我以便排查。')
                    print(tb)
                    return

            # Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            try:
                # Run inference
                if pt and device.type != 'cpu':
                    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
                dt, seen = [0.0, 0.0, 0.0], 0
                for path, im, im0s, vid_cap, s in dataset:
                    t1 = time_sync()
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1
                    # Inference
                    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2
                    # NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    dt[2] += time_sync() - t3
                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))

                        # Print time (inference-only)
                        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                        # Stream results
                        im0 = annotator.result()
                        # if view_img:
                        #     cv2.imshow(str(p), im0)
                        #     cv2.waitKey(1)  # 1 millisecond
                        # Save results (image with detections)
                        resize_scale = output_size / im0.shape[0]
                        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                        cv2.imwrite("images/tmp/single_result.jpg", im0)
                        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
                        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            except Exception:
                tb = traceback.format_exc()
                log_path = os.path.join('images', 'tmp', 'detect_error.log')
                try:
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write(tb)
                except Exception:
                    pass
                QMessageBox.critical(self, '检测出错', f'检测过程中发生错误，已记录到 {log_path}。请将该文件内容粘贴给我以便排查。')
                print(tb)
                return

    # 视频检测，逻辑基本一致，有两个功能，分别是检测摄像头的功能和检测视频文件的功能，先做检测摄像头的功能。

    '''
    ### 界面关闭事件 ### 
    '''

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):
        # pass
        model = self.model
        output_size = self.output_size
        # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    print(f"检测到 {len(det)} 个目标")
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        class_name = names[int(c)]
                        s += f"{n} {class_name}{'s' * (n > 1)}, "  # add to string
                        print(f"检测到 {n} 个 {class_name}")
                        # 检测到火灾时播放警报
                        if class_name == 'fire':
                            import time
                            current_time = time.time()
                            # 检查是否在冷却期内
                            if current_time - self.last_alert_time > self.alert_cooldown:
                                # 图片检测：只需要一次检测
                                if not webcam:
                                    print("检测到火灾（图片），准备播放警报")
                                    # 更新最后播放时间
                                    self.last_alert_time = current_time
                                    # 清空检测记录
                                    self.fire_detections = []
                                    # 异步播放音频，避免阻塞主线程
                                    def play_alert():
                                        try:
                                            # 获取当前脚本所在目录
                                            script_dir = os.path.dirname(os.path.abspath(__file__))
                                            # 音频文件路径
                                            alert_path = os.path.join(script_dir, 'fire_alert.wav.wav')
                                            print(f"尝试播放音频: {alert_path}")
                                            if os.path.exists(alert_path):
                                                print(f"音频文件存在，大小: {os.path.getsize(alert_path)} 字节")
                                                # 优先使用 winsound
                                                try:
                                                    import winsound
                                                    print("使用 winsound 播放音频")
                                                    # 使用异步播放，避免阻塞
                                                    winsound.PlaySound(alert_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                                                    print("使用 winsound 播放成功")
                                                except Exception as e:
                                                    print(f"winsound 失败: {e}")
                                                    # 尝试使用 QSound
                                                    try:
                                                        from PyQt5.QtMultimedia import QSound
                                                        alert = QSound(alert_path)
                                                        print(f"QSound 创建成功")
                                                        alert.play()
                                                        print("音频播放命令已发送")
                                                    except Exception as e:
                                                        print(f"QSound 失败: {e}")
                                            else:
                                                print(f"音频文件不存在: {alert_path}")
                                        except Exception as e:
                                            print(f"播放音频失败: {e}")
                                    threading.Thread(target=play_alert).start()
                                # 视频检测：需要连续检测
                                else:
                                    # 添加当前检测时间到列表
                                    self.fire_detections.append(current_time)
                                    # 过滤掉时间窗口外的检测
                                    self.fire_detections = [t for t in self.fire_detections if current_time - t <= self.detection_window]
                                    print(f"火灾检测记录: {len(self.fire_detections)} 次在 {self.detection_window} 秒内")
                                    # 检查是否满足连续检测条件
                                    if len(self.fire_detections) >= self.required_detections:
                                        print("检测到火灾（视频），准备播放警报")
                                        # 更新最后播放时间
                                        self.last_alert_time = current_time
                                        # 清空检测记录
                                        self.fire_detections = []
                                        # 异步播放音频，避免阻塞主线程
                                        def play_alert():
                                            try:
                                                # 获取当前脚本所在目录
                                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                                # 音频文件路径
                                                alert_path = os.path.join(script_dir, 'fire_alert.wav.wav')
                                                print(f"尝试播放音频: {alert_path}")
                                                if os.path.exists(alert_path):
                                                    print(f"音频文件存在，大小: {os.path.getsize(alert_path)} 字节")
                                                    # 优先使用 winsound
                                                    try:
                                                        import winsound
                                                        print("使用 winsound 播放音频")
                                                        # 使用异步播放，避免阻塞
                                                        winsound.PlaySound(alert_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                                                        print("使用 winsound 播放成功")
                                                    except Exception as e:
                                                        print(f"winsound 失败: {e}")
                                                        # 尝试使用 QSound
                                                        try:
                                                            from PyQt5.QtMultimedia import QSound
                                                            alert = QSound(alert_path)
                                                            print(f"QSound 创建成功")
                                                            alert.play()
                                                            print("音频播放命令已发送")
                                                        except Exception as e:
                                                            print(f"QSound 失败: {e}")
                                                else:
                                                    print(f"音频文件不存在: {alert_path}")
                                            except Exception as e:
                                                print(f"播放音频失败: {e}")
                                        threading.Thread(target=play_alert).start()
                                    else:
                                        print("检测到火灾，但未达到连续检测次数要求")
                            else:
                                print(f"在冷却期内，跳过音频播放 (剩余 {self.alert_cooldown - (current_time - self.last_alert_time):.1f} 秒)")

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                            #                  BGR=True)
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results
                # Save results (image with detections)
                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # self.vid_img
                # if view_img:
                # cv2.imshow(str(p), im0)
                # self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # cv2.waitKey(1)  # 1 millisecond
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                self.reset_vid()
                break
        # self.reset_vid()

    '''
    ### 界面重置事件 ### 
    '''

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_source = '0'
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
