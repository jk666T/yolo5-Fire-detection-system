#本项目基于yolov5-fire-detection独立发展，已进行重构。保留原许可证。原作者：xiaofuqing13
# YOLOv5 火灾检测系统

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-blue) ![License](https://img.shields.io/badge/License-MIT-green)


基于 YOLOv5 的实时火灾检测系统，集成 PyQt5 图形化操作界面，支持摄像头实时检测、视频文件检测和图片检测，并具备火灾报警功能。

## 痛点与目的

- **问题**：传统消防监控依赖人眼盯视频画面，7×24 小时值守不现实，火情发现滞后导致损失扩大
- **方案**：用 YOLOv5 训练火焰检测模型，接入摄像头或视频源自动识别火灾，检测到火焰立刻报警提醒
- **效果**：PyQt5 图形界面一键操作，支持摄像头实时流、视频文件和图片三种检测模式，发现火情自动播放报警音效

## 核心功能

- **实时摄像头检测**：调用本地摄像头实时进行火灾检测
- **视频文件检测**：对录制的监控视频进行逐帧火灾识别
- **图片检测**：对单张图片进行火灾区域标注与识别
- **PyQt5 图形界面**：友好的可视化操作窗口，支持一键切换检测模式
- **火灾报警提示**：检测到火灾时自动播放报警音效
- **YOLOv5 模型训练**：提供完整的模型训练流程，支持自定义数据集训练

## 技术架构

```
输入源（摄像头 / 视频 / 图片）
    ↓
PyQt5 图形界面（window.py）
    ↓
YOLOv5 推理引擎（detect + 预训练权重）
    ↓
NMS 后处理 + 目标框绘制
    ↓
实时显示检测结果 + 火灾报警
```

## 系统要求

### 操作系统
- **Windows**：Windows 10/11（推荐）
- **Linux**：Ubuntu 18.04+
- **macOS**：macOS 10.15+

### 编程工具版本
- **Python**：3.7-3.13（当前版本：3.13.9）
- **PyTorch**：1.7.0+（当前版本：2.7.1+cu118）
- **TorchVision**：0.8.1+（当前版本：0.22.1+cu118）
- **CUDA**：10.2+（当前版本：11.8，仅 GPU 版本需要）

### 硬件要求
- **CPU**：至少 4 核
- **内存**：至少 8GB
- **GPU**：可选，支持 CUDA 的 NVIDIA 显卡（推荐，可加速推理）
- **摄像头**：用于实时检测

### 依赖库版本
- **核心依赖**：
  - matplotlib>=3.2.2（当前版本：3.10.8）
  - numpy>=1.18.5（当前版本：2.4.4）
  - opencv-python>=4.1.2（当前版本：4.13.0）
  - Pillow>=7.1.2（当前版本：12.2.0）
  - PyYAML>=5.3.1（当前版本：6.0.3）
  - requests>=2.23.0（当前版本：2.33.1）
  - scipy>=1.4.1（当前版本：1.17.1）
  - tqdm>=4.41.0（当前版本：4.67.3）
  - tensorboard>=2.4.1（当前版本：2.20.0）
  - pandas>=1.1.4（当前版本：3.0.2）
  - seaborn>=0.11.0（当前版本：0.13.2）
  - thop（当前版本：0.1.1）
- **图形界面依赖**：
  - PyQt5（当前版本：5.15.2）
- **可选依赖**：
  - CUDA 相关库（GPU 加速）
  - pycocotools（COCO 评估）

## 使用说明

### 环境安装

```bash
pip install -r yolov5-fire-42-master/requirements.txt
pip install PyQt5
```

### 启动图形界面

```bash
python window.py
```

### 训练自定义模型

```bash
cd yolov5-fire-42-master
python train.py --img 640 --batch 16 --epochs 100 --data data/fire_data.yaml --weights yolov5s.pt
```

## 项目结构

```
.
├── window.py                     # PyQt5 图形化检测界面（主程序）
├── fire_alert.wav.wav            # 火灾报警音效
├── images/                       # 示例图片
└── yolov5-fire-42-master/        # YOLOv5 核心代码
    ├── train.py                  # 模型训练
    ├── detect.py                 # 推理检测
    ├── val.py                    # 模型验证
    ├── export.py                 # 模型导出
    ├── models/                   # 网络结构定义
    ├── utils/                    # 工具函数
    ├── data/                     # 数据集配置
    ├── images/                   # 测试图片
    ├── docs/                     # 文档
    └── requirements.txt          # 依赖列表
```

## 检测效果展示

### 图片检测结果

![火灾检测结果](images/fire_detection_result.jpg)

### 视频帧检测结果

![视频检测结果](images/fire_video_result.jpg)

### 夜间场景检测

![夜间检测](images/fire_detection_night.png)

### PyQt5 操作界面

![操作界面](images/pyqt5_gui.png)

## 适用场景

- 智慧消防与火灾预警
- 工业厂房安全监控
- 森林防火巡检
- 视频监控智能分析

## 技术栈

| 组件 | 技术 |
|------|------|
| 目标检测 | YOLOv5 |
| 图形界面 | PyQt5 |
| 深度学习 | PyTorch |
| 图像处理 | OpenCV |
| 报警音效 | winsound (Windows) |

## 许可证

GPL-3.0 License
