> **注意：本项目是xiaofuqing13的重写版本，已进行大规模修改，由T_T维护，继续采用 GPL v3 协议。**
> # YOLOv5 火灾检测系统
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
pip install -r requirements.txt
pip install PyQt5
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
