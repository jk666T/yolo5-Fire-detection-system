> **注意：本项目是xiaofuqing13的重写版本，已进行大规模修改，由jk666T维护，继续采用 GPL v3 协议。**
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

# YOLOv5 火灾检测系统 - 改进功能说明

> 基于 yolov5modelfire 版本相对于 yolov5-fire-detection-master 的改进点

---

## 1. 智能报警机制 ⭐

### 问题
原版本检测到火焰立即报警，容易产生误报。

### 改进方案
```python
# 音频播放冷却机制
self.last_alert_time = 0
self.alert_cooldown = 20  # 20秒冷却期

# 连续检测机制
self.fire_detections = []  # 存储最近的火灾检测时间
self.required_detections = 2  # 需要连续检测的次数
self.detection_window = 1.0  # 检测时间窗口（秒）
```

### 效果
- 需要在 **1秒时间窗口内连续2次检测到火焰** 才触发报警
- 报警后有 **20秒冷却期**，避免重复报警

---

## 2. 模型加载容错处理 ⭐

### 问题
原版本模型加载失败会导致程序直接崩溃。

### 改进方案
```python
try:
    self.model = self.model_load(weights="runs/train/exp_yolov5s/weights/best.pt",
                                 device=self.device)
except Exception as e:
    self.model = None
    print(f"模型加载失败: {e}")
```

### 效果
- 捕获加载异常，程序继续运行
- 打印友好错误提示，便于排查问题

---

## 3. 目录自动创建 ⭐

### 问题
原版本需要手动创建 tmp 目录，否则可能报错。

### 改进方案
```python
os.makedirs(os.path.join('images', 'tmp'), exist_ok=True)
```

### 效果
- 启动时自动创建所需目录
- `exist_ok=True` 避免目录已存在时报错

---

## 4. 音频播放模块优化 ⭐

### 问题
原版本音频播放模块功能单一。

### 改进方案
```python
from PyQt5.QtMultimedia import QSound
```

### 效果
- 引入 Qt 多媒体库，支持更丰富的音频播放功能

---

## 5. 依赖库完整声明 ⭐

### 问题
原版本 requirements.txt 中 PyTorch 和 PyQt5 被注释或缺失。

### 改进方案
```txt
# 明确声明所有运行时依赖
torch>=1.7.0
torchvision>=0.8.1
PyQt5>=5.15.2
```

### 效果
- 环境搭建更简单，一键安装所有依赖
- 避免因依赖缺失导致的运行错误

---

## 6. 详细系统要求文档 ⭐

### 问题
原版本缺少运行环境说明。

### 改进方案
README.md 新增完整系统要求：

| 组件 | 版本要求 |
|------|----------|
| Python | 3.7-3.13 |
| PyTorch | 1.7.0+ (实测 2.7.1+cu118) |
| CUDA | 10.2+ (实测 11.8) |
| 内存 | 至少 8GB |
| GPU | 可选，推荐支持 CUDA 的 NVIDIA 显卡 |

### 效果
- 用户可快速判断环境是否满足要求
- 减少环境配置问题的发生

---

## 7. 数据集路径配置 ⭐

### 改进说明
fire_data.yaml 同时支持相对路径和绝对路径配置：

```yaml
# 相对路径（便携性强）
train: data/images/train
val: data/images/val

# 绝对路径（适合训练场景）
train: C:/Users/86178/Desktop/42/demo/fire_yolo_format/images/train
val: C:/Users/86178/Desktop/42/demo/fire_yolo_format/images/val
```

### 效果
- 相对路径版本适合直接部署
- 绝对路径版本方便开发者定位原始训练数据

---

## 📋 改进功能清单

| 序号 | 功能 | 改进类型 |
|------|------|----------|
| 1 | 智能报警机制 | 核心功能 |
| 2 | 模型加载容错处理 | 稳定性 |
| 3 | 目录自动创建 | 便利性 |
| 4 | 音频播放模块优化 | 功能增强 |
| 5 | 依赖库完整声明 | 环境配置 |
| 6 | 详细系统要求文档 | 文档完善 |
| 7 | 数据集路径配置 | 配置灵活 |


## 许可证

GPL-3.0 License
