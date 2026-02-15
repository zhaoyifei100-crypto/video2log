---
name: vision
description: |
  通用视觉能力，让 AI 能看懂图片和视频。
  
  触发场景：
  (1) 用户要求描述、分析、识别图片或视频内容
  (2) 用户要求检测、监控画面异常（黑屏、闪断、物体检测）
  (3) 用户要求找出图中特定物体的位置
  (4) 涉及屏幕监控、电视测试、示波器读数等视觉任务
---

# Vision Skill

Nanobot 的眼睛 🐱

---

## 核心 Flow

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────┐
│           1. 判定视觉模式                    │
├─────────────────────────────────────────────┤
│                                             │
│   静态关键词: 描述/识别/这是什么/有什么/位置   │
│   动态关键词: 检测/监控/盯着/黑屏/闪断/异常  │
│                                             │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   ┌─────────────┐          ┌─────────────┐
   │  静态视觉   │          │  动态视觉   │
   └──────┬──────┘          └──────┬──────┘
          │                        │
          ▼                        ▼
   ┌─────────────┐          ┌─────────────┐
   │ 直接调用    │          │  状态机     │
   │ QwenVL     │          │  INIT      │
   │            │          │    ↓       │
   │            │          │  MONITOR   │
   │            │          │    ↓       │
   │            │          │  ALERT     │
   └─────────────┘          └─────────────┘
```

---

## 模式详解

### 静态视觉

直接分析单帧图片，返回 LLM 描述。

**流程**:
```
截取帧 → 缩放(1280px) → 编码(base64) → 调用 QwenVL → 返回描述
```

**使用**:
```python
from src.static_vision import StaticVision

result = StaticVision().analyze(frame)
# 返回: "图中有一个电视屏幕，显示..."
```

---

### 动态视觉

持续监控，检测异常。需要维护状态机。

**状态机 Flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         DYNAMIC VISION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐                                                  │
│   │   INIT   │  ← 第一次调用                                    │
│   │  初始化   │    (1) 截取当前帧                               │
│   └────┬─────┘    (2) 静态分析，建立基准                        │
│        │         (3) 保存基准帧和基准描述                      │
│        ▼         (4) 进入 MONITOR                              │
│   ┌──────────┐                                                  │
│   │ MONITOR  │  ← 持续监控                                      │
│   │  监测中   │    (1) 执行 CV 函数 (亮度/边缘/帧差)            │
│   └────┬─────┘    (2) 对比基准，判断是否异常                    │
│        │         (3) 异常? → 进入 ALERT                        │
│        │         (4) 正常 → 记录结果，继续监控                  │
│        ▼                                                          │
│   ┌──────────┐    异常触发    ┌──────────┐                      │
│   │  ALERT   │ ────────────▶ │   DONE   │                      │
│   │  通知用户 │               │  结束    │                      │
│   └──────────┘               └──────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**状态存储**:
- 内存: `skill.baseline_frame`, `skill.baseline_analysis`, `skill.cv_results`
- 文件: `/tmp/vision_context.json` (可选，供 LLM 读取)

**使用**:
```python
from src.dynamic_vision import DynamicVision

skill = DynamicVision()

# 第一次 (INIT)
result = skill.process(frame)
# 返回: "已建立基准：电视屏幕在画面中央，开始监控..."

# 后续调用 (MONITOR)
result = skill.process(frame)
# 返回: "正常，画面亮度 80lux"

# 异常时 (ALERT)
result = skill.process(frame)  
# 返回: "⚠️ 异常！亮度仅 10lux，可能黑屏"
```

---

## CV 函数库

这些函数在动态视觉的 MONITOR 阶段调用。

| 函数 | 说明 | 返回 |
|------|------|------|
| `detect_brightness` | 亮度检测，判断黑屏 | `{"brightness": 80, "is_black": false}` |
| `detect_edges` | Canny 边缘检测 | `{"edge_count": 150}` |
| `find_contours` | 轮廓查找 | `{"contours": [[x,y,w,h], ...]}` |
| `calc_frame_diff` | 与基准帧的差异 | `{"diff_percent": 5.2}` |
| `detect_motion` | 运动检测 | `{"has_motion": true, "regions": [...]}` |
| `crop_region` | 裁剪 ROI | 返回裁剪后的 frame |
| `resize_frame` | 缩放 | 返回缩放后的 frame |

---

## 使用方式

### 命令行

```bash
# 静态模式 (单次)
python main.py --mode static --once --input image.jpg

# 动态模式 (持续监控)
python main.py --mode dynamic --input rtsp://192.168.1.31:8554/mjpeg
```

### 作为 Skill 调用

```python
from src import VisionSkill

skill = VisionSkill()

# 用户: "帮我看看这张图"
result = skill.process(frame, mode="static")

# 用户: "帮我盯着有没有黑屏"
result = skill.process(frame, mode="dynamic")
```

---

## 配置

环境变量:
- `SILICONFLOW_API_KEY`: LLM API 密钥

配置文件: `config/config.yaml`

---

## 依赖

```
opencv-python
requests
pyyaml
```
