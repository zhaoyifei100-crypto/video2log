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

> Nanobot 自己写的视觉技能 🐱

---

## 核心亮点

**动态视觉中，LLM 会自己写 CV 代码进行监控！**

```
用户: "帮我盯着电视有没有黑屏"

Nanobot:
  1. 看一眼 → "电视在画面中央，显示正常"
  2. 自己写一段 CV 代码 → "检测中心区域亮度，低于 30 就报警"
  3. 循环执行这段代码监控
  4. 异常时通知你
```

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
   │            │          │  LLM生成代码│ ← 核心！
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

---

### 动态视觉 (核心)

**状态机 Flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         DYNAMIC VISION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐                                                  │
│   │   INIT   │  ← 第一次调用                                    │
│   │  初始化   │    (1) 截取当前帧                               │
│   └────┬─────┘    (2) LLM 静态分析 + 生成监控代码             │
│        │         (3) 进入 MONITOR                              │
│        ▼                                                          │
│   ┌──────────┐                                                  │
│   │ MONITOR  │  ← 执行 LLM 生成的代码                          │
│   │  监测中   │    (1) 用 exec() 执行代码                       │
│   └────┬─────┘    (2) 判断结果是否异常                         │
│        │         (3) 异常? → 进入 ALERT                        │
│        ▼         (4) 正常 → 记录结果，继续监控                  │
│   ┌──────────┐    异常触发    ┌──────────┐                      │
│   │  ALERT   │ ────────────▶ │   DONE   │                      │
│   │ LLM决策  │   (continue/  │  结束    │                      │
│   │          │    adjust/    │          │                      │
│   └──────────┘    stop)       └──────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## LLM 如何生成代码

INIT 阶段，LLM 会收到：

```
1. 当前画面截图
2. 可用的 CV 函数列表
3. 要求生成监控代码

LLM 返回:
{
  "description": "电视在画面中央，显示正常",
  "code": "brightness = detect_brightness(frame); result = {'brightness': brightness, 'is_anomaly': brightness < 30}"
}
```

然后 MONITOR 阶段用 `exec()` 执行这段代码！

---

## CV 函数库

LLM 可以使用的函数：

| 函数 | 说明 | 返回 |
|------|------|------|
| `detect_brightness(frame)` | 亮度检测 | `80.5` |
| `detect_dark_ratio(frame)` | 暗像素比例 | `0.02` |
| `detect_motion(frame, prev_frame)` | 运动检测 | `1500.0` |
| `detect_edges(frame)` | 边缘检测 | `{"edge_count": 150}` |
| `compare_frame(frame1, frame2)` | 帧对比 | `{"diff_score": 5.2, "is_same": false}` |
| `crop_region(frame, x1, y1, x2, y2)` | 裁剪区域 | frame |
| `frame` | 当前帧 | numpy array |
| `prev_frame` | 上一帧 | numpy array |

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
numpy
```
