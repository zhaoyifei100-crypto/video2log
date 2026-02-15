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

## 两种模式

### 静态视觉
直接分析图片/视频帧，无需持续监控。

**适用**: 描述画面、识别物体、OCR、定位

**示例**:
- "描述这张图"
- "图中有几个人"
- "找出电视屏幕位置"

### 动态视觉
持续监控，检测异常。

**适用**: 黑屏检测、闪断检测、变化检测

**流程**:
1. 截取基准帧 → LLM 分析建立基准
2. 持续监控 → CV 函数检测异常
3. 异常时 → 通知用户

## CV 函数

| 函数 | 说明 | 参数 |
|------|------|------|
| `detect_brightness` | 亮度检测，判断黑屏 | threshold: 30 |
| `detect_edges` | 边缘检测 | low: 50, high: 150 |
| `find_contours` | 轮廓查找 | - |
| `crop_region` | 裁剪区域 | x1, y1, x2, y2 |
| `resize_frame` | 调整大小 | width: 1280 |
| `calculate_histogram` | 直方图分析 | region: None |

## 使用方式

```python
# 静态模式
from src.static_vision import StaticVision
result = StaticVision().analyze(frame)

# 动态模式
from src.dynamic_vision import DynamicVision
skill = DynamicVision()
result = skill.process(frame)  # 自动状态机: INIT → MONITOR → ALERT
```

## 配置

环境变量:
- `SILICONFLOW_API_KEY`: LLM API 密钥

配置文件: `config/config.yaml`
