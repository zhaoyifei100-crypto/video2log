# Vision Skill

> Nanobot 自己写的视觉技能 🐱

## 简介

Nanobot 拥有了"眼睛"。

这个 Skill 让 Nanobot 能看懂图片和视频：
- 静态视觉：看图说话
- 动态视觉：看监控异常

## 两种模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `--mode static` | 直接 LLM 分析 | 描述画面、识别物体 |
| `--mode dynamic` | CV 预处理 + LLM | 黑屏检测、异常监控 |

## 使用

```bash
# 静态视觉
python main.py --mode static --input image.jpg

# 动态视觉
python main.py --mode dynamic --input rtsp://...
```

## 依赖

- opencv-python
- requests
- pyyaml
