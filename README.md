# video2log 视频视觉分析

定时从视频流抓取图像，调用 LLM 描述画面内容。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 (默认静态模式)
python main.py --once

# 动态模式 (OpenCV 预处理 + LLM)
python main.py --mode dynamic --once

# 循环运行
python main.py --mode dynamic --interval 30
```

## 两种模式

| 模式 | 说明 |
|------|------|
| `static` | 静态模式，直接调用 LLM 描述图像 |
| `dynamic` | 动态模式，OpenCV 预处理，异常时才调用 LLM |

## 配置

编辑 `config/config.yaml`:

```yaml
mode: "dynamic"           # 模式: dynamic / static
interval: 60              # 抓取间隔(秒)
stream_url: "http://192.168.1.15:8554/stream"

opencv:
  brightness_threshold: 30   # 亮度阈值
  dark_ratio_threshold: 0.9 # 暗像素比例

llm:
  provider: "siliconflow"
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
```

## 架构

```
┌─────────────────────────────────────────────┐
│              视频流 / 图片                    │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│         VisionProcessor                     │
│  ─────────────────────────────────────────  │
│  mode = static → 直接 LLM                   │
│  mode = dynamic → OpenCV → (异常)→ LLM      │
└────────────────────┬────────────────────────┘
                     ▼
              输出结果
```

## 文件结构

```
video2log/
├── config/
│   └── config.yaml       # 配置文件
├── src/
│   ├── vision.py         # 核心处理器
│   ├── opencv_helper.py  # OpenCV 预处理
│   ├── llm_client.py      # LLM 调用
│   ├── config.py         # 配置加载
│   └── logger.py         # 日志
├── main.py               # 入口
└── requirements.txt
```
