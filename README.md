# video2log
# 定时拍照 + LLM 描述 + 异常检测

定时拍照并使用 AI 描述图像内容，支持 Telegram 推送和黑屏检测。

## 功能

- ⏱️ 定时拍照（支持树莓派摄像头 / 网络流 / fswebcam）
- 🤖 LLM 图像描述（OpenAI / Anthropic / 硅基流动 Qwen-VL）
- 📱 Telegram 推送通知
- 📝 日志记录
- 🖥️ **黑屏检测** - 自动判定 Link Test PASS/FAIL

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置

编辑 `config/config.yaml`:

```yaml
# 定时拍照
interval: 60  # 拍照间隔（秒）
output_dir: "photos"

# 输入源: local (树莓派摄像头) / stream (网络流)
source:
  type: "stream"
  stream_url: "http://192.168.1.15:8554/stream"

# LLM 配置 (推荐硅基流动 Qwen-VL)
llm:
  provider: "siliconflow"
  api_key: "${SILICONFLOW_API_KEY}"
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  base_url: "https://api.siliconflow.cn/v1"

# 黑屏检测
detection:
  enabled: true
  black_screen:
    enabled: true
    threshold: 30  # 亮度阈值 (0-255)
```

### 2. 运行

```bash
# 设置环境变量
export SILICONFLOW_API_KEY="sk-..."

# 运行
python -m src.capture_timer
```

## 网络流模式 (Mac 摄像头 → 树莓派)

### Mac 端运行推流脚本

```bash
cd ~/download/video2log
pip install opencv-python
python mac_stream.py
```

### 树莓派端配置

```yaml
source:
  type: "stream"
  stream_url: "http://192.168.1.15:8554/stream"
```

## 黑屏检测说明

| 配置 | 说明 |
|------|------|
| `detection.black_screen.threshold` | 亮度阈值，低于此值认为暗 (默认30) |
| `detection.black_screen.dark_pixel_ratio` | 暗像素比例阈值 (默认0.9) |
| `detection.black_screen.auto_detect_screens` | 用 Qwen 自动检测屏幕边界 (默认true) |

检测结果:
- **PASS** - 所有屏幕亮度正常
- **FAIL** - 任意屏幕黑屏

### 多屏幕检测

自动模式 (默认):
```yaml
detection:
  black_screen:
    enabled: true
    auto_detect_screens: true  # 调用 Qwen 识别屏幕边界
```

手动指定区域:
```yaml
detection:
  black_screen:
    enabled: true
    auto_detect_screens: false
    manual_regions:
      - name: "TV1"
        x1: 100, y1: 50, x2: 600, y2: 400
      - name: "TV2"
        x1: 700, y1: 50, x2: 1200, y2: 400
```

## 项目结构

```
video2log/
├── src/
│   ├── __init__.py
│   ├── config.py        # 配置加载
│   ├── logger.py        # 日志
│   ├── llm_client.py    # LLM API
│   ├── screen_detector.py  # 屏幕边界检测 (Qwen)
│   ├── detector.py      # 黑屏检测
│   └── capture_timer.py # 定时拍照
├── config/
│   └── config.yaml
├── mac_stream.py        # Mac 摄像头推流脚本
├── requirements.txt
└── README.md
```

## 环境变量

| 变量 | 说明 |
|------|------|
| SILICONFLOW_API_KEY | 硅基流动 API Key |
| OPENAI_API_KEY | OpenAI API Key |
| ANTHROPIC_API_KEY | Anthropic API Key |
| TELEGRAM_BOT_TOKEN | Telegram Bot Token |
| TELEGRAM_CHAT_ID | Telegram Chat ID |

## License

MIT
