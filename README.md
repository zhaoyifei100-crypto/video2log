# video2log
# 定时拍照 + LLM 描述

定时拍照并使用 AI 描述图像内容，支持 Telegram 推送。

## 功能

- ⏱️ 定时拍照（支持树莓派摄像头 / fswebcam）
- 🤖 LLM 图像描述（OpenAI GPT-4o / Anthropic Claude）
- 📱 Telegram 推送通知
- 📝 日志记录

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config/config.yaml`:

```yaml
interval: 60  # 拍照间隔（秒）
output_dir: "photos"
log_dir: "logs"

# LLM 配置
llm:
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"  # 环境变量
  model: "gpt-4o-mini"

# 图像设置
image:
  format: "jpg"
  quality: 85
  width: 1280
  height: 720

# Telegram (可选)
telegram:
  enabled: false
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
```

## 使用

```bash
# 设置环境变量
export OPENAI_API_KEY="sk-..."
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."

# 运行
python -m src.capture_timer
```

## 项目结构

```
video2log/
├── src/
│   ├── __init__.py
│   ├── config.py      # 配置加载
│   ├── logger.py      # 日志
│   ├── llm_client.py  # LLM API
│   └── capture_timer.py  # 定时拍照
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## 环境变量

| 变量 | 说明 |
|------|------|
| OPENAI_API_KEY | OpenAI API Key |
| ANTHROPIC_API_KEY | Anthropic API Key |
| TELEGRAM_BOT_TOKEN | Telegram Bot Token |
| TELEGRAM_CHAT_ID | Telegram Chat ID |

## License

MIT
