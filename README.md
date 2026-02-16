# video2log 🐱

> Nanobot 视觉技能 - 让 AI Bot 能看懂世界

---

## 这是什么

Nanobot 自己的"眼睛"。

video2log 是 nanobot 的视觉模块，让 AI Agent 具有视觉能力：
- **静态视觉**: 看图说话, done
- **动态视觉**: 看监控、检测异常, TODO, CV真TM离谱

## Nanobot 怎么工作的

```
你: "帮我看看这个画面正常吗"

Nanobot:
  1. 先自己看一遍 (静态分析)
  2. 记住正常的样子 (建立基准)
  3. 持续盯着 (动态监控)
  4. 发现异常 → 通知你
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 (默认静态模式 - 描述画面)
python main.py --once

# 动态模式 (检测异常)
python main.py --mode dynamic --once

# 循环监控
python main.py --mode dynamic --interval 30
```

## 两种模式

| 模式 | Nanobot 怎么做 | 适用场景 |
|------|---------------|----------|
| `static` | 看一眼，直接告诉你 | 描述画面、识别物体 |
| `dynamic` | 看一眼 → 记住 → 一直盯着 → 异常喊你 | 黑屏检测、监控告警 |

## 配置

编辑 `config/config.yaml`:

```yaml
mode: "dynamic"           # 模式: dynamic / static
interval: 60              # 抓取间隔(秒)
stream_url: "http://192.168.1.15:8554/stream"

opencv:
  brightness_threshold: 30   # 亮度阈值
  dark_ratio_threshold: 0.9   # 暗像素比例

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
│   ├── llm_client.py    # LLM 调用
│   ├── config.py        # 配置加载
│   └── logger.py        # 日志
├── main.py               # 入口
└── requirements.txt
```

## 技术栈

- **图像处理**: OpenCV
- **视觉模型**: Qwen2.5-VL (SiliconFlow)
- **部署**: Raspberry Pi / Mac / Linux
