# AGENTS.md - video2log MCP Vision Server

AI Agent 开发指南 - 本项目是 OpenClaw/nanobot 的 MCP 视觉服务器

## 项目定位

**video2log** 现在是一个 **MCP Server**，为本地 AI Agent 提供视觉能力：
- 多路摄像头支持（CSI/USB/HTTP）
- 运动检测自动触发 + VLM 描述
- 通过 MCP notification 推送告警
- 树莓派本地部署，stdio 传输

## 技术栈

- **协议**: MCP (Model Context Protocol)
- **传输**: stdio（本地进程通信）
- **视觉**: OpenCV + 640x480
- **VLM**: OpenAI 兼容接口（SiliconFlow/Qwen2.5-VL）
- **配置**: YAML + Pydantic

---

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 MCP Server（stdio 模式）
python -m video2log.server

# 测试单个 tool（使用 mcp CLI）
mcp dev src/server.py
```

---

## 架构概览

```
┌─────────────────────────────────────┐
│         OpenClaw / nanobot          │
│           (MCP Client)              │
└───────────┬─────────────────────────┘
            │ MCP stdio
            ▼
┌─────────────────────────────────────┐
│        video2log.server             │
│  ┌─────────┐  ┌─────────────────┐   │
│  │ Cameras │  │ Monitor Sessions│   │
│  │ Manager │  │    Manager      │   │
│  └────┬────┘  └────────┬────────┘   │
│       │                │            │
│       └────────────────┘            │
│              │                      │
│       ┌──────▼──────┐              │
│       │   Motion    │              │
│       │  Detector   │              │
│       └──────┬──────┘              │
│              │                      │
│       ┌──────▼──────┐              │
│       │  VLM Call   │              │
│       └─────────────┘              │
└─────────────────────────────────────┘
```

---

## MCP Tools

### capture
拍摄单张照片
```python
async def capture(camera_id: str = "default") -> ImageContent
```

### describe
拍摄并描述画面
```python
async def describe(
    camera_id: str = "default",
    question: str = "描述当前画面"
) -> str
```

### start_monitoring
开始监控（运动检测）
```python
async def start_monitoring(
    camera_id: str = "default",
    detector: Literal["motion"] = "motion",
    sensitivity: float = 0.05,
    auto_describe: bool = True,
    describe_prompt: str = "描述画面中发生了什么"
) -> str  # session_id
```

### stop_monitoring
停止监控
```python
async def stop_monitoring(session_id: str) -> bool
```

### list_cameras
列出可用摄像头
```python
async def list_cameras() -> List[CameraInfo]
```

### get_monitoring_status
获取监控状态
```python
async def get_monitoring_status(
    session_id: Optional[str] = None
) -> Union[MonitoringStatus, List[MonitoringStatus]]
```

---

## MCP Notification

### vision/alert
运动检测触发时推送
```python
{
    "session_id": str,
    "camera_id": str,
    "timestamp": str,
    "trigger_type": "motion",
    "description": Optional[str],  # VLM 描述
    "image_base64": str
}
```

---

## 代码规范

### 类型标注
必须使用类型注解
```python
from typing import Optional, Dict, Any, List
from numpy.typing import NDArray

def process_frame(frame: NDArray) -> Optional[Dict[str, Any]]:
    pass
```

### 导入顺序
```python
# 标准库
import time
from typing import Optional
from dataclasses import dataclass

# 第三方
import cv2
import numpy as np
from mcp.server import Server
from openai import AsyncOpenAI

# 项目模块
from .camera import CameraManager
from .config import Config
```

### 命名约定
- **类**: PascalCase (`CameraManager`, `MotionDetector`)
- **函数/变量**: snake_case (`capture_frame()`, `motion_threshold`)
- **常量**: UPPER_SNAKE_CASE (`DEFAULT_RESOLUTION = (640, 480)`)
- **私有**: 单下划线前缀 (`_encode_image()`)

### 错误处理
捕获具体异常，返回 None 或空值，不抛出
```python
try:
    frame = camera.capture()
except cv2.error as e:
    logger.error(f"Capture failed: {e}")
    return None
```

### 日志
使用 Python 标准 logging
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Monitoring started: {session_id}")
logger.warning(f"Camera disconnected: {camera_id}")
logger.error(f"LLM call failed: {e}")
```

---

## 目录结构

```
video2log/
├── pyproject.toml
├── requirements.txt
├── README.md
├── AGENTS.md          # 本文件
├── REWRITE_TODO.md    # 重构任务清单
├── config/
│   └── config.yaml    # 配置模板
└── src/
    ├── __init__.py
    ├── server.py      # MCP Server 主入口
    ├── config.py      # 配置管理（Pydantic）
    ├── camera.py      # 摄像头管理
    ├── llm.py         # VLM 客户端
    ├── monitor.py     # 监控会话管理
    └── detectors/
        ├── __init__.py
        ├── base.py    # 检测器基类
        └── motion.py  # 运动检测
```

---

## 关键设计

### 图像尺寸
固定 **640x480**，运动检测和 VLM 都使用这个尺寸

### 监控循环
- 每 0.5 秒检查一次
- 运动检测阈值默认 0.05
- 告警冷却 10 秒

### VLM 调用
- MCP Server 直接调用（使用配置的 API Key）
- OpenAI 兼容接口
- 图像 base64 编码

### 多摄像头
每个摄像头独立，可以同时监控多个

---

## 配置文件示例

```yaml
# config/config.yaml
llm:
  api_key: "${SILICONFLOW_API_KEY}"
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  base_url: "https://api.siliconflow.cn/v1"

cameras:
  default:
    type: "csi"
    source: 0
    resolution: [640, 480]
  
  usb_cam:
    type: "usb"
    source: "/dev/video2"
    resolution: [640, 480]

monitoring:
  motion_threshold: 0.05
  check_interval: 0.5
  alert_cooldown: 10
```

---

## OpenClaw 集成

用户在 OpenClaw config 中添加：

```json
{
  "mcpServers": {
    "vision": {
      "command": "python",
      "args": ["-m", "video2log.server"],
      "env": {
        "VIDEO2LOG_CONFIG": "/home/pi/.config/video2log/config.yaml"
      }
    }
  }
}
```

---

## 注意事项

1. **不要在代码中提交 API Key** - 使用配置文件或环境变量
2. **树莓派 CSI 摄像头** - 需要 picamera2 库（Raspberry Pi OS）
3. **性能考虑** - 640x480 在树莓派 4 上运行流畅
4. **错误恢复** - 摄像头断开后应自动重连
