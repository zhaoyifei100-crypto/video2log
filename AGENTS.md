# AGENTS.md - video2log 开发指南

## 项目概述

video2log 是 nanobot 的视觉模块，让 AI Agent 具有视觉能力：
- **静态视觉**: 看图说话
- **动态视觉**: 看监控、检测异常

技术栈：Python + OpenCV + LLM (Qwen2.5-VL)

---

## 开发命令

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
# 静态模式（描述画面）
python main.py --once

# 动态模式（检测异常）
python main.py --mode dynamic --once

# 循环监控
python main.py --mode dynamic --interval 30
```

### 测试

项目使用 pytest，测试文件位于 `tests/` 目录。

```bash
# 运行所有测试
pytest

# 运行单个测试文件
pytest tests/test_file.py

# 运行单个测试函数
pytest tests/test_file.py::test_function_name

# 带详细输出
pytest -v

# 显示打印输出
pytest -s

# 显示覆盖率
pytest --cov=src --cov-report=html
```

---

## 代码规范

### 类型标注

- **必须**使用类型注解（type hints）
- 使用 `typing` 模块：`Optional`, `Dict`, `List`, `Any`, `Callable`

```python
from typing import Optional, Dict, Any, List

def process_frame(frame, mode: str = "static") -> Optional[Dict[str, Any]]:
    pass
```

### 导入规范

- 使用相对导入（针对项目内部模块）
- 标准库 → 第三方库 → 项目模块分组

```python
# 标准库
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

# 第三方库
import cv2
import numpy as np
import requests

# 项目模块（相对导入）
from .config import config
from .logger import logger
from .llm_client import get_llm_client
```

### 命名约定

- **类名**: PascalCase
  ```python
  class VisionProcessor:
      pass
  
  class VisionMode(Enum):
      DYNAMIC = "dynamic"
  ```

- **函数/变量**: snake_case
  ```python
  def process_frame(self, frame):
      last_frame_base64 = ""
  ```

- **常量**: 全大写 + 下划线
  ```python
  MAX_RETRIES = 3
  DEFAULT_INTERVAL = 60
  ```

- **私有方法/变量**: 单下划线前缀
  ```python
  def _encode_image(self):
      self._config = {}
  ```

### 数据结构

- 使用 `@dataclass` 定义数据结构
- 使用 `Enum` 定义枚举类型

```python
from dataclasses import dataclass, field
from enum import Enum

class VisionMode(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"

@dataclass
class VisionResult:
    mode: VisionMode
    state: VisionState = VisionState.INIT
    frame: Optional[cv2.Mat] = None
    cv_result: Optional[Dict[str, Any]] = None
```

### 错误处理

- 使用 `try/except` 捕获具体异常
- 始终记录日志（使用 `logger` 模块）
- 返回 `None` 或空值表示失败，而非抛出异常

```python
try:
    response = self.llm_client.chat(prompt)
except requests.exceptions.RequestException as e:
    logger.error(f"LLM API 调用失败: {e}")
    return None
except json.JSONDecodeError as e:
    logger.error(f"JSON 解析失败: {e}")
    return None
```

### 日志规范

- 使用 `src/logger.py` 中的 `logger` 实例
- 日志级别：
  - `logger.debug()`: 调试信息
  - `logger.info()`: 正常流程
  - `logger.warning()`: 可恢复的异常
  - `logger.error()`: 错误

```python
logger.info(f"VisionProcessor: mode={self.mode.value}")
logger.warning(f"检测到异常: {reason}")
logger.error(f"LLM 调用失败: {e}")
```

### Docstring

- 使用中文 docstring（项目惯例）
- 模块级 docstring 说明文件用途
- 类和方法 docstring 说明功能

```python
"""
视觉处理核心模块 - 动态/静态模式
支持 LLM 自己生成 CV 代码进行监控
"""

class VisionProcessor:
    """视觉处理器"""
    
    def capture_frame(self) -> Optional[cv2.Mat]:
        """从视频流捕获一帧"""
```

### 代码结构

- 每个模块一个主要类（如 `VisionProcessor`, `LLMClient`）
- 使用配置类加载 YAML 配置
- 公开 API 放在模块顶层（如 `get_llm_client()`）

```python
# 全局单例
llm_client = None

def get_llm_client() -> LLMClient:
    """获取 LLM 客户端实例"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client
```

### 配置管理

- 配置文件：`config/config.yaml`
- 使用 `src/config.py` 中的 `Config` 类加载
- 支持环境变量 `${VAR}` 替换

```python
from .config import config

stream_url = config.get('stream_url', 'default_url')
interval = config.get('interval', 60)
```

### 路径处理

- 使用 `pathlib.Path` 处理路径
- 相对路径基于项目根目录

```python
from pathlib import Path

output_dir = Path("photos")
output_dir.mkdir(parents=True, exist_ok=True)
```

---

## 目录结构

```
video2log/
├── config/
│   └── config.yaml       # 配置文件
├── src/
│   ├── __init__.py
│   ├── vision.py         # 核心处理器
│   ├── opencv_helper.py  # OpenCV 辅助函数
│   ├── llm_client.py    # LLM 调用
│   ├── config.py        # 配置加载
│   ├── logger.py        # 日志
│   ├── detector.py      # 检测器
│   └── capture_timer.py # 定时捕获
├── tests/                 # 测试目录
├── main.py               # 主入口
└── requirements.txt     # 依赖
```

---

## 常用开发模式

### 动态模式状态机

```python
class VisionState(Enum):
    INIT = "init"       # 初始：截帧 + LLM 静态分析
    MONITOR = "monitor" # 监控：执行 LLM 生成的 CV 代码
    ALERT = "alert"     # 异常：LLM 决策下一步
    DONE = "done"       # 完成
```

### CV 函数命名

```python
def cv_detect_brightness(self, frame, region=None) -> float:
    """检测亮度"""

def cv_detect_dark_ratio(self, frame, threshold=30) -> float:
    """检测暗像素比例"""
```

---

## 注意事项

1. **敏感信息**: 不要提交 API Key，使用环境变量
2. **图像处理**: OpenCV 使用 BGR 格式
3. **性能**: 大图像先 resize 再处理，减少 token 消耗
4. **测试**: 新功能应添加测试，覆盖核心逻辑
