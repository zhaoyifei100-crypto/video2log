# MCP Vision Server 重构 TODO

## 项目目标
将 video2log 改造为 MCP Server，为 OpenClaw/nanobot 提供视觉能力：
- 支持多路摄像头（CSI/USB）
- 运动检测自动触发 + VLM 描述
- 通过 MCP notification 推送告警
- 树莓派本地部署（stdio 模式）

---

## Phase 1: 架构设计与依赖更新

### 1.1 更新 requirements.txt
```
# MCP 依赖
mcp>=1.0.0

# 视觉
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0

# LLM
openai>=1.0.0

# 配置
PyYAML>=6.0
pydantic>=2.0.0

# 可选：树莓派 CSI 摄像头
# picamera2>=0.3.0  # Raspberry Pi OS only
```

### 1.2 创建新目录结构
```
video2log/
├── pyproject.toml          # 项目配置
├── README.md               # 更新为新项目描述
├── src/
│   ├── __init__.py         # 包版本信息
│   ├── config.py           # MCP Server 配置（简化版）
│   ├── camera.py           # 摄像头管理器
│   ├── detectors/          # 检测器（保留现有）
│   │   ├── __init__.py
│   │   ├── base.py         # DetectionResult 等
│   │   ├── motion.py       # 运动检测
│   │   └── black_screen.py # 黑屏检测（可选）
│   ├── llm.py              # VLM 客户端（简化版）
│   ├── monitor.py          # 监控会话管理
│   └── server.py           # MCP Server 主入口
└── config/
    └── config.yaml         # MCP Server 配置模板
```

---

## Phase 2: 核心模块重构

### 2.1 简化 Config（src/config.py）
- 移除复杂的 env var 替换逻辑
- 使用 Pydantic 模型验证配置
- 只保留必要字段：
  - LLM API Key/Model/BaseURL
  - 摄像头配置列表
  - 默认监控参数

### 2.2 重写 Camera 模块（src/camera.py）
```python
class Camera:
    """单个摄像头封装"""
    - camera_id: str
    - source: Union[int, str]  # 0, "/dev/video0", "http://..."
    - resolution: Tuple[int, int]  # 640x480
    - capture(): -> np.ndarray
    - release()

class CameraManager:
    """多摄像头管理"""
    - cameras: Dict[str, Camera]
    - get_camera(id) -> Camera
    - list_cameras() -> List[CameraInfo]
```

### 2.3 重写 LLM 模块（src/llm.py）
- 只保留同步调用（OpenAI 兼容）
- 单函数：`describe_image(image: np.ndarray, prompt: str) -> str`
- 图像处理：resize 到 640x480，JPEG encode，base64

### 2.4 重写 Monitor（src/monitor.py）
```python
class MonitoringSession:
    """单个监控会话"""
    - session_id: str
    - camera_id: str
    - detector: BaseDetector
    - callback: Callable[[Alert], None]  # 触发时调用
    - is_running: bool
    - start()/stop()

class MonitorManager:
    """管理所有监控会话"""
    - sessions: Dict[str, MonitoringSession]
    - start_monitoring(config) -> session_id
    - stop_monitoring(session_id)
    - get_status(session_id?) -> Status
```

### 2.5 检测器保留但简化（src/detectors/）
- **保留**：base.py（DetectionResult dataclass）
- **保留**：motion.py（运动检测）
- **可选**：black_screen.py（黑屏检测）
- **移除**：复杂的 LLM 菜单系统
- **修改**：remove 文件系统的 detector 发现机制

---

## Phase 3: MCP Server 实现

### 3.1 实现 Tools（src/server.py）

```python
@mcp.tool()
async def capture(camera_id: str = "default") -> ImageContent:
    """拍摄单张照片"""
    pass

@mcp.tool()
async def describe(
    camera_id: str = "default",
    question: str = "描述当前画面"
) -> str:
    """拍摄并描述"""
    pass

@mcp.tool()
async def start_monitoring(
    camera_id: str = "default",
    detector: Literal["motion"] = "motion",
    sensitivity: float = 0.05,  # 运动检测阈值
    auto_describe: bool = True,
    describe_prompt: str = "描述画面中发生了什么"
) -> str:  # 返回 session_id
    """开始监控"""
    pass

@mcp.tool()
async def stop_monitoring(session_id: str) -> bool:
    """停止监控"""
    pass

@mcp.tool()
async def list_cameras() -> List[CameraInfo]:
    """列出可用摄像头"""
    pass

@mcp.tool()
async def get_monitoring_status(
    session_id: Optional[str] = None
) -> Union[MonitoringStatus, List[MonitoringStatus]]:
    """获取监控状态"""
    pass
```

### 3.2 实现 Notification

```python
@mcp.notification("vision/alert")
class VisionAlert(Notification):
    session_id: str
    camera_id: str
    timestamp: str
    trigger_type: str  # "motion"
    description: Optional[str]  # VLM 描述（如果 auto_describe=True）
    image_base64: str  # 触发时的画面
```

### 3.3 主入口（src/server.py 底部）

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## Phase 4: 配置与部署

### 4.1 配置文件（config/config.yaml）

```yaml
# LLM 配置（VLM）
llm:
  api_key: "${SILICONFLOW_API_KEY}"  # 从环境变量读取
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  base_url: "https://api.siliconflow.cn/v1"

# 摄像头配置
cameras:
  default:
    type: "csi"  # csi, usb, http
    source: 0    # CSI 索引
    resolution: [640, 480]
  
  usb_cam:
    type: "usb"
    source: "/dev/video2"
    resolution: [640, 480]

# 默认监控参数
monitoring:
  motion_threshold: 0.05
  check_interval: 0.5  # 秒
  alert_cooldown: 10   # 秒（同类型告警冷却）
```

### 4.2 OpenClaw 配置示例

用户需要在 OpenClaw/nanobot 的 config 中添加：

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

## Phase 5: 清理旧代码

### 5.1 删除文件
- [x] tests/ - 整个测试目录
- [x] monitor.py - 旧监控脚本
- [x] mac_stream.py - 流媒体服务器
- [x] monitor_config.json - JSON 配置
- [x] monitor_output/ - 输出目录
- [x] logs/ - 日志目录  
- [x] photos/ - 照片目录
- [x] VISION_ALERT.md
- [x] SKILL.md
- [x] vllm_api.log
- [x] .pytest_cache/
- [x] .ruff_cache/

### 5.2 重写文件
- [ ] main.py - 改为 MCP Server 入口或删除
- [ ] src/vision.py - 完全重写为 MCP 服务
- [ ] src/llm_client.py - 简化为 src/llm.py
- [ ] src/config.py - 使用 Pydantic
- [ ] src/logger.py - 简化为 print 或移除
- [ ] src/opencv_helper.py - 合并到 camera.py

### 5.3 保留但修改
- [ ] src/detectors/base.py - 简化
- [ ] src/detectors/motion.py - 保留核心逻辑
- [ ] src/detectors/black_screen.py - 可选

---

## Phase 6: 测试与验证

### 6.1 本地测试
```bash
# 安装依赖
pip install -r requirements.txt

# 运行 MCP Server（stdio 模式）
python -m video2log.server

# 手动测试 tools
capture -> 应该返回 base64 图片
describe -> 应该返回 VLM 描述
start_monitoring -> 返回 session_id
# 然后用手在摄像头前晃动，应该触发 notification
```

### 6.2 OpenClaw 集成测试
```bash
# 启动 OpenClaw
openclaw

# 在 OpenClaw 中测试
> 看看周围有什么？
# Agent 应该自动调用 vision/capture 和 vision/describe

> 帮我监控门口
# Agent 应该调用 vision/start_monitoring
# 当有人经过时，Agent 收到 notification
```

---

## 优先级排序

**P0（核心功能）**
1. MCP Server 骨架 + stdio 传输
2. capture tool
3. motion detector + start_monitoring
4. vision/alert notification

**P1（增强功能）**
5. describe tool（集成 VLM）
6. 多摄像头支持
7. stop_monitoring / get_status

**P2（优化）**
8. 黑屏检测
9. 配置热加载
10. 错误重试机制

---

## 关键设计决策

### 图像尺寸
- **640x480**（用户指定）
- 运动检测在 640x480 上进行
- VLM 描述时保持 640x480（减小 token 消耗）

### 监控循环
- 每个 session 一个独立线程
- 检查间隔 0.5 秒（平衡性能和响应）
- 告警冷却 10 秒（避免重复触发）

### VLM 调用
- MCP Server 端直接调用（使用配置的 API Key）
- 不需要 Agent 提供 LLM 配置
- 只使用 OpenAI 兼容接口（简化）

### 错误处理
- Camera 初始化失败：tool 返回错误信息
- LLM 调用失败：返回空描述，不影响监控
- 监控线程异常：记录日志，停止 session

---

## 完成标志

- [ ] `python -m video2log.server` 能启动且不报错
- [ ] `capture` tool 返回有效图片
- [ ] `describe` tool 返回有效描述
- [ ] `start_monitoring` 后，运动检测能触发 notification
- [ ] OpenClaw 能通过 MCP 调用所有 tools
- [ ] 树莓派上能稳定运行（测试 24 小时）
