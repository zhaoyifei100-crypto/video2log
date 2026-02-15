# Dynamic Vision 监控系统 - 纯Skill实现方案

## 概述

基于 nanobot 现有架构的视频监控系统，**无需修改任何 nanobot 代码**，通过 Skill 指导 Agent 组合使用现有工具实现。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        主 Agent (AgentLoop)                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   spawn      │───▶│ 独立监控脚本  │───▶│ VISION_ALERT │       │
│  │    tool      │    │(后台进程)     │    │    .md       │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│       ▲                                         │                │
│       │                                         │                │
│       └──────────────  read_file ──────────────┘                │
│                                                                 │
│  ┌──────────────┐                                              │
│  │     cron     │───▶ 定期检查 VISION_ALERT.md                 │
│  │    tool      │     异常时通知用户                            │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 为什么不用修改 nanobot?

| 需求 | 原方案(需改代码) | 新方案(纯Skill) |
|-----|-----------------|----------------|
| 后台监控 | DynamicVisionTool + Subagent | `spawn` 工具启动独立Python脚本 |
| 异常通知 | MessageBus系统消息 | 写入文件 + `cron`定期检查 |
| 执行检测代码 | 沙箱exec() | 独立脚本自主执行 |
| LLM分析 | 在Tool中调用 | Agent通过`read_file`获取结果后分析 |

## 实现步骤

### Step 1: 创建独立监控脚本

**文件**: `workspace/video_monitor.py`

```python
#!/usr/bin/env python3
"""Dynamic Vision Monitor - 独立后台监控脚本"""

import sys
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

# 配置
ALERT_FILE = Path(__file__).parent / "VISION_ALERT.md"
LOG_FILE = Path(__file__).parent / ".video_monitor_logs.jsonl"
CHECK_INTERVAL = 2  # 秒

def capture_frame(source: str) -> Optional[bytes]:
    """截图 - 支持桌面、摄像头、RTSP等"""
    try:
        import pyautogui
        screenshot = pyautogui.screenshot()
        from io import BytesIO
        buffer = BytesIO()
        screenshot.save(buffer, format='PNG')
        return buffer.getvalue()
    except Exception as e:
        print(f"截图失败: {e}", file=sys.stderr)
        return None

def analyze_frame(frame: bytes, monitor_type: str) -> dict:
    """分析画面 - 简单示例，实际可用OpenCV/ML模型"""
    # TODO: 接入实际检测逻辑
    return {
        "status": "normal",  # or "anomaly"
        "confidence": 0.0,
        "description": "检测正常",
        "timestamp": datetime.now().isoformat()
    }

def write_alert(result: dict, frame: Optional[bytes] = None):
    """写入异常警报到文件"""
    alert_content = f"""# 🚨 视频监控异常报告

**检测时间**: {result['timestamp']}
**异常类型**: {result.get('description', '未知')}
**置信度**: {result.get('confidence', 'N/A')}

## 检测详情

```json
{json.dumps(result, indent=2, ensure_ascii=False)}
```

---
**监控任务**: {sys.argv[1] if len(sys.argv) > 1 else '未指定'}

请决定下一步操作:
- **continue**: 继续监控（忽略本次异常）
- **adjust**: 调整监控参数
- **stop**: 停止监控任务
"""
    ALERT_FILE.write_text(alert_content, encoding='utf-8')
    
    # 同时保存截图
    if frame:
        img_path = ALERT_FILE.parent / f"anomaly_{result['timestamp'].replace(':', '-')}.png"
        img_path.write_bytes(frame)

def log_result(result: dict):
    """记录检测日志"""
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_monitor.py <monitor_type> [options]")
        print("  monitor_type: desktop, camera, rtsp://...")
        sys.exit(1)
    
    image_source = sys.argv[1]
    print(f"🎥 启动视频监控: {image_source}")
    print(f"   检查间隔: {CHECK_INTERVAL}秒")
    print(f"   警报文件: {ALERT_FILE}")
    
    check_count = 0
    try:
        while True:
            check_count += 1
            
            # 1. 截图
            frame = capture_frame(image_source)
            if frame is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # 2. 分析（这里用简单示例）
            result = analyze_frame(frame, image_source)
            result['check_count'] = check_count
            
            # 3. 记录日志
            log_result(result)
            
            # 4. 检测异常
            if result['status'] == 'anomaly':
                print(f"⚠️ 检测到异常！check_count={check_count}")
                write_alert(result, frame)
                # 异常后退出，等待Agent处理
                print("   已写入警报文件，等待Agent处理...")
                sys.exit(0)
            
            # 5. 正常等待
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ 监控已停止，共检测 {check_count} 次")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

**安装依赖**:
```bash
pip install pyautogui Pillow
```

### Step 2: 创建 Skill

**文件**: `workspace/skills/video-monitor/SKILL.md`

```markdown
---
name: video-monitor
description: 动态视觉监控系统 - 后台监控 + 定时检查
metadata:
  nanobot:
    requires:
      models: ["vision"]
      bins: ["python"]
---

# 动态视觉监控系统

当用户需要进行视频监控时，按以下步骤操作：

## 1. 初始化监控

**1.1** 捕获初始帧进行分析：
```python
exec("python3 -c 'import pyautogui; pyautogui.screenshot().save(\"init_frame.png\")'")
```

**1.2** 使用 vision 工具分析初始帧，理解监控目标：
- 识别监控区域
- 理解用户意图（如"检测人"、"检测变化"等）

**1.3** 启动后台监控脚本：
```python
spawn(task="启动视频监控: python3 workspace/video_monitor.py desktop")
```

**1.4** 创建定期检查任务：
```python
cron(
    action="add",
    message="检查 VISION_ALERT.md 文件，如果有内容则读取并报告异常给用户",
    every_seconds=30
)
```

## 2. 处理异常警报

当 cron 触发检查时：

**2.1** 使用 read_file 工具读取 `VISION_ALERT.md`

**2.2** 如果有内容，分析异常报告：
- 异常类型
- 置信度
- 时间戳
- 截图位置

**2.3** 向用户报告并询问决策：
- **continue**: 清空 VISION_ALERT.md，重新启动监控脚本
- **adjust**: 重新初始化，调整监控参数
- **stop**: 停止监控（移除cron任务）

## 3. 停止监控

**3.1** 列出并移除相关cron任务：
```python
cron(action="list")
cron(action="remove", job_id="<检查任务的job_id>")
```

**3.2** 清理文件：
- VISION_ALERT.md
- anomaly_*.png（异常截图）

## 使用示例

**用户**: 监控我的桌面，如果有人出现告诉我

**Agent执行流程**:
1. 截图 → vision分析 → 确定监控目标
2. spawn启动 video_monitor.py desktop
3. cron创建每30秒检查VISION_ALERT.md的任务
4. （等待）
5. 脚本检测到异常 → 写入VISION_ALERT.md → 退出
6. Cron触发 → Agent读取文件 → 报告用户
7. 询问用户决策 (continue/adjust/stop)

## 注意事项

- video_monitor.py 是独立进程，会在异常时自动退出
- 每次异常后需要重新spawn启动监控
- 监控脚本依赖 pyautogui 和 Pillow
- 日志保存在 .video_monitor_logs.jsonl
```

### Step 3: 可选 - 增强版检测脚本

如果需要更复杂的检测（如YOLO人体检测），可以扩展 `video_monitor.py`：

```python
# 在 video_monitor.py 中添加

def detect_person_opencv(frame: bytes) -> dict:
    """使用OpenCV HOG检测人体"""
    import cv2
    import numpy as np
    from io import BytesIO
    
    # 将bytes转为OpenCV格式
    nparr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # HOG人体检测
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    boxes, weights = hog.detectMultiScale(img, winStride=(8,8))
    
    if len(boxes) > 0:
        return {
            "status": "anomaly",
            "confidence": float(max(weights)) if len(weights) > 0 else 0.5,
            "description": f"检测到 {len(boxes)} 个人",
            "details": {"people_count": len(boxes)}
        }
    
    return {
        "status": "normal",
        "confidence": 1.0,
        "description": "未检测到人体"
    }
```

## 工作流程

```
用户: 监控桌面，有人出现告诉我

├─► Agent读取skill指导
│
├─► Agent执行:
│   ├─ 截图分析 (exec + vision)
│   ├─ spawn启动监控脚本
│   └─ cron创建检查任务 (每30秒)
│
├─► 监控脚本后台运行:
│   ├─ 截图 → 检测 → 正常 → 循环
│   └─ 检测到人 → 写VISION_ALERT.md → 退出
│
├─► Cron触发 (30秒后):
│   └─ Agent发现文件有内容 → 读取并报告用户
│
└─► 用户决策:
    ├─ continue → 清空文件 → 重新spawn
    ├─ adjust → 重新初始化
    └─ stop → 移除cron → 清理文件
```

## 与传统方案的对比

| 特性 | 原方案(改代码) | 新方案(纯Skill) |
|-----|--------------|----------------|
| 侵入性 | 需改nanobot核心代码 | 零侵入 |
| 维护成本 | 高 | 低 |
| 灵活性 | 受限于Tool实现 | 可动态调整 |
| 复杂度 | 高（Subagent+MessageBus） | 低（文件+定时检查） |
| 适用场景 | 复杂实时监控 | 常规监控需求 |

## 未来扩展

1. **多路监控**: 同时监控多个图像源（多个spawn任务）
2. **流式支持**: 扩展脚本支持RTSP/HTTP视频流
3. **GPU加速**: 在脚本中加入CUDA支持
4. **云端分析**: 将截图发送到云端API进行检测

---

**设计原则**: 利用现有工具组合实现功能，避免修改核心代码。
