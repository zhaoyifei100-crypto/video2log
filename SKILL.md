---
name: vision
description: |
  通用视觉能力，让 AI 能看懂图片和视频。
  
  触发场景：
  (1) 用户要求描述、分析、识别图片或视频内容
  (2) 用户要求检测、监控画面异常（黑屏、闪断、物体检测）
  (3) 用户要求找出图中特定物体的位置
  (4) 涉及屏幕监控、电视测试、示波器读数等视觉任务
metadata:
  nanobot:
    requires:
      models: ["vision"]
      bins: ["python3"]
---

# Vision Skill - 非阻塞动态视觉监控

> 基于 nanobot 现有架构的纯 Skill 实现，**无需修改任何 nanobot 代码**

---

## 核心亮点

**动态视觉中，CV 算法检测 + VLLM 二次确认，确保不误报！**

```
用户: "帮我盯着电视有没有黑屏"

Agent:
  1. 看一眼 → "电视在画面中央，显示正常"
  2. spawn 启动独立监控脚本（后台运行，不阻塞）
  3. cron 创建定期检查任务（每30秒检查 VISION_ALERT.md）
  4. 监控脚本后台运行:
     - 定期截图
     - CV 算法检测可疑情况
     - VLLM 二次确认（过滤误报）
     - 确认异常 → 写入 VISION_ALERT.md → 退出
  5. cron 触发 → Agent 读取警报 → 通知用户
  6. 用户决策 (continue/adjust/stop)
```

---

## 架构设计（非阻塞）

```
┌─────────────────────────────────────────────────────────────────┐
│                      主 Agent (AgentLoop)                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   spawn      │───▶│  monitor.py  │───▶│ VISION_ALERT │       │
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

### 为什么不会卡死？

| 问题 | 原方案 | 新方案 |
|-----|--------|--------|
| 监控执行 | 主进程阻塞循环 | `spawn` 启动独立 Python 脚本 |
| 异常检测 | CV 直接判定 | CV 可疑 + VLLM 确认 |
| 异常通知 | 内部状态机 | 写入文件 + `cron` 定时检查 |
| LLM 分析 | 循环中同步调用 | Agent 通过 `read_file` 异步获取 |

---

## 核心流程

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────┐
│           1. 判定视觉模式                    │
├─────────────────────────────────────────────┤
│                                             │
│   静态关键词: 描述/识别/这是什么/有什么/位置   │
│   动态关键词: 检测/监控/盯着/黑屏/闪断/异常  │
│                                             │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   ┌─────────────┐          ┌─────────────┐
   │  静态视觉   │          │  动态视觉   │
   └──────┬──────┘          └──────┬──────┘
          │                        │
          ▼                        ▼
   ┌─────────────┐          ┌─────────────┐
   │ vision工具  │          │   INIT     │
   │ 直接分析    │          │  初始化    │
   │            │          └──────┬──────┘
   │            │                 │
   └────────────┘                 ▼
                           ┌─────────────┐
                           │  spawn启动  │
                           │  monitor.py │
                           └──────┬──────┘
                                  │
                                  ▼
                           ┌─────────────┐
                           │   cron创建  │
                           │ 定时检查任务 │
                           └──────┬──────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
        ┌──────────────┐                 ┌──────────────┐
        │  监控正常     │                 │  CV可疑+VLLM │
        │  持续运行     │                 │  确认异常     │
        └──────────────┘                 └──────┬───────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  写入警报    │
                                        │  脚本退出    │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  cron触发    │
                                        │ Agent读取文件 │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  报告用户    │
                                        │ 询问决策     │
                                        └──────┬───────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
            ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
            │  continue    │          │   adjust     │          │    stop      │
            │ 重新spawn启动 │          │ 调整参数重启  │          │ 移除cron清理  │
            └──────────────┘          └──────────────┘          └──────────────┘
```

---

## 详细步骤

### 静态视觉（单次分析）

当用户说"帮我看看这张图有什么"：

1. **截图/获取图片**
   ```python
   # 如果是屏幕截图
   exec("""
   python3 -c "
   from src.opencv_helper import capture_screen
   capture_screen('current_frame.jpg')
   print('截图已保存')
   "
   """)
   ```

2. **使用 vision 工具分析**
   ```python
   # vision 工具会自动分析图片内容
   # 返回描述给用户
   ```

### 动态视觉（持续监控）- 非阻塞实现

当用户说"帮我盯着电视有没有黑屏"：

#### Step 1: 初始化（INIT）

**1.1** 捕获初始帧分析：
```python
exec("""
python3 -c "
from src.opencv_helper import capture_screen
capture_screen('init_frame.jpg')
print('初始帧已保存')
"
""")
```

**1.2** 使用 vision 工具分析初始帧：
- 识别监控区域（电视/显示器位置）
- 理解用户意图（黑屏检测、运动检测等）

**1.3** 创建监控配置：
```python
exec("""
import json
import os

# 从环境变量获取 API key，或从现有配置读取
config = {
    'monitor_type': 'black_screen',  # black_screen / motion / custom
    'target': 'TV1',
    'interval': 5,  # 检查间隔秒
    'threshold': 30,  # 亮度阈值
    'output_dir': 'monitor_output',
    'alert_file': 'VISION_ALERT.md',
    'stream_url': 'desktop',  # desktop 或 RTSP 地址
    # VLLM 配置（用于异常确认）
    'vllm': {
        'provider': 'siliconflow',
        'api_key': os.getenv('SILICONFLOW_API_KEY', ''),
        'model': 'Qwen/Qwen2.5-VL-72B-Instruct',
        'base_url': 'https://api.siliconflow.cn/v1'
    }
}

with open('monitor_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('监控配置已保存')
""")
```

**1.4** 启动后台监控脚本（关键：使用 spawn 不阻塞）：
```python
spawn(
    command="python3 monitor.py --config monitor_config.json",
    description="启动视频监控进程（后台运行，异常时写入 VISION_ALERT.md）"
)
```

**1.5** 创建定期检查任务：
```python
cron(
    action="add",
    message="检查 VISION_ALERT.md 文件，如果有内容则读取并报告异常给用户",
    every_seconds=30
)
```

**1.6** 向用户报告监控已启动：
```
✅ 已开始监控电视画面
- 检查间隔: 5秒
- 检测类型: 黑屏检测
- 确认机制: CV算法 + VLLM二次确认
- 警报检查: 每30秒检查一次

监控在后台运行中，检测到异常时会立即通知你。
```

#### Step 2: 处理异常警报（ALERT）

当 cron 触发检查发现 VISION_ALERT.md 有内容时：

**2.1** 读取警报文件：
```python
read_file(path="VISION_ALERT.md")
```

**2.2** 解析异常报告并报告用户：
```
🚨 检测到异常！

**时间**: 2024-01-15 14:32:18
**监控类型**: 电视1黑屏

## CV 检测结果
- 平均亮度: 12.3 (阈值: 30)
- 暗像素比例: 94.2%

## VLLM 确认结果
- **确认异常**: ✅ 是
- **置信度**: 0.92
- **原因**: 画面显示为全黑，确认电视已黑屏

## 截图
**文件**: alert_20240115_143218.jpg

**下一步操作**：
1. **continue** - 继续监控（可能是临时故障）
2. **adjust** - 调整监控参数（如提高阈值避免误报）
3. **stop** - 停止监控
```

**2.3** 根据用户决策执行：

- **continue**: 
  ```python
  # 清空警报文件
  exec("open('VISION_ALERT.md', 'w').close()")
  # 重新启动监控
  spawn(command="python3 monitor.py --config monitor_config.json")
  ```

- **adjust**:
  ```python
  # 更新配置
  exec("""
  import json
  with open('monitor_config.json', 'r') as f:
      config = json.load(f)
  config['threshold'] = 50  # 调整阈值
  with open('monitor_config.json', 'w') as f:
      json.dump(config, f)
  """)
  # 重新启动
  spawn(command="python3 monitor.py --config monitor_config.json")
  ```

- **stop**:
  ```python
  # 移除cron任务
  cron(action="list")  # 获取任务ID
  cron(action="remove", job_id="<任务ID>")
  # 清理文件
  exec("""
  import os
  import glob
  for f in ['VISION_ALERT.md', 'monitor_config.json']:
      if os.path.exists(f):
          os.remove(f)
  for f in glob.glob('monitor_output/*'):
      if os.path.exists(f):
          os.remove(f)
  """)
  ```

---

## 监控脚本工作原理

### 文件: `monitor.py`

独立的后台监控脚本，由 `spawn` 工具启动：

```
monitor.py 工作流程:
├─ 1. 加载配置 (monitor_config.json)
├─ 2. 初始化
│   ├─ OpenCVHelper (CV 检测)
│   ├─ VLLMClient (独立 LLM 客户端)
│   └─ MonitorLogger (独立日志)
├─ 3. 监控循环
│   ├─ 截图
│   ├─ CV 检测 (detect_suspicious)
│   │   └─ 返回 is_suspicious + 可疑类型 + 置信度
│   ├─ 连续可疑? ──Yes──┐
│   │                    ▼
│   │            VLLM 确认 (confirm_anomaly)
│   │                    ├─ 确认异常 ──► 写入 VISION_ALERT.md
│   │                    │                保存异常截图
│   │                    │                退出脚本
│   │                    └─ 误报 ─────► 继续监控
│   └─ 正常 ────────────► 继续循环
└─ 4. 退出（异常或用户中断）
```

### CV 检测与 VLLM 确认分离

**CV 算法职责**:
- 快速检测可疑情况（低延迟）
- 只报告 `is_suspicious`，不最终判定
- 连续多次可疑才触发 VLLM（避免频繁调用）

**VLLM 确认职责**:
- 分析截图内容
- 基于 CV 结果做最终判断
- 过滤误报（如：画面变暗但不是黑屏）

### 日志分离

- **VISION_ALERT.md**: 只保留关键异常信息（简洁）
- **monitor_logs.jsonl**: 详细检测日志（包括误报记录）
- **alert_*.jpg**: 确认的异常截图
- **suspicious_*.jpg**: 可疑但未确认的截图（可选清理）

---

## CV 函数库（供参考）

监控脚本可用的检测函数：

| 函数 | 说明 | 返回 |
|------|------|------|
| `detect_brightness(frame, region=None)` | 亮度检测 | `float: 0-255` |
| `detect_dark_ratio(frame, threshold=30)` | 暗像素比例 | `float: 0.0-1.0` |
| `detect_motion(frame, prev_frame)` | 运动检测 | `float: 差异分数` |
| `detect_edges(frame, low=50, high=150)` | 边缘检测 | `dict: edge_count` |
| `compare_frames(frame1, frame2)` | 帧对比 | `dict: diff_score` |
| `crop_region(frame, x1, y1, x2, y2)` | 裁剪区域 | `np.ndarray` |

---

## 配置说明

### config.yaml 结构

```yaml
# LLM 配置 (用于静态分析)
llm:
  provider: "siliconflow"
  api_key: "${SILICONFLOW_API_KEY}"
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  base_url: "https://api.siliconflow.cn/v1"

# VLLM 配置 (用于监控异常确认，可独立配置)
vllm:
  provider: "siliconflow"
  api_key: "${SILICONFLOW_API_KEY}"
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  base_url: "https://api.siliconflow.cn/v1"
```

### monitor_config.json 结构

```json
{
  "monitor_type": "black_screen",
  "target": "TV1",
  "interval": 5,
  "threshold": 30,
  "output_dir": "monitor_output",
  "alert_file": "VISION_ALERT.md",
  "stream_url": "desktop",
  "vllm": {
    "provider": "siliconflow",
    "api_key": "sk-xxx",
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "base_url": "https://api.siliconflow.cn/v1"
  }
}
```

---

## 使用示例

### 示例 1: 监控桌面黑屏

**用户**: "帮我盯着电视，如果黑屏了告诉我"

**Agent 执行**:
1. `exec`: 截图保存到 `init_frame.jpg`
2. `vision`: 分析画面，识别电视位置
3. `exec`: 创建 `monitor_config.json`（黑屏检测配置 + VLLM 配置）
4. `spawn`: 启动 `python3 monitor.py --config monitor_config.json`
5. `cron`: 创建每30秒检查 VISION_ALERT.md 的任务
6. （监控在后台运行...）
7. CV 连续检测到可疑亮度 → 触发 VLLM 确认
8. VLLM 确认是黑屏 → 写入 VISION_ALERT.md → 退出
9. `cron` 触发 → Agent 读取文件 → 报告用户
10. 用户选择 "continue"
11. `exec`: 清空 VISION_ALERT.md
12. `spawn`: 重新启动监控脚本

### 示例 2: 检测运动

**用户**: "监控这个摄像头画面，有人出现时告诉我"

**Agent 执行**:
1. `exec`: 截图
2. `vision`: 分析画面区域
3. `exec`: 创建配置（`monitor_type: motion`）
4. `spawn`: 启动监控
5. `cron`: 创建检查任务
6. CV 检测到运动 → VLLM 确认是人物 → 写入警报

### 示例 3: 静态分析

**用户**: "这张图片里有什么？"

**Agent 执行**:
1. `vision`: 直接分析图片（单次，无需监控）
2. 返回描述结果

---

## 关键注意事项

1. **monitor.py 是独立进程**，会在异常时自动退出，不会卡死主进程
2. **VLLM 用于二次确认**，过滤 CV 算法的误报
3. **spawn 工具是关键**，用于启动后台进程
4. **cron 工具定期检查**，不占用主进程时间
5. **文件通信机制**：
   - `VISION_ALERT.md` - 异常警报（简洁）
   - `monitor_config.json` - 监控配置
   - `monitor_logs.jsonl` - 详细日志
6. **VLLM 可独立配置**，可与主 LLM 不同（如使用更轻量的模型）

---

## 与传统方案的对比

| 特性 | 原方案（阻塞） | 新方案（非阻塞） |
|-----|---------------|-----------------|
| 监控执行 | 主进程循环阻塞 | spawn 后台进程 |
| 异常判定 | CV 直接判定 | CV 可疑 + VLLM 确认 |
| 误报率 | 较高 | 较低（VLLM 过滤） |
| 异常通知 | 内部状态变量 | 文件 + cron |
| 主进程卡死 | 是 | 否 |
| 侵入性 | 需改核心代码 | 纯 Skill 实现 |

---

## 依赖

```
opencv-python
numpy
requests
pyautogui  # 桌面截图用（可选）
```

**安装**:
```bash
pip install opencv-python numpy requests pyautogui
```

---

**设计原则**: 
- CV 算法 + VLLM 确认 = 低误报率
- spawn + cron = 非阻塞
- 文件通信 = 零侵入核心代码
