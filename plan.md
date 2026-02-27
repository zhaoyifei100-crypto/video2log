# LokmEye iOS 技术路线图 v1.0

**基于 OpenClaw SKILL + iOS Node 架构**

---

## 文档信息

- **版本**: 1.0
- **创建日期**: 2026-02-27
- **状态**: Draft (待Review)
- **关联项目**: LokmEye (LookMyEye) - 智能监控解决方案

---

## 1. 项目概述

### 1.1 项目愿景
让旧iPhone变身AI智能监控眼，通过OpenClaw SKILL架构实现零门槛配置、智能化检测、自然语言交互。

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenClaw Gateway (Mac/Server)               │
│  ┌──────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │lokmeye-skill │    │ nodes-tool  │    │    camsnap skill    │ │
│  │  (业务逻辑)   │◄───│ (节点控制)  │────│   (相机管理)         │ │
│  │  · 工作流编排 │    │  · 命令分发 │    │   · RTSP支持        │ │
│  │  · 智能分析   │    │  · 事件收集 │    │   · 网络摄像头       │ │
│  │  · LLM决策   │    │  · 配置同步 │    │                     │ │
│  └──────┬───────┘    └──────┬──────┘    └─────────────────────┘ │
└─────────┼──────────────────┼─────────────────────────────────────┘
          │                  │
          │ WebSocket        │ node.invoke (RPC)
          │ Gateway Protocol │
          │                  │
    ┌─────▼──────────────────▼─────────────────────────────────┐
    │              LokmEye iOS App (Node模式)                   │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │           Gateway Node Client                       │ │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
    │  │  │  Bonjour发现 │  │ WebSocket连接│  │ 配对认证   │ │ │
    │  │  └──────────────┘  └──────────────┘  └────────────┘ │ │
    │  └─────────────────────────────────────────────────────┘ │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │           Command Handlers (命令处理器)              │ │
    │  │  · camera.snap      · detection.start              │ │
    │  │  · camera.clip      · detection.stop               │ │
    │  │  · camera.list      · detection.config             │ │
    │  └─────────────────────────────────────────────────────┘ │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │           Vision Engine (视觉引擎)                   │ │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
    │  │  │人形检测(VN)  │  │运动检测(光流)│  │ 物体分类   │ │ │
    │  │  │· 边界框输出  │  │· 区域变化   │  │· Core ML  │ │ │
    │  │  │· 置信度评分  │  │· 阈值可调   │  │· 本地推理 │ │ │
    │  │  └──────────────┘  └──────────────┘  └────────────┘ │ │
    │  └─────────────────────────────────────────────────────┘ │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │           Event Pipeline (事件管道)                  │ │
    │  │  采集 → 预处理 → 检测 → 过滤 → 上报 → 确认         │ │
    │  │  · 本地缓存(断网)  · 批量上报  · 去重合并           │ │
    │  └─────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────┘
                              │
                              │ APNs Push Notification
                              ▼
                    ┌─────────────────┐
                    │   用户手机       │
                    │ (Telegram/微信) │
                    └─────────────────┘
```

### 1.3 核心概念定义

| 术语 | 定义 | 说明 |
|------|------|------|
| **SKILL** | OpenClaw的技能单元 | 包含`SKILL.md` + 可选代码，定义一组相关工具和命令 |
| **Node** | 设备节点 | iOS App作为OpenClaw Gateway的子节点，通过`node.*`协议通信 |
| **Gateway** | OpenClaw网关 | 控制平面，管理所有Node、处理命令分发、运行SKILL |
| **Command** | 远程命令 | Gateway通过WebSocket向iOS Node发送的执行指令 |
| **Event** | 检测事件 | iOS Node主动向Gateway上报的视觉检测结果 |
| **APNs** | 苹果推送服务 | 用于后台事件通知唤醒用户 |

---

## 2. Phase 2: OpenClaw SKILL + iOS Node 开发

**目标**: 建立完整的SKILL-Node通信体系，实现基础监控能力  
**时间**: 3-4周  
**状态**: 🔴 Not Started

### 2.1 里程碑 M3: 基础框架搭建 (Week 1)

**目标**: 建立SKILL框架和Node连接能力

#### 2.1.1 OpenClaw端: LokmEye SKILL 框架

**任务清单**:
- [ ] 创建 `skills/lokmeye/SKILL.md` 基础结构
  ```yaml
  ---
  name: lokmeye
  description: 智能视觉监控SKILL - 将iPhone变为AI监控眼
  metadata:
    openclaw:
      emoji: 👁️
      requires:
        node: ios  # 需要iOS node支持
  ---
  ```
- [ ] 定义前置条件检查逻辑（检查iOS Node是否已配对）
- [ ] 实现SKILL加载时的依赖验证
- [ ] 配置权限门控（仅授权用户可以访问监控命令）

**交付物**:
- `skills/lokmeye/SKILL.md` (已可被OpenClaw识别)
- 前置条件检查通过
- SKILL在`openclaw doctor`中显示为可用

#### 2.1.2 iOS端: Gateway Node Client

**任务清单**:
- [ ] 实现 `GatewayNodeClient` 类 (WebSocket管理)
  ```swift
  actor GatewayNodeClient {
      func connect(to host: String, port: Int) async throws
      func disconnect() async
      func send(event: NodeEvent) async throws
      func handle(command: NodeCommand) async -> NodeResult
  }
  ```
- [ ] Bonjour服务发现 (`NetServiceBrowser`)
  - 发现局域网内OpenClaw Gateway
  - 自动填充连接配置
- [ ] 配对流程实现
  - 生成/显示配对码
  - 调用 `/pair` 和 `/pair approve`
  - 存储配对凭证 (Keychain)
- [ ] Node能力注册 (`node.describe`)
  - 返回支持的命令列表
  - 返回权限状态

**交付物**:
- iOS App成功注册为Node
- 在OpenClaw Gateway中显示为已连接设备
- 配对流程完整跑通

**关键检查点** ✅:
- [ ] LokmEye SKILL 可被OpenClaw加载
- [ ] iOS App成功注册为Node
- [ ] 两端WebSocket连接稳定

---

### 2.2 里程碑 M4: 相机与检测命令 (Week 2)

**目标**: 实现SKILL命令集和基础检测能力

#### 2.2.1 SKILL命令定义 (SKILL.md)

**新增命令**:

| 命令 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `lokmeye.camera.snap` | `camera?: front\|back`, `quality?: high\|medium\|low` | `{image: base64, timestamp, metadata}` | 拍照 |
| `lokmeye.camera.clip` | `duration: number`, `camera?` | `{video: base64, timestamp}` | 录视频片段 |
| `lokmeye.camera.list` | - | `[{id, name, position, available}]` | 列出相机 |
| `lokmeye.detection.start` | `mode: person\|motion\|all`, `config?: DetectionConfig` | `{sessionId, status}` | 开始检测 |
| `lokmeye.detection.stop` | `sessionId` | `{status}` | 停止检测 |
| `lokmeye.detection.status` | - | `{running, mode, uptime, eventsCount}` | 检测状态 |

**DetectionConfig Schema**:
```typescript
interface DetectionConfig {
  personConfidence?: number;      // 人形置信度阈值 (0-1)
  motionThreshold?: number;       // 运动检测阈值
  detectionRegion?: BoundingBox;  // 检测区域 (归一化坐标)
  cooldownMs?: number;            // 事件冷却时间
  maxFPS?: number;                // 最大处理帧率
}
```

#### 2.2.2 iOS Node命令处理器

**任务清单**:
- [ ] 实现 `CameraCommandHandler`
  - `camera.snap`: 拍照 → Base64编码 → 返回
  - `camera.clip`: 录制 → 临时文件 → Base64 → 返回
  - `camera.list`: 查询AVCaptureDevice
- [ ] 实现 `DetectionCommandHandler`
  - `detection.start`: 启动Vision检测循环
  - `detection.stop`: 停止检测循环
  - 配置实时应用（无需重启）
- [ ] 权限检查中间件
  - 相机权限未授权时返回 `PERMISSION_MISSING`
  - 包含权限申请引导

#### 2.2.3 事件上报机制

**事件类型**:

| 事件 | 字段 | 说明 |
|------|------|------|
| `lokmeye.event.person_detected` | `timestamp`, `confidence`, `boundingBox`, `thumbnail` | 人形检测 |
| `lokmeye.event.motion_detected` | `timestamp`, `region`, `intensity`, `thumbnail` | 运动检测 |
| `lokmeye.event.camera_error` | `error`, `timestamp` | 相机错误 |
| `lokmeye.event.detection_started` | `mode`, `timestamp` | 检测启动 |
| `lokmeye.event.detection_stopped` | `timestamp`, `reason` | 检测停止 |

**上报策略**:
- 实时上报（WebSocket连接正常时）
- 本地队列缓存（断网时）
- 批量重传（恢复连接后）

**交付物**:
- 完整的SKILL命令实现
- 命令处理器单元测试
- 事件上报端到端测试

**关键检查点** ✅:
- [ ] Telegram/WhatsApp中输入 `/lokmeye camera snap` 触发iOS拍照
- [ ] 检测到人物时收到OpenClaw消息通知
- [ ] 断网恢复后事件自动补传

---

### 2.3 里程碑 M5: 智能工作流 (Week 3-4)

**目标**: 实现智能分析和工作流编排

#### 2.3.1 高级SKILL命令

| 命令 | 说明 |
|------|------|
| `lokmeye.monitor` | 一键启动监控（包含检测+上报+通知） |
| `lokmeye.analyze` | 分析最近N个事件（调用LLM生成报告） |
| `lokmeye.alert.config` | 配置告警规则 |
| `lokmeye.stats` | 查看检测统计 |
| `lokmeye.snapshot` | 获取当前实时画面 |

#### 2.3.2 工作流编排示例

**场景1: 简单监控模式**
```
用户: /lokmeye monitor start
OpenClaw:
  1. 发送 detection.start 到 iOS Node
  2. 订阅 lokmeye.event.* 事件
  3. 收到 person_detected:
     - 发送APNs推送通知用户
     - 保存事件到本地存储
     - 等待用户确认
```

**场景2: 智能分析**
```
用户: /lokmeye analyze last-hour
OpenClaw:
  1. 查询最近1小时事件
  2. 调用LLM分析异常模式
  3. 生成报告并发送给用户
```

#### 2.3.3 配置同步机制

**配置层级**:
```
1. OpenClaw Gateway配置 (最高优先级)
   ~/.openclaw/openclaw.json → agents.*.skills.lokmeye
   
2. SKILL运行时配置
   通过 node.invoke 下发到iOS
   
3. iOS本地缓存
   断网时使用最后同步的配置
```

**同步触发时机**:
- Node连接时自动同步
- 配置变更时实时推送
- 每5分钟心跳同步

**交付物**:
- 完整工作流实现
- 配置同步文档
- 性能基准测试

**关键检查点** ✅:
- [ ] `/lokmeye monitor start` 一键启动完整监控
- [ ] 配置变更实时生效
- [ ] 7x24小时稳定性测试通过

---

## 3. Phase 3: 闭环优化与产品化

**目标**: 实现完整闭环反馈和产品级体验  
**时间**: 2-3周  
**状态**: 🔴 Not Started

### 3.1 里程碑 M6: 确认反馈与学习 (Week 1)

**目标**: 建立事件确认-反馈-优化闭环

#### 3.1.1 事件确认流程

**用户交互流程**:
```
1. iOS检测到人物 → 上报OpenClaw
2. OpenClaw发送APNs推送:
   "👁️ LokmEye: 检测到人物 [缩略图]
    [确认正常] [确认异常] [误报]"
3. 用户点击按钮 → 回传确认结果
4. OpenClaw记录反馈
```

**SKILL命令**:
- `lokmeye.event.confirm` - 确认事件
- `lokmeye.event.dismiss` - 忽略事件
- `lokmeye.event.false_positive` - 标记误报

#### 3.1.2 智能降噪

**去重策略**:
- 空间去重：同一区域内事件合并
- 时间去重：5秒内同类事件合并
- 置信度去重：只保留最高置信度

**动态阈值调整**:
```typescript
// 基于反馈自动调整
if (falsePositiveRate > 0.3) {
  config.personConfidence += 0.1;  // 提高阈值
}
if (missRate > 0.2) {
  config.personConfidence -= 0.05; // 降低阈值
}
```

**时间规则**:
- 夜间模式 (22:00-06:00): 高灵敏度
- 白天模式 (06:00-22:00): 标准灵敏度
- 可自定义时间段规则

**交付物**:
- 事件确认UI/UX
- 智能降噪算法
- 反馈数据统计

**关键检查点** ✅:
- [ ] 用户可收到推送并确认事件
- [ ] 误报率随使用降低
- [ ] 事件去重准确率 > 95%

---

### 3.2 里程碑 M7: 协同优化 (Week 2)

**目标**: OpenClaw调度优化与省电策略

#### 3.2.1 协同省电策略

**OpenClaw调度决策**:
```typescript
enum PowerMode {
  HIGH,    // 30 FPS, 全检测
  BALANCED,// 15 FPS, 人形检测
  LOW,     // 5 FPS, 运动检测
  SLEEP    // 1 FPS/分钟, 仅心跳
}

// 自动调度逻辑
if (noEventFor(30min)) → switchTo(LOW)
if (userAway) → switchTo(SLEEP)
if (eventDetected) → switchTo(HIGH)
```

**iOS实现**:
- 帧率动态调整 (`videoMinFrameDuration`)
- 检测器动态切换（运动检测优先）
- 后台保活策略（Location更新触发）

#### 3.2.2 网络优化

**传输优化**:
- 缩略图WebP压缩（比JPEG小30%）
- 渐进式上传（先传缩略图，再传原图）
- 批量上报（5个事件一起发送）

**断网处理**:
- 指数退避重连（1s → 2s → 4s → ... → 60s）
- 本地SQLite缓存（最多7天）
- 恢复后优先级队列（紧急事件优先）

**交付物**:
- 省电模式实现
- 网络优化方案
- 性能监控数据

**关键检查点** ✅:
- [ ] 低功耗模式下续航 > 8小时
- [ ] 断网恢复后100%事件补传
- [ ] 内存占用 < 150MB

---

### 3.3 里程碑 M8: 配置中心 (Week 3)

**目标**: 多设备管理与可视化配置

#### 3.3.1 Web UI配置中心

**功能模块**:
- 设备列表（多个LokmEye iOS设备）
- 实时预览（Canvas显示当前画面）
- ROI设置（可视化绘制检测区域）
- 规则引擎（IF-THEN配置）
- 事件时间线（筛选/搜索/导出）

#### 3.3.2 场景模板

**预设模板**:
- 门口监控: 检测人形，高灵敏度，24小时
- 客厅监控: 检测运动+人形，中灵敏度，仅夜间
- 车库监控: 检测运动，低灵敏度，仅检测时段

#### 3.3.3 SKILL版本管理

- ClawHub发布流程
- 版本兼容性检查
- 自动更新机制

**交付物**:
- Web配置中心
- 场景模板系统
- SKILL发布到ClawHub

**关键检查点** ✅:
- [ ] 可视化配置ROI
- [ ] 多设备统一管理
- [ ] SKILL在ClawHub可下载

---

## 4. 时间线与里程碑总览

### 4.1 甘特图

```
Week:    | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
         ├───┼───┼───┼───┼───┼───┼───┤
Phase 2
  M3     ████                                    基础框架
  M4         ████                               相机命令
  M5             ████████                       智能工作流
Phase 3
  M6                     ████                   确认反馈
  M7                         ████               协同优化
  M8                             ████           配置中心
         └───────────────────────────────────┘
               MVP可用 (Week 5)   完整产品 (Week 8)
```

### 4.2 关键交付节点

| 日期 | 里程碑 | 交付物 | 验收标准 |
|------|--------|--------|----------|
| Week 1 | M3 | Node连接 + SKILL框架 | iOS显示为已连接Node |
| Week 2 | M4 | 相机命令 + 事件上报 | Telegram可触发拍照 |
| Week 4 | M5 | 工作流 + 配置同步 | `/monitor`一键启动 |
| Week 5 | **MVP** | 可用监控系统 | 检测→上报→通知完整闭环 |
| Week 6 | M6 | 确认反馈机制 | 用户可确认/误报事件 |
| Week 7 | M7 | 省电优化 | 8小时续航 |
| Week 8 | M8 | 配置中心 | Web可视化配置 |

---

## 5. 人员配置与分工

### 5.1 推荐配置 (3-4人)

| 角色 | 人数 | 核心职责 | 技能要求 |
|------|------|----------|----------|
| **OpenClaw工程师** | 1 | SKILL开发 + Gateway集成 | Node.js/TS, OpenClaw架构, SKILL规范 |
| **iOS Node工程师** | 1 | Node客户端 + 通信层 | Swift, Combine, WebSocket, Bonjour |
| **视觉工程师** | 1 | AVFoundation + Vision优化 | Core ML, 实时视频, 性能调优 |
| **产品/QA** | 1 | 需求 + 验收 + 文档 | 监控场景理解, iOS测试, 用户研究 |

### 5.2 详细分工

#### OpenClaw工程师 (Alice)

**Phase 2**:
- Week 1: SKILL.md定义, 前置条件检查
- Week 2: 命令实现, 事件处理逻辑
- Week 3-4: 工作流编排, 配置同步

**Phase 3**:
- Week 5: 确认反馈流程, 数据统计
- Week 6: 省电调度策略
- Week 7: Web配置中心后端

**产出物**:
- `skills/lokmeye/` 目录
- 命令处理器代码
- 工作流编排逻辑

#### iOS Node工程师 (Bob)

**Phase 2**:
- Week 1: GatewayNodeClient, Bonjour, 配对
- Week 2: CameraCommandHandler, DetectionCommandHandler
- Week 3-4: 事件上报, 配置应用, 后台保活

**Phase 3**:
- Week 5: APNs推送处理, 确认UI
- Week 6: 省电模式实现, 帧率控制
- Week 7: Web配置中心前端(iOS端)

**产出物**:
- `apps/LokmEye-iOS/` 扩展
- `core/` 中Node相关代码
- 通信协议实现

#### 视觉工程师 (Carol)

**Phase 2**:
- Week 1-2: Vision检测优化, 多线程处理
- Week 3-4: 性能调优, 电池优化

**Phase 3**:
- Week 5: 本地LLM推理 (Core ML)
- Week 6: 智能抽帧, 动态分辨率

**产出物**:
- `LokmVision/` 模块
- 性能基准测试报告
- 电池消耗优化方案

#### 产品/QA (David)

**全程**:
- 需求澄清与验收标准定义
- 编写测试用例
- 用户测试与反馈收集
- 文档编写 (用户手册, API文档)

### 5.3 协作机制

**每日站会** (15分钟):
- 昨日进展
- 今日计划
- 阻塞问题

**每周Review**:
- Demo本周成果
- 调整下周计划
- 技术决策确认

**接口契约** (关键节点):
- Week 1结束: SKILL.md接口定义冻结
- Week 2结束: 命令格式版本锁定
- Week 4结束: 事件Schema版本锁定

---

## 6. 技术规范

### 6.1 代码规范

**OpenClaw (TypeScript)**:
- 遵循OpenClaw AGENTS.md规范
- 使用 `pnpm check` 检查
- 测试覆盖 > 70%

**iOS (Swift)**:
- Swift 5.9+
- 使用 `@Observable` 替代 `ObservableObject`
- 4空格缩进
- 文件长度 < 700行

### 6.2 接口规范

**WebSocket消息格式**:
```typescript
// 命令请求
interface NodeCommand {
  id: string;
  type: 'command';
  skill: 'lokmeye';
  action: string;
  params: Record<string, unknown>;
  timestamp: number;
}

// 命令响应
interface NodeResult {
  commandId: string;
  status: 'success' | 'error';
  data?: unknown;
  error?: {
    code: string;
    message: string;
  };
}

// 事件上报
interface NodeEvent {
  type: 'event';
  skill: 'lokmeye';
  eventType: string;
  payload: unknown;
  timestamp: number;
}
```

### 6.3 事件Schema (v1.0)

**person_detected**:
```typescript
{
  eventType: 'lokmeye.person_detected',
  timestamp: '2026-02-27T10:30:00Z',
  payload: {
    confidence: 0.92,           // 0-1
    boundingBox: {              // 归一化坐标
      x: 0.25, y: 0.30,
      width: 0.30, height: 0.50
    },
    thumbnail: 'base64...',     // WebP, max 100KB
    camera: 'back',             // front/back
    processingTimeMs: 45        // 处理耗时
  }
}
```

---

## 7. 风险与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| iOS后台限制严格 | 高 | 高 | 实施Location保活 + 后台获取 + 用户教育 |
| OpenClaw协议变更 | 中 | 高 | 跟随OpenClaw main分支, 及时适配 |
| WebSocket不稳定 | 中 | 中 | 实现指数退避重连 + 本地缓存 |
| 电池消耗过快 | 高 | 高 | 动态帧率 + 智能休眠 + 功耗监控 |
| App Store审核 | 中 | 高 | 准备隐私政策 + 权限说明 + 测试flight先行 |
| 多设备同步复杂 | 中 | 中 | 使用OpenClaw Gateway作为单一数据源 |

---

## 8. 附录

### 8.1 参考文档

- [OpenClaw AGENTS.md](/Users/zhao/Projects/lokmeye/libs/openclaw/AGENTS.md)
- [OpenClaw Skills文档](/Users/zhao/Projects/lokmeye/libs/openclaw/docs/tools/skills.md)
- [OpenClaw iOS Node指南](/Users/zhao/Projects/lokmeye/libs/openclaw/apps/ios/README.md)
- [OpenClaw Nodes工具](/Users/zhao/Projects/lokmeye/libs/openclaw/src/agents/tools/nodes-tool.ts)

### 8.2 相关项目

- OpenClaw: `libs/openclaw/`
- LokmCore: `core/` (Swift Package)
- LokmEye iOS: `apps/LokmEye-iOS/`

### 8.3 术语表

- **SKILL**: OpenClaw技能单元
- **Node**: 设备节点 (iOS App)
- **Gateway**: OpenClaw网关
- **APNs**: Apple Push Notification Service
- **ROI**: Region of Interest (检测区域)
- **VN**: Vision Framework (Apple)

---

## 9. 审批记录

| 版本 | 日期 | 作者 | 审批人 | 备注 |
|------|------|------|--------|------|
| 1.0 | 2026-02-27 | Claude | - | Draft版本, 待Review |

---

**下一步行动**:
1. [ ] Review本计划并提出修改意见
2. [ ] 确认人员配置和时间安排
3. [ ] 创建GitHub Project Board跟踪进度
4. [ ] Week 1 Kickoff会议
