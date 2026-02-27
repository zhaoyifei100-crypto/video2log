# LokmEye iOS 技术路线图 v1.2

**基于 OpenClaw Node 协议 + 独立 iOS App + AR预览 架构**

*Look My Eye = 智能监控 + AR视觉体验*

---

## 文档信息

- **版本**: 1.2
- **更新日期**: 2026-02-27
- **状态**: Draft (待Review)
- **关联项目**: LokmEye (LookMyEye) - 智能监控+AR解决方案
- **变更说明**: 
  - 采用Node模式接入OpenClaw，独立开发iOS App
  - 新增AR预览模块（Matrix猫头），体现"Look My Eye"第二层含义
  - 4人团队配置：iOS + AR + Backend + QA

---

## 1. 项目概述

### 1.1 项目愿景
让旧iPhone变身AI智能监控眼，作为OpenClaw Node接入，实现：
- **AR即界面** (Look My Eye): **Matrix猫头 = 系统主UI**，用户通过AR猫头进行语音/手势交互
- **零门槛配置**: 通过OpenClaw Gateway自动发现与配对，配置通过AR界面完成
- **智能化检测**: 本地Vision框架人形/运动检测，结果反馈在AR界面
- **自然语言交互**: 对Matrix猫头说话，通过OpenClaw SKILL控制监控

**核心理念**: "The EYE is the Interface" - 屏幕上必须有那只猫，它是产品的灵魂

### 1.2 技术架构 (AR优先)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         用户 (通过AR界面交互)                              │
│                           ↓ 语音/手势/凝视                                  │
│                    ┌──────────────────────┐                               │
│                    │   Matrix猫头 (AR UI)  │ ← 始终显示，产品灵魂          │
│                    │  · 状态显示           │                               │
│                    │  · 语音交互           │                               │
│                    │  · 手势控制           │                               │
│                    │  · 情绪反馈           │                               │
│                    └──────────┬───────────┘                               │
└───────────────────────────────┼──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                      LokmEye iOS App - AR Layer                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                AR Interface (David - Week 2-5)                      │  │
│  │  ┌───────────────┐ ┌───────────────┐ ┌──────────────────────┐     │  │
│  │  │ Matrix猫头    │ │ RealityKit    │ │ Metal Shaders        │     │  │
│  │  │ 3D Model      │ │ Scene         │ │ (Digital Rain)       │     │  │
│  │  └───────────────┘ └───────────────┘ └──────────────────────┘     │  │
│  │  ┌───────────────┐ ┌───────────────┐ ┌──────────────────────┐     │  │
│  │  │ ARKit Tracking│ │ Interaction   │ │ Visual Effects       │     │  │
│  │  │ (Face/World)  │ │ (Voice/Gesture│ │ (Bloom/Animations)   │     │  │
│  │  └───────────────┘ └───────────────┘ └──────────────────────┘     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │           AR-Vision Bridge (Alex - 架构预留)                        │  │
│  │   · 检测结果 → 猫头反馈 (眨眼/变色/旋转)                            │  │
│  │   · 语音命令 ← 猫头交互                                            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                │
                                │ 后台监控功能 (Alex - Week 1-5)
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                    LokmEye iOS App - Backend Layer                       │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │      Node Client             │  │      Vision Engine               │  │
│  │  · WebSocket Manager         │  │  · Person Detection (VN)         │  │
│  │  · Bonjour Discovery         │  │  · Motion Detection              │  │
│  │  · Pairing/Auth              │  │  · Event Stream                  │  │
│  │  · Command Router            │  │  · Local Storage                 │  │
│  └──────────────┬───────────────┘  └──────────────────────────────────┘  │
│                 │                                                         │
│                 │ WebSocket                                               │
└─────────────────┼─────────────────────────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────────────────────────┐
│                    OpenClaw Gateway (外部系统，不修改)                    │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐ │
│  │   nodes-tool        │  │   webhook           │  │   SKILL Loader    │ │
│  │   (command router)  │  │   (event receiver)  │  │   (lokmeye-skill) │ │
│  └─────────────────────┘  └─────────────────────┘  └───────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心概念与边界

| 组件 | 责任方 | 说明 |
|------|--------|------|
| **OpenClaw Gateway** | OpenClaw团队维护 | 我们不修改，只使用标准Node协议 |
| **Node Protocol** | OpenClaw定义 | WebSocket + JSON-RPC，支持`node.invoke`和事件推送 |
| **LokmEye iOS** | 我们开发 | 独立App，实现Node客户端 + Vision检测 |
| **LokmEye SKILL** | 我们开发 | 用户侧配置，通过OpenClaw标准工具调用 |
| **Webhook** | Gateway配置 | 接收LokmEye事件，转发给Main Agent |

### 1.4 与OpenClaw的关系

**不复用OpenClaw iOS App代码**，原因：
1. 简化依赖，降低耦合
2. 独立迭代，不受OpenClaw发布周期影响
3. 专注监控场景，UI更简洁

**复用OpenClaw的设计思想和协议**：
1. Node协议 (WebSocket连接、配对、命令处理)
2. Camera实现参考 (但不直接复制)
3. 事件格式规范

---

## 2. 角色分工与工作分配

### 2.1 团队配置 (推荐 4人)

| 角色 | 核心技能 | 主要职责 | 入场时间 |
|------|----------|----------|----------|
| **iOS Lead Engineer (Alex)** | Swift, AVFoundation, Vision, WebSocket | **AR架构预留** + Node协议 + Vision引擎 + 系统集成 | **Week 1** (立即) |
| **AR Graphics Engineer (David)** | Swift, RealityKit, Metal, ARKit | **Matrix猫头核心UI** + RealityKit渲染 + 3D交互 | **Week 2-3** (后续加入) |
| **Backend/SKILL Engineer (Bob)** | TypeScript, Node.js, OpenClaw | SKILL开发 + Gateway配置 + 事件处理 | **Week 1** (并行) |
| **Product/QA (Carol)** | iOS测试, 产品思维 | 需求确认 + 测试用例 + 验收 | 全程 |

**核心架构理念**：
> **"The EYE is the Interface"** - Matrix猫头不是附加功能，而是主UI

- **David负责**: Matrix猫头 = 系统主界面，用户通过猫头进行所有交互
- **Alex负责**: 架构预留AR层 + 后台监控功能 + 系统集成
- **关键理解**: AR猫头**始终显示**（或作为唤醒后的主界面），监控通过AR界面操作

### 2.2 详细工作分配

#### Alex (iOS Lead) - 约 4.5周工作量

**Week 1-2: Node Client 基础设施**
- [ ] **2.1.1** WebSocket连接管理器
  - 建立/断开连接，心跳保活，重连机制
  - 依赖: 无
  - 产出: `WebSocketManager.swift`
  
- [ ] **2.1.2** Node协议实现
  - `node.describe` 响应
  - `node.invoke` 命令路由
  - 依赖: 2.1.1
  - 产出: `NodeClient.swift`, `CommandRouter.swift`

- [ ] **2.1.3** 配对与认证
  - Bonjour服务发现
  - 配对流程 (setup code → approve)
  - Token存储 (Keychain)
  - 依赖: 2.1.1
  - 产出: `PairingService.swift`, `KeychainStore.swift`

- [ ] **2.1.4** 标准Camera命令
  - `camera.list`, `camera.snap`, `camera.clip`
  - 参考OpenClaw实现，但不直接复制
  - 依赖: 2.1.2
  - 产出: `CameraCommandHandler.swift`

- [ ] **2.1.5** 【关键】AR架构预留 (IC0)
  - 定义 `ARInterface` 协议 (供David实现)
  - 预留AR层调用点（检测反馈、命令触发）
  - 相机共享协议设计 (ARSession vs AVCaptureSession)
  - **产出**: `ARInterface.swift` (协议定义), 架构文档
  - **阻塞David进场**: 必须在Week 1结束完成

**Week 2-3: Vision Engine + AR集成**
- [ ] **2.2.1** 相机预览与采集
  - AVCaptureSession管理
  - 实时帧获取 (CVPixelBuffer)
  - 依赖: 2.1.4
  - 产出: `CameraCaptureService.swift`

- [ ] **2.2.2** 人形检测 (VN)
  - VNDetectHumanRectanglesRequest
  - 边界框转换 (normalized coordinates)
  - 依赖: 2.2.1
  - 产出: `PersonDetector.swift`

- [ ] **2.2.3** 运动检测
  - 帧差法 / 光流算法
  - 运动区域计算
  - 依赖: 2.2.1
  - 产出: `MotionDetector.swift`

- [ ] **2.2.4** Vision命令处理器
  - `lokmeye.vision.start/stop/status`
  - 检测配置应用 (threshold, ROI)
  - 依赖: 2.1.2, 2.2.2, 2.2.3
  - 产出: `VisionCommandHandler.swift`

**Week 3-4: 事件与优化**
- [ ] **2.3.1** 事件发射器
  - 主动推送事件到Gateway (扩展协议)
  - 事件队列与重传
  - 依赖: 2.1.1
  - 产出: `EventEmitter.swift`

- [ ] **2.3.2** 本地配置管理
  - 配置持久化 (UserDefaults/JSON)
  - 运行时配置更新
  - 依赖: 2.2.4
  - 产出: `ConfigManager.swift`

- [ ] **2.3.3** 后台保活
  - Location更新触发检测
  - 省电模式切换
  - 依赖: 2.2.4
  - 产出: `BackgroundTaskManager.swift`

- [ ] **2.3.4** iOS UI (简化版)
  - 连接状态显示
  - 实时预览 (可选)
  - 设置页面
  - 依赖: 2.1.3
  - 产出: `ContentView.swift`, `SettingsView.swift`

**Week 4-5: 集成与测试**
- [ ] **2.4.1** 端到端联调
  - 与Bob的SKILL对接
  - 命令-响应完整测试
  - 依赖: 2.1.2, 2.2.4, 3.1.1

- [ ] **2.4.2** 性能优化
  - 内存优化 (< 150MB)
  - 电池优化 (8小时续航)
  - FPS稳定性

- [ ] **2.4.3** 错误处理与边界情况
  - 网络断开恢复
  - 权限拒绝处理
  - 相机占用冲突

#### David (AR Graphics Engineer) - 约 3.5周工作量

**与Alex的协作模式**：
- **AR是主UI**，David负责核心交互层，Alex负责后台数据层
- 架构预留：Alex Week 1定义ARInterface协议，David Week 2实现
- 数据流：Vision检测结果 → ARInterface → Matrix猫头反馈（情绪/状态）
- 用户流：用户与猫头交互 → ARInterface → Node命令 → Gateway

**David入场时间：Week 2（Alex架构预留完成后）**

**Week 2-3: Matrix猫头核心UI**
- [ ] **4.1.1** Matrix数字雨着色器 (Metal)
  - 顶点着色器：字符位置矩阵变换
  - 片段着色器：绿色荧光 + 数字纹理采样 + 时间uniform
  - 噪声纹理生成（随机数字列）
  - 依赖: 无
  - 产出: `MatrixRainShader.metal`

- [ ] **4.1.2** 程序化猫头轮廓生成
  - 使用SceneKit/RealityKit几何体构建猫头形状
  - UV映射适配数字雨纹理
  - 边缘高亮材质（Fresnel效果）
  - 依赖: 4.1.1
  - 产出: `CatHeadGeometry.swift`

- [ ] **4.1.3** RealityKit自定义材质系统
  - 将Metal着色器应用到RealityKit实体
  - 动态uniform更新（时间、速度）
  - 性能优化（GPU Instancing）
  - 依赖: 4.1.1, 4.1.2
  - 产出: `MatrixMaterial.swift`

- [ ] **4.1.4** 主AR界面搭建（核心UI）
  - ARView配置（World Tracking或Face Tracking）
  - **Matrix猫头始终显示在屏幕中央**
  - 环境光照估计
  - 相机背景渲染
  - 依赖: 无
  - 产出: `AREyeInterface.swift`（主UI层）

**Week 2-3: 主UI交互系统**
- [ ] **4.2.1** ARKit空间追踪集成
  - ARWorldTrackingConfiguration配置
  - 平面检测（放置猫头于现实世界）
  - 或Face Tracking模式（猫头跟随人脸）
  - 依赖: 4.1.4
  - 产出: `ARTrackingManager.swift`

- [ ] **4.2.2** 平滑旋转与物理效果
  - 球面线性插值（Slerp）平滑旋转
  - 弹簧物理系统（惯性回弹）
  - 最大角度限制（Yaw ±60°, Pitch ±45°）
  - 依赖: 4.2.1
  - 产出: `CatPhysicsController.swift`

- [ ] **4.2.3** 视觉反馈系统
  - 眨眼检测 → 眼睛高亮闪烁
  - 距离感知 → 数字雨速度变化
  - 检测触发 → 猫头出现/消失动画
  - 依赖: 4.2.1, Alex的2.2.2
  - 产出: `ARFeedbackSystem.swift`

**Week 3-4: 与监控系统集成**
- [ ] **4.3.1** AR状态反馈系统（主UI反馈后台状态）
  - Vision检测结果 → 猫头情绪/颜色变化
  - 连接状态 → 猫头眼神/亮度反馈
  - 系统状态 → 猫头旋转速度/数字雨密度
  - 依赖: Alex的2.2.2, 4.2.1
  - 产出: `AREyeFeedback.swift`

- [ ] **4.3.2** 扩展交互与功能
  - 长按猫头 → 快捷菜单
  - 双击猫头 → 拍照/录像
  - 语音唤醒 → 猫头激活动画
  - 依赖: 4.3.1
  - 产出: `AREyeInteractions.swift`

- [ ] **4.3.3** AR录制与分享（内容创作）
  - RealityKit屏幕录制（带猫头）
  - AR场景+真实背景合成
  - 社交媒体分享
  - 依赖: 4.3.2
  - 产出: `AREyeRecorder.swift`

- [ ] **4.3.4** 性能优化与适配
  - iPhone 12以下机型降级方案（简化效果）
  - 帧率稳定60FPS（UI流畅度）
  - 内存占用控制
  - 依赖: 4.3.1-4.3.3
  - 产出: `AREyeOptimizer.swift`

**Week 4-5: 集成与打磨**
- [ ] **4.4.1** 与Alex主App集成
  - Swift Package Manager集成
  - 相机共享协调（ARSession vs AVCaptureSession）
  - 生命周期管理（前后台切换）
  - 依赖: Alex的2.3.4, 4.3.4

- [ ] **4.4.2** AR命令实现
  - `lokmeye.ar.start` - 启动AR预览
  - `lokmeye.ar.stop` - 停止AR预览
  - `lokmeye.ar.record` - 录制AR场景
  - 依赖: Alex的2.1.2, 4.4.1
  - 产出: `ARCommandHandler.swift`

- [ ] **4.4.3** UI/UX打磨
  - AR模式切换UI
  - 手势控制（缩放、旋转猫头）
  - 视觉效果调优（Bloom、泛光）
  - 依赖: 4.4.1
  - 产出: `ARUIOverlay.swift`

#### Bob (Backend/SKILL) - 约 3.5周工作量

**Week 1: Gateway配置与SKILL框架**
- [ ] **3.1.1** LokmEye SKILL框架
  - `SKILL.md` 基础定义
  - 命令映射到 `nodes.invoke`
  - 依赖: 无
  - 产出: `skills/lokmeye/SKILL.md`

- [ ] **3.1.2** Gateway Webhook配置
  - 配置事件接收端点
  - 事件过滤与转发
  - 依赖: 无
  - 产出: `gateway-webhook-config.md`

- [ ] **3.1.3** 标准命令包装
  - `/lokmeye camera snap` → `nodes camera snap`
  - `/lokmeye camera list` → `nodes camera list`
  - 依赖: 3.1.1
  - 产出: `skills/lokmeye/tools/camera.ts`

**Week 2: Vision命令与事件处理**
- [ ] **3.2.1** Vision命令包装
  - `/lokmeye vision start` → `nodes invoke lokmeye.vision.start`
  - `/lokmeye vision stop` → `nodes invoke lokmeye.vision.stop`
  - `/lokmeye vision status`
  - 依赖: 3.1.1, Alex的2.2.4
  - 产出: `skills/lokmeye/tools/vision.ts`

- [ ] **3.2.2** 事件接收处理
  - Webhook接收 `lokmeye.event.*`
  - 事件解析与存储
  - 依赖: 3.1.2, Alex的2.3.1
  - 产出: `skills/lokmeye/events/handler.ts`

- [ ] **3.2.3** 通知集成
  - 事件 → APNs推送
  - 推送内容格式化
  - 依赖: 3.2.2
  - 产出: `skills/lokmeye/notifications.ts`

**Week 3: 工作流与配置**
- [ ] **3.3.1** 一键监控工作流
  - `/lokmeye monitor start` 命令
  - 启动检测 + 订阅事件 + 配置通知
  - 依赖: 3.2.1, 3.2.3
  - 产出: `skills/lokmeye/workflows/monitor.ts`

- [ ] **3.3.2** 配置管理
  - `/lokmeye config set` 命令
  - 配置下发到iOS Node
  - 依赖: 3.2.1
  - 产出: `skills/lokmeye/tools/config.ts`

- [ ] **3.3.3** 统计与报告
  - `/lokmeye stats` 命令
  - 事件统计查询
  - 依赖: 3.2.2
  - 产出: `skills/lokmeye/tools/stats.ts`

**Week 4: 高级功能**
- [ ] **3.4.1** 确认反馈机制
  - 推送通知带确认按钮
  - 确认结果回传到iOS
  - 依赖: 3.2.3, 3.3.1
  - 产出: `skills/lokmeye/feedback.ts`

- [ ] **3.4.2** 智能分析集成
  - 调用LLM分析异常模式
  - 生成自然语言报告
  - 依赖: 3.2.2
  - 产出: `skills/lokmeye/analysis.ts`

- [ ] **3.4.3** 场景模板
  - 门口/客厅/车库预设配置
  - 一键切换场景
  - 依赖: 3.3.2
  - 产出: `skills/lokmeye/templates/*.json`

#### Carol (Product/QA) - 全程并行

**持续进行**:
- [ ] 编写用户故事和验收标准
- [ ] 设计测试用例 (功能/性能/边界)
- [ ] 进行手动测试和验收
- [ ] 编写用户文档和快速入门指南
- [ ] 收集用户反馈并创建Issue

**关键节点**:
- Week 1结束: 确认Node协议接口定义
- Week 2结束: 确认Vision检测准确性标准
- Week 3结束: 确认事件流完整性
- Week 4结束: MVP验收测试
- Week 5结束: 完整产品验收

### 2.3 协作接口与依赖

#### 接口契约 (关键同步点)

**IC0: AR架构契约 (Week 1结束，阻塞David进场)**
```swift
// Alex定义 → David实现
protocol ARInterface: AnyObject {
    // 系统状态反馈到AR层
    func showConnectionState(_ state: ConnectionState)
    func showDetectionAlert(_ event: DetectionEvent)
    func showSystemStatus(_ status: SystemStatus)
    
    // AR层触发系统命令
    var onVoiceCommand: ((String) -> Void)? { get set }
    var onGesture: ((GestureType) -> Void)? { get set }
    var onStatusTap: (() -> Void)? { get set }
}

// 数据模型
struct DetectionEvent {
    let type: DetectionType
    let confidence: Double
    let position: SIMD3<Float>?  // AR空间位置
}

enum ConnectionState {
    case connecting, connected, disconnected, error
}

// 相机共享协议
defaultCameraMode: .arPriority  // AR优先，Vision后台运行
```

**IC1: Node协议基础 (Week 1结束)**
```typescript
// Alex提供: Swift结构
struct NodeDescription {
    commands: [String]      // 支持的命令列表
    capabilities: [String]  // 能力标识
    version: String         // 协议版本
}

// Bob依赖: 用于SKILL命令定义
```

**IC2: Camera命令 (Week 2开始)**
```typescript
// 标准命令 (复用OpenClaw)
camera.list → {devices: [{id, name, position}]}
camera.snap → {format, base64, width, height}
camera.clip → {format, base64, durationMs, hasAudio}
```

**IC3: Vision命令 (Week 3开始)**
```typescript
// 自定义命令
lokmeye.vision.start(params: {
    mode: 'person' | 'motion' | 'all'
    confidence?: number
    threshold?: number
    region?: BoundingBox
}) → {sessionId, status}

lokmeye.vision.stop(sessionId) → {status}
lokmeye.vision.status() → {running, mode, eventsCount}
```

**IC4: 事件格式 (Week 3开始)**
```typescript
// Alex发送, Bob接收
interface LokmeyeEvent {
    type: 'lokmeye.person_detected' | 'lokmeye.motion_detected'
    timestamp: string
    payload: {
        confidence: number
        boundingBox?: {x, y, width, height}
        thumbnail?: string  // base64, webp
        camera: 'front' | 'back'
    }
}
```

**IC5: 配置Schema (Week 2结束)**
```typescript
interface LokmeyeConfig {
    detection: {
        mode: 'person' | 'motion' | 'all'
        confidence: number      // 0-1
        threshold: number       // 运动阈值
        cooldownMs: number      // 事件冷却
        maxFPS: number          // 最大帧率
    }
    region?: BoundingBox        // 检测区域
    powerMode: 'high' | 'balanced' | 'low'
}
```

**IC6: AR检测桥接 (Week 3开始)**
```swift
// Alex提供检测事件 → David消费
struct VisionDetectionEvent {
    let type: DetectionType  // .person, .motion
    let boundingBox: CGRect  // 归一化坐标 (0-1)
    let confidence: Double
    let timestamp: Date
    let depth: Float?        // AR深度信息 (如果可用)
}

// David提供AR状态 → Alex查询
struct ARSessionState {
    let isRunning: Bool
    let trackingState: ARTrackingState
    let cameraMode: ARCameraMode  // .arSession or .avCapture
}
```

**IC7: AR命令 (Week 4开始)**
```typescript
// 新增AR命令
lokmeye.ar.start(params: {
    mode: 'world' | 'face' | 'off'
    showOnDetection: boolean  // 检测到时自动显示
}) → {status, sessionId}

lokmeye.ar.stop(sessionId) → {status}

lokmeye.ar.record(params: {
    duration?: number
    quality: 'high' | 'medium' | 'low'
}) → {videoBase64}
```

**IC8: 相机共享协议 (Week 2结束)**
```swift
// Alex和David协调相机使用
enum CameraMode {
    case visionOnly      // 仅监控模式
    case arOnly          // 仅AR预览模式
    case hybrid          // 混合模式（AR优先，间歇检测）
}

// 相机切换时序
// 1. AR启动时 → Alex暂停AVCapture，David启动ARSession
// 2. AR关闭时 → David停止ARSession，Alex恢复AVCapture
// 3. Hybrid模式 → ARSession运行，Vision从ARFrame获取图像
```

#### 依赖关系图

```
Week 1:
  Alex:  2.1.1 → 2.1.2 → 2.1.3 → 2.1.4
  David: 4.1.1 → 4.1.2 → 4.1.3 → 4.1.4
  Bob:   3.1.1 → 3.1.2 → 3.1.3
  
Week 2:
  Alex:  2.2.1 → 2.2.2 → 2.2.3 → 2.2.4
         ↓              ↑
  David: 4.2.1 → 4.2.2 → 4.2.3
         ↓ (IC8相机协议)
  Bob:   3.2.1 (依赖 Alex 2.2.4)
  
Week 3:
  Alex:  2.3.1 → 2.3.2 → 2.3.3 → 2.3.4
         ↓ (IC6检测事件) ↓
  David: 4.3.1 → 4.3.2 → 4.3.3 → 4.3.4
                ↑
  Bob:   3.2.2 (依赖 Alex 2.3.1)
         3.2.3 (依赖 Alex 2.3.1)
         3.3.1 (依赖 Alex 2.2.4, 2.3.1)
         
Week 4:
  Alex:  2.4.1 → 2.4.2 → 2.4.3
         ↑      ↑ (IC7 AR命令)
  David: 4.4.1 → 4.4.2 → 4.4.3
         ↑
  Bob:   3.3.2 → 3.3.3 → 3.4.1 → 3.4.2 → 3.4.3
         
Week 5:
  联合测试与Bug修复
  AR-MVP验收 (David 4.3.x 完成)
```

---

## 3. Phase 详细规划

### Phase 1: Node Client 基础设施 (Week 1-2)

**目标**: 建立与OpenClaw Gateway的稳定连接，实现标准Camera命令

#### Week 1 详细任务

**Alex - Day 1-2: WebSocket与Node协议**
```swift
// NodeProtocol.swift
protocol NodeProtocol {
    func connect(to url: URL) async throws
    func disconnect() async
    func sendEvent(_ event: NodeEvent) async throws
}

// NodeClient.swift (Actor)
actor NodeClient: NodeProtocol {
    private var webSocketTask: URLSessionWebSocketTask?
    private var commandHandlers: [String: CommandHandler] = [:]
    
    func handleIncomingMessage(_ message: String) async {
        // 解析JSON-RPC
        // 路由到对应handler
    }
}
```

**验收标准**:
- [ ] 成功连接到Gateway
- [ ] 发送`node.describe`得到正确响应
- [ ] 心跳保活正常

**Alex - Day 3-4: 配对与认证**
```swift
// PairingService.swift
actor PairingService {
    func startPairing() async throws -> String  // 返回setup code
    func completePairing(with token: String) async throws
    func storeToken(_ token: String) throws  // Keychain
    func retrieveToken() throws -> String?
}
```

**验收标准**:
- [ ] Bonjour发现Gateway
- [ ] 生成setup code
- [ ] 用户approve后保存token
- [ ] 断开后用token自动重连

**Alex - Day 5: Camera基础**
- 实现`camera.list` (列出设备)
- 实现`camera.snap`基础版 (拍照返回base64)

**Bob - Day 1-2: SKILL框架**
```markdown
// skills/lokmeye/SKILL.md
---
name: lokmeye
description: LokmEye智能监控
metadata:
  openclaw:
    emoji: 👁️
    requires:
      node: ios
      capabilities: [lokmeye.vision]
---

## 命令

- `/lokmeye camera snap` - 拍照
- `/lokmeye camera list` - 列出摄像头
```

**Bob - Day 3-5: Gateway配置**
```typescript
// gateway-webhook-config.md
{
  "gateway": {
    "webhooks": [{
      "events": ["node.event.lokmeye.*"],
      "url": "http://localhost:8080/webhook/lokmeye"
    }]
  }
}
```

#### Week 2 详细任务

**Alex - Vision Engine基础**
- Day 1-2: CameraCaptureService (实时帧获取)
- Day 3-4: PersonDetector (VN实现)
- Day 5: MotionDetector (帧差法)

**验收标准**:
- [ ] 实时预览帧率 > 15 FPS
- [ ] 人形检测延迟 < 200ms
- [ ] 运动检测误报率 < 10%

**Bob - Camera命令包装**
```typescript
// tools/camera.ts
export async function snap(nodeId: string, facing?: string) {
  return await nodesTool.execute({
    action: "camera_snap",
    node: nodeId,
    facing: facing || "front"
  });
}
```

### Phase 2: Vision与事件流 (Week 2-3)

**目标**: 实现检测能力和事件主动上报

#### Week 3 详细任务

**Alex - Vision命令与事件**
- Day 1: `lokmeye.vision.start/stop/status` 命令
- Day 2: EventEmitter (主动推送)
- Day 3: 配置管理 (`lokmeye.config.set/get`)
- Day 4: 后台保活 (Location策略)
- Day 5: iOS UI (简化版)

**关键代码 - EventEmitter**:
```swift
actor EventEmitter {
    private var eventQueue: [LokmeyeEvent] = []
    private var webSocket: URLSessionWebSocketTask?
    
    func emit(_ event: LokmeyeEvent) async {
        if isConnected {
            await sendImmediate(event)
        } else {
            queueForLater(event)
        }
    }
    
    private func sendImmediate(_ event: LokmeyeEvent) async {
        let message = try! JSONEncoder().encode(event)
        try? await webSocket?.send(.string(String(data: message, encoding: .utf8)!))
    }
}
```

**Bob - Vision命令与事件处理**
- Day 1-2: Vision命令包装
- Day 3: Webhook接收器
- Day 4: 事件解析与存储
- Day 5: APNs推送集成

### Phase 3: 闭环与优化 (Week 4-5)

**目标**: 完整监控闭环，产品级体验

#### Week 4-5 详细任务

**Alex - 优化与集成**
- 省电模式实现
- 错误处理完善
- 与Bob联调
- 性能优化 (内存、电池)

**Bob - 工作流与高级功能**
- 一键监控工作流
- 确认反馈机制
- LLM分析集成
- 场景模板

---

## 4. 时间线与里程碑 (更新)

### 甘特图

```
Week:    | 1       | 2       | 3       | 4       | 5       |
         |---------|---------|---------|---------|---------|
Alex     ██████████
         2.1.1-2.1.4
                  ██████████
                  2.2.1-2.2.4
                            ██████████
                            2.3.1-2.3.4
                                      ██████████
                                      2.4.1-2.4.3
                                       
David    ██████████
         4.1.1-4.1.4
                  ██████████
                  4.2.1-4.2.3
                            ██████████
                            4.3.1-4.3.4
                                      ██████████
                                      4.4.1-4.4.3
                                       
Bob                ████████
                   3.1.1-3.1.3
                            ██████████
                            3.2.1-3.2.3
                                      ██████████
                                      3.3.1-3.4.3

Carol    ════════════════════════════════════════════════════
         需求+测试+文档 (全程)

         └── IC1 ──┘ (Week 1结束)
                   └── IC2 ──┘ (Week 2开始)
                             └── IC6 ──┘ (Week 3开始, AR桥接)
                   └── IC8 ──┘ (Week 2结束, 相机协议)
                             └── IC3 ──┘ (Week 3开始)
                                       └── IC4 ──┘ (Week 3开始)
                                                 └── IC7 ──┘ (Week 4开始, AR命令)

Milestone:
  M1 (Week 1): Node连接成功
  M2 (Week 2): Camera命令可用 + Matrix猫头渲染  
  M3 (Week 3): Vision检测+事件流 ✅ MVP
  M4 (Week 4): AR-Vision集成完成 ✅ AR-MVP
  M5 (Week 5): 完整产品+AR体验 ✅ Release
```

### 里程碑详情

| 里程碑 | 日期 | 验收标准 | 负责人 |
|--------|------|----------|--------|
| **M1** | Week 1 Fri | 1. iOS显示为已连接Node<br>2. `node.describe`返回正确能力列表<br>3. 配对流程完整<br>4. **ARInterface协议定义完成 (IC0)** ✅ | Alex |
| **M2** | Week 2 Fri | 1. `camera.snap`可用<br>2. Matrix猫头基础渲染正常<br>3. ARKit追踪+平滑旋转正常工作<br>4. **屏幕上能看到猫头** ✅ | David + Alex |
| **M3** | Week 3 Fri | 1. 人形检测正常工作<br>2. Vision检测→AR猫头反馈通<br>3. 事件主动上报到Gateway<br>4. **核心MVP: 监控+AR集成** ✅ | Alex + David + Bob |
| **M4** | Week 4 Fri | 1. `/lokmeye monitor start`一键启动<br>2. 用户确认反馈正常工作<br>3. 语音/手势交互完成<br>4. **完整AR体验** ✅ | 全员 |
| **M5** | Week 5 Fri | 1. 8小时续航测试通过<br>2. 7x24稳定性测试通过<br>3. AR录制分享功能完成<br>4. **产品发布** ✅ | 全员 |

---

## 5. 技术风险与缓解

### 5.1 高风险项

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Node协议文档不完整 | 中 | 高 | Week 1前半专注逆向OpenClaw iOS代码，验证协议 |
| iOS后台限制严格 | 高 | 高 | 采用Location保活 + 用户教育 + 降低后台期望 |
| WebSocket稳定性 | 中 | 中 | 实现指数退避重连 + 本地事件队列 |
| 电池消耗过高 | 高 | 高 | 动态帧率 + 智能休眠 + Week 3专项优化 |
| **AR性能不足** | **中** | **高** | **降级方案：简化效果/降低分辨率/关闭AR** |
| **相机AR冲突** | **中** | **高** | **明确IC8相机协议，Hybrid模式优先级** |

### 5.2 应急预案

**如果Node协议过于复杂**:
- Plan B: 复用OpenClaw iOS App源码，在其基础上添加Vision功能
- 工作量增加: +1周

**如果Vision检测性能不足**:
- Plan B: 降低分辨率 (720p → 480p)
- Plan C: 只做人形检测，不做运动检测

**如果事件推送Gateway不支持**:
- Plan B: 使用HTTP轮询 (iOS定期pull)
- Plan C: 使用MQTT broker中转

---

## 6. 资源需求

### 6.1 开发环境

| 资源 | 数量 | 用途 |
|------|------|------|
| MacBook Pro | 3台 | iOS开发 + AR开发 + Gateway运行 |
| iPhone (旧款) | 4-5台 | 测试设备 (iPhone X/11/12/13) |
| **iPhone 12 Pro+ (LiDAR)** | 1-2台 | David的AR深度测试 |
| iPad Pro (可选) | 1台 | AR大屏预览测试 |
| OpenClaw Gateway | 1个 | Bob的开发环境 |
| TestFlight账号 | 1个 | 内测分发 |
| **Blender** | 1套 | David的猫头模型调整（如需要）|

### 6.2 第三方服务

| 服务 | 用途 | 成本 |
|------|------|------|
| OpenClaw | Gateway运行 | 免费 (自托管) |
| Apple Developer | 签名 + APNs | $99/年 |
| Tailscale (可选) | 远程访问 | 免费版足够 |

---

## 7. 成功指标

### 7.1 技术指标

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| 检测延迟 | < 500ms | 从画面变化到事件上报 |
| 误报率 | < 5% | 用户标记误报 / 总事件数 |
| 续航时间 | > 8小时 | 连续监控电池消耗 |
| 内存占用 | < 150MB | Xcode Instruments |
| 连接稳定性 | > 99% | 7x24小时在线率 |
| **AR帧率** | **> 30 FPS** | **RealityKit统计** |
| **AR追踪精度** | **< 5cm误差** | **ARKit世界坐标比对** |
| **AR响应延迟** | **< 200ms** | **检测到显示Matrix猫头** |

### 7.2 产品指标

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| 配对成功率 | > 95% | 成功配对 / 尝试次数 |
| 一键监控成功率 | > 90% | 成功启动 / 尝试次数 |
| **AR猫头启动率** | **> 95%** | **成功显示猫头 / 启动App** |
| **AR交互成功率** | **> 85%** | **语音/手势命令成功执行** |
| 用户满意度 | > 4.0/5 | 内测用户评分 |
| **"屏幕有猫头"满意度** | **> 4.5/5** | **AR视觉体验评分** |

---

## 8. 附录

### 8.1 参考代码

**OpenClaw参考实现**:
- `libs/openclaw/apps/ios/Sources/Camera/CameraController.swift` - Camera实现
- `libs/openclaw/apps/shared/OpenClawKit/Sources/OpenClawKit/GatewayNodeSession.swift` - Node协议
- `libs/openclaw/src/agents/tools/nodes-tool.ts` - Gateway端工具

### 8.2 接口文档

- [OpenClaw iOS Node指南](libs/openclaw/docs/platforms/ios.md)
- [OpenClaw Pairing文档](libs/openclaw/docs/gateway/pairing.md)
- [OpenClaw Camera文档](libs/openclaw/docs/nodes/camera.md)

### 8.3 变更日志

**v1.0 → v1.1**:
- 技术架构从"复用OpenClaw iOS"改为"独立Node客户端"
- 明确不复用OpenClaw源代码，只使用协议
- 重新分配角色工作，明确接口契约
- 细化Week 1-5的具体任务和依赖
- 添加风险评估和应急预案

**v1.1 → v1.2** (重大架构调整):
- **核心理念转变**: AR不是附加功能，而是**核心UI层**（"The EYE is the Interface"）
- Matrix猫头 = 系统主界面，用户通过AR进行所有交互
- David（AR工程师）**Week 2进场**（前期Alex预留架构）
- 新增**IC0架构契约**: Alex Week 1定义ARInterface协议，阻塞David进场
- 重新设计架构图: AR层作为顶层UI，后台监控作为数据层
- 里程碑调整: M3 MVP必须包含基础AR功能（屏幕上能看到猫头）
- 团队入场时间明确: Alex（Week 1）→ David（Week 2）→ 全员（Week 3+）
- 强调: 没有AR猫头 = 产品失去灵魂（乔布斯式产品定义）

---

## 9. 审批记录

| 版本 | 日期 | 作者 | 审批人 | 状态 |
|------|------|------|--------|------|
| 1.0 | 2026-02-27 | Claude | - | Superseded |
| 1.1 | 2026-02-27 | Claude | - | Superseded |
| 1.2 | 2026-02-27 | Claude | - | **Draft (待Review)** |


---

## 10. 下一步行动

1. **Review会议** (建议本周内)
   - Alex, Bob, Carol 一起Review本计划
   - 确认技术方案可行性
   - 确认时间线是否合理

2. **环境准备** (Week 0)
   - Alex: 准备iOS开发环境，测试旧iPhone
   - Bob: 安装配置OpenClaw Gateway
   - Carol: 准备项目管理工具 (GitHub Projects?)

3. **Week 1 Kickoff**
   - 详细拆解Day 1-5任务
   - 确认IC1 (Node协议) 输出格式
   - 每日站会启动

**准备好了就开始执行！**
