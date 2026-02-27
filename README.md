# LokmEye - LookMyEye(TellEeWhy)

让旧iPhone变身智能监控眼的0成本方案

## 架构

```
lokmeye/
├── apps/
│   ├── LokmEye-iOS/          # iOS端 (旧手机作为Eye)
│   ├── LokmEye-macOS/        # macOS端 (Mac作为Eye)
│   └── LokmHub/              # macOS中控端
├── core/                     # Swift Package共享核心
│   ├── LokmCore/             # 基础类型和工具
│   ├── LokmCamera/           # AVFoundation封装
│   ├── LokmVision/           # Apple Vision + Core ML
│   └── LokmLLM/              # Ollama/LLM客户端
├── hardware/
│   └── cad/                  # 3D打印支架模型
├── docs/                     # 文档
└── legacy/                   # 原有Python代码(video2log)

```

## Phase 0 目标 (Month 0-1)

- [ ] LokmEye iOS App - 检测运动/人并上报
- [ ] LokmHub macOS App - 接收事件并展示
- [ ] 3D打印支架设计
- [ ] 端到端闭环: 检测 → 上报 → 确认

## 技术栈

- **Camera**: AVFoundation (原生)
- **Vision**: Apple Vision Framework + Core ML
- **UI**: SwiftUI (跨平台)
- **Network**: WebSocket + Bonjour
- **LLM**: Ollama API (可选)

## 开源协议

MIT License - 完全开源，自由使用

## 项目背景

这个项目源于一个硬件创业想法：为边缘视觉设计超低功耗AI芯片。
但芯片开发极其复杂，所以先用软件验证市场需求和用户体验。
如果软件获得关注，再考虑硬件产品化。

详见 [docs/PATENT_DISCLOSURE.md](docs/PATENT_DISCLOSURE.md)
