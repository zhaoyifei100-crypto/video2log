# Phase 0 Roadmap (Month 0-6)

## Goals
- 0 external funding; personal time only
- Complete "sense → report → confirm" loop (Mac/iPhone)
- Build trust and technical reputation via open source

## Deliverables
- [1.1] OpenClaw Eye macOS Native Demo (Swift/SwiftUI + Vision Framework)
- [1.2] OpenClaw Hub macOS app (Swift/SwiftUI Open Source)

- [ ] OpenClaw Eye iOS App (Swift codebase shared with macOS Demo)
- [ ] 3D-printed stand STL (open source, Qi charging slot)

## Commercial Model (No Revenue)
- Software: fully open source
- Hardware: stand design open; users print themselves
- Revenue: none

## Exit Criteria (Directional)
- [ ] Community attention grows clearly
- [ ] Users ask to buy a ready-made stand

## Roadmap

### Month 0-1: Foundation (Swift First)
- [ ] Create GitHub org and repos (Eye & Hub monorepo or separate)
- [ ] **Core Logic (Swift Package):**
    - [ ] `CameraService`: AVFoundation wrapper (macOS/iOS compatible)
    - [ ] `VisionService`: Apple Vision Framework for motion/person detection
    - [ ] `LLMService`: Swift actor for Ollama API calls
- [ ] OpenClaw Eye macOS Demo (CLI or simple Menu Bar App) to test above logic

### Month 1-2: MVP Loop (Cross-Platform)
- [ ] **Port to iOS**: Reuse Core Logic package, build iOS UI
- [ ] **macOS Hub**: WebSocket server (using NWListener or SwiftNI)
- [ ] End-to-end flow: Eye (iOS/Mac) → Hub (Mac) → user confirms
- [ ] Settings UI: Persistence (UserDefaults/AppStorage) for LLM configs

### Month 2-3: UX + Reliability
- [ ] Add status UI (FPS, latency, network state)
- [ ] Implement "Heartbeat" & Auto-reconnect (Swift Concurrency)
- [ ] Improve false-positive handling (Vision confidence thresholds)

### Month 3-4: 3D Stand V1
- [ ] Design V1 stand (15° angle, Qi slot, cable routing)
- [ ] Publish STL + simple assembly guide

### Month 4-6: Community Validation
- [ ] Release alpha builds + demo video
- [ ] Collect feedback (issues/discussions)
- [ ] Iterate stand design based on feedback