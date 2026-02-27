# Phase 0 Roadmap (Month 0-6)

## Goals
- 0 external funding; personal time only
- Complete "sense → report → confirm" loop (Mac/iPhone)
- Build trust and technical reputation via open source

## Deliverables
- [ ] OpenClaw Eye iOS App (MIT License)
- [ ] OpenClaw Hub macOS app (open source)
- [ ] 3D-printed stand STL (open source, Qi charging slot)

## Commercial Model (No Revenue)
- Software: fully open source
- Hardware: stand design open; users print themselves
- Revenue: none

## Exit Criteria (Directional)
- [ ] Community attention grows clearly
- [ ] Users ask to buy a ready-made stand

## Roadmap

### Month 0-1: Foundation
- [ ] Create GitHub org and repos (Eye iOS, Hub macOS)
- [ ] Define minimal protocol for "event → confirm" flow
- [ ] Set up basic CI and README scaffolding
- [ ] Expose LLM config UI (model/provider/base URL/API key)
- [ ] Ensure local/LAN Ollama endpoint support from day one

### Month 1-2: MVP Loop
- [ ] iOS Eye: camera capture + local motion detection
- [ ] macOS Hub: WebSocket server + event viewer
- [ ] End-to-end flow: Eye detects → Hub receives → user confirms
- [ ] LLM config persistence + validation in app settings
- [ ] Ollama local/LAN smoke test (self-hosted model)

### Month 2-3: UX + Reliability
- [ ] Add status UI (FPS, latency, network state)
- [ ] Add reconnect/heartbeat for WebSocket
- [ ] Improve false-positive handling

### Month 3-4: 3D Stand V1
- [ ] Design V1 stand (15° angle, Qi slot, cable routing)
- [ ] Publish STL + simple assembly guide

### Month 4-6: Community Validation
- [ ] Release alpha builds + demo video
- [ ] Collect feedback (issues/discussions)
- [ ] Iterate stand design based on feedback
