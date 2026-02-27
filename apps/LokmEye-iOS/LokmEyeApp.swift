import SwiftUI
import LokmCamera
import LokmVision
import LokmCore

@main
struct LokmEyeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = EyeViewModel()
    
    var body: some View {
        NavigationView {
            VStack {
                CameraPreviewView(session: viewModel.cameraService.captureSession)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black)
                
                VStack(spacing: 12) {
                    StatusBar(
                        isRunning: viewModel.isRunning,
                        hubConnected: viewModel.hubConnected,
                        fps: viewModel.fps
                    )
                    
                    HStack(spacing: 20) {
                        Button(action: { viewModel.toggleMonitoring() }) {
                            Label(
                                viewModel.isRunning ? "Stop" : "Start",
                                systemImage: viewModel.isRunning ? "stop.fill" : "play.fill"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(viewModel.isRunning ? .red : .green)
                        
                        Button(action: { viewModel.showSettings = true }) {
                            Label("Settings", systemImage: "gear")
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.horizontal)
                }
                .padding()
                .background(.ultraThinMaterial)
            }
            .navigationTitle("LokmEye 👁️")
        }
        .sheet(isPresented: $viewModel.showSettings) {
            SettingsView()
        }
    }
}

@MainActor
class EyeViewModel: ObservableObject {
    @Published var isRunning = false
    @Published var hubConnected = false
    @Published var fps: Double = 0
    @Published var showSettings = false
    
    let cameraService = LokmCameraService()
    let visionService = LokmVisionService()
    
    init() {
        setupVisionCallback()
    }
    
    private func setupVisionCallback() {
        Task {
            await visionService.onEventDetected = { [weak self] event in
                await self?.handleEvent(event)
            }
            
            await cameraService.onFrame = { [weak self] sampleBuffer in
                await self?.visionService.processFrame(sampleBuffer)
            }
        }
    }
    
    func toggleMonitoring() {
        Task {
            if isRunning {
                await cameraService.stopCapture()
                isRunning = false
            } else {
                try? await cameraService.startCapture()
                isRunning = true
            }
        }
    }
    
    private func handleEvent(_ event: LokmEvent) async {
        // TODO: Send to Hub via WebSocket
        print("Event detected: \(event)")
    }
}

struct StatusBar: View {
    let isRunning: Bool
    let hubConnected: Bool
    let fps: Double
    
    var body: some View {
        HStack {
            HStack(spacing: 4) {
                Circle()
                    .fill(isRunning ? .green : .gray)
                    .frame(width: 8, height: 8)
                Text(isRunning ? "Running" : "Idle")
                    .font(.caption)
            }
            
            Spacer()
            
            Text(String(format: "%.1f FPS", fps))
                .font(.caption.monospacedDigit())
            
            Spacer()
            
            HStack(spacing: 4) {
                Image(systemName: hubConnected ? "wifi" : "wifi.slash")
                    .foregroundColor(hubConnected ? .green : .red)
                Text(hubConnected ? "Hub" : "No Hub")
                    .font(.caption)
            }
        }
    }
}

struct CameraPreviewView: UIViewControllerRepresentable {
    var session: AVCaptureSession?
    
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Preview layer setup
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section("Hub Connection") {
                    TextField("Hub IP Address", text: .constant(""))
                    TextField("Port", text: .constant("8080"))
                }
                
                Section("Detection") {
                    Toggle("Person Detection", isOn: .constant(true))
                    Toggle("Motion Detection", isOn: .constant(false))
                    Slider(value: .constant(0.8), in: 0...1) {
                        Text("Confidence Threshold")
                    }
                }
                
                Section("About") {
                    Text("LokmEye v0.1.0")
                    Text("Look, My Eye! 👁️")
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}
