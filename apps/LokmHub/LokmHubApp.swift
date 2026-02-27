import SwiftUI

@main
struct LokmHubApp: App {
    var body: some Scene {
        WindowGroup {
            HubContentView()
        }
        .windowResizability(.contentSize)
    }
}

struct HubContentView: View {
    @StateObject private var viewModel = HubViewModel()
    
    var body: some View {
        NavigationSplitView {
            SidebarView(devices: viewModel.connectedDevices)
        } detail: {
            EventLogView(events: viewModel.events)
        }
        .navigationTitle("LokmHub 👁️")
        .frame(minWidth: 800, minHeight: 600)
    }
}

@MainActor
class HubViewModel: ObservableObject {
    @Published var connectedDevices: [EyeDevice] = []
    @Published var events: [EyeEvent] = []
    
    init() {
        startServer()
    }
    
    private func startServer() {
        // TODO: Start WebSocket server
        // TODO: Start Bonjour service
    }
}

struct EyeDevice: Identifiable {
    let id = UUID()
    let name: String
    let ipAddress: String
    let lastSeen: Date
    let isOnline: Bool
}

struct EyeEvent: Identifiable {
    let id = UUID()
    let deviceName: String
    let timestamp: Date
    let type: String
    let confidence: Double
    let thumbnailData: Data?
}

struct SidebarView: View {
    let devices: [EyeDevice]
    
    var body: some View {
        List {
            Section("Connected Eyes") {
                ForEach(devices) { device in
                    DeviceRow(device: device)
                }
            }
            
            if devices.isEmpty {
                Text("No devices connected")
                    .foregroundColor(.secondary)
                    .italic()
            }
            
            Section("Actions") {
                Button(action: {}) {
                    Label("Add Device Manually", systemImage: "plus")
                }
                
                Button(action: {}) {
                    Label("Scan Network", systemImage: "magnifyingglass")
                }
            }
        }
        .listStyle(.sidebar)
    }
}

struct DeviceRow: View {
    let device: EyeDevice
    
    var body: some View {
        HStack {
            Image(systemName: "eye.fill")
                .foregroundColor(device.isOnline ? .green : .gray)
            
            VStack(alignment: .leading) {
                Text(device.name)
                    .font(.headline)
                Text(device.ipAddress)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Circle()
                .fill(device.isOnline ? .green : .red)
                .frame(width: 8, height: 8)
        }
    }
}

struct EventLogView: View {
    let events: [EyeEvent]
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Event Log")
                .font(.headline)
                .padding()
            
            if events.isEmpty {
                VStack {
                    Spacer()
                    Image(systemName: "eye.slash")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("No events yet")
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(events) { event in
                    EventRow(event: event)
                }
            }
        }
    }
}

struct EventRow: View {
    let event: EyeEvent
    
    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.gray.opacity(0.3))
                .frame(width: 60, height: 45)
                .overlay(
                    Image(systemName: "photo")
                        .foregroundColor(.secondary)
                )
            
            VStack(alignment: .leading, spacing: 4) {
                Text(event.type.capitalized)
                    .font(.headline)
                Text("From: \(event.deviceName)")
                    .font(.caption)
                Text(event.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(String(format: "%.0f%%", event.confidence * 100))
                .font(.caption.monospacedDigit())
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.blue.opacity(0.2))
                .cornerRadius(4)
        }
        .padding(.vertical, 4)
    }
}
