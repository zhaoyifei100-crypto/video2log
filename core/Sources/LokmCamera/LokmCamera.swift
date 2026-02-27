import Foundation
import AVFoundation
import LokmCore

@available(iOS 16.0, macOS 13.0, *)
public actor LokmCameraService {
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var device: AVCaptureDevice?
    
    public var onFrame: ((CMSampleBuffer) async -> Void)?
    
    public init() {}
    
    public func checkAuthorization() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .notDetermined {
            return await AVCaptureDevice.requestAccess(for: .video)
        }
        return status == .authorized
    }
    
    public func startCapture(preferredDevice: AVCaptureDevice? = nil) async throws {
        guard await checkAuthorization() else {
            throw LokmError.cameraNotAuthorized
        }
        
        let session = AVCaptureSession()
        session.sessionPreset = .medium  // 640x480
        
        let device = preferredDevice ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        guard let device = device else {
            throw LokmError.invalidConfiguration
        }
        self.device = device
        
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "lokm.camera"))
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        
        self.videoOutput = output
        self.captureSession = session
        
        session.startRunning()
        await LokmLogger.shared.log("Camera started", level: .info)
    }
    
    public func stopCapture() async {
        captureSession?.stopRunning()
        captureSession = nil
        videoOutput = nil
        await LokmLogger.shared.log("Camera stopped", level: .info)
    }
}

@available(iOS 16.0, macOS 13.0, *)
extension LokmCameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated public func captureOutput(_ output: AVCaptureOutput, 
                                          didOutput sampleBuffer: CMSampleBuffer, 
                                          from connection: AVCaptureConnection) {
        Task {
            await onFrame?(sampleBuffer)
        }
    }
}
