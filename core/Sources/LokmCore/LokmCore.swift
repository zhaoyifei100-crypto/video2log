import Foundation

/// LokmCore - 共享基础类型

public enum LokmEvent: Sendable {
    case motionDetected(region: CGRect, confidence: Double)
    case personDetected(bbox: CGRect, confidence: Double)
    case unknownObject(bbox: CGRect)
}

public struct LokmEventPayload: Codable {
    public let timestamp: Date
    public let eventType: String
    public let boundingBox: CGRect
    public let confidence: Double
    public let imageData: Data?  // 可选缩略图
    
    public init(event: LokmEvent, imageData: Data? = nil) {
        self.timestamp = Date()
        self.imageData = imageData
        
        switch event {
        case .motionDetected(let region, let conf):
            self.eventType = "motion"
            self.boundingBox = region
            self.confidence = conf
        case .personDetected(let bbox, let conf):
            self.eventType = "person"
            self.boundingBox = bbox
            self.confidence = conf
        case .unknownObject(let bbox):
            self.eventType = "unknown"
            self.boundingBox = bbox
            self.confidence = 0.0
        }
    }
}

public enum LokmError: Error {
    case cameraNotAuthorized
    case networkDisconnected
    case invalidConfiguration
}

public actor LokmLogger {
    public static let shared = LokmLogger()
    
    public func log(_ message: String, level: LogLevel = .info) {
        let prefix = "[Lokm]"
        print("\(prefix) [\(level)] \(message)")
    }
    
    public enum LogLevel: String {
        case debug = "DEBUG"
        case info = "INFO"
        case warning = "WARN"
        case error = "ERROR"
    }
}
