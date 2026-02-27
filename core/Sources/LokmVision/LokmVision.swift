import Foundation
import Vision
import CoreML
import LokmCore

@available(iOS 16.0, macOS 13.0, *)
public actor LokmVisionService {
    
    private var request: VNGenerateObjectnessBasedSaliencyImageRequest?
    private var personRequest: VNDetectHumanRectanglesRequest?
    
    public var onEventDetected: ((LokmEvent) async -> Void)?
    
    public init() {
        setupRequests()
    }
    
    private func setupRequests() {
        // 运动显著性检测 (轻量级)
        let saliencyRequest = VNGenerateObjectnessBasedSaliencyImageRequest { [weak self] request, error in
            guard let self = self else { return }
            Task {
                await self.handleSaliencyResult(request: request, error: error)
            }
        }
        self.request = saliencyRequest
        
        // 人形检测
        let personReq = VNDetectHumanRectanglesRequest { [weak self] request, error in
            guard let self = self else { return }
            Task {
                await self.handlePersonResult(request: request, error: error)
            }
        }
        personReq.upperBodyOnly = false
        self.personRequest = personReq
    }
    
    public func processFrame(_ sampleBuffer: CMSampleBuffer) async {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            if let personReq = personRequest {
                try handler.perform([personReq])
            }
            // 显著性检测暂不使用，优先人形检测
            // if let saliencyReq = request {
            //     try handler.perform([saliencyReq])
            // }
        } catch {
            await LokmLogger.shared.log("Vision processing error: \(error)", level: .error)
        }
    }
    
    private func handlePersonResult(request: VNRequest, error: Error?) async {
        guard let results = request.results as? [VNHumanObservation], !results.isEmpty else { return }
        
        for observation in results {
            let event = LokmEvent.personDetected(
                bbox: observation.boundingBox,
                confidence: Double(observation.confidence)
            )
            await onEventDetected?(event)
        }
    }
    
    private func handleSaliencyResult(request: VNRequest, error: Error?) async {
        // 显著区域检测，可用于运动检测
        // 暂时简化实现
    }
}
