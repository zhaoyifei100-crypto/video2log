import Foundation
import LokmCore

public actor LokmLLMService {
    private var baseURL: URL
    private var modelName: String
    private var session: URLSession
    
    public init(baseURL: String = "http://localhost:11434", 
                modelName: String = "llava:7b") {
        self.baseURL = URL(string: baseURL)!
        self.modelName = modelName
        self.session = URLSession.shared
    }
    
    public func configure(baseURL: String, modelName: String) {
        self.baseURL = URL(string: baseURL)!
        self.modelName = modelName
    }
    
    public func describeImage(_ imageData: Data) async throws -> String {
        let url = baseURL.appendingPathComponent("api/generate")
        
        let requestBody: [String: Any] = [
            "model": modelName,
            "prompt": "What do you see in this image? Describe briefly.",
            "images": [imageData.base64EncodedString()],
            "stream": false
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        let (data, _) = try await session.data(for: request)
        
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let response = json["response"] as? String else {
            throw LokmError.invalidConfiguration
        }
        
        return response
    }
    
    public func healthCheck() async -> Bool {
        do {
            let url = baseURL.appendingPathComponent("api/tags")
            let (_, response) = try await session.data(from: url)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }
}
