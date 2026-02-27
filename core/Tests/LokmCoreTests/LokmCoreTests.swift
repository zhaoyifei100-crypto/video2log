import XCTest
@testable import LokmCore

final class LokmCoreTests: XCTestCase {
    func testEventCreation() {
        let event = LokmEvent.personDetected(bbox: CGRect(x: 0.1, y: 0.1, width: 0.3, height: 0.4), 
                                             confidence: 0.95)
        XCTAssertNotNil(event)
    }
    
    func testEventPayload() {
        let event = LokmEvent.motionDetected(region: CGRect(x: 0, y: 0, width: 1, height: 1), 
                                             confidence: 0.8)
        let payload = LokmEventPayload(event: event)
        
        XCTAssertEqual(payload.eventType, "motion")
        XCTAssertEqual(payload.confidence, 0.8, accuracy: 0.01)
    }
}
