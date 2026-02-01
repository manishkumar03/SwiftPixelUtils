import XCTest
@testable import SwiftPixelUtils

/// Tests for ``BoundingBox`` - Bounding box conversion and manipulation utilities.
///
/// ## Topics
///
/// ### Format Conversion Tests
/// - Converts between XYXY, XYWH, and CXCYWH formats
/// - Validates round-trip conversions preserve coordinates
///
/// ### IoU (Intersection over Union) Tests
/// - Tests perfect overlap (IoU = 1.0), no overlap (IoU = 0.0)
/// - Partial overlap and contained box scenarios
///
/// ### Non-Maximum Suppression Tests
/// - Standard NMS with various IoU thresholds
/// - Score threshold filtering
/// - Empty input handling
///
/// ### Box Scaling & Clipping Tests
/// - Scales boxes between different image dimensions
/// - Clips boxes to image boundaries
final class BoundingBoxTests: XCTestCase {
    
    // MARK: - Format Conversion Tests
    
    func testXYXYToXYWH() {
        let xyxy: [[Float]] = [[100, 100, 200, 250]]
        let xywh = BoundingBox.convertFormat(xyxy, from: .xyxy, to: .xywh)
        
        XCTAssertEqual(xywh[0][0], 100, accuracy: 0.01)  // x
        XCTAssertEqual(xywh[0][1], 100, accuracy: 0.01)  // y
        XCTAssertEqual(xywh[0][2], 100, accuracy: 0.01)  // w
        XCTAssertEqual(xywh[0][3], 150, accuracy: 0.01)  // h
    }
    
    func testXYXYToCXCYWH() {
        let xyxy: [[Float]] = [[100, 100, 200, 200]]
        let cxcywh = BoundingBox.convertFormat(xyxy, from: .xyxy, to: .cxcywh)
        
        XCTAssertEqual(cxcywh[0][0], 150, accuracy: 0.01)  // cx
        XCTAssertEqual(cxcywh[0][1], 150, accuracy: 0.01)  // cy
        XCTAssertEqual(cxcywh[0][2], 100, accuracy: 0.01)  // w
        XCTAssertEqual(cxcywh[0][3], 100, accuracy: 0.01)  // h
    }
    
    func testCXCYWHToXYXY() {
        let cxcywh: [[Float]] = [[320, 240, 100, 80]]
        let xyxy = BoundingBox.convertFormat(cxcywh, from: .cxcywh, to: .xyxy)
        
        XCTAssertEqual(xyxy[0][0], 270, accuracy: 0.1)  // x1
        XCTAssertEqual(xyxy[0][1], 200, accuracy: 0.1)  // y1
        XCTAssertEqual(xyxy[0][2], 370, accuracy: 0.1)  // x2
        XCTAssertEqual(xyxy[0][3], 280, accuracy: 0.1)  // y2
    }
    
    func testXYWHToXYXY() {
        let xywh: [[Float]] = [[50, 50, 100, 150]]
        let xyxy = BoundingBox.convertFormat(xywh, from: .xywh, to: .xyxy)
        
        XCTAssertEqual(xyxy[0][0], 50, accuracy: 0.01)   // x1
        XCTAssertEqual(xyxy[0][1], 50, accuracy: 0.01)   // y1
        XCTAssertEqual(xyxy[0][2], 150, accuracy: 0.01)  // x2
        XCTAssertEqual(xyxy[0][3], 200, accuracy: 0.01)  // y2
    }
    
    func testXYWHToCXCYWH() {
        let xywh: [[Float]] = [[100, 100, 200, 100]]
        let cxcywh = BoundingBox.convertFormat(xywh, from: .xywh, to: .cxcywh)
        
        XCTAssertEqual(cxcywh[0][0], 200, accuracy: 0.01)  // cx
        XCTAssertEqual(cxcywh[0][1], 150, accuracy: 0.01)  // cy
        XCTAssertEqual(cxcywh[0][2], 200, accuracy: 0.01)  // w
        XCTAssertEqual(cxcywh[0][3], 100, accuracy: 0.01)  // h
    }
    
    func testCXCYWHToXYWH() {
        let cxcywh: [[Float]] = [[150, 100, 100, 50]]
        let xywh = BoundingBox.convertFormat(cxcywh, from: .cxcywh, to: .xywh)
        
        XCTAssertEqual(xywh[0][0], 100, accuracy: 0.01)  // x
        XCTAssertEqual(xywh[0][1], 75, accuracy: 0.01)   // y
        XCTAssertEqual(xywh[0][2], 100, accuracy: 0.01)  // w
        XCTAssertEqual(xywh[0][3], 50, accuracy: 0.01)   // h
    }
    
    func testSameFormatConversion() {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let result = BoundingBox.convertFormat(boxes, from: .xyxy, to: .xyxy)
        XCTAssertEqual(boxes, result)
    }
    
    func testMultipleBoxesConversion() {
        let boxes: [[Float]] = [
            [0, 0, 100, 100],
            [200, 200, 300, 300],
            [50, 50, 150, 150]
        ]
        let converted = BoundingBox.convertFormat(boxes, from: .xyxy, to: .cxcywh)
        XCTAssertEqual(converted.count, 3)
        
        // First box: center should be at (50, 50)
        XCTAssertEqual(converted[0][0], 50, accuracy: 0.01)
        XCTAssertEqual(converted[0][1], 50, accuracy: 0.01)
    }
    
    func testRoundTripConversion() {
        let original: [[Float]] = [[100, 100, 200, 200]]
        
        // XYXY -> CXCYWH -> XYXY
        let cxcywh = BoundingBox.convertFormat(original, from: .xyxy, to: .cxcywh)
        let backToXYXY = BoundingBox.convertFormat(cxcywh, from: .cxcywh, to: .xyxy)
        
        XCTAssertEqual(original[0][0], backToXYXY[0][0], accuracy: 0.01)
        XCTAssertEqual(original[0][1], backToXYXY[0][1], accuracy: 0.01)
        XCTAssertEqual(original[0][2], backToXYXY[0][2], accuracy: 0.01)
        XCTAssertEqual(original[0][3], backToXYXY[0][3], accuracy: 0.01)
    }
    
    // MARK: - Scaling Tests
    
    func testBoxScalingUpXYXY() {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 640, height: 640),
            to: CGSize(width: 1920, height: 1080),
            format: .xyxy
        )
        
        XCTAssertEqual(scaled[0][0], 300, accuracy: 0.1)     // x1 * 3
        XCTAssertEqual(scaled[0][1], 168.75, accuracy: 0.1)  // y1 * 1.6875
        XCTAssertEqual(scaled[0][2], 600, accuracy: 0.1)     // x2 * 3
        XCTAssertEqual(scaled[0][3], 337.5, accuracy: 0.1)   // y2 * 1.6875
    }
    
    func testBoxScalingDownXYXY() {
        let boxes: [[Float]] = [[300, 300, 600, 600]]
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 1000, height: 1000),
            to: CGSize(width: 500, height: 500),
            format: .xyxy
        )
        
        XCTAssertEqual(scaled[0][0], 150, accuracy: 0.1)
        XCTAssertEqual(scaled[0][1], 150, accuracy: 0.1)
        XCTAssertEqual(scaled[0][2], 300, accuracy: 0.1)
        XCTAssertEqual(scaled[0][3], 300, accuracy: 0.1)
    }
    
    func testBoxScalingXYWH() {
        let boxes: [[Float]] = [[100, 100, 100, 100]]  // x, y, w, h
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 500, height: 500),
            to: CGSize(width: 1000, height: 1000),
            format: .xywh
        )
        
        XCTAssertEqual(scaled[0][0], 200, accuracy: 0.1)  // x * 2
        XCTAssertEqual(scaled[0][1], 200, accuracy: 0.1)  // y * 2
        XCTAssertEqual(scaled[0][2], 200, accuracy: 0.1)  // w * 2
        XCTAssertEqual(scaled[0][3], 200, accuracy: 0.1)  // h * 2
    }
    
    func testBoxScalingCXCYWH() {
        let boxes: [[Float]] = [[250, 250, 100, 100]]  // cx, cy, w, h
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 500, height: 500),
            to: CGSize(width: 1000, height: 1000),
            format: .cxcywh
        )
        
        XCTAssertEqual(scaled[0][0], 500, accuracy: 0.1)  // cx * 2
        XCTAssertEqual(scaled[0][1], 500, accuracy: 0.1)  // cy * 2
        XCTAssertEqual(scaled[0][2], 200, accuracy: 0.1)  // w * 2
        XCTAssertEqual(scaled[0][3], 200, accuracy: 0.1)  // h * 2
    }
    
    func testBoxScalingSameSize() {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 640, height: 640),
            to: CGSize(width: 640, height: 640),
            format: .xyxy
        )
        
        XCTAssertEqual(boxes[0], scaled[0])
    }
    
    // MARK: - Clipping Tests
    
    func testClipBoxOutOfBoundsXYXY() {
        let boxes: [[Float]] = [[-10, 50, 700, 500]]
        let clipped = BoundingBox.clip(
            boxes,
            imageSize: CGSize(width: 640, height: 480),
            format: .xyxy
        )
        
        XCTAssertEqual(clipped[0][0], 0)
        XCTAssertEqual(clipped[0][1], 50)
        XCTAssertEqual(clipped[0][2], 640)
        XCTAssertEqual(clipped[0][3], 480)
    }
    
    func testClipBoxAllNegative() {
        let boxes: [[Float]] = [[-100, -100, -50, -50]]
        let clipped = BoundingBox.clip(
            boxes,
            imageSize: CGSize(width: 640, height: 480),
            format: .xyxy
        )
        
        XCTAssertEqual(clipped[0][0], 0)
        XCTAssertEqual(clipped[0][1], 0)
        XCTAssertEqual(clipped[0][2], 0)
        XCTAssertEqual(clipped[0][3], 0)
    }
    
    func testClipBoxWithinBounds() {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let clipped = BoundingBox.clip(
            boxes,
            imageSize: CGSize(width: 640, height: 480),
            format: .xyxy
        )
        
        XCTAssertEqual(clipped[0], boxes[0])
    }
    
    func testClipMultipleBoxes() {
        let boxes: [[Float]] = [
            [-10, -10, 50, 50],
            [600, 400, 700, 500],
            [100, 100, 200, 200]
        ]
        let clipped = BoundingBox.clip(
            boxes,
            imageSize: CGSize(width: 640, height: 480),
            format: .xyxy
        )
        
        XCTAssertEqual(clipped.count, 3)
        XCTAssertEqual(clipped[0][0], 0)
        XCTAssertEqual(clipped[1][2], 640)
        XCTAssertEqual(clipped[2], boxes[2])  // Within bounds, unchanged
    }
    
    // MARK: - IoU Calculation Tests
    
    func testIoUPerfectOverlap() {
        let box1: [Float] = [100, 100, 200, 200]
        let box2: [Float] = [100, 100, 200, 200]
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        XCTAssertEqual(iou, 1.0, accuracy: 0.001)
    }
    
    func testIoUNoOverlap() {
        let box1: [Float] = [0, 0, 100, 100]
        let box2: [Float] = [200, 200, 300, 300]
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        XCTAssertEqual(iou, 0.0, accuracy: 0.001)
    }
    
    func testIoUPartialOverlap() {
        let box1: [Float] = [100, 100, 200, 200]
        let box2: [Float] = [150, 150, 250, 250]
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        
        // Intersection: 50x50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU = 2500 / 17500 ≈ 0.1428
        XCTAssertEqual(iou, 0.1428, accuracy: 0.01)
    }
    
    func testIoUHalfOverlap() {
        let box1: [Float] = [0, 0, 100, 100]
        let box2: [Float] = [50, 0, 150, 100]
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        
        // Intersection: 50x100 = 5000
        // Union: 10000 + 10000 - 5000 = 15000
        // IoU = 5000 / 15000 ≈ 0.333
        XCTAssertEqual(iou, 0.333, accuracy: 0.01)
    }
    
    func testIoUTouchingBoxes() {
        let box1: [Float] = [0, 0, 100, 100]
        let box2: [Float] = [100, 0, 200, 100]
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        XCTAssertEqual(iou, 0.0, accuracy: 0.001)
    }
    
    func testIoUContainedBox() {
        let box1: [Float] = [0, 0, 200, 200]  // Large box
        let box2: [Float] = [50, 50, 150, 150]  // Contained box
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        
        // Intersection: 10000 (entire small box)
        // Union: 40000 (large box area)
        // IoU = 10000 / 40000 = 0.25
        XCTAssertEqual(iou, 0.25, accuracy: 0.01)
    }
    
    // MARK: - NMS Tests
    
    func testNMSBasic() {
        let detections = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0),
            Detection(box: [300, 300, 400, 400], score: 0.7, classIndex: 1)
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        // First two boxes overlap significantly, should keep only the higher score
        // Third box is separate class/location
        XCTAssertEqual(filtered.count, 2)
        XCTAssertEqual(filtered[0].score, 0.9)
        XCTAssertEqual(filtered[1].score, 0.7)
    }
    
    func testNMSAllSuppressed() {
        let detections = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [100, 100, 200, 200], score: 0.8, classIndex: 0),
            Detection(box: [100, 100, 200, 200], score: 0.7, classIndex: 0)
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].score, 0.9)
    }
    
    func testNMSNoneSuppressed() {
        let detections = [
            Detection(box: [0, 0, 100, 100], score: 0.9, classIndex: 0),
            Detection(box: [200, 200, 300, 300], score: 0.8, classIndex: 0),
            Detection(box: [400, 400, 500, 500], score: 0.7, classIndex: 0)
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        XCTAssertEqual(filtered.count, 3)
    }
    
    func testNMSScoreThreshold() {
        let detections = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [300, 300, 400, 400], score: 0.2, classIndex: 0)
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.5
        )
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].score, 0.9)
    }
    
    func testNMSHighIoUThreshold() {
        let detections = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0)
        ]
        
        // High IoU threshold means more boxes are kept
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.9,
            scoreThreshold: 0.3
        )
        
        XCTAssertEqual(filtered.count, 2)
    }
    
    func testNMSEmptyInput() {
        let detections: [Detection] = []
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        XCTAssertEqual(filtered.count, 0)
    }
    
    func testNMSSingleDetection() {
        let detections = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0)
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        XCTAssertEqual(filtered.count, 1)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyBoxArray() {
        let boxes: [[Float]] = []
        let converted = BoundingBox.convertFormat(boxes, from: .xyxy, to: .cxcywh)
        XCTAssertEqual(converted.count, 0)
    }
    
    func testZeroSizeBox() {
        let boxes: [[Float]] = [[100, 100, 100, 100]]  // Zero size in XYXY
        let cxcywh = BoundingBox.convertFormat(boxes, from: .xyxy, to: .cxcywh)
        
        XCTAssertEqual(cxcywh[0][2], 0)  // width
        XCTAssertEqual(cxcywh[0][3], 0)  // height
    }
    
    func testNegativeCoordinates() {
        // Some models can output negative coordinates before clipping
        let boxes: [[Float]] = [[-50, -50, 50, 50]]
        let cxcywh = BoundingBox.convertFormat(boxes, from: .xyxy, to: .cxcywh)
        
        XCTAssertEqual(cxcywh[0][0], 0, accuracy: 0.01)   // cx
        XCTAssertEqual(cxcywh[0][1], 0, accuracy: 0.01)   // cy
        XCTAssertEqual(cxcywh[0][2], 100, accuracy: 0.01) // w
        XCTAssertEqual(cxcywh[0][3], 100, accuracy: 0.01) // h
    }
    
    // MARK: - Performance Tests
    
    func testBoundingBoxConversionPerformance() {
        var boxes: [[Float]] = []
        for i in 0..<1000 {
            let fi = Float(i)
            boxes.append([fi, fi, fi + 100, fi + 100])
        }
        
        measure {
            let _ = BoundingBox.convertFormat(boxes, from: .xyxy, to: .cxcywh)
        }
    }
    
    func testNMSPerformance() {
        var detections: [Detection] = []
        for i in 0..<500 {
            let x = Float(i % 20) * 50
            let y = Float(i / 20) * 50
            detections.append(Detection(
                box: [x, y, x + 40, y + 40],
                score: Float.random(in: 0.5...1.0),
                classIndex: i % 10
            ))
        }
        
        measure {
            let _ = BoundingBox.nonMaxSuppression(
                detections: detections,
                iouThreshold: 0.5,
                scoreThreshold: 0.3
            )
        }
    }
}
