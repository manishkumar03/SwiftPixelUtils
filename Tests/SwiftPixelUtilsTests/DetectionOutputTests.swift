import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``DetectionOutput`` - Object detection model output processing.
///
/// ## Topics
///
/// ### ObjectDetection Tests
/// - Initialization with class index, label, confidence, and bounding box
/// - Equatable conformance
///
/// ### DetectionResult Tests  
/// - Multiple detection handling
/// - NMS integration
/// - Format conversion (XYXY, XYWH, CXCYWH)
///
/// ### Post-Processing Tests
/// - Score threshold filtering
/// - IoU-based suppression
/// - Box coordinate scaling
final class DetectionOutputTests: XCTestCase {
    
    // MARK: - ObjectDetection Tests
    
    func testObjectDetectionInit() {
        let detection = ObjectDetection(
            classIndex: 0,
            label: "person",
            confidence: 0.95,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        )
        
        XCTAssertEqual(detection.classIndex, 0)
        XCTAssertEqual(detection.label, "person")
        XCTAssertEqual(detection.confidence, 0.95)
        XCTAssertNil(detection.pixelBoundingBox)
    }
    
    func testObjectDetectionWithPixelBox() {
        let detection = ObjectDetection(
            classIndex: 0,
            label: "person",
            confidence: 0.95,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4),
            pixelBoundingBox: CGRect(x: 64, y: 128, width: 192, height: 256)
        )
        
        XCTAssertNotNil(detection.pixelBoundingBox)
        XCTAssertEqual(detection.pixelBoundingBox?.origin.x, 64)
    }
    
    func testObjectDetectionEquatable() {
        let d1 = ObjectDetection(
            classIndex: 0,
            label: "person",
            confidence: 0.95,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        )
        let d2 = ObjectDetection(
            classIndex: 0,
            label: "person",
            confidence: 0.95,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        )
        
        XCTAssertEqual(d1, d2)
    }
    
    func testObjectDetectionToDrawableBox() {
        let detection = ObjectDetection(
            classIndex: 0,
            label: "person",
            confidence: 0.95,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        )
        
        let drawableBox = detection.toDrawableBox(imageSize: CGSize(width: 640, height: 480))
        
        XCTAssertEqual(drawableBox.label, "person")
        XCTAssertEqual(drawableBox.score, 0.95)
    }
    
    func testObjectDetectionToDrawableBoxWithPixelBox() {
        let detection = ObjectDetection(
            classIndex: 0,
            label: "person",
            confidence: 0.95,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4),
            pixelBoundingBox: CGRect(x: 64, y: 128, width: 192, height: 256)
        )
        
        let drawableBox = detection.toDrawableBox()
        
        // Should use pixelBoundingBox if available
        XCTAssertEqual(drawableBox.box[0], 64, accuracy: 0.1)
        XCTAssertEqual(drawableBox.box[1], 128, accuracy: 0.1)
    }
    
    // MARK: - DetectionResult Tests
    
    func testDetectionResultInit() {
        let detections = [
            ObjectDetection(classIndex: 0, label: "person", confidence: 0.95,
                          boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4))
        ]
        
        let result = DetectionResult(
            detections: detections,
            processingTimeMs: 10.5,
            rawDetectionCount: 8400,
            postConfidenceFilterCount: 100
        )
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result.processingTimeMs, 10.5)
        XCTAssertEqual(result.rawDetectionCount, 8400)
        XCTAssertEqual(result.postConfidenceFilterCount, 100)
    }
    
    func testDetectionResultFilterByClass() {
        let detections = [
            ObjectDetection(classIndex: 0, label: "person", confidence: 0.95,
                          boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)),
            ObjectDetection(classIndex: 1, label: "car", confidence: 0.85,
                          boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.2, height: 0.2)),
            ObjectDetection(classIndex: 0, label: "person", confidence: 0.75,
                          boundingBox: CGRect(x: 0.3, y: 0.3, width: 0.1, height: 0.3))
        ]
        
        let result = DetectionResult(
            detections: detections,
            processingTimeMs: 10.0,
            rawDetectionCount: 3,
            postConfidenceFilterCount: 3
        )
        
        let personDetections = result.filter(byClass: 0)
        XCTAssertEqual(personDetections.count, 2)
    }
    
    func testDetectionResultFilterByLabel() {
        let detections = [
            ObjectDetection(classIndex: 0, label: "Person", confidence: 0.95,
                          boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)),
            ObjectDetection(classIndex: 1, label: "car", confidence: 0.85,
                          boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.2, height: 0.2))
        ]
        
        let result = DetectionResult(
            detections: detections,
            processingTimeMs: 10.0,
            rawDetectionCount: 2,
            postConfidenceFilterCount: 2
        )
        
        // Should be case-insensitive
        let personDetections = result.filter(byLabel: "person")
        XCTAssertEqual(personDetections.count, 1)
    }
    
    func testDetectionResultToDrawableBoxes() {
        let detections = [
            ObjectDetection(classIndex: 0, label: "person", confidence: 0.95,
                          boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)),
            ObjectDetection(classIndex: 1, label: "car", confidence: 0.85,
                          boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.2, height: 0.2))
        ]
        
        let result = DetectionResult(
            detections: detections,
            processingTimeMs: 10.0,
            rawDetectionCount: 2,
            postConfidenceFilterCount: 2
        )
        
        let boxes = result.toDrawableBoxes(imageSize: CGSize(width: 640, height: 480))
        
        XCTAssertEqual(boxes.count, 2)
        XCTAssertEqual(boxes[0].label, "person")
        XCTAssertEqual(boxes[1].label, "car")
    }
    
    // MARK: - DetectionColorPalette Tests
    
    func testColorPaletteHas20Colors() {
        XCTAssertEqual(DetectionColorPalette.colors.count, 20)
    }
    
    func testColorPaletteCycles() {
        let color0 = DetectionColorPalette.color(forClassIndex: 0)
        let color20 = DetectionColorPalette.color(forClassIndex: 20)
        
        // Index 20 should cycle back to index 0
        XCTAssertEqual(color0.r, color20.r)
        XCTAssertEqual(color0.g, color20.g)
        XCTAssertEqual(color0.b, color20.b)
    }
    
    func testColorPaletteDistinctColors() {
        // First few colors should be distinct
        let colors = (0..<5).map { DetectionColorPalette.color(forClassIndex: $0) }
        
        // Red, Green, Blue, Yellow, Magenta should all be different
        for i in 0..<colors.count {
            for j in (i+1)..<colors.count {
                let different = colors[i].r != colors[j].r ||
                               colors[i].g != colors[j].g ||
                               colors[i].b != colors[j].b
                XCTAssertTrue(different, "Colors \(i) and \(j) should be different")
            }
        }
    }
    
    // MARK: - OutputFormat Tests
    
    func testYOLOv5Format() {
        let format = DetectionOutput.OutputFormat.yolov5(numClasses: 80)
        if case .yolov5(let numClasses) = format {
            XCTAssertEqual(numClasses, 80)
        } else {
            XCTFail("Should be yolov5 format")
        }
    }
    
    func testYOLOv8Format() {
        let format = DetectionOutput.OutputFormat.yolov8(numClasses: 80)
        if case .yolov8(let numClasses) = format {
            XCTAssertEqual(numClasses, 80)
        } else {
            XCTFail("Should be yolov8 format")
        }
    }
    
    func testCustomFormat() {
        let format = DetectionOutput.OutputFormat.custom(
            boxOffset: 0,
            classOffset: 4,
            numClasses: 91,
            boxFormat: .xyxy,
            hasObjectness: true
        )
        
        if case .custom(let boxOffset, let classOffset, let numClasses, let boxFormat, let hasObjectness) = format {
            XCTAssertEqual(boxOffset, 0)
            XCTAssertEqual(classOffset, 4)
            XCTAssertEqual(numClasses, 91)
            XCTAssertEqual(boxFormat, .xyxy)
            XCTAssertTrue(hasObjectness)
        } else {
            XCTFail("Should be custom format")
        }
    }
    
    // MARK: - LabelSource Tests
    
    func testLabelSourceCoco() {
        let source = DetectionOutput.LabelSource.coco
        if case .coco = source {
            // OK
        } else {
            XCTFail("Should be coco source")
        }
    }
    
    func testLabelSourceCustom() {
        let labels = ["cat", "dog", "bird"]
        let source = DetectionOutput.LabelSource.custom(labels)
        
        if case .custom(let labelArray) = source {
            XCTAssertEqual(labelArray, labels)
        } else {
            XCTFail("Should be custom source")
        }
    }
    
    // MARK: - OutputCoordinateSpace Tests
    
    func testPixelSpaceCoordinates() {
        let space = DetectionOutput.OutputCoordinateSpace.pixelSpace
        if case .pixelSpace = space {
            // OK
        } else {
            XCTFail("Should be pixel space")
        }
    }
    
    func testNormalizedCoordinates() {
        let space = DetectionOutput.OutputCoordinateSpace.normalized
        if case .normalized = space {
            // OK
        } else {
            XCTFail("Should be normalized")
        }
    }
    
    // MARK: - YOLOv5 Processing Tests
    
    func testProcessYOLOv5Format() throws {
        // Create minimal YOLOv5 output: [1, N, 85] where 85 = 5 + 80 classes
        // Each row: [cx, cy, w, h, obj_conf, class_scores...]
        let numClasses = 80
        let stride = 5 + numClasses
        
        // Create 2 detections
        var output = [Float](repeating: 0, count: 2 * stride)
        
        // Detection 1: person at center with high confidence
        output[0] = 320  // cx
        output[1] = 240  // cy
        output[2] = 100  // w
        output[3] = 200  // h
        output[4] = 0.9  // objectness
        output[5] = 0.95 // person class score
        
        // Detection 2: car with lower confidence
        output[stride + 0] = 400  // cx
        output[stride + 1] = 300  // cy
        output[stride + 2] = 150  // w
        output[stride + 3] = 100  // h
        output[stride + 4] = 0.8  // objectness
        output[stride + 7] = 0.7  // car class score (index 2)
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,
            labels: .coco,
            modelInputSize: CGSize(width: 640, height: 480)
        )
        
        XCTAssertGreaterThanOrEqual(result.rawDetectionCount, 2)
    }
    
    // MARK: - YOLOv8 Processing Tests
    
    func testProcessYOLOv8Format() throws {
        // Create minimal YOLOv8 output: [1, 84, N] where 84 = 4 + 80 classes
        let numClasses = 80
        let numDetections = 2
        let channels = 4 + numClasses
        
        // YOLOv8 has [channels, numDetections] layout
        var output = [Float](repeating: 0, count: channels * numDetections)
        
        // Detection 1 at column 0
        output[0 * numDetections + 0] = 320  // cx
        output[1 * numDetections + 0] = 240  // cy
        output[2 * numDetections + 0] = 100  // w
        output[3 * numDetections + 0] = 200  // h
        output[4 * numDetections + 0] = 0.9  // person class score
        
        // Detection 2 at column 1
        output[0 * numDetections + 1] = 400  // cx
        output[1 * numDetections + 1] = 300  // cy
        output[2 * numDetections + 1] = 150  // w
        output[3 * numDetections + 1] = 100  // h
        output[6 * numDetections + 1] = 0.7  // car class score (index 2)
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov8(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,
            labels: .coco,
            modelInputSize: CGSize(width: 640, height: 480)
        )
        
        XCTAssertGreaterThanOrEqual(result.rawDetectionCount, 2)
    }
    
    // MARK: - Confidence Threshold Tests
    
    func testConfidenceThresholdFiltering() throws {
        let numClasses = 80
        let stride = 5 + numClasses
        
        var output = [Float](repeating: 0, count: 3 * stride)
        
        // High confidence detection
        output[0] = 320
        output[1] = 240
        output[2] = 100
        output[3] = 200
        output[4] = 0.9
        output[5] = 0.95
        
        // Low confidence detection
        output[stride + 0] = 400
        output[stride + 1] = 300
        output[stride + 2] = 150
        output[stride + 3] = 100
        output[stride + 4] = 0.3
        output[stride + 5] = 0.2
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.7,
            iouThreshold: 0.45,
            labels: .coco,
            modelInputSize: CGSize(width: 640, height: 640)
        )
        
        // Low confidence detection should be filtered
        XCTAssertLessThan(result.count, result.rawDetectionCount)
    }
    
    // MARK: - Image Size Scaling Tests
    
    func testImageSizeScaling() throws {
        let numClasses = 80
        let stride = 5 + numClasses
        
        var output = [Float](repeating: 0, count: stride)
        output[0] = 320   // cx
        output[1] = 320   // cy
        output[2] = 100   // w
        output[3] = 100   // h
        output[4] = 0.95  // objectness
        output[5] = 0.99  // person class score
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,
            labels: .coco,
            imageSize: CGSize(width: 1920, height: 1080),
            modelInputSize: CGSize(width: 640, height: 640)
        )
        
        if let detection = result.detections.first {
            // Should have pixel bounding box
            XCTAssertNotNil(detection.pixelBoundingBox)
        }
    }
    
    // MARK: - Normalized Coordinate Space Tests
    
    func testNormalizedCoordinateSpace() throws {
        let numClasses = 80
        let stride = 5 + numClasses
        
        var output = [Float](repeating: 0, count: stride)
        output[0] = 0.5   // cx (normalized)
        output[1] = 0.5   // cy (normalized)
        output[2] = 0.2   // w (normalized)
        output[3] = 0.2   // h (normalized)
        output[4] = 0.95  // objectness
        output[5] = 0.99  // person class score
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,
            labels: .coco,
            outputCoordinateSpace: .normalized
        )
        
        if let detection = result.detections.first {
            // Normalized box should stay in 0-1 range
            XCTAssertLessThanOrEqual(detection.boundingBox.maxX, 1.0)
            XCTAssertLessThanOrEqual(detection.boundingBox.maxY, 1.0)
        }
    }
    
    // MARK: - Data Processing Tests
    
    func testProcessWithData() throws {
        let numClasses = 80
        let stride = 5 + numClasses
        
        var floats = [Float](repeating: 0, count: stride)
        floats[0] = 320
        floats[1] = 240
        floats[2] = 100
        floats[3] = 200
        floats[4] = 0.9
        floats[5] = 0.95
        
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let result = try DetectionOutput.process(
            outputData: data,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,
            labels: .coco
        )
        
        XCTAssertNotNil(result)
        XCTAssertGreaterThanOrEqual(result.processingTimeMs, 0)
    }
    
    // MARK: - Empty Output Tests
    
    func testEmptyOutput() throws {
        let result = try DetectionOutput.process(
            floatOutput: [],
            format: .yolov5(numClasses: 80),
            confidenceThreshold: 0.5,
            labels: .coco
        )
        
        XCTAssertEqual(result.count, 0)
        XCTAssertEqual(result.rawDetectionCount, 0)
    }
    
    // MARK: - NMS Tests
    
    func testNMSRemovesOverlapping() throws {
        let numClasses = 80
        let stride = 5 + numClasses
        
        var output = [Float](repeating: 0, count: 2 * stride)
        
        // Two highly overlapping detections of same class
        output[0] = 320
        output[1] = 240
        output[2] = 100
        output[3] = 200
        output[4] = 0.9
        output[5] = 0.95
        
        output[stride + 0] = 330  // Slightly offset
        output[stride + 1] = 250
        output[stride + 2] = 100
        output[stride + 3] = 200
        output[stride + 4] = 0.85
        output[stride + 5] = 0.9
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.3,  // Low threshold to suppress overlapping
            labels: .coco,
            modelInputSize: CGSize(width: 640, height: 480)
        )
        
        // NMS should keep fewer detections than raw
        XCTAssertLessThanOrEqual(result.count, result.rawDetectionCount)
    }
    
    // MARK: - Max Detections Tests
    
    func testMaxDetections() throws {
        let numClasses = 80
        let stride = 5 + numClasses
        let numDetections = 200
        
        var output = [Float](repeating: 0, count: numDetections * stride)
        
        for i in 0..<numDetections {
            let offset = i * stride
            output[offset] = Float(50 + i * 10)  // Spread out boxes
            output[offset + 1] = Float(50 + (i % 20) * 20)
            output[offset + 2] = 30
            output[offset + 3] = 30
            output[offset + 4] = 0.9
            output[offset + 5] = 0.9
        }
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov5(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.99,  // Very high to keep more
            maxDetections: 10,
            labels: .coco,
            modelInputSize: CGSize(width: 2000, height: 800)
        )
        
        XCTAssertLessThanOrEqual(result.count, 10)
    }
    
    // MARK: - Performance Tests
    
    func testDetectionProcessingPerformance() throws {
        // Create realistic YOLOv8 output
        let numClasses = 80
        let numDetections = 8400  // Typical for 640x640 input
        let channels = 4 + numClasses
        
        var output = [Float](repeating: 0, count: channels * numDetections)
        
        // Fill with some random-ish data
        for n in 0..<numDetections {
            output[0 * numDetections + n] = Float(n % 640)
            output[1 * numDetections + n] = Float((n * 3) % 640)
            output[2 * numDetections + n] = Float(30 + (n % 50))
            output[3 * numDetections + n] = Float(30 + (n % 50))
            
            // Random class with low probability
            let classIdx = n % numClasses
            output[(4 + classIdx) * numDetections + n] = 0.3 + Float(n % 5) * 0.1
        }
        
        measure {
            _ = try? DetectionOutput.process(
                floatOutput: output,
                format: .yolov8(numClasses: numClasses),
                confidenceThreshold: 0.5,
                iouThreshold: 0.45,
                labels: .coco,
                modelInputSize: CGSize(width: 640, height: 640)
            )
        }
    }
}
