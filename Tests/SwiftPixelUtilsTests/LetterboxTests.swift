import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``Letterbox`` - YOLO-style letterbox padding operations.
///
/// ## Topics
///
/// ### Letterbox Application Tests
/// - Maintains aspect ratio during resize
/// - Padding calculation for various aspect ratios
/// - Square target (e.g., 640x640 for YOLO)
///
/// ### Coordinate Reversal Tests
/// - Maps letterboxed coordinates back to original
/// - Handles bounding box adjustment
///
/// ### Options Tests
/// - Custom pad color (gray, black, white)
/// - Center vs corner alignment
/// - Stride-based padding
final class LetterboxTests: XCTestCase {
    
    // MARK: - Helper Methods
    
    private func createTestCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 128, count: width * height * bytesPerPixel)
        
        for i in stride(from: 3, to: pixels.count, by: bytesPerPixel) {
            pixels[i] = 255  // Alpha
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        
        return context.makeImage()!
    }
    
    // MARK: - LetterboxOptions Tests
    
    func testLetterboxOptionsDefaults() {
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        XCTAssertEqual(options.targetWidth, 640)
        XCTAssertEqual(options.targetHeight, 640)
        XCTAssertEqual(options.fillColor.r, 114)  // YOLO gray
        XCTAssertEqual(options.fillColor.g, 114)
        XCTAssertEqual(options.fillColor.b, 114)
        XCTAssertTrue(options.scaleUp)
        XCTAssertTrue(options.center)
    }
    
    func testLetterboxOptionsCustom() {
        let options = LetterboxOptions(
            targetWidth: 416,
            targetHeight: 416,
            fillColor: (0, 0, 0),  // Black
            scaleUp: false,
            center: false
        )
        
        XCTAssertEqual(options.targetWidth, 416)
        XCTAssertEqual(options.targetHeight, 416)
        XCTAssertEqual(options.fillColor.r, 0)
        XCTAssertFalse(options.scaleUp)
        XCTAssertFalse(options.center)
    }
    
    // MARK: - Square Image Tests
    
    func testLetterboxSquareToSquare() throws {
        let image = createTestCGImage(width: 640, height: 640)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        XCTAssertEqual(result.originalWidth, 640)
        XCTAssertEqual(result.originalHeight, 640)
        XCTAssertEqual(result.scale, 1.0, accuracy: 0.01)
        XCTAssertEqual(result.offsetX, 0, accuracy: 0.01)
        XCTAssertEqual(result.offsetY, 0, accuracy: 0.01)
    }
    
    func testLetterboxSmallSquareToLargeSquare() throws {
        let image = createTestCGImage(width: 320, height: 320)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        XCTAssertEqual(result.scale, 2.0, accuracy: 0.01)
        XCTAssertEqual(result.offsetX, 0, accuracy: 0.01)
        XCTAssertEqual(result.offsetY, 0, accuracy: 0.01)
    }
    
    func testLetterboxLargeSquareToSmallSquare() throws {
        let image = createTestCGImage(width: 1280, height: 1280)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        XCTAssertEqual(result.scale, 0.5, accuracy: 0.01)
    }
    
    // MARK: - Wide Image Tests
    
    func testLetterboxWideImage() throws {
        let image = createTestCGImage(width: 1920, height: 1080)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        
        // Scale should be determined by width (wider image)
        let expectedScale = 640.0 / 1920.0
        XCTAssertEqual(result.scale, expectedScale, accuracy: 0.01)
        
        // Should have vertical padding (top and bottom bars)
        XCTAssertEqual(result.offsetX, 0, accuracy: 0.5)
        XCTAssertGreaterThan(result.offsetY, 0)
    }
    
    // MARK: - Tall Image Tests
    
    func testLetterboxTallImage() throws {
        let image = createTestCGImage(width: 1080, height: 1920)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        
        // Scale should be determined by height (taller image)
        let expectedScale = 640.0 / 1920.0
        XCTAssertEqual(result.scale, expectedScale, accuracy: 0.01)
        
        // Should have horizontal padding (left and right bars)
        XCTAssertGreaterThan(result.offsetX, 0)
        XCTAssertEqual(result.offsetY, 0, accuracy: 0.5)
    }
    
    // MARK: - Scale Up Option Tests
    
    func testLetterboxNoScaleUp() throws {
        let image = createTestCGImage(width: 320, height: 320)
        let options = LetterboxOptions(
            targetWidth: 640,
            targetHeight: 640,
            scaleUp: false
        )
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        // Should not scale up beyond 1.0
        XCTAssertEqual(result.scale, 1.0, accuracy: 0.01)
        
        // Image should be centered with padding
        let expectedOffset = (640.0 - 320.0) / 2.0
        XCTAssertEqual(result.offsetX, expectedOffset, accuracy: 0.5)
        XCTAssertEqual(result.offsetY, expectedOffset, accuracy: 0.5)
    }
    
    func testLetterboxNoScaleUpLargeImage() throws {
        let image = createTestCGImage(width: 1280, height: 1280)
        let options = LetterboxOptions(
            targetWidth: 640,
            targetHeight: 640,
            scaleUp: false
        )
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        // Should still scale down
        XCTAssertEqual(result.scale, 0.5, accuracy: 0.01)
    }
    
    // MARK: - Center Option Tests
    
    func testLetterboxNotCentered() throws {
        let image = createTestCGImage(width: 1920, height: 1080)
        let options = LetterboxOptions(
            targetWidth: 640,
            targetHeight: 640,
            center: false
        )
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        // Should have no offset (top-left aligned)
        XCTAssertEqual(result.offsetX, 0, accuracy: 0.01)
        XCTAssertEqual(result.offsetY, 0, accuracy: 0.01)
    }
    
    // MARK: - Fill Color Tests
    
    func testLetterboxCustomFillColor() throws {
        let image = createTestCGImage(width: 320, height: 480)
        let options = LetterboxOptions(
            targetWidth: 640,
            targetHeight: 640,
            fillColor: (255, 0, 0)  // Red
        )
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.fillColor.r, 255)
        XCTAssertEqual(result.fillColor.g, 0)
        XCTAssertEqual(result.fillColor.b, 0)
    }
    
    // MARK: - Result Properties Tests
    
    func testLetterboxExtendedResult() throws {
        let image = createTestCGImage(width: 800, height: 600)
        let options = LetterboxOptions(
            targetWidth: 640,
            targetHeight: 640,
            fillColor: (114, 114, 114)
        )
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        XCTAssertFalse(result.imageBase64.isEmpty)
        XCTAssertNotNil(result.cgImage)
        XCTAssertEqual(result.cgImage.width, 640)
        XCTAssertEqual(result.cgImage.height, 640)
        XCTAssertEqual(result.originalWidth, 800)
        XCTAssertEqual(result.originalHeight, 600)
        XCTAssertGreaterThan(result.processingTimeMs, 0)
    }
    
    // MARK: - Reverse Transform Tests
    
    func testReverseTransformBoxes() {
        let boxes: [[Double]] = [
            [100, 150, 200, 250],  // x1, y1, x2, y2
            [300, 350, 400, 450]
        ]
        
        let scale = 0.5
        let offsetX = 80.0
        let offsetY = 60.0
        
        let transformed = Letterbox.reverseTransformBoxes(
            boxes: boxes,
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY
        )
        
        XCTAssertEqual(transformed.count, 2)
        
        // First box: (100-80)/0.5, (150-60)/0.5, etc.
        XCTAssertEqual(transformed[0][0], (100 - offsetX) / scale, accuracy: 0.01)
        XCTAssertEqual(transformed[0][1], (150 - offsetY) / scale, accuracy: 0.01)
        XCTAssertEqual(transformed[0][2], (200 - offsetX) / scale, accuracy: 0.01)
        XCTAssertEqual(transformed[0][3], (250 - offsetY) / scale, accuracy: 0.01)
    }
    
    func testReverseTransformSingleBox() {
        let box: [Double] = [100, 150, 200, 250]
        
        let transformed = Letterbox.reverseTransformBox(
            box: box,
            scale: 0.5,
            offsetX: 80,
            offsetY: 60
        )
        
        XCTAssertEqual(transformed.count, 4)
        XCTAssertEqual(transformed[0], (100 - 80) / 0.5, accuracy: 0.01)
        XCTAssertEqual(transformed[1], (150 - 60) / 0.5, accuracy: 0.01)
    }
    
    func testReverseTransformDetections() {
        let detections = [
            Detection(box: [100, 150, 200, 250], score: 0.9, classIndex: 0, label: "person"),
            Detection(box: [300, 350, 400, 450], score: 0.8, classIndex: 1, label: "car")
        ]
        
        let transformed = Letterbox.reverseTransformDetections(
            detections: detections,
            scale: 0.5,
            offsetX: 80,
            offsetY: 60,
            format: .xyxy
        )
        
        XCTAssertEqual(transformed.count, 2)
        XCTAssertEqual(transformed[0].label, "person")
        XCTAssertEqual(transformed[0].score, 0.9, accuracy: 0.001)
        XCTAssertEqual(transformed[0].classIndex, 0)
    }
    
    func testReverseTransformDetectionsXYWH() {
        let detections = [
            Detection(box: [100, 150, 50, 60], score: 0.9, classIndex: 0, label: "person")  // x, y, w, h
        ]
        
        let transformed = Letterbox.reverseTransformDetections(
            detections: detections,
            scale: 0.5,
            offsetX: 80,
            offsetY: 60,
            format: .xywh
        )
        
        XCTAssertEqual(transformed.count, 1)
        // Output should also be xywh
        XCTAssertEqual(transformed[0].box.count, 4)
    }
    
    func testReverseTransformDetectionsCXCYWH() {
        let detections = [
            Detection(box: [150, 200, 100, 120], score: 0.9, classIndex: 0, label: "person")  // cx, cy, w, h
        ]
        
        let transformed = Letterbox.reverseTransformDetections(
            detections: detections,
            scale: 0.5,
            offsetX: 80,
            offsetY: 60,
            format: .cxcywh
        )
        
        XCTAssertEqual(transformed.count, 1)
        // Output should also be cxcywh
        XCTAssertEqual(transformed[0].box.count, 4)
    }
    
    // MARK: - Roundtrip Transform Tests
    
    func testLetterboxAndReverseTransform() throws {
        let originalWidth = 1920
        let originalHeight = 1080
        let image = createTestCGImage(width: originalWidth, height: originalHeight)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try Letterbox.applySync(to: .cgImage(image), options: options)
        
        // Create a detection in letterboxed space
        // Say we detect something at center of letterboxed image
        let letterboxedBox: [Double] = [280, 280, 360, 360]  // 80x80 box at center
        
        let originalBox = Letterbox.reverseTransformBox(
            box: letterboxedBox,
            scale: result.scale,
            offsetX: result.offsetX,
            offsetY: result.offsetY
        )
        
        // Verify the box is in valid original image space
        XCTAssertGreaterThanOrEqual(originalBox[0], 0)
        XCTAssertGreaterThanOrEqual(originalBox[1], 0)
        XCTAssertLessThanOrEqual(originalBox[2], Double(originalWidth))
        XCTAssertLessThanOrEqual(originalBox[3], Double(originalHeight))
    }
    
    // MARK: - Edge Cases
    
    func testLetterboxEmptyBox() {
        let result = Letterbox.reverseTransformBoxes(
            boxes: [],
            scale: 0.5,
            offsetX: 80,
            offsetY: 60
        )
        
        XCTAssertTrue(result.isEmpty)
    }
    
    func testLetterboxBoxWithInsufficientCoordinates() {
        let boxes: [[Double]] = [[100, 150]]  // Only 2 values
        
        let result = Letterbox.reverseTransformBoxes(
            boxes: boxes,
            scale: 0.5,
            offsetX: 80,
            offsetY: 60
        )
        
        // Should return original box unchanged
        XCTAssertEqual(result[0], [100, 150])
    }
    
    // MARK: - Async API Tests
    
    func testLetterboxAsync() async throws {
        let image = createTestCGImage(width: 800, height: 600)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        let result = try await Letterbox.apply(to: .cgImage(image), options: options)
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
    }
    
    // MARK: - Combined Operations Tests
    
    func testApplyAndExtract() async throws {
        let image = createTestCGImage(width: 800, height: 600)
        let letterboxOptions = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        let pixelOptions = PixelDataOptions()
        
        let (pixels, letterbox) = try await Letterbox.applyAndExtract(
            from: .cgImage(image),
            letterboxOptions: letterboxOptions,
            pixelOptions: pixelOptions
        )
        
        XCTAssertEqual(letterbox.width, 640)
        XCTAssertEqual(letterbox.height, 640)
        XCTAssertFalse(pixels.data.isEmpty)
    }
    
    // MARK: - Performance Tests
    
    func testLetterboxPerformance() throws {
        let image = createTestCGImage(width: 1920, height: 1080)
        let options = LetterboxOptions(targetWidth: 640, targetHeight: 640)
        
        measure {
            _ = try? Letterbox.applySync(to: .cgImage(image), options: options)
        }
    }
    
    func testReverseTransformPerformance() {
        var boxes: [[Double]] = []
        for i in 0..<1000 {
            boxes.append([Double(i * 10), Double(i * 10), Double(i * 10 + 50), Double(i * 10 + 50)])
        }
        
        measure {
            _ = Letterbox.reverseTransformBoxes(
                boxes: boxes,
                scale: 0.5,
                offsetX: 80,
                offsetY: 60
            )
        }
    }
}
