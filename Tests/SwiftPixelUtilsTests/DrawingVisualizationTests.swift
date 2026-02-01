import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``Drawing`` - Visualization and annotation utilities.
///
/// ## Topics
///
/// ### Bounding Box Drawing Tests
/// - Basic box rendering with labels
/// - Custom colors and line widths
/// - Multiple box rendering
///
/// ### Keypoint Drawing Tests
/// - Pose keypoint visualization
/// - Confidence-based filtering
/// - Custom styling options
///
/// ### Mask Overlay Tests
/// - Segmentation mask blending
/// - Alpha transparency handling
/// - Color mapping options
///
/// ### Heatmap Overlay Tests
/// - Attention map visualization
/// - Color scheme selection (jet, viridis, grayscale)
final class DrawingVisualizationTests: XCTestCase {
    
    // MARK: - Helper Methods
    
    private func createTestCGImage(width: Int = 200, height: Int = 200) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 200, count: width * height * bytesPerPixel)
        
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
    
    // MARK: - BoxDrawingOptions Tests
    
    func testBoxDrawingOptionsDefaults() {
        let options = BoxDrawingOptions()
        
        XCTAssertEqual(options.lineWidth, 2.0)
        XCTAssertEqual(options.fontSize, 12.0)
        XCTAssertTrue(options.drawLabels)
        XCTAssertTrue(options.drawScores)
        XCTAssertEqual(options.labelBackgroundAlpha, 0.7)
        XCTAssertEqual(options.defaultColor.r, 255)
        XCTAssertEqual(options.defaultColor.g, 0)
        XCTAssertEqual(options.defaultColor.b, 0)
    }
    
    func testBoxDrawingOptionsCustom() {
        let options = BoxDrawingOptions(
            lineWidth: 4.0,
            fontSize: 18.0,
            drawLabels: false,
            drawScores: false,
            labelBackgroundAlpha: 0.5,
            defaultColor: (0, 255, 0, 255)
        )
        
        XCTAssertEqual(options.lineWidth, 4.0)
        XCTAssertEqual(options.fontSize, 18.0)
        XCTAssertFalse(options.drawLabels)
        XCTAssertFalse(options.drawScores)
        XCTAssertEqual(options.defaultColor.g, 255)
    }
    
    // MARK: - DrawableBox Tests
    
    func testDrawableBoxInit() {
        let box = DrawableBox(box: [10, 20, 100, 150])
        
        XCTAssertEqual(box.box, [10, 20, 100, 150])
        XCTAssertNil(box.label)
        XCTAssertNil(box.score)
        XCTAssertNil(box.color)
    }
    
    func testDrawableBoxWithAllProperties() {
        let box = DrawableBox(
            box: [10, 20, 100, 150],
            label: "person",
            score: 0.95,
            color: (255, 0, 0, 255)
        )
        
        XCTAssertEqual(box.box, [10, 20, 100, 150])
        XCTAssertEqual(box.label, "person")
        XCTAssertEqual(box.score, 0.95)
        XCTAssertNotNil(box.color)
        XCTAssertEqual(box.color?.r, 255)
    }
    
    // MARK: - KeypointDrawingOptions Tests
    
    func testKeypointDrawingOptionsDefaults() {
        let options = KeypointDrawingOptions()
        
        XCTAssertEqual(options.pointRadius, 4.0)
        XCTAssertEqual(options.lineWidth, 2.0)
        XCTAssertTrue(options.drawSkeleton)
        XCTAssertNil(options.skeleton)
        XCTAssertEqual(options.confidenceThreshold, 0.5)
    }
    
    func testKeypointDrawingOptionsWithSkeleton() {
        let options = KeypointDrawingOptions(
            pointRadius: 6.0,
            skeleton: KeypointDrawingOptions.cocoSkeleton,
            confidenceThreshold: 0.3
        )
        
        XCTAssertEqual(options.pointRadius, 6.0)
        XCTAssertNotNil(options.skeleton)
        XCTAssertEqual(options.confidenceThreshold, 0.3)
    }
    
    func testCocoSkeletonConnections() {
        let skeleton = KeypointDrawingOptions.cocoSkeleton
        
        XCTAssertFalse(skeleton.isEmpty)
        XCTAssertEqual(skeleton.count, 16)  // COCO has 16 skeleton connections
        
        // Check first connection (nose to left eye)
        XCTAssertEqual(skeleton[0].0, 0)
        XCTAssertEqual(skeleton[0].1, 1)
    }
    
    // MARK: - DrawableKeypoint Tests
    
    func testDrawableKeypointInit() {
        let keypoint = DrawableKeypoint(x: 100, y: 150)
        
        XCTAssertEqual(keypoint.x, 100)
        XCTAssertEqual(keypoint.y, 150)
        XCTAssertEqual(keypoint.confidence, 1.0)
        XCTAssertNil(keypoint.color)
    }
    
    func testDrawableKeypointWithConfidence() {
        let keypoint = DrawableKeypoint(x: 100, y: 150, confidence: 0.8)
        
        XCTAssertEqual(keypoint.confidence, 0.8)
    }
    
    func testDrawableKeypointWithColor() {
        let keypoint = DrawableKeypoint(
            x: 100,
            y: 150,
            confidence: 0.9,
            color: (255, 128, 0, 255)
        )
        
        XCTAssertNotNil(keypoint.color)
        XCTAssertEqual(keypoint.color?.r, 255)
        XCTAssertEqual(keypoint.color?.g, 128)
    }
    
    // MARK: - MaskOverlayOptions Tests
    
    func testMaskOverlayOptionsDefaults() {
        let options = MaskOverlayOptions(maskWidth: 100, maskHeight: 100)
        
        XCTAssertEqual(options.alpha, 0.5)
        XCTAssertEqual(options.color.r, 0)
        XCTAssertEqual(options.color.g, 128)
        XCTAssertEqual(options.color.b, 255)
        XCTAssertEqual(options.threshold, 0.5)
    }
    
    func testMaskOverlayOptionsCustom() {
        let options = MaskOverlayOptions(
            alpha: 0.8,
            color: (255, 0, 0),
            maskWidth: 256,
            maskHeight: 256,
            threshold: 0.7
        )
        
        XCTAssertEqual(options.alpha, 0.8)
        XCTAssertEqual(options.color.r, 255)
        XCTAssertEqual(options.maskWidth, 256)
        XCTAssertEqual(options.maskHeight, 256)
        XCTAssertEqual(options.threshold, 0.7)
    }
    
    // MARK: - HeatmapOverlayOptions Tests
    
    func testHeatmapOverlayOptionsDefaults() {
        let options = HeatmapOverlayOptions(heatmapWidth: 64, heatmapHeight: 64)
        
        XCTAssertEqual(options.alpha, 0.6)
        XCTAssertEqual(options.colorScheme, .jet)
        XCTAssertEqual(options.heatmapWidth, 64)
        XCTAssertEqual(options.heatmapHeight, 64)
    }
    
    func testHeatmapOverlayOptionsCustomScheme() {
        let options = HeatmapOverlayOptions(
            alpha: 0.9,
            colorScheme: .viridis,
            heatmapWidth: 128,
            heatmapHeight: 128
        )
        
        XCTAssertEqual(options.colorScheme, .viridis)
        XCTAssertEqual(options.alpha, 0.9)
    }
    
    // MARK: - HeatmapColorScheme Tests
    
    func testHeatmapColorSchemes() {
        // Test all schemes exist
        _ = HeatmapColorScheme.jet
        _ = HeatmapColorScheme.viridis
        _ = HeatmapColorScheme.hot
        _ = HeatmapColorScheme.grayscale
    }
    
    // MARK: - Drawing.drawBoxes Tests
    
    func testDrawBoxesBasic() async throws {
        let image = createTestCGImage()
        let boxes = [
            DrawableBox(box: [10, 10, 50, 50], label: "person", score: 0.9)
        ]
        
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawBoxesMultiple() async throws {
        let image = createTestCGImage()
        let boxes = [
            DrawableBox(box: [10, 10, 50, 50], label: "person", score: 0.9),
            DrawableBox(box: [60, 60, 100, 100], label: "car", score: 0.85),
            DrawableBox(box: [120, 30, 180, 90], label: "dog", score: 0.75)
        ]
        
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawBoxesWithOptions() async throws {
        let image = createTestCGImage()
        let boxes = [
            DrawableBox(box: [10, 10, 100, 100])
        ]
        let options = BoxDrawingOptions(
            lineWidth: 4.0,
            drawLabels: false,
            drawScores: false
        )
        
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawBoxesWithColors() async throws {
        let image = createTestCGImage()
        let boxes = [
            DrawableBox(box: [10, 10, 50, 50], label: "red", color: (255, 0, 0, 255)),
            DrawableBox(box: [60, 60, 100, 100], label: "green", color: (0, 255, 0, 255)),
            DrawableBox(box: [120, 30, 160, 70], label: "blue", color: (0, 0, 255, 255))
        ]
        
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawBoxesEmpty() async throws {
        let image = createTestCGImage()
        
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: []
        )
        
        // Should return original image unchanged
        XCTAssertNotNil(result)
    }
    
    // MARK: - Drawing.drawKeypoints Tests
    
    func testDrawKeypointsBasic() async throws {
        let image = createTestCGImage()
        let keypoints = [
            DrawableKeypoint(x: 50, y: 50),
            DrawableKeypoint(x: 100, y: 100),
            DrawableKeypoint(x: 150, y: 75)
        ]
        
        let result = try await Drawing.drawKeypoints(
            on: .cgImage(image),
            keypoints: keypoints
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawKeypointsWithSkeleton() async throws {
        let image = createTestCGImage(width: 400, height: 400)
        
        // Create 17 COCO keypoints
        let keypoints = (0..<17).map { i in
            DrawableKeypoint(
                x: Float(50 + i * 20),
                y: Float(100 + (i % 5) * 40),
                confidence: 0.9
            )
        }
        
        let options = KeypointDrawingOptions(
            skeleton: KeypointDrawingOptions.cocoSkeleton
        )
        
        let result = try await Drawing.drawKeypoints(
            on: .cgImage(image),
            keypoints: keypoints,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawKeypointsWithConfidenceThreshold() async throws {
        let image = createTestCGImage()
        let keypoints = [
            DrawableKeypoint(x: 50, y: 50, confidence: 0.9),
            DrawableKeypoint(x: 100, y: 100, confidence: 0.3),  // Below threshold
            DrawableKeypoint(x: 150, y: 75, confidence: 0.8)
        ]
        
        let options = KeypointDrawingOptions(confidenceThreshold: 0.5)
        
        let result = try await Drawing.drawKeypoints(
            on: .cgImage(image),
            keypoints: keypoints,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    // MARK: - Drawing.overlayMask Tests
    
    func testOverlayMaskBasic() async throws {
        let image = createTestCGImage(width: 100, height: 100)
        let mask = [Float](repeating: 0.7, count: 100 * 100)
        let options = MaskOverlayOptions(maskWidth: 100, maskHeight: 100)
        
        let result = try await Drawing.overlayMask(
            on: .cgImage(image),
            mask: mask,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    func testOverlayMaskWithThreshold() async throws {
        let image = createTestCGImage(width: 100, height: 100)
        
        // Create mask with varying values
        var mask = [Float](repeating: 0, count: 100 * 100)
        for i in 0..<50 * 100 {
            mask[i] = 0.8  // Above threshold
        }
        
        let options = MaskOverlayOptions(
            maskWidth: 100,
            maskHeight: 100,
            threshold: 0.5
        )
        
        let result = try await Drawing.overlayMask(
            on: .cgImage(image),
            mask: mask,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    // MARK: - Drawing.overlayHeatmap Tests
    
    func testOverlayHeatmapBasic() async throws {
        let image = createTestCGImage(width: 256, height: 256)
        let heatmap = [Float](repeating: 0.5, count: 64 * 64)
        let options = HeatmapOverlayOptions(heatmapWidth: 64, heatmapHeight: 64)
        
        let result = try await Drawing.overlayHeatmap(
            on: .cgImage(image),
            heatmap: heatmap,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    func testOverlayHeatmapWithGradient() async throws {
        let image = createTestCGImage(width: 256, height: 256)
        
        // Create gradient heatmap
        var heatmap = [Float](repeating: 0, count: 64 * 64)
        for y in 0..<64 {
            for x in 0..<64 {
                heatmap[y * 64 + x] = Float(x) / 64.0
            }
        }
        
        let options = HeatmapOverlayOptions(
            colorScheme: .viridis,
            heatmapWidth: 64,
            heatmapHeight: 64
        )
        
        let result = try await Drawing.overlayHeatmap(
            on: .cgImage(image),
            heatmap: heatmap,
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    // MARK: - Combined Drawing Tests
    
    func testDrawBoxesAndKeypoints() async throws {
        let image = createTestCGImage(width: 300, height: 300)
        
        // Draw boxes
        let boxes = [
            DrawableBox(box: [50, 50, 150, 250], label: "person", score: 0.9)
        ]
        
        var result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes
        )
        
        // Draw keypoints on top
        let keypoints = [
            DrawableKeypoint(x: 100, y: 80, confidence: 0.9),
            DrawableKeypoint(x: 90, y: 100, confidence: 0.85),
            DrawableKeypoint(x: 110, y: 100, confidence: 0.88)
        ]
        
        #if canImport(UIKit)
        if let uiImage = result as? UIImage, let cgImage = uiImage.cgImage {
            result = try await Drawing.drawKeypoints(
                on: .cgImage(cgImage),
                keypoints: keypoints
            )
        }
        #elseif canImport(AppKit)
        if let nsImage = result as? NSImage {
            var rect = NSRect(origin: .zero, size: nsImage.size)
            if let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) {
                result = try await Drawing.drawKeypoints(
                    on: .cgImage(cgImage),
                    keypoints: keypoints
                )
            }
        }
        #endif
        
        XCTAssertNotNil(result)
    }
    
    // MARK: - Edge Cases
    
    func testDrawBoxOutOfBounds() async throws {
        let image = createTestCGImage(width: 100, height: 100)
        let boxes = [
            DrawableBox(box: [-10, -10, 50, 50]),  // Partially out of bounds
            DrawableBox(box: [80, 80, 150, 150])   // Partially out of bounds
        ]
        
        // Should handle gracefully
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes
        )
        
        XCTAssertNotNil(result)
    }
    
    func testDrawBoxInvalidCoordinates() async throws {
        let image = createTestCGImage()
        let boxes = [
            DrawableBox(box: [100, 100, 10, 10])  // x2 < x1, y2 < y1
        ]
        
        // Should handle gracefully
        let result = try await Drawing.drawBoxes(
            on: .cgImage(image),
            boxes: boxes
        )
        
        XCTAssertNotNil(result)
    }
    
    // MARK: - Performance Tests
    
    func testDrawBoxesPerformance() async throws {
        let image = createTestCGImage(width: 1920, height: 1080)
        let boxes = (0..<100).map { i in
            DrawableBox(
                box: [Float(i * 10), Float(i * 5), Float(i * 10 + 50), Float(i * 5 + 30)],
                label: "object_\(i)",
                score: Float(90 - i) / 100.0
            )
        }
        
        measure {
            let expectation = self.expectation(description: "draw")
            Task {
                _ = try? await Drawing.drawBoxes(
                    on: .cgImage(image),
                    boxes: boxes
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }
    }
}
