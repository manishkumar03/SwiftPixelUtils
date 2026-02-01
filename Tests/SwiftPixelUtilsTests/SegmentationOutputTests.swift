import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``SegmentationOutput`` - Semantic segmentation output processing.
///
/// ## Topics
///
/// ### SegmentationResult Tests
/// - Class mask with pixel-wise labels
/// - Confidence map for each pixel
/// - Per-class pixel counts
///
/// ### Post-Processing Tests
/// - Argmax-based class assignment
/// - Confidence threshold filtering
/// - Mask resizing to original dimensions
///
/// ### Visualization Tests
/// - Color-coded class visualization
/// - Overlay on original image
/// - Legend generation
final class SegmentationOutputTests: XCTestCase {
    
    // MARK: - SegmentationResult Tests
    
    func testSegmentationResultInit() {
        let result = SegmentationResult(
            classMask: [0, 1, 1, 0],
            confidenceMap: [0.9, 0.8, 0.85, 0.95],
            width: 2,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.5,
            classPixelCounts: [0: 2, 1: 2]
        )
        
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
        XCTAssertEqual(result.numClasses, 21)
        XCTAssertEqual(result.processingTimeMs, 10.5)
    }
    
    func testClassAtPosition() {
        let result = SegmentationResult(
            classMask: [0, 1, 2, 3],  // 2x2 grid
            confidenceMap: [0.9, 0.8, 0.85, 0.95],
            width: 2,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [0: 1, 1: 1, 2: 1, 3: 1]
        )
        
        XCTAssertEqual(result.classAt(x: 0, y: 0), 0)
        XCTAssertEqual(result.classAt(x: 1, y: 0), 1)
        XCTAssertEqual(result.classAt(x: 0, y: 1), 2)
        XCTAssertEqual(result.classAt(x: 1, y: 1), 3)
    }
    
    func testClassAtOutOfBounds() {
        let result = SegmentationResult(
            classMask: [0, 1, 2, 3],
            confidenceMap: [0.9, 0.8, 0.85, 0.95],
            width: 2,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [:]
        )
        
        XCTAssertEqual(result.classAt(x: -1, y: 0), 0)
        XCTAssertEqual(result.classAt(x: 5, y: 0), 0)
        XCTAssertEqual(result.classAt(x: 0, y: 10), 0)
    }
    
    func testConfidenceAtPosition() {
        let result = SegmentationResult(
            classMask: [0, 1, 2, 3],
            confidenceMap: [0.9, 0.8, 0.85, 0.95],
            width: 2,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [:]
        )
        
        XCTAssertEqual(result.confidenceAt(x: 0, y: 0), 0.9, accuracy: 0.001)
        XCTAssertEqual(result.confidenceAt(x: 1, y: 1), 0.95, accuracy: 0.001)
    }
    
    func testPresentClasses() {
        let result = SegmentationResult(
            classMask: [0, 1, 0, 15, 15, 0],
            confidenceMap: [Float](repeating: 0.9, count: 6),
            width: 3,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [0: 3, 1: 1, 15: 2]
        )
        
        let present = result.presentClasses
        
        // Should exclude background (0)
        XCTAssertFalse(present.contains(0))
        XCTAssertTrue(present.contains(1))
        XCTAssertTrue(present.contains(15))
        XCTAssertEqual(present, [1, 15])  // Sorted
    }
    
    func testClassSummary() {
        let result = SegmentationResult(
            classMask: [0, 0, 0, 1, 1, 15],
            confidenceMap: [Float](repeating: 0.9, count: 6),
            width: 3,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [0: 3, 1: 2, 15: 1],
            labels: ["background", "aeroplane", "", "", "", "", "", "", "", "", "", "", "", "", "", "person"]
        )
        
        let summary = result.classSummary
        
        // Should be sorted by pixel count (excluding background)
        XCTAssertEqual(summary.count, 2)
        XCTAssertEqual(summary[0].classIndex, 1)  // 2 pixels
        XCTAssertEqual(summary[0].label, "aeroplane")
        XCTAssertEqual(summary[1].classIndex, 15)  // 1 pixel
    }
    
    func testBinaryMaskForClass() {
        let result = SegmentationResult(
            classMask: [0, 1, 0, 1, 2, 1],
            confidenceMap: [Float](repeating: 0.9, count: 6),
            width: 3,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [:]
        )
        
        let binaryMask = result.binaryMask(forClass: 1)
        
        XCTAssertEqual(binaryMask, [0, 1, 0, 1, 0, 1])
    }
    
    func testToColoredMask() {
        let result = SegmentationResult(
            classMask: [0, 1, 2, 0],
            confidenceMap: [Float](repeating: 0.9, count: 4),
            width: 2,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [:]
        )
        
        let rgbData = result.toColoredMask(palette: .voc)
        
        // Should have 4 pixels * 3 channels = 12 bytes
        XCTAssertEqual(rgbData.count, 12)
        
        // First pixel (class 0) should be black in VOC palette
        XCTAssertEqual(rgbData[0], 0)  // R
        XCTAssertEqual(rgbData[1], 0)  // G
        XCTAssertEqual(rgbData[2], 0)  // B
    }
    
    func testToColoredMaskRGBA() {
        let result = SegmentationResult(
            classMask: [0, 1],
            confidenceMap: [0.9, 0.8],
            width: 2,
            height: 1,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [:]
        )
        
        let rgbaData = result.toColoredMaskRGBA(palette: .voc, alpha: 200)
        
        // Should have 2 pixels * 4 channels = 8 bytes
        XCTAssertEqual(rgbaData.count, 8)
        
        // Check alpha values
        XCTAssertEqual(rgbaData[3], 200)
        XCTAssertEqual(rgbaData[7], 200)
    }
    
    func testToColoredCGImage() {
        let result = SegmentationResult(
            classMask: [0, 1, 2, 3],
            confidenceMap: [Float](repeating: 0.9, count: 4),
            width: 2,
            height: 2,
            numClasses: 21,
            processingTimeMs: 10.0,
            classPixelCounts: [:]
        )
        
        let cgImage = result.toColoredCGImage(palette: .voc)
        
        XCTAssertNotNil(cgImage)
        XCTAssertEqual(cgImage?.width, 2)
        XCTAssertEqual(cgImage?.height, 2)
    }
    
    // MARK: - SegmentationColorPalette Tests
    
    func testVOCPalette() {
        let palette = SegmentationColorPalette.voc
        
        XCTAssertEqual(palette.name, "PASCAL VOC")
        XCTAssertEqual(palette.colors.count, 21)
        
        // Background should be black
        let bg = palette.color(forClassIndex: 0)
        XCTAssertEqual(bg.r, 0)
        XCTAssertEqual(bg.g, 0)
        XCTAssertEqual(bg.b, 0)
    }
    
    func testCityscapesPalette() {
        let palette = SegmentationColorPalette.cityscapes
        
        XCTAssertEqual(palette.name, "Cityscapes")
        XCTAssertEqual(palette.colors.count, 19)
    }
    
    func testADE20KPalette() {
        let palette = SegmentationColorPalette.ade20k
        
        XCTAssertEqual(palette.name, "ADE20K")
        XCTAssertEqual(palette.colors.count, 150)
    }
    
    func testRainbowPalette() {
        let palette = SegmentationColorPalette.rainbow(numClasses: 10)
        
        XCTAssertEqual(palette.colors.count, 10)
        XCTAssertTrue(palette.name.contains("Rainbow"))
        
        // First color (background) should be black
        XCTAssertEqual(palette.colors[0].r, 0)
        XCTAssertEqual(palette.colors[0].g, 0)
        XCTAssertEqual(palette.colors[0].b, 0)
    }
    
    func testPaletteCycles() {
        let palette = SegmentationColorPalette.voc
        
        let color0 = palette.color(forClassIndex: 0)
        let color21 = palette.color(forClassIndex: 21)
        
        // Should cycle back
        XCTAssertEqual(color0.r, color21.r)
        XCTAssertEqual(color0.g, color21.g)
        XCTAssertEqual(color0.b, color21.b)
    }
    
    func testCustomPalette() {
        let customPalette = SegmentationColorPalette(
            name: "Custom",
            colors: [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        )
        
        XCTAssertEqual(customPalette.name, "Custom")
        XCTAssertEqual(customPalette.colors.count, 3)
        
        let color = customPalette.color(forClassIndex: 1)
        XCTAssertEqual(color.r, 0)
        XCTAssertEqual(color.g, 255)
        XCTAssertEqual(color.b, 0)
    }
    
    func testEmptyPaletteFallback() {
        let emptyPalette = SegmentationColorPalette(name: "Empty", colors: [])
        
        let color = emptyPalette.color(forClassIndex: 0)
        
        // Should return gray fallback
        XCTAssertEqual(color.r, 128)
        XCTAssertEqual(color.g, 128)
        XCTAssertEqual(color.b, 128)
    }
    
    // MARK: - OutputFormat Tests
    
    func testLogitsFormat() {
        let format = SegmentationOutput.OutputFormat.logits(height: 257, width: 257, numClasses: 21)
        
        if case .logits(let h, let w, let n) = format {
            XCTAssertEqual(h, 257)
            XCTAssertEqual(w, 257)
            XCTAssertEqual(n, 21)
        } else {
            XCTFail("Should be logits format")
        }
    }
    
    func testLogitsNCHWFormat() {
        let format = SegmentationOutput.OutputFormat.logitsNCHW(height: 224, width: 224, numClasses: 150)
        
        if case .logitsNCHW(let h, let w, let n) = format {
            XCTAssertEqual(h, 224)
            XCTAssertEqual(w, 224)
            XCTAssertEqual(n, 150)
        } else {
            XCTFail("Should be logitsNCHW format")
        }
    }
    
    func testProbabilitiesFormat() {
        let format = SegmentationOutput.OutputFormat.probabilities(height: 512, width: 512, numClasses: 19)
        
        if case .probabilities(let h, let w, let n) = format {
            XCTAssertEqual(h, 512)
            XCTAssertEqual(w, 512)
            XCTAssertEqual(n, 19)
        } else {
            XCTFail("Should be probabilities format")
        }
    }
    
    func testArgmaxFormat() {
        let format = SegmentationOutput.OutputFormat.argmax(height: 256, width: 256, numClasses: 21)
        
        if case .argmax(let h, let w, let n) = format {
            XCTAssertEqual(h, 256)
            XCTAssertEqual(w, 256)
            XCTAssertEqual(n, 21)
        } else {
            XCTFail("Should be argmax format")
        }
    }
    
    // MARK: - LabelSource Tests
    
    func testLabelSourceVOC() {
        let source = SegmentationOutput.LabelSource.voc
        if case .voc = source {
            // OK
        } else {
            XCTFail("Should be voc source")
        }
    }
    
    func testLabelSourceCustom() {
        let labels = ["background", "cat", "dog"]
        let source = SegmentationOutput.LabelSource.custom(labels)
        
        if case .custom(let labelArray) = source {
            XCTAssertEqual(labelArray, labels)
        } else {
            XCTFail("Should be custom source")
        }
    }
    
    // MARK: - NHWC Processing Tests
    
    func testProcessLogitsNHWC() throws {
        // Create small 2x2 output with 3 classes
        // NHWC: [1, 2, 2, 3] = 12 elements
        let height = 2
        let width = 2
        let numClasses = 3
        
        // Logits for each pixel: pixel0 -> class 1, pixel1 -> class 2, etc.
        var output = [Float](repeating: 0, count: height * width * numClasses)
        
        // Pixel (0,0): class 1 wins
        output[0] = 0.1  // class 0
        output[1] = 0.9  // class 1 (highest)
        output[2] = 0.2  // class 2
        
        // Pixel (1,0): class 2 wins
        output[3] = 0.1  // class 0
        output[4] = 0.3  // class 1
        output[5] = 0.8  // class 2 (highest)
        
        // Pixel (0,1): class 0 wins
        output[6] = 0.95 // class 0 (highest)
        output[7] = 0.1  // class 1
        output[8] = 0.2  // class 2
        
        // Pixel (1,1): class 1 wins
        output[9] = 0.1   // class 0
        output[10] = 0.85 // class 1 (highest)
        output[11] = 0.2  // class 2
        
        let result = try SegmentationOutput.process(
            floatOutput: output,
            format: .logits(height: height, width: width, numClasses: numClasses),
            labels: .none
        )
        
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
        XCTAssertEqual(result.classMask.count, 4)
        
        // Check class assignments
        XCTAssertEqual(result.classAt(x: 0, y: 0), 1)
        XCTAssertEqual(result.classAt(x: 1, y: 0), 2)
        XCTAssertEqual(result.classAt(x: 0, y: 1), 0)
        XCTAssertEqual(result.classAt(x: 1, y: 1), 1)
    }
    
    // MARK: - NCHW Processing Tests
    
    func testProcessLogitsNCHW() throws {
        // Create small 2x2 output with 3 classes
        // NCHW: [1, 3, 2, 2] = 12 elements
        let height = 2
        let width = 2
        let numClasses = 3
        let channelStride = height * width  // 4
        
        var output = [Float](repeating: 0, count: height * width * numClasses)
        
        // Class 0 channel
        output[0 * channelStride + 0] = 0.1  // pixel (0,0)
        output[0 * channelStride + 1] = 0.1  // pixel (1,0)
        output[0 * channelStride + 2] = 0.95 // pixel (0,1) -> wins
        output[0 * channelStride + 3] = 0.1  // pixel (1,1)
        
        // Class 1 channel
        output[1 * channelStride + 0] = 0.9  // pixel (0,0) -> wins
        output[1 * channelStride + 1] = 0.3  // pixel (1,0)
        output[1 * channelStride + 2] = 0.1  // pixel (0,1)
        output[1 * channelStride + 3] = 0.85 // pixel (1,1) -> wins
        
        // Class 2 channel
        output[2 * channelStride + 0] = 0.2  // pixel (0,0)
        output[2 * channelStride + 1] = 0.8  // pixel (1,0) -> wins
        output[2 * channelStride + 2] = 0.2  // pixel (0,1)
        output[2 * channelStride + 3] = 0.2  // pixel (1,1)
        
        let result = try SegmentationOutput.process(
            floatOutput: output,
            format: .logitsNCHW(height: height, width: width, numClasses: numClasses),
            labels: .none
        )
        
        XCTAssertEqual(result.classAt(x: 0, y: 0), 1)
        XCTAssertEqual(result.classAt(x: 1, y: 0), 2)
        XCTAssertEqual(result.classAt(x: 0, y: 1), 0)
        XCTAssertEqual(result.classAt(x: 1, y: 1), 1)
    }
    
    // MARK: - Argmax Processing Tests
    
    func testProcessArgmax() throws {
        // Direct class indices
        let output: [Float] = [0, 1, 2, 1]  // 2x2
        
        let result = try SegmentationOutput.process(
            floatOutput: output,
            format: .argmax(height: 2, width: 2, numClasses: 3),
            labels: .none
        )
        
        XCTAssertEqual(result.classMask, [0, 1, 2, 1])
        
        // Confidence should be 1.0 for argmax format
        XCTAssertEqual(result.confidenceAt(x: 0, y: 0), 1.0)
    }
    
    // MARK: - Data Processing Tests
    
    func testProcessWithData() throws {
        var floats: [Float] = [0.1, 0.9, 0.8, 0.2]  // Simple 2x2 with 2 classes -> 2x1 argmax
        // Actually need proper shape: 2x2 with 1 class each
        floats = [0, 1, 2, 0]  // argmax format
        
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let result = try SegmentationOutput.process(
            outputData: data,
            format: .argmax(height: 2, width: 2, numClasses: 3),
            labels: .none
        )
        
        XCTAssertNotNil(result)
        XCTAssertGreaterThanOrEqual(result.processingTimeMs, 0)
    }
    
    // MARK: - Invalid Size Tests
    
    func testInvalidOutputSize() {
        let output = [Float](repeating: 0, count: 10)  // Wrong size
        
        XCTAssertThrowsError(try SegmentationOutput.process(
            floatOutput: output,
            format: .logits(height: 100, width: 100, numClasses: 21),
            labels: .none
        )) { error in
            if case PixelUtilsError.invalidOptions = error {
                // Expected
            } else {
                XCTFail("Should throw invalidOptions error")
            }
        }
    }
    
    // MARK: - Class Statistics Tests
    
    func testClassPixelCounts() throws {
        // 3x3 mask with various classes
        let height = 3
        let width = 3
        let numClasses = 3
        
        // All pixels are class 0 except center is class 1
        var output = [Float](repeating: 0, count: height * width * numClasses)
        
        for i in 0..<9 {
            let baseIdx = i * numClasses
            if i == 4 {  // Center pixel -> class 1
                output[baseIdx + 1] = 1.0
            } else {
                output[baseIdx + 0] = 1.0  // class 0
            }
        }
        
        let result = try SegmentationOutput.process(
            floatOutput: output,
            format: .logits(height: height, width: width, numClasses: numClasses),
            labels: .none
        )
        
        XCTAssertEqual(result.classPixelCounts[0], 8)
        XCTAssertEqual(result.classPixelCounts[1], 1)
    }
    
    // MARK: - Performance Tests
    
    func testSegmentationProcessingPerformance() throws {
        // Typical DeepLabV3 output: 257x257x21
        let height = 257
        let width = 257
        let numClasses = 21
        
        let output = [Float](repeating: 0.5, count: height * width * numClasses)
        
        measure {
            _ = try? SegmentationOutput.process(
                floatOutput: output,
                format: .logits(height: height, width: width, numClasses: numClasses),
                labels: .voc
            )
        }
    }
    
    func testColoredMaskPerformance() {
        let result = SegmentationResult(
            classMask: [Int](repeating: 0, count: 512 * 512),
            confidenceMap: [Float](repeating: 0.9, count: 512 * 512),
            width: 512,
            height: 512,
            numClasses: 21,
            processingTimeMs: 0,
            classPixelCounts: [:]
        )
        
        measure {
            _ = result.toColoredMaskRGBA(palette: .voc)
        }
    }
}
