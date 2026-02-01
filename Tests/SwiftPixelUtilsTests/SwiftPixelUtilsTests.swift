import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Integration tests for SwiftPixelUtils - End-to-end API validation.
///
/// These tests verify the main public API works correctly across modules.
/// For detailed unit tests, see the individual test files for each component.
///
/// ## Topics
///
/// ### Pixel Extraction Integration
/// - Full pipeline from image to tensor
/// - Options combination testing
///
/// ### Model Preset Integration
/// - YOLO, MobileNet, ResNet presets
/// - Preset customization
///
/// ### Output Processing Integration
/// - Classification, detection, segmentation pipelines
/// - End-to-end workflow validation
final class SwiftPixelUtilsTests: XCTestCase {
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        
        for y in 0..<height {
            for x in 0..<width {
                let r = CGFloat(x) / CGFloat(width)
                let g = CGFloat(y) / CGFloat(height)
                context.setFillColor(red: r, green: g, blue: 0.5, alpha: 1.0)
                context.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        
        return context.makeImage()!
    }
    
    // MARK: - Integration Tests
    
    func testPixelExtractionIntegration() async throws {
        let image = createTestImage(width: 100, height: 100)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(result.width, 100)
        XCTAssertEqual(result.height, 100)
        XCTAssertEqual(result.channels, 3)
        XCTAssertGreaterThan(result.data.count, 0)
    }
    
    func testYOLOPresetIntegration() async throws {
        let image = createTestImage(width: 640, height: 480)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: ModelPresets.yolov8
        )
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        XCTAssertEqual(result.channels, 3)
        XCTAssertEqual(result.dataLayout, .nchw)
    }
    
    func testMobileNetPresetIntegration() async throws {
        let image = createTestImage(width: 300, height: 300)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: ModelPresets.mobilenet
        )
        
        XCTAssertEqual(result.width, 224)
        XCTAssertEqual(result.height, 224)
        XCTAssertEqual(result.dataLayout, .nhwc)
    }
    
    func testBatchProcessingIntegration() async throws {
        let sources: [ImageSource] = [
            .cgImage(createTestImage(width: 100, height: 100)),
            .cgImage(createTestImage(width: 100, height: 100)),
            .cgImage(createTestImage(width: 100, height: 100))
        ]
        
        let results = try await PixelExtractor.batchGetPixelData(
            sources: sources,
            options: PixelDataOptions(),
            concurrency: 2
        )
        
        XCTAssertEqual(results.count, 3)
        for result in results {
            XCTAssertEqual(result.width, 100)
            XCTAssertEqual(result.height, 100)
        }
    }
    
    func testLetterboxIntegration() async throws {
        let image = createTestImage(width: 300, height: 200)
        
        let result = try await Letterbox.apply(
            to: .cgImage(image),
            options: LetterboxOptions(targetWidth: 640, targetHeight: 640)
        )
        
        XCTAssertNotNil(result)
    }
    
    func testImageAnalysisIntegration() async throws {
        let image = createTestImage(width: 100, height: 100)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
        
        XCTAssertEqual(metadata.width, 100)
        XCTAssertEqual(metadata.height, 100)
        XCTAssertEqual(stats.mean.count, 3)
        XCTAssertEqual(stats.std.count, 3)
    }
    
    func testFiveCropIntegration() async throws {
        let image = createTestImage(width: 300, height: 300)
        let options = CropOptions(width: 100, height: 100, normalization: .scale)
        
        let result = try await MultiCropOperations.fiveCrop(
            from: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.crops.count, 5)
        XCTAssertEqual(result.positions.count, 5)
    }
    
    func testTensorValidationIntegration() throws {
        let tensor = [Float](repeating: 0.5, count: 1 * 3 * 224 * 224)
        let shape = [1, 3, 224, 224]
        let spec = TensorSpec(shape: shape)
        
        let result = TensorValidation.validate(data: tensor, shape: shape, spec: spec)
        
        XCTAssertTrue(result.isValid)
    }
    
    func testClassificationIntegration() throws {
        // Test with float32 data
        let logits = (0..<1000).map { Float($0) / 1000.0 }
        let data = Data(bytes: logits, count: logits.count * MemoryLayout<Float>.size)
        
        let result = try ClassificationOutput.process(
            outputData: data,
            quantization: .none,
            topK: 5,
            labels: .none
        )
        
        XCTAssertGreaterThan(result.predictions.count, 0)
        XCTAssertLessThanOrEqual(result.predictions.count, 5)
    }
    
    func testAllModelPresetsValid() throws {
        let presets: [PixelDataOptions] = [
            ModelPresets.yolo,
            ModelPresets.yolov8,
            ModelPresets.mobilenet,
            ModelPresets.resnet,
            ModelPresets.efficientnet,
            ModelPresets.clip
        ]
        
        for preset in presets {
            XCTAssertNotNil(preset.resize)
            XCTAssertNotNil(preset.colorFormat)
        }
    }
}
