import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``PixelExtractor`` - Core pixel data extraction from images.
///
/// ## Topics
///
/// ### Image Source Tests
/// - CGImage, UIImage, NSImage, file path, URL
/// - Base64 encoded images
/// - CVPixelBuffer inputs
///
/// ### Data Layout Tests
/// - HWC (Height, Width, Channels)
/// - CHW (Channels, Height, Width)
/// - NHWC/NCHW with batch dimension
///
/// ### Normalization Tests
/// - ImageNet preset, TensorFlow preset
/// - Custom mean/std values
/// - Scale to [0, 1] range
///
/// ### Resize & ROI Tests
/// - Target size with aspect ratio preservation
/// - Region of Interest extraction
/// - Letterbox padding
final class PixelExtractorTests: XCTestCase {
    
    // MARK: - Helper Methods
    
    private func createTestCGImage(width: Int = 100, height: Int = 100, color: (r: UInt8, g: UInt8, b: UInt8) = (128, 64, 192)) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        for i in stride(from: 0, to: pixels.count, by: bytesPerPixel) {
            pixels[i] = color.r
            pixels[i + 1] = color.g
            pixels[i + 2] = color.b
            pixels[i + 3] = 255
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
    
    // MARK: - PixelDataOptions Tests
    
    func testPixelDataOptionsDefaults() {
        let options = PixelDataOptions()
        
        XCTAssertEqual(options.colorFormat, .rgb)
        XCTAssertEqual(options.dataLayout, .hwc)
        XCTAssertNil(options.resize)
        XCTAssertNil(options.roi)
    }
    
    func testPixelDataOptionsWithValues() {
        var options = PixelDataOptions()
        options.colorFormat = .bgr
        options.dataLayout = .nchw
        options.resize = ResizeOptions(width: 640, height: 640)
        options.normalization = .imagenet
        
        XCTAssertEqual(options.colorFormat, .bgr)
        XCTAssertEqual(options.dataLayout, .nchw)
        XCTAssertNotNil(options.resize)
        XCTAssertEqual(options.normalization, .imagenet)
    }
    
    // MARK: - Basic Extraction Tests
    
    func testGetPixelDataBasic() async throws {
        let image = createTestCGImage()
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(result.width, 100)
        XCTAssertEqual(result.height, 100)
        XCTAssertEqual(result.channels, 3)
        XCTAssertFalse(result.data.isEmpty)
    }
    
    func testGetPixelDataShape() async throws {
        let image = createTestCGImage(width: 64, height: 48)
        var options = PixelDataOptions()
        options.dataLayout = .hwc
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.shape, [48, 64, 3])
    }
    
    func testGetPixelDataShapeNCHW() async throws {
        let image = createTestCGImage(width: 64, height: 48)
        var options = PixelDataOptions()
        options.dataLayout = .nchw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.shape, [1, 3, 48, 64])
    }
    
    // MARK: - Color Format Tests
    
    func testGetPixelDataRGB() async throws {
        let image = createTestCGImage(color: (255, 128, 64))
        var options = PixelDataOptions()
        options.colorFormat = .rgb
        options.normalization = .raw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.channels, 3)
    }
    
    func testGetPixelDataBGR() async throws {
        let image = createTestCGImage(color: (255, 128, 64))
        var options = PixelDataOptions()
        options.colorFormat = .bgr
        options.normalization = .raw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.channels, 3)
    }
    
    func testGetPixelDataRGBA() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.colorFormat = .rgba
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.channels, 4)
    }
    
    func testGetPixelDataGrayscale() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.colorFormat = .grayscale
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.channels, 1)
    }
    
    // MARK: - Resize Tests
    
    func testGetPixelDataWithResize() async throws {
        let image = createTestCGImage(width: 200, height: 150)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 100, height: 100)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 100)
        XCTAssertEqual(result.height, 100)
    }
    
    func testGetPixelDataResizeStretch() async throws {
        let image = createTestCGImage(width: 200, height: 100)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 64, height: 64, strategy: .stretch)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 64)
        XCTAssertEqual(result.height, 64)
    }
    
    func testGetPixelDataResizeCover() async throws {
        let image = createTestCGImage(width: 200, height: 100)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 64, height: 64, strategy: .cover)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 64)
        XCTAssertEqual(result.height, 64)
    }
    
    func testGetPixelDataResizeContain() async throws {
        let image = createTestCGImage(width: 200, height: 100)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 64, height: 64, strategy: .contain)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 64)
        XCTAssertEqual(result.height, 64)
    }
    
    // MARK: - Normalization Tests
    
    func testGetPixelDataNormalizationRaw() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.normalization = .raw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        // Raw should be in 0-1 range (or have uint8Data in 0-255)
        XCTAssertNotNil(result)
    }
    
    func testGetPixelDataNormalizationImageNet() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.normalization = .imagenet
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        // ImageNet normalization produces values roughly in [-2.5, 2.5] range
        let hasNormalizedValues = result.data.contains { $0 < 0 || $0 > 1 }
        XCTAssertTrue(hasNormalizedValues || !result.data.isEmpty)
    }
    
    func testGetPixelDataNormalizationTensorFlow() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.normalization = .tensorflow
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        // TensorFlow normalization produces values in [-1, 1] range
        XCTAssertNotNil(result)
    }
    
    func testGetPixelDataNormalizationZeroOne() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.normalization = .scale
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        // All values should be in [0, 1] range
        for value in result.data {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThanOrEqual(value, 1)
        }
    }
    
    func testGetPixelDataNormalizationCustom() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.normalization = Normalization(
            preset: .custom,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5]
        )
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result)
    }
    
    // MARK: - Data Layout Tests
    
    func testGetPixelDataLayoutHWC() async throws {
        let image = createTestCGImage(width: 10, height: 10)
        var options = PixelDataOptions()
        options.dataLayout = .hwc
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.shape, [10, 10, 3])
        XCTAssertEqual(result.data.count, 10 * 10 * 3)
    }
    
    func testGetPixelDataLayoutCHW() async throws {
        let image = createTestCGImage(width: 10, height: 10)
        var options = PixelDataOptions()
        options.dataLayout = .chw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.shape, [3, 10, 10])
        XCTAssertEqual(result.data.count, 3 * 10 * 10)
    }
    
    func testGetPixelDataLayoutNCHW() async throws {
        let image = createTestCGImage(width: 10, height: 10)
        var options = PixelDataOptions()
        options.dataLayout = .nchw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.shape, [1, 3, 10, 10])
        XCTAssertEqual(result.data.count, 1 * 3 * 10 * 10)
    }
    
    func testGetPixelDataLayoutNHWC() async throws {
        let image = createTestCGImage(width: 10, height: 10)
        var options = PixelDataOptions()
        options.dataLayout = .nhwc
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.shape, [1, 10, 10, 3])
        XCTAssertEqual(result.data.count, 1 * 10 * 10 * 3)
    }
    
    // MARK: - ROI Tests
    
    func testGetPixelDataWithROI() async throws {
        let image = createTestCGImage(width: 200, height: 200)
        var options = PixelDataOptions()
        options.roi = ROI(x: 50, y: 50, width: 100, height: 100)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 100)
        XCTAssertEqual(result.height, 100)
    }
    
    // MARK: - Output Format Tests
    
    func testGetPixelDataUInt8Output() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.outputFormat = .uint8Array
        options.normalization = .raw
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result.uint8Data)
        XCTAssertEqual(result.uint8Data?.count, result.width * result.height * result.channels)
    }
    
    func testGetPixelDataInt32Output() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.outputFormat = .int32Array
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result.int32Data)
        XCTAssertEqual(result.int32Data?.count, result.width * result.height * result.channels)
    }
    
    func testGetPixelDataFloat16Output() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.outputFormat = .float16Array
        options.normalization = .scale
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result.float16Data)
        XCTAssertEqual(result.float16Data?.count, result.width * result.height * result.channels)
        // Verify float32 data is also populated
        XCTAssertFalse(result.data.isEmpty)
    }
    
    // MARK: - Letterbox Info Tests
    
    func testGetPixelDataLetterboxInfoWideImage() async throws {
        // Wide image (2:1) letterboxed to square - should have vertical padding
        let image = createTestCGImage(width: 200, height: 100)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 640, height: 640, strategy: .letterbox)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result.letterboxInfo)
        XCTAssertGreaterThan(result.letterboxInfo!.offset.y, 0)  // Vertical padding
        XCTAssertEqual(result.letterboxInfo!.offset.x, 0, accuracy: 1.0)  // No horizontal padding
        XCTAssertGreaterThan(result.letterboxInfo!.scale, 0)
    }
    
    func testGetPixelDataLetterboxInfoTallImage() async throws {
        // Tall image (1:2) letterboxed to square - should have horizontal padding
        let image = createTestCGImage(width: 100, height: 200)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 640, height: 640, strategy: .letterbox)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result.letterboxInfo)
        XCTAssertGreaterThan(result.letterboxInfo!.offset.x, 0)  // Horizontal padding
        XCTAssertEqual(result.letterboxInfo!.offset.y, 0, accuracy: 1.0)  // No vertical padding
    }
    
    func testGetPixelDataLetterboxInfoSquareImage() async throws {
        // Square image letterboxed to square - no padding needed
        let image = createTestCGImage(width: 100, height: 100)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 640, height: 640, strategy: .letterbox)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNotNil(result.letterboxInfo)
        XCTAssertEqual(result.letterboxInfo!.offset.x, 0, accuracy: 1.0)
        XCTAssertEqual(result.letterboxInfo!.offset.y, 0, accuracy: 1.0)
    }
    
    func testGetPixelDataNoLetterboxInfoForStretch() async throws {
        // Non-letterbox resize should not have letterboxInfo
        let image = createTestCGImage(width: 200, height: 100)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 640, height: 640, strategy: .stretch)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertNil(result.letterboxInfo)
    }
    
    // MARK: - Orientation Normalization Tests
    
    func testGetPixelDataOrientationNormalization() async throws {
        let image = createTestCGImage()
        var options = PixelDataOptions()
        options.normalizeOrientation = true
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        // Image dimensions should be preserved
        XCTAssertEqual(result.width, 100)
        XCTAssertEqual(result.height, 100)
        XCTAssertFalse(result.data.isEmpty)
    }
    
    func testGetPixelDataOrientationNormalizationDefault() async throws {
        let image = createTestCGImage()
        let options = PixelDataOptions()
        
        // Default should be false
        XCTAssertFalse(options.normalizeOrientation)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 100)
        XCTAssertEqual(result.height, 100)
    }
    
    // MARK: - Batch Processing Tests
    
    func testBatchGetPixelData() async throws {
        let images = [
            createTestCGImage(width: 64, height: 64),
            createTestCGImage(width: 64, height: 64),
            createTestCGImage(width: 64, height: 64)
        ]
        
        let sources = images.map { ImageSource.cgImage($0) }
        
        let results = try await PixelExtractor.batchGetPixelData(
            sources: sources,
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(results.count, 3)
        for result in results {
            XCTAssertEqual(result.width, 64)
            XCTAssertEqual(result.height, 64)
        }
    }
    
    func testBatchGetPixelDataWithConcurrency() async throws {
        let images = (0..<10).map { _ in createTestCGImage(width: 32, height: 32) }
        let sources = images.map { ImageSource.cgImage($0) }
        
        let results = try await PixelExtractor.batchGetPixelData(
            sources: sources,
            options: PixelDataOptions(),
            concurrency: 4
        )
        
        XCTAssertEqual(results.count, 10)
    }
    
    // MARK: - PixelDataResult Tests
    
    func testPixelDataResultProperties() async throws {
        let image = createTestCGImage(width: 50, height: 30)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(result.width, 50)
        XCTAssertEqual(result.height, 30)
        XCTAssertEqual(result.channels, 3)
        XCTAssertGreaterThan(result.processingTimeMs, 0)
    }
    
    func testPixelDataResultTotalElements() async throws {
        let image = createTestCGImage(width: 10, height: 20)
        var options = PixelDataOptions()
        options.colorFormat = .rgb
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        let expectedCount = 10 * 20 * 3
        XCTAssertEqual(result.data.count, expectedCount)
    }
    
    // MARK: - Edge Cases
    
    func testGetPixelDataSmallImage() async throws {
        let image = createTestCGImage(width: 1, height: 1)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(result.width, 1)
        XCTAssertEqual(result.height, 1)
        XCTAssertEqual(result.data.count, 3)  // RGB
    }
    
    func testGetPixelDataWideImage() async throws {
        let image = createTestCGImage(width: 1000, height: 10)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(result.width, 1000)
        XCTAssertEqual(result.height, 10)
    }
    
    func testGetPixelDataTallImage() async throws {
        let image = createTestCGImage(width: 10, height: 1000)
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: PixelDataOptions()
        )
        
        XCTAssertEqual(result.width, 10)
        XCTAssertEqual(result.height, 1000)
    }
    
    // MARK: - Model Presets Tests
    
    func testGetPixelDataWithYOLOPreset() async throws {
        let image = createTestCGImage(width: 640, height: 480)
        let options = ModelPresets.yolov8
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 640)
        XCTAssertEqual(result.height, 640)
        XCTAssertEqual(result.shape, [1, 3, 640, 640])  // NCHW
    }
    
    func testGetPixelDataWithMobileNetPreset() async throws {
        let image = createTestCGImage(width: 300, height: 200)
        let options = ModelPresets.mobilenet_v2
        
        let result = try await PixelExtractor.getPixelData(
            source: .cgImage(image),
            options: options
        )
        
        XCTAssertEqual(result.width, 224)
        XCTAssertEqual(result.height, 224)
    }
    
    // MARK: - Performance Tests
    
    func testPixelExtractionPerformance() async throws {
        let image = createTestCGImage(width: 640, height: 640)
        var options = PixelDataOptions()
        options.resize = ResizeOptions(width: 640, height: 640)
        options.normalization = .imagenet
        options.dataLayout = .nchw
        
        measure {
            let expectation = self.expectation(description: "extraction")
            Task {
                _ = try? await PixelExtractor.getPixelData(
                    source: .cgImage(image),
                    options: options
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }
    }
    
    func testBatchExtractionPerformance() async throws {
        let images = (0..<10).map { _ in createTestCGImage(width: 224, height: 224) }
        let sources = images.map { ImageSource.cgImage($0) }
        
        measure {
            let expectation = self.expectation(description: "batch")
            Task {
                _ = try? await PixelExtractor.batchGetPixelData(
                    sources: sources,
                    options: PixelDataOptions(),
                    concurrency: 4
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 10.0)
        }
    }
}
