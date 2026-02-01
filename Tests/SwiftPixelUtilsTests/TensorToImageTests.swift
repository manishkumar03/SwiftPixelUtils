import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``TensorToImage`` - Convert tensor data back to images.
///
/// ## Topics
///
/// ### Options Tests
/// - Default values for channels, layout, normalization
/// - Output format selection (PNG, JPEG)
/// - JPEG quality setting
///
/// ### Denormalization Tests
/// - Reverse ImageNet normalization
/// - Custom mean/std reversal
/// - Scale from [0, 1] to [0, 255]
///
/// ### Output Format Tests
/// - CGImage generation
/// - PNG/JPEG data encoding
/// - Base64 string output
///
/// ### Layout Handling Tests
/// - HWC and CHW input support
/// - Grayscale (1 channel) images
final class TensorToImageTests: XCTestCase {
    
    // MARK: - TensorToImageOptions Tests
    
    func testTensorToImageOptionsDefaults() {
        let options = TensorToImageOptions()
        
        XCTAssertEqual(options.channels, 3)
        XCTAssertEqual(options.dataLayout, .hwc)
        XCTAssertTrue(options.denormalize)
        XCTAssertEqual(options.mean, [0.485, 0.456, 0.406])
        XCTAssertEqual(options.std, [0.229, 0.224, 0.225])
        XCTAssertEqual(options.outputFormat, .png)
        XCTAssertEqual(options.jpegQuality, 90)
    }
    
    func testTensorToImageOptionsCustom() {
        let options = TensorToImageOptions(
            channels: 1,
            dataLayout: .chw,
            denormalize: false,
            mean: [0.5],
            std: [0.5],
            outputFormat: .jpeg,
            jpegQuality: 75
        )
        
        XCTAssertEqual(options.channels, 1)
        XCTAssertEqual(options.dataLayout, .chw)
        XCTAssertFalse(options.denormalize)
        XCTAssertEqual(options.mean, [0.5])
        XCTAssertEqual(options.std, [0.5])
        XCTAssertEqual(options.outputFormat, .jpeg)
        XCTAssertEqual(options.jpegQuality, 75)
    }
    
    // MARK: - ImageOutputFormat Tests
    
    func testImageOutputFormatValues() {
        XCTAssertEqual(ImageOutputFormat.png.rawValue, "png")
        XCTAssertEqual(ImageOutputFormat.jpeg.rawValue, "jpeg")
    }
    
    // MARK: - Basic Conversion Tests
    
    func testConvertBasicRGB() async throws {
        // Create simple 2x2 RGB data (in HWC format)
        let data: [Float] = [
            0.5, 0.5, 0.5,  // Pixel (0,0)
            0.5, 0.5, 0.5,  // Pixel (1,0)
            0.5, 0.5, 0.5,  // Pixel (0,1)
            0.5, 0.5, 0.5   // Pixel (1,1)
        ]
        
        let options = TensorToImageOptions(
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 2,
            height: 2,
            options: options
        )
        
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
        XCTAssertFalse(result.imageBase64.isEmpty)
    }
    
    func testConvertSyncBasic() throws {
        let data: [Float] = [Float](repeating: 0.5, count: 10 * 10 * 3)
        
        let result = try TensorToImage.convertSync(
            data: data,
            width: 10,
            height: 10,
            options: TensorToImageOptions(denormalize: false)
        )
        
        XCTAssertEqual(result.width, 10)
        XCTAssertEqual(result.height, 10)
        XCTAssertNotNil(result.cgImage)
    }
    
    // MARK: - Denormalization Tests
    
    func testConvertWithDenormalization() async throws {
        // Normalized ImageNet data
        let normalizedData = [Float](repeating: 0.0, count: 4 * 4 * 3)
        
        let options = TensorToImageOptions(
            denormalize: true,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225]
        )
        
        let result = try await TensorToImage.convert(
            data: normalizedData,
            width: 4,
            height: 4,
            options: options
        )
        
        XCTAssertEqual(result.width, 4)
        XCTAssertEqual(result.height, 4)
    }
    
    func testConvertWithoutDenormalization() async throws {
        // Data already in [0, 1] range
        let data = [Float](repeating: 0.5, count: 4 * 4 * 3)
        
        let options = TensorToImageOptions(
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 4,
            height: 4,
            options: options
        )
        
        XCTAssertNotNil(result.cgImage)
    }
    
    // MARK: - Data Layout Tests
    
    func testConvertHWCLayout() async throws {
        // HWC: [height, width, channels]
        let data = [Float](repeating: 0.5, count: 8 * 8 * 3)
        
        let options = TensorToImageOptions(
            dataLayout: .hwc,
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 8,
            height: 8,
            options: options
        )
        
        XCTAssertEqual(result.width, 8)
        XCTAssertEqual(result.height, 8)
    }
    
    func testConvertCHWLayout() async throws {
        // CHW: [channels, height, width]
        let data = [Float](repeating: 0.5, count: 3 * 8 * 8)
        
        let options = TensorToImageOptions(
            dataLayout: .chw,
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 8,
            height: 8,
            options: options
        )
        
        XCTAssertEqual(result.width, 8)
        XCTAssertEqual(result.height, 8)
    }
    
    func testConvertNCHWLayout() async throws {
        // NCHW: [batch, channels, height, width]
        let data = [Float](repeating: 0.5, count: 1 * 3 * 8 * 8)
        
        let options = TensorToImageOptions(
            dataLayout: .nchw,
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 8,
            height: 8,
            options: options
        )
        
        XCTAssertEqual(result.width, 8)
        XCTAssertEqual(result.height, 8)
    }
    
    // MARK: - Channel Count Tests
    
    func testConvertGrayscale() async throws {
        let data = [Float](repeating: 0.5, count: 10 * 10 * 1)
        
        let options = TensorToImageOptions(
            channels: 1,
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 10,
            height: 10,
            options: options
        )
        
        XCTAssertEqual(result.width, 10)
        XCTAssertEqual(result.height, 10)
    }
    
    func testConvertRGBA() async throws {
        let data = [Float](repeating: 0.5, count: 10 * 10 * 4)
        
        let options = TensorToImageOptions(
            channels: 4,
            denormalize: false
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 10,
            height: 10,
            options: options
        )
        
        XCTAssertEqual(result.width, 10)
        XCTAssertEqual(result.height, 10)
    }
    
    // MARK: - Output Format Tests
    
    func testConvertToPNG() async throws {
        let data = [Float](repeating: 0.5, count: 8 * 8 * 3)
        
        let options = TensorToImageOptions(
            denormalize: false,
            outputFormat: .png
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 8,
            height: 8,
            options: options
        )
        
        // PNG base64 starts with specific characters
        XCTAssertFalse(result.imageBase64.isEmpty)
    }
    
    func testConvertToJPEG() async throws {
        let data = [Float](repeating: 0.5, count: 8 * 8 * 3)
        
        let options = TensorToImageOptions(
            denormalize: false,
            outputFormat: .jpeg,
            jpegQuality: 80
        )
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 8,
            height: 8,
            options: options
        )
        
        XCTAssertFalse(result.imageBase64.isEmpty)
    }
    
    func testJPEGQualityAffectsSize() async throws {
        let data = [Float](repeating: 0.5, count: 100 * 100 * 3)
        
        let highQualityOptions = TensorToImageOptions(
            denormalize: false,
            outputFormat: .jpeg,
            jpegQuality: 100
        )
        
        let lowQualityOptions = TensorToImageOptions(
            denormalize: false,
            outputFormat: .jpeg,
            jpegQuality: 10
        )
        
        let highQualityResult = try await TensorToImage.convert(
            data: data,
            width: 100,
            height: 100,
            options: highQualityOptions
        )
        
        let lowQualityResult = try await TensorToImage.convert(
            data: data,
            width: 100,
            height: 100,
            options: lowQualityOptions
        )
        
        // Lower quality should produce smaller base64 string
        XCTAssertLessThan(lowQualityResult.imageBase64.count, highQualityResult.imageBase64.count)
    }
    
    // MARK: - Error Handling Tests
    
    func testConvertInvalidDataSize() async {
        let data = [Float](repeating: 0.5, count: 10)  // Too small
        
        do {
            _ = try await TensorToImage.convert(
                data: data,
                width: 100,
                height: 100,
                options: TensorToImageOptions()
            )
            XCTFail("Should throw error for invalid data size")
        } catch {
            if case PixelUtilsError.invalidOptions = error {
                // Expected
            } else {
                XCTFail("Should throw invalidOptions error")
            }
        }
    }
    
    // MARK: - TensorToImageResult Tests
    
    func testResultProperties() async throws {
        let data = [Float](repeating: 0.5, count: 32 * 24 * 3)
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 32,
            height: 24,
            options: TensorToImageOptions(denormalize: false)
        )
        
        XCTAssertEqual(result.width, 32)
        XCTAssertEqual(result.height, 24)
        XCTAssertNotNil(result.cgImage)
        XCTAssertEqual(result.cgImage.width, 32)
        XCTAssertEqual(result.cgImage.height, 24)
        XCTAssertGreaterThan(result.processingTimeMs, 0)
        XCTAssertFalse(result.imageBase64.isEmpty)
    }
    
    // MARK: - Roundtrip Tests
    
    func testRoundtripPreservesShape() async throws {
        // Create a simple gradient pattern
        var data = [Float]()
        for y in 0..<16 {
            for x in 0..<16 {
                let r = Float(x) / 16.0
                let g = Float(y) / 16.0
                let b = Float(x + y) / 32.0
                data.append(contentsOf: [r, g, b])
            }
        }
        
        let result = try await TensorToImage.convert(
            data: data,
            width: 16,
            height: 16,
            options: TensorToImageOptions(denormalize: false)
        )
        
        XCTAssertEqual(result.width, 16)
        XCTAssertEqual(result.height, 16)
    }
    
    // MARK: - Performance Tests
    
    func testConversionPerformance() async throws {
        let data = [Float](repeating: 0.5, count: 640 * 640 * 3)
        let options = TensorToImageOptions(
            denormalize: true,
            outputFormat: .png
        )
        
        measure {
            let expectation = self.expectation(description: "convert")
            Task {
                _ = try? await TensorToImage.convert(
                    data: data,
                    width: 640,
                    height: 640,
                    options: options
                )
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }
    }
}
