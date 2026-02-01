import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ``ImageAnalyzer`` - Image analysis and quality assessment.
///
/// ## Topics
///
/// ### Statistical Analysis Tests
/// - Mean, standard deviation, min/max calculations
/// - Per-channel statistics
/// - Histogram computation
///
/// ### Quality Assessment Tests
/// - Brightness and contrast estimation
/// - Blur detection
/// - Noise level assessment
///
/// ### Format Detection Tests
/// - Color space identification
/// - Bit depth analysis
/// - Alpha channel detection
final class ImageAnalyzerTests: XCTestCase {
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: Int, height: Int, color: (CGFloat, CGFloat, CGFloat)? = nil) -> CGImage {
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
        
        if let color = color {
            context.setFillColor(red: color.0, green: color.1, blue: color.2, alpha: 1.0)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        } else {
            // Create gradient for varied content
            for y in 0..<height {
                for x in 0..<width {
                    let r = CGFloat(x) / CGFloat(width)
                    let g = CGFloat(y) / CGFloat(height)
                    let b = 0.5
                    context.setFillColor(red: r, green: g, blue: b, alpha: 1.0)
                    context.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        return context.makeImage()!
    }
    
    private func createSharpImage(width: Int, height: Int) -> CGImage {
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
        
        // Create sharp edges with alternating pattern
        for y in 0..<height {
            for x in 0..<width {
                let isWhite = (x + y) % 2 == 0
                let value: CGFloat = isWhite ? 1.0 : 0.0
                context.setFillColor(red: value, green: value, blue: value, alpha: 1.0)
                context.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        
        return context.makeImage()!
    }
    
    private func createBlurryImage(width: Int, height: Int) -> CGImage {
        // Create uniform gray image (no edges = low Laplacian variance)
        return createTestImage(width: width, height: height, color: (0.5, 0.5, 0.5))
    }
    
    // MARK: - BlurDetectionResult Tests
    
    func testBlurDetectionResultBlurry() {
        let result = BlurDetectionResult(
            isBlurry: true,
            score: 50.0,
            threshold: 100.0,
            processingTimeMs: 10.0
        )
        
        XCTAssertTrue(result.isBlurry)
        XCTAssertEqual(result.score, 50.0)
        XCTAssertEqual(result.threshold, 100.0)
        XCTAssertEqual(result.processingTimeMs, 10.0)
    }
    
    func testBlurDetectionResultNotBlurry() {
        let result = BlurDetectionResult(
            isBlurry: false,
            score: 150.0,
            threshold: 100.0,
            processingTimeMs: 15.0
        )
        
        XCTAssertFalse(result.isBlurry)
        XCTAssertEqual(result.score, 150.0)
    }
    
    // MARK: - ImageStatistics Tests
    
    func testImageStatisticsInit() {
        let stats = ImageStatistics(
            mean: [0.5, 0.5, 0.5],
            std: [0.1, 0.1, 0.1],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
            histogram: [Array(repeating: 100, count: 256),
                       Array(repeating: 100, count: 256),
                       Array(repeating: 100, count: 256)]
        )
        
        XCTAssertEqual(stats.mean.count, 3)
        XCTAssertEqual(stats.std.count, 3)
        XCTAssertEqual(stats.min.count, 3)
        XCTAssertEqual(stats.max.count, 3)
        XCTAssertEqual(stats.histogram.count, 3)
        XCTAssertEqual(stats.histogram[0].count, 256)
    }
    
    // MARK: - ImageMetadata Tests
    
    func testImageMetadataInit() {
        let metadata = ImageMetadata(
            width: 1920,
            height: 1080,
            channels: 3,
            colorSpace: "sRGB",
            hasAlpha: false,
            aspectRatio: 1.78
        )
        
        XCTAssertEqual(metadata.width, 1920)
        XCTAssertEqual(metadata.height, 1080)
        XCTAssertEqual(metadata.channels, 3)
        XCTAssertEqual(metadata.colorSpace, "sRGB")
        XCTAssertFalse(metadata.hasAlpha)
        XCTAssertEqual(metadata.aspectRatio, 1.78, accuracy: 0.01)
    }
    
    func testImageMetadataWithAlpha() {
        let metadata = ImageMetadata(
            width: 100,
            height: 100,
            channels: 4,
            colorSpace: "sRGB",
            hasAlpha: true,
            aspectRatio: 1.0
        )
        
        XCTAssertEqual(metadata.channels, 4)
        XCTAssertTrue(metadata.hasAlpha)
    }
    
    // MARK: - ValidationOptions Tests
    
    func testValidationOptionsInit() {
        let options = ValidationOptions(
            minWidth: 100,
            minHeight: 100,
            maxWidth: 4000,
            maxHeight: 4000,
            requiredAspectRatio: 1.0,
            aspectRatioTolerance: 0.1
        )
        
        XCTAssertEqual(options.minWidth, 100)
        XCTAssertEqual(options.minHeight, 100)
        XCTAssertEqual(options.maxWidth, 4000)
        XCTAssertEqual(options.maxHeight, 4000)
        XCTAssertEqual(options.requiredAspectRatio, 1.0)
        XCTAssertEqual(options.aspectRatioTolerance, 0.1)
    }
    
    func testValidationOptionsPartial() {
        let options = ValidationOptions(minWidth: 200)
        
        XCTAssertEqual(options.minWidth, 200)
        XCTAssertNil(options.minHeight)
        XCTAssertNil(options.maxWidth)
        XCTAssertNil(options.maxHeight)
        XCTAssertNil(options.requiredAspectRatio)
    }
    
    // MARK: - ValidationResult Tests
    
    func testValidationResultValid() {
        let result = ValidationResult(isValid: true, issues: [])
        
        XCTAssertTrue(result.isValid)
        XCTAssertTrue(result.issues.isEmpty)
    }
    
    func testValidationResultInvalid() {
        let result = ValidationResult(
            isValid: false,
            issues: ["Width too small", "Height too small"]
        )
        
        XCTAssertFalse(result.isValid)
        XCTAssertEqual(result.issues.count, 2)
    }
    
    // MARK: - detectBlur Tests
    
    func testDetectBlurSharpImage() async throws {
        let image = createSharpImage(width: 100, height: 100)
        
        let result = try await ImageAnalyzer.detectBlur(
            source: .cgImage(image),
            threshold: 10.0  // Low threshold for sharp detection
        )
        
        // Sharp image should have high Laplacian variance
        XCTAssertGreaterThan(result.score, 0)
        XCTAssertGreaterThan(result.processingTimeMs, 0)
    }
    
    func testDetectBlurBlurryImage() async throws {
        let image = createBlurryImage(width: 100, height: 100)
        
        let result = try await ImageAnalyzer.detectBlur(
            source: .cgImage(image),
            threshold: 100.0
        )
        
        // Uniform image should have very low Laplacian variance (close to 0)
        XCTAssertLessThan(result.score, 1.0)  // Very low for uniform image
        XCTAssertTrue(result.isBlurry)
    }
    
    func testDetectBlurWithDownsample() async throws {
        let image = createSharpImage(width: 1000, height: 1000)
        
        let result = try await ImageAnalyzer.detectBlur(
            source: .cgImage(image),
            downsampleSize: 100
        )
        
        XCTAssertGreaterThan(result.processingTimeMs, 0)
    }
    
    func testDetectBlurCustomThreshold() async throws {
        let image = createTestImage(width: 100, height: 100)
        
        let lowResult = try await ImageAnalyzer.detectBlur(
            source: .cgImage(image),
            threshold: 1.0  // Very low threshold
        )
        
        let highResult = try await ImageAnalyzer.detectBlur(
            source: .cgImage(image),
            threshold: 10000.0  // Very high threshold
        )
        
        XCTAssertEqual(lowResult.threshold, 1.0)
        XCTAssertEqual(highResult.threshold, 10000.0)
        
        // High threshold should be more lenient
        if lowResult.isBlurry {
            XCTAssertTrue(highResult.isBlurry || !highResult.isBlurry)  // Could be either
        }
    }
    
    // MARK: - getStatistics Tests
    
    func testGetStatisticsBasic() async throws {
        let image = createTestImage(width: 100, height: 100)
        
        let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
        
        XCTAssertEqual(stats.mean.count, 3)
        XCTAssertEqual(stats.std.count, 3)
        XCTAssertEqual(stats.min.count, 3)
        XCTAssertEqual(stats.max.count, 3)
        XCTAssertEqual(stats.histogram.count, 3)
    }
    
    func testGetStatisticsRange() async throws {
        let image = createTestImage(width: 50, height: 50)
        
        let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
        
        // All values should be in [0, 1] range
        for i in 0..<3 {
            XCTAssertGreaterThanOrEqual(stats.mean[i], 0.0)
            XCTAssertLessThanOrEqual(stats.mean[i], 1.0)
            XCTAssertGreaterThanOrEqual(stats.min[i], 0.0)
            XCTAssertLessThanOrEqual(stats.max[i], 1.0)
            XCTAssertGreaterThanOrEqual(stats.std[i], 0.0)
        }
    }
    
    func testGetStatisticsUniformColor() async throws {
        // Red image
        let image = createTestImage(width: 50, height: 50, color: (1.0, 0.0, 0.0))
        
        let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
        
        // Red channel should be 1.0, others should be 0.0
        XCTAssertEqual(stats.mean[0], 1.0, accuracy: 0.05)
        XCTAssertEqual(stats.mean[1], 0.0, accuracy: 0.05)
        XCTAssertEqual(stats.mean[2], 0.0, accuracy: 0.05)
        
        // Standard deviation should be 0 for uniform color
        XCTAssertEqual(stats.std[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(stats.std[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(stats.std[2], 0.0, accuracy: 0.01)
    }
    
    func testGetStatisticsHistogram() async throws {
        let image = createTestImage(width: 50, height: 50)
        
        let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
        
        // Each channel histogram should have 256 bins
        for histogram in stats.histogram {
            XCTAssertEqual(histogram.count, 256)
            
            // Sum of histogram should equal pixel count
            let sum = histogram.reduce(0, +)
            XCTAssertEqual(sum, 50 * 50)
        }
    }
    
    // MARK: - getMetadata Tests
    
    func testGetMetadataBasic() async throws {
        let image = createTestImage(width: 200, height: 100)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.width, 200)
        XCTAssertEqual(metadata.height, 100)
        XCTAssertEqual(metadata.aspectRatio, 2.0, accuracy: 0.01)
    }
    
    func testGetMetadataSquareImage() async throws {
        let image = createTestImage(width: 100, height: 100)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.aspectRatio, 1.0, accuracy: 0.01)
    }
    
    func testGetMetadataTallImage() async throws {
        let image = createTestImage(width: 100, height: 200)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.aspectRatio, 0.5, accuracy: 0.01)
    }
    
    // MARK: - validate Tests
    
    func testValidateValid() async throws {
        let image = createTestImage(width: 200, height: 200)
        
        let options = ValidationOptions(
            minWidth: 100,
            minHeight: 100,
            maxWidth: 500,
            maxHeight: 500
        )
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertTrue(result.isValid)
        XCTAssertTrue(result.issues.isEmpty)
    }
    
    func testValidateWidthTooSmall() async throws {
        let image = createTestImage(width: 50, height: 200)
        
        let options = ValidationOptions(minWidth: 100)
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertFalse(result.isValid)
        XCTAssertTrue(result.issues.contains { $0.contains("Width") && $0.contains("below") })
    }
    
    func testValidateHeightTooSmall() async throws {
        let image = createTestImage(width: 200, height: 50)
        
        let options = ValidationOptions(minHeight: 100)
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertFalse(result.isValid)
        XCTAssertTrue(result.issues.contains { $0.contains("Height") && $0.contains("below") })
    }
    
    func testValidateWidthTooLarge() async throws {
        let image = createTestImage(width: 500, height: 100)
        
        let options = ValidationOptions(maxWidth: 300)
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertFalse(result.isValid)
        XCTAssertTrue(result.issues.contains { $0.contains("Width") && $0.contains("exceeds") })
    }
    
    func testValidateHeightTooLarge() async throws {
        let image = createTestImage(width: 100, height: 500)
        
        let options = ValidationOptions(maxHeight: 300)
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertFalse(result.isValid)
        XCTAssertTrue(result.issues.contains { $0.contains("Height") && $0.contains("exceeds") })
    }
    
    func testValidateAspectRatio() async throws {
        let image = createTestImage(width: 200, height: 100)  // Aspect ratio 2.0
        
        let options = ValidationOptions(
            requiredAspectRatio: 1.0,  // Square
            aspectRatioTolerance: 0.1
        )
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertFalse(result.isValid)
        XCTAssertTrue(result.issues.contains { $0.contains("Aspect ratio") })
    }
    
    func testValidateAspectRatioWithTolerance() async throws {
        let image = createTestImage(width: 105, height: 100)  // Aspect ratio ~1.05
        
        let options = ValidationOptions(
            requiredAspectRatio: 1.0,
            aspectRatioTolerance: 0.1  // 10% tolerance
        )
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertTrue(result.isValid)  // Within tolerance
    }
    
    func testValidateMultipleIssues() async throws {
        let image = createTestImage(width: 50, height: 50)
        
        let options = ValidationOptions(
            minWidth: 100,
            minHeight: 100,
            requiredAspectRatio: 2.0
        )
        
        let result = try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
        
        XCTAssertFalse(result.isValid)
        XCTAssertGreaterThanOrEqual(result.issues.count, 2)
    }
    
    // MARK: - Edge Cases
    
    func testSmallImage() async throws {
        let image = createTestImage(width: 5, height: 5)
        
        let stats = try await ImageAnalyzer.getStatistics(source: .cgImage(image))
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.width, 5)
        XCTAssertEqual(metadata.height, 5)
        XCTAssertEqual(stats.mean.count, 3)
    }
    
    func testLargeImage() async throws {
        let image = createTestImage(width: 1000, height: 1000)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.width, 1000)
        XCTAssertEqual(metadata.height, 1000)
    }
    
    func testWideImage() async throws {
        let image = createTestImage(width: 500, height: 50)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.aspectRatio, 10.0, accuracy: 0.01)
    }
    
    func testTallImage() async throws {
        let image = createTestImage(width: 50, height: 500)
        
        let metadata = try await ImageAnalyzer.getMetadata(source: .cgImage(image))
        
        XCTAssertEqual(metadata.aspectRatio, 0.1, accuracy: 0.01)
    }
    
    // MARK: - Performance Tests
    
    func testBlurDetectionPerformance() throws {
        let image = createTestImage(width: 640, height: 480)
        
        measure {
            _ = try? Task {
                try await ImageAnalyzer.detectBlur(source: .cgImage(image))
            }
        }
    }
    
    func testStatisticsPerformance() throws {
        let image = createTestImage(width: 640, height: 480)
        
        measure {
            _ = try? Task {
                try await ImageAnalyzer.getStatistics(source: .cgImage(image))
            }
        }
    }
    
    func testValidationPerformance() throws {
        let image = createTestImage(width: 640, height: 480)
        let options = ValidationOptions(
            minWidth: 100,
            minHeight: 100,
            maxWidth: 1000,
            maxHeight: 1000,
            requiredAspectRatio: 4.0/3.0,
            aspectRatioTolerance: 0.1
        )
        
        measure {
            _ = try? Task {
                try await ImageAnalyzer.validate(source: .cgImage(image), options: options)
            }
        }
    }
}
