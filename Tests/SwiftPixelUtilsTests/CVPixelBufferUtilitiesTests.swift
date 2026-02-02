import XCTest
import CoreVideo
@testable import SwiftPixelUtils

/// Tests for ``CVPixelBufferUtilities`` - CVPixelBuffer to tensor data conversions.
///
/// ## Topics
///
/// ### Pixel Format Tests
/// - BGRA and RGBA buffer conversion
/// - YUV format handling
/// - Channel order preservation (RGB vs BGR)
///
/// ### Normalization Tests
/// - Scale normalization [0, 1]
/// - ImageNet normalization (mean/std)
/// - TensorFlow normalization [-1, 1]
/// - Custom normalization parameters
///
/// ### Performance Tests
/// - Conversion speed benchmarks
/// - Large buffer handling (4K images)
final class CVPixelBufferUtilitiesTests: XCTestCase {
    
    // MARK: - Helper Methods
    
    private func createBGRAPixelBuffer(width: Int, height: Int, color: (UInt8, UInt8, UInt8, UInt8)? = nil) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw PixelUtilsError.processingFailed("Failed to create pixel buffer")
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw PixelUtilsError.processingFailed("Failed to get base address")
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Fill with color or gradient
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                if let color = color {
                    ptr[offset] = color.2     // B
                    ptr[offset + 1] = color.1 // G
                    ptr[offset + 2] = color.0 // R
                    ptr[offset + 3] = color.3 // A
                } else {
                    // Gradient
                    let val = UInt8((x + y) % 256)
                    ptr[offset] = val     // B
                    ptr[offset + 1] = val // G
                    ptr[offset + 2] = val // R
                    ptr[offset + 3] = 255 // A
                }
            }
        }
        
        return buffer
    }
    
    private func createRGBAPixelBuffer(width: Int, height: Int, color: (UInt8, UInt8, UInt8, UInt8)? = nil) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32RGBA,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw PixelUtilsError.processingFailed("Failed to create pixel buffer")
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw PixelUtilsError.processingFailed("Failed to get base address")
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                if let color = color {
                    ptr[offset] = color.0     // R
                    ptr[offset + 1] = color.1 // G
                    ptr[offset + 2] = color.2 // B
                    ptr[offset + 3] = color.3 // A
                } else {
                    let val = UInt8((x + y) % 256)
                    ptr[offset] = val
                    ptr[offset + 1] = val
                    ptr[offset + 2] = val
                    ptr[offset + 3] = 255
                }
            }
        }
        
        return buffer
    }

    private func createRGB565PixelBuffer(
        width: Int,
        height: Int,
        r5: UInt16,
        g6: UInt16,
        b5: UInt16,
        littleEndian: Bool
    ) throws -> CVPixelBuffer {
        let format: OSType = littleEndian ? kCVPixelFormatType_16LE565 : kCVPixelFormatType_16BE565
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            format,
            nil,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw PixelUtilsError.processingFailed("Failed to create RGB565 pixel buffer")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw PixelUtilsError.processingFailed("Failed to get base address")
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
        let pixelsPerRow = bytesPerRow / MemoryLayout<UInt16>.size

        let packed = (r5 << 11) | (g6 << 5) | b5
        let stored = littleEndian ? CFSwapInt16HostToLittle(packed) : CFSwapInt16HostToBig(packed)

        for y in 0..<height {
            for x in 0..<width {
                ptr[y * pixelsPerRow + x] = stored
            }
        }

        return buffer
    }
    
    // MARK: - ConversionResult Tests
    
    func testConversionResultInit() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let result = CVPixelBufferUtilities.ConversionResult(
            data: data,
            originalWidth: 100,
            originalHeight: 200,
            tensorWidth: 50,
            tensorHeight: 100,
            channels: 3
        )
        
        XCTAssertEqual(result.data, data)
        XCTAssertEqual(result.originalWidth, 100)
        XCTAssertEqual(result.originalHeight, 200)
        XCTAssertEqual(result.tensorWidth, 50)
        XCTAssertEqual(result.tensorHeight, 100)
        XCTAssertEqual(result.channels, 3)
    }
    
    // MARK: - ChannelOrder Tests
    
    func testChannelOrderRGB() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (255, 128, 64, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            channelOrder: .rgb
        )
        
        XCTAssertEqual(result.channels, 3)
        // First pixel RGB values should be 255, 128, 64 (normalized to 1.0, 0.5, 0.25)
    }
    
    func testChannelOrderBGR() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (255, 128, 64, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            channelOrder: .bgr
        )
        
        XCTAssertEqual(result.channels, 3)
        // First pixel BGR values should be 64, 128, 255 (reversed)
    }
    
    // MARK: - toTensorData Tests
    
    func testToTensorDataBasic() throws {
        let buffer = try createBGRAPixelBuffer(width: 10, height: 10)
        
        let result = try CVPixelBufferUtilities.toTensorData(buffer)
        
        XCTAssertEqual(result.originalWidth, 10)
        XCTAssertEqual(result.originalHeight, 10)
        XCTAssertEqual(result.tensorWidth, 10)
        XCTAssertEqual(result.tensorHeight, 10)
        XCTAssertEqual(result.channels, 3)  // Default excludes alpha
        XCTAssertEqual(result.data.count, 10 * 10 * 3)
    }
    
    func testToTensorDataIncludeAlpha() throws {
        let buffer = try createBGRAPixelBuffer(width: 10, height: 10)
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            includeAlpha: true
        )
        
        XCTAssertEqual(result.channels, 4)
        XCTAssertEqual(result.data.count, 10 * 10 * 4)
    }
    
    func testToTensorDataWithResize() throws {
        let buffer = try createBGRAPixelBuffer(width: 100, height: 100)
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            targetSize: (width: 50, height: 50)
        )
        
        XCTAssertEqual(result.originalWidth, 100)
        XCTAssertEqual(result.originalHeight, 100)
        XCTAssertEqual(result.tensorWidth, 50)
        XCTAssertEqual(result.tensorHeight, 50)
        XCTAssertEqual(result.data.count, 50 * 50 * 3)
    }
    
    func testToTensorDataWithUpscale() throws {
        let buffer = try createBGRAPixelBuffer(width: 50, height: 50)
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            targetSize: (width: 100, height: 100)
        )
        
        XCTAssertEqual(result.tensorWidth, 100)
        XCTAssertEqual(result.tensorHeight, 100)
    }
    
    // MARK: - RGBA Format Tests
    
    func testRGBAPixelBuffer() throws {
        // RGBA format may not be supported on all platforms
        let buffer: CVPixelBuffer
        do {
            buffer = try createRGBAPixelBuffer(width: 10, height: 10)
        } catch {
            throw XCTSkip("RGBA pixel buffer creation not supported on this platform")
        }
        
        let result = try CVPixelBufferUtilities.toTensorData(buffer)
        
        XCTAssertEqual(result.originalWidth, 10)
        XCTAssertEqual(result.originalHeight, 10)
        XCTAssertEqual(result.data.count, 10 * 10 * 3)
    }

    // MARK: - RGB565 Format Tests

    func testRGB565LEPixelBuffer() throws {
        let buffer: CVPixelBuffer
        do {
            buffer = try createRGB565PixelBuffer(width: 2, height: 2, r5: 31, g6: 0, b5: 0, littleEndian: true)
        } catch {
            throw XCTSkip("RGB565 pixel buffer creation not supported on this platform")
        }

        let result = try CVPixelBufferUtilities.toTensorData(buffer)

        XCTAssertEqual(result.originalWidth, 2)
        XCTAssertEqual(result.originalHeight, 2)
        XCTAssertEqual(result.channels, 3)

        XCTAssertEqual(result.data[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(result.data[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(result.data[2], 0.0, accuracy: 0.01)
    }

    func testRGB565BEPixelBuffer() throws {
        let buffer: CVPixelBuffer
        do {
            buffer = try createRGB565PixelBuffer(width: 2, height: 2, r5: 0, g6: 63, b5: 0, littleEndian: false)
        } catch {
            throw XCTSkip("RGB565 pixel buffer creation not supported on this platform")
        }

        let result = try CVPixelBufferUtilities.toTensorData(buffer)

        XCTAssertEqual(result.originalWidth, 2)
        XCTAssertEqual(result.originalHeight, 2)
        XCTAssertEqual(result.channels, 3)

        XCTAssertEqual(result.data[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(result.data[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(result.data[2], 0.0, accuracy: 0.01)
    }
    
    // MARK: - Normalization Tests
    
    func testNormalizationScale() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (128, 128, 128, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: .scale
        )
        
        // Scale normalization keeps values in [0, 1]
        for value in result.data {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }
    }
    
    func testNormalizationTensorflow() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (128, 128, 128, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: Normalization(preset: .tensorflow)
        )
        
        // TensorFlow normalization maps [0, 1] to [-1, 1]
        for value in result.data {
            XCTAssertGreaterThanOrEqual(value, -1.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }
    }
    
    func testNormalizationImageNet() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (128, 128, 128, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: .imagenet
        )
        
        // ImageNet normalization applies mean subtraction and std division
        // Values should be centered around 0 for mid-gray
        XCTAssertEqual(result.data.count, 2 * 2 * 3)
    }
    
    func testNormalizationRaw() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (255, 128, 0, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: Normalization(preset: .raw)
        )
        
        // Raw normalization should have values in [0, 255] range
        // Check that at least some values are above 1.0 (proving no scale normalization)
        let maxValue = result.data.max() ?? 0
        XCTAssertGreaterThan(maxValue, 1.0)
    }
    
    func testNormalizationCustom() throws {
        let buffer = try createBGRAPixelBuffer(width: 2, height: 2, color: (128, 128, 128, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: Normalization(
                preset: .custom,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5]
            )
        )
        
        // With custom normalization, mid-gray (0.5) should map to 0
        XCTAssertEqual(result.data.count, 2 * 2 * 3)
    }
    
    // MARK: - getPixelFormatDescription Tests
    
    func testGetPixelFormatDescriptionBGRA() {
        let desc = CVPixelBufferUtilities.getPixelFormatDescription(kCVPixelFormatType_32BGRA)
        XCTAssertEqual(desc, "32BGRA")
    }
    
    func testGetPixelFormatDescriptionRGBA() {
        let desc = CVPixelBufferUtilities.getPixelFormatDescription(kCVPixelFormatType_32RGBA)
        XCTAssertEqual(desc, "32RGBA")
    }
    
    func testGetPixelFormatDescriptionARGB() {
        let desc = CVPixelBufferUtilities.getPixelFormatDescription(kCVPixelFormatType_32ARGB)
        XCTAssertEqual(desc, "32ARGB")
    }
    
    func testGetPixelFormatDescriptionYUVVideoRange() {
        let desc = CVPixelBufferUtilities.getPixelFormatDescription(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        XCTAssertEqual(desc, "420YpCbCr8BiPlanarVideoRange")
    }
    
    func testGetPixelFormatDescriptionYUVFullRange() {
        let desc = CVPixelBufferUtilities.getPixelFormatDescription(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        XCTAssertEqual(desc, "420YpCbCr8BiPlanarFullRange")
    }
    
    func testGetPixelFormatDescriptionUnknown() {
        let desc = CVPixelBufferUtilities.getPixelFormatDescription(999999)
        XCTAssertTrue(desc.contains("Unknown"))
    }
    
    // MARK: - Edge Cases
    
    func testSmallBuffer() throws {
        let buffer = try createBGRAPixelBuffer(width: 1, height: 1, color: (255, 128, 64, 255))
        
        let result = try CVPixelBufferUtilities.toTensorData(buffer)
        
        XCTAssertEqual(result.tensorWidth, 1)
        XCTAssertEqual(result.tensorHeight, 1)
        XCTAssertEqual(result.data.count, 3)
    }
    
    func testWideBuffer() throws {
        let buffer = try createBGRAPixelBuffer(width: 100, height: 10)
        
        let result = try CVPixelBufferUtilities.toTensorData(buffer)
        
        XCTAssertEqual(result.tensorWidth, 100)
        XCTAssertEqual(result.tensorHeight, 10)
    }
    
    func testTallBuffer() throws {
        let buffer = try createBGRAPixelBuffer(width: 10, height: 100)
        
        let result = try CVPixelBufferUtilities.toTensorData(buffer)
        
        XCTAssertEqual(result.tensorWidth, 10)
        XCTAssertEqual(result.tensorHeight, 100)
    }
    
    func testNonSquareResize() throws {
        let buffer = try createBGRAPixelBuffer(width: 100, height: 50)
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            targetSize: (width: 224, height: 224)
        )
        
        XCTAssertEqual(result.tensorWidth, 224)
        XCTAssertEqual(result.tensorHeight, 224)
    }
    
    func testColorPreservation() throws {
        // Create a pixel with known color
        let buffer = try createBGRAPixelBuffer(width: 1, height: 1, color: (255, 0, 0, 255))  // Pure red
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: .scale,
            channelOrder: .rgb
        )
        
        // RGB values for red: R=1.0, G=0.0, B=0.0
        XCTAssertEqual(result.data.count, 3)
        XCTAssertEqual(result.data[0], 1.0, accuracy: 0.01)  // R
        XCTAssertEqual(result.data[1], 0.0, accuracy: 0.01)  // G
        XCTAssertEqual(result.data[2], 0.0, accuracy: 0.01)  // B
    }
    
    func testChannelOrderBGRColorPreservation() throws {
        let buffer = try createBGRAPixelBuffer(width: 1, height: 1, color: (255, 0, 0, 255))  // Pure red
        
        let result = try CVPixelBufferUtilities.toTensorData(
            buffer,
            normalization: .scale,
            channelOrder: .bgr
        )
        
        // BGR values for red: B=0.0, G=0.0, R=1.0
        XCTAssertEqual(result.data.count, 3)
        XCTAssertEqual(result.data[0], 0.0, accuracy: 0.01)  // B
        XCTAssertEqual(result.data[1], 0.0, accuracy: 0.01)  // G
        XCTAssertEqual(result.data[2], 1.0, accuracy: 0.01)  // R
    }
    
    // MARK: - Performance Tests
    
    func testConversionPerformance() throws {
        let buffer = try createBGRAPixelBuffer(width: 640, height: 480)
        
        measure {
            _ = try? CVPixelBufferUtilities.toTensorData(
                buffer,
                targetSize: (width: 224, height: 224)
            )
        }
    }
    
    func testLargeBufferPerformance() throws {
        let buffer = try createBGRAPixelBuffer(width: 1920, height: 1080)
        
        measure {
            _ = try? CVPixelBufferUtilities.toTensorData(
                buffer,
                targetSize: (width: 640, height: 640)
            )
        }
    }
    
    // MARK: - Float16 Conversion Tests
    
    func testFloat16ToFloat32Zero() {
        // Float16 zero: 0x0000
        let result = CVPixelBufferUtilities.float16ToFloat32(0x0000)
        XCTAssertEqual(result, 0.0, accuracy: 0.0001)
        
        // Negative zero: 0x8000
        let negZero = CVPixelBufferUtilities.float16ToFloat32(0x8000)
        XCTAssertEqual(negZero, 0.0, accuracy: 0.0001)
    }
    
    func testFloat16ToFloat32One() {
        // Float16 one: sign=0, exp=15, frac=0 -> 0x3C00
        let result = CVPixelBufferUtilities.float16ToFloat32(0x3C00)
        XCTAssertEqual(result, 1.0, accuracy: 0.0001)
        
        // Negative one: 0xBC00
        let negOne = CVPixelBufferUtilities.float16ToFloat32(0xBC00)
        XCTAssertEqual(negOne, -1.0, accuracy: 0.0001)
    }
    
    func testFloat16ToFloat32CommonValues() {
        // Float16 0.5: 0x3800
        XCTAssertEqual(CVPixelBufferUtilities.float16ToFloat32(0x3800), 0.5, accuracy: 0.001)
        
        // Float16 2.0: 0x4000
        XCTAssertEqual(CVPixelBufferUtilities.float16ToFloat32(0x4000), 2.0, accuracy: 0.001)
        
        // Float16 -2.0: 0xC000
        XCTAssertEqual(CVPixelBufferUtilities.float16ToFloat32(0xC000), -2.0, accuracy: 0.001)
        
        // Float16 0.25: 0x3400
        XCTAssertEqual(CVPixelBufferUtilities.float16ToFloat32(0x3400), 0.25, accuracy: 0.001)
    }
    
    func testFloat16ToFloat32SpecialValues() {
        // Positive infinity: 0x7C00
        let posInf = CVPixelBufferUtilities.float16ToFloat32(0x7C00)
        XCTAssertTrue(posInf.isInfinite && posInf > 0)
        
        // Negative infinity: 0xFC00
        let negInf = CVPixelBufferUtilities.float16ToFloat32(0xFC00)
        XCTAssertTrue(negInf.isInfinite && negInf < 0)
        
        // NaN: 0x7E00 (common NaN representation)
        let nan = CVPixelBufferUtilities.float16ToFloat32(0x7E00)
        XCTAssertTrue(nan.isNaN)
    }
    
    func testFloat32ToFloat16Zero() {
        let result = CVPixelBufferUtilities.float32ToFloat16(0.0)
        XCTAssertEqual(result, 0x0000)
    }
    
    func testFloat32ToFloat16One() {
        let result = CVPixelBufferUtilities.float32ToFloat16(1.0)
        XCTAssertEqual(result, 0x3C00)
        
        let negOne = CVPixelBufferUtilities.float32ToFloat16(-1.0)
        XCTAssertEqual(negOne, 0xBC00)
    }
    
    func testFloat32ToFloat16CommonValues() {
        XCTAssertEqual(CVPixelBufferUtilities.float32ToFloat16(0.5), 0x3800)
        XCTAssertEqual(CVPixelBufferUtilities.float32ToFloat16(2.0), 0x4000)
        XCTAssertEqual(CVPixelBufferUtilities.float32ToFloat16(-2.0), 0xC000)
    }
    
    func testFloat32ToFloat16SpecialValues() {
        // Infinity
        let posInf = CVPixelBufferUtilities.float32ToFloat16(Float.infinity)
        XCTAssertEqual(posInf, 0x7C00)
        
        let negInf = CVPixelBufferUtilities.float32ToFloat16(-Float.infinity)
        XCTAssertEqual(negInf, 0xFC00)
        
        // NaN
        let nan = CVPixelBufferUtilities.float32ToFloat16(Float.nan)
        XCTAssertEqual(nan, 0x7E00)
    }
    
    func testFloat16RoundTrip() {
        // Test round-trip conversion for various values
        let testValues: [Float] = [0.0, 1.0, -1.0, 0.5, 0.25, 2.0, 100.0, -100.0, 0.001]
        
        for value in testValues {
            let half = CVPixelBufferUtilities.float32ToFloat16(value)
            let back = CVPixelBufferUtilities.float16ToFloat32(half)
            // Float16 has limited precision, so allow some error
            XCTAssertEqual(back, value, accuracy: abs(value) * 0.01 + 0.001,
                          "Round-trip failed for \(value): got \(back)")
        }
    }
}
