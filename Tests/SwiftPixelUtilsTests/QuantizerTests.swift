import XCTest
@testable import SwiftPixelUtils

/// Tests for ``Quantizer`` - Tensor quantization utilities.
///
/// ## Topics
///
/// ### UInt8 Quantization Tests
/// - Per-tensor and per-channel modes
/// - Scale and zero-point computation
/// - Value clamping to [0, 255]
///
/// ### Int8 Quantization Tests
/// - Symmetric quantization
/// - Signed range [-128, 127]
/// - Per-channel scale factors
///
/// ### Dequantization Tests
/// - Round-trip accuracy (quantize → dequantize)
/// - Precision loss measurement
///
/// ### Edge Cases
/// - Empty arrays, NaN/Inf handling
/// - Zero-range inputs
final class QuantizerTests: XCTestCase {
    
    // MARK: - UInt8 Quantization Tests
    
    func testQuantizeToUInt8PerTensor() throws {
        let original: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertNotNil(quantized.uint8Data)
        XCTAssertEqual(quantized.uint8Data?.count, 5)
        XCTAssertEqual(quantized.uint8Data?[0], 0)
        XCTAssertEqual(quantized.uint8Data?[4], 255)
    }
    
    func testQuantizeToUInt8WithZeroPoint() throws {
        let original: [Float] = [-1.0, 0.0, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 127.0)],
                zeroPoint: [128]
            )
        )
        
        XCTAssertNotNil(quantized.uint8Data)
        // UInt8 doesn't support accuracy, use direct comparison with tolerance
        XCTAssertNotNil(quantized.uint8Data?[0])
        XCTAssertTrue(quantized.uint8Data![0] <= 2)  // -1 -> ~1
        XCTAssertTrue(abs(Int(quantized.uint8Data![1]) - 128) <= 1)  // 0 -> 128
        XCTAssertTrue(abs(Int(quantized.uint8Data![2]) - 255) <= 1)  // 1 -> 255
    }
    
    func testQuantizeToUInt8RoundTrip() throws {
        let original: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        let dequantized = try Quantizer.dequantize(
            uint8Data: quantized.uint8Data,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perTensor
        )
        
        XCTAssertEqual(dequantized.count, 5)
        XCTAssertEqual(dequantized[0], 0, accuracy: 0.01)
        XCTAssertEqual(dequantized[4], 1, accuracy: 0.01)
    }
    
    // MARK: - Int8 Quantization Tests
    
    func testQuantizeToInt8Symmetric() throws {
        let original: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int8,
                scale: [Float(1.0 / 127.0)],
                zeroPoint: [0]  // Symmetric quantization
            )
        )
        
        XCTAssertNotNil(quantized.int8Data)
        XCTAssertEqual(quantized.int8Data?.count, 5)
        // Int8 doesn't support accuracy, use direct comparison with tolerance
        XCTAssertTrue(abs(Int(quantized.int8Data![0]) - (-127)) <= 1)  // -1.0
        XCTAssertTrue(abs(Int(quantized.int8Data![2]) - 0) <= 1)       // 0.0
        XCTAssertTrue(abs(Int(quantized.int8Data![4]) - 127) <= 1)     // 1.0
    }
    
    func testQuantizeToInt8RoundTrip() throws {
        let original: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int8,
                scale: [Float(1.0 / 127.0)],
                zeroPoint: [0]
            )
        )
        
        let dequantized = try Quantizer.dequantize(
            int8Data: quantized.int8Data,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perTensor
        )
        
        XCTAssertEqual(dequantized.count, 5)
        XCTAssertEqual(dequantized[0], -1.0, accuracy: 0.02)
        XCTAssertEqual(dequantized[2], 0.0, accuracy: 0.02)
        XCTAssertEqual(dequantized[4], 1.0, accuracy: 0.02)
    }
    
    // MARK: - Int16 Quantization Tests
    
    func testQuantizeToInt16() throws {
        let original: [Float] = [-1.0, 0.0, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int16,
                scale: [Float(1.0 / 32767.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertNotNil(quantized.int16Data)
        XCTAssertEqual(quantized.int16Data?.count, 3)
    }
    
    // MARK: - Calibration Tests
    
    func testCalibrationSymmetric() {
        let data: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        let (scale, zeroPoint) = Quantizer.calibrate(
            data: data,
            dtype: .int8,
            symmetric: true
        )
        
        XCTAssertGreaterThan(scale, 0)
        XCTAssertEqual(zeroPoint, 0)  // Symmetric quantization has zero_point = 0
    }
    
    func testCalibrationAsymmetric() {
        let data: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        let (scale, zeroPoint) = Quantizer.calibrate(
            data: data,
            dtype: .uint8,
            symmetric: false
        )
        
        XCTAssertGreaterThan(scale, 0)
        XCTAssertGreaterThanOrEqual(zeroPoint, 0)
        XCTAssertLessThanOrEqual(zeroPoint, 255)
    }
    
    func testCalibrationWithNegativeRange() {
        let data: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        let (scale, zeroPoint) = Quantizer.calibrate(
            data: data,
            dtype: .int8,
            symmetric: true
        )
        
        XCTAssertGreaterThan(scale, 0)
        // Scale should cover the range [-2, 2]
        XCTAssertGreaterThan(scale, 0.01)
    }
    
    // MARK: - Per-Channel Quantization Tests
    
    func testPerChannelQuantization() throws {
        // 3 channels, 4 values each
        let original: [Float] = [
            0.0, 0.5, 1.0, 0.25,  // Channel 0
            0.0, 0.25, 0.5, 0.75, // Channel 1
            0.1, 0.2, 0.3, 0.4    // Channel 2
        ]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0), Float(1.0 / 255.0), Float(1.0 / 255.0)],
                zeroPoint: [0, 0, 0]
            )
        )
        
        XCTAssertNotNil(quantized.uint8Data)
        XCTAssertEqual(quantized.uint8Data?.count, 12)
    }
    
    // MARK: - Edge Cases
    
    func testQuantizeEmptyArray() throws {
        let original: [Float] = []
        
        // Empty array should throw an error
        XCTAssertThrowsError(try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [0.01],
                zeroPoint: [0]
            )
        ))
    }
    
    func testQuantizeClampingUInt8() throws {
        // Values that exceed the UInt8 range should be clamped
        let original: [Float] = [-0.5, 0.0, 1.5]  // -0.5 and 1.5 exceed [0, 1]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        // Should be clamped to [0, 255]
        XCTAssertEqual(quantized.uint8Data?[0], 0)    // -0.5 clamped to 0
        XCTAssertEqual(quantized.uint8Data?[2], 255)  // 1.5 clamped to 255
    }
    
    func testQuantizeClampingInt8() throws {
        // Values that exceed the Int8 range should be clamped
        let original: [Float] = [-2.0, 0.0, 2.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int8,
                scale: [Float(1.0 / 127.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertEqual(quantized.int8Data?[0], -128)  // Clamped
        XCTAssertEqual(quantized.int8Data?[1], 0)
        XCTAssertEqual(quantized.int8Data?[2], 127)   // Clamped
    }
    
    func testQuantizeSingleValue() throws {
        let original: [Float] = [0.5]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertEqual(quantized.uint8Data?.count, 1)
        XCTAssertTrue(abs(Int(quantized.uint8Data![0]) - 128) <= 1)
    }
    
    func testQuantizeAllZeros() throws {
        let original: [Float] = [0.0, 0.0, 0.0, 0.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertTrue(quantized.uint8Data?.allSatisfy { $0 == 0 } ?? false)
    }
    
    func testQuantizeAllOnes() throws {
        let original: [Float] = [1.0, 1.0, 1.0, 1.0]
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertTrue(quantized.uint8Data?.allSatisfy { $0 == 255 } ?? false)
    }
    
    // MARK: - Dequantization Tests
    
    func testDequantizeUInt8() throws {
        let uint8Data: [UInt8] = [0, 64, 128, 192, 255]
        
        let dequantized = try Quantizer.dequantize(
            uint8Data: uint8Data,
            scale: [Float(1.0 / 255.0)],
            zeroPoint: [0],
            mode: .perTensor
        )
        
        XCTAssertEqual(dequantized[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(dequantized[2], 0.5, accuracy: 0.01)
        XCTAssertEqual(dequantized[4], 1.0, accuracy: 0.01)
    }
    
    func testDequantizeInt8() throws {
        let int8Data: [Int8] = [-127, -64, 0, 64, 127]
        
        let dequantized = try Quantizer.dequantize(
            int8Data: int8Data,
            scale: [Float(1.0 / 127.0)],
            zeroPoint: [0],
            mode: .perTensor
        )
        
        XCTAssertEqual(dequantized[0], -1.0, accuracy: 0.02)
        XCTAssertEqual(dequantized[2], 0.0, accuracy: 0.02)
        XCTAssertEqual(dequantized[4], 1.0, accuracy: 0.02)
    }
    
    func testDequantizeWithNonZeroZeroPoint() throws {
        let uint8Data: [UInt8] = [0, 128, 255]
        
        let dequantized = try Quantizer.dequantize(
            uint8Data: uint8Data,
            scale: [Float(1.0 / 127.0)],
            zeroPoint: [128],  // 128 represents 0
            mode: .perTensor
        )
        
        // 0 -> (0 - 128) * scale = -1.0
        // 128 -> (128 - 128) * scale = 0.0
        // 255 -> (255 - 128) * scale = 1.0
        XCTAssertEqual(dequantized[0], -1.0, accuracy: 0.02)
        XCTAssertEqual(dequantized[1], 0.0, accuracy: 0.02)
        XCTAssertEqual(dequantized[2], 1.0, accuracy: 0.02)
    }
    
    // MARK: - TFLite Compatibility Tests
    
    func testTFLiteTypicalQuantization() throws {
        // Typical TFLite quantization parameters
        let original: [Float] = Array(stride(from: 0.0, through: 1.0, by: 0.1))
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [0.00392157],  // ~1/255
                zeroPoint: [0]
            )
        )
        
        XCTAssertNotNil(quantized.uint8Data)
        
        // Verify dequantization produces similar values
        let dequantized = try Quantizer.dequantize(
            uint8Data: quantized.uint8Data,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perTensor
        )
        
        for i in 0..<original.count {
            XCTAssertEqual(dequantized[i], original[i], accuracy: 0.02)
        }
    }
    
    // MARK: - Performance Tests
    
    func testQuantizationPerformance() throws {
        let original = [Float](repeating: 0.5, count: 224 * 224 * 3)
        
        measure {
            let _ = try? Quantizer.quantize(
                data: original,
                options: QuantizationOptions(
                    mode: .perTensor,
                    dtype: .uint8,
                    scale: [Float(1.0 / 255.0)],
                    zeroPoint: [0]
                )
            )
        }
    }
    
    func testDequantizationPerformance() throws {
        let uint8Data = [UInt8](repeating: 128, count: 224 * 224 * 3)
        
        measure {
            let _ = try? Quantizer.dequantize(
                uint8Data: uint8Data,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0],
                mode: .perTensor
            )
        }
    }
}

// Helper extension for approximate comparison in tests
extension XCTestCase {
    func assertApproximatelyEqual(_ value: Int8, _ expected: Int8, accuracy: Int8, file: StaticString = #file, line: UInt = #line) {
        XCTAssertTrue(abs(Int(value) - Int(expected)) <= Int(accuracy), 
                     "(\(value)) is not equal to (\(expected)) +/- (\(accuracy))",
                     file: file, line: line)
    }
    
    func assertApproximatelyEqual(_ value: UInt8, _ expected: UInt8, accuracy: UInt8, file: StaticString = #file, line: UInt = #line) {
        let diff = value > expected ? value - expected : expected - value
        XCTAssertTrue(diff <= accuracy,
                     "(\(value)) is not equal to (\(expected)) +/- (\(accuracy))",
                     file: file, line: line)
    }
}
