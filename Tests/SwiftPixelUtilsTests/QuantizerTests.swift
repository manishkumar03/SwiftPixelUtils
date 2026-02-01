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
    
    func testPerChannelQuantizationCHWLayout() throws {
        // 3 channels, 4 values each (CHW layout: all channel 0, then all channel 1, etc.)
        let numChannels = 3
        let spatialSize = 4
        let original: [Float] = [
            0.0, 0.25, 0.5, 0.75,   // Channel 0 (R)
            -1.0, -0.5, 0.5, 1.0,   // Channel 1 (G) 
            -2.0, -1.0, 1.0, 2.0    // Channel 2 (B)
        ]
        
        // Calibrate per-channel
        let params = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int8
        )
        
        XCTAssertEqual(params.numChannels, 3)
        XCTAssertEqual(params.scales.count, 3)
        XCTAssertEqual(params.zeroPoints.count, 3)
        
        // Verify different scales for different ranges
        // Channel 2 has largest range, should have largest scale
        XCTAssertGreaterThan(params.scales[2], params.scales[0])
        
        // Quantize
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .int8,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
        )
        
        XCTAssertNotNil(quantized.int8Data)
        XCTAssertEqual(quantized.int8Data?.count, 12)
        XCTAssertEqual(quantized.mode, .perChannel)
    }
    
    func testPerChannelQuantizationHWCLayout() throws {
        // 4 pixels, 3 channels each (HWC layout: interleaved RGBRGBRGB...)
        let numChannels = 3
        let spatialSize = 4
        let original: [Float] = [
            0.0, -1.0, -2.0,   // Pixel 0: R, G, B
            0.25, -0.5, -1.0,  // Pixel 1
            0.5, 0.5, 1.0,     // Pixel 2
            0.75, 1.0, 2.0     // Pixel 3
        ]
        
        // Calibrate per-channel (HWC)
        let params = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 2,  // HWC layout
            dtype: .int8
        )
        
        XCTAssertEqual(params.scales.count, 3)
        
        // Quantize
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .int8,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: 2,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
        )
        
        XCTAssertNotNil(quantized.int8Data)
        XCTAssertEqual(quantized.int8Data?.count, 12)
    }
    
    func testPerChannelRoundTripCHW() throws {
        let numChannels = 3
        let spatialSize = 4
        let original: [Float] = [
            0.0, 0.25, 0.5, 0.75,   // Channel 0
            -1.0, -0.5, 0.5, 1.0,   // Channel 1
            -2.0, -1.0, 1.0, 2.0    // Channel 2
        ]
        
        // Calibrate
        let params = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int8
        )
        
        // Quantize
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .int8,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
        )
        
        // Dequantize
        let restored = try Quantizer.dequantize(
            int8Data: quantized.int8Data,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perChannel,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0
        )
        
        // Verify round-trip accuracy
        XCTAssertEqual(restored.count, original.count)
        for i in 0..<original.count {
            XCTAssertEqual(restored[i], original[i], accuracy: 0.05)
        }
    }
    
    func testPerChannelVsPerTensorAccuracy() throws {
        // Create data where per-channel should clearly outperform per-tensor
        let numChannels = 3
        let spatialSize = 4
        
        // Deliberately different ranges per channel
        let rChannel: [Float] = [0.0, 0.1, 0.2, 0.3]       // Range: 0.3
        let gChannel: [Float] = [-1.0, -0.3, 0.3, 1.0]     // Range: 2.0
        let bChannel: [Float] = [-10.0, -5.0, 5.0, 10.0]   // Range: 20.0
        let original = rChannel + gChannel + bChannel
        
        // Per-Tensor
        let tensorParams = Quantizer.calibrate(data: original, dtype: .int8)
        let tensorQuantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int8,
                scale: [tensorParams.scale],
                zeroPoint: [tensorParams.zeroPoint]
            )
        )
        let tensorRestored = try Quantizer.dequantize(
            int8Data: tensorQuantized.int8Data,
            scale: tensorQuantized.scale,
            zeroPoint: tensorQuantized.zeroPoint,
            mode: .perTensor
        )
        
        // Per-Channel
        let channelParams = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int8
        )
        let channelQuantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .int8,
                scale: channelParams.scales,
                zeroPoint: channelParams.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
        )
        let channelRestored = try Quantizer.dequantize(
            int8Data: channelQuantized.int8Data,
            scale: channelQuantized.scale,
            zeroPoint: channelQuantized.zeroPoint,
            mode: .perChannel,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0
        )
        
        // Calculate errors
        let tensorErrors = zip(original, tensorRestored).map { abs($0 - $1) }
        let channelErrors = zip(original, channelRestored).map { abs($0 - $1) }
        
        let tensorAvgError = tensorErrors.reduce(0, +) / Float(tensorErrors.count)
        let channelAvgError = channelErrors.reduce(0, +) / Float(channelErrors.count)
        
        // Per-channel should be more accurate (lower error)
        XCTAssertLessThan(channelAvgError, tensorAvgError)
        
        // Specifically check R channel which has small range
        let rTensorErrors = zip(rChannel, Array(tensorRestored[0..<spatialSize])).map { abs($0 - $1) }
        let rChannelErrors = zip(rChannel, Array(channelRestored[0..<spatialSize])).map { abs($0 - $1) }
        
        let rTensorAvg = rTensorErrors.reduce(0, +) / Float(rTensorErrors.count)
        let rChannelAvg = rChannelErrors.reduce(0, +) / Float(rChannelErrors.count)
        
        // R channel error should be much lower with per-channel
        XCTAssertLessThan(rChannelAvg, rTensorAvg)
    }
    
    func testPerChannelCalibrationDetectsRanges() {
        let numChannels = 3
        let spatialSize = 4
        
        // CHW layout with known ranges
        let rChannel: [Float] = [0.0, 0.1, 0.2, 0.3]       // min=0, max=0.3
        let gChannel: [Float] = [-1.0, -0.5, 0.5, 1.0]     // min=-1, max=1
        let bChannel: [Float] = [-5.0, -2.0, 2.0, 5.0]     // min=-5, max=5
        let data = rChannel + gChannel + bChannel
        
        let params = Quantizer.calibratePerChannel(
            data: data,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int8
        )
        
        // Verify min/max detection
        XCTAssertEqual(params.minValues[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(params.maxValues[0], 0.3, accuracy: 0.001)
        XCTAssertEqual(params.minValues[1], -1.0, accuracy: 0.001)
        XCTAssertEqual(params.maxValues[1], 1.0, accuracy: 0.001)
        XCTAssertEqual(params.minValues[2], -5.0, accuracy: 0.001)
        XCTAssertEqual(params.maxValues[2], 5.0, accuracy: 0.001)
    }
    
    func testPerChannelQuantizationUInt8() throws {
        let numChannels = 3
        let spatialSize = 4
        let original: [Float] = [
            0.0, 0.25, 0.5, 0.75,
            0.0, 0.33, 0.66, 1.0,
            0.1, 0.2, 0.3, 0.4
        ]
        
        let params = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .uint8
        )
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .uint8,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
        )
        
        XCTAssertNotNil(quantized.uint8Data)
        XCTAssertEqual(quantized.uint8Data?.count, 12)
        
        // Verify round-trip
        let restored = try Quantizer.dequantize(
            uint8Data: quantized.uint8Data,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perChannel,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0
        )
        
        for i in 0..<original.count {
            XCTAssertEqual(restored[i], original[i], accuracy: 0.02)
        }
    }
    
    func testPerChannelPerformance() throws {
        let numChannels = 3
        let spatialSize = 224 * 224
        let original = [Float](repeating: 0.5, count: numChannels * spatialSize)
        
        let params = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int8
        )
        
        measure {
            let _ = try? Quantizer.quantize(
                data: original,
                options: QuantizationOptions(
                    mode: .perChannel,
                    dtype: .int8,
                    scale: params.scales,
                    zeroPoint: params.zeroPoints,
                    channelAxis: 0,
                    numChannels: numChannels,
                    spatialSize: spatialSize
                )
            )
        }
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
    
    // MARK: - INT4 Quantization Tests
    
    func testInt4QuantizationBasic() throws {
        // Basic INT4 quantization test
        let original: [Float] = [-0.8, -0.4, 0.0, 0.4, 0.8]
        
        let params = Quantizer.calibrate(data: original, dtype: .int4)
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int4,
                scale: [params.scale],
                zeroPoint: [params.zeroPoint]
            )
        )
        
        XCTAssertNotNil(quantized.packedInt4Data)
        XCTAssertEqual(quantized.originalCount, 5)
        
        // 5 values -> 3 packed bytes (ceiling of 5/2)
        XCTAssertEqual(quantized.packedInt4Data?.count, 3)
        
        // Verify compression ratio
        XCTAssertEqual(quantized.compressionRatio, 8.0)
    }
    
    func testUInt4QuantizationBasic() throws {
        // UINT4 quantization for non-negative data
        let original: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        let params = Quantizer.calibrate(data: original, dtype: .uint4)
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint4,
                scale: [params.scale],
                zeroPoint: [params.zeroPoint]
            )
        )
        
        XCTAssertNotNil(quantized.packedInt4Data)
        XCTAssertEqual(quantized.originalCount, 5)
        XCTAssertEqual(quantized.packedInt4Data?.count, 3)
    }
    
    func testInt4PackingUnpacking() throws {
        // Test packing and unpacking utilities
        let values: [Int8] = [-8, -4, 0, 4, 7]  // Valid INT4 range
        
        let packed = Quantizer.packInt4(values)
        XCTAssertEqual(packed.count, 3)  // 5 values -> 3 bytes
        
        let unpacked = Quantizer.unpackInt4(packed, count: 5)
        XCTAssertEqual(unpacked.count, 5)
        
        for i in 0..<values.count {
            XCTAssertEqual(unpacked[i], values[i], "Value at index \(i) mismatch")
        }
    }
    
    func testUInt4PackingUnpacking() throws {
        // Test UINT4 packing and unpacking
        let values: [UInt8] = [0, 4, 8, 12, 15]  // Valid UINT4 range
        
        // Pack manually by converting to Int8 and using packInt4
        let int8Values = values.map { Int8(bitPattern: $0) }
        let packed = Quantizer.packInt4(int8Values)
        
        let unpacked = Quantizer.unpackUInt4(packed, count: 5)
        XCTAssertEqual(unpacked.count, 5)
        
        for i in 0..<values.count {
            XCTAssertEqual(unpacked[i], values[i], "Value at index \(i) mismatch")
        }
    }
    
    func testInt4RoundTrip() throws {
        let original: [Float] = [-0.7, -0.3, 0.0, 0.3, 0.7]
        
        let params = Quantizer.calibrate(data: original, dtype: .int4)
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int4,
                scale: [params.scale],
                zeroPoint: [params.zeroPoint]
            )
        )
        
        let dequantized = try Quantizer.dequantize(
            packedInt4Data: quantized.packedInt4Data,
            originalCount: quantized.originalCount,
            dtype: .int4,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perTensor
        )
        
        XCTAssertEqual(dequantized.count, original.count)
        
        // INT4 has larger quantization error than INT8
        // With 16 levels for range [-0.7, 0.7] = 1.4, step size ~ 0.0875
        for i in 0..<original.count {
            XCTAssertEqual(dequantized[i], original[i], accuracy: 0.15)
        }
    }
    
    func testInt4VsInt8Accuracy() throws {
        // Compare INT4 vs INT8 accuracy
        let original: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        // INT4
        let int4Params = Quantizer.calibrate(data: original, dtype: .int4)
        let int4Quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int4,
                scale: [int4Params.scale],
                zeroPoint: [int4Params.zeroPoint]
            )
        )
        let int4Dequantized = try Quantizer.dequantize(
            packedInt4Data: int4Quantized.packedInt4Data,
            originalCount: int4Quantized.originalCount,
            dtype: .int4,
            scale: int4Quantized.scale,
            zeroPoint: int4Quantized.zeroPoint,
            mode: .perTensor
        )
        
        // INT8
        let int8Params = Quantizer.calibrate(data: original, dtype: .int8)
        let int8Quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int8,
                scale: [int8Params.scale],
                zeroPoint: [int8Params.zeroPoint]
            )
        )
        let int8Dequantized = try Quantizer.dequantize(
            int8Data: int8Quantized.int8Data,
            scale: int8Quantized.scale,
            zeroPoint: int8Quantized.zeroPoint,
            mode: .perTensor
        )
        
        // Calculate errors
        let int4Errors = zip(original, int4Dequantized).map { abs($0 - $1) }
        let int8Errors = zip(original, int8Dequantized).map { abs($0 - $1) }
        
        let int4AvgError = int4Errors.reduce(0, +) / Float(int4Errors.count)
        let int8AvgError = int8Errors.reduce(0, +) / Float(int8Errors.count)
        
        // INT8 should have lower error than INT4
        XCTAssertLessThan(int8AvgError, int4AvgError, "INT8 should be more accurate than INT4")
        
        // But INT4 should still be usable (error < 0.2 for normalized data)
        XCTAssertLessThan(int4AvgError, 0.2, "INT4 error should be acceptable")
    }
    
    func testInt4Calibration() throws {
        let data: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        let params = Quantizer.calibrate(data: data, dtype: .int4)
        
        // For range [-1, 1] = 2.0, with 16 levels (range 15)
        // scale = 2.0 / 15 = 0.133...
        XCTAssertGreaterThan(params.scale, 0.1)
        XCTAssertLessThan(params.scale, 0.2)
    }
    
    func testInt4PerChannelQuantization() throws {
        let numChannels = 3
        let spatialSize = 4
        
        // CHW layout with different ranges per channel
        let rChannel: [Float] = [-0.3, -0.1, 0.1, 0.3]   // Small range
        let gChannel: [Float] = [-0.6, -0.2, 0.2, 0.6]   // Medium range
        let bChannel: [Float] = [-1.0, -0.5, 0.5, 1.0]   // Large range
        let original = rChannel + gChannel + bChannel
        
        let params = Quantizer.calibratePerChannel(
            data: original,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int4
        )
        
        XCTAssertEqual(params.scales.count, 3)
        XCTAssertEqual(params.zeroPoints.count, 3)
        
        // R channel should have smallest scale (smallest range)
        XCTAssertLessThan(params.scales[0], params.scales[2])
        
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perChannel,
                dtype: .int4,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
        )
        
        XCTAssertNotNil(quantized.packedInt4Data)
        XCTAssertEqual(quantized.originalCount, 12)
        XCTAssertEqual(quantized.packedInt4Data?.count, 6)  // 12 values -> 6 packed bytes
    }
    
    func testInt4EdgeCases() throws {
        // Test with single value
        let singleValue: [Float] = [0.5]
        let singleParams = Quantizer.calibrate(data: singleValue, dtype: .int4)
        let singleQuantized = try Quantizer.quantize(
            data: singleValue,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int4,
                scale: [singleParams.scale],
                zeroPoint: [singleParams.zeroPoint]
            )
        )
        XCTAssertEqual(singleQuantized.packedInt4Data?.count, 1)
        
        // Test with even number of values (no padding needed)
        let evenValues: [Float] = [0.0, 0.5, 1.0, 1.5]
        let evenParams = Quantizer.calibrate(data: evenValues, dtype: .int4)
        let evenQuantized = try Quantizer.quantize(
            data: evenValues,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .int4,
                scale: [evenParams.scale],
                zeroPoint: [evenParams.zeroPoint]
            )
        )
        XCTAssertEqual(evenQuantized.packedInt4Data?.count, 2)  // 4 values -> 2 bytes
    }
    
    func testInt4Performance() throws {
        let original = [Float](repeating: 0.5, count: 224 * 224 * 3)
        let params = Quantizer.calibrate(data: original, dtype: .int4)
        
        measure {
            let _ = try? Quantizer.quantize(
                data: original,
                options: QuantizationOptions(
                    mode: .perTensor,
                    dtype: .int4,
                    scale: [params.scale],
                    zeroPoint: [params.zeroPoint]
                )
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
