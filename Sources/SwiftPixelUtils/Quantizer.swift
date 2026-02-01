//
//  Quantizer.swift
//  SwiftPixelUtils
//
//  Quantization utilities for ML model deployment
//

import Foundation
import Accelerate

/// Quantization utilities for converting between float and integer representations.
///
/// ## Overview
///
/// Quantization reduces model size and inference latency by representing weights
/// and activations with lower-precision integers instead of 32-bit floats.
/// This module handles the preprocessing/postprocessing required for quantized models.
///
/// ## Quantization Basics
///
/// ### The Quantization Formula
///
/// ```
/// quantized = round((float_value / scale) + zero_point)
/// float_value = (quantized - zero_point) × scale
/// ```
///
/// Where:
/// - **scale**: Maps the float range to integer range
/// - **zero_point**: The integer value representing 0.0
///
/// ### Data Types
///
/// | Type | Range | Typical Use |
/// |------|-------|-------------|
/// | **Int8** | [-128, 127] | TFLite, ONNX, ExecuTorch (symmetric) |
/// | **UInt8** | [0, 255] | TFLite (asymmetric), CoreML |
/// | **Int16** | [-32768, 32767] | High-precision quantization |
///
/// ## Quantization Modes
///
/// ### Per-Tensor Quantization
///
/// Single scale/zero_point for entire tensor:
/// ```
/// scale = (max - min) / (qmax - qmin)
/// zero_point = round(-min / scale)
/// ```
///
/// - **Pros**: Simple, compact metadata
/// - **Cons**: Less accurate if value distributions vary across channels
///
/// ### Per-Channel Quantization
///
/// Separate scale/zero_point per channel:
/// ```
/// scale[c] = (max[c] - min[c]) / (qmax - qmin)
/// zero_point[c] = round(-min[c] / scale[c])
/// ```
///
/// - **Pros**: More accurate, better preserves per-channel distributions
/// - **Cons**: More metadata, slightly more complex inference
///
/// ## Usage
///
/// ```swift
/// // Quantize float data for TFLite model
/// let result = try Quantizer.quantize(
///     data: floatPixels,
///     options: QuantizationOptions(
///         mode: .perTensor,
///         dtype: .uint8,
///         scale: [0.00784314],  // 1/127.5
///         zeroPoint: [128]
///     )
/// )
///
/// // Use result.int8Data or result.uint8Data with your model
/// ```
///
/// ## TensorFlow Lite Compatibility
///
/// TFLite quantized models typically use:
/// - **Full integer**: Int8 or UInt8, per-tensor or per-channel
/// - **Dynamic range**: Float weights, int8 activations at runtime
///
/// Check your model's quantization config to match scale/zero_point values.
///
/// ## ExecuTorch Compatibility
///
/// ExecuTorch (PyTorch's on-device runtime) uses the same quantization scheme.
/// For ExecuTorch quantized models:
/// - Use Int8 with symmetric quantization (zero_point = 0)
/// - NCHW data layout (PyTorch convention)
/// - Per-channel quantization for weights, per-tensor for activations
///
/// ```swift
/// // Preprocess for ExecuTorch quantized model
/// let pixels = try await PixelExtractor.getPixelData(
///     from: image,
///     options: PixelDataOptions(
///         resize: .fit(width: 224, height: 224),
///         normalization: .imagenet,
///         dataLayout: .nchw
///     )
/// )
///
/// // Quantize for Int8 ExecuTorch model
/// let quantized = try Quantizer.quantize(
///     data: pixels.pixelData,
///     options: QuantizationOptions(
///         mode: .perTensor,
///         dtype: .int8,
///         scale: [model_scale],
///         zeroPoint: [0]  // Symmetric
///     )
/// )
/// // Pass quantized.int8Data to ExecuTorch tensor
/// ```
public enum Quantizer {
    
    // MARK: - Quantization
    
    /// Quantizes float data to integer representation.
    ///
    /// ## Formula
    ///
    /// For each value:
    /// ```
    /// quantized = clamp(round(value / scale + zero_point), dtype_min, dtype_max)
    /// ```
    ///
    /// ## Scale and Zero Point
    ///
    /// If not provided, scale and zero_point are computed from data range:
    /// ```
    /// min_val = min(data)
    /// max_val = max(data)
    /// scale = (max_val - min_val) / (dtype_max - dtype_min)
    /// zero_point = dtype_min - round(min_val / scale)
    /// ```
    ///
    /// - Parameters:
    ///   - data: Float data to quantize
    ///   - options: Quantization configuration (mode, dtype, scale, zeroPoint)
    /// - Returns: ``QuantizationResult`` with quantized data and computed parameters
    /// - Throws: ``PixelUtilsError`` if quantization fails
    public static func quantize(
        data: [Float],
        options: QuantizationOptions
    ) throws -> QuantizationResult {
        guard !data.isEmpty else {
            throw PixelUtilsError.emptyBatch("Cannot quantize empty data")
        }
        
        let scale = options.scale
        let zeroPoint = options.zeroPoint
        
        switch options.dtype {
        case .int8:
            let result = quantizeToInt8(data: data, scale: scale, zeroPoint: zeroPoint, mode: options.mode)
            return QuantizationResult(
                int8Data: result.data,
                uint8Data: nil,
                int16Data: nil,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .int8,
                mode: options.mode
            )
            
        case .uint8:
            let result = quantizeToUInt8(data: data, scale: scale, zeroPoint: zeroPoint, mode: options.mode)
            return QuantizationResult(
                int8Data: nil,
                uint8Data: result.data,
                int16Data: nil,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .uint8,
                mode: options.mode
            )
            
        case .int16:
            let result = quantizeToInt16(data: data, scale: scale, zeroPoint: zeroPoint, mode: options.mode)
            return QuantizationResult(
                int8Data: nil,
                uint8Data: nil,
                int16Data: result.data,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .int16,
                mode: options.mode
            )
        }
    }
    
    // MARK: - Dequantization
    
    /// Dequantizes integer data back to float representation.
    ///
    /// ## Formula
    ///
    /// For each value:
    /// ```
    /// float_value = (quantized - zero_point) × scale
    /// ```
    ///
    /// - Parameters:
    ///   - int8Data: Int8 data to dequantize (optional)
    ///   - uint8Data: UInt8 data to dequantize (optional)
    ///   - int16Data: Int16 data to dequantize (optional)
    ///   - scale: Scale factor(s) used during quantization
    ///   - zeroPoint: Zero point(s) used during quantization
    ///   - mode: Quantization mode (perTensor or perChannel)
    /// - Returns: Dequantized float data
    /// - Throws: ``PixelUtilsError`` if dequantization fails
    public static func dequantize(
        int8Data: [Int8]? = nil,
        uint8Data: [UInt8]? = nil,
        int16Data: [Int16]? = nil,
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode = .perTensor
    ) throws -> [Float] {
        if let int8Data = int8Data {
            return dequantizeInt8(data: int8Data, scale: scale, zeroPoint: zeroPoint, mode: mode)
        } else if let uint8Data = uint8Data {
            return dequantizeUInt8(data: uint8Data, scale: scale, zeroPoint: zeroPoint, mode: mode)
        } else if let int16Data = int16Data {
            return dequantizeInt16(data: int16Data, scale: scale, zeroPoint: zeroPoint, mode: mode)
        } else {
            throw PixelUtilsError.invalidOptions("No quantized data provided")
        }
    }
    
    // MARK: - Calibration
    
    /// Computes optimal scale and zero point from calibration data.
    ///
    /// Run representative data through your model and collect activation ranges,
    /// then use this function to compute quantization parameters.
    ///
    /// ## Algorithm
    ///
    /// For asymmetric quantization (UInt8):
    /// ```
    /// scale = (max - min) / 255
    /// zero_point = round(-min / scale)
    /// ```
    ///
    /// For symmetric quantization (Int8):
    /// ```
    /// scale = max(|min|, |max|) / 127
    /// zero_point = 0
    /// ```
    ///
    /// - Parameters:
    ///   - data: Calibration data (representative activation values)
    ///   - dtype: Target data type
    ///   - symmetric: Whether to use symmetric quantization (default: false)
    /// - Returns: Tuple of (scale, zeroPoint)
    public static func calibrate(
        data: [Float],
        dtype: QuantizationDType,
        symmetric: Bool = false
    ) -> (scale: Float, zeroPoint: Int) {
        guard !data.isEmpty else {
            return (scale: 1.0, zeroPoint: 0)
        }
        
        let minVal = data.min() ?? 0
        let maxVal = data.max() ?? 0
        
        let (qmin, qmax): (Float, Float)
        switch dtype {
        case .int8:
            qmin = -128
            qmax = 127
        case .uint8:
            qmin = 0
            qmax = 255
        case .int16:
            qmin = -32768
            qmax = 32767
        }
        
        if symmetric {
            // Symmetric quantization: zero_point = 0
            let absMax = max(abs(minVal), abs(maxVal))
            let scale = absMax / ((qmax - qmin) / 2)
            return (scale: max(scale, Float.leastNormalMagnitude), zeroPoint: 0)
        } else {
            // Asymmetric quantization
            let scale = (maxVal - minVal) / (qmax - qmin)
            let zeroPoint = Int(round(qmin - minVal / max(scale, Float.leastNormalMagnitude)))
            return (scale: max(scale, Float.leastNormalMagnitude), zeroPoint: clamp(zeroPoint, Int(qmin), Int(qmax)))
        }
    }
    
    // MARK: - Private Helpers - Int8
    
    private static func quantizeToInt8(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode
    ) -> (data: [Int8], scale: [Float], zeroPoint: [Int]) {
        var result = [Int8](repeating: 0, count: data.count)
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        if scale.isEmpty {
            let (s, z) = calibrate(data: data, dtype: .int8)
            actualScale = [s]
            actualZeroPoint = [z]
        }
        
        let s = actualScale[0]
        let z = actualZeroPoint[0]
        
        for i in 0..<data.count {
            let quantized = round(data[i] / s) + Float(z)
            result[i] = Int8(clamp(Int(quantized), -128, 127))
        }
        
        return (data: result, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    private static func dequantizeInt8(
        data: [Int8],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        let s = scale[0]
        let z = zeroPoint[0]
        
        for i in 0..<data.count {
            result[i] = (Float(data[i]) - Float(z)) * s
        }
        
        return result
    }
    
    // MARK: - Private Helpers - UInt8
    
    private static func quantizeToUInt8(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode
    ) -> (data: [UInt8], scale: [Float], zeroPoint: [Int]) {
        var result = [UInt8](repeating: 0, count: data.count)
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        if scale.isEmpty {
            let (s, z) = calibrate(data: data, dtype: .uint8)
            actualScale = [s]
            actualZeroPoint = [z]
        }
        
        let s = actualScale[0]
        let z = actualZeroPoint[0]
        
        for i in 0..<data.count {
            let quantized = round(data[i] / s) + Float(z)
            result[i] = UInt8(clamp(Int(quantized), 0, 255))
        }
        
        return (data: result, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    private static func dequantizeUInt8(
        data: [UInt8],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        let s = scale[0]
        let z = zeroPoint[0]
        
        for i in 0..<data.count {
            result[i] = (Float(data[i]) - Float(z)) * s
        }
        
        return result
    }
    
    // MARK: - Private Helpers - Int16
    
    private static func quantizeToInt16(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode
    ) -> (data: [Int16], scale: [Float], zeroPoint: [Int]) {
        var result = [Int16](repeating: 0, count: data.count)
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        if scale.isEmpty {
            let (s, z) = calibrate(data: data, dtype: .int16)
            actualScale = [s]
            actualZeroPoint = [z]
        }
        
        let s = actualScale[0]
        let z = actualZeroPoint[0]
        
        for i in 0..<data.count {
            let quantized = round(data[i] / s) + Float(z)
            result[i] = Int16(clamp(Int(quantized), -32768, 32767))
        }
        
        return (data: result, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    private static func dequantizeInt16(
        data: [Int16],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        let s = scale[0]
        let z = zeroPoint[0]
        
        for i in 0..<data.count {
            result[i] = (Float(data[i]) - Float(z)) * s
        }
        
        return result
    }
    
    // MARK: - Utility
    
    private static func clamp<T: Comparable>(_ value: T, _ minVal: T, _ maxVal: T) -> T {
        return min(max(value, minVal), maxVal)
    }
}

// MARK: - Quantization Result

/// Result from quantization operation.
///
/// Contains the quantized data in the appropriate type, plus the scale and
/// zero point values needed for dequantization.
public struct QuantizationResult {
    /// Int8 quantized data (nil if dtype is not int8)
    public let int8Data: [Int8]?
    
    /// UInt8 quantized data (nil if dtype is not uint8)
    public let uint8Data: [UInt8]?
    
    /// Int16 quantized data (nil if dtype is not int16)
    public let int16Data: [Int16]?
    
    /// Scale factor(s) used for quantization
    public let scale: [Float]
    
    /// Zero point(s) used for quantization
    public let zeroPoint: [Int]
    
    /// Data type of the quantized values
    public let dtype: QuantizationDType
    
    /// Quantization mode (per-tensor or per-channel)
    public let mode: QuantizationMode
    
    public init(
        int8Data: [Int8]?,
        uint8Data: [UInt8]?,
        int16Data: [Int16]?,
        scale: [Float],
        zeroPoint: [Int],
        dtype: QuantizationDType,
        mode: QuantizationMode
    ) {
        self.int8Data = int8Data
        self.uint8Data = uint8Data
        self.int16Data = int16Data
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.dtype = dtype
        self.mode = mode
    }
}
