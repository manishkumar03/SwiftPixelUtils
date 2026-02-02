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
/// | Type | Range | Bits | Typical Use |
/// |------|-------|------|-------------|
/// | **Int4** | [-8, 7] | 4 | LLM weights, edge deployment |
/// | **UInt4** | [0, 15] | 4 | LLM weights, aggressive compression |
/// | **Int8** | [-128, 127] | 8 | TFLite, ONNX, ExecuTorch (symmetric) |
/// | **UInt8** | [0, 255] | 8 | TFLite (asymmetric), CoreML |
/// | **Int16** | [-32768, 32767] | 16 | High-precision quantization |
///
/// ### INT4 for LLM/Edge Deployment
///
/// INT4 quantization provides 8× compression vs Float32, ideal for:
/// - Large Language Models (LLMs): GPT, Llama, Mistral, etc.
/// - Edge devices with limited memory
/// - Mobile apps requiring minimal model size
///
/// ```swift
/// // INT4 quantization for LLM weights
/// let result = try Quantizer.quantize(
///     data: modelWeights,
///     options: QuantizationOptions(
///         mode: .perTensor,
///         dtype: .int4,
///         scale: [scale],
///         zeroPoint: [zeroPoint]
///     )
/// )
/// // result.packedInt4Data contains packed bytes (2 values per byte)
/// ```
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
/// - **Use case**: Convolution weights, models with varying channel ranges
///
/// ## Per-Channel Usage
///
/// ```swift
/// // Calibrate per-channel
/// let perChannelParams = Quantizer.calibratePerChannel(
///     data: floatPixels,
///     numChannels: 3,
///     spatialSize: 224 * 224,  // H * W
///     channelAxis: 0,          // CHW layout
///     dtype: .int8
/// )
///
/// // Quantize with per-channel parameters
/// let result = try Quantizer.quantize(
///     data: floatPixels,
///     options: QuantizationOptions(
///         mode: .perChannel,
///         dtype: .int8,
///         scale: perChannelParams.scales,
///         zeroPoint: perChannelParams.zeroPoints,
///         channelAxis: 0,
///         numChannels: 3,
///         spatialSize: 224 * 224
///     )
/// )
/// ```
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
        let numChannels = options.numChannels
        let spatialSize = options.spatialSize
        let channelAxis = options.channelAxis
        
        switch options.dtype {
        case .int8:
            let result = quantizeToInt8(
                data: data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            return QuantizationResult(
                int8Data: result.data,
                uint8Data: nil,
                int16Data: nil,
                packedInt4Data: nil,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .int8,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            
        case .uint8:
            let result = quantizeToUInt8(
                data: data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            return QuantizationResult(
                int8Data: nil,
                uint8Data: result.data,
                int16Data: nil,
                packedInt4Data: nil,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .uint8,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            
        case .int16:
            let result = quantizeToInt16(
                data: data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            return QuantizationResult(
                int8Data: nil,
                uint8Data: nil,
                int16Data: result.data,
                packedInt4Data: nil,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .int16,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            
        case .int4:
            let result = quantizeToInt4(
                data: data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            return QuantizationResult(
                int8Data: nil,
                uint8Data: nil,
                int16Data: nil,
                packedInt4Data: result.data,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .int4,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis,
                originalCount: data.count
            )
            
        case .uint4:
            let result = quantizeToUInt4(
                data: data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
            return QuantizationResult(
                int8Data: nil,
                uint8Data: nil,
                int16Data: nil,
                packedInt4Data: result.data,
                scale: result.scale,
                zeroPoint: result.zeroPoint,
                dtype: .uint4,
                mode: options.mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis,
                originalCount: data.count
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
    /// ## Per-Channel Dequantization
    ///
    /// For per-channel mode, provide the same parameters used during quantization:
    /// ```swift
    /// let restored = try Quantizer.dequantize(
    ///     int8Data: quantized.int8Data,
    ///     scale: quantized.scale,
    ///     zeroPoint: quantized.zeroPoint,
    ///     mode: .perChannel,
    ///     numChannels: 3,
    ///     spatialSize: 224 * 224,
    ///     channelAxis: 0
    /// )
    /// ```
    ///
    /// ## INT4 Dequantization
    ///
    /// For INT4 data, provide packed bytes and original count:
    /// ```swift
    /// let restored = try Quantizer.dequantize(
    ///     packedInt4Data: quantized.packedInt4Data,
    ///     originalCount: quantized.originalCount,
    ///     dtype: .int4,
    ///     scale: quantized.scale,
    ///     zeroPoint: quantized.zeroPoint
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - int8Data: Int8 data to dequantize (optional)
    ///   - uint8Data: UInt8 data to dequantize (optional)
    ///   - int16Data: Int16 data to dequantize (optional)
    ///   - packedInt4Data: Packed INT4 data to dequantize (optional)
    ///   - originalCount: Original number of values (required for INT4)
    ///   - dtype: Data type for INT4 (`.int4` or `.uint4`)
    ///   - scale: Scale factor(s) used during quantization
    ///   - zeroPoint: Zero point(s) used during quantization
    ///   - mode: Quantization mode (perTensor or perChannel)
    ///   - numChannels: Number of channels (required for perChannel mode)
    ///   - spatialSize: Spatial size H×W (required for perChannel mode with CHW layout)
    ///   - channelAxis: Channel axis (0 for CHW, 2 for HWC)
    /// - Returns: Dequantized float data
    /// - Throws: ``PixelUtilsError`` if dequantization fails
    public static func dequantize(
        int8Data: [Int8]? = nil,
        uint8Data: [UInt8]? = nil,
        int16Data: [Int16]? = nil,
        packedInt4Data: [UInt8]? = nil,
        originalCount: Int? = nil,
        dtype: QuantizationDType? = nil,
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode = .perTensor,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) throws -> [Float] {
        if let int8Data = int8Data {
            return dequantizeInt8(
                data: int8Data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
        } else if let uint8Data = uint8Data {
            return dequantizeUInt8(
                data: uint8Data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
        } else if let int16Data = int16Data {
            return dequantizeInt16(
                data: int16Data,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
        } else if let packedInt4Data = packedInt4Data, let count = originalCount, let dataType = dtype {
            return dequantizeInt4(
                packedData: packedInt4Data,
                originalCount: count,
                dtype: dataType,
                scale: scale,
                zeroPoint: zeroPoint,
                mode: mode,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: channelAxis
            )
        } else {
            throw PixelUtilsError.invalidOptions("No quantized data provided")
        }
    }
    
    // MARK: - Calibration
    
    /// Computes optimal scale and zero point from calibration data (per-tensor).
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
    /// For INT4 (signed):
    /// ```
    /// scale = (max - min) / 15  // [-8, 7] = 16 levels
    /// zero_point = round(-min / scale) - 8
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
        case .int4:
            qmin = -8
            qmax = 7
        case .uint4:
            qmin = 0
            qmax = 15
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
            let safeScale = max(scale, Float.leastNormalMagnitude)
            let rawZeroPoint = qmin - minVal / safeScale
            // Clamp to prevent overflow when converting to Int
            let clampedZeroPoint = max(min(rawZeroPoint, Float(Int.max)), Float(Int.min))
            let zeroPoint = Int(round(clampedZeroPoint))
            return (scale: safeScale, zeroPoint: clamp(zeroPoint, Int(qmin), Int(qmax)))
        }
    }
    
    // MARK: - Per-Channel Calibration
    
    /// Result from per-channel calibration
    public struct PerChannelCalibrationResult {
        /// Scale factors, one per channel
        public let scales: [Float]
        /// Zero points, one per channel
        public let zeroPoints: [Int]
        /// Number of channels
        public let numChannels: Int
        /// Min values per channel (for debugging)
        public let minValues: [Float]
        /// Max values per channel (for debugging)
        public let maxValues: [Float]
    }
    
    /// Computes optimal scale and zero point for each channel.
    ///
    /// Per-channel quantization provides better accuracy than per-tensor by
    /// computing separate quantization parameters for each channel, preserving
    /// the unique value distribution of each channel.
    ///
    /// ## Data Layout
    ///
    /// For CHW layout (channelAxis = 0):
    /// ```
    /// data = [R₀, R₁, ..., Rₙ, G₀, G₁, ..., Gₙ, B₀, B₁, ..., Bₙ]
    ///        |--- channel 0 ---|--- channel 1 ---|--- channel 2 ---|
    /// ```
    ///
    /// For HWC layout (channelAxis = 2):
    /// ```
    /// data = [R₀, G₀, B₀, R₁, G₁, B₁, ...]
    ///        |pixel 0| |pixel 1| ...
    /// ```
    ///
    /// - Parameters:
    ///   - data: Calibration data (flattened tensor)
    ///   - numChannels: Number of channels (e.g., 3 for RGB)
    ///   - spatialSize: Spatial size (H × W) for CHW layout, or total pixels for HWC
    ///   - channelAxis: Axis along which channels are arranged (0 for CHW, 2 for HWC)
    ///   - dtype: Target data type
    ///   - symmetric: Whether to use symmetric quantization (default: false)
    /// - Returns: ``PerChannelCalibrationResult`` with per-channel parameters
    public static func calibratePerChannel(
        data: [Float],
        numChannels: Int,
        spatialSize: Int,
        channelAxis: Int = 0,
        dtype: QuantizationDType,
        symmetric: Bool = false
    ) -> PerChannelCalibrationResult {
        guard !data.isEmpty, numChannels > 0, spatialSize > 0 else {
            return PerChannelCalibrationResult(
                scales: [1.0],
                zeroPoints: [0],
                numChannels: 1,
                minValues: [0],
                maxValues: [0]
            )
        }
        
        var scales = [Float](repeating: 1.0, count: numChannels)
        var zeroPoints = [Int](repeating: 0, count: numChannels)
        var minValues = [Float](repeating: Float.greatestFiniteMagnitude, count: numChannels)
        var maxValues = [Float](repeating: -Float.greatestFiniteMagnitude, count: numChannels)
        
        // Calculate min/max per channel
        if channelAxis == 0 {
            // CHW layout: channels are contiguous blocks
            for c in 0..<numChannels {
                let startIdx = c * spatialSize
                let endIdx = min(startIdx + spatialSize, data.count)
                
                for i in startIdx..<endIdx {
                    minValues[c] = min(minValues[c], data[i])
                    maxValues[c] = max(maxValues[c], data[i])
                }
            }
        } else {
            // HWC layout: channels are interleaved
            for i in stride(from: 0, to: data.count, by: numChannels) {
                for c in 0..<numChannels {
                    let idx = i + c
                    if idx < data.count {
                        minValues[c] = min(minValues[c], data[idx])
                        maxValues[c] = max(maxValues[c], data[idx])
                    }
                }
            }
        }
        
        // Calculate scale and zero point per channel
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
        case .int4:
            qmin = -8
            qmax = 7
        case .uint4:
            qmin = 0
            qmax = 15
        }
        
        for c in 0..<numChannels {
            if symmetric {
                let absMax = max(abs(minValues[c]), abs(maxValues[c]))
                scales[c] = max(absMax / ((qmax - qmin) / 2), Float.leastNormalMagnitude)
                zeroPoints[c] = 0
            } else {
                let range = maxValues[c] - minValues[c]
                scales[c] = max(range / (qmax - qmin), Float.leastNormalMagnitude)
                let rawZP = qmin - minValues[c] / scales[c]
                // Clamp to prevent overflow when converting to Int
                let clampedZP = max(min(rawZP, Float(Int.max)), Float(Int.min))
                let zp = Int(round(clampedZP))
                zeroPoints[c] = clamp(zp, Int(qmin), Int(qmax))
            }
        }
        
        return PerChannelCalibrationResult(
            scales: scales,
            zeroPoints: zeroPoints,
            numChannels: numChannels,
            minValues: minValues,
            maxValues: maxValues
        )
    }
    
    // MARK: - Private Helpers - Int8
    
    private static func quantizeToInt8(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> (data: [Int8], scale: [Float], zeroPoint: [Int]) {
        var result = [Int8](repeating: 0, count: data.count)
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel quantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = max(actualScale[c], Float.leastNormalMagnitude)
                    let z = actualZeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        let quantized = round(data[i] / s) + Float(z)
                        // Clamp to prevent overflow before Int conversion
                        let clamped = max(min(quantized, 127), -128)
                        result[i] = Int8(clamped)
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = max(actualScale[c], Float.leastNormalMagnitude)
                            let z = actualZeroPoint[c]
                            let quantized = round(data[idx] / s) + Float(z)
                            // Clamp to prevent overflow before Int conversion
                            let clamped = max(min(quantized, 127), -128)
                            result[idx] = Int8(clamped)
                        }
                    }
                }
            }
        } else {
            // Per-tensor quantization
            if scale.isEmpty {
                let (s, z) = calibrate(data: data, dtype: .int8)
                actualScale = [s]
                actualZeroPoint = [z]
            }
            
            let s = max(actualScale[0], Float.leastNormalMagnitude)
            let z = actualZeroPoint[0]
            
            for i in 0..<data.count {
                let quantized = round(data[i] / s) + Float(z)
                // Clamp to prevent overflow before Int conversion
                let clamped = max(min(quantized, 127), -128)
                result[i] = Int8(clamped)
            }
        }
        
        return (data: result, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    private static func dequantizeInt8(
        data: [Int8],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel dequantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = scale[c]
                    let z = zeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        result[i] = (Float(data[i]) - Float(z)) * s
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = scale[c]
                            let z = zeroPoint[c]
                            result[idx] = (Float(data[idx]) - Float(z)) * s
                        }
                    }
                }
            }
        } else {
            // Per-tensor dequantization
            let s = scale[0]
            let z = zeroPoint[0]
            
            for i in 0..<data.count {
                result[i] = (Float(data[i]) - Float(z)) * s
            }
        }
        
        return result
    }
    
    // MARK: - Private Helpers - UInt8
    
    private static func quantizeToUInt8(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> (data: [UInt8], scale: [Float], zeroPoint: [Int]) {
        var result = [UInt8](repeating: 0, count: data.count)
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel quantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = max(actualScale[c], Float.leastNormalMagnitude)
                    let z = actualZeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        let quantized = round(data[i] / s) + Float(z)
                        // Clamp to prevent overflow before Int conversion
                        let clamped = max(min(quantized, 255), 0)
                        result[i] = UInt8(clamped)
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = max(actualScale[c], Float.leastNormalMagnitude)
                            let z = actualZeroPoint[c]
                            let quantized = round(data[idx] / s) + Float(z)
                            // Clamp to prevent overflow before Int conversion
                            let clamped = max(min(quantized, 255), 0)
                            result[idx] = UInt8(clamped)
                        }
                    }
                }
            }
        } else {
            // Per-tensor quantization
            if scale.isEmpty {
                let (s, z) = calibrate(data: data, dtype: .uint8)
                actualScale = [s]
                actualZeroPoint = [z]
            }
            
            let s = max(actualScale[0], Float.leastNormalMagnitude)
            let z = actualZeroPoint[0]
            
            for i in 0..<data.count {
                let quantized = round(data[i] / s) + Float(z)
                // Clamp to prevent overflow before Int conversion
                let clamped = max(min(quantized, 255), 0)
                result[i] = UInt8(clamped)
            }
        }
        
        return (data: result, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    private static func dequantizeUInt8(
        data: [UInt8],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel dequantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = scale[c]
                    let z = zeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        result[i] = (Float(data[i]) - Float(z)) * s
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = scale[c]
                            let z = zeroPoint[c]
                            result[idx] = (Float(data[idx]) - Float(z)) * s
                        }
                    }
                }
            }
        } else {
            // Per-tensor dequantization
            let s = scale[0]
            let z = zeroPoint[0]
            
            for i in 0..<data.count {
                result[i] = (Float(data[i]) - Float(z)) * s
            }
        }
        
        return result
    }
    
    // MARK: - Private Helpers - Int16
    
    private static func quantizeToInt16(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> (data: [Int16], scale: [Float], zeroPoint: [Int]) {
        var result = [Int16](repeating: 0, count: data.count)
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel quantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = actualScale[c]
                    let z = actualZeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        let quantized = round(data[i] / s) + Float(z)
                        result[i] = Int16(clamp(Int(quantized), -32768, 32767))
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = actualScale[c]
                            let z = actualZeroPoint[c]
                            let quantized = round(data[idx] / s) + Float(z)
                            result[idx] = Int16(clamp(Int(quantized), -32768, 32767))
                        }
                    }
                }
            }
        } else {
            // Per-tensor quantization
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
        }
        
        return (data: result, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    private static func dequantizeInt16(
        data: [Int16],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel dequantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = scale[c]
                    let z = zeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        result[i] = (Float(data[i]) - Float(z)) * s
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = scale[c]
                            let z = zeroPoint[c]
                            result[idx] = (Float(data[idx]) - Float(z)) * s
                        }
                    }
                }
            }
        } else {
            // Per-tensor dequantization
            let s = scale[0]
            let z = zeroPoint[0]
            
            for i in 0..<data.count {
                result[i] = (Float(data[i]) - Float(z)) * s
            }
        }
        
        return result
    }
    
    // MARK: - Private Helpers - Int4
    
    /// Quantizes float data to INT4 and packs into bytes (2 values per byte)
    private static func quantizeToInt4(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> (data: [UInt8], scale: [Float], zeroPoint: [Int]) {
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        // First quantize to int4 values (unpacked)
        var int4Values = [Int8](repeating: 0, count: data.count)
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel quantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = max(actualScale[c], Float.leastNormalMagnitude)
                    let z = actualZeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        let quantized = round(data[i] / s) + Float(z)
                        // Clamp to prevent overflow before Int conversion
                        let clamped = max(min(quantized, 7), -8)
                        int4Values[i] = Int8(clamped)
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = max(actualScale[c], Float.leastNormalMagnitude)
                            let z = actualZeroPoint[c]
                            let quantized = round(data[idx] / s) + Float(z)
                            // Clamp to prevent overflow before Int conversion
                            let clamped = max(min(quantized, 7), -8)
                            int4Values[idx] = Int8(clamped)
                        }
                    }
                }
            }
        } else {
            // Per-tensor quantization
            if scale.isEmpty {
                let (s, z) = calibrate(data: data, dtype: .int4)
                actualScale = [s]
                actualZeroPoint = [z]
            }
            
            let s = max(actualScale[0], Float.leastNormalMagnitude)
            let z = actualZeroPoint[0]
            
            for i in 0..<data.count {
                let quantized = round(data[i] / s) + Float(z)
                // Clamp to prevent overflow before Int conversion
                let clamped = max(min(quantized, 7), -8)
                int4Values[i] = Int8(clamped)
            }
        }
        
        // Pack into bytes: 2 int4 values per byte
        // Low nibble = even index, high nibble = odd index
        let packedCount = (data.count + 1) / 2
        var packed = [UInt8](repeating: 0, count: packedCount)
        
        for i in 0..<data.count {
            let packedIdx = i / 2
            let value = UInt8(bitPattern: int4Values[i]) & 0x0F  // Mask to 4 bits
            
            if i % 2 == 0 {
                // Low nibble (bits 0-3)
                packed[packedIdx] |= value
            } else {
                // High nibble (bits 4-7)
                packed[packedIdx] |= (value << 4)
            }
        }
        
        return (data: packed, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    /// Quantizes float data to UINT4 and packs into bytes (2 values per byte)
    private static func quantizeToUInt4(
        data: [Float],
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> (data: [UInt8], scale: [Float], zeroPoint: [Int]) {
        var actualScale = scale
        var actualZeroPoint = zeroPoint
        
        // First quantize to uint4 values (unpacked)
        var uint4Values = [UInt8](repeating: 0, count: data.count)
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel quantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = max(actualScale[c], Float.leastNormalMagnitude)
                    let z = actualZeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, data.count)
                    
                    for i in startIdx..<endIdx {
                        let quantized = round(data[i] / s) + Float(z)
                        // Clamp to prevent overflow before conversion
                        let clamped = max(min(quantized, 15), 0)
                        uint4Values[i] = UInt8(clamped)
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: data.count, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < data.count {
                            let s = max(actualScale[c], Float.leastNormalMagnitude)
                            let z = actualZeroPoint[c]
                            let quantized = round(data[idx] / s) + Float(z)
                            // Clamp to prevent overflow before conversion
                            let clamped = max(min(quantized, 15), 0)
                            uint4Values[idx] = UInt8(clamped)
                        }
                    }
                }
            }
        } else {
            // Per-tensor quantization
            if scale.isEmpty {
                let (s, z) = calibrate(data: data, dtype: .uint4)
                actualScale = [s]
                actualZeroPoint = [z]
            }
            
            let s = max(actualScale[0], Float.leastNormalMagnitude)
            let z = actualZeroPoint[0]
            
            for i in 0..<data.count {
                let quantized = round(data[i] / s) + Float(z)
                // Clamp to prevent overflow before conversion
                let clamped = max(min(quantized, 15), 0)
                uint4Values[i] = UInt8(clamped)
            }
        }
        
        // Pack into bytes: 2 uint4 values per byte
        // Low nibble = even index, high nibble = odd index
        let packedCount = (data.count + 1) / 2
        var packed = [UInt8](repeating: 0, count: packedCount)
        
        for i in 0..<data.count {
            let packedIdx = i / 2
            let value = uint4Values[i] & 0x0F  // Mask to 4 bits
            
            if i % 2 == 0 {
                // Low nibble (bits 0-3)
                packed[packedIdx] |= value
            } else {
                // High nibble (bits 4-7)
                packed[packedIdx] |= (value << 4)
            }
        }
        
        return (data: packed, scale: actualScale, zeroPoint: actualZeroPoint)
    }
    
    /// Dequantizes packed INT4/UINT4 data back to float
    private static func dequantizeInt4(
        packedData: [UInt8],
        originalCount: Int,
        dtype: QuantizationDType,
        scale: [Float],
        zeroPoint: [Int],
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0
    ) -> [Float] {
        // Unpack the values
        var unpacked = [Int](repeating: 0, count: originalCount)
        
        for i in 0..<originalCount {
            let packedIdx = i / 2
            let nibble: UInt8
            
            if i % 2 == 0 {
                // Low nibble
                nibble = packedData[packedIdx] & 0x0F
            } else {
                // High nibble
                nibble = (packedData[packedIdx] >> 4) & 0x0F
            }
            
            // Sign extend for int4
            if dtype == .int4 {
                if nibble & 0x08 != 0 {
                    // Negative value - sign extend
                    unpacked[i] = Int(nibble) - 16
                } else {
                    unpacked[i] = Int(nibble)
                }
            } else {
                unpacked[i] = Int(nibble)
            }
        }
        
        // Dequantize
        var result = [Float](repeating: 0, count: originalCount)
        
        if mode == .perChannel, let channels = numChannels, let spatial = spatialSize, scale.count >= channels {
            // Per-channel dequantization
            if channelAxis == 0 {
                // CHW layout
                for c in 0..<channels {
                    let s = scale[c]
                    let z = zeroPoint[c]
                    let startIdx = c * spatial
                    let endIdx = min(startIdx + spatial, originalCount)
                    
                    for i in startIdx..<endIdx {
                        result[i] = (Float(unpacked[i]) - Float(z)) * s
                    }
                }
            } else {
                // HWC layout
                for i in stride(from: 0, to: originalCount, by: channels) {
                    for c in 0..<channels {
                        let idx = i + c
                        if idx < originalCount {
                            let s = scale[c]
                            let z = zeroPoint[c]
                            result[idx] = (Float(unpacked[idx]) - Float(z)) * s
                        }
                    }
                }
            }
        } else {
            // Per-tensor dequantization
            let s = scale[0]
            let z = zeroPoint[0]
            
            for i in 0..<originalCount {
                result[i] = (Float(unpacked[i]) - Float(z)) * s
            }
        }
        
        return result
    }
    
    // MARK: - INT4 Packing Utilities
    
    /// Packs an array of 4-bit values into bytes (2 values per byte)
    ///
    /// - Parameter values: Array of values in range [-8, 7] for int4 or [0, 15] for uint4
    /// - Returns: Packed byte array where each byte contains 2 values
    ///           (low nibble = even index, high nibble = odd index)
    public static func packInt4(_ values: [Int8]) -> [UInt8] {
        let packedCount = (values.count + 1) / 2
        var packed = [UInt8](repeating: 0, count: packedCount)
        
        for i in 0..<values.count {
            let packedIdx = i / 2
            let value = UInt8(bitPattern: values[i]) & 0x0F
            
            if i % 2 == 0 {
                packed[packedIdx] |= value
            } else {
                packed[packedIdx] |= (value << 4)
            }
        }
        
        return packed
    }
    
    /// Unpacks bytes into 4-bit signed values
    ///
    /// - Parameters:
    ///   - packed: Packed byte array
    ///   - count: Original number of values
    /// - Returns: Array of int4 values in range [-8, 7]
    public static func unpackInt4(_ packed: [UInt8], count: Int) -> [Int8] {
        var unpacked = [Int8](repeating: 0, count: count)
        
        for i in 0..<count {
            let packedIdx = i / 2
            let nibble: UInt8
            
            if i % 2 == 0 {
                nibble = packed[packedIdx] & 0x0F
            } else {
                nibble = (packed[packedIdx] >> 4) & 0x0F
            }
            
            // Sign extend
            if nibble & 0x08 != 0 {
                unpacked[i] = Int8(Int(nibble) - 16)
            } else {
                unpacked[i] = Int8(nibble)
            }
        }
        
        return unpacked
    }
    
    /// Unpacks bytes into 4-bit unsigned values
    ///
    /// - Parameters:
    ///   - packed: Packed byte array
    ///   - count: Original number of values
    /// - Returns: Array of uint4 values in range [0, 15]
    public static func unpackUInt4(_ packed: [UInt8], count: Int) -> [UInt8] {
        var unpacked = [UInt8](repeating: 0, count: count)
        
        for i in 0..<count {
            let packedIdx = i / 2
            
            if i % 2 == 0 {
                unpacked[i] = packed[packedIdx] & 0x0F
            } else {
                unpacked[i] = (packed[packedIdx] >> 4) & 0x0F
            }
        }
        
        return unpacked
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
///
/// ## Per-Channel Results
///
/// For per-channel quantization, `scale` and `zeroPoint` arrays contain
/// one value per channel. Use `numChannels`, `spatialSize`, and `channelAxis`
/// to correctly interpret the data layout for dequantization.
public struct QuantizationResult {
    /// Int8 quantized data (nil if dtype is not int8)
    public let int8Data: [Int8]?
    
    /// UInt8 quantized data (nil if dtype is not uint8)
    public let uint8Data: [UInt8]?
    
    /// Int16 quantized data (nil if dtype is not int16)
    public let int16Data: [Int16]?
    
    /// Packed INT4/UINT4 quantized data (nil if dtype is not int4/uint4)
    /// Each byte contains 2 values: low nibble = even index, high nibble = odd index
    /// Use `Quantizer.unpackInt4` or `Quantizer.unpackUInt4` to extract values
    public let packedInt4Data: [UInt8]?
    
    /// Scale factor(s) used for quantization
    /// - Per-tensor: single value
    /// - Per-channel: one value per channel
    public let scale: [Float]
    
    /// Zero point(s) used for quantization
    /// - Per-tensor: single value
    /// - Per-channel: one value per channel
    public let zeroPoint: [Int]
    
    /// Data type of the quantized values
    public let dtype: QuantizationDType
    
    /// Quantization mode (per-tensor or per-channel)
    public let mode: QuantizationMode
    
    /// Number of channels (for per-channel mode)
    public let numChannels: Int?
    
    /// Spatial size H×W (for per-channel mode with CHW layout)
    public let spatialSize: Int?
    
    /// Channel axis (0 for CHW, 2 for HWC)
    public let channelAxis: Int
    
    /// Original number of values (required for INT4 unpacking)
    public let originalCount: Int?
    
    /// Compression ratio vs Float32
    public var compressionRatio: Float {
        switch dtype {
        case .int4, .uint4: return 8.0
        case .int8, .uint8: return 4.0
        case .int16: return 2.0
        }
    }
    
    /// Size in bytes of the quantized data
    public var sizeInBytes: Int {
        if let data = int8Data { return data.count }
        if let data = uint8Data { return data.count }
        if let data = int16Data { return data.count * 2 }
        if let data = packedInt4Data { return data.count }
        return 0
    }
    
    public init(
        int8Data: [Int8]?,
        uint8Data: [UInt8]?,
        int16Data: [Int16]?,
        packedInt4Data: [UInt8]? = nil,
        scale: [Float],
        zeroPoint: [Int],
        dtype: QuantizationDType,
        mode: QuantizationMode,
        numChannels: Int? = nil,
        spatialSize: Int? = nil,
        channelAxis: Int = 0,
        originalCount: Int? = nil
    ) {
        self.int8Data = int8Data
        self.uint8Data = uint8Data
        self.int16Data = int16Data
        self.packedInt4Data = packedInt4Data
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.dtype = dtype
        self.mode = mode
        self.numChannels = numChannels
        self.spatialSize = spatialSize
        self.channelAxis = channelAxis
        self.originalCount = originalCount
    }
}
