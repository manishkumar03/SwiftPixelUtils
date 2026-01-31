//
//  TensorValidation.swift
//  SwiftPixelUtils
//
//  Validate tensor shapes, dtypes, and value ranges
//

import Foundation

// MARK: - Validation Types

/// Specification for expected tensor properties.
public struct TensorSpec {
    /// Expected shape (use -1 for any dimension).
    public let shape: [Int]?
    
    /// Expected minimum value.
    public let minValue: Float?
    
    /// Expected maximum value.
    public let maxValue: Float?
    
    /// Whether to check for NaN values.
    public var checkNaN: Bool = true
    
    /// Whether to check for Inf values.
    public var checkInf: Bool = true
    
    /// Name for error messages.
    public var name: String = "tensor"
    
    public init(
        shape: [Int]? = nil,
        minValue: Float? = nil,
        maxValue: Float? = nil,
        checkNaN: Bool = true,
        checkInf: Bool = true,
        name: String = "tensor"
    ) {
        self.shape = shape
        self.minValue = minValue
        self.maxValue = maxValue
        self.checkNaN = checkNaN
        self.checkInf = checkInf
        self.name = name
    }
}

/// Result of tensor validation.
public struct TensorValidationResult {
    /// Whether the tensor is valid.
    public let isValid: Bool
    
    /// List of validation errors (empty if valid).
    public let errors: [String]
    
    /// Tensor statistics.
    public let statistics: TensorStatistics?
}

/// Statistics about tensor values.
public struct TensorStatistics {
    /// Minimum value in the tensor.
    public let min: Float
    
    /// Maximum value in the tensor.
    public let max: Float
    
    /// Mean value.
    public let mean: Float
    
    /// Standard deviation.
    public let std: Float
    
    /// Number of NaN values.
    public let nanCount: Int
    
    /// Number of Inf values.
    public let infCount: Int
    
    /// Total number of elements.
    public let count: Int
}

// MARK: - Tensor Validation

/// Validate tensor data for ML inference.
///
/// Provides comprehensive validation including:
/// - Shape validation with wildcards
/// - Value range checks
/// - NaN/Inf detection
/// - Statistical analysis
///
/// ## Example
///
/// ```swift
/// // Validate ImageNet tensor
/// let spec = TensorSpec(
///     shape: [1, 3, 224, 224],
///     minValue: -3.0,
///     maxValue: 3.0,
///     name: "input_tensor"
/// )
///
/// let result = TensorValidation.validate(data: tensor, shape: [1, 3, 224, 224], spec: spec)
/// if !result.isValid {
///     print("Validation errors: \(result.errors)")
/// }
/// ```
public enum TensorValidation {
    
    // MARK: - Validate
    
    /// Validate tensor data against a specification.
    ///
    /// - Parameters:
    ///   - data: The tensor data
    ///   - shape: Actual shape of the tensor
    ///   - spec: Validation specification
    /// - Returns: Validation result
    public static func validate(
        data: [Float],
        shape: [Int],
        spec: TensorSpec
    ) -> TensorValidationResult {
        var errors: [String] = []
        
        // Calculate statistics
        let stats = calculateStatistics(data)
        
        // Validate shape
        if let expectedShape = spec.shape {
            if expectedShape.count != shape.count {
                errors.append("\(spec.name): Expected \(expectedShape.count) dimensions, got \(shape.count)")
            } else {
                for (i, (expected, actual)) in zip(expectedShape, shape).enumerated() {
                    if expected != -1 && expected != actual {
                        errors.append("\(spec.name): Dimension \(i) expected \(expected), got \(actual)")
                    }
                }
            }
        }
        
        // Validate expected element count
        let expectedCount = shape.reduce(1, *)
        if data.count != expectedCount {
            errors.append("\(spec.name): Expected \(expectedCount) elements for shape \(shape), got \(data.count)")
        }
        
        // Check for NaN
        if spec.checkNaN && stats.nanCount > 0 {
            errors.append("\(spec.name): Contains \(stats.nanCount) NaN values")
        }
        
        // Check for Inf
        if spec.checkInf && stats.infCount > 0 {
            errors.append("\(spec.name): Contains \(stats.infCount) Inf values")
        }
        
        // Check value range
        if let minValue = spec.minValue, stats.min < minValue {
            errors.append("\(spec.name): Minimum value \(stats.min) is below expected \(minValue)")
        }
        
        if let maxValue = spec.maxValue, stats.max > maxValue {
            errors.append("\(spec.name): Maximum value \(stats.max) is above expected \(maxValue)")
        }
        
        return TensorValidationResult(
            isValid: errors.isEmpty,
            errors: errors,
            statistics: stats
        )
    }
    
    /// Validate tensor for ImageNet-normalized input.
    ///
    /// - Parameters:
    ///   - data: The tensor data
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Validation result
    public static func validateImageNetTensor(
        data: [Float],
        width: Int,
        height: Int
    ) -> TensorValidationResult {
        let spec = TensorSpec(
            shape: [-1, 3, height, width],  // NCHW or just CHW
            minValue: -3.0,  // Approximately 3 std deviations
            maxValue: 3.0,
            name: "ImageNet input"
        )
        
        // Determine actual shape
        let expectedElements = width * height * 3
        let shape: [Int]
        if data.count == expectedElements {
            shape = [3, height, width]
        } else if data.count == expectedElements * 1 {
            shape = [1, 3, height, width]
        } else {
            shape = [data.count]  // Will fail validation
        }
        
        return validate(data: data, shape: shape, spec: spec)
    }
    
    /// Validate tensor for TensorFlow-style [-1, 1] normalization.
    ///
    /// - Parameters:
    ///   - data: The tensor data
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Validation result
    public static func validateTensorFlowTensor(
        data: [Float],
        width: Int,
        height: Int
    ) -> TensorValidationResult {
        let spec = TensorSpec(
            shape: nil,
            minValue: -1.1,  // Allow small tolerance
            maxValue: 1.1,
            name: "TensorFlow input"
        )
        
        let shape = [data.count]  // Just validate count
        return validate(data: data, shape: shape, spec: spec)
    }
    
    /// Validate tensor for [0, 1] scaled input.
    ///
    /// - Parameters:
    ///   - data: The tensor data
    /// - Returns: Validation result
    public static func validateScaledTensor(data: [Float]) -> TensorValidationResult {
        let spec = TensorSpec(
            shape: nil,
            minValue: -0.01,  // Allow small tolerance
            maxValue: 1.01,
            name: "Scaled input"
        )
        
        let shape = [data.count]
        return validate(data: data, shape: shape, spec: spec)
    }
    
    // MARK: - Statistics
    
    /// Calculate statistics for tensor data.
    ///
    /// - Parameter data: The tensor data
    /// - Returns: Tensor statistics
    public static func calculateStatistics(_ data: [Float]) -> TensorStatistics {
        guard !data.isEmpty else {
            return TensorStatistics(min: 0, max: 0, mean: 0, std: 0, nanCount: 0, infCount: 0, count: 0)
        }
        
        var minVal: Float = .infinity
        var maxVal: Float = -.infinity
        var sum: Double = 0
        var nanCount = 0
        var infCount = 0
        
        for value in data {
            if value.isNaN {
                nanCount += 1
                continue
            }
            if value.isInfinite {
                infCount += 1
                continue
            }
            
            minVal = min(minVal, value)
            maxVal = max(maxVal, value)
            sum += Double(value)
        }
        
        let validCount = data.count - nanCount - infCount
        let mean = validCount > 0 ? Float(sum / Double(validCount)) : 0
        
        // Calculate std
        var sumSquaredDiff: Double = 0
        for value in data {
            if !value.isNaN && !value.isInfinite {
                let diff = Double(value) - Double(mean)
                sumSquaredDiff += diff * diff
            }
        }
        let std = validCount > 0 ? Float(sqrt(sumSquaredDiff / Double(validCount))) : 0
        
        return TensorStatistics(
            min: minVal.isInfinite ? 0 : minVal,
            max: maxVal.isInfinite ? 0 : maxVal,
            mean: mean,
            std: std,
            nanCount: nanCount,
            infCount: infCount,
            count: data.count
        )
    }
}

// MARK: - Batch Assembly

/// Result of batch assembly.
public struct BatchAssemblyResult {
    /// Combined tensor data for the batch.
    public let data: [Float]
    
    /// Shape of the batch tensor.
    public let shape: [Int]
    
    /// Number of items in the batch.
    public let batchSize: Int
    
    /// Width of each item.
    public let width: Int
    
    /// Height of each item.
    public let height: Int
    
    /// Channels of each item.
    public let channels: Int
}

/// Assemble multiple tensors into a batch.
///
/// ## Example
///
/// ```swift
/// let results = try await images.asyncMap { image in
///     try await PixelExtractor.getPixelData(source: .cgImage(image), options: options)
/// }
///
/// let batch = try BatchAssembly.assemble(results)
/// // batch.shape = [N, C, H, W] or [N, H, W, C] depending on layout
/// ```
public enum BatchAssembly {
    
    /// Assemble pixel data results into a batch.
    ///
    /// - Parameters:
    ///   - results: Array of pixel data results
    /// - Returns: Assembled batch
    /// - Throws: `PixelUtilsError` if dimensions don't match
    public static func assemble(_ results: [PixelDataResult]) throws -> BatchAssemblyResult {
        guard !results.isEmpty else {
            throw PixelUtilsError.invalidOptions("Cannot assemble empty batch")
        }
        
        let first = results[0]
        
        // Validate all have same dimensions
        for (i, result) in results.enumerated() {
            if result.width != first.width || result.height != first.height || result.channels != first.channels {
                throw PixelUtilsError.invalidOptions(
                    "Dimension mismatch at index \(i): expected \(first.width)x\(first.height)x\(first.channels), got \(result.width)x\(result.height)x\(result.channels)"
                )
            }
        }
        
        // Concatenate data
        var batchData: [Float] = []
        batchData.reserveCapacity(results.count * first.data.count)
        
        for result in results {
            batchData.append(contentsOf: result.data)
        }
        
        // Calculate shape
        let shape: [Int]
        switch first.dataLayout {
        case .hwc:
            shape = [results.count, first.height, first.width, first.channels]
        case .chw:
            shape = [results.count, first.channels, first.height, first.width]
        case .nhwc:
            shape = [results.count, first.height, first.width, first.channels]
        case .nchw:
            shape = [results.count, first.channels, first.height, first.width]
        }
        
        return BatchAssemblyResult(
            data: batchData,
            shape: shape,
            batchSize: results.count,
            width: first.width,
            height: first.height,
            channels: first.channels
        )
    }
}
