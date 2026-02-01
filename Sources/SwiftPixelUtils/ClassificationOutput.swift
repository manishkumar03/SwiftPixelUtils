//
//  ClassificationOutput.swift
//  SwiftPixelUtils
//
//  High-level API for processing classification model outputs
//

import Foundation

/// A single classification prediction with label and confidence
public struct ClassificationPrediction: Equatable {
    /// Class index from the model
    public let classIndex: Int
    /// Human-readable label
    public let label: String
    /// Confidence score (0-1 after softmax)
    public let confidence: Float
    
    public init(classIndex: Int, label: String, confidence: Float) {
        self.classIndex = classIndex
        self.label = label
        self.confidence = confidence
    }
}

/// Result from classification output processing
public struct ClassificationResult {
    /// Top predictions sorted by confidence
    public let predictions: [ClassificationPrediction]
    /// Processing time in milliseconds
    public let processingTimeMs: Double
    /// Whether softmax was applied
    public let softmaxApplied: Bool
    /// Whether output was dequantized
    public let dequantized: Bool
    
    public init(
        predictions: [ClassificationPrediction],
        processingTimeMs: Double,
        softmaxApplied: Bool,
        dequantized: Bool
    ) {
        self.predictions = predictions
        self.processingTimeMs = processingTimeMs
        self.softmaxApplied = softmaxApplied
        self.dequantized = dequantized
    }
    
    /// Convenience accessor for the top prediction
    public var topPrediction: ClassificationPrediction? {
        predictions.first
    }
}

/// High-level utilities for processing classification model outputs.
///
/// This class provides a simplified API for the common pattern of:
/// 1. Dequantizing output (if quantized)
/// 2. Applying softmax
/// 3. Getting top-K predictions
/// 4. Mapping to human-readable labels
///
/// ## Usage
///
/// ```swift
/// // One-line output processing for TFLite quantized model
/// let result = try ClassificationOutput.process(
///     outputData: outputTensor.data,
///     quantization: .uint8(scale: quantParams.scale, zeroPoint: quantParams.zeroPoint),
///     topK: 5,
///     labels: .imagenet(hasBackgroundClass: true)
/// )
///
/// for prediction in result.predictions {
///     print("\(prediction.label): \(String(format: "%.1f%%", prediction.confidence * 100))")
/// }
/// ```
public enum ClassificationOutput {
    
    /// Quantization specification for model outputs
    public enum QuantizationType {
        /// No quantization (raw float32)
        case none
        /// UInt8 quantized output
        case uint8(scale: Float, zeroPoint: Int)
        /// Int8 quantized output
        case int8(scale: Float, zeroPoint: Int)
    }
    
    /// Label source specification
    public enum LabelSource {
        /// ImageNet-1K labels (1000 classes)
        /// - Parameter hasBackgroundClass: If true, index 0 is background and ImageNet classes start at 1
        case imagenet(hasBackgroundClass: Bool = false)
        /// COCO labels (80 classes)
        case coco
        /// CIFAR-10 labels (10 classes)
        case cifar10
        /// CIFAR-100 labels (100 classes)
        case cifar100
        /// Custom labels array
        case custom([String])
        /// No labels (returns indices as labels)
        case none
    }
    
    /// Process classification model output with automatic dequantization, softmax, and label mapping.
    ///
    /// This is the main entry point for processing classification outputs. It handles:
    /// - Dequantization (for quantized models)
    /// - Softmax activation
    /// - Top-K extraction
    /// - Label mapping
    ///
    /// - Parameters:
    ///   - outputData: Raw output data from the model (Data type)
    ///   - quantization: Quantization specification (use `.none` for float models)
    ///   - topK: Number of top predictions to return
    ///   - labels: Label source for mapping indices to names
    ///   - applySoftmax: Whether to apply softmax (default: true)
    ///   - temperature: Softmax temperature (default: 1.0)
    /// - Returns: ClassificationResult with top predictions
    /// - Throws: PixelUtilsError if processing fails
    public static func process(
        outputData: Data,
        quantization: QuantizationType,
        topK: Int = 5,
        labels: LabelSource = .none,
        applySoftmax: Bool = true,
        temperature: Float = 1.0
    ) throws -> ClassificationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Step 1: Convert Data to appropriate array and dequantize if needed
        let logits: [Float]
        let wasDequantized: Bool
        
        switch quantization {
        case .none:
            // Float32 data - just convert
            logits = outputData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float.self))
            }
            wasDequantized = false
            
        case .uint8(let scale, let zeroPoint):
            let uint8Array = [UInt8](outputData)
            logits = try Quantizer.dequantize(
                uint8Data: uint8Array,
                scale: [scale],
                zeroPoint: [zeroPoint],
                mode: .perTensor
            )
            wasDequantized = true
            
        case .int8(let scale, let zeroPoint):
            let int8Array = outputData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Int8.self))
            }
            logits = try Quantizer.dequantize(
                int8Data: int8Array,
                scale: [scale],
                zeroPoint: [zeroPoint],
                mode: .perTensor
            )
            wasDequantized = true
        }
        
        // Step 2: Apply softmax and get top-K
        let topKResult: TopKExtractor.TopKResult
        if applySoftmax {
            topKResult = TopKExtractor.extractTopKWithSoftmax(logits, k: topK, temperature: temperature)
        } else {
            topKResult = TopKExtractor.extractTopK(logits, k: topK)
        }
        
        // Step 3: Map indices to labels
        let predictions = mapToLabels(
            indices: topKResult.indices,
            confidences: topKResult.values,
            labelSource: labels
        )
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return ClassificationResult(
            predictions: predictions,
            processingTimeMs: processingTime,
            softmaxApplied: applySoftmax,
            dequantized: wasDequantized
        )
    }
    
    /// Process classification output from a UInt8 array (convenience method)
    public static func process(
        uint8Output: [UInt8],
        scale: Float,
        zeroPoint: Int,
        topK: Int = 5,
        labels: LabelSource = .none,
        applySoftmax: Bool = true,
        temperature: Float = 1.0
    ) throws -> ClassificationResult {
        let data = Data(uint8Output)
        return try process(
            outputData: data,
            quantization: .uint8(scale: scale, zeroPoint: zeroPoint),
            topK: topK,
            labels: labels,
            applySoftmax: applySoftmax,
            temperature: temperature
        )
    }
    
    /// Process classification output from a Float array (convenience method)
    public static func process(
        floatOutput: [Float],
        topK: Int = 5,
        labels: LabelSource = .none,
        applySoftmax: Bool = true,
        temperature: Float = 1.0
    ) throws -> ClassificationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Apply softmax and get top-K
        let topKResult: TopKExtractor.TopKResult
        if applySoftmax {
            topKResult = TopKExtractor.extractTopKWithSoftmax(floatOutput, k: topK, temperature: temperature)
        } else {
            topKResult = TopKExtractor.extractTopK(floatOutput, k: topK)
        }
        
        // Map indices to labels
        let predictions = mapToLabels(
            indices: topKResult.indices,
            confidences: topKResult.values,
            labelSource: labels
        )
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return ClassificationResult(
            predictions: predictions,
            processingTimeMs: processingTime,
            softmaxApplied: applySoftmax,
            dequantized: false
        )
    }
    
    // MARK: - Private Helpers
    
    private static func mapToLabels(
        indices: [Int],
        confidences: [Float],
        labelSource: LabelSource
    ) -> [ClassificationPrediction] {
        return zip(indices, confidences).map { (index, confidence) in
            let label: String
            
            switch labelSource {
            case .imagenet(let hasBackgroundClass):
                // If model has background class at index 0, ImageNet classes start at 1
                let adjustedIndex = hasBackgroundClass ? (index > 0 ? index - 1 : index) : index
                label = LabelDatabase.getLabel(adjustedIndex, dataset: .imagenet) ?? "Unknown (\(index))"
                
            case .coco:
                label = LabelDatabase.getLabel(index, dataset: .coco) ?? "Unknown (\(index))"
                
            case .cifar10:
                label = LabelDatabase.getLabel(index, dataset: .cifar10) ?? "Unknown (\(index))"
                
            case .cifar100:
                label = LabelDatabase.getLabel(index, dataset: .cifar100) ?? "Unknown (\(index))"
                
            case .custom(let labels):
                label = index < labels.count ? labels[index] : "Unknown (\(index))"
                
            case .none:
                label = "Class \(index)"
            }
            
            return ClassificationPrediction(
                classIndex: index,
                label: label,
                confidence: confidence
            )
        }
    }
}
