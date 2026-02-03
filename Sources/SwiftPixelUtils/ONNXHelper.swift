//
//  ONNXHelper.swift
//  SwiftPixelUtils
//
//  Helper utilities for ONNX Runtime integration
//

import Foundation
import CoreGraphics

/// Helper utilities for ONNX Runtime integration.
///
/// ## Overview
///
/// This module provides convenience methods for preparing data for ONNX Runtime
/// inference and parsing common output formats. While SwiftPixelUtils doesn't
/// include ONNX Runtime as a dependency, these helpers make integration seamless.
///
/// ## Creating Tensor Data
///
/// ```swift
/// // Prepare image for ONNX model
/// let tensorInput = try await ONNXHelper.createTensorData(
///     from: .uiImage(image),
///     config: .yolov8
/// )
///
/// // Use with onnxruntime-swift
/// let ortValue = try OrtValue.createTensor(
///     ortEnv: env,
///     ortAllocator: allocator,
///     shape: tensorInput.shape.map { Int64($0) },
///     data: tensorInput.data
/// )
/// ```
///
/// ## Parsing Detection Output
///
/// ```swift
/// // Parse YOLOv8 detection output
/// let detections = ONNXHelper.parseDetectionOutput(
///     data: outputData,
///     shape: [1, 84, 8400],
///     format: .yoloV8,
///     confidenceThreshold: 0.5
/// )
/// ```
public enum ONNXHelper {
    
    // MARK: - Tensor Creation
    
    /// Create ONNX tensor data from an image source using a model configuration.
    ///
    /// - Parameters:
    ///   - source: The image to process
    ///   - config: ONNX model configuration
    /// - Returns: Tensor input ready for ONNX Runtime
    /// - Throws: ``PixelUtilsError`` if preprocessing fails
    public static func createTensorData(
        from source: ImageSource,
        config: ONNXModelConfig
    ) throws -> ONNXTensorInput {
        // Determine size from config shape (assumes NCHW: [batch, channels, height, width])
        guard config.inputShape.count >= 4 else {
            throw PixelUtilsError.invalidOptions("ONNX config shape must have at least 4 dimensions")
        }
        
        let height = config.inputShape[2]
        let width = config.inputShape[3]
        
        // Get model input using the framework preset
        let result = try PixelExtractor.getModelInput(
            source: source,
            framework: config.framework,
            width: width,
            height: height,
            resizeStrategy: config.detectionFormat != nil ? .letterbox : .cover
        )
        
        return ONNXTensorInput(
            name: config.inputName,
            data: result.data,
            shape: config.inputShape,
            dataType: config.inputDataType
        )
    }
    
    /// Create ONNX tensor data from an image source with explicit parameters.
    ///
    /// - Parameters:
    ///   - source: The image to process
    ///   - inputName: Tensor name (must match model's input name)
    ///   - width: Target width
    ///   - height: Target height
    ///   - framework: ML framework preset to use
    ///   - dataType: Output data type
    /// - Returns: Tensor input ready for ONNX Runtime
    public static func createTensorData(
        from source: ImageSource,
        inputName: String,
        width: Int,
        height: Int,
        framework: MLFramework = .onnx,
        dataType: ONNXDataType = .float32
    ) throws -> ONNXTensorInput {
        let result = try PixelExtractor.getModelInput(
            source: source,
            framework: framework,
            width: width,
            height: height
        )
        
        return ONNXTensorInput(
            name: inputName,
            data: result.data,
            shape: [1, result.channels, height, width],
            dataType: dataType
        )
    }
    
    /// Create multiple tensor inputs for batch inference.
    ///
    /// - Parameters:
    ///   - sources: Array of images to process
    ///   - config: ONNX model configuration
    /// - Returns: Tensor input with batched data
    public static func createBatchTensorData(
        from sources: [ImageSource],
        config: ONNXModelConfig
    ) throws -> ONNXTensorInput {
        guard !sources.isEmpty else {
            throw PixelUtilsError.emptyBatch("Cannot create batch tensor from empty source array")
        }
        
        guard config.inputShape.count >= 4 else {
            throw PixelUtilsError.invalidOptions("ONNX config shape must have at least 4 dimensions")
        }
        
        let height = config.inputShape[2]
        let width = config.inputShape[3]
        
        // Process all images
        var allData = Data()
        for source in sources {
            let result = try PixelExtractor.getModelInput(
                source: source,
                framework: config.framework,
                width: width,
                height: height,
                resizeStrategy: config.detectionFormat != nil ? .letterbox : .cover
            )
            allData.append(result.data)
        }
        
        // Update shape with actual batch size
        var batchShape = config.inputShape
        batchShape[0] = sources.count
        
        return ONNXTensorInput(
            name: config.inputName,
            data: allData,
            shape: batchShape,
            dataType: config.inputDataType
        )
    }
    
    // MARK: - Classification Output Parsing
    
    /// Parse classification output from ONNX model.
    ///
    /// - Parameters:
    ///   - data: Raw output data from ONNX inference
    ///   - numClasses: Number of classes
    ///   - topK: Number of top predictions to return
    ///   - applyingSoftmax: Whether to apply softmax to logits
    /// - Returns: Array of (classIndex, probability) tuples sorted by probability
    public static func parseClassificationOutput(
        data: Data,
        numClasses: Int,
        topK: Int = 5,
        applyingSoftmax: Bool = true
    ) -> [(classIndex: Int, probability: Float)] {
        let floats = data.withUnsafeBytes { ptr -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr.prefix(numClasses))
        }
        
        var scores = floats
        
        // Apply softmax if needed
        if applyingSoftmax {
            let maxScore = scores.max() ?? 0
            let expScores = scores.map { exp($0 - maxScore) }
            let sumExp = expScores.reduce(0, +)
            scores = expScores.map { $0 / sumExp }
        }
        
        // Get top-k
        let indexed = scores.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        let topResults = Array(sorted.prefix(topK))
        
        return topResults.map { (classIndex: $0.0, probability: $0.1) }
    }
    
    // MARK: - Detection Output Parsing
    
    /// Parse detection output from ONNX model in YOLOv8 format.
    ///
    /// YOLOv8 output shape: [1, 84, 8400] where 84 = 4 bbox coords + 80 classes
    ///
    /// - Parameters:
    ///   - data: Raw output data from ONNX inference
    ///   - numClasses: Number of detection classes (default 80 for COCO)
    ///   - confidenceThreshold: Minimum confidence to keep
    ///   - nmsThreshold: IoU threshold for NMS
    /// - Returns: Array of Detection objects
    public static func parseYOLOv8Output(
        data: Data,
        numClasses: Int = 80,
        confidenceThreshold: Float = 0.25,
        nmsThreshold: Float = 0.45
    ) -> [Detection] {
        let numBoxes = 8400
        let valuesPerBox = 4 + numClasses // 4 bbox + class scores
        
        // Parse float data
        let floats = data.withUnsafeBytes { ptr -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr.prefix(valuesPerBox * numBoxes))
        }
        
        // YOLOv8 format is transposed: [4+classes, num_boxes]
        // So we need to read column by column
        var candidates: [Detection] = []
        
        for boxIdx in 0..<numBoxes {
            // Extract bbox (cx, cy, w, h) - reading column-wise
            let cx = floats[0 * numBoxes + boxIdx]
            let cy = floats[1 * numBoxes + boxIdx]
            let w = floats[2 * numBoxes + boxIdx]
            let h = floats[3 * numBoxes + boxIdx]
            
            // Find best class
            var bestClass = 0
            var bestScore: Float = 0
            
            for classIdx in 0..<numClasses {
                let score = floats[(4 + classIdx) * numBoxes + boxIdx]
                if score > bestScore {
                    bestScore = score
                    bestClass = classIdx
                }
            }
            
            // Apply confidence threshold
            if bestScore >= confidenceThreshold {
                // Convert to x1,y1,x2,y2
                let x1 = cx - w / 2
                let y1 = cy - h / 2
                let x2 = cx + w / 2
                let y2 = cy + h / 2
                
                candidates.append(Detection(
                    box: [x1, y1, x2, y2],
                    score: bestScore,
                    classIndex: bestClass,
                    label: nil
                ))
            }
        }
        
        // Apply NMS
        return applyNMS(candidates, threshold: nmsThreshold)
    }
    
    /// Parse detection output from ONNX model in RT-DETR format.
    ///
    /// RT-DETR output shape: [1, 300, 4+num_classes]
    ///
    /// - Parameters:
    ///   - data: Raw output data from ONNX inference
    ///   - numClasses: Number of detection classes
    ///   - confidenceThreshold: Minimum confidence to keep
    /// - Returns: Array of Detection objects
    public static func parseRTDETROutput(
        data: Data,
        numClasses: Int = 80,
        confidenceThreshold: Float = 0.5
    ) -> [Detection] {
        let numQueries = 300
        let valuesPerQuery = 4 + numClasses
        
        // Parse float data
        let floats = data.withUnsafeBytes { ptr -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr.prefix(numQueries * valuesPerQuery))
        }
        
        var detections: [Detection] = []
        
        for queryIdx in 0..<numQueries {
            let offset = queryIdx * valuesPerQuery
            
            // RT-DETR outputs normalized coordinates (0-1)
            let cx = floats[offset + 0]
            let cy = floats[offset + 1]
            let w = floats[offset + 2]
            let h = floats[offset + 3]
            
            // Find best class (RT-DETR uses softmax, scores are already probabilities)
            var bestClass = 0
            var bestScore: Float = 0
            
            for classIdx in 0..<numClasses {
                let score = floats[offset + 4 + classIdx]
                if score > bestScore {
                    bestScore = score
                    bestClass = classIdx
                }
            }
            
            if bestScore >= confidenceThreshold {
                // Convert to x1,y1,x2,y2 (still normalized)
                let x1 = cx - w / 2
                let y1 = cy - h / 2
                let x2 = cx + w / 2
                let y2 = cy + h / 2
                
                detections.append(Detection(
                    box: [x1, y1, x2, y2],
                    score: bestScore,
                    classIndex: bestClass,
                    label: nil
                ))
            }
        }
        
        return detections
    }
    
    /// Generic detection output parser that handles multiple formats.
    ///
    /// - Parameters:
    ///   - data: Raw output data from ONNX inference
    ///   - shape: Output tensor shape
    ///   - format: Detection format specification
    ///   - numClasses: Number of classes
    ///   - confidenceThreshold: Minimum confidence to keep
    ///   - nmsThreshold: IoU threshold for NMS (if applicable)
    /// - Returns: Array of Detection objects
    public static func parseDetectionOutput(
        data: Data,
        shape: [Int],
        format: ONNXDetectionFormat,
        numClasses: Int = 80,
        confidenceThreshold: Float = 0.25,
        nmsThreshold: Float = 0.45
    ) -> [Detection] {
        switch format {
        case .yoloV8:
            return parseYOLOv8Output(
                data: data,
                numClasses: numClasses,
                confidenceThreshold: confidenceThreshold,
                nmsThreshold: nmsThreshold
            )
        case .rtdetr:
            return parseRTDETROutput(
                data: data,
                numClasses: numClasses,
                confidenceThreshold: confidenceThreshold
            )
        case .yoloV5, .ssd, .generic:
            return parseGenericDetectionOutput(
                data: data,
                shape: shape,
                numClasses: numClasses,
                confidenceThreshold: confidenceThreshold,
                nmsThreshold: nmsThreshold
            )
        }
    }
    
    /// Parse generic detection output: [batch, num_boxes, 4+1+classes] or [batch, num_boxes, 4+classes]
    private static func parseGenericDetectionOutput(
        data: Data,
        shape: [Int],
        numClasses: Int,
        confidenceThreshold: Float,
        nmsThreshold: Float
    ) -> [Detection] {
        guard shape.count >= 2 else { return [] }
        
        let numBoxes = shape[1]
        let valuesPerBox = shape.count > 2 ? shape[2] : (4 + numClasses)
        let hasObjectness = valuesPerBox == (4 + 1 + numClasses) // YOLOv5 style
        
        let floats = data.withUnsafeBytes { ptr -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr.prefix(numBoxes * valuesPerBox))
        }
        
        var candidates: [Detection] = []
        
        for boxIdx in 0..<numBoxes {
            let offset = boxIdx * valuesPerBox
            
            let cx = floats[offset + 0]
            let cy = floats[offset + 1]
            let w = floats[offset + 2]
            let h = floats[offset + 3]
            
            let classOffset = hasObjectness ? 5 : 4
            let objectness = hasObjectness ? sigmoid(floats[offset + 4]) : Float(1.0)
            
            var bestClass = 0
            var bestScore: Float = 0
            
            for classIdx in 0..<numClasses {
                let rawScore = floats[offset + classOffset + classIdx]
                let score = objectness * sigmoid(rawScore)
                if score > bestScore {
                    bestScore = score
                    bestClass = classIdx
                }
            }
            
            if bestScore >= confidenceThreshold {
                let x1 = cx - w / 2
                let y1 = cy - h / 2
                let x2 = cx + w / 2
                let y2 = cy + h / 2
                
                candidates.append(Detection(
                    box: [x1, y1, x2, y2],
                    score: bestScore,
                    classIndex: bestClass,
                    label: nil
                ))
            }
        }
        
        return applyNMS(candidates, threshold: nmsThreshold)
    }
    
    // MARK: - Segmentation Output Parsing
    
    /// Parse segmentation output from ONNX model.
    ///
    /// - Parameters:
    ///   - data: Raw output data (logits)
    ///   - shape: Output shape [batch, num_classes, height, width]
    ///   - applyArgmax: Whether to apply argmax to get class indices
    /// - Returns: Array of class indices per pixel or raw probabilities
    public static func parseSegmentationOutput(
        data: Data,
        shape: [Int],
        applyArgmax: Bool = true
    ) -> [Int] {
        guard shape.count >= 4 else { return [] }
        
        let numClasses = shape[1]
        let height = shape[2]
        let width = shape[3]
        let numPixels = height * width
        
        let floats = data.withUnsafeBytes { ptr -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr.prefix(numClasses * numPixels))
        }
        
        guard applyArgmax else {
            // Return raw data interpretation not applicable here
            return []
        }
        
        // Apply argmax per pixel
        var classIndices: [Int] = []
        classIndices.reserveCapacity(numPixels)
        
        for pixelIdx in 0..<numPixels {
            var bestClass = 0
            var bestScore: Float = -.infinity
            
            for classIdx in 0..<numClasses {
                // NCHW layout: [batch, class, h*w position]
                let score = floats[classIdx * numPixels + pixelIdx]
                if score > bestScore {
                    bestScore = score
                    bestClass = classIdx
                }
            }
            
            classIndices.append(bestClass)
        }
        
        return classIndices
    }
    
    // MARK: - Helpers
    
    /// Sigmoid activation function.
    @inlinable
    public static func sigmoid(_ x: Float) -> Float {
        1 / (1 + exp(-x))
    }
    
    /// Apply Non-Maximum Suppression to detections.
    public static func applyNMS(_ detections: [Detection], threshold: Float) -> [Detection] {
        guard !detections.isEmpty else { return [] }
        
        // Sort by score (confidence)
        var sorted = detections.sorted { $0.score > $1.score }
        var kept: [Detection] = []
        
        while !sorted.isEmpty {
            let best = sorted.removeFirst()
            kept.append(best)
            
            sorted = sorted.filter { detection in
                let iou = computeIoU(best.box, detection.box)
                return iou < threshold || detection.classIndex != best.classIndex
            }
        }
        
        return kept
    }
    
    /// Compute Intersection over Union between two boxes in [x1, y1, x2, y2] format.
    @inlinable
    public static func computeIoU(_ box1: [Float], _ box2: [Float]) -> Float {
        let x1 = max(box1[0], box2[0])
        let y1 = max(box1[1], box2[1])
        let x2 = min(box1[2], box2[2])
        let y2 = min(box1[3], box2[3])
        
        let intersectionArea = max(0, x2 - x1) * max(0, y2 - y1)
        
        let box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        let box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        let unionArea = box1Area + box2Area - intersectionArea
        
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
    
    /// Scale detection boxes from model coordinates to original image coordinates.
    ///
    /// - Parameters:
    ///   - detections: Detection results with boxes in model space
    ///   - modelSize: Size of model input (e.g., 640x640)
    ///   - originalSize: Original image size
    ///   - letterboxInfo: Optional letterbox info if letterboxing was used
    /// - Returns: Detections with scaled boxes
    public static func scaleDetections(
        _ detections: [Detection],
        modelSize: CGSize,
        originalSize: CGSize,
        letterboxInfo: LetterboxInfo? = nil
    ) -> [Detection] {
        return detections.map { detection in
            let scaledBox: [Float]
            
            if let info = letterboxInfo {
                // Remove letterbox padding and scale
                scaledBox = detection.box.enumerated().map { idx, value in
                    if idx % 2 == 0 {
                        // x coordinates
                        return (value - Float(info.offset.x)) / info.scale
                    } else {
                        // y coordinates
                        return (value - Float(info.offset.y)) / info.scale
                    }
                }
            } else {
                // Simple scaling
                let scaleX = Float(originalSize.width / modelSize.width)
                let scaleY = Float(originalSize.height / modelSize.height)
                
                scaledBox = detection.box.enumerated().map { idx, value in
                    idx % 2 == 0 ? value * scaleX : value * scaleY
                }
            }
            
            return Detection(
                box: scaledBox,
                score: detection.score,
                classIndex: detection.classIndex,
                label: detection.label
            )
        }
    }
}
