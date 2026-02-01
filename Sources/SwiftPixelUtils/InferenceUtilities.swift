//
//  InferenceUtilities.swift
//  SwiftPixelUtils
//
//  Post-processing utilities for ML inference pipelines
//

import Foundation
import Accelerate
import CoreGraphics

#if canImport(CoreML)
import CoreML
#endif

// MARK: - Activation Functions

/// Activation functions for converting raw model outputs to probabilities.
///
/// Neural networks often output raw logits that need to be transformed into
/// interpretable probability distributions. This enum provides common activation
/// functions used in inference post-processing.
///
/// ## Example
///
/// ```swift
/// // Convert classifier logits to probabilities
/// let logits: [Float] = [-1.5, 2.0, 0.5, -0.3]
/// let probs = ActivationFunctions.softmax(logits)
/// // probs ≈ [0.025, 0.822, 0.118, 0.035]
///
/// // Convert binary output to probability
/// let binaryLogit: Float = 1.5
/// let prob = ActivationFunctions.sigmoid(binaryLogit)
/// // prob ≈ 0.818
/// ```
public enum ActivationFunctions {
    
    /// Apply softmax activation to convert logits to probabilities.
    ///
    /// The softmax function normalizes a vector of real numbers into a probability
    /// distribution where all values are positive and sum to 1.
    ///
    /// ## Formula
    /// $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
    ///
    /// Uses numerically stable computation by subtracting the maximum value
    /// to prevent floating-point overflow.
    ///
    /// - Parameter logits: Raw model output values
    /// - Returns: Probability distribution (values sum to 1.0)
    ///
    /// ## Example
    /// ```swift
    /// let logits: [Float] = [2.0, 1.0, 0.1]
    /// let probs = ActivationFunctions.softmax(logits)
    /// // probs ≈ [0.659, 0.242, 0.099]
    /// ```
    public static func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        // Use Accelerate for performance
        var maxValue: Float = 0
        vDSP_maxv(logits, 1, &maxValue, vDSP_Length(logits.count))
        
        // Subtract max for numerical stability and compute exp
        var shifted = logits.map { $0 - maxValue }
        var result = [Float](repeating: 0, count: logits.count)
        var count = Int32(logits.count)
        vvexpf(&result, &shifted, &count)
        
        // Compute sum and normalize
        var sum: Float = 0
        vDSP_sve(result, 1, &sum, vDSP_Length(result.count))
        
        guard sum > 0 else {
            // Fallback to uniform distribution
            return Array(repeating: 1.0 / Float(logits.count), count: logits.count)
        }
        
        vDSP_vsdiv(result, 1, &sum, &result, 1, vDSP_Length(result.count))
        return result
    }
    
    /// Apply softmax along a specific dimension of a multi-dimensional tensor.
    ///
    /// Useful for semantic segmentation outputs where softmax is applied
    /// per-pixel across the class dimension.
    ///
    /// - Parameters:
    ///   - tensor: Flattened tensor data
    ///   - shape: Shape of the tensor [C, H, W] or [H, W, C]
    ///   - axis: Axis along which to apply softmax (typically the channel axis)
    ///   - layout: Data layout (CHW or HWC)
    /// - Returns: Tensor with softmax applied along the specified axis
    public static func softmaxAlongAxis(
        _ tensor: [Float],
        shape: [Int],
        axis: Int = 0,
        layout: DataLayout = .chw
    ) -> [Float] {
        guard shape.count == 3 else { return tensor }
        
        let (channels, height, width): (Int, Int, Int)
        switch layout {
        case .chw, .nchw:
            channels = shape[0]
            height = shape[1]
            width = shape[2]
        case .hwc, .nhwc:
            height = shape[0]
            width = shape[1]
            channels = shape[2]
        }
        
        var result = [Float](repeating: 0, count: tensor.count)
        
        // Apply softmax per spatial location
        for y in 0..<height {
            for x in 0..<width {
                var logits = [Float](repeating: 0, count: channels)
                
                // Extract values along channel axis
                for c in 0..<channels {
                    let idx: Int
                    switch layout {
                    case .chw, .nchw:
                        idx = c * height * width + y * width + x
                    case .hwc, .nhwc:
                        idx = y * width * channels + x * channels + c
                    }
                    logits[c] = tensor[idx]
                }
                
                // Apply softmax
                let probs = softmax(logits)
                
                // Write back
                for c in 0..<channels {
                    let idx: Int
                    switch layout {
                    case .chw, .nchw:
                        idx = c * height * width + y * width + x
                    case .hwc, .nhwc:
                        idx = y * width * channels + x * channels + c
                    }
                    result[idx] = probs[c]
                }
            }
        }
        
        return result
    }
    
    /// Apply sigmoid activation to convert logits to probabilities.
    ///
    /// The sigmoid function maps any real value to the range (0, 1),
    /// commonly used for binary classification or multi-label outputs.
    ///
    /// ## Formula
    /// $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
    ///
    /// - Parameter logit: Raw model output value
    /// - Returns: Probability in range (0, 1)
    ///
    /// ## Example
    /// ```swift
    /// let logit: Float = 2.0
    /// let prob = ActivationFunctions.sigmoid(logit)
    /// // prob ≈ 0.881
    /// ```
    public static func sigmoid(_ logit: Float) -> Float {
        // Handle extreme values to prevent overflow
        if logit >= 20 { return 1.0 }
        if logit <= -20 { return 0.0 }
        return 1.0 / (1.0 + exp(-logit))
    }
    
    /// Apply sigmoid activation to an array of values.
    ///
    /// Efficiently processes multiple values using Accelerate framework.
    ///
    /// - Parameter logits: Array of raw model output values
    /// - Returns: Array of probabilities in range (0, 1)
    public static func sigmoid(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        // Negate values
        var negated = [Float](repeating: 0, count: logits.count)
        var minusOne: Float = -1.0
        vDSP_vsmul(logits, 1, &minusOne, &negated, 1, vDSP_Length(logits.count))
        
        // Compute exp(-x)
        var expValues = [Float](repeating: 0, count: logits.count)
        var count = Int32(logits.count)
        vvexpf(&expValues, &negated, &count)
        
        // Add 1
        var one: Float = 1.0
        var denominator = [Float](repeating: 0, count: logits.count)
        vDSP_vsadd(expValues, 1, &one, &denominator, 1, vDSP_Length(logits.count))
        
        // Compute 1 / (1 + exp(-x))
        var result = [Float](repeating: 0, count: logits.count)
        vDSP_svdiv(&one, denominator, 1, &result, 1, vDSP_Length(logits.count))
        
        return result
    }
    
    /// Apply log-softmax for numerical stability in loss computation.
    ///
    /// ## Formula
    /// $$\log\text{softmax}(x_i) = x_i - \log\sum_{j} e^{x_j}$$
    ///
    /// - Parameter logits: Raw model output values
    /// - Returns: Log-probabilities
    public static func logSoftmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        // Find max for numerical stability
        var maxValue: Float = 0
        vDSP_maxv(logits, 1, &maxValue, vDSP_Length(logits.count))
        
        // Compute log(sum(exp(x - max)))
        let shifted = logits.map { $0 - maxValue }
        var expValues = [Float](repeating: 0, count: logits.count)
        var count = Int32(logits.count)
        vvexpf(&expValues, shifted, &count)
        
        var sum: Float = 0
        vDSP_sve(expValues, 1, &sum, vDSP_Length(expValues.count))
        let logSum = log(sum) + maxValue
        
        // Subtract logSum from each logit
        return logits.map { $0 - logSum }
    }
}

// MARK: - Top-K Extraction

/// Extract top-K predictions from model outputs.
///
/// Provides utilities for extracting the most confident predictions
/// from classification model outputs.
public enum TopKExtractor {
    
    /// Result of top-K extraction.
    public struct TopKResult: Sendable {
        /// Indices of the top-K elements (in descending order by value).
        public let indices: [Int]
        /// Values of the top-K elements.
        public let values: [Float]
        /// Processing time in milliseconds.
        public let processingTimeMs: Double
    }
    
    /// Extract top-K values and their indices from an array.
    ///
    /// Efficiently finds the K largest values in an unsorted array
    /// along with their original indices.
    ///
    /// - Parameters:
    ///   - values: Array of values to search
    ///   - k: Number of top elements to return
    ///   - minValue: Optional minimum threshold (values below this are ignored)
    /// - Returns: TopKResult containing indices and values
    ///
    /// ## Example
    /// ```swift
    /// let scores: [Float] = [0.1, 0.8, 0.05, 0.3, 0.9]
    /// let top3 = TopKExtractor.extractTopK(values: scores, k: 3)
    /// // top3.indices = [4, 1, 3]  // indices of 0.9, 0.8, 0.3
    /// // top3.values = [0.9, 0.8, 0.3]
    /// ```
    public static func extractTopK(
        values: [Float],
        k: Int,
        minValue: Float? = nil
    ) -> TopKResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create indexed values
        var indexedValues = values.enumerated().map { ($0.offset, $0.element) }
        
        // Filter by minimum if specified
        if let min = minValue {
            indexedValues = indexedValues.filter { $0.1 >= min }
        }
        
        // Sort by value descending
        indexedValues.sort { $0.1 > $1.1 }
        
        // Take top K
        let topK = Array(indexedValues.prefix(k))
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return TopKResult(
            indices: topK.map { $0.0 },
            values: topK.map { $0.1 },
            processingTimeMs: processingTime
        )
    }
    
    /// Extract top-K with softmax applied first.
    ///
    /// Combines softmax normalization with top-K extraction in a single call.
    ///
    /// - Parameters:
    ///   - logits: Raw model output logits
    ///   - k: Number of top elements to return
    ///   - minProbability: Minimum probability threshold
    /// - Returns: TopKResult with probabilities instead of raw logits
    public static func extractTopKWithSoftmax(
        logits: [Float],
        k: Int,
        minProbability: Float = 0.0
    ) -> TopKResult {
        let probabilities = ActivationFunctions.softmax(logits)
        return extractTopK(values: probabilities, k: k, minValue: minProbability)
    }
    
    /// Find the argmax (index of maximum value) in an array.
    ///
    /// - Parameter values: Array of values
    /// - Returns: Tuple of (index, value) for the maximum element
    public static func argmax(_ values: [Float]) -> (index: Int, value: Float)? {
        guard !values.isEmpty else { return nil }
        
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(values, 1, &maxValue, &maxIndex, vDSP_Length(values.count))
        
        return (Int(maxIndex), maxValue)
    }
    
    /// Find the argmin (index of minimum value) in an array.
    ///
    /// - Parameter values: Array of values
    /// - Returns: Tuple of (index, value) for the minimum element
    public static func argmin(_ values: [Float]) -> (index: Int, value: Float)? {
        guard !values.isEmpty else { return nil }
        
        var minValue: Float = 0
        var minIndex: vDSP_Length = 0
        vDSP_minvi(values, 1, &minValue, &minIndex, vDSP_Length(values.count))
        
        return (Int(minIndex), minValue)
    }
}

// MARK: - Soft-NMS

/// Non-Maximum Suppression variations for object detection post-processing.
///
/// Provides different NMS algorithms including the standard greedy NMS,
/// Soft-NMS (which reduces scores instead of eliminating boxes), and
/// batched variants for processing multiple images.
public enum NMSVariants {
    
    /// Soft-NMS mode determining how overlapping box scores are reduced.
    public enum SoftNMSMode: Sendable {
        /// Linear decay: score = score × (1 - iou) if iou > threshold
        case linear
        /// Gaussian decay: score = score × exp(-iou²/sigma)
        case gaussian(sigma: Float)
    }
    
    /// Apply Soft-NMS to detection results.
    ///
    /// Unlike standard NMS which completely removes overlapping boxes,
    /// Soft-NMS reduces their scores based on overlap. This helps when
    /// objects are close together or partially occluded.
    ///
    /// ## Paper Reference
    /// "Soft-NMS -- Improving Object Detection With One Line of Code"
    /// Bodla et al., ICCV 2017
    ///
    /// - Parameters:
    ///   - detections: Array of detections to process
    ///   - iouThreshold: IoU threshold for score reduction (linear mode)
    ///   - scoreThreshold: Minimum score to keep detection
    ///   - mode: Score reduction mode (linear or gaussian)
    /// - Returns: Filtered detections with adjusted scores
    ///
    /// ## Example
    /// ```swift
    /// let boxes: [[Float]] = [[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]]
    /// let scores: [Float] = [0.9, 0.85, 0.8]
    /// let classes: [Int] = [0, 0, 1]
    ///
    /// let detections = zip(zip(boxes, scores), classes).map {
    ///     Detection(box: $0.0.0, score: $0.0.1, classIndex: $0.1)
    /// }
    ///
    /// let filtered = NMSVariants.softNMS(
    ///     detections: detections,
    ///     iouThreshold: 0.5,
    ///     scoreThreshold: 0.3,
    ///     mode: .gaussian(sigma: 0.5)
    /// )
    /// ```
    public static func softNMS(
        detections: [Detection],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.3,
        mode: SoftNMSMode = .linear
    ) -> [Detection] {
        guard !detections.isEmpty else { return [] }
        
        // Work with mutable copy of scores
        var mutableDetections = detections.map { detection -> (detection: Detection, score: Float) in
            (detection, detection.score)
        }
        
        var results: [Detection] = []
        
        while !mutableDetections.isEmpty {
            // Find detection with highest score
            guard let maxIdx = mutableDetections.enumerated().max(by: { $0.element.score < $1.element.score })?.offset else {
                break
            }
            
            let selected = mutableDetections[maxIdx]
            
            // Skip if below threshold
            if selected.score < scoreThreshold {
                mutableDetections.remove(at: maxIdx)
                continue
            }
            
            // Add to results with updated score
            var resultDetection = selected.detection
            resultDetection = Detection(
                box: resultDetection.box,
                score: selected.score,
                classIndex: resultDetection.classIndex,
                label: resultDetection.label
            )
            results.append(resultDetection)
            
            mutableDetections.remove(at: maxIdx)
            
            // Update scores of remaining detections based on IoU
            for i in 0..<mutableDetections.count {
                let iou = BoundingBox.calculateIoU(
                    selected.detection.box,
                    mutableDetections[i].detection.box,
                    format: .xyxy
                )
                
                if iou > 0 {
                    let scoreReduction: Float
                    switch mode {
                    case .linear:
                        if iou > iouThreshold {
                            scoreReduction = 1 - iou
                        } else {
                            scoreReduction = 1.0
                        }
                    case .gaussian(let sigma):
                        let iouSquared = iou * iou
                        scoreReduction = exp(-iouSquared / sigma)
                    }
                    
                    mutableDetections[i].score *= scoreReduction
                }
            }
            
            // Remove detections below threshold
            mutableDetections = mutableDetections.filter { $0.score >= scoreThreshold }
        }
        
        return results
    }
    
    /// Apply NMS separately for each class (class-agnostic = false).
    ///
    /// This is the standard approach for multi-class object detection
    /// where overlapping boxes of different classes should not suppress each other.
    ///
    /// - Parameters:
    ///   - detections: Array of detections to process
    ///   - iouThreshold: IoU threshold for suppression
    ///   - scoreThreshold: Minimum score to keep detection
    /// - Returns: Filtered detections
    public static func perClassNMS(
        detections: [Detection],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.3
    ) -> [Detection] {
        // Group by class
        var byClass: [Int: [Detection]] = [:]
        for detection in detections where detection.score >= scoreThreshold {
            byClass[detection.classIndex, default: []].append(detection)
        }
        
        // Apply NMS per class
        var results: [Detection] = []
        for (_, classDetections) in byClass {
            let filtered = BoundingBox.nonMaxSuppression(
                detections: classDetections,
                iouThreshold: iouThreshold
            )
            results.append(contentsOf: filtered)
        }
        
        return results.sorted { $0.score > $1.score }
    }
    
    /// Apply batched NMS for processing multiple images efficiently.
    ///
    /// When running inference on multiple images, this processes all
    /// detections while keeping track of which image each detection belongs to.
    ///
    /// - Parameters:
    ///   - batchDetections: Array of detection arrays (one per image in batch)
    ///   - iouThreshold: IoU threshold for suppression
    ///   - scoreThreshold: Minimum score to keep detection
    ///   - maxDetectionsPerImage: Maximum detections to return per image
    /// - Returns: Array of filtered detection arrays (one per image)
    ///
    /// ## Example
    /// ```swift
    /// let batch = [image1Detections, image2Detections, image3Detections]
    /// let filteredBatch = NMSVariants.batchedNMS(
    ///     batchDetections: batch,
    ///     iouThreshold: 0.5,
    ///     maxDetectionsPerImage: 100
    /// )
    /// ```
    public static func batchedNMS(
        batchDetections: [[Detection]],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.3,
        maxDetectionsPerImage: Int = 300
    ) -> [[Detection]] {
        return batchDetections.map { imageDetections in
            let filtered = perClassNMS(
                detections: imageDetections,
                iouThreshold: iouThreshold,
                scoreThreshold: scoreThreshold
            )
            return Array(filtered.prefix(maxDetectionsPerImage))
        }
    }
    
    /// Apply class-agnostic NMS (treats all classes the same).
    ///
    /// Useful for single-class detection or when you want to suppress
    /// overlapping boxes regardless of their predicted class.
    ///
    /// - Parameters:
    ///   - detections: Array of detections to process
    ///   - iouThreshold: IoU threshold for suppression
    ///   - scoreThreshold: Minimum score to keep detection
    /// - Returns: Filtered detections
    public static func classAgnosticNMS(
        detections: [Detection],
        iouThreshold: Float = 0.5,
        scoreThreshold: Float = 0.3
    ) -> [Detection] {
        return BoundingBox.nonMaxSuppression(
            detections: detections.filter { $0.score >= scoreThreshold },
            iouThreshold: iouThreshold
        )
    }
}

// MARK: - Confidence Filtering

/// Utilities for filtering detections based on confidence scores.
public enum ConfidenceFilter {
    
    /// Filter detections by minimum confidence score.
    ///
    /// - Parameters:
    ///   - detections: Array of detections to filter
    ///   - minConfidence: Minimum confidence threshold
    /// - Returns: Filtered detections
    public static func filter(
        detections: [Detection],
        minConfidence: Float
    ) -> [Detection] {
        return detections.filter { $0.score >= minConfidence }
    }
    
    /// Filter detections with class-specific thresholds.
    ///
    /// Different classes may require different confidence thresholds
    /// based on their detection difficulty or importance.
    ///
    /// - Parameters:
    ///   - detections: Array of detections to filter
    ///   - thresholds: Dictionary mapping class index to minimum threshold
    ///   - defaultThreshold: Threshold for classes not in dictionary
    /// - Returns: Filtered detections
    ///
    /// ## Example
    /// ```swift
    /// let thresholds: [Int: Float] = [
    ///     0: 0.7,  // person - high threshold
    ///     2: 0.5,  // car - medium threshold
    ///     67: 0.3  // cell phone - low threshold (small objects)
    /// ]
    /// let filtered = ConfidenceFilter.filterWithClassThresholds(
    ///     detections: detections,
    ///     thresholds: thresholds,
    ///     defaultThreshold: 0.5
    /// )
    /// ```
    public static func filterWithClassThresholds(
        detections: [Detection],
        thresholds: [Int: Float],
        defaultThreshold: Float = 0.5
    ) -> [Detection] {
        return detections.filter { detection in
            let threshold = thresholds[detection.classIndex] ?? defaultThreshold
            return detection.score >= threshold
        }
    }
    
    /// Apply dynamic thresholding based on score distribution.
    ///
    /// Useful when you want to keep a relative percentage of top detections
    /// rather than using a fixed threshold.
    ///
    /// - Parameters:
    ///   - detections: Array of detections to filter
    ///   - keepRatio: Fraction of top-scoring detections to keep (0.0 - 1.0)
    ///   - minKeep: Minimum number of detections to keep regardless of ratio
    ///   - maxKeep: Maximum number of detections to return
    /// - Returns: Filtered detections sorted by score
    public static func filterByRatio(
        detections: [Detection],
        keepRatio: Float,
        minKeep: Int = 1,
        maxKeep: Int = 300
    ) -> [Detection] {
        let sorted = detections.sorted { $0.score > $1.score }
        let keepCount = max(minKeep, min(maxKeep, Int(Float(sorted.count) * keepRatio)))
        return Array(sorted.prefix(keepCount))
    }
}

// MARK: - Mask Utilities

/// Utilities for processing segmentation masks.
///
/// Handles common post-processing operations for semantic and instance
/// segmentation model outputs.
public enum MaskUtilities {
    
    /// Result of mask processing.
    public struct MaskResult: Sendable {
        /// Processed mask data
        public let mask: [Float]
        /// Width of the mask
        public let width: Int
        /// Height of the mask
        public let height: Int
        /// Processing time in milliseconds
        public let processingTimeMs: Double
    }
    
    /// Resize a segmentation mask using nearest-neighbor interpolation.
    ///
    /// Nearest-neighbor is preferred for masks to preserve sharp class boundaries.
    ///
    /// - Parameters:
    ///   - mask: Input mask data (flattened HxW)
    ///   - sourceWidth: Original width
    ///   - sourceHeight: Original height
    ///   - targetWidth: Target width
    ///   - targetHeight: Target height
    /// - Returns: Resized mask
    public static func resizeMask(
        mask: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> MaskResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var result = [Float](repeating: 0, count: targetWidth * targetHeight)
        
        let scaleX = Float(sourceWidth) / Float(targetWidth)
        let scaleY = Float(sourceHeight) / Float(targetHeight)
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIdx = srcY * sourceWidth + srcX
                let dstIdx = y * targetWidth + x
                
                if srcIdx < mask.count && dstIdx < result.count {
                    result[dstIdx] = mask[srcIdx]
                }
            }
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return MaskResult(
            mask: result,
            width: targetWidth,
            height: targetHeight,
            processingTimeMs: processingTime
        )
    }
    
    /// Resize a segmentation mask using bilinear interpolation.
    ///
    /// Use for probability maps where smooth transitions are acceptable.
    ///
    /// - Parameters:
    ///   - mask: Input mask data
    ///   - sourceWidth: Original width
    ///   - sourceHeight: Original height
    ///   - targetWidth: Target width
    ///   - targetHeight: Target height
    /// - Returns: Resized mask with interpolated values
    public static func resizeMaskBilinear(
        mask: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> MaskResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var result = [Float](repeating: 0, count: targetWidth * targetHeight)
        
        let scaleX = Float(sourceWidth - 1) / Float(targetWidth - 1)
        let scaleY = Float(sourceHeight - 1) / Float(targetHeight - 1)
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let srcX = Float(x) * scaleX
                let srcY = Float(y) * scaleY
                
                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, sourceWidth - 1)
                let y1 = min(y0 + 1, sourceHeight - 1)
                
                let xFrac = srcX - Float(x0)
                let yFrac = srcY - Float(y0)
                
                let v00 = mask[y0 * sourceWidth + x0]
                let v10 = mask[y0 * sourceWidth + x1]
                let v01 = mask[y1 * sourceWidth + x0]
                let v11 = mask[y1 * sourceWidth + x1]
                
                let top = v00 * (1 - xFrac) + v10 * xFrac
                let bottom = v01 * (1 - xFrac) + v11 * xFrac
                let value = top * (1 - yFrac) + bottom * yFrac
                
                result[y * targetWidth + x] = value
            }
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return MaskResult(
            mask: result,
            width: targetWidth,
            height: targetHeight,
            processingTimeMs: processingTime
        )
    }
    
    /// Apply threshold to convert probability mask to binary mask.
    ///
    /// - Parameters:
    ///   - mask: Input probability mask (values 0-1)
    ///   - threshold: Threshold value (pixels >= threshold become 1, others become 0)
    /// - Returns: Binary mask
    public static func threshold(
        mask: [Float],
        threshold: Float = 0.5
    ) -> [Float] {
        return mask.map { $0 >= threshold ? 1.0 : 0.0 }
    }
    
    /// Apply argmax across class dimension to get per-pixel class predictions.
    ///
    /// For semantic segmentation outputs with shape [C, H, W] or [H, W, C].
    ///
    /// - Parameters:
    ///   - tensor: Segmentation output tensor
    ///   - shape: Shape of the tensor
    ///   - layout: Data layout (CHW or HWC)
    /// - Returns: 2D mask with class indices per pixel
    public static func argmaxMask(
        tensor: [Float],
        shape: [Int],
        layout: DataLayout = .chw
    ) -> [Int] {
        guard shape.count == 3 else { return [] }
        
        let (channels, height, width): (Int, Int, Int)
        switch layout {
        case .chw, .nchw:
            channels = shape[0]
            height = shape[1]
            width = shape[2]
        case .hwc, .nhwc:
            height = shape[0]
            width = shape[1]
            channels = shape[2]
        }
        
        var result = [Int](repeating: 0, count: height * width)
        
        for y in 0..<height {
            for x in 0..<width {
                var maxValue: Float = -.greatestFiniteMagnitude
                var maxClass = 0
                
                for c in 0..<channels {
                    let idx: Int
                    switch layout {
                    case .chw, .nchw:
                        idx = c * height * width + y * width + x
                    case .hwc, .nhwc:
                        idx = y * width * channels + x * channels + c
                    }
                    
                    if tensor[idx] > maxValue {
                        maxValue = tensor[idx]
                        maxClass = c
                    }
                }
                
                result[y * width + x] = maxClass
            }
        }
        
        return result
    }
    
    /// Compute mask area (number of non-zero pixels).
    ///
    /// - Parameter mask: Binary or soft mask
    /// - Returns: Number of pixels above 0.5 threshold
    public static func computeArea(mask: [Float]) -> Int {
        return mask.filter { $0 >= 0.5 }.count
    }
    
    /// Compute intersection over union between two masks.
    ///
    /// - Parameters:
    ///   - mask1: First binary mask
    ///   - mask2: Second binary mask
    /// - Returns: IoU value between 0 and 1
    public static func maskIoU(mask1: [Float], mask2: [Float]) -> Float {
        guard mask1.count == mask2.count, !mask1.isEmpty else { return 0 }
        
        var intersection = 0
        var union = 0
        
        for i in 0..<mask1.count {
            let a = mask1[i] >= 0.5
            let b = mask2[i] >= 0.5
            
            if a && b { intersection += 1 }
            if a || b { union += 1 }
        }
        
        return union > 0 ? Float(intersection) / Float(union) : 0
    }
}

// MARK: - CoreML MLMultiArray Conversion

#if canImport(CoreML)

/// Utilities for converting between SwiftPixelUtils data and CoreML MLMultiArray.
///
/// Provides seamless integration with CoreML models by converting
/// preprocessed pixel data to MLMultiArray format and vice versa.
public enum CoreMLConversion {
    
    /// Create an MLMultiArray from preprocessed pixel data.
    ///
    /// This is the primary integration point for using SwiftPixelUtils
    /// with CoreML models.
    ///
    /// - Parameters:
    ///   - data: Preprocessed float array from PixelExtractor
    ///   - shape: Shape of the tensor (e.g., [1, 3, 224, 224] for NCHW)
    /// - Returns: MLMultiArray suitable for CoreML model input
    /// - Throws: Error if MLMultiArray creation fails
    ///
    /// ## Example
    /// ```swift
    /// // Preprocess image
    /// let result = try await PixelExtractor.getPixelData(
    ///     source: .cgImage(image),
    ///     options: ModelPresets.mobileNetV3.options
    /// )
    ///
    /// // Convert to MLMultiArray for CoreML
    /// let inputArray = try CoreMLConversion.toMLMultiArray(
    ///     data: result.data,
    ///     shape: result.shape
    /// )
    ///
    /// // Use with CoreML model
    /// let prediction = try model.prediction(input: inputArray)
    /// ```
    public static func toMLMultiArray(
        data: [Float],
        shape: [Int]
    ) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let multiArray = try MLMultiArray(shape: nsShape, dataType: .float32)
        
        // Copy data to MLMultiArray
        let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: data.count)
        for i in 0..<data.count {
            pointer[i] = data[i]
        }
        
        return multiArray
    }
    
    /// Convert MLMultiArray to Float array.
    ///
    /// Useful for post-processing CoreML model outputs with SwiftPixelUtils.
    ///
    /// - Parameter multiArray: CoreML MLMultiArray
    /// - Returns: Tuple of (data array, shape)
    public static func fromMLMultiArray(_ multiArray: MLMultiArray) -> (data: [Float], shape: [Int]) {
        let count = multiArray.count
        let shape = multiArray.shape.map { $0.intValue }
        
        var data = [Float](repeating: 0, count: count)
        let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        
        for i in 0..<count {
            data[i] = pointer[i]
        }
        
        return (data, shape)
    }
    
    /// Create an MLMultiArray from a PixelDataResult.
    ///
    /// Convenience method that extracts shape information automatically.
    ///
    /// - Parameter result: Result from PixelExtractor.getPixelData
    /// - Returns: MLMultiArray ready for CoreML inference
    /// - Throws: Error if conversion fails
    public static func toMLMultiArray(from result: PixelDataResult) throws -> MLMultiArray {
        return try toMLMultiArray(data: result.data, shape: result.shape)
    }
    
    /// Convert model output MLMultiArray to class probabilities.
    ///
    /// - Parameters:
    ///   - output: Model output MLMultiArray
    ///   - applySoftmax: Whether to apply softmax (for logit outputs)
    /// - Returns: Array of probabilities
    public static func toProbabilities(
        _ output: MLMultiArray,
        applySoftmax: Bool = false
    ) -> [Float] {
        let (data, _) = fromMLMultiArray(output)
        return applySoftmax ? ActivationFunctions.softmax(data) : data
    }
}

#endif

// MARK: - CVPixelBuffer Utilities

import CoreVideo

/// Optimized utilities for CVPixelBuffer processing.
///
/// CVPixelBuffer is commonly used in camera capture pipelines and
/// video processing. These utilities provide efficient conversion
/// to formats suitable for ML inference.
public enum CVPixelBufferUtilities {
    
    /// Result of CVPixelBuffer to tensor conversion.
    public struct ConversionResult: Sendable {
        /// Preprocessed tensor data
        public let data: [Float]
        /// Width of the processed image
        public let width: Int
        /// Height of the processed image
        public let height: Int
        /// Processing time in milliseconds
        public let processingTimeMs: Double
    }
    
    /// Convert CVPixelBuffer directly to normalized tensor data.
    ///
    /// This is optimized for real-time camera inference by avoiding
    /// intermediate image formats where possible.
    ///
    /// - Parameters:
    ///   - pixelBuffer: CVPixelBuffer from camera or video
    ///   - targetWidth: Target width for model input (nil to keep original)
    ///   - targetHeight: Target height for model input (nil to keep original)
    ///   - normalization: Normalization to apply
    ///   - colorFormat: Target color format
    /// - Returns: Conversion result with tensor data
    /// - Throws: Error if conversion fails
    ///
    /// ## Example
    /// ```swift
    /// // In AVCaptureVideoDataOutputSampleBufferDelegate
    /// func captureOutput(_ output: AVCaptureOutput, 
    ///                    didOutput sampleBuffer: CMSampleBuffer,
    ///                    from connection: AVCaptureConnection) {
    ///     guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    ///     
    ///     let result = try CVPixelBufferUtilities.toTensorData(
    ///         pixelBuffer,
    ///         targetWidth: 640,
    ///         targetHeight: 640,
    ///         normalization: .zeroToOne,
    ///         colorFormat: .rgb
    ///     )
    ///     
    ///     // Use result.data for inference
    /// }
    /// ```
    public static func toTensorData(
        _ pixelBuffer: CVPixelBuffer,
        targetWidth: Int? = nil,
        targetHeight: Int? = nil,
        normalization: Normalization = .scale,
        colorFormat: ColorFormat = .rgb
    ) throws -> ConversionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw PixelUtilsError.processingFailed("Cannot access pixel buffer base address")
        }
        
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        // Extract RGBA data based on pixel format
        let rgbaData: [UInt8]
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            rgbaData = extractBGRA(baseAddress: baseAddress, width: sourceWidth, height: sourceHeight, bytesPerRow: bytesPerRow)
        case kCVPixelFormatType_32RGBA:
            rgbaData = extractRGBA(baseAddress: baseAddress, width: sourceWidth, height: sourceHeight, bytesPerRow: bytesPerRow)
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            rgbaData = try extractYUV420BiPlanar(pixelBuffer: pixelBuffer)
        default:
            throw PixelUtilsError.invalidOptions("Unsupported pixel format: \(pixelFormat)")
        }
        
        // Resize if needed
        let finalWidth = targetWidth ?? sourceWidth
        let finalHeight = targetHeight ?? sourceHeight
        
        var processedRGBA = rgbaData
        if finalWidth != sourceWidth || finalHeight != sourceHeight {
            processedRGBA = resizeRGBA(
                data: rgbaData,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                targetWidth: finalWidth,
                targetHeight: finalHeight
            )
        }
        
        // Convert to float and apply normalization
        let channels = colorFormat.channelCount
        var floatData = [Float](repeating: 0, count: finalWidth * finalHeight * channels)
        
        for i in 0..<(finalWidth * finalHeight) {
            let r = Float(processedRGBA[i * 4]) / 255.0
            let g = Float(processedRGBA[i * 4 + 1]) / 255.0
            let b = Float(processedRGBA[i * 4 + 2]) / 255.0
            
            switch colorFormat {
            case .rgb:
                floatData[i * 3] = r
                floatData[i * 3 + 1] = g
                floatData[i * 3 + 2] = b
            case .bgr:
                floatData[i * 3] = b
                floatData[i * 3 + 1] = g
                floatData[i * 3 + 2] = r
            case .rgba:
                let a = Float(processedRGBA[i * 4 + 3]) / 255.0
                floatData[i * 4] = r
                floatData[i * 4 + 1] = g
                floatData[i * 4 + 2] = b
                floatData[i * 4 + 3] = a
            case .bgra:
                let a = Float(processedRGBA[i * 4 + 3]) / 255.0
                floatData[i * 4] = b
                floatData[i * 4 + 1] = g
                floatData[i * 4 + 2] = r
                floatData[i * 4 + 3] = a
            case .grayscale:
                floatData[i] = 0.299 * r + 0.587 * g + 0.114 * b
            default:
                floatData[i * 3] = r
                floatData[i * 3 + 1] = g
                floatData[i * 3 + 2] = b
            }
        }
        
        // Apply normalization
        floatData = applyNormalization(floatData, normalization: normalization, channels: channels)
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return ConversionResult(
            data: floatData,
            width: finalWidth,
            height: finalHeight,
            processingTimeMs: processingTime
        )
    }
    
    /// Get pixel format description for a CVPixelBuffer.
    ///
    /// - Parameter pixelBuffer: CVPixelBuffer to inspect
    /// - Returns: Human-readable format description
    public static func getPixelFormatDescription(_ pixelBuffer: CVPixelBuffer) -> String {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        switch format {
        case kCVPixelFormatType_32BGRA: return "32BGRA"
        case kCVPixelFormatType_32RGBA: return "32RGBA"
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return "420YpCbCr8BiPlanarFullRange"
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return "420YpCbCr8BiPlanarVideoRange"
        default: return "Unknown (\(format))"
        }
    }
    
    // MARK: Private Helpers
    
    private static func extractBGRA(baseAddress: UnsafeMutableRawPointer, width: Int, height: Int, bytesPerRow: Int) -> [UInt8] {
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let dstIdx = (y * width + x) * 4
                rgba[dstIdx] = ptr[srcIdx + 2]     // R
                rgba[dstIdx + 1] = ptr[srcIdx + 1] // G
                rgba[dstIdx + 2] = ptr[srcIdx]     // B
                rgba[dstIdx + 3] = ptr[srcIdx + 3] // A
            }
        }
        
        return rgba
    }
    
    private static func extractRGBA(baseAddress: UnsafeMutableRawPointer, width: Int, height: Int, bytesPerRow: Int) -> [UInt8] {
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let dstIdx = (y * width + x) * 4
                rgba[dstIdx] = ptr[srcIdx]
                rgba[dstIdx + 1] = ptr[srcIdx + 1]
                rgba[dstIdx + 2] = ptr[srcIdx + 2]
                rgba[dstIdx + 3] = ptr[srcIdx + 3]
            }
        }
        
        return rgba
    }
    
    private static func extractYUV420BiPlanar(pixelBuffer: CVPixelBuffer) throws -> [UInt8] {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 else {
            throw PixelUtilsError.processingFailed("Expected bi-planar format")
        }
        
        guard let yPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
              let uvPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
            throw PixelUtilsError.processingFailed("Cannot access YUV planes")
        }
        
        let yPtr = yPlane.assumingMemoryBound(to: UInt8.self)
        let uvPtr = uvPlane.assumingMemoryBound(to: UInt8.self)
        
        let yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let uvBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
        
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let yIdx = y * yBytesPerRow + x
                let uvIdx = (y / 2) * uvBytesPerRow + (x / 2) * 2
                
                let yValue = Float(yPtr[yIdx])
                let uValue = Float(uvPtr[uvIdx]) - 128
                let vValue = Float(uvPtr[uvIdx + 1]) - 128
                
                // YUV to RGB conversion (BT.601)
                var r = yValue + 1.402 * vValue
                var g = yValue - 0.344 * uValue - 0.714 * vValue
                var b = yValue + 1.772 * uValue
                
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                let idx = (y * width + x) * 4
                rgba[idx] = UInt8(r)
                rgba[idx + 1] = UInt8(g)
                rgba[idx + 2] = UInt8(b)
                rgba[idx + 3] = 255
            }
        }
        
        return rgba
    }
    
    private static func resizeRGBA(data: [UInt8], sourceWidth: Int, sourceHeight: Int, targetWidth: Int, targetHeight: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        
        let scaleX = Float(sourceWidth) / Float(targetWidth)
        let scaleY = Float(sourceHeight) / Float(targetHeight)
        
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIdx = (srcY * sourceWidth + srcX) * 4
                let dstIdx = (y * targetWidth + x) * 4
                
                result[dstIdx] = data[srcIdx]
                result[dstIdx + 1] = data[srcIdx + 1]
                result[dstIdx + 2] = data[srcIdx + 2]
                result[dstIdx + 3] = data[srcIdx + 3]
            }
        }
        
        return result
    }
    
    private static func applyNormalization(_ data: [Float], normalization: Normalization, channels: Int) -> [Float] {
        var result = data
        
        switch normalization.preset {
        case .raw:
            // No normalization, but data is already 0-1, so multiply back to 0-255
            for i in 0..<result.count {
                result[i] = result[i] * 255.0
            }
        case .scale:
            break // Already in 0-1 range
        case .tensorflow:
            // Map [0, 1] to [-1, 1]
            for i in 0..<result.count {
                result[i] = result[i] * 2.0 - 1.0
            }
        case .imagenet:
            let mean = normalization.mean ?? [0.485, 0.456, 0.406]
            let std = normalization.std ?? [0.229, 0.224, 0.225]
            let pixelCount = result.count / channels
            for i in 0..<pixelCount {
                for c in 0..<min(channels, mean.count) {
                    let idx = i * channels + c
                    result[idx] = (result[idx] - mean[c]) / std[c]
                }
            }
        case .custom:
            if let mean = normalization.mean, let std = normalization.std {
                let pixelCount = result.count / channels
                for i in 0..<pixelCount {
                    for c in 0..<min(channels, mean.count) {
                        let idx = i * channels + c
                        let m = c < mean.count ? mean[c] : 0
                        let s = c < std.count ? std[c] : 1
                        result[idx] = (result[idx] - m) / s
                    }
                }
            }
        }
        
        return result
    }
}
