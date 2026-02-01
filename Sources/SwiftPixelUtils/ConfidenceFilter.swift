import Foundation

// MARK: - Confidence Filtering

/// Utilities for filtering detections by confidence thresholds
public enum ConfidenceFilter {
    
    /// Filters detections by a single confidence threshold.
    /// - Parameters:
    ///   - detections: Array of detection tuples (index, classId, confidence)
    ///   - threshold: Minimum confidence to keep
    /// - Returns: Filtered detections
    public static func filter<T>(
        _ detections: [(item: T, confidence: Float)],
        threshold: Float
    ) -> [(item: T, confidence: Float)] {
        return detections.filter { $0.confidence >= threshold }
    }
    
    /// Filters with per-class thresholds.
    /// - Parameters:
    ///   - detections: Array of (item, classId, confidence) tuples
    ///   - classThresholds: Dictionary mapping class IDs to thresholds
    ///   - defaultThreshold: Threshold for classes not in dictionary
    /// - Returns: Filtered detections
    public static func filterWithClassThresholds<T>(
        _ detections: [(item: T, classId: Int, confidence: Float)],
        classThresholds: [Int: Float],
        defaultThreshold: Float = 0.5
    ) -> [(item: T, classId: Int, confidence: Float)] {
        return detections.filter { detection in
            let threshold = classThresholds[detection.classId] ?? defaultThreshold
            return detection.confidence >= threshold
        }
    }
    
    /// Filters to keep only the top percentage of detections by confidence.
    /// - Parameters:
    ///   - detections: Array of (item, confidence) tuples
    ///   - ratio: Fraction of detections to keep (0.0 to 1.0)
    /// - Returns: Top detections by confidence
    public static func filterByRatio<T>(
        _ detections: [(item: T, confidence: Float)],
        ratio: Float
    ) -> [(item: T, confidence: Float)] {
        guard !detections.isEmpty, ratio > 0 else { return [] }
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        let keepCount = max(1, Int(Float(detections.count) * min(1.0, ratio)))
        
        return Array(sorted.prefix(keepCount))
    }
}
