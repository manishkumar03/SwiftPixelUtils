import Foundation
import Accelerate

// MARK: - Top-K Extraction

/// Utilities for extracting top predictions from model outputs
public enum TopKExtractor {
    
    /// Result of top-K extraction
    public struct TopKResult {
        /// Indices of top-K elements
        public let indices: [Int]
        /// Values of top-K elements
        public let values: [Float]
        /// Softmax probabilities if requested
        public let probabilities: [Float]?
        
        public init(indices: [Int], values: [Float], probabilities: [Float]? = nil) {
            self.indices = indices
            self.values = values
            self.probabilities = probabilities
        }
    }
    
    /// Extracts the top-K highest values and their indices.
    /// - Parameters:
    ///   - values: Array of values to search
    ///   - k: Number of top elements to return
    /// - Returns: TopKResult containing indices and values
    public static func extractTopK(_ values: [Float], k: Int) -> TopKResult {
        guard !values.isEmpty else {
            return TopKResult(indices: [], values: [])
        }
        
        let actualK = min(k, values.count)
        
        // Create indexed array and sort
        let indexed = values.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        
        let topK = Array(sorted.prefix(actualK))
        let indices = topK.map { $0.0 }
        let topValues = topK.map { $0.1 }
        
        return TopKResult(indices: indices, values: topValues)
    }
    
    /// Extracts top-K with softmax probabilities applied.
    /// - Parameters:
    ///   - logits: Raw model output logits
    ///   - k: Number of top elements
    ///   - temperature: Softmax temperature
    /// - Returns: TopKResult with probabilities
    public static func extractTopKWithSoftmax(_ logits: [Float], k: Int, temperature: Float = 1.0) -> TopKResult {
        let probabilities = ActivationFunctions.softmax(logits, temperature: temperature)
        let topK = extractTopK(probabilities, k: k)
        
        return TopKResult(
            indices: topK.indices,
            values: topK.values,
            probabilities: topK.values
        )
    }
    
    /// Returns the index of the maximum value.
    /// - Parameter values: Array to search
    /// - Returns: Index of maximum, or nil if empty
    public static func argmax(_ values: [Float]) -> Int? {
        guard !values.isEmpty else { return nil }
        
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(values, 1, &maxVal, &maxIdx, vDSP_Length(values.count))
        
        return Int(maxIdx)
    }
    
    /// Returns the index of the minimum value.
    /// - Parameter values: Array to search
    /// - Returns: Index of minimum, or nil if empty
    public static func argmin(_ values: [Float]) -> Int? {
        guard !values.isEmpty else { return nil }
        
        var minVal: Float = 0
        var minIdx: vDSP_Length = 0
        vDSP_minvi(values, 1, &minVal, &minIdx, vDSP_Length(values.count))
        
        return Int(minIdx)
    }
}
