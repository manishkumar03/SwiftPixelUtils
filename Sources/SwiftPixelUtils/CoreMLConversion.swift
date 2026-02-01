import Foundation

#if canImport(CoreML)
import CoreML

// MARK: - CoreML Conversion Utilities

/// Utilities for converting between Swift arrays and CoreML MLMultiArray
public enum CoreMLConversion {
    
    /// Converts a Float array to MLMultiArray.
    /// - Parameters:
    ///   - array: Input float array
    ///   - shape: Shape of the multi-array
    /// - Returns: MLMultiArray ready for CoreML inference
    /// - Throws: PixelUtilsError if conversion fails
    public static func toMLMultiArray(_ array: [Float], shape: [Int]) throws -> MLMultiArray {
        let totalSize = shape.reduce(1, *)
        guard array.count == totalSize else {
            throw PixelUtilsError.processingFailed("Array size \(array.count) doesn't match shape \(shape)")
        }
        
        let nsShape = shape.map { NSNumber(value: $0) }
        guard let multiArray = try? MLMultiArray(shape: nsShape, dataType: .float32) else {
            throw PixelUtilsError.processingFailed("Failed to create MLMultiArray")
        }
        
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        for i in 0..<array.count {
            ptr[i] = array[i]
        }
        
        return multiArray
    }
    
    /// Converts MLMultiArray to Float array.
    /// - Parameter multiArray: CoreML multi-array
    /// - Returns: Flattened Float array
    public static func fromMLMultiArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var result = [Float](repeating: 0, count: count)
        
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = ptr[i]
        }
        
        return result
    }
    
    /// Converts MLMultiArray classification output to probability dictionary.
    /// - Parameters:
    ///   - multiArray: Output from classification model
    ///   - labels: Optional class labels
    /// - Returns: Dictionary mapping labels/indices to probabilities
    public static func toProbabilities(
        _ multiArray: MLMultiArray,
        labels: [String]? = nil
    ) -> [String: Float] {
        let values = fromMLMultiArray(multiArray)
        let probs = ActivationFunctions.softmax(values)
        
        var result = [String: Float]()
        for (idx, prob) in probs.enumerated() {
            let key = labels?[safe: idx] ?? "class_\(idx)"
            result[key] = prob
        }
        
        return result
    }
}

// Safe array access helper
private extension Array {
    subscript(safe index: Int) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

#endif
