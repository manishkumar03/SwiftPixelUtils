import Foundation
import Accelerate

// MARK: - Activation Functions

/// Collection of activation functions for neural network output post-processing
public enum ActivationFunctions {
    
    // MARK: - Softmax
    
    /// Applies softmax activation to convert logits to probabilities.
    /// - Parameters:
    ///   - logits: Raw model output values
    ///   - temperature: Temperature for softmax (default 1.0). Higher values create more uniform distributions.
    /// - Returns: Probability distribution that sums to 1.0
    public static func softmax(_ logits: [Float], temperature: Float = 1.0) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        var scaledLogits = logits
        if temperature != 1.0 {
            var temp = temperature
            vDSP_vsdiv(logits, 1, &temp, &scaledLogits, 1, vDSP_Length(logits.count))
        }
        
        // Subtract max for numerical stability
        var maxVal: Float = 0
        vDSP_maxv(scaledLogits, 1, &maxVal, vDSP_Length(scaledLogits.count))
        
        var negMax = -maxVal
        var shifted = [Float](repeating: 0, count: scaledLogits.count)
        vDSP_vsadd(scaledLogits, 1, &negMax, &shifted, 1, vDSP_Length(scaledLogits.count))
        
        // Compute exp
        var count = Int32(shifted.count)
        var expValues = [Float](repeating: 0, count: shifted.count)
        vvexpf(&expValues, shifted, &count)
        
        // Sum and divide
        var sum: Float = 0
        vDSP_sve(expValues, 1, &sum, vDSP_Length(expValues.count))
        
        var result = [Float](repeating: 0, count: expValues.count)
        vDSP_vsdiv(expValues, 1, &sum, &result, 1, vDSP_Length(expValues.count))
        
        return result
    }
    
    /// Applies softmax along a specific axis of a multi-dimensional array.
    /// - Parameters:
    ///   - logits: Flattened array of logits
    ///   - shape: Shape of the array (e.g., [batch, classes] or [height, width, classes])
    ///   - axis: Axis along which to apply softmax
    ///   - temperature: Temperature for softmax
    /// - Returns: Probabilities with same shape as input
    public static func softmaxAlongAxis(_ logits: [Float], shape: [Int], axis: Int, temperature: Float = 1.0) -> [Float] {
        guard !logits.isEmpty, !shape.isEmpty else { return [] }
        
        let normalizedAxis = axis < 0 ? shape.count + axis : axis
        guard normalizedAxis >= 0, normalizedAxis < shape.count else { return logits }
        
        let totalElements = shape.reduce(1, *)
        guard logits.count == totalElements else { return logits }
        
        var result = [Float](repeating: 0, count: logits.count)
        
        // Calculate strides
        var strides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        
        let axisSize = shape[normalizedAxis]
        let axisStride = strides[normalizedAxis]
        let outerSize = strides[0] * shape[0] / (axisSize * axisStride)
        
        // Process each slice along the axis
        for outer in 0..<outerSize {
            for inner in 0..<axisStride {
                // Extract values along axis
                var slice = [Float](repeating: 0, count: axisSize)
                for i in 0..<axisSize {
                    let idx = outer * axisSize * axisStride + i * axisStride + inner
                    slice[i] = logits[idx]
                }
                
                // Apply softmax to slice
                let softmaxSlice = softmax(slice, temperature: temperature)
                
                // Put back
                for i in 0..<axisSize {
                    let idx = outer * axisSize * axisStride + i * axisStride + inner
                    result[idx] = softmaxSlice[i]
                }
            }
        }
        
        return result
    }
    
    // MARK: - Sigmoid
    
    /// Applies sigmoid activation to a single value.
    /// - Parameter x: Input value
    /// - Returns: Output in range (0, 1)
    public static func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    /// Applies sigmoid activation element-wise to an array.
    /// - Parameter logits: Input values
    /// - Returns: Outputs in range (0, 1)
    public static func sigmoid(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        var negated = [Float](repeating: 0, count: logits.count)
        var negOne: Float = -1.0
        vDSP_vsmul(logits, 1, &negOne, &negated, 1, vDSP_Length(logits.count))
        
        var count = Int32(negated.count)
        var expValues = [Float](repeating: 0, count: negated.count)
        vvexpf(&expValues, negated, &count)
        
        var one: Float = 1.0
        var onePlusExp = [Float](repeating: 0, count: expValues.count)
        vDSP_vsadd(expValues, 1, &one, &onePlusExp, 1, vDSP_Length(expValues.count))
        
        var result = [Float](repeating: 0, count: onePlusExp.count)
        vDSP_svdiv(&one, onePlusExp, 1, &result, 1, vDSP_Length(onePlusExp.count))
        
        return result
    }
    
    // MARK: - Log Softmax
    
    /// Applies log-softmax for numerical stability in loss calculations.
    /// - Parameters:
    ///   - logits: Raw model output values
    ///   - temperature: Temperature parameter
    /// - Returns: Log probabilities
    public static func logSoftmax(_ logits: [Float], temperature: Float = 1.0) -> [Float] {
        guard !logits.isEmpty else { return [] }
        
        var scaledLogits = logits
        if temperature != 1.0 {
            var temp = temperature
            vDSP_vsdiv(logits, 1, &temp, &scaledLogits, 1, vDSP_Length(logits.count))
        }
        
        // max for stability
        var maxVal: Float = 0
        vDSP_maxv(scaledLogits, 1, &maxVal, vDSP_Length(scaledLogits.count))
        
        // shifted = logits - max
        var negMax = -maxVal
        var shifted = [Float](repeating: 0, count: scaledLogits.count)
        vDSP_vsadd(scaledLogits, 1, &negMax, &shifted, 1, vDSP_Length(scaledLogits.count))
        
        // exp(shifted)
        var count = Int32(shifted.count)
        var expValues = [Float](repeating: 0, count: shifted.count)
        vvexpf(&expValues, shifted, &count)
        
        // sum(exp)
        var sumExp: Float = 0
        vDSP_sve(expValues, 1, &sumExp, vDSP_Length(expValues.count))
        
        // log(sum(exp))
        let logSumExp = log(sumExp)
        
        // result = shifted - log(sum(exp))
        var negLogSumExp = -logSumExp
        var result = [Float](repeating: 0, count: shifted.count)
        vDSP_vsadd(shifted, 1, &negLogSumExp, &result, 1, vDSP_Length(shifted.count))
        
        return result
    }
}
