import XCTest
@testable import SwiftPixelUtils

/// Tests for ``ActivationFunctions`` - Neural network activation functions.
///
/// ## Topics
///
/// ### Softmax Tests
/// - Validates probability distribution output sums to 1.0
/// - Tests empty input, single element, and numerical stability
/// - Verifies temperature parameter effects
///
/// ### Sigmoid Tests
/// - Tests single values and array operations
/// - Validates output range [0, 1]
/// - Checks symmetry property: σ(-x) = 1 - σ(x)
///
/// ### Log-Softmax Tests
/// - Verifies log probabilities are negative
/// - Tests consistency with softmax: exp(log_softmax) ≈ softmax
final class ActivationFunctionsTests: XCTestCase {
    
    // MARK: - Softmax Tests
    
    func testSoftmaxBasic() {
        let logits: [Float] = [1, 2, 3]
        let probs = ActivationFunctions.softmax(logits)
        
        XCTAssertEqual(probs.count, 3)
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 0.001)
        
        // Highest logit should have highest probability
        XCTAssertGreaterThan(probs[2], probs[1])
        XCTAssertGreaterThan(probs[1], probs[0])
    }
    
    func testSoftmaxEmpty() {
        let probs = ActivationFunctions.softmax([])
        XCTAssertTrue(probs.isEmpty)
    }
    
    func testSoftmaxSingleElement() {
        let probs = ActivationFunctions.softmax([5.0])
        XCTAssertEqual(probs.count, 1)
        XCTAssertEqual(probs[0], 1.0, accuracy: 0.001)
    }
    
    func testSoftmaxEqualInputs() {
        let logits: [Float] = [2, 2, 2, 2]
        let probs = ActivationFunctions.softmax(logits)
        
        for prob in probs {
            XCTAssertEqual(prob, 0.25, accuracy: 0.001)
        }
    }
    
    func testSoftmaxNumericalStability() {
        // Large values should not overflow
        let logits: [Float] = [1000, 1001, 1002]
        let probs = ActivationFunctions.softmax(logits)
        
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 0.001)
        XCTAssertFalse(probs.contains(where: { $0.isNaN || $0.isInfinite }))
    }
    
    func testSoftmaxNegativeValues() {
        let logits: [Float] = [-5, -2, -1]
        let probs = ActivationFunctions.softmax(logits)
        
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 0.001)
        XCTAssertGreaterThan(probs[2], probs[0])
    }
    
    func testSoftmaxWithTemperature() {
        let logits: [Float] = [1, 2, 3]
        
        let probs1 = ActivationFunctions.softmax(logits, temperature: 1.0)
        let probs2 = ActivationFunctions.softmax(logits, temperature: 2.0)
        
        // Higher temperature = more uniform distribution
        let gap1 = probs1[2] - probs1[0]
        let gap2 = probs2[2] - probs2[0]
        
        XCTAssertLessThan(gap2, gap1)
    }
    
    func testSoftmaxLowTemperature() {
        let logits: [Float] = [1, 2, 3]
        let probs = ActivationFunctions.softmax(logits, temperature: 0.5)
        
        // Low temperature = more peaked distribution
        XCTAssertGreaterThan(probs[2], 0.8)
    }
    
    func testSoftmaxZeroTemperature() {
        // Very low temperature approaches argmax
        let logits: [Float] = [1, 5, 2]
        let probs = ActivationFunctions.softmax(logits, temperature: 0.1)
        
        XCTAssertGreaterThan(probs[1], 0.99)
    }
    
    // MARK: - Softmax Along Axis Tests
    
    func testSoftmaxAlongAxisLastAxis() {
        // 2x3 array, softmax along last axis
        let logits: [Float] = [1, 2, 3, 1, 1, 1]
        let shape = [2, 3]
        
        let probs = ActivationFunctions.softmaxAlongAxis(logits, shape: shape, axis: 1)
        
        XCTAssertEqual(probs.count, 6)
        
        // First row should sum to 1
        XCTAssertEqual(probs[0] + probs[1] + probs[2], 1.0, accuracy: 0.001)
        // Second row should sum to 1
        XCTAssertEqual(probs[3] + probs[4] + probs[5], 1.0, accuracy: 0.001)
    }
    
    func testSoftmaxAlongAxisEmpty() {
        let probs = ActivationFunctions.softmaxAlongAxis([], shape: [], axis: 0)
        XCTAssertTrue(probs.isEmpty)
    }
    
    func testSoftmaxAlongAxisInvalidAxis() {
        let logits: [Float] = [1, 2, 3]
        let probs = ActivationFunctions.softmaxAlongAxis(logits, shape: [3], axis: 5)
        
        // Should return original logits on invalid axis
        XCTAssertEqual(probs, logits)
    }
    
    func testSoftmaxAlongAxisNegativeAxis() {
        let logits: [Float] = [1, 2, 3, 4, 5, 6]
        let shape = [2, 3]
        
        let probs = ActivationFunctions.softmaxAlongAxis(logits, shape: shape, axis: -1)
        
        // axis -1 is the last axis (same as axis 1)
        XCTAssertEqual(probs[0] + probs[1] + probs[2], 1.0, accuracy: 0.001)
    }
    
    func testSoftmaxAlongAxisMismatchedShape() {
        let logits: [Float] = [1, 2, 3]
        let shape = [5]  // Claims 5 elements but only 3 provided
        
        let probs = ActivationFunctions.softmaxAlongAxis(logits, shape: shape, axis: 0)
        
        // Should return original on mismatch
        XCTAssertEqual(probs, logits)
    }
    
    // MARK: - Sigmoid Tests
    
    func testSigmoidSingleValue() {
        let result = ActivationFunctions.sigmoid(0.0)
        XCTAssertEqual(result, 0.5, accuracy: 0.001)
    }
    
    func testSigmoidPositive() {
        let result = ActivationFunctions.sigmoid(10.0)
        XCTAssertGreaterThan(result, 0.99)
    }
    
    func testSigmoidNegative() {
        let result = ActivationFunctions.sigmoid(-10.0)
        XCTAssertLessThan(result, 0.01)
    }
    
    func testSigmoidSymmetry() {
        let pos = ActivationFunctions.sigmoid(2.0)
        let neg = ActivationFunctions.sigmoid(-2.0)
        
        XCTAssertEqual(pos + neg, 1.0, accuracy: 0.001)
    }
    
    func testSigmoidArray() {
        let logits: [Float] = [-10, 0, 10]
        let results = ActivationFunctions.sigmoid(logits)
        
        XCTAssertEqual(results.count, 3)
        XCTAssertLessThan(results[0], 0.01)
        XCTAssertEqual(results[1], 0.5, accuracy: 0.001)
        XCTAssertGreaterThan(results[2], 0.99)
    }
    
    func testSigmoidArrayEmpty() {
        let results = ActivationFunctions.sigmoid([])
        XCTAssertTrue(results.isEmpty)
    }
    
    func testSigmoidArrayRange() {
        let logits: [Float] = [-5, -2, 0, 2, 5]
        let results = ActivationFunctions.sigmoid(logits)
        
        for result in results {
            XCTAssertGreaterThan(result, 0)
            XCTAssertLessThan(result, 1)
        }
    }
    
    // MARK: - Log Softmax Tests
    
    func testLogSoftmaxBasic() {
        let logits: [Float] = [1, 2, 3]
        let logProbs = ActivationFunctions.logSoftmax(logits)
        
        XCTAssertEqual(logProbs.count, 3)
        
        // All log probabilities should be negative (log of value < 1)
        for lp in logProbs {
            XCTAssertLessThanOrEqual(lp, 0)
        }
    }
    
    func testLogSoftmaxEmpty() {
        let logProbs = ActivationFunctions.logSoftmax([])
        XCTAssertTrue(logProbs.isEmpty)
    }
    
    func testLogSoftmaxConsistency() {
        let logits: [Float] = [1, 2, 3]
        
        // exp(log_softmax) should equal softmax
        let logProbs = ActivationFunctions.logSoftmax(logits)
        let probs = ActivationFunctions.softmax(logits)
        
        for i in 0..<logits.count {
            XCTAssertEqual(exp(logProbs[i]), probs[i], accuracy: 0.001)
        }
    }
    
    func testLogSoftmaxNumericalStability() {
        let logits: [Float] = [1000, 1001, 1002]
        let logProbs = ActivationFunctions.logSoftmax(logits)
        
        XCTAssertFalse(logProbs.contains(where: { $0.isNaN || $0.isInfinite }))
    }
    
    func testLogSoftmaxWithTemperature() {
        let logits: [Float] = [1, 2, 3]
        
        let logProbs1 = ActivationFunctions.logSoftmax(logits, temperature: 1.0)
        let logProbs2 = ActivationFunctions.logSoftmax(logits, temperature: 2.0)
        
        // Higher temperature should give less extreme differences
        let diff1 = logProbs1[2] - logProbs1[0]
        let diff2 = logProbs2[2] - logProbs2[0]
        
        XCTAssertLessThan(abs(diff2), abs(diff1))
    }
    
    func testLogSoftmaxSingleElement() {
        let logProbs = ActivationFunctions.logSoftmax([5.0])
        
        XCTAssertEqual(logProbs.count, 1)
        XCTAssertEqual(logProbs[0], 0, accuracy: 0.001)  // log(1) = 0
    }
    
    // MARK: - Performance Tests
    
    func testSoftmaxPerformance() {
        let logits = [Float](repeating: 1.0, count: 1000)
        
        measure {
            _ = ActivationFunctions.softmax(logits)
        }
    }
    
    func testSigmoidArrayPerformance() {
        let logits = [Float](repeating: 0.0, count: 10000)
        
        measure {
            _ = ActivationFunctions.sigmoid(logits)
        }
    }
    
    func testLogSoftmaxPerformance() {
        let logits = [Float](repeating: 1.0, count: 1000)
        
        measure {
            _ = ActivationFunctions.logSoftmax(logits)
        }
    }
}
