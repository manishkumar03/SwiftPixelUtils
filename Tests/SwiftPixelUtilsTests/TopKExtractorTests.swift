import XCTest
@testable import SwiftPixelUtils

/// Tests for ``TopKExtractor`` - Top-K value extraction utilities.
///
/// ## Topics
///
/// ### TopKResult Tests
/// - Indices, values, and probabilities storage
/// - Proper sorting (descending by value)
///
/// ### Extraction Tests
/// - Extract top-k from classification logits
/// - Handle k > array length
/// - Empty array handling
///
/// ### Performance Tests
/// - Large array extraction benchmarks
/// - Partial sort optimization validation
///
/// ### Edge Cases
/// - Duplicate values, negative values
/// - k = 0, k = 1, k = all
final class TopKExtractorTests: XCTestCase {
    
    // MARK: - TopKResult Tests
    
    func testTopKResultInit() {
        let result = TopKExtractor.TopKResult(
            indices: [0, 1, 2],
            values: [0.9, 0.08, 0.02],
            probabilities: [0.9, 0.08, 0.02]
        )
        
        XCTAssertEqual(result.indices, [0, 1, 2])
        XCTAssertEqual(result.values, [0.9, 0.08, 0.02])
        XCTAssertEqual(result.probabilities, [0.9, 0.08, 0.02])
    }
    
    func testTopKResultWithoutProbabilities() {
        let result = TopKExtractor.TopKResult(
            indices: [0],
            values: [1.0]
        )
        
        XCTAssertNil(result.probabilities)
    }
    
    // MARK: - extractTopK Tests
    
    func testExtractTopKBasic() {
        let values: [Float] = [0.1, 0.9, 0.5, 0.3]
        
        let result = TopKExtractor.extractTopK(values, k: 2)
        
        XCTAssertEqual(result.indices.count, 2)
        XCTAssertEqual(result.values.count, 2)
        XCTAssertEqual(result.indices[0], 1)  // Index of 0.9
        XCTAssertEqual(result.values[0], 0.9, accuracy: 0.001)
    }
    
    func testExtractTopKAll() {
        let values: [Float] = [0.3, 0.5, 0.2]
        
        let result = TopKExtractor.extractTopK(values, k: 10)  // k > count
        
        XCTAssertEqual(result.indices.count, 3)
        XCTAssertEqual(result.values.count, 3)
    }
    
    func testExtractTopK1() {
        let values: [Float] = [0.1, 0.9, 0.5]
        
        let result = TopKExtractor.extractTopK(values, k: 1)
        
        XCTAssertEqual(result.indices.count, 1)
        XCTAssertEqual(result.indices[0], 1)
        XCTAssertEqual(result.values[0], 0.9, accuracy: 0.001)
    }
    
    func testExtractTopKEmpty() {
        let values: [Float] = []
        
        let result = TopKExtractor.extractTopK(values, k: 5)
        
        XCTAssertTrue(result.indices.isEmpty)
        XCTAssertTrue(result.values.isEmpty)
    }
    
    func testExtractTopKSameValues() {
        let values: [Float] = [0.5, 0.5, 0.5, 0.5]
        
        let result = TopKExtractor.extractTopK(values, k: 2)
        
        XCTAssertEqual(result.indices.count, 2)
        XCTAssertEqual(result.values[0], 0.5, accuracy: 0.001)
        XCTAssertEqual(result.values[1], 0.5, accuracy: 0.001)
    }
    
    func testExtractTopKNegativeValues() {
        let values: [Float] = [-5, -2, -10, -1]
        
        let result = TopKExtractor.extractTopK(values, k: 2)
        
        XCTAssertEqual(result.indices[0], 3)  // -1 is highest
        XCTAssertEqual(result.indices[1], 1)  // -2 is second highest
    }
    
    func testExtractTopKOrdering() {
        let values: [Float] = [0.1, 0.4, 0.2, 0.3]
        
        let result = TopKExtractor.extractTopK(values, k: 4)
        
        // Should be in descending order
        XCTAssertEqual(result.indices, [1, 3, 2, 0])  // 0.4, 0.3, 0.2, 0.1
    }
    
    func testExtractTopKSingleElement() {
        let values: [Float] = [0.5]
        
        let result = TopKExtractor.extractTopK(values, k: 5)
        
        XCTAssertEqual(result.indices.count, 1)
        XCTAssertEqual(result.indices[0], 0)
    }
    
    func testExtractTopKZero() {
        let values: [Float] = [0.1, 0.9, 0.5]
        
        let result = TopKExtractor.extractTopK(values, k: 0)
        
        XCTAssertTrue(result.indices.isEmpty)
    }
    
    // MARK: - extractTopKWithSoftmax Tests
    
    func testExtractTopKWithSoftmax() {
        let logits: [Float] = [10, 1, 0.1]  // Strong preference for first
        
        let result = TopKExtractor.extractTopKWithSoftmax(logits, k: 2)
        
        XCTAssertEqual(result.indices.count, 2)
        XCTAssertEqual(result.indices[0], 0)  // Highest after softmax
        XCTAssertNotNil(result.probabilities)
        XCTAssertEqual(result.probabilities?.reduce(0, +) ?? 0 + (result.values.count > 2 ? 0 : 0), result.values.reduce(0, +), accuracy: 0.001)
    }
    
    func testExtractTopKWithSoftmaxSumsToOne() {
        let logits: [Float] = [1, 2, 3, 4, 5]
        
        let result = TopKExtractor.extractTopKWithSoftmax(logits, k: 5)
        
        // All probabilities should sum close to 1
        let sum = result.values.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
    }
    
    func testExtractTopKWithSoftmaxTemperature() {
        let logits: [Float] = [1, 2, 3]
        
        let result1 = TopKExtractor.extractTopKWithSoftmax(logits, k: 3, temperature: 1.0)
        let result2 = TopKExtractor.extractTopKWithSoftmax(logits, k: 3, temperature: 2.0)
        
        // Higher temperature = more uniform distribution
        let gap1 = result1.values[0] - result1.values[2]
        let gap2 = result2.values[0] - result2.values[2]
        
        XCTAssertLessThan(gap2, gap1)
    }
    
    func testExtractTopKWithSoftmaxEmpty() {
        let logits: [Float] = []
        
        let result = TopKExtractor.extractTopKWithSoftmax(logits, k: 5)
        
        XCTAssertTrue(result.indices.isEmpty)
    }
    
    func testExtractTopKWithSoftmaxProbabilitiesMatchValues() {
        let logits: [Float] = [1, 2, 3]
        
        let result = TopKExtractor.extractTopKWithSoftmax(logits, k: 3)
        
        // For this implementation, probabilities should equal values
        XCTAssertEqual(result.values, result.probabilities ?? [])
    }
    
    // MARK: - argmax Tests
    
    func testArgmaxBasic() {
        let values: [Float] = [0.1, 0.9, 0.5, 0.3]
        
        let index = TopKExtractor.argmax(values)
        
        XCTAssertEqual(index, 1)
    }
    
    func testArgmaxEmpty() {
        let values: [Float] = []
        
        let index = TopKExtractor.argmax(values)
        
        XCTAssertNil(index)
    }
    
    func testArgmaxSingleElement() {
        let values: [Float] = [0.5]
        
        let index = TopKExtractor.argmax(values)
        
        XCTAssertEqual(index, 0)
    }
    
    func testArgmaxNegativeValues() {
        let values: [Float] = [-10, -5, -1, -7]
        
        let index = TopKExtractor.argmax(values)
        
        XCTAssertEqual(index, 2)  // -1 is highest
    }
    
    func testArgmaxFirstOccurrence() {
        // When there are ties, should return first occurrence
        let values: [Float] = [0.5, 0.5, 0.3]
        
        let index = TopKExtractor.argmax(values)
        
        XCTAssertEqual(index, 0)  // First 0.5
    }
    
    func testArgmaxAllSame() {
        let values: [Float] = [1, 1, 1, 1]
        
        let index = TopKExtractor.argmax(values)
        
        XCTAssertEqual(index, 0)  // First occurrence
    }
    
    // MARK: - argmin Tests
    
    func testArgminBasic() {
        let values: [Float] = [0.5, 0.1, 0.3, 0.9]
        
        let index = TopKExtractor.argmin(values)
        
        XCTAssertEqual(index, 1)
    }
    
    func testArgminEmpty() {
        let values: [Float] = []
        
        let index = TopKExtractor.argmin(values)
        
        XCTAssertNil(index)
    }
    
    func testArgminSingleElement() {
        let values: [Float] = [0.5]
        
        let index = TopKExtractor.argmin(values)
        
        XCTAssertEqual(index, 0)
    }
    
    func testArgminNegativeValues() {
        let values: [Float] = [-1, -10, -5, -2]
        
        let index = TopKExtractor.argmin(values)
        
        XCTAssertEqual(index, 1)  // -10 is lowest
    }
    
    func testArgminFirstOccurrence() {
        let values: [Float] = [0.3, 0.1, 0.1, 0.5]
        
        let index = TopKExtractor.argmin(values)
        
        XCTAssertEqual(index, 1)  // First 0.1
    }
    
    func testArgminAllSame() {
        let values: [Float] = [2, 2, 2, 2]
        
        let index = TopKExtractor.argmin(values)
        
        XCTAssertEqual(index, 0)  // First occurrence
    }
    
    // MARK: - Edge Cases
    
    func testLargeArray() {
        let values = (0..<10000).map { Float($0) / 10000 }
        
        let result = TopKExtractor.extractTopK(values, k: 5)
        
        XCTAssertEqual(result.indices.count, 5)
        XCTAssertEqual(result.indices[0], 9999)  // Highest value
    }
    
    func testVerySmallValues() {
        let values: [Float] = [1e-10, 1e-11, 1e-9, 1e-12]
        
        let result = TopKExtractor.extractTopK(values, k: 2)
        
        XCTAssertEqual(result.indices[0], 2)  // 1e-9 is highest
    }
    
    func testMixedSignValues() {
        let values: [Float] = [-100, 0, 100, 50, -50]
        
        let argmaxIdx = TopKExtractor.argmax(values)
        let argminIdx = TopKExtractor.argmin(values)
        
        XCTAssertEqual(argmaxIdx, 2)   // 100
        XCTAssertEqual(argminIdx, 0)  // -100
    }
    
    // MARK: - Performance Tests
    
    func testExtractTopKPerformance() {
        let values = [Float](repeating: 0.5, count: 10000)
        
        measure {
            _ = TopKExtractor.extractTopK(values, k: 10)
        }
    }
    
    func testExtractTopKWithSoftmaxPerformance() {
        let logits = [Float](repeating: 1.0, count: 1000)
        
        measure {
            _ = TopKExtractor.extractTopKWithSoftmax(logits, k: 5)
        }
    }
    
    func testArgmaxPerformance() {
        let values = [Float](repeating: 0.5, count: 100000)
        
        measure {
            _ = TopKExtractor.argmax(values)
        }
    }
}
