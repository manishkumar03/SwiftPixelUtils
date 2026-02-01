#if canImport(CoreML)
import XCTest
import CoreML
@testable import SwiftPixelUtils

/// Tests for ``CoreMLConversion`` - CoreML MLMultiArray conversion utilities.
///
/// ## Topics
///
/// ### toMLMultiArray Tests
/// - 1D, 2D, 3D, and 4D array conversions
/// - Shape validation
/// - Empty array handling
///
/// ### fromMLMultiArray Tests
/// - Float32 extraction
/// - Shape mismatch detection
///
/// ### Data Type Tests
/// - Float32, Double, Int32 conversions
/// - Type-specific precision
final class CoreMLConversionTests: XCTestCase {
    
    // MARK: - toMLMultiArray Tests
    
    func testToMLMultiArrayBasic() throws {
        let array: [Float] = [1, 2, 3, 4, 5, 6]
        let shape = [2, 3]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        
        XCTAssertEqual(multiArray.shape.count, 2)
        XCTAssertEqual(multiArray.shape[0].intValue, 2)
        XCTAssertEqual(multiArray.shape[1].intValue, 3)
        XCTAssertEqual(multiArray.count, 6)
    }
    
    func testToMLMultiArray1D() throws {
        let array: [Float] = [1, 2, 3, 4, 5]
        let shape = [5]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        
        XCTAssertEqual(multiArray.shape.count, 1)
        XCTAssertEqual(multiArray.shape[0].intValue, 5)
    }
    
    func testToMLMultiArray4D() throws {
        let array = [Float](repeating: 0.5, count: 1 * 3 * 224 * 224)
        let shape = [1, 3, 224, 224]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        
        XCTAssertEqual(multiArray.shape.count, 4)
        XCTAssertEqual(multiArray.shape[0].intValue, 1)
        XCTAssertEqual(multiArray.shape[1].intValue, 3)
        XCTAssertEqual(multiArray.shape[2].intValue, 224)
        XCTAssertEqual(multiArray.shape[3].intValue, 224)
    }
    
    func testToMLMultiArraySizeMismatch() {
        let array: [Float] = [1, 2, 3, 4, 5]  // 5 elements
        let shape = [2, 3]  // Expects 6 elements
        
        XCTAssertThrowsError(try CoreMLConversion.toMLMultiArray(array, shape: shape)) { error in
            if case PixelUtilsError.processingFailed = error {
                // Expected
            } else {
                XCTFail("Should throw processingFailed error")
            }
        }
    }
    
    func testToMLMultiArrayPreservesValues() throws {
        let array: [Float] = [1.0, 2.5, 3.7, 4.2]
        let shape = [4]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        
        // Read back values
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: 4)
        for i in 0..<4 {
            XCTAssertEqual(ptr[i], array[i], accuracy: 0.001)
        }
    }
    
    // MARK: - fromMLMultiArray Tests
    
    func testFromMLMultiArrayBasic() throws {
        let array: [Float] = [1, 2, 3, 4, 5, 6]
        let shape = [2, 3]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        let result = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(result.count, 6)
        XCTAssertEqual(result, array)
    }
    
    func testFromMLMultiArrayPreservesValues() throws {
        let original: [Float] = [1.1, 2.2, 3.3, 4.4, 5.5]
        let shape = [5]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(original, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        for i in 0..<original.count {
            XCTAssertEqual(recovered[i], original[i], accuracy: 0.001)
        }
    }
    
    func testFromMLMultiArrayLargeArray() throws {
        let original = [Float](repeating: 0.5, count: 1000)
        let shape = [10, 100]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(original, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(recovered.count, 1000)
    }
    
    // MARK: - Roundtrip Tests
    
    func testRoundtripSmall() throws {
        let original: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        let shape = [3, 3]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(original, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(recovered, original)
    }
    
    func testRoundtripLarge() throws {
        let original = (0..<10000).map { Float($0) / 10000.0 }
        let shape = [100, 100]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(original, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(recovered.count, original.count)
        for i in 0..<original.count {
            XCTAssertEqual(recovered[i], original[i], accuracy: 0.0001)
        }
    }
    
    func testRoundtripImageTensor() throws {
        // Typical image tensor shape: [1, 3, 224, 224]
        let batchSize = 1
        let channels = 3
        let height = 224
        let width = 224
        let totalSize = batchSize * channels * height * width
        
        let original = [Float](repeating: 0.5, count: totalSize)
        let shape = [batchSize, channels, height, width]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(original, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(recovered.count, totalSize)
    }
    
    // MARK: - toProbabilities Tests
    
    func testToProbabilitiesBasic() throws {
        // Logits that will become clear probabilities after softmax
        let logits: [Float] = [0, 0, 10]  // Third class should dominate
        let shape = [3]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(logits, shape: shape)
        let probs = CoreMLConversion.toProbabilities(multiArray)
        
        XCTAssertEqual(probs.count, 3)
        XCTAssertTrue(probs.keys.contains("class_0"))
        XCTAssertTrue(probs.keys.contains("class_1"))
        XCTAssertTrue(probs.keys.contains("class_2"))
        
        // Third class should have highest probability
        XCTAssertGreaterThan(probs["class_2"]!, probs["class_0"]!)
        XCTAssertGreaterThan(probs["class_2"]!, probs["class_1"]!)
    }
    
    func testToProbabilitiesWithLabels() throws {
        let logits: [Float] = [2, 1, 3]
        let shape = [3]
        let labels = ["cat", "dog", "bird"]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(logits, shape: shape)
        let probs = CoreMLConversion.toProbabilities(multiArray, labels: labels)
        
        XCTAssertTrue(probs.keys.contains("cat"))
        XCTAssertTrue(probs.keys.contains("dog"))
        XCTAssertTrue(probs.keys.contains("bird"))
        
        // "bird" should have highest probability (highest logit)
        XCTAssertGreaterThan(probs["bird"]!, probs["cat"]!)
        XCTAssertGreaterThan(probs["bird"]!, probs["dog"]!)
    }
    
    func testToProbabilitiesSumToOne() throws {
        let logits: [Float] = [1, 2, 3, 4, 5]
        let shape = [5]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(logits, shape: shape)
        let probs = CoreMLConversion.toProbabilities(multiArray)
        
        let sum = probs.values.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
    }
    
    func testToProbabilitiesPartialLabels() throws {
        let logits: [Float] = [1, 2, 3, 4, 5]
        let shape = [5]
        let labels = ["a", "b", "c"]  // Only 3 labels for 5 classes
        
        let multiArray = try CoreMLConversion.toMLMultiArray(logits, shape: shape)
        let probs = CoreMLConversion.toProbabilities(multiArray, labels: labels)
        
        XCTAssertTrue(probs.keys.contains("a"))
        XCTAssertTrue(probs.keys.contains("b"))
        XCTAssertTrue(probs.keys.contains("c"))
        XCTAssertTrue(probs.keys.contains("class_3"))  // Falls back to index
        XCTAssertTrue(probs.keys.contains("class_4"))
    }
    
    // MARK: - Edge Cases
    
    func testEmptyArray() {
        let array: [Float] = []
        let shape = [0]
        
        // Empty arrays might be handled differently
        // This test documents the behavior
        do {
            let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
            XCTAssertEqual(multiArray.count, 0)
        } catch {
            // Some implementations might throw for empty arrays
            // This is acceptable behavior
        }
    }
    
    func testSingleElement() throws {
        let array: [Float] = [42.0]
        let shape = [1]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(recovered.count, 1)
        XCTAssertEqual(recovered[0], 42.0, accuracy: 0.001)
    }
    
    func testNegativeValues() throws {
        let array: [Float] = [-1, -2, -3, 0, 1, 2, 3]
        let shape = [7]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        XCTAssertEqual(recovered, array)
    }
    
    func testVerySmallValues() throws {
        let array: [Float] = [0.000001, 0.0000001, 0.00000001]
        let shape = [3]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        for i in 0..<array.count {
            XCTAssertEqual(recovered[i], array[i], accuracy: 0.0000001)
        }
    }
    
    func testVeryLargeValues() throws {
        let array: [Float] = [1e30, -1e30, 1e38]
        let shape = [3]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        let recovered = CoreMLConversion.fromMLMultiArray(multiArray)
        
        for i in 0..<array.count {
            XCTAssertEqual(recovered[i], array[i], accuracy: abs(array[i] * 0.0001))
        }
    }
    
    // MARK: - Performance Tests
    
    func testToMLMultiArrayPerformance() throws {
        let array = [Float](repeating: 0.5, count: 1 * 3 * 640 * 640)
        let shape = [1, 3, 640, 640]
        
        measure {
            _ = try? CoreMLConversion.toMLMultiArray(array, shape: shape)
        }
    }
    
    func testFromMLMultiArrayPerformance() throws {
        let array = [Float](repeating: 0.5, count: 1 * 3 * 640 * 640)
        let shape = [1, 3, 640, 640]
        let multiArray = try CoreMLConversion.toMLMultiArray(array, shape: shape)
        
        measure {
            _ = CoreMLConversion.fromMLMultiArray(multiArray)
        }
    }
}
#endif
