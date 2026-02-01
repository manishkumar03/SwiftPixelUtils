import XCTest
@testable import SwiftPixelUtils

/// Tests for ``MaskUtilities`` - Segmentation mask processing utilities.
///
/// ## Topics
///
/// ### MaskResult Tests
/// - Initialization with mask data, dimensions, and class count
/// - Memory layout validation
///
/// ### Mask Processing Tests
/// - Argmax across channels for class prediction
/// - Threshold-based binary mask generation
/// - Mask resizing and interpolation
///
/// ### Mask Analysis Tests
/// - Per-class pixel counts
/// - Confidence aggregation
/// - Connected component analysis
final class MaskUtilitiesTests: XCTestCase {
    
    // MARK: - MaskResult Tests
    
    func testMaskResultInit() {
        let mask: [Float] = [0, 1, 1, 0]
        let result = MaskUtilities.MaskResult(
            mask: mask,
            width: 2,
            height: 2,
            numClasses: 2
        )
        
        XCTAssertEqual(result.mask, mask)
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
        XCTAssertEqual(result.numClasses, 2)
    }
    
    func testMaskResultWithoutNumClasses() {
        let result = MaskUtilities.MaskResult(
            mask: [1, 0],
            width: 2,
            height: 1
        )
        
        XCTAssertNil(result.numClasses)
    }
    
    // MARK: - Resize Mask (Nearest Neighbor) Tests
    
    func testResizeMaskUpscale() {
        // 2x2 mask
        let mask: [Float] = [0, 1, 1, 0]
        
        let result = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 4, height: 4)
        )
        
        XCTAssertEqual(result.width, 4)
        XCTAssertEqual(result.height, 4)
        XCTAssertEqual(result.mask.count, 16)
    }
    
    func testResizeMaskDownscale() {
        // 4x4 mask
        let mask = [Float](repeating: 1, count: 16)
        
        let result = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 4, height: 4),
            toSize: (width: 2, height: 2)
        )
        
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
        XCTAssertEqual(result.mask.count, 4)
    }
    
    func testResizeMaskSameSize() {
        let mask: [Float] = [0, 1, 2, 3]
        
        let result = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 2, height: 2)
        )
        
        XCTAssertEqual(result.mask, mask)
    }
    
    func testResizeMaskEmpty() {
        let result = MaskUtilities.resizeMask(
            [],
            fromSize: (width: 0, height: 0),
            toSize: (width: 4, height: 4)
        )
        
        XCTAssertTrue(result.mask.isEmpty)
        XCTAssertEqual(result.width, 4)
        XCTAssertEqual(result.height, 4)
    }
    
    func testResizeMaskPreservesValues() {
        // Simple pattern
        let mask: [Float] = [0, 1, 1, 0]
        
        let result = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 2, height: 2)
        )
        
        // Values should be preserved
        XCTAssertEqual(result.mask[0], 0)
        XCTAssertEqual(result.mask[1], 1)
    }
    
    // MARK: - Resize Mask (Bilinear) Tests
    
    func testResizeMaskBilinearUpscale() {
        let mask: [Float] = [0, 1, 1, 0]
        
        let result = MaskUtilities.resizeMaskBilinear(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 4, height: 4)
        )
        
        XCTAssertEqual(result.width, 4)
        XCTAssertEqual(result.height, 4)
        XCTAssertEqual(result.mask.count, 16)
    }
    
    func testResizeMaskBilinearDownscale() {
        let mask = [Float](repeating: 0.5, count: 16)
        
        let result = MaskUtilities.resizeMaskBilinear(
            mask,
            fromSize: (width: 4, height: 4),
            toSize: (width: 2, height: 2)
        )
        
        XCTAssertEqual(result.width, 2)
        XCTAssertEqual(result.height, 2)
    }
    
    func testResizeMaskBilinearSmooth() {
        // Bilinear should create intermediate values
        let mask: [Float] = [0, 1, 0, 1]
        
        let result = MaskUtilities.resizeMaskBilinear(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 4, height: 4)
        )
        
        // Should have some intermediate values
        let hasIntermediateValues = result.mask.contains { $0 > 0 && $0 < 1 }
        XCTAssertTrue(hasIntermediateValues, "Bilinear should create smooth transitions")
    }
    
    func testResizeMaskBilinearEmpty() {
        let result = MaskUtilities.resizeMaskBilinear(
            [],
            fromSize: (width: 0, height: 0),
            toSize: (width: 4, height: 4)
        )
        
        XCTAssertTrue(result.mask.isEmpty)
    }
    
    func testResizeMaskBilinearSinglePixel() {
        let mask: [Float] = [0.5]
        
        let result = MaskUtilities.resizeMaskBilinear(
            mask,
            fromSize: (width: 1, height: 1),
            toSize: (width: 3, height: 3)
        )
        
        XCTAssertEqual(result.mask.count, 9)
        // All values should be 0.5 when upscaling from single pixel
        for value in result.mask {
            XCTAssertEqual(value, 0.5, accuracy: 0.01)
        }
    }
    
    // MARK: - Threshold Tests
    
    func testThresholdBasic() {
        let mask: [Float] = [0.3, 0.5, 0.7, 0.9]
        
        let binary = MaskUtilities.threshold(mask, threshold: 0.5)
        
        XCTAssertEqual(binary, [0, 1, 1, 1])
    }
    
    func testThresholdExactValue() {
        let mask: [Float] = [0.5, 0.49, 0.51]
        
        let binary = MaskUtilities.threshold(mask, threshold: 0.5)
        
        XCTAssertEqual(binary, [1, 0, 1])  // 0.5 >= 0.5 is true
    }
    
    func testThresholdAllAbove() {
        let mask: [Float] = [0.6, 0.7, 0.8, 0.9]
        
        let binary = MaskUtilities.threshold(mask, threshold: 0.5)
        
        XCTAssertTrue(binary.allSatisfy { $0 == 1.0 })
    }
    
    func testThresholdAllBelow() {
        let mask: [Float] = [0.1, 0.2, 0.3, 0.4]
        
        let binary = MaskUtilities.threshold(mask, threshold: 0.5)
        
        XCTAssertTrue(binary.allSatisfy { $0 == 0.0 })
    }
    
    func testThresholdEmpty() {
        let binary = MaskUtilities.threshold([], threshold: 0.5)
        XCTAssertTrue(binary.isEmpty)
    }
    
    func testThresholdZero() {
        let mask: [Float] = [0, 0.5, 1]
        
        let binary = MaskUtilities.threshold(mask, threshold: 0.0)
        
        XCTAssertEqual(binary, [1, 1, 1])  // All >= 0
    }
    
    func testThresholdOne() {
        let mask: [Float] = [0, 0.99, 1]
        
        let binary = MaskUtilities.threshold(mask, threshold: 1.0)
        
        XCTAssertEqual(binary, [0, 0, 1])  // Only 1.0 passes
    }
    
    func testThresholdDefaultValue() {
        let mask: [Float] = [0.3, 0.5, 0.7]
        
        let binary = MaskUtilities.threshold(mask)  // Default threshold is 0.5
        
        XCTAssertEqual(binary, [0, 1, 1])
    }
    
    // MARK: - Argmax Mask Tests
    
    func testArgmaxMaskBasic() {
        // 2x2 image, 3 classes
        // Logits arranged as [h, w, c]
        let logits: [Float] = [
            0.1, 0.8, 0.1,  // pixel (0,0) -> class 1
            0.9, 0.05, 0.05, // pixel (1,0) -> class 0
            0.1, 0.1, 0.8,  // pixel (0,1) -> class 2
            0.3, 0.4, 0.3   // pixel (1,1) -> class 1
        ]
        
        let result = MaskUtilities.argmaxMask(
            logits: logits,
            width: 2,
            height: 2,
            numClasses: 3
        )
        
        XCTAssertEqual(result, [1, 0, 2, 1])
    }
    
    func testArgmaxMaskSingleClass() {
        let logits: [Float] = [0.5, 0.5, 0.5, 0.5]
        
        let result = MaskUtilities.argmaxMask(
            logits: logits,
            width: 2,
            height: 2,
            numClasses: 1
        )
        
        XCTAssertEqual(result, [0, 0, 0, 0])  // All class 0
    }
    
    func testArgmaxMaskInvalidSize() {
        // Wrong number of elements
        let logits: [Float] = [0.1, 0.9]
        
        let result = MaskUtilities.argmaxMask(
            logits: logits,
            width: 2,
            height: 2,
            numClasses: 3
        )
        
        XCTAssertTrue(result.isEmpty)
    }
    
    func testArgmaxMaskTieBreaking() {
        // Tied values - should return first max
        let logits: [Float] = [
            0.5, 0.5, 0.5  // All equal
        ]
        
        let result = MaskUtilities.argmaxMask(
            logits: logits,
            width: 1,
            height: 1,
            numClasses: 3
        )
        
        XCTAssertEqual(result, [0])  // First class wins ties
    }
    
    // MARK: - Edge Cases
    
    func testResizeMaskNonSquare() {
        let mask: [Float] = [0, 1, 2, 3, 4, 5]  // 3x2
        
        let result = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 3, height: 2),
            toSize: (width: 6, height: 4)
        )
        
        XCTAssertEqual(result.width, 6)
        XCTAssertEqual(result.height, 4)
        XCTAssertEqual(result.mask.count, 24)
    }
    
    func testResizeMaskToOnePixel() {
        let mask: [Float] = [1, 2, 3, 4]
        
        let result = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 1, height: 1)
        )
        
        XCTAssertEqual(result.mask.count, 1)
    }
    
    // MARK: - Performance Tests
    
    func testResizeMaskPerformance() {
        let mask = [Float](repeating: 0.5, count: 224 * 224)
        
        measure {
            _ = MaskUtilities.resizeMask(
                mask,
                fromSize: (width: 224, height: 224),
                toSize: (width: 640, height: 640)
            )
        }
    }
    
    func testResizeMaskBilinearPerformance() {
        let mask = [Float](repeating: 0.5, count: 224 * 224)
        
        measure {
            _ = MaskUtilities.resizeMaskBilinear(
                mask,
                fromSize: (width: 224, height: 224),
                toSize: (width: 640, height: 640)
            )
        }
    }
    
    func testArgmaxMaskPerformance() {
        let logits = [Float](repeating: 0.1, count: 224 * 224 * 80)
        
        measure {
            _ = MaskUtilities.argmaxMask(
                logits: logits,
                width: 224,
                height: 224,
                numClasses: 80
            )
        }
    }
}
