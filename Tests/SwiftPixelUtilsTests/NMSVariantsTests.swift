import XCTest
@testable import SwiftPixelUtils

/// Tests for ``NMSVariants`` - Non-Maximum Suppression algorithm variants.
///
/// ## Topics
///
/// ### Soft-NMS Tests
/// - Linear decay mode
/// - Gaussian decay mode
/// - Sigma parameter effects
///
/// ### Class-Aware NMS Tests
/// - Per-class suppression
/// - Multi-class detection handling
///
/// ### Batched NMS Tests
/// - Multiple images in batch
/// - Memory-efficient processing
///
/// ### Performance Tests
/// - Large detection count benchmarks
/// - IoU computation optimization
final class NMSVariantsTests: XCTestCase {
    
    // MARK: - Soft-NMS Tests
    
    func testSoftNMSLinearBasic() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [1, 1, 11, 11],  // Overlaps with first
            [50, 50, 60, 60]  // No overlap
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = NMSVariants.softNMS(
            boxes: boxes,
            scores: scores,
            mode: .linear,
            iouThreshold: 0.3,
            scoreThreshold: 0.1
        )
        
        // All should be kept, but overlapping box should have reduced score
        XCTAssertTrue(result.indices.contains(0))
        XCTAssertTrue(result.indices.contains(2))
    }
    
    func testSoftNMSGaussianBasic() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [2, 2, 12, 12],
            [100, 100, 110, 110]
        ]
        let scores: [Float] = [0.9, 0.85, 0.7]
        
        let result = NMSVariants.softNMS(
            boxes: boxes,
            scores: scores,
            mode: .gaussian(sigma: 0.5),
            iouThreshold: 0.3,
            scoreThreshold: 0.1
        )
        
        XCTAssertFalse(result.indices.isEmpty)
        XCTAssertEqual(result.indices.count, result.scores.count)
    }
    
    func testSoftNMSEmpty() {
        let result = NMSVariants.softNMS(
            boxes: [],
            scores: [],
            mode: .linear
        )
        
        XCTAssertTrue(result.indices.isEmpty)
        XCTAssertTrue(result.scores.isEmpty)
    }
    
    func testSoftNMSMismatchedCount() {
        let boxes: [[Float]] = [[0, 0, 10, 10]]
        let scores: [Float] = [0.9, 0.8]  // More scores than boxes
        
        let result = NMSVariants.softNMS(boxes: boxes, scores: scores, mode: .linear)
        
        XCTAssertTrue(result.indices.isEmpty)
    }
    
    func testSoftNMSSingleBox() {
        let boxes: [[Float]] = [[0, 0, 10, 10]]
        let scores: [Float] = [0.9]
        
        let result = NMSVariants.softNMS(boxes: boxes, scores: scores, mode: .linear)
        
        XCTAssertEqual(result.indices, [0])
        XCTAssertEqual(result.scores[0], 0.9, accuracy: 0.001)
    }
    
    func testSoftNMSNoOverlap() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [50, 50, 60, 60],
            [100, 100, 110, 110]
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = NMSVariants.softNMS(boxes: boxes, scores: scores, mode: .linear)
        
        // All boxes should be kept with original scores
        XCTAssertEqual(result.indices.count, 3)
        XCTAssertEqual(result.scores, [0.9, 0.8, 0.7])
    }
    
    func testSoftNMSAllBelowThreshold() {
        let boxes: [[Float]] = [[0, 0, 10, 10], [5, 5, 15, 15]]
        let scores: [Float] = [0.05, 0.03]
        
        let result = NMSVariants.softNMS(
            boxes: boxes,
            scores: scores,
            mode: .linear,
            scoreThreshold: 0.1
        )
        
        XCTAssertTrue(result.indices.isEmpty)
    }
    
    func testSoftNMSScoreDecay() {
        // Two highly overlapping boxes
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [1, 1, 11, 11]
        ]
        let scores: [Float] = [0.9, 0.8]
        
        let result = NMSVariants.softNMS(
            boxes: boxes,
            scores: scores,
            mode: .linear,
            iouThreshold: 0.0,  // Always apply decay
            scoreThreshold: 0.001
        )
        
        // Second box's score should be reduced
        if result.indices.count == 2 {
            let secondBoxScoreIndex = result.indices.firstIndex(of: 1)!
            XCTAssertLessThan(result.scores[secondBoxScoreIndex], 0.8)
        }
    }
    
    // MARK: - Per-Class NMS Tests
    
    func testPerClassNMSBasic() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60]
        ]
        // 3 boxes, 2 classes
        let scores: [[Float]] = [
            [0.9, 0.1],  // Box 0: high score for class 0
            [0.8, 0.1],  // Box 1: high score for class 0 (overlaps box 0)
            [0.1, 0.9]   // Box 2: high score for class 1
        ]
        
        let result = NMSVariants.perClassNMS(
            boxes: boxes,
            scores: scores,
            iouThreshold: 0.5,
            scoreThreshold: 0.5
        )
        
        // Should have at least one detection per class with high score
        let class0Detections = result.filter { $0.classIndex == 0 }
        let class1Detections = result.filter { $0.classIndex == 1 }
        
        XCTAssertFalse(class0Detections.isEmpty)
        XCTAssertFalse(class1Detections.isEmpty)
    }
    
    func testPerClassNMSEmpty() {
        let result = NMSVariants.perClassNMS(
            boxes: [],
            scores: [],
            iouThreshold: 0.5
        )
        
        XCTAssertTrue(result.isEmpty)
    }
    
    func testPerClassNMSMismatchedCount() {
        let boxes: [[Float]] = [[0, 0, 10, 10]]
        let scores: [[Float]] = [[0.9, 0.1], [0.8, 0.2]]  // More scores than boxes
        
        let result = NMSVariants.perClassNMS(boxes: boxes, scores: scores)
        
        XCTAssertTrue(result.isEmpty)
    }
    
    func testPerClassNMSMaxDetectionsPerClass() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
            [60, 60, 70, 70]
        ]
        let scores: [[Float]] = [
            [0.9], [0.8], [0.7], [0.6]
        ]
        
        let result = NMSVariants.perClassNMS(
            boxes: boxes,
            scores: scores,
            scoreThreshold: 0.5,
            maxDetectionsPerClass: 2
        )
        
        XCTAssertLessThanOrEqual(result.count, 2)
    }
    
    func testPerClassNMSSortedByScore() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [50, 50, 60, 60]
        ]
        let scores: [[Float]] = [
            [0.6],
            [0.9]
        ]
        
        let result = NMSVariants.perClassNMS(
            boxes: boxes,
            scores: scores,
            scoreThreshold: 0.5
        )
        
        // Results should be sorted by score descending
        if result.count >= 2 {
            XCTAssertGreaterThanOrEqual(result[0].score, result[1].score)
        }
    }
    
    // MARK: - Batched NMS Tests
    
    func testBatchedNMSBasic() {
        let batchBoxes: [[[Float]]] = [
            [[0, 0, 10, 10], [1, 1, 11, 11]],  // Image 1
            [[0, 0, 10, 10]]  // Image 2
        ]
        let batchScores: [[Float]] = [
            [0.9, 0.8],
            [0.7]
        ]
        
        let results = NMSVariants.batchedNMS(
            batchBoxes: batchBoxes,
            batchScores: batchScores,
            iouThreshold: 0.5,
            scoreThreshold: 0.5
        )
        
        XCTAssertEqual(results.count, 2)  // Two images
    }
    
    func testBatchedNMSEmpty() {
        let results = NMSVariants.batchedNMS(
            batchBoxes: [],
            batchScores: []
        )
        
        XCTAssertTrue(results.isEmpty)
    }
    
    func testBatchedNMSMismatchedBatches() {
        let batchBoxes: [[[Float]]] = [[[0, 0, 10, 10]]]
        let batchScores: [[Float]] = [[0.9], [0.8]]  // Different batch count
        
        let results = NMSVariants.batchedNMS(
            batchBoxes: batchBoxes,
            batchScores: batchScores
        )
        
        XCTAssertTrue(results.isEmpty)
    }
    
    // MARK: - Class-Agnostic NMS Tests
    
    func testClassAgnosticNMSBasic() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60]
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let kept = NMSVariants.classAgnosticNMS(
            boxes: boxes,
            scores: scores,
            iouThreshold: 0.5,
            scoreThreshold: 0.5
        )
        
        XCTAssertTrue(kept.contains(0))  // Highest scoring
        XCTAssertTrue(kept.contains(2))  // Non-overlapping
    }
    
    func testClassAgnosticNMSEmpty() {
        let kept = NMSVariants.classAgnosticNMS(
            boxes: [],
            scores: []
        )
        
        XCTAssertTrue(kept.isEmpty)
    }
    
    func testClassAgnosticNMSMismatchedCount() {
        let boxes: [[Float]] = [[0, 0, 10, 10]]
        let scores: [Float] = [0.9, 0.8]
        
        let kept = NMSVariants.classAgnosticNMS(boxes: boxes, scores: scores)
        
        XCTAssertTrue(kept.isEmpty)
    }
    
    func testClassAgnosticNMSSingleBox() {
        let boxes: [[Float]] = [[0, 0, 10, 10]]
        let scores: [Float] = [0.9]
        
        let kept = NMSVariants.classAgnosticNMS(boxes: boxes, scores: scores)
        
        XCTAssertEqual(kept, [0])
    }
    
    func testClassAgnosticNMSNoOverlap() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [100, 100, 110, 110]
        ]
        let scores: [Float] = [0.9, 0.8]
        
        let kept = NMSVariants.classAgnosticNMS(
            boxes: boxes,
            scores: scores,
            scoreThreshold: 0.5
        )
        
        XCTAssertEqual(kept.sorted(), [0, 1])
    }
    
    func testClassAgnosticNMSHighOverlap() {
        // Two nearly identical boxes
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [0, 0, 10, 10]
        ]
        let scores: [Float] = [0.9, 0.8]
        
        let kept = NMSVariants.classAgnosticNMS(
            boxes: boxes,
            scores: scores,
            iouThreshold: 0.5
        )
        
        // Should only keep one box
        XCTAssertEqual(kept.count, 1)
        XCTAssertEqual(kept[0], 0)  // Higher scoring one
    }
    
    func testClassAgnosticNMSAllBelowThreshold() {
        let boxes: [[Float]] = [[0, 0, 10, 10], [50, 50, 60, 60]]
        let scores: [Float] = [0.3, 0.2]
        
        let kept = NMSVariants.classAgnosticNMS(
            boxes: boxes,
            scores: scores,
            scoreThreshold: 0.5
        )
        
        XCTAssertTrue(kept.isEmpty)
    }
    
    func testClassAgnosticNMSPreservesHighScoreOrder() {
        let boxes: [[Float]] = [
            [0, 0, 10, 10],
            [50, 50, 60, 60],
            [100, 100, 110, 110]
        ]
        let scores: [Float] = [0.7, 0.9, 0.8]  // Not in order
        
        let kept = NMSVariants.classAgnosticNMS(
            boxes: boxes,
            scores: scores,
            scoreThreshold: 0.5
        )
        
        // First kept should be highest score
        if !kept.isEmpty {
            XCTAssertEqual(kept[0], 1)  // Index 1 has 0.9
        }
    }
    
    // MARK: - SoftNMSMode Tests
    
    func testSoftNMSModeLinear() {
        let mode = NMSVariants.SoftNMSMode.linear
        if case .linear = mode {
            // Success
        } else {
            XCTFail("Expected linear mode")
        }
    }
    
    func testSoftNMSModeGaussian() {
        let mode = NMSVariants.SoftNMSMode.gaussian(sigma: 0.5)
        if case .gaussian(let sigma) = mode {
            XCTAssertEqual(sigma, 0.5)
        } else {
            XCTFail("Expected gaussian mode")
        }
    }
    
    // MARK: - Performance Tests
    
    func testClassAgnosticNMSPerformance() {
        // Generate 1000 random boxes
        var boxes: [[Float]] = []
        var scores: [Float] = []
        
        for i in 0..<1000 {
            let x = Float(i % 100) * 10
            let y = Float(i / 100) * 10
            boxes.append([x, y, x + 10, y + 10])
            scores.append(Float.random(in: 0.5...1.0))
        }
        
        measure {
            _ = NMSVariants.classAgnosticNMS(
                boxes: boxes,
                scores: scores,
                iouThreshold: 0.5,
                scoreThreshold: 0.5
            )
        }
    }
    
    func testSoftNMSPerformance() {
        var boxes: [[Float]] = []
        var scores: [Float] = []
        
        for i in 0..<500 {
            let x = Float(i % 50) * 10
            let y = Float(i / 50) * 10
            boxes.append([x, y, x + 10, y + 10])
            scores.append(Float.random(in: 0.5...1.0))
        }
        
        measure {
            _ = NMSVariants.softNMS(
                boxes: boxes,
                scores: scores,
                mode: .gaussian(sigma: 0.5)
            )
        }
    }
}
