import XCTest
@testable import SwiftPixelUtils

/// Tests for ``ConfidenceFilter`` - Detection confidence filtering utilities.
///
/// ## Topics
///
/// ### Basic Filtering Tests
/// - Threshold-based filtering
/// - All pass, all fail, partial pass scenarios
///
/// ### Edge Cases
/// - Empty input arrays
/// - Boundary threshold values (0.0, 1.0)
/// - Equal confidence scores
///
/// ### Performance Tests
/// - Large array filtering benchmarks
final class ConfidenceFilterTests: XCTestCase {
    
    // MARK: - Basic Filter Tests
    
    func testFilterBasic() {
        let detections = [
            (item: "a", confidence: Float(0.9)),
            (item: "b", confidence: Float(0.5)),
            (item: "c", confidence: Float(0.3))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertEqual(filtered.count, 2)
        XCTAssertTrue(filtered.contains { $0.item == "a" })
        XCTAssertTrue(filtered.contains { $0.item == "b" })
    }
    
    func testFilterAllPass() {
        let detections = [
            (item: 1, confidence: Float(0.9)),
            (item: 2, confidence: Float(0.8)),
            (item: 3, confidence: Float(0.7))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertEqual(filtered.count, 3)
    }
    
    func testFilterNonePass() {
        let detections = [
            (item: 1, confidence: Float(0.3)),
            (item: 2, confidence: Float(0.2)),
            (item: 3, confidence: Float(0.1))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertTrue(filtered.isEmpty)
    }
    
    func testFilterEmpty() {
        let detections: [(item: Int, confidence: Float)] = []
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertTrue(filtered.isEmpty)
    }
    
    func testFilterExactThreshold() {
        let detections = [
            (item: 1, confidence: Float(0.5)),
            (item: 2, confidence: Float(0.49))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].item, 1)
    }
    
    func testFilterZeroThreshold() {
        let detections = [
            (item: 1, confidence: Float(0.9)),
            (item: 2, confidence: Float(0.0)),
            (item: 3, confidence: Float(0.001))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.0)
        
        XCTAssertEqual(filtered.count, 3)
    }
    
    func testFilterOneThreshold() {
        let detections = [
            (item: 1, confidence: Float(1.0)),
            (item: 2, confidence: Float(0.99)),
            (item: 3, confidence: Float(0.9))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 1.0)
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].item, 1)
    }
    
    // MARK: - Per-Class Threshold Tests
    
    func testFilterWithClassThresholdsBasic() {
        let detections = [
            (item: "box1", classId: 0, confidence: Float(0.8)),
            (item: "box2", classId: 1, confidence: Float(0.6)),
            (item: "box3", classId: 0, confidence: Float(0.4))
        ]
        
        let classThresholds = [0: Float(0.5), 1: Float(0.7)]
        
        let filtered = ConfidenceFilter.filterWithClassThresholds(
            detections,
            classThresholds: classThresholds
        )
        
        XCTAssertEqual(filtered.count, 1)  // Only box1 passes (class 0, 0.8 >= 0.5)
        XCTAssertEqual(filtered[0].item, "box1")
    }
    
    func testFilterWithClassThresholdsDefaultThreshold() {
        let detections = [
            (item: "a", classId: 0, confidence: Float(0.8)),
            (item: "b", classId: 99, confidence: Float(0.6)),  // Unknown class
            (item: "c", classId: 99, confidence: Float(0.4))   // Unknown class
        ]
        
        let classThresholds = [0: Float(0.5)]
        
        let filtered = ConfidenceFilter.filterWithClassThresholds(
            detections,
            classThresholds: classThresholds,
            defaultThreshold: 0.5
        )
        
        // Class 0 box passes, class 99 uses default 0.5
        XCTAssertEqual(filtered.count, 2)
    }
    
    func testFilterWithClassThresholdsEmpty() {
        let detections: [(item: Int, classId: Int, confidence: Float)] = []
        
        let filtered = ConfidenceFilter.filterWithClassThresholds(
            detections,
            classThresholds: [:]
        )
        
        XCTAssertTrue(filtered.isEmpty)
    }
    
    func testFilterWithClassThresholdsAllClasses() {
        let detections = [
            (item: "a", classId: 0, confidence: Float(0.9)),
            (item: "b", classId: 1, confidence: Float(0.7)),
            (item: "c", classId: 2, confidence: Float(0.5))
        ]
        
        let classThresholds = [
            0: Float(0.8),
            1: Float(0.6),
            2: Float(0.4)
        ]
        
        let filtered = ConfidenceFilter.filterWithClassThresholds(
            detections,
            classThresholds: classThresholds
        )
        
        XCTAssertEqual(filtered.count, 3)
    }
    
    func testFilterWithClassThresholdsNonePass() {
        let detections = [
            (item: "a", classId: 0, confidence: Float(0.5)),
            (item: "b", classId: 1, confidence: Float(0.5))
        ]
        
        let classThresholds = [
            0: Float(0.9),
            1: Float(0.9)
        ]
        
        let filtered = ConfidenceFilter.filterWithClassThresholds(
            detections,
            classThresholds: classThresholds
        )
        
        XCTAssertTrue(filtered.isEmpty)
    }
    
    // MARK: - Filter by Ratio Tests
    
    func testFilterByRatioBasic() {
        let detections = [
            (item: "a", confidence: Float(0.9)),
            (item: "b", confidence: Float(0.8)),
            (item: "c", confidence: Float(0.7)),
            (item: "d", confidence: Float(0.6))
        ]
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 0.5)
        
        XCTAssertEqual(filtered.count, 2)
        XCTAssertEqual(filtered[0].item, "a")  // Highest
        XCTAssertEqual(filtered[1].item, "b")  // Second highest
    }
    
    func testFilterByRatioEmpty() {
        let detections: [(item: Int, confidence: Float)] = []
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 0.5)
        
        XCTAssertTrue(filtered.isEmpty)
    }
    
    func testFilterByRatioZero() {
        let detections = [
            (item: "a", confidence: Float(0.9)),
            (item: "b", confidence: Float(0.8))
        ]
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 0.0)
        
        XCTAssertTrue(filtered.isEmpty)
    }
    
    func testFilterByRatioOne() {
        let detections = [
            (item: "a", confidence: Float(0.9)),
            (item: "b", confidence: Float(0.8)),
            (item: "c", confidence: Float(0.7))
        ]
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 1.0)
        
        XCTAssertEqual(filtered.count, 3)
    }
    
    func testFilterByRatioGreaterThanOne() {
        let detections = [
            (item: "a", confidence: Float(0.9)),
            (item: "b", confidence: Float(0.8))
        ]
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 2.0)
        
        // Should clamp to 1.0
        XCTAssertEqual(filtered.count, 2)
    }
    
    func testFilterByRatioMinimumOneKept() {
        let detections = [
            (item: "a", confidence: Float(0.9)),
            (item: "b", confidence: Float(0.8)),
            (item: "c", confidence: Float(0.7)),
            (item: "d", confidence: Float(0.6))
        ]
        
        // Very small ratio should still keep at least 1
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 0.001)
        
        XCTAssertGreaterThanOrEqual(filtered.count, 1)
    }
    
    func testFilterByRatioSortedByConfidence() {
        let detections = [
            (item: "low", confidence: Float(0.3)),
            (item: "high", confidence: Float(0.9)),
            (item: "mid", confidence: Float(0.6))
        ]
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 1.0)
        
        // Should be sorted descending
        XCTAssertEqual(filtered[0].item, "high")
        XCTAssertEqual(filtered[1].item, "mid")
        XCTAssertEqual(filtered[2].item, "low")
    }
    
    func testFilterByRatioSingleElement() {
        let detections = [(item: "only", confidence: Float(0.5))]
        
        let filtered = ConfidenceFilter.filterByRatio(detections, ratio: 0.5)
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].item, "only")
    }
    
    // MARK: - Generic Type Tests
    
    func testFilterWithCustomType() {
        struct Detection {
            let id: Int
            let name: String
        }
        
        let detections = [
            (item: Detection(id: 1, name: "cat"), confidence: Float(0.9)),
            (item: Detection(id: 2, name: "dog"), confidence: Float(0.4))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].item.name, "cat")
    }
    
    // MARK: - Edge Cases
    
    func testFilterNegativeConfidence() {
        let detections = [
            (item: 1, confidence: Float(-0.5)),
            (item: 2, confidence: Float(0.5))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.0)
        
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].item, 2)
    }
    
    func testFilterNaNConfidence() {
        let detections = [
            (item: 1, confidence: Float.nan),
            (item: 2, confidence: Float(0.5))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        // NaN should not pass any threshold comparison
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered[0].item, 2)
    }
    
    func testFilterInfiniteConfidence() {
        let detections = [
            (item: 1, confidence: Float.infinity),
            (item: 2, confidence: Float(0.5))
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertEqual(filtered.count, 2)
    }
    
    // MARK: - Performance Tests
    
    func testFilterPerformance() {
        let detections = (0..<10000).map { i in
            (item: i, confidence: Float.random(in: 0...1))
        }
        
        measure {
            _ = ConfidenceFilter.filter(detections, threshold: 0.5)
        }
    }
    
    func testFilterWithClassThresholdsPerformance() {
        let detections = (0..<10000).map { i in
            (item: i, classId: i % 100, confidence: Float.random(in: 0...1))
        }
        
        var thresholds: [Int: Float] = [:]
        for i in 0..<100 {
            thresholds[i] = Float(i) / 100.0
        }
        
        measure {
            _ = ConfidenceFilter.filterWithClassThresholds(
                detections,
                classThresholds: thresholds
            )
        }
    }
    
    func testFilterByRatioPerformance() {
        let detections = (0..<10000).map { i in
            (item: i, confidence: Float.random(in: 0...1))
        }
        
        measure {
            _ = ConfidenceFilter.filterByRatio(detections, ratio: 0.1)
        }
    }
}
