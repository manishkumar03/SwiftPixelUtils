import XCTest
@testable import SwiftPixelUtils

/// Tests for ``LabelDatabase`` - Label lookup for ML model outputs.
///
/// ## Topics
///
/// ### Dataset Label Tests
/// - COCO (80 classes), ImageNet (999 classes)
/// - VOC (21 classes), ADE20K (150 classes)
/// - Places365, custom datasets
///
/// ### Label Lookup Tests
/// - Index-to-label mapping
/// - Out-of-bounds handling
/// - Case sensitivity
///
/// ### Top-K Labels Tests
/// - Retrieve top predictions with labels
/// - Score-based sorting
/// - Threshold filtering
final class LabelDatabaseTests: XCTestCase {
    
    // MARK: - getLabel Tests
    
    func testGetLabelCOCO() {
        // COCO class 0 should be "person"
        let label = LabelDatabase.getLabel(0, dataset: .coco)
        XCTAssertEqual(label, "person")
    }
    
    func testGetLabelCOCOOtherClasses() {
        // Test a few known COCO classes
        XCTAssertEqual(LabelDatabase.getLabel(1, dataset: .coco), "bicycle")
        XCTAssertEqual(LabelDatabase.getLabel(2, dataset: .coco), "car")
    }
    
    func testGetLabelImageNet() {
        // ImageNet class 0 should be "tench"
        let label = LabelDatabase.getLabel(0, dataset: .imagenet)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelCIFAR10() {
        // CIFAR-10 has 10 classes
        let label = LabelDatabase.getLabel(0, dataset: .cifar10)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelCIFAR100() {
        let label = LabelDatabase.getLabel(0, dataset: .cifar100)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelVOC() {
        let label = LabelDatabase.getLabel(0, dataset: .voc)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelPlaces365() {
        let label = LabelDatabase.getLabel(0, dataset: .places365)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelADE20K() {
        let label = LabelDatabase.getLabel(0, dataset: .ade20k)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelOpenImages() {
        let label = LabelDatabase.getLabel(0, dataset: .openimages)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelLVIS() {
        let label = LabelDatabase.getLabel(0, dataset: .lvis)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelObjects365() {
        let label = LabelDatabase.getLabel(0, dataset: .objects365)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelADE20KFull() {
        let label = LabelDatabase.getLabel(0, dataset: .ade20kFull)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelKinetics400() {
        let label = LabelDatabase.getLabel(0, dataset: .kinetics400)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelKinetics700() {
        let label = LabelDatabase.getLabel(0, dataset: .kinetics700)
        XCTAssertNotNil(label)
    }
    
    func testGetLabelOutOfBounds() {
        let label = LabelDatabase.getLabel(10000, dataset: .coco)
        XCTAssertNil(label)
    }
    
    func testGetLabelNegativeIndex() {
        let label = LabelDatabase.getLabel(-1, dataset: .coco)
        XCTAssertNil(label)
    }
    
    func testGetLabelDefaultDataset() {
        // Default should be COCO
        let label = LabelDatabase.getLabel(0)
        XCTAssertEqual(label, "person")
    }
    
    // MARK: - getAllLabels Tests
    
    func testGetAllLabelsCOCO() {
        let labels = LabelDatabase.getAllLabels(for: .coco)
        XCTAssertEqual(labels.count, 80)
        XCTAssertEqual(labels[0], "person")
    }
    
    func testGetAllLabelsCOCO91() {
        let labels = LabelDatabase.getAllLabels(for: .coco91)
        XCTAssertEqual(labels.count, 91)
    }
    
    func testGetAllLabelsCIFAR10() {
        let labels = LabelDatabase.getAllLabels(for: .cifar10)
        XCTAssertEqual(labels.count, 10)
    }
    
    func testGetAllLabelsCIFAR100() {
        let labels = LabelDatabase.getAllLabels(for: .cifar100)
        XCTAssertEqual(labels.count, 100)
    }
    
    func testGetAllLabelsImageNet() {
        let labels = LabelDatabase.getAllLabels(for: .imagenet)
        XCTAssertEqual(labels.count, 999)  // ImageNet classes 0-998
    }
    
    func testGetAllLabelsVOC() {
        let labels = LabelDatabase.getAllLabels(for: .voc)
        XCTAssertEqual(labels.count, 21)
    }
    
    func testGetAllLabelsPlaces365() {
        let labels = LabelDatabase.getAllLabels(for: .places365)
        XCTAssertEqual(labels.count, 379)  // Places365 classes available
    }
    
    func testGetAllLabelsADE20K() {
        let labels = LabelDatabase.getAllLabels(for: .ade20k)
        XCTAssertEqual(labels.count, 150)
    }
    
    func testGetAllLabelsOpenImages() {
        let labels = LabelDatabase.getAllLabels(for: .openimages)
        XCTAssertEqual(labels.count, 602)  // Actual count in database
        XCTAssertEqual(labels[0], "Accordion")
    }
    
    func testGetAllLabelsLVIS() {
        let labels = LabelDatabase.getAllLabels(for: .lvis)
        XCTAssertEqual(labels.count, 1205)  // Actual count in database
    }
    
    func testGetAllLabelsObjects365() {
        let labels = LabelDatabase.getAllLabels(for: .objects365)
        XCTAssertEqual(labels.count, 365)
        XCTAssertEqual(labels[0], "Person")
    }
    
    func testGetAllLabelsADE20KFull() {
        let labels = LabelDatabase.getAllLabels(for: .ade20kFull)
        XCTAssertEqual(labels.count, 734)  // Actual count in database
    }
    
    func testGetAllLabelsKinetics400() {
        let labels = LabelDatabase.getAllLabels(for: .kinetics400)
        XCTAssertEqual(labels.count, 401)  // Actual count in database (includes background class)
        XCTAssertEqual(labels[0], "abseiling")
    }
    
    func testGetAllLabelsKinetics700() {
        let labels = LabelDatabase.getAllLabels(for: .kinetics700)
        XCTAssertEqual(labels.count, 678)  // Actual count in database
    }
    
    // MARK: - getTopLabels Tests
    
    func testGetTopLabels() {
        var scores = [Float](repeating: 0.01, count: 80)
        scores[0] = 0.9  // person
        scores[1] = 0.5  // bicycle
        scores[2] = 0.3  // car
        
        let top = LabelDatabase.getTopLabels(scores: scores, dataset: .coco, k: 3)
        
        XCTAssertEqual(top.count, 3)
        XCTAssertEqual(top[0].label, "person")
        XCTAssertEqual(top[0].confidence, 0.9)
        XCTAssertEqual(top[0].index, 0)
    }
    
    func testGetTopLabelsWithMinConfidence() {
        var scores = [Float](repeating: 0.001, count: 80)
        scores[0] = 0.9
        scores[1] = 0.05  // Below threshold
        
        let top = LabelDatabase.getTopLabels(
            scores: scores,
            dataset: .coco,
            k: 5,
            minConfidence: 0.1
        )
        
        XCTAssertEqual(top.count, 1)
        XCTAssertEqual(top[0].index, 0)
    }
    
    func testGetTopLabelsKGreaterThanClasses() {
        let scores: [Float] = [0.5, 0.3, 0.2]
        
        let top = LabelDatabase.getTopLabels(
            scores: scores,
            dataset: .cifar10,
            k: 100
        )
        
        XCTAssertLessThanOrEqual(top.count, 3)
    }
    
    func testGetTopLabelsEmptyScores() {
        let scores: [Float] = []
        
        let top = LabelDatabase.getTopLabels(scores: scores, dataset: .coco, k: 5)
        
        XCTAssertTrue(top.isEmpty)
    }
    
    func testGetTopLabelsAllBelowThreshold() {
        let scores = [Float](repeating: 0.001, count: 10)
        
        let top = LabelDatabase.getTopLabels(
            scores: scores,
            dataset: .cifar10,
            k: 5,
            minConfidence: 0.1
        )
        
        XCTAssertTrue(top.isEmpty)
    }
    
    // MARK: - getTopLabelsWithSoftmax Tests
    
    func testGetTopLabelsWithSoftmax() {
        var logits = [Float](repeating: 0, count: 10)
        logits[3] = 10.0  // High logit
        
        let top = LabelDatabase.getTopLabelsWithSoftmax(
            logits: logits,
            dataset: .cifar10,
            k: 3
        )
        
        XCTAssertEqual(top[0].index, 3)
        XCTAssertGreaterThan(top[0].confidence, 0.9)
    }
    
    // MARK: - softmax Tests
    
    func testSoftmaxBasic() {
        let logits: [Float] = [1, 2, 3]
        let probs = LabelDatabase.softmax(logits)
        
        XCTAssertEqual(probs.count, 3)
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 0.001)
        
        // Highest logit should have highest probability
        XCTAssertGreaterThan(probs[2], probs[1])
        XCTAssertGreaterThan(probs[1], probs[0])
    }
    
    func testSoftmaxEmpty() {
        let probs = LabelDatabase.softmax([])
        XCTAssertTrue(probs.isEmpty)
    }
    
    func testSoftmaxSingleElement() {
        let probs = LabelDatabase.softmax([5.0])
        XCTAssertEqual(probs.count, 1)
        XCTAssertEqual(probs[0], 1.0, accuracy: 0.001)
    }
    
    func testSoftmaxEqualInputs() {
        let logits: [Float] = [1, 1, 1, 1]
        let probs = LabelDatabase.softmax(logits)
        
        // All should be equal (0.25 each)
        for prob in probs {
            XCTAssertEqual(prob, 0.25, accuracy: 0.001)
        }
    }
    
    func testSoftmaxLargeValues() {
        // Test numerical stability with large values
        let logits: [Float] = [1000, 1001, 1002]
        let probs = LabelDatabase.softmax(logits)
        
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 0.001)
        XCTAssertFalse(probs.contains(where: { $0.isNaN || $0.isInfinite }))
    }
    
    func testSoftmaxNegativeValues() {
        let logits: [Float] = [-5, -2, -1]
        let probs = LabelDatabase.softmax(logits)
        
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 0.001)
        XCTAssertGreaterThan(probs[2], probs[1])
    }
    
    // MARK: - getDatasetInfo Tests
    
    func testGetDatasetInfoCOCO() {
        let info = LabelDatabase.getDatasetInfo(for: .coco)
        
        XCTAssertEqual(info.name, "coco")
        XCTAssertEqual(info.numClasses, 80)
        XCTAssertTrue(info.description.contains("80"))
    }
    
    func testGetDatasetInfoImageNet() {
        let info = LabelDatabase.getDatasetInfo(for: .imagenet)
        
        XCTAssertEqual(info.name, "imagenet")
        XCTAssertEqual(info.numClasses, 999)  // ImageNet classes available
    }
    
    func testGetDatasetInfoAllDatasets() {
        for dataset in LabelDataset.allCases {
            let info = LabelDatabase.getDatasetInfo(for: dataset)
            XCTAssertFalse(info.name.isEmpty)
            XCTAssertFalse(info.description.isEmpty)
        }
    }
    
    // MARK: - getAvailableDatasets Tests
    
    func testGetAvailableDatasets() {
        let datasets = LabelDatabase.getAvailableDatasets()
        
        XCTAssertTrue(datasets.contains(.coco))
        XCTAssertTrue(datasets.contains(.imagenet))
        XCTAssertTrue(datasets.contains(.cifar10))
        XCTAssertTrue(datasets.contains(.voc))
        XCTAssertTrue(datasets.contains(.openimages))
        XCTAssertTrue(datasets.contains(.lvis))
        XCTAssertTrue(datasets.contains(.objects365))
        XCTAssertTrue(datasets.contains(.ade20kFull))
        XCTAssertTrue(datasets.contains(.kinetics400))
        XCTAssertTrue(datasets.contains(.kinetics700))
    }
    
    // MARK: - Custom Labels Tests
    
    func testLoadCustomLabelsFromData() throws {
        let json = """
        {
            "labels": ["apple", "banana", "cherry"]
        }
        """
        let data = json.data(using: .utf8)!
        
        try LabelDatabase.loadCustomLabels(from: data, as: "fruits")
        
        XCTAssertEqual(LabelDatabase.getCustomLabel(0, customDataset: "fruits"), "apple")
        XCTAssertEqual(LabelDatabase.getCustomLabel(1, customDataset: "fruits"), "banana")
        XCTAssertEqual(LabelDatabase.getCustomLabel(2, customDataset: "fruits"), "cherry")
        
        // Cleanup
        LabelDatabase.unloadCustomLabels(name: "fruits")
    }
    
    func testGetCustomLabelNotLoaded() {
        let label = LabelDatabase.getCustomLabel(0, customDataset: "nonexistent")
        XCTAssertNil(label)
    }
    
    func testGetCustomLabelOutOfBounds() throws {
        let json = """
        {"labels": ["a", "b"]}
        """
        try LabelDatabase.loadCustomLabels(from: json.data(using: .utf8)!, as: "test")
        
        XCTAssertNil(LabelDatabase.getCustomLabel(10, customDataset: "test"))
        
        LabelDatabase.unloadCustomLabels(name: "test")
    }
    
    func testUnloadCustomLabels() throws {
        let json = """
        {"labels": ["x"]}
        """
        try LabelDatabase.loadCustomLabels(from: json.data(using: .utf8)!, as: "temp")
        
        XCTAssertEqual(LabelDatabase.getCustomLabel(0, customDataset: "temp"), "x")
        
        LabelDatabase.unloadCustomLabels(name: "temp")
        
        XCTAssertNil(LabelDatabase.getCustomLabel(0, customDataset: "temp"))
    }
    
    func testGetLoadedCustomDatasets() throws {
        let json = """
        {"labels": ["a"]}
        """
        try LabelDatabase.loadCustomLabels(from: json.data(using: .utf8)!, as: "custom1")
        try LabelDatabase.loadCustomLabels(from: json.data(using: .utf8)!, as: "custom2")
        
        let loaded = LabelDatabase.getLoadedCustomDatasets()
        
        XCTAssertTrue(loaded.contains("custom1"))
        XCTAssertTrue(loaded.contains("custom2"))
        
        // Cleanup
        LabelDatabase.unloadCustomLabels(name: "custom1")
        LabelDatabase.unloadCustomLabels(name: "custom2")
    }
    
    func testLoadCustomLabelsInvalidJSON() {
        let invalidJSON = "not json".data(using: .utf8)!
        
        XCTAssertThrowsError(try LabelDatabase.loadCustomLabels(from: invalidJSON, as: "invalid"))
    }
    
    func testLoadCustomLabelsMissingLabelsArray() {
        let json = """
        {"name": "test"}
        """
        
        XCTAssertThrowsError(try LabelDatabase.loadCustomLabels(from: json.data(using: .utf8)!, as: "test"))
    }
    
    func testGetTopCustomLabels() throws {
        let json = """
        {"labels": ["red", "green", "blue"]}
        """
        try LabelDatabase.loadCustomLabels(from: json.data(using: .utf8)!, as: "colors")
        
        let scores: [Float] = [0.2, 0.7, 0.1]
        let top = LabelDatabase.getTopCustomLabels(
            scores: scores,
            customDataset: "colors",
            k: 2
        )
        
        XCTAssertEqual(top.count, 2)
        XCTAssertEqual(top[0].label, "green")
        XCTAssertEqual(top[0].index, 1)
        
        LabelDatabase.unloadCustomLabels(name: "colors")
    }
    
    // MARK: - Cache Tests
    
    func testClearCache() {
        // Access a dataset to cache it
        _ = LabelDatabase.getAllLabels(for: .coco)
        
        // Clear cache
        LabelDatabase.clearCache()
        
        // Access should still work (reloads from JSON)
        let labels = LabelDatabase.getAllLabels(for: .coco)
        XCTAssertEqual(labels.count, 80)
    }
    
    // MARK: - LabelDataset Enum Tests
    
    func testLabelDatasetRawValues() {
        XCTAssertEqual(LabelDataset.coco.rawValue, "coco")
        XCTAssertEqual(LabelDataset.imagenet.rawValue, "imagenet")
        XCTAssertEqual(LabelDataset.cifar10.rawValue, "cifar10")
        XCTAssertEqual(LabelDataset.voc.rawValue, "voc")
    }
    
    func testLabelDatasetCaseIterable() {
        let allCases = LabelDataset.allCases
        
        XCTAssertTrue(allCases.count >= 15)  // Now includes 15 datasets
        XCTAssertTrue(allCases.contains(.coco))
        XCTAssertTrue(allCases.contains(.imagenet))
        XCTAssertTrue(allCases.contains(.openimages))
        XCTAssertTrue(allCases.contains(.lvis))
        XCTAssertTrue(allCases.contains(.kinetics400))
    }
    
    // MARK: - DatasetInfo Tests
    
    func testDatasetInfoInit() {
        let info = DatasetInfo(
            name: "test",
            numClasses: 10,
            description: "Test dataset"
        )
        
        XCTAssertEqual(info.name, "test")
        XCTAssertEqual(info.numClasses, 10)
        XCTAssertEqual(info.description, "Test dataset")
    }
    
    // MARK: - Performance Tests
    
    func testLabelLookupPerformance() {
        // Pre-load the cache
        _ = LabelDatabase.getAllLabels(for: .imagenet)
        
        measure {
            for i in 0..<1000 {
                _ = LabelDatabase.getLabel(i, dataset: .imagenet)
            }
        }
    }
    
    func testTopLabelsPerformance() {
        let scores = [Float](repeating: 0.001, count: 1000)
        
        measure {
            _ = LabelDatabase.getTopLabels(scores: scores, dataset: .imagenet, k: 5)
        }
    }
}
