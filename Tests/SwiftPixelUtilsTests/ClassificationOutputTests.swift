import XCTest
@testable import SwiftPixelUtils

/// Tests for ``ClassificationOutput`` - Classification model output processing.
///
/// ## Topics
///
/// ### Prediction Tests
/// - ClassificationPrediction initialization and equality
/// - Label and confidence handling
///
/// ### Result Processing Tests
/// - Top-K prediction extraction
/// - Confidence threshold filtering
/// - Label database integration
///
/// ### Edge Cases
/// - Empty logits, single class, equal probabilities
/// - Very high/low confidence values
final class ClassificationOutputTests: XCTestCase {
    
    // MARK: - ClassificationPrediction Tests
    
    func testClassificationPredictionInit() {
        let prediction = ClassificationPrediction(
            classIndex: 5,
            label: "dog",
            confidence: 0.95
        )
        
        XCTAssertEqual(prediction.classIndex, 5)
        XCTAssertEqual(prediction.label, "dog")
        XCTAssertEqual(prediction.confidence, 0.95)
    }
    
    func testClassificationPredictionEquatable() {
        let pred1 = ClassificationPrediction(classIndex: 1, label: "cat", confidence: 0.9)
        let pred2 = ClassificationPrediction(classIndex: 1, label: "cat", confidence: 0.9)
        let pred3 = ClassificationPrediction(classIndex: 2, label: "dog", confidence: 0.8)
        
        XCTAssertEqual(pred1, pred2)
        XCTAssertNotEqual(pred1, pred3)
    }
    
    // MARK: - ClassificationResult Tests
    
    func testClassificationResultInit() {
        let predictions = [
            ClassificationPrediction(classIndex: 0, label: "cat", confidence: 0.8),
            ClassificationPrediction(classIndex: 1, label: "dog", confidence: 0.15)
        ]
        
        let result = ClassificationResult(
            predictions: predictions,
            processingTimeMs: 5.5,
            softmaxApplied: true,
            dequantized: false
        )
        
        XCTAssertEqual(result.predictions.count, 2)
        XCTAssertEqual(result.processingTimeMs, 5.5)
        XCTAssertTrue(result.softmaxApplied)
        XCTAssertFalse(result.dequantized)
    }
    
    func testClassificationResultTopPrediction() {
        let predictions = [
            ClassificationPrediction(classIndex: 0, label: "cat", confidence: 0.9),
            ClassificationPrediction(classIndex: 1, label: "dog", confidence: 0.1)
        ]
        
        let result = ClassificationResult(
            predictions: predictions,
            processingTimeMs: 1.0,
            softmaxApplied: true,
            dequantized: false
        )
        
        XCTAssertEqual(result.topPrediction?.label, "cat")
        XCTAssertEqual(result.topPrediction?.confidence, 0.9)
    }
    
    func testClassificationResultTopPredictionEmpty() {
        let result = ClassificationResult(
            predictions: [],
            processingTimeMs: 1.0,
            softmaxApplied: true,
            dequantized: false
        )
        
        XCTAssertNil(result.topPrediction)
    }
    
    // MARK: - Process Float Array Tests
    
    func testProcessFloatArrayWithSoftmax() throws {
        // Logits that will become clear after softmax
        let logits: [Float] = [10, 1, 0.1, 0.01, 0.001]
        
        let result = try ClassificationOutput.process(
            floatOutput: logits,
            topK: 3,
            labels: .none,
            applySoftmax: true
        )
        
        XCTAssertEqual(result.predictions.count, 3)
        XCTAssertTrue(result.softmaxApplied)
        XCTAssertFalse(result.dequantized)
        
        // First prediction should have highest confidence
        XCTAssertEqual(result.predictions[0].classIndex, 0)
        XCTAssertGreaterThan(result.predictions[0].confidence, 0.9)
    }
    
    func testProcessFloatArrayWithoutSoftmax() throws {
        let scores: [Float] = [0.1, 0.5, 0.3, 0.05, 0.05]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 2,
            labels: .none,
            applySoftmax: false
        )
        
        XCTAssertFalse(result.softmaxApplied)
        XCTAssertEqual(result.predictions.count, 2)
        XCTAssertEqual(result.predictions[0].classIndex, 1)  // Index 1 has 0.5
    }
    
    func testProcessFloatArrayTopK1() throws {
        let scores: [Float] = [0.1, 0.9]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 1,
            labels: .none,
            applySoftmax: false
        )
        
        XCTAssertEqual(result.predictions.count, 1)
        XCTAssertEqual(result.predictions[0].classIndex, 1)
    }
    
    func testProcessFloatArrayAllClasses() throws {
        let scores: [Float] = [0.1, 0.2, 0.3, 0.4]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 10,  // More than available
            labels: .none,
            applySoftmax: false
        )
        
        // Should return all 4 classes
        XCTAssertLessThanOrEqual(result.predictions.count, 4)
    }
    
    // MARK: - Process Data Tests
    
    func testProcessDataNoQuantization() throws {
        var logits: [Float] = [5.0, 2.0, 1.0]
        let data = Data(bytes: &logits, count: logits.count * MemoryLayout<Float>.size)
        
        let result = try ClassificationOutput.process(
            outputData: data,
            quantization: .none,
            topK: 2,
            labels: .none
        )
        
        XCTAssertEqual(result.predictions.count, 2)
        XCTAssertFalse(result.dequantized)
    }
    
    func testProcessDataUInt8Quantization() throws {
        let uint8Data: [UInt8] = [200, 100, 50, 25]  // Quantized values
        let data = Data(uint8Data)
        
        let result = try ClassificationOutput.process(
            outputData: data,
            quantization: .uint8(scale: 0.00392157, zeroPoint: 0),
            topK: 2,
            labels: .none
        )
        
        XCTAssertEqual(result.predictions.count, 2)
        XCTAssertTrue(result.dequantized)
        XCTAssertEqual(result.predictions[0].classIndex, 0)  // Highest value
    }
    
    func testProcessDataInt8Quantization() throws {
        var int8Data: [Int8] = [100, 50, -10, -50]
        let data = Data(bytes: &int8Data, count: int8Data.count)
        
        let result = try ClassificationOutput.process(
            outputData: data,
            quantization: .int8(scale: 0.01, zeroPoint: 0),
            topK: 2,
            labels: .none
        )
        
        XCTAssertEqual(result.predictions.count, 2)
        XCTAssertTrue(result.dequantized)
    }
    
    // MARK: - Process UInt8 Convenience Method Tests
    
    func testProcessUInt8ArrayConvenience() throws {
        let uint8Output: [UInt8] = [255, 128, 64, 32]
        
        let result = try ClassificationOutput.process(
            uint8Output: uint8Output,
            scale: 0.00392157,
            zeroPoint: 0,
            topK: 2
        )
        
        XCTAssertEqual(result.predictions.count, 2)
        XCTAssertTrue(result.dequantized)
    }
    
    // MARK: - Label Mapping Tests
    
    func testProcessWithNoLabels() throws {
        let scores: [Float] = [0.1, 0.9]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 2,
            labels: .none
        )
        
        XCTAssertEqual(result.predictions[0].label, "Class 1")
        XCTAssertEqual(result.predictions[1].label, "Class 0")
    }
    
    func testProcessWithCustomLabels() throws {
        let scores: [Float] = [0.1, 0.9]
        let customLabels = ["cat", "dog"]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 2,
            labels: .custom(customLabels)
        )
        
        XCTAssertEqual(result.predictions[0].label, "dog")
        XCTAssertEqual(result.predictions[1].label, "cat")
    }
    
    func testProcessWithCustomLabelsOutOfRange() throws {
        let scores: [Float] = [0.1, 0.8, 0.1]
        let customLabels = ["cat", "dog"]  // Only 2 labels for 3 classes
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 3,
            labels: .custom(customLabels)
        )
        
        // Third class should have unknown label
        XCTAssertTrue(result.predictions.last?.label.contains("Unknown") ?? false)
    }
    
    func testProcessWithImageNetLabels() throws {
        // Create a simple logit array where index 0 has highest score
        var scores = [Float](repeating: 0.001, count: 1000)
        scores[281] = 0.9  // 281 is "tabby cat" in ImageNet
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 1,
            labels: .imagenet(hasBackgroundClass: false),
            applySoftmax: false
        )
        
        XCTAssertEqual(result.predictions[0].classIndex, 281)
        // Label should be something from ImageNet (depends on LabelDatabase)
        XCTAssertFalse(result.predictions[0].label.isEmpty)
    }
    
    func testProcessWithImageNetBackgroundClass() throws {
        var scores = [Float](repeating: 0.001, count: 1001)  // 1001 for background class
        scores[282] = 0.9  // 282 because 0 is background
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 1,
            labels: .imagenet(hasBackgroundClass: true),
            applySoftmax: false
        )
        
        // Index 282 with background class maps to ImageNet index 281
        XCTAssertEqual(result.predictions[0].classIndex, 282)
    }
    
    func testProcessWithCOCOLabels() throws {
        var scores = [Float](repeating: 0.01, count: 80)
        scores[0] = 0.9  // Person in COCO
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 1,
            labels: .coco,
            applySoftmax: false
        )
        
        XCTAssertEqual(result.predictions[0].classIndex, 0)
    }
    
    func testProcessWithCIFAR10Labels() throws {
        var scores = [Float](repeating: 0.05, count: 10)
        scores[3] = 0.9  // cat in CIFAR-10
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 1,
            labels: .cifar10,
            applySoftmax: false
        )
        
        XCTAssertEqual(result.predictions[0].classIndex, 3)
    }
    
    func testProcessWithCIFAR100Labels() throws {
        var scores = [Float](repeating: 0.005, count: 100)
        scores[8] = 0.9
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 1,
            labels: .cifar100,
            applySoftmax: false
        )
        
        XCTAssertEqual(result.predictions[0].classIndex, 8)
    }
    
    // MARK: - Temperature Tests
    
    func testProcessWithTemperature() throws {
        let logits: [Float] = [2.0, 1.0, 0.5]
        
        // Higher temperature makes distribution more uniform
        let result1 = try ClassificationOutput.process(
            floatOutput: logits,
            topK: 3,
            labels: .none,
            applySoftmax: true,
            temperature: 1.0
        )
        
        let result2 = try ClassificationOutput.process(
            floatOutput: logits,
            topK: 3,
            labels: .none,
            applySoftmax: true,
            temperature: 2.0
        )
        
        // With higher temperature, the gap between probabilities should be smaller
        let gap1 = result1.predictions[0].confidence - result1.predictions[2].confidence
        let gap2 = result2.predictions[0].confidence - result2.predictions[2].confidence
        
        XCTAssertLessThan(gap2, gap1)
    }
    
    func testProcessWithLowTemperature() throws {
        let logits: [Float] = [2.0, 1.0, 0.5]
        
        let result = try ClassificationOutput.process(
            floatOutput: logits,
            topK: 3,
            labels: .none,
            applySoftmax: true,
            temperature: 0.5  // Lower temperature = more confident
        )
        
        // Top prediction should be very confident with low temperature
        XCTAssertGreaterThan(result.predictions[0].confidence, 0.7)
    }
    
    // MARK: - Edge Cases
    
    func testProcessEmptyOutput() throws {
        let scores: [Float] = []
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 5,
            labels: .none
        )
        
        XCTAssertTrue(result.predictions.isEmpty)
    }
    
    func testProcessSingleClass() throws {
        let scores: [Float] = [1.0]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 5,
            labels: .none,
            applySoftmax: true
        )
        
        XCTAssertEqual(result.predictions.count, 1)
        XCTAssertEqual(result.predictions[0].confidence, 1.0, accuracy: 0.01)
    }
    
    func testProcessAllEqualScores() throws {
        let scores: [Float] = [0.5, 0.5, 0.5, 0.5]
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 4,
            labels: .none,
            applySoftmax: true
        )
        
        // All probabilities should be equal after softmax
        for prediction in result.predictions {
            XCTAssertEqual(prediction.confidence, 0.25, accuracy: 0.01)
        }
    }
    
    func testProcessNegativeLogits() throws {
        let logits: [Float] = [-1.0, -2.0, -3.0]
        
        let result = try ClassificationOutput.process(
            floatOutput: logits,
            topK: 3,
            labels: .none,
            applySoftmax: true
        )
        
        XCTAssertEqual(result.predictions.count, 3)
        XCTAssertEqual(result.predictions[0].classIndex, 0)  // Highest (least negative)
    }
    
    func testProcessLargeLogits() throws {
        let logits: [Float] = [100, 50, 0]  // Large values
        
        let result = try ClassificationOutput.process(
            floatOutput: logits,
            topK: 3,
            labels: .none,
            applySoftmax: true
        )
        
        // Should handle large logits without overflow
        XCTAssertEqual(result.predictions[0].classIndex, 0)
        XCTAssertGreaterThan(result.predictions[0].confidence, 0.99)
    }
    
    // MARK: - Processing Time Tests
    
    func testProcessingTimeRecorded() throws {
        let scores: [Float] = Array(repeating: 0.01, count: 1000)
        
        let result = try ClassificationOutput.process(
            floatOutput: scores,
            topK: 5,
            labels: .imagenet(hasBackgroundClass: false)
        )
        
        XCTAssertGreaterThan(result.processingTimeMs, 0)
    }
    
    // MARK: - QuantizationType Tests
    
    func testQuantizationTypeNone() {
        let qtype = ClassificationOutput.QuantizationType.none
        if case .none = qtype {
            // Success
        } else {
            XCTFail("Expected .none")
        }
    }
    
    func testQuantizationTypeUInt8() {
        let qtype = ClassificationOutput.QuantizationType.uint8(scale: 0.5, zeroPoint: 128)
        if case .uint8(let scale, let zp) = qtype {
            XCTAssertEqual(scale, 0.5)
            XCTAssertEqual(zp, 128)
        } else {
            XCTFail("Expected .uint8")
        }
    }
    
    func testQuantizationTypeInt8() {
        let qtype = ClassificationOutput.QuantizationType.int8(scale: 0.01, zeroPoint: 0)
        if case .int8(let scale, let zp) = qtype {
            XCTAssertEqual(scale, 0.01)
            XCTAssertEqual(zp, 0)
        } else {
            XCTFail("Expected .int8")
        }
    }
    
    // MARK: - LabelSource Tests
    
    func testLabelSourceImageNet() {
        let source = ClassificationOutput.LabelSource.imagenet(hasBackgroundClass: true)
        if case .imagenet(let hasBackground) = source {
            XCTAssertTrue(hasBackground)
        } else {
            XCTFail("Expected .imagenet")
        }
    }
    
    func testLabelSourceCustom() {
        let labels = ["a", "b", "c"]
        let source = ClassificationOutput.LabelSource.custom(labels)
        if case .custom(let customLabels) = source {
            XCTAssertEqual(customLabels, labels)
        } else {
            XCTFail("Expected .custom")
        }
    }
    
    // MARK: - Performance Tests
    
    func testProcessPerformance() {
        let scores = [Float](repeating: 0.001, count: 1000)
        
        measure {
            let _ = try? ClassificationOutput.process(
                floatOutput: scores,
                topK: 5,
                labels: .imagenet(hasBackgroundClass: false),
                applySoftmax: true
            )
        }
    }
}
