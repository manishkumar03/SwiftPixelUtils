import XCTest
@testable import SwiftPixelUtils

final class SwiftPixelUtilsTests: XCTestCase {
    
    // MARK: - Color Format Tests
    
    func testColorFormatChannelCount() throws {
        XCTAssertEqual(ColorFormat.rgb.channelCount, 3)
        XCTAssertEqual(ColorFormat.rgba.channelCount, 4)
        XCTAssertEqual(ColorFormat.grayscale.channelCount, 1)
        XCTAssertEqual(ColorFormat.hsv.channelCount, 3)
    }
    
    // MARK: - Bounding Box Tests
    
    func testBoxFormatConversion() throws {
        // Test CXCYWH to XYXY conversion
        let cxcywhBox: [[Float]] = [[320, 240, 100, 80]]
        let xyxyBox = BoundingBox.convertFormat(
            cxcywhBox,
            from: .cxcywh,
            to: .xyxy
        )
        
        // Center (320, 240) with width 100, height 80
        // Should be: x1 = 320 - 50 = 270, y1 = 240 - 40 = 200
        //           x2 = 320 + 50 = 370, y2 = 240 + 40 = 280
        XCTAssertEqual(xyxyBox[0][0], 270, accuracy: 0.1)
        XCTAssertEqual(xyxyBox[0][1], 200, accuracy: 0.1)
        XCTAssertEqual(xyxyBox[0][2], 370, accuracy: 0.1)
        XCTAssertEqual(xyxyBox[0][3], 280, accuracy: 0.1)
    }
    
    func testBoxScaling() throws {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 640, height: 640),
            to: CGSize(width: 1920, height: 1080),
            format: .xyxy
        )
        
        // Scale factors: x=3, y=1.6875
        XCTAssertEqual(scaled[0][0], 300, accuracy: 0.1)
        XCTAssertEqual(scaled[0][1], 168.75, accuracy: 0.1)
        XCTAssertEqual(scaled[0][2], 600, accuracy: 0.1)
        XCTAssertEqual(scaled[0][3], 337.5, accuracy: 0.1)
    }
    
    func testBoxClipping() throws {
        let boxes: [[Float]] = [[-10, 50, 700, 500]]
        let clipped = BoundingBox.clip(
            boxes,
            imageSize: CGSize(width: 640, height: 480),
            format: .xyxy
        )
        
        XCTAssertEqual(clipped[0][0], 0)
        XCTAssertEqual(clipped[0][1], 50)
        XCTAssertEqual(clipped[0][2], 640)
        XCTAssertEqual(clipped[0][3], 480)
    }
    
    func testIoUCalculation() throws {
        let box1: [Float] = [100, 100, 200, 200]
        let box2: [Float] = [150, 150, 250, 250]
        
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        
        // Intersection: 50x50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU = 2500 / 17500 ≈ 0.1428
        XCTAssertEqual(iou, 0.1428, accuracy: 0.01)
    }
    
    func testNonMaxSuppression() throws {
        let detections = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0),
            Detection(box: [300, 300, 400, 400], score: 0.7, classIndex: 1)
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        // Should keep first and third detection (second is suppressed due to overlap)
        XCTAssertEqual(filtered.count, 2)
        XCTAssertEqual(filtered[0].score, 0.9)
        XCTAssertEqual(filtered[1].score, 0.7)
    }
    
    // MARK: - Model Presets Tests
    
    func testModelPresets() throws {
        // Test YOLO preset
        let yolo = ModelPresets.yolov8
        XCTAssertEqual(yolo.colorFormat, .rgb)
        XCTAssertEqual(yolo.resize?.width, 640)
        XCTAssertEqual(yolo.resize?.height, 640)
        XCTAssertEqual(yolo.resize?.strategy, .letterbox)
        XCTAssertEqual(yolo.dataLayout, .nchw)
        
        // Test MobileNet preset
        let mobilenet = ModelPresets.mobilenet
        XCTAssertEqual(mobilenet.colorFormat, .rgb)
        XCTAssertEqual(mobilenet.resize?.width, 224)
        XCTAssertEqual(mobilenet.resize?.height, 224)
        XCTAssertEqual(mobilenet.resize?.strategy, .cover)
        XCTAssertEqual(mobilenet.dataLayout, .nhwc)
    }
    
    // MARK: - Normalization Tests
    
    func testNormalizationPresets() throws {
        let imagenet = Normalization.imagenet
        XCTAssertEqual(imagenet.preset, .imagenet)
        XCTAssertNotNil(imagenet.mean)
        XCTAssertNotNil(imagenet.std)
        XCTAssertEqual(imagenet.mean?.count, 3)
        XCTAssertEqual(imagenet.std?.count, 3)
        
        let scale = Normalization.scale
        XCTAssertEqual(scale.preset, .scale)
    }
    
    // MARK: - Error Handling Tests
    
    func testPixelUtilsError() throws {
        let error = PixelUtilsError.invalidSource("Test error")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("Test error"))
    }
    
    // MARK: - Performance Tests
    
    func testBoundingBoxPerformance() throws {
        var boxes: [[Float]] = []
        for i in 0..<1000 {
            let fi = Float(i)
            boxes.append([fi, fi, fi + 100, fi + 100])
        }
        
        measure {
            let _ = BoundingBox.convertFormat(boxes, from: .xyxy, to: .cxcywh)
        }
    }
    
    // MARK: - Tensor Operations Tests
    
    func testChannelExtraction() throws {
        // HWC data: 2x2 RGB image
        let hwcData: [Float] = [
            1, 2, 3,   // pixel (0,0): R=1, G=2, B=3
            4, 5, 6,   // pixel (0,1): R=4, G=5, B=6
            7, 8, 9,   // pixel (1,0): R=7, G=8, B=9
            10, 11, 12 // pixel (1,1): R=10, G=11, B=12
        ]
        
        // Extract red channel
        let red = try TensorOperations.extractChannel(
            data: hwcData,
            width: 2,
            height: 2,
            channels: 3,
            channelIndex: 0,
            dataLayout: .hwc
        )
        
        XCTAssertEqual(red, [1, 4, 7, 10])
        
        // Extract green channel
        let green = try TensorOperations.extractChannel(
            data: hwcData,
            width: 2,
            height: 2,
            channels: 3,
            channelIndex: 1,
            dataLayout: .hwc
        )
        
        XCTAssertEqual(green, [2, 5, 8, 11])
    }
    
    func testTensorPermute() throws {
        // HWC: 2x2x3 tensor
        let hwcData: [Float] = [
            1, 2, 3,   4, 5, 6,   // row 0
            7, 8, 9,   10, 11, 12 // row 1
        ]
        
        // Permute HWC to CHW: [2, 0, 1]
        let result = try TensorOperations.permute(
            data: hwcData,
            shape: [2, 2, 3],  // H, W, C
            order: [2, 0, 1]   // C, H, W
        )
        
        // Expected shape: [3, 2, 2]
        XCTAssertEqual(result.shape, [3, 2, 2])
        
        // Expected data: all reds, all greens, all blues
        XCTAssertEqual(result.data[0], 1)  // R(0,0)
        XCTAssertEqual(result.data[1], 4)  // R(0,1)
        XCTAssertEqual(result.data[2], 7)  // R(1,0)
        XCTAssertEqual(result.data[3], 10) // R(1,1)
        XCTAssertEqual(result.data[4], 2)  // G(0,0)
    }
    
    func testSqueeze() throws {
        let shape = [1, 3, 224, 224]
        let squeezed = TensorOperations.squeeze(shape: shape, dims: [0])
        XCTAssertEqual(squeezed, [3, 224, 224])
        
        let shape2 = [1, 1, 3, 1]
        let squeezedAll = TensorOperations.squeeze(shape: shape2, dims: nil)
        XCTAssertEqual(squeezedAll, [3])
    }
    
    func testUnsqueeze() throws {
        let shape = [3, 224, 224]
        let unsqueezed = TensorOperations.unsqueeze(shape: shape, dim: 0)
        XCTAssertEqual(unsqueezed, [1, 3, 224, 224])
    }
    
    // MARK: - Quantization Tests
    
    func testQuantizationRoundTrip() throws {
        let original: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        // Quantize to UInt8
        let quantized = try Quantizer.quantize(
            data: original,
            options: QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [Float(1.0 / 255.0)],
                zeroPoint: [0]
            )
        )
        
        XCTAssertNotNil(quantized.uint8Data)
        XCTAssertEqual(quantized.uint8Data?.count, 5)
        
        // Expected: 0, 64, 128, 191, 255 (approximately)
        XCTAssertEqual(quantized.uint8Data?[0], 0)
        XCTAssertEqual(quantized.uint8Data?[4], 255)
        
        // Dequantize
        let dequantized = try Quantizer.dequantize(
            uint8Data: quantized.uint8Data,
            scale: quantized.scale,
            zeroPoint: quantized.zeroPoint,
            mode: .perTensor
        )
        
        XCTAssertEqual(dequantized.count, 5)
        XCTAssertEqual(dequantized[0], 0, accuracy: 0.01)
        XCTAssertEqual(dequantized[4], 1, accuracy: 0.01)
    }
    
    func testCalibration() throws {
        let data: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        let (scale, zeroPoint) = Quantizer.calibrate(
            data: data,
            dtype: .int8,
            symmetric: true
        )
        
        XCTAssertGreaterThan(scale, 0)
        XCTAssertEqual(zeroPoint, 0) // Symmetric quantization has zero_point = 0
    }
    
    // MARK: - LabelDatabase Tests
    
    func testLabelDatabaseCOCO() throws {
        // Test COCO labels (embedded)
        let person = LabelDatabase.getLabel(0, dataset: .coco)
        XCTAssertEqual(person, "person")
        
        let toothbrush = LabelDatabase.getLabel(79, dataset: .coco)
        XCTAssertEqual(toothbrush, "toothbrush")
        
        // Out of bounds returns nil
        XCTAssertNil(LabelDatabase.getLabel(80, dataset: .coco))
        XCTAssertNil(LabelDatabase.getLabel(-1, dataset: .coco))
    }
    
    func testLabelDatabaseImageNet() throws {
        // Test ImageNet labels (loaded from JSON)
        let tench = LabelDatabase.getLabel(0, dataset: .imagenet)
        XCTAssertEqual(tench, "tench")
        
        let goldfish = LabelDatabase.getLabel(1, dataset: .imagenet)
        XCTAssertEqual(goldfish, "goldfish")
        
        // Check that labels are loaded (may vary by version)
        let allLabels = LabelDatabase.getAllLabels(for: .imagenet)
        XCTAssertGreaterThanOrEqual(allLabels.count, 999)
    }
    
    func testLabelDatabaseCIFAR100() throws {
        // Test CIFAR-100 labels (loaded from JSON)
        let allLabels = LabelDatabase.getAllLabels(for: .cifar100)
        XCTAssertEqual(allLabels.count, 100)
        
        // First label should be "apple"
        let apple = LabelDatabase.getLabel(0, dataset: .cifar100)
        XCTAssertEqual(apple, "apple")
    }
    
    func testLabelDatabasePlaces365() throws {
        // Test Places365 labels (loaded from JSON)
        let allLabels = LabelDatabase.getAllLabels(for: .places365)
        XCTAssertGreaterThanOrEqual(allLabels.count, 365)
        
        // First label should be "airfield"
        let airfield = LabelDatabase.getLabel(0, dataset: .places365)
        XCTAssertEqual(airfield, "airfield")
    }
    
    func testLabelDatabaseADE20K() throws {
        // Test ADE20K labels (loaded from JSON)
        let allLabels = LabelDatabase.getAllLabels(for: .ade20k)
        XCTAssertEqual(allLabels.count, 150)
        
        // First label should be "wall"
        let wall = LabelDatabase.getLabel(0, dataset: .ade20k)
        XCTAssertEqual(wall, "wall")
    }
    
    func testTopLabels() throws {
        // Create mock scores
        let scores: [Float] = [0.1, 0.05, 0.3, 0.15, 0.2, 0.05, 0.05, 0.05, 0.025, 0.025]
        
        // Get top 3 with CIFAR-10 labels
        let top3 = LabelDatabase.getTopLabels(
            scores: scores,
            dataset: .cifar10,
            k: 3,
            minConfidence: 0.0
        )
        
        XCTAssertEqual(top3.count, 3)
        XCTAssertEqual(top3[0].index, 2) // "bird" has highest score
        XCTAssertEqual(top3[0].label, "bird")
        XCTAssertEqual(top3[0].confidence, 0.3, accuracy: 0.001)
    }
    
    func testSoftmax() throws {
        let logits: [Float] = [1.0, 2.0, 3.0]
        let probs = LabelDatabase.softmax(logits)
        
        // Sum should be 1.0
        let sum = probs.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)
        
        // Highest logit should have highest probability
        XCTAssertGreaterThan(probs[2], probs[1])
        XCTAssertGreaterThan(probs[1], probs[0])
    }
    
    func testDatasetInfo() throws {
        let cocoInfo = LabelDatabase.getDatasetInfo(for: .coco)
        XCTAssertEqual(cocoInfo.name, "coco")
        XCTAssertEqual(cocoInfo.numClasses, 80)
        
        let imagenetInfo = LabelDatabase.getDatasetInfo(for: .imagenet)
        XCTAssertEqual(imagenetInfo.name, "imagenet")
        XCTAssertGreaterThanOrEqual(imagenetInfo.numClasses, 999)
    }
    
    // MARK: - Inference Utilities Tests
    
    func testActivationSoftmax() throws {
        let logits: [Float] = [-1.0, 2.0, 0.5, -0.3]
        let probs = ActivationFunctions.softmax(logits)
        
        // Sum should be 1.0
        let sum = probs.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)
        
        // Highest logit (index 1) should have highest probability
        XCTAssertEqual(probs.firstIndex(of: probs.max()!), 1)
    }
    
    func testActivationSigmoid() throws {
        let logits: [Float] = [-2.0, 0.0, 2.0]
        let probs = ActivationFunctions.sigmoid(logits)
        
        // sigmoid(0) should be 0.5
        XCTAssertEqual(probs[1], 0.5, accuracy: 0.001)
        
        // sigmoid(-2) < 0.5 < sigmoid(2)
        XCTAssertLessThan(probs[0], 0.5)
        XCTAssertGreaterThan(probs[2], 0.5)
        
        // Values should be in (0, 1) range
        for prob in probs {
            XCTAssertGreaterThan(prob, 0)
            XCTAssertLessThan(prob, 1)
        }
    }
    
    func testTopKExtraction() throws {
        let scores: [Float] = [0.1, 0.8, 0.05, 0.3, 0.9]
        let top3 = TopKExtractor.extractTopK(scores, k: 3)
        
        XCTAssertEqual(top3.indices.count, 3)
        XCTAssertEqual(top3.indices[0], 4)  // 0.9
        XCTAssertEqual(top3.indices[1], 1)  // 0.8
        XCTAssertEqual(top3.indices[2], 3)  // 0.3
        
        XCTAssertEqual(top3.values[0], 0.9, accuracy: 0.001)
        XCTAssertEqual(top3.values[1], 0.8, accuracy: 0.001)
        XCTAssertEqual(top3.values[2], 0.3, accuracy: 0.001)
    }
    
    func testArgmax() throws {
        let scores: [Float] = [0.15, 0.72, 0.08, 0.95, 0.43]
        guard let idx = TopKExtractor.argmax(scores) else {
            XCTFail("Argmax returned nil")
            return
        }
        
        XCTAssertEqual(idx, 3)
        XCTAssertEqual(scores[idx], 0.95, accuracy: 0.001)
    }
    
    func testArgmin() throws {
        let scores: [Float] = [0.15, 0.72, 0.08, 0.95, 0.43]
        guard let idx = TopKExtractor.argmin(scores) else {
            XCTFail("Argmin returned nil")
            return
        }
        
        XCTAssertEqual(idx, 2)
        XCTAssertEqual(scores[idx], 0.08, accuracy: 0.001)
    }
    
    func testSoftNMS() throws {
        let boxes: [[Float]] = [
            [10, 10, 50, 50],
            [12, 12, 52, 52], // Overlaps
            [100, 100, 150, 150], // Isolated
        ]
        let scores: [Float] = [0.9, 0.85, 0.8]
        
        let (indices, filteredScores) = NMSVariants.softNMS(
            boxes: boxes,
            scores: scores,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        // Soft-NMS reduces scores instead of removing boxes
        // High-confidence and isolated detections should remain
        XCTAssertGreaterThanOrEqual(indices.count, 2)
        XCTAssertEqual(filteredScores[0], 0.9, accuracy: 0.001) // Top score unchanged
    }
    
    func testConfidenceFiltering() throws {
        let detections: [(item: Int, confidence: Float)] = [
            (0, 0.95),
            (1, 0.45),
            (2, 0.72),
        ]
        
        let filtered = ConfidenceFilter.filter(detections, threshold: 0.5)
        
        XCTAssertEqual(filtered.count, 2)
        XCTAssertTrue(filtered.allSatisfy { $0.confidence >= 0.5 })
    }
    
    func testMaskThreshold() throws {
        let probMask: [Float] = [0.2, 0.6, 0.8, 0.3, 0.9, 0.4]
        let binary = MaskUtilities.threshold(probMask, threshold: 0.5)
        
        XCTAssertEqual(binary, [0, 1, 1, 0, 1, 0])
    }
    
    func testMaskResize() throws {
        let mask: [Float] = [
            1.0, 0.0,
            0.0, 1.0
        ]
        
        let resized = MaskUtilities.resizeMask(
            mask,
            fromSize: (width: 2, height: 2),
            toSize: (width: 4, height: 4)
        )
        
        XCTAssertEqual(resized.width, 4)
        XCTAssertEqual(resized.height, 4)
        XCTAssertEqual(resized.mask.count, 16)
    }
    
    func testMaskIoU() throws {
        let mask1: [Float] = [1, 1, 0, 0, 1, 1, 0, 0, 0]
        let mask2: [Float] = [0, 1, 1, 0, 1, 1, 0, 0, 0]
        
        let iou = MaskUtilities.maskIoU(mask1, mask2)
        
        // Intersection: 3, Union: 5
        XCTAssertEqual(iou, 0.6, accuracy: 0.001)
    }
    
    func testMaskArea() throws {
        let mask: [Float] = [1, 1, 0, 1, 0, 0, 1, 1, 1]
        let area = MaskUtilities.computeArea(mask)
        
        XCTAssertEqual(area, 6)
    }
    
    #if canImport(CoreML)
    func testCoreMLConversion() throws {
        let data: [Float] = Array(repeating: 0.5, count: 48) // 1x3x4x4
        let shape = [1, 3, 4, 4]
        
        let multiArray = try CoreMLConversion.toMLMultiArray(data, shape: shape)
        
        XCTAssertEqual(multiArray.shape.map { $0.intValue }, shape)
        XCTAssertEqual(multiArray.count, 48)
        
        // Round trip
        let roundTripData = CoreMLConversion.fromMLMultiArray(multiArray)
        XCTAssertEqual(roundTripData.count, data.count)
    }
    #endif
    
    // MARK: - DetectionOutput Tests
    
    func testDetectionOutputYOLOv8() throws {
        // Simulate YOLOv8 output: [1, 84, N] format (4 boxes + 80 classes, 3 detections)
        // We'll use the transposed format for easier setup: [N, 84]
        let numClasses = 80
        let numDetections = 3
        let stride = 4 + numClasses  // 84
        
        var output = [Float](repeating: 0, count: numDetections * stride)
        
        // Detection 1: person (class 0) at center (320, 240) with size (100, 200), confidence 0.9
        output[0] = 320  // cx
        output[1] = 240  // cy
        output[2] = 100  // w
        output[3] = 200  // h
        output[4] = 0.9  // class 0 (person) confidence
        
        // Detection 2: car (class 2) at center (400, 300), confidence 0.7
        output[stride + 0] = 400
        output[stride + 1] = 300
        output[stride + 2] = 150
        output[stride + 3] = 100
        output[stride + 6] = 0.7  // class 2 (car) confidence
        
        // Detection 3: low confidence detection, should be filtered
        output[stride * 2 + 0] = 100
        output[stride * 2 + 1] = 100
        output[stride * 2 + 2] = 50
        output[stride * 2 + 3] = 50
        output[stride * 2 + 4] = 0.1  // below threshold
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov8Transposed(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,
            labels: .coco,
            modelInputSize: CGSize(width: 640, height: 640)
        )
        
        // Should have 2 detections after confidence filtering
        XCTAssertEqual(result.detections.count, 2)
        XCTAssertEqual(result.rawDetectionCount, 3)
        
        // Check first detection (highest confidence)
        let topDetection = result.detections[0]
        XCTAssertEqual(topDetection.classIndex, 0)
        XCTAssertEqual(topDetection.label, "person")
        XCTAssertEqual(topDetection.confidence, 0.9, accuracy: 0.01)
        
        // Check second detection
        let secondDetection = result.detections[1]
        XCTAssertEqual(secondDetection.classIndex, 2)
        XCTAssertEqual(secondDetection.label, "car")
        XCTAssertEqual(secondDetection.confidence, 0.7, accuracy: 0.01)
    }
    
    func testDetectionOutputNMS() throws {
        // Test NMS by creating overlapping detections
        let numClasses = 80
        let stride = 4 + numClasses
        
        var output = [Float](repeating: 0, count: 2 * stride)
        
        // Two overlapping detections of the same class
        // Detection 1: high confidence
        output[0] = 320  // cx
        output[1] = 240  // cy
        output[2] = 100  // w
        output[3] = 100  // h
        output[4] = 0.9  // class 0 confidence
        
        // Detection 2: lower confidence, overlapping significantly
        output[stride + 0] = 330  // cx (slightly offset)
        output[stride + 1] = 250  // cy
        output[stride + 2] = 100  // w
        output[stride + 3] = 100  // h
        output[stride + 4] = 0.7  // class 0 confidence
        
        let result = try DetectionOutput.process(
            floatOutput: output,
            format: .yolov8Transposed(numClasses: numClasses),
            confidenceThreshold: 0.5,
            iouThreshold: 0.45,  // Low threshold, should suppress overlapping
            labels: .coco,
            modelInputSize: CGSize(width: 640, height: 640)
        )
        
        // NMS should keep only one detection due to high overlap
        XCTAssertEqual(result.detections.count, 1)
        XCTAssertEqual(result.detections[0].confidence, 0.9, accuracy: 0.01)
    }
    
    func testDetectionResultFiltering() throws {
        // Create a DetectionResult with multiple classes
        let detections = [
            ObjectDetection(classIndex: 0, label: "person", confidence: 0.9, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.3)),
            ObjectDetection(classIndex: 0, label: "person", confidence: 0.7, boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.2, height: 0.3)),
            ObjectDetection(classIndex: 2, label: "car", confidence: 0.8, boundingBox: CGRect(x: 0.3, y: 0.3, width: 0.3, height: 0.2))
        ]
        
        let result = DetectionResult(
            detections: detections,
            processingTimeMs: 10.0,
            rawDetectionCount: 100,
            postConfidenceFilterCount: 10
        )
        
        // Test filtering by class
        let personDetections = result.filter(byClass: 0)
        XCTAssertEqual(personDetections.count, 2)
        
        let carDetections = result.filter(byClass: 2)
        XCTAssertEqual(carDetections.count, 1)
        
        // Test filtering by label
        let personsByLabel = result.filter(byLabel: "person")
        XCTAssertEqual(personsByLabel.count, 2)
        
        let carsByLabel = result.filter(byLabel: "Car")  // Case insensitive
        XCTAssertEqual(carsByLabel.count, 1)
    }
    
    // MARK: - Segmentation Output Tests
    
    func testSegmentationOutputBasic() throws {
        // Create a simple 4x4 segmentation with 3 classes
        // Shape: [1, 4, 4, 3] (NHWC) = 48 floats
        let height = 4
        let width = 4
        let numClasses = 3
        
        var logits = [Float](repeating: 0, count: height * width * numClasses)
        
        // Make specific pixels belong to specific classes by setting higher logits
        // Row 0,1: Class 0 (background)
        // Row 2,3: Class 1 (person)
        for y in 0..<height {
            for x in 0..<width {
                let pixelBase = (y * width + x) * numClasses
                if y < 2 {
                    logits[pixelBase + 0] = 5.0  // Class 0
                    logits[pixelBase + 1] = 0.0
                    logits[pixelBase + 2] = 0.0
                } else {
                    logits[pixelBase + 0] = 0.0
                    logits[pixelBase + 1] = 5.0  // Class 1
                    logits[pixelBase + 2] = 0.0
                }
            }
        }
        
        let result = try SegmentationOutput.process(
            floatOutput: logits,
            format: .logits(height: height, width: width, numClasses: numClasses),
            labels: .custom(["background", "person", "car"])
        )
        
        // Verify dimensions
        XCTAssertEqual(result.width, width)
        XCTAssertEqual(result.height, height)
        XCTAssertEqual(result.numClasses, numClasses)
        
        // Verify class mask
        XCTAssertEqual(result.classMask.count, height * width)
        XCTAssertEqual(result.classAt(x: 0, y: 0), 0)  // Background
        XCTAssertEqual(result.classAt(x: 0, y: 2), 1)  // Person
        
        // Verify class counts
        XCTAssertEqual(result.classPixelCounts[0], 8)  // Top half: 4*2 = 8 pixels
        XCTAssertEqual(result.classPixelCounts[1], 8)  // Bottom half: 4*2 = 8 pixels
        
        // Verify presentClasses excludes background
        XCTAssertEqual(result.presentClasses, [1])
        
        // Verify class summary
        XCTAssertEqual(result.classSummary.count, 1)  // Only non-background
        XCTAssertEqual(result.classSummary[0].label, "person")
        XCTAssertEqual(result.classSummary[0].percentage, 50.0, accuracy: 0.1)
    }
    
    func testSegmentationBinaryMask() throws {
        let height = 2
        let width = 2
        let numClasses = 2
        
        // Create logits: top-left = class 1, rest = class 0
        var logits = [Float](repeating: 0, count: height * width * numClasses)
        logits[0] = 0.0; logits[1] = 5.0  // Pixel (0,0): Class 1
        logits[2] = 5.0; logits[3] = 0.0  // Pixel (1,0): Class 0
        logits[4] = 5.0; logits[5] = 0.0  // Pixel (0,1): Class 0
        logits[6] = 5.0; logits[7] = 0.0  // Pixel (1,1): Class 0
        
        let result = try SegmentationOutput.process(
            floatOutput: logits,
            format: .logits(height: height, width: width, numClasses: numClasses),
            labels: .none
        )
        
        let binaryMask = result.binaryMask(forClass: 1)
        XCTAssertEqual(binaryMask, [1.0, 0.0, 0.0, 0.0])
    }
    
    func testSegmentationColorPalettes() throws {
        // Test VOC palette has correct number of colors
        XCTAssertEqual(SegmentationColorPalette.voc.colors.count, 21)
        
        // Test color cycling
        let color0 = SegmentationColorPalette.voc.color(forClassIndex: 0)
        let color21 = SegmentationColorPalette.voc.color(forClassIndex: 21)
        XCTAssertEqual(color0.r, color21.r)  // Should cycle back
        
        // Test rainbow palette generation
        let rainbow = SegmentationColorPalette.rainbow(numClasses: 10)
        XCTAssertEqual(rainbow.colors.count, 10)
        
        // First color should be black (background)
        let bg = rainbow.color(forClassIndex: 0)
        XCTAssertEqual(bg.r, 0)
        XCTAssertEqual(bg.g, 0)
        XCTAssertEqual(bg.b, 0)
    }
}
