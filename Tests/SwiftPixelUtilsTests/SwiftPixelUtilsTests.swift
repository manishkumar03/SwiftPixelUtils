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
}
