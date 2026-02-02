import XCTest
import CoreGraphics
@testable import SwiftPixelUtils

/// Tests for ONNX Runtime integration helpers.
///
/// ## Topics
///
/// ### ONNXDataType Tests
/// - Element size calculations
/// - ONNX type name mapping
/// - Codable conformance
///
/// ### ONNXTensorInput Tests
/// - Initialization and validation
/// - Element count and data size calculations
///
/// ### ONNXDetectionFormat Tests
/// - Format properties (center format, sigmoid)
/// - Codable conformance
///
/// ### ONNXModelConfig Tests
/// - Pre-configured model configs
/// - Custom config creation
///
/// ### ONNXHelper Tests
/// - Detection output parsing (YOLOv8, RT-DETR)
/// - Classification output parsing
/// - Segmentation output parsing
/// - NMS application
/// - IoU computation
/// - Detection scaling
final class ONNXHelperTests: XCTestCase {
    
    // MARK: - ONNXDataType Tests
    
    func testONNXDataTypeElementSize() {
        XCTAssertEqual(ONNXDataType.float32.elementSize, 4)
        XCTAssertEqual(ONNXDataType.float16.elementSize, 2)
        XCTAssertEqual(ONNXDataType.uint8.elementSize, 1)
        XCTAssertEqual(ONNXDataType.int8.elementSize, 1)
        XCTAssertEqual(ONNXDataType.int32.elementSize, 4)
        XCTAssertEqual(ONNXDataType.int64.elementSize, 8)
        XCTAssertEqual(ONNXDataType.bool.elementSize, 1)
    }
    
    func testONNXDataTypeOnnxTypeName() {
        XCTAssertEqual(ONNXDataType.float32.onnxTypeName, "tensor(float)")
        XCTAssertEqual(ONNXDataType.float16.onnxTypeName, "tensor(float16)")
        XCTAssertEqual(ONNXDataType.uint8.onnxTypeName, "tensor(uint8)")
        XCTAssertEqual(ONNXDataType.int8.onnxTypeName, "tensor(int8)")
        XCTAssertEqual(ONNXDataType.int32.onnxTypeName, "tensor(int32)")
        XCTAssertEqual(ONNXDataType.int64.onnxTypeName, "tensor(int64)")
        XCTAssertEqual(ONNXDataType.bool.onnxTypeName, "tensor(bool)")
    }
    
    func testONNXDataTypeCodable() throws {
        let types: [ONNXDataType] = [.float32, .float16, .uint8, .int8, .int32, .int64, .bool]
        
        for type in types {
            let encoded = try JSONEncoder().encode(type)
            let decoded = try JSONDecoder().decode(ONNXDataType.self, from: encoded)
            XCTAssertEqual(type, decoded)
        }
    }
    
    // MARK: - ONNXTensorInput Tests
    
    func testONNXTensorInputInit() {
        let data = Data(repeating: 0, count: 1 * 3 * 224 * 224 * 4)  // Float32
        let tensor = ONNXTensorInput(
            name: "input",
            data: data,
            shape: [1, 3, 224, 224],
            dataType: .float32
        )
        
        XCTAssertEqual(tensor.name, "input")
        XCTAssertEqual(tensor.shape, [1, 3, 224, 224])
        XCTAssertEqual(tensor.dataType, .float32)
    }
    
    func testONNXTensorInputElementCount() {
        let tensor = ONNXTensorInput(
            name: "images",
            data: Data(),
            shape: [1, 3, 640, 640],
            dataType: .float32
        )
        
        XCTAssertEqual(tensor.elementCount, 1 * 3 * 640 * 640)
        XCTAssertEqual(tensor.elementCount, 1_228_800)
    }
    
    func testONNXTensorInputExpectedDataSize() {
        // Float32: 4 bytes per element
        let tensorFloat32 = ONNXTensorInput(
            name: "input",
            data: Data(),
            shape: [1, 3, 224, 224],
            dataType: .float32
        )
        XCTAssertEqual(tensorFloat32.expectedDataSize, 1 * 3 * 224 * 224 * 4)
        
        // Float16: 2 bytes per element
        let tensorFloat16 = ONNXTensorInput(
            name: "input",
            data: Data(),
            shape: [1, 3, 224, 224],
            dataType: .float16
        )
        XCTAssertEqual(tensorFloat16.expectedDataSize, 1 * 3 * 224 * 224 * 2)
        
        // UInt8: 1 byte per element
        let tensorUInt8 = ONNXTensorInput(
            name: "input",
            data: Data(),
            shape: [1, 3, 224, 224],
            dataType: .uint8
        )
        XCTAssertEqual(tensorUInt8.expectedDataSize, 1 * 3 * 224 * 224)
    }
    
    func testONNXTensorInputValidation() {
        // Valid tensor
        let validData = Data(repeating: 0, count: 1 * 3 * 4 * 4 * 4)  // Float32
        let validTensor = ONNXTensorInput(
            name: "input",
            data: validData,
            shape: [1, 3, 4, 4],
            dataType: .float32
        )
        XCTAssertTrue(validTensor.isValid)
        
        // Invalid tensor (wrong data size)
        let invalidData = Data(repeating: 0, count: 100)  // Wrong size
        let invalidTensor = ONNXTensorInput(
            name: "input",
            data: invalidData,
            shape: [1, 3, 4, 4],
            dataType: .float32
        )
        XCTAssertFalse(invalidTensor.isValid)
    }
    
    func testONNXTensorInputCodable() throws {
        let data = Data([1, 2, 3, 4])
        let tensor = ONNXTensorInput(
            name: "test",
            data: data,
            shape: [1, 1, 1, 1],
            dataType: .uint8
        )
        
        let encoded = try JSONEncoder().encode(tensor)
        let decoded = try JSONDecoder().decode(ONNXTensorInput.self, from: encoded)
        
        XCTAssertEqual(tensor.name, decoded.name)
        XCTAssertEqual(tensor.shape, decoded.shape)
        XCTAssertEqual(tensor.dataType, decoded.dataType)
        XCTAssertEqual(tensor.data, decoded.data)
    }
    
    // MARK: - ONNXDetectionFormat Tests
    
    func testONNXDetectionFormatIsCenterFormat() {
        XCTAssertTrue(ONNXDetectionFormat.yoloV8.isCenterFormat)
        XCTAssertTrue(ONNXDetectionFormat.yoloV5.isCenterFormat)
        XCTAssertTrue(ONNXDetectionFormat.rtdetr.isCenterFormat)
        XCTAssertFalse(ONNXDetectionFormat.ssd.isCenterFormat)
        XCTAssertFalse(ONNXDetectionFormat.generic.isCenterFormat)
    }
    
    func testONNXDetectionFormatNeedsSigmoid() {
        XCTAssertTrue(ONNXDetectionFormat.yoloV8.needsSigmoid)
        XCTAssertTrue(ONNXDetectionFormat.yoloV5.needsSigmoid)
        XCTAssertFalse(ONNXDetectionFormat.rtdetr.needsSigmoid)
        XCTAssertFalse(ONNXDetectionFormat.ssd.needsSigmoid)
        XCTAssertFalse(ONNXDetectionFormat.generic.needsSigmoid)
    }
    
    func testONNXDetectionFormatCodable() throws {
        let formats: [ONNXDetectionFormat] = [.yoloV8, .yoloV5, .rtdetr, .ssd, .generic]
        
        for format in formats {
            let encoded = try JSONEncoder().encode(format)
            let decoded = try JSONDecoder().decode(ONNXDetectionFormat.self, from: encoded)
            XCTAssertEqual(format, decoded)
        }
    }
    
    // MARK: - ONNXModelConfig Tests
    
    func testONNXModelConfigYOLOv8() {
        let config = ONNXModelConfig.yolov8
        
        XCTAssertEqual(config.inputName, "images")
        XCTAssertEqual(config.inputShape, [1, 3, 640, 640])
        XCTAssertEqual(config.inputDataType, .float32)
        XCTAssertEqual(config.framework, .onnxRaw)
        XCTAssertEqual(config.detectionFormat, .yoloV8)
    }
    
    func testONNXModelConfigRTDETR() {
        let config = ONNXModelConfig.rtdetr
        
        XCTAssertEqual(config.inputName, "images")
        XCTAssertEqual(config.inputShape, [1, 3, 640, 640])
        XCTAssertEqual(config.inputDataType, .float32)
        XCTAssertEqual(config.framework, .onnx)
        XCTAssertEqual(config.detectionFormat, .rtdetr)
    }
    
    func testONNXModelConfigResNet() {
        let config = ONNXModelConfig.resnet
        
        XCTAssertEqual(config.inputName, "input")
        XCTAssertEqual(config.inputShape, [1, 3, 224, 224])
        XCTAssertEqual(config.inputDataType, .float32)
        XCTAssertEqual(config.framework, .onnx)
        XCTAssertNil(config.detectionFormat)
    }
    
    func testONNXModelConfigViT() {
        let config = ONNXModelConfig.vit
        
        XCTAssertEqual(config.inputName, "pixel_values")
        XCTAssertEqual(config.inputShape, [1, 3, 224, 224])
        XCTAssertEqual(config.inputDataType, .float32)
        XCTAssertNil(config.detectionFormat)
    }
    
    func testONNXModelConfigCLIP() {
        let config = ONNXModelConfig.clip
        
        XCTAssertEqual(config.inputName, "pixel_values")
        XCTAssertEqual(config.inputShape, [1, 3, 224, 224])
    }
    
    func testONNXModelConfigCustom() {
        let config = ONNXModelConfig(
            inputName: "custom_input",
            inputShape: [1, 3, 512, 512],
            inputDataType: .float16,
            framework: .onnxFloat16,
            detectionFormat: nil
        )
        
        XCTAssertEqual(config.inputName, "custom_input")
        XCTAssertEqual(config.inputShape, [1, 3, 512, 512])
        XCTAssertEqual(config.inputDataType, .float16)
        XCTAssertEqual(config.framework, .onnxFloat16)
        XCTAssertNil(config.detectionFormat)
    }
    
    func testONNXModelConfigCodable() throws {
        let config = ONNXModelConfig.yolov8
        
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(ONNXModelConfig.self, from: encoded)
        
        XCTAssertEqual(config.inputName, decoded.inputName)
        XCTAssertEqual(config.inputShape, decoded.inputShape)
        XCTAssertEqual(config.inputDataType, decoded.inputDataType)
    }
    
    // MARK: - ONNXHelper Sigmoid Tests
    
    func testSigmoid() {
        XCTAssertEqual(ONNXHelper.sigmoid(0), 0.5, accuracy: 0.001)
        XCTAssertEqual(ONNXHelper.sigmoid(1), 0.731, accuracy: 0.001)
        XCTAssertEqual(ONNXHelper.sigmoid(-1), 0.269, accuracy: 0.001)
        XCTAssertEqual(ONNXHelper.sigmoid(10), 0.99995, accuracy: 0.0001)
        XCTAssertEqual(ONNXHelper.sigmoid(-10), 0.00005, accuracy: 0.0001)
    }
    
    // MARK: - ONNXHelper IoU Tests
    
    func testComputeIoUIdenticalBoxes() {
        let box = [Float](arrayLiteral: 0, 0, 100, 100)
        let iou = ONNXHelper.computeIoU(box, box)
        XCTAssertEqual(iou, 1.0, accuracy: 0.001)
    }
    
    func testComputeIoUNoOverlap() {
        let box1: [Float] = [0, 0, 50, 50]
        let box2: [Float] = [100, 100, 150, 150]
        let iou = ONNXHelper.computeIoU(box1, box2)
        XCTAssertEqual(iou, 0.0, accuracy: 0.001)
    }
    
    func testComputeIoUPartialOverlap() {
        let box1: [Float] = [0, 0, 100, 100]
        let box2: [Float] = [50, 50, 150, 150]
        // Intersection: 50x50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU: 2500 / 17500 = 0.143
        let iou = ONNXHelper.computeIoU(box1, box2)
        XCTAssertEqual(iou, 0.143, accuracy: 0.01)
    }
    
    func testComputeIoUContained() {
        let box1: [Float] = [0, 0, 100, 100]
        let box2: [Float] = [25, 25, 75, 75]
        // Intersection: 50x50 = 2500
        // Union: 10000 + 2500 - 2500 = 10000
        // IoU: 2500 / 10000 = 0.25
        let iou = ONNXHelper.computeIoU(box1, box2)
        XCTAssertEqual(iou, 0.25, accuracy: 0.001)
    }
    
    // MARK: - ONNXHelper NMS Tests
    
    func testApplyNMSEmpty() {
        let detections: [Detection] = []
        let result = ONNXHelper.applyNMS(detections, threshold: 0.5)
        XCTAssertTrue(result.isEmpty)
    }
    
    func testApplyNMSSingleDetection() {
        let detection = Detection(box: [0, 0, 100, 100], score: 0.9, classIndex: 0, label: nil)
        let result = ONNXHelper.applyNMS([detection], threshold: 0.5)
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].score, 0.9)
    }
    
    func testApplyNMSSuppressOverlapping() {
        let detections = [
            Detection(box: [0, 0, 100, 100], score: 0.9, classIndex: 0, label: nil),
            Detection(box: [10, 10, 110, 110], score: 0.8, classIndex: 0, label: nil),  // Overlaps, same class
            Detection(box: [200, 200, 300, 300], score: 0.7, classIndex: 0, label: nil)  // No overlap
        ]
        
        let result = ONNXHelper.applyNMS(detections, threshold: 0.5)
        
        // Should keep highest confidence and non-overlapping
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].score, 0.9)
        XCTAssertEqual(result[1].score, 0.7)
    }
    
    func testApplyNMSDifferentClasses() {
        let detections = [
            Detection(box: [0, 0, 100, 100], score: 0.9, classIndex: 0, label: nil),
            Detection(box: [10, 10, 110, 110], score: 0.8, classIndex: 1, label: nil)  // Overlaps, different class
        ]
        
        let result = ONNXHelper.applyNMS(detections, threshold: 0.5)
        
        // Different classes should both be kept
        XCTAssertEqual(result.count, 2)
    }
    
    // MARK: - ONNXHelper Classification Output Tests
    
    func testParseClassificationOutput() {
        // Create logits for 5 classes
        let logits: [Float] = [1.0, 2.0, 5.0, 0.5, 0.1]
        let data = logits.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let results = ONNXHelper.parseClassificationOutput(
            data: data,
            numClasses: 5,
            topK: 3,
            applyingSoftmax: true
        )
        
        XCTAssertEqual(results.count, 3)
        
        // Class 2 (logit=5.0) should be top
        XCTAssertEqual(results[0].classIndex, 2)
        XCTAssertGreaterThan(results[0].probability, 0.9)  // Should be highest after softmax
        
        // Class 1 (logit=2.0) should be second
        XCTAssertEqual(results[1].classIndex, 1)
    }
    
    func testParseClassificationOutputWithoutSoftmax() {
        // Already probabilities that sum to 1
        let probs: [Float] = [0.1, 0.15, 0.5, 0.15, 0.1]
        let data = probs.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let results = ONNXHelper.parseClassificationOutput(
            data: data,
            numClasses: 5,
            topK: 2,
            applyingSoftmax: false
        )
        
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].classIndex, 2)
        XCTAssertEqual(results[0].probability, 0.5, accuracy: 0.001)
    }
    
    // MARK: - ONNXHelper Segmentation Output Tests
    
    func testParseSegmentationOutput() {
        // 3 classes, 2x2 image
        // Shape: [1, 3, 2, 2] = 12 values
        // NCHW layout: class0[pixel0, pixel1, pixel2, pixel3], class1[...], class2[...]
        let logits: [Float] = [
            // Class 0 scores for each pixel
            0.1, 0.5, 0.2, 0.1,
            // Class 1 scores for each pixel
            0.8, 0.3, 0.9, 0.2,
            // Class 2 scores for each pixel
            0.1, 0.2, 0.1, 0.7
        ]
        let data = logits.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let classIndices = ONNXHelper.parseSegmentationOutput(
            data: data,
            shape: [1, 3, 2, 2],
            applyArgmax: true
        )
        
        XCTAssertEqual(classIndices.count, 4)  // 2x2 pixels
        XCTAssertEqual(classIndices[0], 1)  // Pixel 0: class 1 has highest score (0.8)
        XCTAssertEqual(classIndices[1], 0)  // Pixel 1: class 0 has highest score (0.5)
        XCTAssertEqual(classIndices[2], 1)  // Pixel 2: class 1 has highest score (0.9)
        XCTAssertEqual(classIndices[3], 2)  // Pixel 3: class 2 has highest score (0.7)
    }
    
    // MARK: - ONNXHelper Detection Scaling Tests
    
    func testScaleDetectionsSimple() {
        let detections = [
            Detection(box: [64, 64, 192, 192], score: 0.9, classIndex: 0, label: nil)
        ]
        
        let scaled = ONNXHelper.scaleDetections(
            detections,
            modelSize: CGSize(width: 256, height: 256),
            originalSize: CGSize(width: 512, height: 512),
            letterboxInfo: nil
        )
        
        XCTAssertEqual(scaled.count, 1)
        // Scale factor is 2x
        XCTAssertEqual(scaled[0].box[0], 128, accuracy: 0.1)  // x1
        XCTAssertEqual(scaled[0].box[1], 128, accuracy: 0.1)  // y1
        XCTAssertEqual(scaled[0].box[2], 384, accuracy: 0.1)  // x2
        XCTAssertEqual(scaled[0].box[3], 384, accuracy: 0.1)  // y2
        XCTAssertEqual(scaled[0].score, 0.9)
    }
    
    func testScaleDetectionsWithLetterbox() {
        let letterboxInfo = LetterboxInfo(
            scale: 0.5,
            offset: CGPoint(x: 64, y: 0),
            originalSize: CGSize(width: 256, height: 128),
            letterboxedSize: CGSize(width: 256, height: 256)
        )
        
        let detections = [
            Detection(box: [64, 0, 192, 128], score: 0.9, classIndex: 0, label: nil)
        ]
        
        let scaled = ONNXHelper.scaleDetections(
            detections,
            modelSize: CGSize(width: 256, height: 256),
            originalSize: CGSize(width: 256, height: 128),
            letterboxInfo: letterboxInfo
        )
        
        XCTAssertEqual(scaled.count, 1)
        // After removing offset (64) and scaling by 1/0.5 = 2
        XCTAssertEqual(scaled[0].box[0], 0, accuracy: 0.1)    // (64-64)/0.5 = 0
        XCTAssertEqual(scaled[0].box[1], 0, accuracy: 0.1)    // (0-0)/0.5 = 0
        XCTAssertEqual(scaled[0].box[2], 256, accuracy: 0.1)  // (192-64)/0.5 = 256
        XCTAssertEqual(scaled[0].box[3], 256, accuracy: 0.1)  // (128-0)/0.5 = 256
    }
    
    // MARK: - MLFramework ONNX Variants Tests
    
    func testMLFrameworkONNXOptions() {
        // Standard ONNX (ImageNet, Float32)
        let onnxOptions = MLFramework.onnx.options
        XCTAssertEqual(onnxOptions.colorFormat, .rgb)
        XCTAssertEqual(onnxOptions.normalization, .imagenet)
        XCTAssertEqual(onnxOptions.dataLayout, .nchw)
        XCTAssertEqual(onnxOptions.outputFormat, .float32Array)
    }
    
    func testMLFrameworkONNXRawOptions() {
        let options = MLFramework.onnxRaw.options
        XCTAssertEqual(options.colorFormat, .rgb)
        XCTAssertEqual(options.normalization, .scale)
        XCTAssertEqual(options.dataLayout, .nchw)
        XCTAssertEqual(options.outputFormat, .float32Array)
    }
    
    func testMLFrameworkONNXQuantizedUInt8Options() {
        let options = MLFramework.onnxQuantizedUInt8.options
        XCTAssertEqual(options.colorFormat, .rgb)
        XCTAssertEqual(options.normalization, .raw)
        XCTAssertEqual(options.dataLayout, .nchw)
        XCTAssertEqual(options.outputFormat, .uint8Array)
    }
    
    func testMLFrameworkONNXQuantizedInt8Options() {
        let options = MLFramework.onnxQuantizedInt8.options
        XCTAssertEqual(options.colorFormat, .rgb)
        XCTAssertEqual(options.normalization, .raw)
        XCTAssertEqual(options.dataLayout, .nchw)
        XCTAssertEqual(options.outputFormat, .int32Array)
    }
    
    func testMLFrameworkONNXFloat16Options() {
        let options = MLFramework.onnxFloat16.options
        XCTAssertEqual(options.colorFormat, .rgb)
        XCTAssertEqual(options.normalization, .imagenet)
        XCTAssertEqual(options.dataLayout, .nchw)
        XCTAssertEqual(options.outputFormat, .float16Array)
    }
    
    // MARK: - Model Presets Tests
    
    func testONNXYOLOv8Preset() {
        let preset = ModelPresets.onnx_yolov8
        XCTAssertEqual(preset.colorFormat, .rgb)
        XCTAssertEqual(preset.resize?.width, 640)
        XCTAssertEqual(preset.resize?.height, 640)
        XCTAssertEqual(preset.resize?.strategy, .letterbox)
        XCTAssertEqual(preset.normalization, .scale)
        XCTAssertEqual(preset.dataLayout, .nchw)
        XCTAssertEqual(preset.outputFormat, .float32Array)
    }
    
    func testONNXResNetPreset() {
        let preset = ModelPresets.onnx_resnet
        XCTAssertEqual(preset.colorFormat, .rgb)
        XCTAssertEqual(preset.resize?.width, 224)
        XCTAssertEqual(preset.resize?.height, 224)
        XCTAssertEqual(preset.resize?.strategy, .cover)
        XCTAssertEqual(preset.normalization, .imagenet)
        XCTAssertEqual(preset.dataLayout, .nchw)
    }
    
    func testONNXQuantizedPresets() {
        let uint8Preset = ModelPresets.onnx_quantized_uint8
        XCTAssertEqual(uint8Preset.normalization, .raw)
        XCTAssertEqual(uint8Preset.outputFormat, .uint8Array)
        
        let int8Preset = ModelPresets.onnx_quantized_int8
        XCTAssertEqual(int8Preset.normalization, .raw)
        XCTAssertEqual(int8Preset.outputFormat, .int32Array)
    }
    
    func testONNXFloat16Preset() {
        let preset = ModelPresets.onnx_float16
        XCTAssertEqual(preset.normalization, .imagenet)
        XCTAssertEqual(preset.outputFormat, .float16Array)
    }
    
    func testONNXRTDETRPreset() {
        let preset = ModelPresets.onnx_rtdetr
        XCTAssertEqual(preset.resize?.width, 640)
        XCTAssertEqual(preset.resize?.height, 640)
        XCTAssertEqual(preset.resize?.strategy, .letterbox)
        XCTAssertEqual(preset.normalization, .imagenet)
    }
    
    // MARK: - YOLOv8 Preset Variants Tests
    
    func testONNXYOLOv8VariantPresets() {
        // All YOLOv8 variants should have same preprocessing
        let variants = [
            ModelPresets.onnx_yolov8n,
            ModelPresets.onnx_yolov8s,
            ModelPresets.onnx_yolov8m,
            ModelPresets.onnx_yolov8l,
            ModelPresets.onnx_yolov8x
        ]
        
        for variant in variants {
            XCTAssertEqual(variant.resize?.width, 640)
            XCTAssertEqual(variant.resize?.height, 640)
            XCTAssertEqual(variant.normalization, .scale)
            XCTAssertEqual(variant.dataLayout, .nchw)
        }
    }
    
    // MARK: - YOLOv8 Output Parsing Tests
    
    func testParseYOLOv8OutputBasic() {
        // YOLOv8 uses fixed 8400 boxes, format: [1, 84, 8400]
        // 84 = 4 bbox coords + 80 classes
        let numBoxes = 8400
        let numClasses = 80
        let valuesPerBox = 4 + numClasses  // 84
        
        // YOLOv8 format is transposed: [4+classes, num_boxes]
        var output = [Float](repeating: 0, count: valuesPerBox * numBoxes)
        
        // Box 0: cx=100, cy=100, w=50, h=50, class 0 with high score
        output[0 * numBoxes + 0] = 100  // cx
        output[1 * numBoxes + 0] = 100  // cy
        output[2 * numBoxes + 0] = 50   // w
        output[3 * numBoxes + 0] = 50   // h
        output[4 * numBoxes + 0] = 0.9  // class 0 score
        
        // Box 100: cx=200, cy=200, w=30, h=30, class 1 with medium score
        output[0 * numBoxes + 100] = 200  // cx
        output[1 * numBoxes + 100] = 200  // cy
        output[2 * numBoxes + 100] = 30   // w
        output[3 * numBoxes + 100] = 30   // h
        output[5 * numBoxes + 100] = 0.5  // class 1 score
        
        let data = output.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let detections = ONNXHelper.parseYOLOv8Output(
            data: data,
            numClasses: numClasses,
            confidenceThreshold: 0.25,
            nmsThreshold: 0.45
        )
        
        XCTAssertEqual(detections.count, 2)
        
        // First detection (higher confidence)
        XCTAssertEqual(detections[0].classIndex, 0)
        XCTAssertEqual(detections[0].score, 0.9, accuracy: 0.01)
        
        // Verify box conversion from cxcywh to xyxy
        // cx=100, cy=100, w=50, h=50 -> x1=75, y1=75, x2=125, y2=125
        XCTAssertEqual(detections[0].box[0], 75, accuracy: 0.1)
        XCTAssertEqual(detections[0].box[1], 75, accuracy: 0.1)
        XCTAssertEqual(detections[0].box[2], 125, accuracy: 0.1)
        XCTAssertEqual(detections[0].box[3], 125, accuracy: 0.1)
    }
    
    func testParseYOLOv8OutputFiltersLowConfidence() {
        let numBoxes = 8400
        let numClasses = 80
        let valuesPerBox = 4 + numClasses
        
        var output = [Float](repeating: 0, count: valuesPerBox * numBoxes)
        
        // Box 0 with score below threshold
        output[0 * numBoxes + 0] = 100
        output[1 * numBoxes + 0] = 100
        output[2 * numBoxes + 0] = 50
        output[3 * numBoxes + 0] = 50
        output[4 * numBoxes + 0] = 0.1  // Below 0.25 threshold
        
        // Box 1 with score above threshold
        output[0 * numBoxes + 1] = 200
        output[1 * numBoxes + 1] = 200
        output[2 * numBoxes + 1] = 30
        output[3 * numBoxes + 1] = 30
        output[4 * numBoxes + 1] = 0.8
        
        let data = output.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let detections = ONNXHelper.parseYOLOv8Output(
            data: data,
            numClasses: numClasses,
            confidenceThreshold: 0.25,
            nmsThreshold: 0.45
        )
        
        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections[0].score, 0.8, accuracy: 0.01)
    }
    
    func testParseYOLOv8OutputNMSSuppression() {
        let numBoxes = 8400
        let numClasses = 80
        let valuesPerBox = 4 + numClasses
        
        var output = [Float](repeating: 0, count: valuesPerBox * numBoxes)
        
        // Box 0: high confidence
        output[0 * numBoxes + 0] = 100  // cx
        output[1 * numBoxes + 0] = 100  // cy
        output[2 * numBoxes + 0] = 50   // w
        output[3 * numBoxes + 0] = 50   // h
        output[4 * numBoxes + 0] = 0.9  // class 0
        
        // Box 1: overlapping, lower confidence, same class (should be suppressed)
        output[0 * numBoxes + 1] = 110  // cx (overlaps with box 0)
        output[1 * numBoxes + 1] = 110  // cy
        output[2 * numBoxes + 1] = 50   // w
        output[3 * numBoxes + 1] = 50   // h
        output[4 * numBoxes + 1] = 0.7  // class 0
        
        // Box 2: non-overlapping (should be kept)
        output[0 * numBoxes + 2] = 500  // cx
        output[1 * numBoxes + 2] = 500  // cy
        output[2 * numBoxes + 2] = 50   // w
        output[3 * numBoxes + 2] = 50   // h
        output[4 * numBoxes + 2] = 0.6  // class 0
        
        let data = output.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let detections = ONNXHelper.parseYOLOv8Output(
            data: data,
            numClasses: numClasses,
            confidenceThreshold: 0.25,
            nmsThreshold: 0.45
        )
        
        // Box 1 should be suppressed by NMS
        XCTAssertEqual(detections.count, 2)
        XCTAssertEqual(detections[0].score, 0.9, accuracy: 0.01)
        XCTAssertEqual(detections[1].score, 0.6, accuracy: 0.01)
    }
    
    // MARK: - RT-DETR Output Parsing Tests
    
    func testParseRTDETROutputBasic() {
        // RT-DETR output shape: [1, 300, 4+num_classes] - always 300 queries
        let numQueries = 300  // RT-DETR always uses 300 queries
        let numClasses = 80
        let valuesPerQuery = 4 + numClasses
        
        var output = [Float](repeating: 0, count: numQueries * valuesPerQuery)
        
        // Query 0: high confidence detection
        output[0 * valuesPerQuery + 0] = 0.5  // cx (normalized)
        output[0 * valuesPerQuery + 1] = 0.5  // cy (normalized)
        output[0 * valuesPerQuery + 2] = 0.2  // w (normalized)
        output[0 * valuesPerQuery + 3] = 0.2  // h (normalized)
        output[0 * valuesPerQuery + 4] = 0.9  // class 0 score
        
        // Query 1: low confidence (should be filtered)
        output[1 * valuesPerQuery + 0] = 0.3
        output[1 * valuesPerQuery + 1] = 0.3
        output[1 * valuesPerQuery + 2] = 0.1
        output[1 * valuesPerQuery + 3] = 0.1
        output[1 * valuesPerQuery + 4] = 0.3  // Below 0.5 threshold
        
        // All other queries have 0 scores and will be filtered
        
        let data = output.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let detections = ONNXHelper.parseRTDETROutput(
            data: data,
            numClasses: numClasses,
            confidenceThreshold: 0.5
        )
        
        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections[0].classIndex, 0)
        XCTAssertEqual(detections[0].score, 0.9, accuracy: 0.01)
        
        // Verify normalized box conversion
        // cx=0.5, cy=0.5, w=0.2, h=0.2 -> x1=0.4, y1=0.4, x2=0.6, y2=0.6
        XCTAssertEqual(detections[0].box[0], 0.4, accuracy: 0.01)
        XCTAssertEqual(detections[0].box[1], 0.4, accuracy: 0.01)
        XCTAssertEqual(detections[0].box[2], 0.6, accuracy: 0.01)
        XCTAssertEqual(detections[0].box[3], 0.6, accuracy: 0.01)
    }
    
    // MARK: - Generic Detection Output Tests
    
    func testParseDetectionOutputWithYOLOv8Format() {
        // Test that format dispatch works with YOLOv8
        let numBoxes = 8400
        let numClasses = 80
        
        // YOLOv8 format
        var yolov8Output = [Float](repeating: 0, count: (4 + numClasses) * numBoxes)
        yolov8Output[0 * numBoxes + 0] = 100  // cx
        yolov8Output[1 * numBoxes + 0] = 100  // cy
        yolov8Output[2 * numBoxes + 0] = 50   // w
        yolov8Output[3 * numBoxes + 0] = 50   // h
        yolov8Output[4 * numBoxes + 0] = 0.9  // class 0
        
        let data = yolov8Output.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let detections = ONNXHelper.parseDetectionOutput(
            data: data,
            shape: [1, 84, 8400],
            format: .yoloV8,
            numClasses: numClasses,
            confidenceThreshold: 0.25,
            nmsThreshold: 0.45
        )
        
        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections[0].score, 0.9, accuracy: 0.01)
    }
    
    func testParseDetectionOutputWithRTDETRFormat() {
        // Test that format dispatch works with RT-DETR
        let numQueries = 300
        let numClasses = 80
        
        var rtdetrOutput = [Float](repeating: 0, count: numQueries * (4 + numClasses))
        rtdetrOutput[0] = 0.5   // cx
        rtdetrOutput[1] = 0.5   // cy
        rtdetrOutput[2] = 0.2   // w
        rtdetrOutput[3] = 0.2   // h
        rtdetrOutput[4] = 0.9   // class 0
        
        let data = rtdetrOutput.withUnsafeBufferPointer { Data(buffer: $0) }
        
        let detections = ONNXHelper.parseDetectionOutput(
            data: data,
            shape: [1, 300, 84],
            format: .rtdetr,
            numClasses: numClasses,
            confidenceThreshold: 0.5,
            nmsThreshold: 0.45
        )
        
        XCTAssertEqual(detections.count, 1)
    }
}
