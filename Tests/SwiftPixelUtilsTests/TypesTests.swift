import XCTest
@testable import SwiftPixelUtils

/// Tests for core type definitions in SwiftPixelUtils.
///
/// ## Topics
///
/// ### ColorFormat Tests
/// - Channel count for RGB, RGBA, BGR, grayscale, etc.
/// - Format conversion compatibility
///
/// ### Normalization Tests
/// - Preset values (ImageNet, TensorFlow, scale, raw)
/// - Custom mean/std configuration
/// - Codable conformance
///
/// ### DataLayout Tests
/// - HWC, CHW, NHWC, NCHW layouts
/// - Codable serialization
///
/// ### PixelDataResult Tests
/// - Shape computation for different layouts
/// - Data integrity validation
///
/// ### Options Tests
/// - ResizeOptions, QuantizationOptions
/// - Codable round-trip tests
final class TypesTests: XCTestCase {
    
    // MARK: - ColorFormat Tests
    
    func testColorFormatChannelCount() {
        XCTAssertEqual(ColorFormat.rgb.channelCount, 3)
        XCTAssertEqual(ColorFormat.rgba.channelCount, 4)
        XCTAssertEqual(ColorFormat.bgr.channelCount, 3)
        XCTAssertEqual(ColorFormat.bgra.channelCount, 4)
        XCTAssertEqual(ColorFormat.grayscale.channelCount, 1)
        XCTAssertEqual(ColorFormat.hsv.channelCount, 3)
        XCTAssertEqual(ColorFormat.hsl.channelCount, 3)
        XCTAssertEqual(ColorFormat.lab.channelCount, 3)
        XCTAssertEqual(ColorFormat.yuv.channelCount, 3)
        XCTAssertEqual(ColorFormat.ycbcr.channelCount, 3)
    }
    
    func testColorFormatCodable() throws {
        let format = ColorFormat.rgb
        let encoded = try JSONEncoder().encode(format)
        let decoded = try JSONDecoder().decode(ColorFormat.self, from: encoded)
        XCTAssertEqual(format, decoded)
    }
    
    func testAllColorFormats() {
        // Ensure all formats are accounted for
        let allFormats: [ColorFormat] = [.rgb, .rgba, .bgr, .bgra, .grayscale, .hsv, .hsl, .lab, .yuv, .ycbcr]
        XCTAssertEqual(allFormats.count, 10)
        
        // All should have positive channel counts
        for format in allFormats {
            XCTAssertGreaterThan(format.channelCount, 0)
            XCTAssertLessThanOrEqual(format.channelCount, 4)
        }
    }
    
    // MARK: - ResizeStrategy Tests
    
    func testResizeStrategyCodable() throws {
        let strategies: [ResizeStrategy] = [.cover, .contain, .stretch, .letterbox]
        
        for strategy in strategies {
            let encoded = try JSONEncoder().encode(strategy)
            let decoded = try JSONDecoder().decode(ResizeStrategy.self, from: encoded)
            XCTAssertEqual(strategy, decoded)
        }
    }
    
    // MARK: - ResizeOptions Tests
    
    func testResizeOptionsInit() {
        let options = ResizeOptions(width: 640, height: 480, strategy: .letterbox)
        XCTAssertEqual(options.width, 640)
        XCTAssertEqual(options.height, 480)
        XCTAssertEqual(options.strategy, .letterbox)
        XCTAssertNil(options.padColor)
        XCTAssertNotNil(options.letterboxColor)
    }
    
    func testResizeOptionsWithPadColor() {
        let options = ResizeOptions(
            width: 224,
            height: 224,
            strategy: .contain,
            padColor: [0, 0, 0],
            letterboxColor: nil
        )
        XCTAssertNotNil(options.padColor)
        XCTAssertEqual(options.padColor, [0, 0, 0])
    }
    
    func testResizeOptionsCodable() throws {
        let options = ResizeOptions(width: 640, height: 640, strategy: .letterbox)
        let encoded = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(ResizeOptions.self, from: encoded)
        
        XCTAssertEqual(options.width, decoded.width)
        XCTAssertEqual(options.height, decoded.height)
        XCTAssertEqual(options.strategy, decoded.strategy)
    }
    
    // MARK: - Normalization Tests
    
    func testNormalizationPresets() {
        // Scale preset
        let scale = Normalization.scale
        XCTAssertEqual(scale.preset, .scale)
        XCTAssertNil(scale.mean)
        XCTAssertNil(scale.std)
        
        // ImageNet preset
        let imagenet = Normalization.imagenet
        XCTAssertEqual(imagenet.preset, .imagenet)
        XCTAssertNotNil(imagenet.mean)
        XCTAssertNotNil(imagenet.std)
        XCTAssertEqual(imagenet.mean?.count, 3)
        XCTAssertEqual(imagenet.std?.count, 3)
        
        // Verify ImageNet values
        XCTAssertEqual(Double(imagenet.mean?[0] ?? 0), 0.485, accuracy: 0.001)
        XCTAssertEqual(Double(imagenet.mean?[1] ?? 0), 0.456, accuracy: 0.001)
        XCTAssertEqual(Double(imagenet.mean?[2] ?? 0), 0.406, accuracy: 0.001)
        XCTAssertEqual(Double(imagenet.std?[0] ?? 0), 0.229, accuracy: 0.001)
        XCTAssertEqual(Double(imagenet.std?[1] ?? 0), 0.224, accuracy: 0.001)
        XCTAssertEqual(Double(imagenet.std?[2] ?? 0), 0.225, accuracy: 0.001)
        
        // TensorFlow preset
        let tensorflow = Normalization.tensorflow
        XCTAssertEqual(tensorflow.preset, .tensorflow)
        
        // Raw preset
        let raw = Normalization.raw
        XCTAssertEqual(raw.preset, .raw)
    }
    
    func testCustomNormalization() {
        let custom = Normalization(
            preset: .custom,
            mean: [0.5, 0.5, 0.5],
            std: [0.25, 0.25, 0.25]
        )
        XCTAssertEqual(custom.preset, .custom)
        XCTAssertEqual(custom.mean, [0.5, 0.5, 0.5])
        XCTAssertEqual(custom.std, [0.25, 0.25, 0.25])
    }
    
    func testNormalizationEquatable() {
        let n1 = Normalization.imagenet
        let n2 = Normalization(preset: .imagenet, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
        XCTAssertEqual(n1, n2)
        
        let n3 = Normalization.scale
        XCTAssertNotEqual(n1, n3)
    }
    
    func testNormalizationCodable() throws {
        let normalizations: [Normalization] = [.scale, .imagenet, .tensorflow, .raw]
        
        for norm in normalizations {
            let encoded = try JSONEncoder().encode(norm)
            let decoded = try JSONDecoder().decode(Normalization.self, from: encoded)
            XCTAssertEqual(norm, decoded)
        }
    }
    
    // MARK: - DataLayout Tests
    
    func testDataLayoutValues() {
        // Verify all layouts exist
        let layouts: [DataLayout] = [.hwc, .chw, .nhwc, .nchw]
        XCTAssertEqual(layouts.count, 4)
    }
    
    func testDataLayoutCodable() throws {
        let layouts: [DataLayout] = [.hwc, .chw, .nhwc, .nchw]
        
        for layout in layouts {
            let encoded = try JSONEncoder().encode(layout)
            let decoded = try JSONDecoder().decode(DataLayout.self, from: encoded)
            XCTAssertEqual(layout, decoded)
        }
    }
    
    // MARK: - BoxFormat Tests
    
    func testBoxFormatValues() {
        let formats: [BoxFormat] = [.xyxy, .xywh, .cxcywh]
        XCTAssertEqual(formats.count, 3)
    }
    
    func testBoxFormatEquality() {
        XCTAssertEqual(BoxFormat.xyxy, BoxFormat.xyxy)
        XCTAssertNotEqual(BoxFormat.xyxy, BoxFormat.xywh)
        XCTAssertNotEqual(BoxFormat.xywh, BoxFormat.cxcywh)
    }
    
    // MARK: - Detection Tests
    
    func testDetectionInit() {
        let detection = Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0)
        XCTAssertEqual(detection.box, [100, 100, 200, 200])
        XCTAssertEqual(detection.score, 0.9)
        XCTAssertEqual(detection.classIndex, 0)
    }
    
    func testDetectionWithLabel() {
        let detection = Detection(box: [100, 100, 200, 200], score: 0.85, classIndex: 15, label: "person")
        XCTAssertEqual(detection.label, "person")
    }
    
    // MARK: - PixelDataResult Tests
    
    func testPixelDataResultShape() {
        let result = PixelDataResult(
            data: [Float](repeating: 0, count: 224 * 224 * 3),
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: .rgb,
            dataLayout: .hwc,
            shape: [224, 224, 3],
            processingTimeMs: 0.0
        )
        
        XCTAssertEqual(result.shape, [224, 224, 3])
        XCTAssertEqual(result.data.count, 224 * 224 * 3)
    }
    
    func testPixelDataResultNCHWShape() {
        let result = PixelDataResult(
            data: [Float](repeating: 0, count: 1 * 3 * 224 * 224),
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: .rgb,
            dataLayout: .nchw,
            shape: [1, 3, 224, 224],
            processingTimeMs: 0.0
        )
        
        XCTAssertEqual(result.shape, [1, 3, 224, 224])
    }
    
    // MARK: - QuantizationOptions Tests
    
    func testQuantizationOptionsPerTensor() {
        let options = QuantizationOptions(
            mode: .perTensor,
            dtype: .uint8,
            scale: [0.00784314],
            zeroPoint: [128]
        )
        
        XCTAssertEqual(options.mode, .perTensor)
        XCTAssertEqual(options.dtype, .uint8)
        XCTAssertEqual(options.scale, [0.00784314])
        XCTAssertEqual(options.zeroPoint, [128])
    }
    
    func testQuantizationOptionsPerChannel() {
        let options = QuantizationOptions(
            mode: .perChannel,
            dtype: .int8,
            scale: [0.1, 0.2, 0.3],
            zeroPoint: [0, 0, 0]
        )
        
        XCTAssertEqual(options.mode, .perChannel)
        XCTAssertEqual(options.scale.count, 3)
        XCTAssertEqual(options.zeroPoint.count, 3)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyResizeOptions() {
        // Minimum valid resize options
        let options = ResizeOptions(width: 1, height: 1, strategy: .stretch)
        XCTAssertEqual(options.width, 1)
        XCTAssertEqual(options.height, 1)
    }
    
    func testLargeResizeOptions() {
        let options = ResizeOptions(width: 4096, height: 4096, strategy: .cover)
        XCTAssertEqual(options.width, 4096)
        XCTAssertEqual(options.height, 4096)
    }
    
    func testSingleChannelNormalization() {
        let grayscaleNorm = Normalization(
            preset: .custom,
            mean: [0.5],
            std: [0.5]
        )
        XCTAssertEqual(grayscaleNorm.mean?.count, 1)
        XCTAssertEqual(grayscaleNorm.std?.count, 1)
    }
}
