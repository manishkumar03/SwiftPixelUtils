import XCTest
import AVFoundation
@testable import SwiftPixelUtils

/// Tests for ``VideoFrameExtractor`` - Video frame extraction utilities.
///
/// ## Topics
///
/// ### VideoSource Tests
/// - URL and file path sources
/// - Source validation
///
/// ### Extraction Options Tests
/// - Interval-based extraction (every N seconds)
/// - Timestamp-based extraction (specific times)
/// - Maximum frame count limits
///
/// ### Output Format Tests
/// - Pixel data for ML inference
/// - JPEG/PNG image data
/// - Base64 encoded output
///
/// ### Metadata Tests
/// - Video duration, dimensions, frame rate
/// - Codec information
///
/// ### Error Handling Tests
/// - Invalid URLs, missing files
/// - Unsupported formats
@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
final class VideoFrameExtractorTests: XCTestCase {
    
    // MARK: - VideoSource Tests
    
    func testVideoSourceURL() {
        let url = URL(string: "https://example.com/video.mp4")!
        let source = VideoSource.url(url)
        
        if case .url(let sourceURL) = source {
            XCTAssertEqual(sourceURL, url)
        } else {
            XCTFail("Should be URL source")
        }
    }
    
    func testVideoSourceFile() {
        let url = URL(fileURLWithPath: "/path/to/video.mp4")
        let source = VideoSource.file(url)
        
        if case .file(let sourceURL) = source {
            XCTAssertEqual(sourceURL, url)
        } else {
            XCTFail("Should be file source")
        }
    }
    
    // MARK: - VideoFrameExtractionOptions Tests
    
    func testVideoFrameExtractionOptionsDefaults() {
        let options = VideoFrameExtractionOptions()
        
        XCTAssertNil(options.timestamps)
        XCTAssertNil(options.interval)
        XCTAssertNil(options.maxFrames)
        XCTAssertNil(options.resize)
        XCTAssertEqual(options.outputFormat, .pixelData)
        XCTAssertEqual(options.jpegQuality, 80)
        XCTAssertEqual(options.dataLayout, .hwc)
    }
    
    func testVideoFrameExtractionOptionsCustom() {
        let options = VideoFrameExtractionOptions(
            timestamps: [0.0, 1.0, 2.0],
            interval: 0.5,
            maxFrames: 10,
            resize: (width: 224, height: 224),
            outputFormat: .base64,
            jpegQuality: 90,
            normalization: .imagenet,
            dataLayout: .nchw
        )
        
        XCTAssertEqual(options.timestamps, [0.0, 1.0, 2.0])
        XCTAssertEqual(options.interval, 0.5)
        XCTAssertEqual(options.maxFrames, 10)
        XCTAssertEqual(options.resize?.width, 224)
        XCTAssertEqual(options.resize?.height, 224)
        XCTAssertEqual(options.outputFormat, .base64)
        XCTAssertEqual(options.jpegQuality, 90)
        XCTAssertEqual(options.dataLayout, .nchw)
    }
    
    func testVideoFrameExtractionOptionsWithTimestamps() {
        let timestamps = [0.5, 1.0, 1.5, 2.0]
        let options = VideoFrameExtractionOptions(timestamps: timestamps)
        
        XCTAssertEqual(options.timestamps, timestamps)
    }
    
    func testVideoFrameExtractionOptionsWithInterval() {
        let options = VideoFrameExtractionOptions(interval: 0.033)  // 30 FPS
        
        XCTAssertNotNil(options.interval)
        XCTAssertEqual(options.interval!, 0.033, accuracy: 0.001)
    }
    
    // MARK: - VideoOutputFormat Tests
    
    func testVideoOutputFormatRawValues() {
        XCTAssertEqual(VideoOutputFormat.pixelData.rawValue, "pixelData")
        XCTAssertEqual(VideoOutputFormat.base64.rawValue, "base64")
    }
    
    func testVideoOutputFormatFromRawValue() {
        XCTAssertEqual(VideoOutputFormat(rawValue: "pixelData"), .pixelData)
        XCTAssertEqual(VideoOutputFormat(rawValue: "base64"), .base64)
        XCTAssertNil(VideoOutputFormat(rawValue: "invalid"))
    }
    
    // MARK: - VideoMetadata Tests
    
    func testVideoMetadataInit() {
        let metadata = VideoMetadata(
            duration: 120.5,
            frameRate: 30.0,
            width: 1920,
            height: 1080,
            trackCount: 2
        )
        
        XCTAssertEqual(metadata.duration, 120.5)
        XCTAssertEqual(metadata.frameRate, 30.0)
        XCTAssertEqual(metadata.width, 1920)
        XCTAssertEqual(metadata.height, 1080)
        XCTAssertEqual(metadata.trackCount, 2)
    }
    
    func testVideoMetadataVariousFormats() {
        // 4K video
        let metadata4K = VideoMetadata(
            duration: 60.0,
            frameRate: 60.0,
            width: 3840,
            height: 2160,
            trackCount: 1
        )
        
        XCTAssertEqual(metadata4K.width, 3840)
        XCTAssertEqual(metadata4K.height, 2160)
        
        // Portrait video
        let metadataPortrait = VideoMetadata(
            duration: 30.0,
            frameRate: 30.0,
            width: 1080,
            height: 1920,
            trackCount: 1
        )
        
        XCTAssertEqual(metadataPortrait.width, 1080)
        XCTAssertEqual(metadataPortrait.height, 1920)
    }
    
    // MARK: - ExtractedFrame Tests
    
    func testExtractedFrameWithPixelData() {
        let pixelData = PixelDataResult(
            data: [Float](repeating: 0.5, count: 100),
            width: 10,
            height: 10,
            channels: 1,
            colorFormat: .grayscale,
            dataLayout: .hwc,
            shape: [10, 10, 1],
            processingTimeMs: 0
        )
        
        let frame = ExtractedFrame(
            timestamp: 1.5,
            frameIndex: 45,
            data: pixelData,
            width: 1920,
            height: 1080,
            error: nil
        )
        
        XCTAssertEqual(frame.timestamp, 1.5)
        XCTAssertEqual(frame.frameIndex, 45)
        XCTAssertEqual(frame.width, 1920)
        XCTAssertEqual(frame.height, 1080)
        XCTAssertNil(frame.error)
    }
    
    func testExtractedFrameWithBase64() {
        let frame = ExtractedFrame(
            timestamp: 2.0,
            frameIndex: 60,
            data: "base64EncodedJPEG...",
            width: 640,
            height: 480,
            error: nil
        )
        
        XCTAssertEqual(frame.timestamp, 2.0)
        XCTAssertEqual(frame.frameIndex, 60)
        XCTAssertTrue(frame.data is String)
    }
    
    func testExtractedFrameWithError() {
        let frame = ExtractedFrame(
            timestamp: 5.0,
            frameIndex: 150,
            data: "",
            width: 0,
            height: 0,
            error: "Frame extraction failed"
        )
        
        XCTAssertEqual(frame.error, "Frame extraction failed")
        XCTAssertEqual(frame.width, 0)
        XCTAssertEqual(frame.height, 0)
    }
    
    // MARK: - VideoExtractionResult Tests
    
    func testVideoExtractionResultInit() {
        let metadata = VideoMetadata(
            duration: 10.0,
            frameRate: 30.0,
            width: 1920,
            height: 1080,
            trackCount: 1
        )
        
        let frames = [
            ExtractedFrame(
                timestamp: 0.0,
                frameIndex: 0,
                data: "frame0",
                width: 1920,
                height: 1080,
                error: nil
            ),
            ExtractedFrame(
                timestamp: 1.0,
                frameIndex: 1,
                data: "frame1",
                width: 1920,
                height: 1080,
                error: nil
            )
        ]
        
        let result = VideoExtractionResult(
            frames: frames,
            metadata: metadata,
            processingTimeMs: 250.5,
            failedFrames: 0
        )
        
        XCTAssertEqual(result.frames.count, 2)
        XCTAssertEqual(result.metadata.duration, 10.0)
        XCTAssertEqual(result.processingTimeMs, 250.5)
        XCTAssertEqual(result.failedFrames, 0)
    }
    
    func testVideoExtractionResultWithFailures() {
        let metadata = VideoMetadata(
            duration: 10.0,
            frameRate: 30.0,
            width: 1920,
            height: 1080,
            trackCount: 1
        )
        
        let frames = [
            ExtractedFrame(timestamp: 0.0, frameIndex: 0, data: "ok", width: 1920, height: 1080, error: nil),
            ExtractedFrame(timestamp: 1.0, frameIndex: 1, data: "", width: 0, height: 0, error: "Failed"),
            ExtractedFrame(timestamp: 2.0, frameIndex: 2, data: "", width: 0, height: 0, error: "Failed")
        ]
        
        let result = VideoExtractionResult(
            frames: frames,
            metadata: metadata,
            processingTimeMs: 100.0,
            failedFrames: 2
        )
        
        XCTAssertEqual(result.frames.count, 3)
        XCTAssertEqual(result.failedFrames, 2)
    }
    
    // MARK: - extractFrames Error Tests
    
    func testExtractFramesInvalidURL() async {
        let source = VideoSource.url(URL(string: "https://invalid.example.com/nonexistent.mp4")!)
        
        do {
            _ = try await VideoFrameExtractor.extractFrames(from: source)
            XCTFail("Should throw for invalid URL")
        } catch {
            // Expected - file doesn't exist
        }
    }
    
    func testExtractFramesNonexistentFile() async {
        let source = VideoSource.file(URL(fileURLWithPath: "/nonexistent/path/video.mp4"))
        
        do {
            _ = try await VideoFrameExtractor.extractFrames(from: source)
            XCTFail("Should throw for nonexistent file")
        } catch {
            // Expected
        }
    }
    
    // MARK: - getVideoMetadata Error Tests
    
    func testGetVideoMetadataInvalidSource() async {
        let source = VideoSource.file(URL(fileURLWithPath: "/invalid/video.mp4"))
        
        do {
            _ = try await VideoFrameExtractor.getVideoMetadata(from: source)
            XCTFail("Should throw for invalid source")
        } catch {
            // Expected
        }
    }
    
    // MARK: - extractFrame Error Tests
    
    func testExtractFrameInvalidSource() async {
        let source = VideoSource.file(URL(fileURLWithPath: "/invalid/video.mp4"))
        
        do {
            _ = try await VideoFrameExtractor.extractFrame(from: source, at: 1.0)
            XCTFail("Should throw for invalid source")
        } catch {
            // Expected
        }
    }
    
    // MARK: - Options Combinations
    
    func testOptionsWithTimestampsAndMaxFrames() {
        let options = VideoFrameExtractionOptions(
            timestamps: [0.0, 1.0, 2.0, 3.0, 4.0],
            maxFrames: 3  // maxFrames should be ignored when timestamps are provided
        )
        
        XCTAssertEqual(options.timestamps?.count, 5)
        XCTAssertEqual(options.maxFrames, 3)
    }
    
    func testOptionsWithIntervalAndMaxFrames() {
        let options = VideoFrameExtractionOptions(
            interval: 0.1,
            maxFrames: 100
        )
        
        XCTAssertEqual(options.interval, 0.1)
        XCTAssertEqual(options.maxFrames, 100)
    }
    
    // MARK: - Normalization Presets
    
    func testNormalizationOptions() {
        let scaleOptions = VideoFrameExtractionOptions(normalization: .scale)
        let imagenetOptions = VideoFrameExtractionOptions(normalization: .imagenet)
        let rawOptions = VideoFrameExtractionOptions(normalization: Normalization(preset: .raw))
        
        XCTAssertEqual(scaleOptions.normalization.preset, .scale)
        XCTAssertEqual(imagenetOptions.normalization.preset, .imagenet)
        XCTAssertEqual(rawOptions.normalization.preset, .raw)
    }
    
    // MARK: - Data Layout Options
    
    func testDataLayoutOptions() {
        let hwcOptions = VideoFrameExtractionOptions(dataLayout: .hwc)
        let chwOptions = VideoFrameExtractionOptions(dataLayout: .chw)
        let nchwOptions = VideoFrameExtractionOptions(dataLayout: .nchw)
        let nhwcOptions = VideoFrameExtractionOptions(dataLayout: .nhwc)
        
        XCTAssertEqual(hwcOptions.dataLayout, .hwc)
        XCTAssertEqual(chwOptions.dataLayout, .chw)
        XCTAssertEqual(nchwOptions.dataLayout, .nchw)
        XCTAssertEqual(nhwcOptions.dataLayout, .nhwc)
    }
    
    // MARK: - JPEG Quality Tests
    
    func testJPEGQualityValues() {
        let lowQuality = VideoFrameExtractionOptions(jpegQuality: 10)
        let mediumQuality = VideoFrameExtractionOptions(jpegQuality: 50)
        let highQuality = VideoFrameExtractionOptions(jpegQuality: 100)
        
        XCTAssertEqual(lowQuality.jpegQuality, 10)
        XCTAssertEqual(mediumQuality.jpegQuality, 50)
        XCTAssertEqual(highQuality.jpegQuality, 100)
    }
    
    // MARK: - Resize Options
    
    func testResizeOptions() {
        let squareResize = VideoFrameExtractionOptions(resize: (width: 224, height: 224))
        let wideResize = VideoFrameExtractionOptions(resize: (width: 640, height: 480))
        let tallResize = VideoFrameExtractionOptions(resize: (width: 480, height: 640))
        
        XCTAssertNotNil(squareResize.resize)
        XCTAssertEqual(squareResize.resize!.width, 224)
        XCTAssertEqual(squareResize.resize!.height, 224)
        XCTAssertNotNil(wideResize.resize)
        XCTAssertEqual(wideResize.resize!.width, 640)
        XCTAssertEqual(wideResize.resize!.height, 480)
        XCTAssertNotNil(tallResize.resize)
        XCTAssertEqual(tallResize.resize!.width, 480)
        XCTAssertEqual(tallResize.resize!.height, 640)
    }
    
    // MARK: - Frame Index Consistency
    
    func testExtractedFrameIndexConsistency() {
        var frames: [ExtractedFrame] = []
        
        for i in 0..<10 {
            frames.append(ExtractedFrame(
                timestamp: Double(i) * 0.5,
                frameIndex: i,
                data: "frame\(i)",
                width: 1920,
                height: 1080,
                error: nil
            ))
        }
        
        for (index, frame) in frames.enumerated() {
            XCTAssertTrue(frame.frameIndex == index)
            XCTAssertEqual(frame.timestamp, Double(index) * 0.5, accuracy: 0.001)
        }
    }
}
