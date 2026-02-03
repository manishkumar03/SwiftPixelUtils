//
//  VideoFrameExtractor.swift
//  SwiftPixelUtils
//
//  Extract frames from videos for temporal ML models
//

import Foundation
import CoreGraphics
import AVFoundation

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Video Types

/// Video source specification.
public enum VideoSource {
    case url(URL)
    case file(URL)
}

/// Options for video frame extraction.
public struct VideoFrameExtractionOptions {
    /// Timestamps (in seconds) to extract frames at.
    public var timestamps: [Double]?
    
    /// Extract frames at regular intervals (in seconds).
    public var interval: Double?
    
    /// Maximum number of frames to extract.
    public var maxFrames: Int?
    
    /// Target frame size (nil for original size).
    public var resize: (width: Int, height: Int)?
    
    /// Output format for frame data.
    public var outputFormat: VideoOutputFormat = .pixelData
    
    /// JPEG quality (0-100) if outputFormat is base64.
    public var jpegQuality: Int = 80
    
    /// Normalization to apply.
    public var normalization: Normalization = .scale
    
    /// Data layout for output.
    public var dataLayout: DataLayout = .hwc
    
    public init(
        timestamps: [Double]? = nil,
        interval: Double? = nil,
        maxFrames: Int? = nil,
        resize: (width: Int, height: Int)? = nil,
        outputFormat: VideoOutputFormat = .pixelData,
        jpegQuality: Int = 80,
        normalization: Normalization = .scale,
        dataLayout: DataLayout = .hwc
    ) {
        self.timestamps = timestamps
        self.interval = interval
        self.maxFrames = maxFrames
        self.resize = resize
        self.outputFormat = outputFormat
        self.jpegQuality = jpegQuality
        self.normalization = normalization
        self.dataLayout = dataLayout
    }
}

/// Output format for video frames.
public enum VideoOutputFormat: String {
    case pixelData = "pixelData"
    case base64 = "base64"
}

/// Video metadata.
public struct VideoMetadata {
    /// Duration in seconds.
    public let duration: Double
    
    /// Frame rate (frames per second).
    public let frameRate: Float
    
    /// Natural size of the video.
    public let width: Int
    public let height: Int
    
    /// Number of video tracks.
    public let trackCount: Int
}

/// Extracted frame data.
public struct ExtractedFrame {
    /// Frame timestamp in seconds.
    public let timestamp: Double
    
    /// Frame index.
    public let frameIndex: Int
    
    /// Frame data (either PixelDataResult or base64 string depending on outputFormat).
    public let data: Any
    
    /// Width of the frame.
    public let width: Int
    
    /// Height of the frame.
    public let height: Int
    
    /// Error if frame extraction failed.
    public let error: String?
}

/// Result of video frame extraction.
public struct VideoExtractionResult {
    /// Extracted frames.
    public let frames: [ExtractedFrame]
    
    /// Video metadata.
    public let metadata: VideoMetadata
    
    /// Total processing time in milliseconds.
    public let processingTimeMs: Double
    
    /// Number of failed frame extractions.
    public let failedFrames: Int
}

// MARK: - Video Frame Extractor

/// Extract frames from videos for temporal ML models.
///
/// Supports various extraction modes:
/// - Extract frames at specific timestamps
/// - Extract frames at regular intervals
/// - Extract a maximum number of evenly-spaced frames
///
/// ## Example
///
/// ```swift
/// // Extract 10 evenly-spaced frames
/// let result = try await VideoFrameExtractor.extractFrames(
///     from: .url(videoURL),
///     options: VideoFrameExtractionOptions(
///         maxFrames: 10,
///         resize: (width: 224, height: 224),
///         outputFormat: .pixelData
///     )
/// )
///
/// for frame in result.frames {
///     if let pixelData = frame.data as? PixelDataResult {
///         // Process frame...
///     }
/// }
/// ```
@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
public enum VideoFrameExtractor {
    
    // MARK: - Main Extraction
    
    /// Extract frames from a video.
    ///
    /// - Parameters:
    ///   - source: Video source
    ///   - options: Extraction options
    /// - Returns: Extraction result with frames and metadata
    /// - Throws: `PixelUtilsError` if extraction fails
    public static func extractFrames(
        from source: VideoSource,
        options: VideoFrameExtractionOptions = VideoFrameExtractionOptions()
    ) async throws -> VideoExtractionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let url: URL
        switch source {
        case .url(let sourceURL), .file(let sourceURL):
            url = sourceURL
        }
        
        let asset = AVURLAsset(url: url)
        
        // Get video metadata
        let metadata = try await getVideoMetadata(from: asset)
        
        // Determine timestamps to extract
        let timestamps = determineTimestamps(
            duration: metadata.duration,
            options: options
        )
        
        // Create image generator
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)
        
        if let resize = options.resize {
            generator.maximumSize = CGSize(width: resize.width, height: resize.height)
        }
        
        var frames: [ExtractedFrame] = []
        var failedFrames = 0
        
        for (index, timestamp) in timestamps.enumerated() {
            let cmTime = CMTime(seconds: timestamp, preferredTimescale: 600)
            
            do {
                let (cgImage, actualTime) = try await generator.image(at: cmTime)
                let actualTimestamp = CMTimeGetSeconds(actualTime)
                
                let frameData: Any
                switch options.outputFormat {
                case .pixelData:
                    frameData = try await extractPixelData(
                        from: cgImage,
                        normalization: options.normalization,
                        dataLayout: options.dataLayout
                    )
                case .base64:
                    frameData = try encodeToBase64JPEG(cgImage, quality: options.jpegQuality)
                }
                
                frames.append(ExtractedFrame(
                    timestamp: actualTimestamp,
                    frameIndex: index,
                    data: frameData,
                    width: cgImage.width,
                    height: cgImage.height,
                    error: nil
                ))
            } catch {
                failedFrames += 1
                frames.append(ExtractedFrame(
                    timestamp: timestamp,
                    frameIndex: index,
                    data: "",
                    width: 0,
                    height: 0,
                    error: error.localizedDescription
                ))
            }
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        return VideoExtractionResult(
            frames: frames,
            metadata: metadata,
            processingTimeMs: processingTime,
            failedFrames: failedFrames
        )
    }
    
    // MARK: - Metadata
    
    /// Get video metadata.
    ///
    /// - Parameter source: Video source
    /// - Returns: Video metadata
    /// - Throws: `PixelUtilsError` if metadata cannot be read
    public static func getVideoMetadata(from source: VideoSource) async throws -> VideoMetadata {
        let url: URL
        switch source {
        case .url(let sourceURL), .file(let sourceURL):
            url = sourceURL
        }
        
        let asset = AVURLAsset(url: url)
        return try await getVideoMetadata(from: asset)
    }
    
    // MARK: - Single Frame
    
    /// Extract a single frame at a specific timestamp.
    ///
    /// - Parameters:
    ///   - source: Video source
    ///   - timestamp: Timestamp in seconds
    ///   - options: Extraction options
    /// - Returns: Extracted frame
    /// - Throws: `PixelUtilsError` if extraction fails
    public static func extractFrame(
        from source: VideoSource,
        at timestamp: Double,
        options: VideoFrameExtractionOptions = VideoFrameExtractionOptions()
    ) async throws -> ExtractedFrame {
        var opts = options
        opts.timestamps = [timestamp]
        opts.maxFrames = 1
        
        let result = try await extractFrames(from: source, options: opts)
        
        guard let frame = result.frames.first,
              frame.error == nil else {
            throw PixelUtilsError.processingFailed("Failed to extract frame at \(timestamp)s")
        }
        
        return frame
    }
    
    // MARK: - Private Helpers
    
    private static func getVideoMetadata(from asset: AVURLAsset) async throws -> VideoMetadata {
        let duration = try await asset.load(.duration)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        
        guard let track = tracks.first else {
            throw PixelUtilsError.invalidOptions("No video track found")
        }
        
        let size = try await track.load(.naturalSize)
        let frameRate = try await track.load(.nominalFrameRate)
        
        return VideoMetadata(
            duration: CMTimeGetSeconds(duration),
            frameRate: frameRate,
            width: Int(size.width),
            height: Int(size.height),
            trackCount: tracks.count
        )
    }
    
    private static func determineTimestamps(
        duration: Double,
        options: VideoFrameExtractionOptions
    ) -> [Double] {
        // If specific timestamps provided, use them
        if let timestamps = options.timestamps {
            return timestamps.filter { $0 >= 0 && $0 <= duration }
        }
        
        // If interval provided, use it
        if let interval = options.interval {
            var timestamps: [Double] = []
            var t: Double = 0
            while t <= duration {
                timestamps.append(t)
                t += interval
                
                if let maxFrames = options.maxFrames, timestamps.count >= maxFrames {
                    break
                }
            }
            return timestamps
        }
        
        // If maxFrames provided, distribute evenly
        if let maxFrames = options.maxFrames, maxFrames > 0 {
            if maxFrames == 1 {
                return [duration / 2]
            }
            
            var timestamps: [Double] = []
            let interval = duration / Double(maxFrames - 1)
            for i in 0..<maxFrames {
                timestamps.append(Double(i) * interval)
            }
            return timestamps
        }
        
        // Default: single frame at middle
        return [duration / 2]
    }
    
    private static func extractPixelData(
        from cgImage: CGImage,
        normalization: Normalization,
        dataLayout: DataLayout
    ) async throws -> PixelDataResult {
        let pixelOptions = PixelDataOptions(
            colorFormat: .rgb,
            normalization: normalization,
            dataLayout: dataLayout
        )
        return try PixelExtractor.getPixelData(
            source: .cgImage(cgImage),
            options: pixelOptions
        )
    }
    
    private static func encodeToBase64JPEG(_ cgImage: CGImage, quality: Int) throws -> String {
        #if canImport(UIKit)
        let uiImage = UIImage(cgImage: cgImage)
        guard let data = uiImage.jpegData(compressionQuality: CGFloat(quality) / 100.0) else {
            throw PixelUtilsError.processingFailed("Failed to encode frame to JPEG")
        }
        return data.base64EncodedString()
        #elseif canImport(AppKit)
        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        guard let data = bitmap.representation(using: .jpeg, properties: [.compressionFactor: CGFloat(quality) / 100.0]) else {
            throw PixelUtilsError.processingFailed("Failed to encode frame to JPEG")
        }
        return data.base64EncodedString()
        #endif
    }
}
