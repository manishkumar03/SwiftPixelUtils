//
//  SwiftPixelUtils.swift
//  SwiftPixelUtils
//
//  High-performance image preprocessing for ML/AI inference pipelines
//

/// # SwiftPixelUtils
///
/// A comprehensive Swift library for image preprocessing in ML/AI pipelines,
/// providing the same functionality as react-native-vision-utils but with native
/// Swift performance and Apple framework integration.
///
/// ## Overview
///
/// SwiftPixelUtils handles the critical preprocessing step between raw images
/// and machine learning model inference. It provides:
///
/// - **Pixel Extraction**: Convert images to normalized float arrays
/// - **Color Space Conversion**: RGB, BGR, HSV, HSL, LAB, YUV, YCbCr, Grayscale
/// - **Resize Strategies**: Cover, contain, stretch, letterbox
/// - **ML Normalization**: ImageNet, TensorFlow, custom mean/std
/// - **Data Layouts**: HWC, CHW, NHWC, NCHW for different frameworks
/// - **Object Detection**: Bounding box utilities, IoU, NMS
/// - **Model Presets**: YOLO, MobileNet, ResNet, CLIP, SAM, etc.
///
/// ## Quick Start
///
/// ```swift
/// import SwiftPixelUtils
///
/// // Basic usage with YOLO preset
/// let result = try await PixelExtractor.getPixelData(
///     from: .cgImage(image),
///     options: ModelPresets.yolo
/// )
///
/// // Custom configuration
/// let options = PixelDataOptions(
///     colorFormat: .rgb,
///     resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
///     normalization: .imagenet,
///     dataLayout: .nchw
/// )
/// let result = try await PixelExtractor.getPixelData(from: .cgImage(image), options: options)
///
/// // Use result.data with your ML model
/// let inputTensor = result.data  // [Float] ready for inference
/// ```
///
/// ## Architecture
///
/// ```
/// ImageSource → CGImage → Resize → ROI → RGBA → ColorConvert → Normalize → Layout → Output
/// ```
///
/// Each stage is optimized for performance using CoreGraphics and vDSP (Accelerate)
/// where applicable.
///
/// ## Platform Support
///
/// - iOS 15.0+
/// - macOS 12.0+
/// - tvOS 15.0+
/// - watchOS 8.0+
///
/// ## Topics
///
/// ### Essentials
/// - ``PixelExtractor``
/// - ``ModelPresets``
///
/// ### Configuration
/// - ``PixelDataOptions``
/// - ``ColorFormat``
/// - ``Normalization``
/// - ``DataLayout``
///
/// ### Object Detection
/// - ``BoundingBox``
/// - ``Detection``
/// - ``BoxFormat``
///
/// ### Results
/// - ``PixelDataResult``
/// - ``ImageMetadata``
/// - ``ImageStatistics``

import Foundation
import CoreGraphics
import CoreImage

#if canImport(UIKit)
import UIKit
/// Platform-specific image type (UIImage on iOS/tvOS/watchOS).
public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
import AppKit
/// Platform-specific image type (NSImage on macOS).
public typealias PlatformImage = NSImage
#endif

import Accelerate

// MARK: - Public API Entry Point

/// Main entry point for SwiftPixelUtils functionality.
///
/// This enum serves as a namespace for library-wide constants and utilities.
/// The primary functionality is provided by ``PixelExtractor`` and ``BoundingBox``.
public enum SwiftPixelUtils {
    /// Version of the library
    public static let version = "1.0.0"
}
