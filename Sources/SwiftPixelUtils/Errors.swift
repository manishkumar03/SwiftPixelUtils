//
//  Errors.swift
//  SwiftPixelUtils
//
//  Error definitions for the library
//

import Foundation

/// Errors that can occur during pixel utility operations.
///
/// ## Error Handling
///
/// All SwiftPixelUtils operations can throw errors. Handle them with do-catch:
///
/// ```swift
/// do {
///     let result = try await PixelExtractor.getPixelData(from: source, options: options)
///     // Use result...
/// } catch PixelUtilsError.invalidSource(let message) {
///     print("Source error: \(message)")
/// } catch PixelUtilsError.loadFailed(let message) {
///     print("Load error: \(message)")
/// } catch {
///     print("Unexpected error: \(error)")
/// }
/// ```
///
/// ## Common Error Scenarios
///
/// | Error | Common Cause | Solution |
/// |-------|--------------|----------|
/// | ``invalidSource(_:)`` | Nil image, invalid URL | Validate input before calling |
/// | ``loadFailed(_:)`` | Network error, corrupt file | Check file exists, handle network errors |
/// | ``invalidROI(_:)`` | ROI outside image bounds | Ensure ROI fits within image dimensions |
/// | ``invalidOptions(_:)`` | Custom norm without mean/std | Provide mean and std for custom normalization |
public enum PixelUtilsError: Error, LocalizedError {
    case invalidSource(String)
    case loadFailed(String)
    case invalidROI(String)
    case processingFailed(String)
    case invalidOptions(String)
    case invalidChannel(String)
    case invalidPatch(String)
    case dimensionMismatch(String)
    case emptyBatch(String)
    case unknown(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidSource(let message):
            return "Invalid source: \(message)"
        case .loadFailed(let message):
            return "Load failed: \(message)"
        case .invalidROI(let message):
            return "Invalid ROI: \(message)"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .invalidOptions(let message):
            return "Invalid options: \(message)"
        case .invalidChannel(let message):
            return "Invalid channel: \(message)"
        case .invalidPatch(let message):
            return "Invalid patch: \(message)"
        case .dimensionMismatch(let message):
            return "Dimension mismatch: \(message)"
        case .emptyBatch(let message):
            return "Empty batch: \(message)"
        case .unknown(let message):
            return "Unknown error: \(message)"
        }
    }
}
