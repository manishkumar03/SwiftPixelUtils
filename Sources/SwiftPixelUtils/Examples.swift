//
//  ExampleUsage.swift
//  SwiftPixelUtils
//
//  Example usage demonstrations
//

import Foundation

#if canImport(UIKit)
import UIKit
#endif

/// Example usage of SwiftPixelUtils
public class ExampleUsage {
    
    // MARK: - Basic Pixel Extraction
    
    public static func basicExample() async throws {
        // Load image from URL
        let url = URL(string: "https://example.com/image.jpg")!
        let result = try await PixelExtractor.getPixelData(
            source: .url(url),
            options: PixelDataOptions()
        )
        
        print("Image dimensions: \(result.width)x\(result.height)")
        print("Channels: \(result.channels)")
        print("Shape: \(result.shape)")
        print("Processing time: \(result.processingTimeMs)ms")
    }
    
    // MARK: - Model Preset Example
    
    public static func modelPresetExample() async throws {
        let fileURL = URL(fileURLWithPath: "/path/to/image.jpg")
        
        // Use YOLO preset
        let yoloResult = try await PixelExtractor.getPixelData(
            source: .file(fileURL),
            options: ModelPresets.yolov8
        )
        
        print("YOLO preprocessing complete:")
        print("Shape: \(yoloResult.shape)") // [1, 3, 640, 640]
        print("Layout: \(yoloResult.dataLayout)") // NCHW
        
        // Use MobileNet preset
        let mobileNetResult = try await PixelExtractor.getPixelData(
            source: .file(fileURL),
            options: ModelPresets.mobilenet
        )
        
        print("MobileNet preprocessing complete:")
        print("Shape: \(mobileNetResult.shape)") // [1, 224, 224, 3]
    }
    
    // MARK: - Custom Options Example
    
    public static func customOptionsExample() async throws {
        let fileURL = URL(fileURLWithPath: "/path/to/image.jpg")
        
        let options = PixelDataOptions(
            colorFormat: .rgb,
            resize: ResizeOptions(
                width: 224,
                height: 224,
                strategy: .cover
            ),
            roi: nil,
            normalization: .imagenet,
            dataLayout: .nchw,
            outputFormat: .float32Array
        )
        
        let result = try await PixelExtractor.getPixelData(
            source: .file(fileURL),
            options: options
        )
        
        print("Custom preprocessing complete:")
        print("Data points: \(result.data.count)")
    }
    
    // MARK: - Batch Processing Example
    
    public static func batchProcessingExample() async throws {
        let sources: [ImageSource] = [
            .url(URL(string: "https://example.com/1.jpg")!),
            .url(URL(string: "https://example.com/2.jpg")!),
            .url(URL(string: "https://example.com/3.jpg")!)
        ]
        
        let results = try await PixelExtractor.batchGetPixelData(
            sources: sources,
            options: ModelPresets.mobilenet,
            concurrency: 2
        )
        
        print("Processed \(results.count) images")
        for (index, result) in results.enumerated() {
            print("Image \(index + 1): \(result.width)x\(result.height)")
        }
    }
    
    // MARK: - Bounding Box Example
    
    public static func boundingBoxExample() {
        // Convert box format
        let yoloBox: [Float] = [320, 240, 100, 80] // center x, y, width, height
        let xyxyBox = BoundingBox.convertFormat(
            [yoloBox],
            from: .cxcywh,
            to: .xyxy
        )
        print("Converted box: \(xyxyBox[0])") // [270, 200, 370, 280]
        
        // Scale boxes
        let scaledBoxes = BoundingBox.scale(
            [[100, 100, 200, 200]],
            from: CGSize(width: 640, height: 640),
            to: CGSize(width: 1920, height: 1080),
            format: .xyxy
        )
        print("Scaled boxes: \(scaledBoxes)")
        
        // Calculate IoU
        let iou = BoundingBox.calculateIoU(
            [100, 100, 200, 200],
            [150, 150, 250, 250],
            format: .xyxy
        )
        print("IoU: \(iou)")
        
        // Non-Maximum Suppression
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
        print("Filtered detections: \(filtered.count)")
    }
}
