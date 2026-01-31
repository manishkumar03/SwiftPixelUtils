//
//  VideoFrameView.swift
//  SwiftPixelUtilsExampleApp
//
//  Video frame extraction demo
//

import SwiftUI
import SwiftPixelUtils

struct VideoFrameView: View {
    @State private var result = "Tap to test video frame extraction"
    @State private var isLoading = false
    @State private var previewImage: PlatformImage?
    @State private var previewImages: [PlatformImage] = []
    @State private var showImagePreview = false
    @State private var showImagesPreview = false
    
    // Public domain sample video (Big Buck Bunny)
    private let sampleVideoURL = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Video Metadata") {
                    Button("Get Video Metadata") {
                        Task { await getVideoMetadata() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isLoading)
                }
                
                GroupBox("Single Frame Extraction") {
                    VStack(spacing: 12) {
                        Button("Extract Frame at 1.0s") {
                            Task { await extractSingleFrame(at: 1.0) }
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoading)
                        
                        Button("Extract Frame at 2.5s") {
                            Task { await extractSingleFrame(at: 2.5) }
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoading)
                    }
                }
                
                GroupBox("Multiple Frames") {
                    VStack(spacing: 12) {
                        Button("Extract 5 Evenly-Spaced Frames") {
                            Task { await extractMultipleFrames(count: 5) }
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoading)
                        
                        Button("Extract 10 Frames (224x224)") {
                            Task { await extractMultipleFramesResized(count: 10, size: 224) }
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoading)
                    }
                }
                
                if isLoading {
                    ProgressView("Processing video...")
                        .padding()
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Video Frames")
        .sheet(isPresented: $showImagePreview) {
            ImagePreviewSheet(image: previewImage, isPresented: $showImagePreview)
        }
        .sheet(isPresented: $showImagesPreview) {
            MultiImagePreviewSheet(images: previewImages, isPresented: $showImagesPreview)
        }
    }
    
    func getVideoMetadata() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            let source = VideoSource.url(URL(string: sampleVideoURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let metadata = try await VideoFrameExtractor.getVideoMetadata(from: source)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Video Metadata
            Duration: \(String(format: "%.2f", metadata.duration))s
            Size: \(metadata.width)x\(metadata.height)
            Frame Rate: \(String(format: "%.2f", metadata.frameRate)) fps
            Total Frames: ~\(Int(metadata.duration * Double(metadata.frameRate)))
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func extractSingleFrame(at timestamp: Double) async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            let source = VideoSource.url(URL(string: sampleVideoURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let frame = try await VideoFrameExtractor.extractFrame(
                from: source,
                at: timestamp,
                options: VideoFrameExtractionOptions(outputFormat: .base64, jpegQuality: 90)
            )
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Frame Extracted
            Timestamp: \(String(format: "%.2f", timestamp))s
            Frame Size: \(frame.width)x\(frame.height)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            // Decode base64 to image
            if let base64String = frame.data as? String,
               let imageData = Data(base64Encoded: base64String) {
                #if canImport(UIKit)
                previewImage = UIImage(data: imageData)
                #else
                previewImage = NSImage(data: imageData)
                #endif
                showImagePreview = true
            }
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func extractMultipleFrames(count: Int) async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            let source = VideoSource.url(URL(string: sampleVideoURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = VideoFrameExtractionOptions(
                maxFrames: count,
                outputFormat: .base64,
                jpegQuality: 80
            )
            let extraction = try await VideoFrameExtractor.extractFrames(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            let timestamps = extraction.frames.map { 
                String(format: "%.2fs", $0.timestamp) 
            }.joined(separator: ", ")
            
            result = """
            ✅ Multiple Frames Extracted
            Requested: \(count) frames
            Extracted: \(extraction.frames.count) frames
            Video Duration: \(String(format: "%.2f", extraction.metadata.duration))s
            Video Size: \(extraction.metadata.width)x\(extraction.metadata.height)
            
            Timestamps: \(timestamps)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            // Decode base64 frames to images
            var images: [PlatformImage] = []
            for frame in extraction.frames {
                if let base64String = frame.data as? String,
                   let imageData = Data(base64Encoded: base64String) {
                    #if canImport(UIKit)
                    if let image = UIImage(data: imageData) {
                        images.append(image)
                    }
                    #else
                    if let image = NSImage(data: imageData) {
                        images.append(image)
                    }
                    #endif
                }
            }
            if !images.isEmpty {
                previewImages = images
                showImagesPreview = true
            }
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func extractMultipleFramesResized(count: Int, size: Int) async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            let source = VideoSource.url(URL(string: sampleVideoURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = VideoFrameExtractionOptions(
                maxFrames: count,
                resize: (width: size, height: size),
                outputFormat: .base64,
                jpegQuality: 80
            )
            let extraction = try await VideoFrameExtractor.extractFrames(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Resized Frames Extracted
            Requested: \(count) frames at \(size)x\(size)
            Extracted: \(extraction.frames.count) frames
            Video Duration: \(String(format: "%.2f", extraction.metadata.duration))s
            Original Size: \(extraction.metadata.width)x\(extraction.metadata.height)
            Output Size: \(size)x\(size)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
            
            // Decode base64 frames to images
            var images: [PlatformImage] = []
            for frame in extraction.frames {
                if let base64String = frame.data as? String,
                   let imageData = Data(base64Encoded: base64String) {
                    #if canImport(UIKit)
                    if let image = UIImage(data: imageData) {
                        images.append(image)
                    }
                    #else
                    if let image = NSImage(data: imageData) {
                        images.append(image)
                    }
                    #endif
                }
            }
            if !images.isEmpty {
                previewImages = images
                showImagesPreview = true
            }
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
