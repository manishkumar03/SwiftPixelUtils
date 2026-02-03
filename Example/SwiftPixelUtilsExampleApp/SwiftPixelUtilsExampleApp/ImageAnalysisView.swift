//
//  ImageAnalysisView.swift
//  SwiftPixelUtilsExampleApp
//
//  Image analysis demo tab
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Image Analysis Demo
struct ImageAnalysisView: View {
    @State private var statisticsResult = "Tap to analyze"
    @State private var blurResult = "Tap to detect blur"
    @State private var metadataResult = "Tap to get metadata"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Statistics
                    GroupBox("Image Statistics") {
                        VStack(spacing: 12) {
                            Button("Get Statistics") {
                                Task { await getStatistics() }
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Text(statisticsResult)
                                .font(.system(.caption, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    
                    // Blur Detection
                    GroupBox("Blur Detection") {
                        VStack(spacing: 12) {
                            Button("Detect Blur") {
                                Task { await detectBlur() }
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Text(blurResult)
                                .font(.system(.caption, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    
                    // Metadata
                    GroupBox("Image Metadata") {
                        VStack(spacing: 12) {
                            Button("Get Metadata") {
                                Task { await getMetadata() }
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Text(metadataResult)
                                .font(.system(.caption, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Image Analysis")
        }
    }
    
    func getStatistics() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let stats = try ImageAnalyzer.getStatistics(source: source)
            
            statisticsResult = """
            ✅ Statistics:
            Mean RGB: [\(stats.mean.map { String(format: "%.3f", $0) }.joined(separator: ", "))]
            Std RGB: [\(stats.std.map { String(format: "%.3f", $0) }.joined(separator: ", "))]
            Min RGB: [\(stats.min.map { String(format: "%.3f", $0) }.joined(separator: ", "))]
            Max RGB: [\(stats.max.map { String(format: "%.3f", $0) }.joined(separator: ", "))]
            Histogram buckets: \(stats.histogram.count) channels
            """
        } catch {
            statisticsResult = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func detectBlur() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let result = try ImageAnalyzer.detectBlur(source: source)
            
            blurResult = """
            ✅ Blur Detection:
            Is Blurry: \(result.isBlurry ? "Yes" : "No")
            Score: \(String(format: "%.2f", result.score))
            Threshold: \(String(format: "%.2f", result.threshold))
            Processing Time: \(String(format: "%.2f", result.processingTimeMs))ms
            """
        } catch {
            blurResult = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func getMetadata() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let metadata = try ImageAnalyzer.getMetadata(source: source)
            
            metadataResult = """
            ✅ Metadata:
            Size: \(metadata.width)x\(metadata.height)
            Channels: \(metadata.channels)
            Has Alpha: \(metadata.hasAlpha)
            Aspect Ratio: \(String(format: "%.3f", metadata.aspectRatio))
            Color Space: \(metadata.colorSpace)
            """
        } catch {
            metadataResult = "❌ Error: \(error.localizedDescription)"
        }
    }
}
