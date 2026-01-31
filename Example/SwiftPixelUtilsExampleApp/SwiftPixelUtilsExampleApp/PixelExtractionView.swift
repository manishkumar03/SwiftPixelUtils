//
//  PixelExtractionView.swift
//  SwiftPixelUtilsExampleApp
//
//  Pixel extraction demo tab
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Pixel Extraction Demo
struct PixelExtractionView: View {
    @State private var result: String = "Tap a button to extract pixel data"
    @State private var isLoading = false
    @State private var processingTime: Double = 0
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Model Presets Section
                    GroupBox("Model Presets") {
                        VStack(spacing: 12) {
                            PresetButton(title: "YOLOv8", preset: ModelPresets.yolov8) { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "MobileNet", preset: ModelPresets.mobilenet) { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ResNet50", preset: ModelPresets.resnet50) { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ViT", preset: ModelPresets.vit) { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "CLIP", preset: ModelPresets.clip) { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                        }
                    }
                    
                    // Custom Options Section
                    GroupBox("Custom Options") {
                        VStack(spacing: 12) {
                            Button("RGB + ImageNet Norm") {
                                Task { await extractCustom(colorFormat: .rgb, normalization: .imagenet) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Grayscale + Scale") {
                                Task { await extractCustom(colorFormat: .grayscale, normalization: .scale) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("BGR + TensorFlow") {
                                Task { await extractCustom(colorFormat: .bgr, normalization: .tensorflow) }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    // Results Section
                    GroupBox("Results") {
                        VStack(alignment: .leading, spacing: 8) {
                            if processingTime > 0 {
                                Text("Processing Time: \(String(format: "%.2f", processingTime))ms")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Text(result)
                                .font(.system(.body, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Pixel Extraction")
        }
    }
    
    func extractCustom(colorFormat: ColorFormat, normalization: NormalizationPreset) async {
        let options = PixelDataOptions(
            colorFormat: colorFormat,
            resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
            normalization: Normalization(preset: normalization),
            dataLayout: .nchw
        )
        
        do {
            let start = CFAbsoluteTimeGetCurrent()
            let pixelResult = try await PixelExtractor.getPixelData(
                source: .url(URL(string: sampleImageURL)!),
                options: options
            )
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Success!
            Shape: \(pixelResult.shape)
            Size: \(pixelResult.width)x\(pixelResult.height)
            Channels: \(pixelResult.channels)
            Layout: \(pixelResult.dataLayout)
            Color: \(colorFormat)
            Data points: \(pixelResult.data.count)
            Range: [\(String(format: "%.3f", pixelResult.data.min() ?? 0)), \(String(format: "%.3f", pixelResult.data.max() ?? 0))]
            """
            processingTime = time
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Preset Button
struct PresetButton: View {
    let title: String
    let preset: PixelDataOptions
    let onResult: (String, Double) -> Void
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        Button(title) {
            Task {
                do {
                    let start = CFAbsoluteTimeGetCurrent()
                    let result = try await PixelExtractor.getPixelData(
                        source: .url(URL(string: sampleImageURL)!),
                        options: preset
                    )
                    let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    
                    let text = """
                    ✅ \(title) Preset
                    Shape: \(result.shape)
                    Size: \(result.width)x\(result.height)
                    Channels: \(result.channels)
                    Layout: \(result.dataLayout)
                    Data points: \(result.data.count)
                    """
                    onResult(text, time)
                } catch {
                    onResult("❌ Error: \(error.localizedDescription)", 0)
                }
            }
        }
        .buttonStyle(.borderedProminent)
    }
}
