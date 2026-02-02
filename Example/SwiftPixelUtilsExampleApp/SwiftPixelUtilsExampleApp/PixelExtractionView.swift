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
                    GroupBox("Classification Presets") {
                        VStack(spacing: 12) {
                            PresetButton(title: "MobileNet", preset: ModelPresets.mobilenet, accessibilityId: "pixel-preset-mobilenet") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ResNet50", preset: ModelPresets.resnet50, accessibilityId: "pixel-preset-resnet50") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ViT", preset: ModelPresets.vit, accessibilityId: "pixel-preset-vit") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "CLIP", preset: ModelPresets.clip, accessibilityId: "pixel-preset-clip") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                        }
                    }
                    
                    // Detection Presets
                    GroupBox("Detection Presets") {
                        VStack(spacing: 12) {
                            PresetButton(title: "YOLOv8", preset: ModelPresets.yolov8, accessibilityId: "pixel-preset-yolov8") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "RT-DETR", preset: ModelPresets.rtdetr, accessibilityId: "pixel-preset-rtdetr") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                        }
                    }
                    
                    // ONNX Presets Section
                    GroupBox("ONNX Presets") {
                        VStack(spacing: 12) {
                            PresetButton(title: "ONNX YOLOv8", preset: ModelPresets.onnx_yolov8, accessibilityId: "pixel-preset-onnx-yolov8") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ONNX RT-DETR", preset: ModelPresets.onnx_rtdetr, accessibilityId: "pixel-preset-onnx-rtdetr") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ONNX ResNet", preset: ModelPresets.onnx_resnet, accessibilityId: "pixel-preset-onnx-resnet") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ONNX MobileNetV2", preset: ModelPresets.onnx_mobilenetv2, accessibilityId: "pixel-preset-onnx-mobilenetv2") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ONNX ViT", preset: ModelPresets.onnx_vit, accessibilityId: "pixel-preset-onnx-vit") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                        }
                    }
                    
                    // ONNX Quantized Presets
                    GroupBox("ONNX Quantized") {
                        VStack(spacing: 12) {
                            PresetButton(title: "ONNX UInt8", preset: ModelPresets.onnx_quantized_uint8, accessibilityId: "pixel-preset-onnx-uint8") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ONNX Int8", preset: ModelPresets.onnx_quantized_int8, accessibilityId: "pixel-preset-onnx-int8") { result, time in
                                self.result = result
                                self.processingTime = time
                            }
                            PresetButton(title: "ONNX Float16", preset: ModelPresets.onnx_float16, accessibilityId: "pixel-preset-onnx-float16") { result, time in
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
                            .accessibilityIdentifier("pixel-custom-rgb-imagenet")
                            
                            Button("Grayscale + Scale") {
                                Task { await extractCustom(colorFormat: .grayscale, normalization: .scale) }
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("pixel-custom-grayscale")
                            
                            Button("BGR + TensorFlow") {
                                Task { await extractCustom(colorFormat: .bgr, normalization: .tensorflow) }
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("pixel-custom-bgr-tensorflow")
                        }
                    }
                    
                    // Results Section
                    GroupBox("Results") {
                        VStack(alignment: .leading, spacing: 8) {
                            if processingTime > 0 {
                                Text("Processing Time: \(String(format: "%.2f", processingTime))ms")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .accessibilityIdentifier("pixel-processing-time")
                            }
                            Text(result)
                                .font(.system(.body, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .accessibilityIdentifier("pixel-result-text")
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
    let accessibilityId: String
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
        .accessibilityIdentifier(accessibilityId)
    }
}
