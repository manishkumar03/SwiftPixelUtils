//
//  MoreFeaturesViews.swift
//  SwiftPixelUtilsExampleApp
//
//  More features menu, Label Database, and About views
//

import SwiftUI
import SwiftPixelUtils

// MARK: - More Features Menu
struct MoreFeaturesView: View {
    var body: some View {
        NavigationView {
            List {
                Section("Augmentation") {
                    NavigationLink(destination: AugmentationView()) {
                        Label("Augmentation", systemImage: "wand.and.stars")
                    }
                    
                    NavigationLink(destination: IndividualAugmentationsView()) {
                        Label("Individual Augmentations", systemImage: "slider.horizontal.3")
                    }
                    
                    NavigationLink(destination: CutoutView()) {
                        Label("Cutout", systemImage: "scissors")
                    }
                }
                
                Section("Analysis") {
                    NavigationLink(destination: ImageAnalysisView()) {
                        Label("Image Analysis", systemImage: "chart.bar.xaxis")
                    }
                }
                
                Section("Inference Post-Processing") {
                    NavigationLink(destination: InferenceUtilitiesView()) {
                        Label("Inference Utilities", systemImage: "brain")
                    }
                }
                
                Section("Data Processing") {
                    NavigationLink(destination: LabelDatabaseView()) {
                        Label("Label Database", systemImage: "tag")
                    }
                    
                    NavigationLink(destination: TensorOperationsView()) {
                        Label("Tensor Operations", systemImage: "cube")
                    }
                    
                    NavigationLink(destination: LetterboxView()) {
                        Label("Letterbox", systemImage: "rectangle.center.inset.filled")
                    }
                    
                    NavigationLink(destination: QuantizationView()) {
                        Label("Quantization", systemImage: "number.circle")
                    }
                    
                    NavigationLink(destination: MultiCropView()) {
                        Label("Multi-Crop", systemImage: "rectangle.split.3x3")
                    }
                }
                
                Section("Visualization") {
                    NavigationLink(destination: DrawingView()) {
                        Label("Drawing/Boxes", systemImage: "pencil.and.outline")
                    }
                    
                    NavigationLink(destination: TensorToImageView()) {
                        Label("Tensor → Image", systemImage: "photo")
                    }
                }
                
                Section("Validation & Batch") {
                    NavigationLink(destination: TensorValidationView()) {
                        Label("Tensor Validation", systemImage: "checkmark.shield")
                    }
                    
                    NavigationLink(destination: BatchOperationsView()) {
                        Label("Batch Operations", systemImage: "square.stack.3d.up")
                    }
                    
                    NavigationLink(destination: ImageValidationView()) {
                        Label("Image Validation", systemImage: "checkmark.rectangle")
                    }
                }
                
                Section("Video & Media") {
                    NavigationLink(destination: VideoFrameView()) {
                        Label("Video Frames", systemImage: "film")
                    }
                }
                
                Section("Info") {
                    NavigationLink(destination: AboutView()) {
                        Label("About", systemImage: "info.circle")
                    }
                }
            }
            .navigationTitle("More Features")
        }
    }
}

// MARK: - About View
struct AboutView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("SwiftPixelUtils")
                    .font(.largeTitle)
                    .bold()
                
                Text("High-performance Swift library for image preprocessing optimized for ML/AI inference pipelines.")
                    .foregroundColor(.secondary)
                
                GroupBox("Features") {
                    VStack(alignment: .leading, spacing: 8) {
                        FeatureRow(icon: "square.grid.3x3", title: "Pixel Extraction", description: "Raw pixel data with multiple formats")
                        FeatureRow(icon: "chart.bar", title: "Image Analysis", description: "Statistics, blur detection, metadata")
                        FeatureRow(icon: "wand.and.stars", title: "Augmentation", description: "Rotate, flip, color jitter, blur")
                        FeatureRow(icon: "rectangle.dashed", title: "Bounding Boxes", description: "Format conversion, IoU, NMS")
                        FeatureRow(icon: "tag", title: "Label Database", description: "COCO, ImageNet, CIFAR, VOC, Places365")
                        FeatureRow(icon: "cube", title: "Tensor Ops", description: "Channel extraction, permute, softmax")
                        FeatureRow(icon: "rectangle.center.inset.filled", title: "Letterbox", description: "YOLO-style padding with transforms")
                        FeatureRow(icon: "number.circle", title: "Quantization", description: "Float to int8/uint8/int16")
                        FeatureRow(icon: "rectangle.split.3x3", title: "Multi-Crop", description: "Five-crop, ten-crop, grid, random")
                        FeatureRow(icon: "brain", title: "Inference Utils", description: "Softmax, Sigmoid, Soft-NMS, Top-K, CoreML")
                        FeatureRow(icon: "cpu", title: "CoreML Integration", description: "MLMultiArray conversion, CVPixelBuffer")
                    }
                }
                
                GroupBox("Model Presets") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("• YOLO / YOLOv8")
                        Text("• MobileNet v1/v2/v3")
                        Text("• EfficientNet")
                        Text("• ResNet / ResNet50")
                        Text("• Vision Transformer (ViT)")
                        Text("• CLIP")
                        Text("• SAM")
                        Text("• DINO")
                        Text("• DETR")
                    }
                    .font(.caption)
                }
                
                GroupBox("Platforms") {
                    HStack {
                        Label("iOS 15+", systemImage: "iphone")
                        Spacer()
                        Label("macOS 12+", systemImage: "desktopcomputer")
                        Spacer()
                        Label("tvOS 15+", systemImage: "appletv")
                    }
                    .font(.caption)
                }
            }
            .padding()
        }
        .navigationTitle("About")
    }
}

// MARK: - Feature Row
struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundColor(.accentColor)
            VStack(alignment: .leading) {
                Text(title).font(.subheadline).bold()
                Text(description).font(.caption).foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - Label Database Demo
struct LabelDatabaseView: View {
    @State private var result = "Tap to query labels"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Single Label Lookup") {
                    VStack(spacing: 12) {
                        Button("COCO: Index 0") {
                            lookupLabel(index: 0, dataset: .coco)
                        }
                        .buttonStyle(.bordered)
                        
                        Button("ImageNet: Index 281") {
                            lookupLabel(index: 281, dataset: .imagenet)
                        }
                        .buttonStyle(.bordered)
                        
                        Button("CIFAR-10: Index 3") {
                            lookupLabel(index: 3, dataset: .cifar10)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Top-K Labels") {
                    VStack(spacing: 12) {
                        Button("Simulate ImageNet Top-5") {
                            simulateTopK()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Top-5 with Softmax (from logits)") {
                            simulateTopKWithSoftmax()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Test Softmax Function") {
                            testSoftmax()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Dataset Info") {
                    VStack(spacing: 12) {
                        ForEach([LabelDataset.coco, .imagenet, .imagenet21k, .cifar10, .places365], id: \.self) { dataset in
                            Button(dataset.rawValue.uppercased()) {
                                showDatasetInfo(dataset)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                
                GroupBox("Available Datasets") {
                    Button("List All Datasets") {
                        listAllDatasets()
                    }
                    .buttonStyle(.borderedProminent)
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Label Database")
    }
    
    func lookupLabel(index: Int, dataset: LabelDataset) {
        if let label = LabelDatabase.getLabel(index, dataset: dataset) {
            result = """
            ✅ Label Lookup
            Dataset: \(dataset.rawValue)
            Index: \(index)
            Label: "\(label)"
            """
        } else {
            result = "❌ Index \(index) not found in \(dataset.rawValue)"
        }
    }
    
    func simulateTopK() {
        // Simulate softmax output with high confidence for cat classes
        var scores = [Float](repeating: 0.001, count: 1000)
        scores[281] = 0.85  // tabby cat
        scores[282] = 0.08  // tiger cat
        scores[285] = 0.03  // Egyptian Mau
        scores[283] = 0.02  // Persian cat
        scores[287] = 0.01  // lynx
        
        let topLabels = LabelDatabase.getTopLabels(
            scores: scores,
            dataset: .imagenet,
            k: 5,
            minConfidence: 0.005
        )
        
        result = """
        ✅ Top-5 ImageNet Predictions
        (Simulated cat image)
        
        \(topLabels.enumerated().map { i, item in
            "\(i + 1). \(item.label): \(String(format: "%.1f", item.confidence * 100))% (idx: \(item.index))"
        }.joined(separator: "\n"))
        """
    }
    
    func simulateTopKWithSoftmax() {
        // Simulate raw logits (unnormalized model output)
        var logits = [Float](repeating: -5.0, count: 1000)
        logits[281] = 8.5   // tabby cat - high logit
        logits[282] = 5.2   // tiger cat
        logits[285] = 3.8   // Egyptian Mau
        logits[283] = 3.1   // Persian cat
        logits[287] = 2.4   // lynx
        
        let topLabels = LabelDatabase.getTopLabelsWithSoftmax(
            logits: logits,
            dataset: .imagenet,
            k: 5,
            minConfidence: 0.001
        )
        
        result = """
        ✅ Top-5 with Softmax (from raw logits)
        
        Input logits (sample):
        tabby: 8.5, tiger cat: 5.2, ...
        
        After softmax normalization:
        \(topLabels.enumerated().map { i, item in
            "\(i + 1). \(item.label): \(String(format: "%.1f", item.confidence * 100))% (idx: \(item.index))"
        }.joined(separator: "\n"))
        """
    }
    
    func testSoftmax() {
        // Simple example showing softmax
        let logits: [Float] = [2.0, 1.0, 0.1]
        let probabilities = LabelDatabase.softmax(logits)
        
        result = """
        ✅ Softmax Function
        
        Input logits: [\(logits.map { String(format: "%.1f", $0) }.joined(separator: ", "))]
        
        Formula: softmax(x_i) = exp(x_i) / Σexp(x_j)
        
        Output probabilities:
        [\(probabilities.map { String(format: "%.4f", $0) }.joined(separator: ", "))]
        
        Sum: \(String(format: "%.4f", probabilities.reduce(0, +)))
        (Should equal 1.0)
        """
    }
    
    func showDatasetInfo(_ dataset: LabelDataset) {
        let info = LabelDatabase.getDatasetInfo(for: dataset)
        let labels = LabelDatabase.getAllLabels(for: dataset)
        let sampleLabels = labels.prefix(5).joined(separator: ", ")
        
        result = """
        ✅ Dataset: \(info.name.uppercased())
        Classes: \(info.numClasses)
        Description: \(info.description)
        
        Sample labels:
        \(sampleLabels)...
        """
    }
    
    func listAllDatasets() {
        let datasets = LabelDatabase.getAvailableDatasets()
        
        result = """
        ✅ Available Datasets (\(datasets.count))
        
        \(datasets.map { dataset in
            let info = LabelDatabase.getDatasetInfo(for: dataset)
            return "• \(dataset.rawValue): \(info.numClasses) classes"
        }.joined(separator: "\n"))
        """
    }
}
