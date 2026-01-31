//
//  ContentView.swift
//  SwiftPixelUtilsExampleApp
//
//  Comprehensive demo of SwiftPixelUtils features
//  Mirrors the react-native-vision-utils example app
//

import SwiftUI
import SwiftPixelUtils

struct ContentView: View {
    var body: some View {
        TabView {
            PixelExtractionView()
                .tabItem {
                    Label("Pixels", systemImage: "square.grid.3x3")
                }
            
            ImageAnalysisView()
                .tabItem {
                    Label("Analyze", systemImage: "chart.bar.xaxis")
                }
            
            AugmentationView()
                .tabItem {
                    Label("Augment", systemImage: "wand.and.stars")
                }
            
            BoundingBoxView()
                .tabItem {
                    Label("Boxes", systemImage: "rectangle.dashed")
                }
            
            MoreFeaturesView()
                .tabItem {
                    Label("More", systemImage: "ellipsis.circle")
                }
        }
    }
}

// MARK: - More Features Menu
struct MoreFeaturesView: View {
    var body: some View {
        NavigationView {
            List {
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
                
                Section("Augmentation") {
                    NavigationLink(destination: CutoutView()) {
                        Label("Cutout", systemImage: "scissors")
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
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let stats = try await ImageAnalyzer.getStatistics(source: source)
            
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
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let result = try await ImageAnalyzer.detectBlur(source: source)
            
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
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let metadata = try await ImageAnalyzer.getMetadata(source: source)
            
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

// MARK: - Augmentation Demo
struct AugmentationView: View {
    @State private var result = "Tap to apply augmentations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Single Augmentations") {
                        VStack(spacing: 12) {
                            Button("Rotate 45°") {
                                Task { await applyAugmentation(AugmentationOptions(rotation: 45)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Horizontal Flip") {
                                Task { await applyAugmentation(AugmentationOptions(horizontalFlip: true)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Brightness +20%") {
                                Task { await applyAugmentation(AugmentationOptions(brightness: 0.2)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Contrast +30%") {
                                Task { await applyAugmentation(AugmentationOptions(contrast: 0.3)) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Gaussian Blur") {
                                Task { await applyAugmentation(AugmentationOptions(blur: BlurOptions(type: .gaussian, radius: 5))) }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Saturation -20%") {
                                Task { await applyAugmentation(AugmentationOptions(saturation: -0.2)) }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Combined Augmentations") {
                        Button("Rotate + Flip + Brightness") {
                            Task {
                                await applyAugmentation(AugmentationOptions(
                                    rotation: 15,
                                    horizontalFlip: true,
                                    brightness: 0.1
                                ))
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    GroupBox("Color Jitter") {
                        Button("Apply Color Jitter") {
                            Task { await applyColorJitter() }
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
            .navigationTitle("Augmentation")
        }
    }
    
    func applyAugmentation(_ options: AugmentationOptions) async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            let augmented = try await ImageAugmentor.applyAugmentations(to: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            #if canImport(UIKit)
            let size = augmented.size
            result = """
            ✅ Augmentation Applied!
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            #else
            let size = augmented.size
            result = """
            ✅ Augmentation Applied!
            Output Size: \(Int(size.width))x\(Int(size.height))
            Processing Time: \(String(format: "%.2f", time))ms
            """
            #endif
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func applyColorJitter() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            let jittered = try await ImageAugmentor.colorJitter(
                source: source,
                options: ColorJitterOptions(
                    brightness: 0.2,
                    contrast: 0.2,
                    saturation: 0.3,
                    hue: 0.1
                )
            )
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Color Jitter Applied!
            Brightness: ±0.2
            Contrast: ±0.2
            Saturation: ±0.3
            Hue: ±0.1
            Applied values:
              Brightness: \(String(format: "%.3f", jittered.appliedBrightness))
              Contrast: \(String(format: "%.3f", jittered.appliedContrast))
              Saturation: \(String(format: "%.3f", jittered.appliedSaturation))
              Hue: \(String(format: "%.3f", jittered.appliedHue))
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Bounding Box Demo
struct BoundingBoxView: View {
    @State private var result = "Tap to test bounding box utilities"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Format Conversion") {
                        VStack(spacing: 12) {
                            Button("cxcywh → xyxy") {
                                testFormatConversion(from: .cxcywh, to: .xyxy)
                            }
                            .buttonStyle(.bordered)
                            
                            Button("xyxy → xywh") {
                                testFormatConversion(from: .xyxy, to: .xywh)
                            }
                            .buttonStyle(.bordered)
                            
                            Button("xywh → cxcywh") {
                                testFormatConversion(from: .xywh, to: .cxcywh)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Box Operations") {
                        VStack(spacing: 12) {
                            Button("Calculate IoU") {
                                testIoU()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Scale Boxes") {
                                testScaling()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Clip Boxes") {
                                testClipping()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Non-Max Suppression") {
                                testNMS()
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                    
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding()
            }
            .navigationTitle("Bounding Boxes")
        }
    }
    
    func testFormatConversion(from: BoxFormat, to: BoxFormat) {
        let inputBox: [Float] = [320, 240, 100, 80] // Example box
        let converted = BoundingBox.convertFormat([inputBox], from: from, to: to)
        
        result = """
        ✅ Format Conversion
        From: \(from) → To: \(to)
        Input: \(inputBox)
        Output: \(converted.first ?? [])
        """
    }
    
    func testIoU() {
        let box1: [Float] = [100, 100, 200, 200]
        let box2: [Float] = [150, 150, 250, 250]
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        
        result = """
        ✅ IoU Calculation
        Box 1: \(box1)
        Box 2: \(box2)
        IoU: \(String(format: "%.4f", iou))
        Overlap: \(String(format: "%.1f", iou * 100))%
        """
    }
    
    func testScaling() {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 640, height: 640),
            to: CGSize(width: 1920, height: 1080),
            format: .xyxy
        )
        
        result = """
        ✅ Box Scaling
        From: 640x640
        To: 1920x1080
        Input: \(boxes.first ?? [])
        Output: \(scaled.first ?? [])
        """
    }
    
    func testClipping() {
        let boxes: [[Float]] = [[-10, 50, 700, 500]]
        let clipped = BoundingBox.clip(boxes, imageSize: CGSize(width: 640, height: 480), format: .xyxy)
        
        result = """
        ✅ Box Clipping
        Image: 640x480
        Input: \(boxes.first ?? [])
        Clipped: \(clipped.first ?? [])
        """
    }
    
    func testNMS() {
        let detections: [Detection] = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0),
            Detection(box: [300, 300, 400, 400], score: 0.7, classIndex: 1),
            Detection(box: [105, 105, 205, 205], score: 0.85, classIndex: 0),
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        result = """
        ✅ Non-Max Suppression
        Input detections: \(detections.count)
        IoU threshold: 0.5
        Score threshold: 0.3
        Output detections: \(filtered.count)
        
        Kept boxes:
        \(filtered.map { "- Class \($0.classIndex): score \(String(format: "%.2f", $0.score))" }.joined(separator: "\n"))
        """
    }
}

// MARK: - Label Database Demo
struct LabelDatabaseView: View {
    @State private var result = "Tap to query labels"
    
    var body: some View {
        NavigationView {
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
                        Button("Simulate ImageNet Top-5") {
                            simulateTopK()
                        }
                        .buttonStyle(.borderedProminent)
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

#Preview {
    ContentView()
}
