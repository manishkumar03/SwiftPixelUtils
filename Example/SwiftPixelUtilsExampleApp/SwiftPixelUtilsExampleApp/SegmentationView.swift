//
//  SegmentationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Semantic Segmentation using DeepLabV3 with SwiftPixelUtils
//
//  This demonstrates how to use SwiftPixelUtils for preprocessing images
//  for semantic segmentation models and post-processing with SegmentationOutput.
//

import SwiftUI
import SwiftPixelUtils
import TensorFlowLite

/// # DeepLabV3 Semantic Segmentation
///
/// This view demonstrates semantic segmentation using DeepLabV3 MobileNetV2
/// with TensorFlow Lite and SwiftPixelUtils for preprocessing and post-processing.
///
/// ## Model Details
/// - **Architecture**: DeepLabV3 with MobileNetV2 backbone
/// - **Input**: 257×257×3 RGB (Float32, [0-1] normalized)
/// - **Output**: 257×257×21 class logits (Pascal VOC classes)
/// - **Size**: ~2.4MB
///
/// ## SwiftPixelUtils Integration
///
/// ### Preprocessing
/// ```swift
/// let modelInput = try await PixelExtractor.getModelInput(
///     source: .uiImage(image),
///     framework: .tfliteFloat,
///     width: 257,
///     height: 257
/// )
/// ```
///
/// ### Post-processing with SegmentationOutput
/// ```swift
/// let result = try SegmentationOutput.process(
///     floatOutput: outputArray,
///     format: .logits(height: 257, width: 257, numClasses: 21),
///     labels: .voc
/// )
/// ```
struct SegmentationView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dog", "car", "aeroplane"]

    @State private var selectedImage: String = "dog"
    @State private var loadedImage: UIImage?
    @State private var segmentedImage: UIImage?
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Segmentation'"
    @State private var classSummary: [(classIndex: Int, label: String, pixelCount: Int, percentage: Float)] = []
    @State private var inferenceTime: Double = 0
    @State private var overlayAlpha: Double = 0.5
    @State private var showOverlay: Bool = true
    
    // Model constants
    private let modelWidth = 257
    private let modelHeight = 257
    private let numClasses = 21
    
    var body: some View {
        List {
            // MARK: - Model Info Section
            Section {
                HStack {
                    Image(systemName: "square.grid.3x3.fill")
                        .font(.title)
                        .foregroundColor(.purple)
                    VStack(alignment: .leading) {
                        Text("DeepLabV3 MobileNetV2")
                            .font(.headline)
                        Text("Float32 • 257×257 • 21 VOC Classes")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            } header: {
                Text("Model")
            }
            
            // MARK: - Image Selection Section
            Section {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(sampleImages, id: \.self) { imageName in
                            Button {
                                selectedImage = imageName
                                loadSelectedImage()
                            } label: {
                                VStack {
                                    if let uiImage = loadBundleImage(named: imageName) {
                                        Image(uiImage: uiImage)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 70, height: 70)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 8)
                                                    .stroke(selectedImage == imageName ? Color.purple : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 70, height: 70)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption2)
                                        .foregroundColor(selectedImage == imageName ? .purple : .primary)
                                }
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("segmentation-image-\(imageName)")
                        }
                    }
                    .padding(.vertical, 4)
                }
            } header: {
                Text("Input Image")
            }
            
            // MARK: - Run Inference Section
            Section {
                Button {
                    Task {
                        await runSegmentation()
                    }
                } label: {
                    HStack {
                        Spacer()
                        if isRunning {
                            ProgressView()
                                .padding(.trailing, 8)
                        }
                        Text(isRunning ? "Running..." : "Run Segmentation")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .disabled(isRunning || loadedImage == nil)
                .accessibilityIdentifier("segmentation-run-button")
            }
            
            // MARK: - Results Section
            if let segmented = segmentedImage {
                Section {
                    VStack(spacing: 12) {
                        // Overlay controls
                        Toggle("Show Segmentation Overlay", isOn: $showOverlay)
                            .accessibilityIdentifier("segmentation-overlay-toggle")
                        
                        if showOverlay {
                            HStack {
                                Text("Opacity")
                                Slider(value: $overlayAlpha, in: 0.1...0.9)
                                    .accessibilityIdentifier("segmentation-opacity-slider")
                                Text(String(format: "%.0f%%", overlayAlpha * 100))
                                    .frame(width: 40)
                            }
                        }
                        
                        // Display image
                        HStack {
                            Spacer()
                            Image(uiImage: segmented)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .accessibilityIdentifier("segmentation-result-image")
                            Spacer()
                        }
                    }
                } header: {
                    Label("Segmentation Results", systemImage: "square.grid.3x3.fill")
                        .accessibilityIdentifier("segmentation-results-header")
                }
                .onChange(of: overlayAlpha) { _ in
                    Task {
                        await updateOverlay()
                    }
                }
                .onChange(of: showOverlay) { _ in
                    Task {
                        await updateOverlay()
                    }
                }
                
                // Class summary
                if !classSummary.isEmpty {
                    Section {
                        ForEach(Array(classSummary.prefix(10).enumerated()), id: \.offset) { index, item in
                            HStack {
                                // Color swatch
                                let color = SegmentationColorPalette.voc.color(forClassIndex: item.classIndex)
                                Circle()
                                    .fill(Color(
                                        red: Double(color.r) / 255.0,
                                        green: Double(color.g) / 255.0,
                                        blue: Double(color.b) / 255.0
                                    ))
                                    .frame(width: 16, height: 16)
                                
                                Text(item.label.capitalized)
                                    .font(.subheadline)
                                
                                Spacer()
                                
                                Text(String(format: "%.1f%%", item.percentage))
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        }
                    } header: {
                        Label("Detected Classes", systemImage: "list.bullet.rectangle")
                    }
                }
            }
            
            // MARK: - Performance Section
            if inferenceTime > 0 {
                Section {
                    HStack {
                        Text("Inference Time")
                        Spacer()
                        Text(String(format: "%.2f ms", inferenceTime))
                            .fontWeight(.medium)
                            .foregroundColor(.purple)
                            .accessibilityIdentifier("segmentation-inference-time")
                    }
                } header: {
                    Text("Performance")
                }
            }
            
            // MARK: - Status Section
            Section {
                Text(resultText)
                    .font(.system(.caption, design: .monospaced))
            } header: {
                Text("Status")
            }
            
            // MARK: - VOC Classes Reference
            Section {
                DisclosureGroup("Pascal VOC Classes (21)") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: 8) {
                        ForEach(vocClasses.indices, id: \.self) { index in
                            let color = SegmentationColorPalette.voc.color(forClassIndex: index)
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(Color(
                                        red: Double(color.r) / 255.0,
                                        green: Double(color.g) / 255.0,
                                        blue: Double(color.b) / 255.0
                                    ))
                                    .frame(width: 12, height: 12)
                                Text(vocClasses[index])
                                    .font(.caption2)
                            }
                        }
                    }
                }
            } header: {
                Text("Reference")
            }
            
            // MARK: - Code Example Section
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("SwiftPixelUtils Post-processing:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("""
                    let result = try SegmentationOutput.process(
                        floatOutput: outputArray,
                        format: .logits(
                            height: 257,
                            width: 257,
                            numClasses: 21
                        ),
                        labels: .voc
                    )
                    
                    // Get class summary
                    for item in result.classSummary {
                        print("\\(item.label): \\(item.percentage)%")
                    }
                    
                    // Overlay on image
                    let overlay = try Drawing.overlaySegmentation(
                        on: .uiImage(image),
                        segmentation: result,
                        palette: .voc,
                        alpha: 0.5
                    )
                    """)
                    .font(.system(.caption2, design: .monospaced))
                    .padding(8)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                }
            } header: {
                Text("Code Example")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Segmentation")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadSelectedImage()
        }
    }
    
    // MARK: - VOC Classes (loaded from LabelDatabase with fallback)
    
    private var vocClasses: [String] {
        let labels = LabelDatabase.getAllLabels(for: .voc)
        if labels.isEmpty {
            // Fallback if JSON fails to load
            return [
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
        }
        return labels
    }
    
    // MARK: - Image Loading
    
    private func loadSelectedImage() {
        loadedImage = loadBundleImage(named: selectedImage)
        segmentedImage = nil
        classSummary = []
        inferenceTime = 0
        resultText = "Ready to run segmentation"
    }
    
    // MARK: - Cached results for overlay updates
    @State private var cachedResult: SegmentationResult?
    
    private func updateOverlay() async {
        guard let image = loadedImage, let result = cachedResult else { return }
        
        do {
            if showOverlay {
                let overlay = try Drawing.overlaySegmentation(
                    on: .uiImage(image),
                    segmentation: result,
                    palette: .voc,
                    alpha: CGFloat(overlayAlpha),
                    excludeBackground: true
                )
                
                await MainActor.run {
                    segmentedImage = UIImage(cgImage: overlay.cgImage)
                }
            } else {
                await MainActor.run {
                    segmentedImage = image
                }
            }
        } catch {
            print("Failed to update overlay: \(error)")
        }
    }
    
    // MARK: - Segmentation
    
    private func runSegmentation() async {
        guard let image = loadedImage else { return }
        
        await MainActor.run {
            isRunning = true
            resultText = "Preprocessing image..."
        }
        
        do {
            // Step 1: Preprocess image using SwiftPixelUtils
            let preprocessStart = CFAbsoluteTimeGetCurrent()
            
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: .tfliteFloat,  // Float32 [0-1] in NHWC
                width: modelWidth,
                height: modelHeight
            )
            
            let preprocessTime = (CFAbsoluteTimeGetCurrent() - preprocessStart) * 1000
            
            await MainActor.run {
                resultText = "Loading model..."
            }
            
            // Step 2: Load TFLite model
            var modelPath: String? = Bundle.main.path(forResource: "deeplabv3_257_mv_gpu", ofType: "tflite", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "deeplabv3_257_mv_gpu", ofType: "tflite")
            }
            
            guard let finalModelPath = modelPath else {
                await MainActor.run {
                    isRunning = false
                    resultText = "❌ Model not found. Download deeplabv3_257_mv_gpu.tflite"
                }
                return
            }
            
            var options = Interpreter.Options()
            options.threadCount = 4
            
            let interpreter = try Interpreter(modelPath: finalModelPath, options: options)
            try interpreter.allocateTensors()
            
            await MainActor.run {
                resultText = "Running inference..."
            }
            
            // Step 3: Run inference
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            
            try interpreter.copy(modelInput.data, toInputAt: 0)
            try interpreter.invoke()
            
            let inferenceMs = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000
            
            // Step 4: Get output
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            await MainActor.run {
                resultText = "Processing segmentation output..."
            }
            
            // Step 5: Process output using SwiftPixelUtils SegmentationOutput
            let outputArray = outputData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float.self))
            }
            
            let result = try SegmentationOutput.process(
                floatOutput: outputArray,
                format: .logits(height: modelHeight, width: modelWidth, numClasses: numClasses),
                labels: .voc
            )
            
            // Cache result for overlay updates
            await MainActor.run {
                cachedResult = result
            }
            
            // Step 6: Create visualization
            let overlay = try Drawing.overlaySegmentation(
                on: .uiImage(image),
                segmentation: result,
                palette: .voc,
                alpha: CGFloat(overlayAlpha),
                excludeBackground: true
            )
            
            // Update UI
            await MainActor.run {
                segmentedImage = UIImage(cgImage: overlay.cgImage)
                classSummary = result.classSummary
                inferenceTime = inferenceMs
                isRunning = false
                
                let classCount = result.presentClasses.count
                resultText = """
                ✅ Segmentation complete
                • Preprocess: \(String(format: "%.1f", preprocessTime))ms
                • Inference: \(String(format: "%.1f", inferenceMs))ms
                • Post-process: \(String(format: "%.1f", result.processingTimeMs))ms
                • Classes detected: \(classCount)
                """
            }
            
        } catch {
            await MainActor.run {
                isRunning = false
                resultText = "❌ Error: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Preview

struct SegmentationView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            SegmentationView()
        }
    }
}
