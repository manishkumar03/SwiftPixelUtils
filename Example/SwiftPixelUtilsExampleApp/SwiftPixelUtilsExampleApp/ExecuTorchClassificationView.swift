//
//  ExecuTorchClassificationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Image Classification using ExecuTorch with SwiftPixelUtils preprocessing
//
//  This demonstrates how to use SwiftPixelUtils for preprocessing images
//  for ExecuTorch/PyTorch models (NCHW layout with ImageNet normalization).
//

import SwiftUI
import SwiftPixelUtils
import ExecuTorch

/// # ExecuTorch Image Classification
///
/// This view demonstrates image classification using ExecuTorch with a
/// MobileNet V2 model exported from PyTorch.
///
/// ## Model Details
/// - **Architecture**: MobileNet V2
/// - **Input**: 224×224×3 RGB (Float32, ImageNet normalized)
/// - **Layout**: NCHW (Channels first - PyTorch convention)
/// - **Output**: 1000 classes (ImageNet)
///
/// ## SwiftPixelUtils Integration
///
/// ```swift
/// // Preprocess image for ExecuTorch (PyTorch-style preprocessing)
/// let modelInput = try await PixelExtractor.getModelInput(
///     source: .uiImage(image),
///     framework: .execuTorch,  // NCHW, ImageNet normalization, Float32
///     width: 224,
///     height: 224
/// )
///
/// // Get tensor input directly (array + shape ready for ExecuTorch)
/// var tensorInput = modelInput.floatTensorInput()!
/// let tensor = Tensor<Float>(&tensorInput.array, shape: tensorInput.shape)
/// ```
///
/// ## Key Differences from TFLite
///
/// | Aspect | TFLite | ExecuTorch |
/// |--------|--------|------------|
/// | Layout | NHWC | NCHW |
/// | Normalization | Raw [0-255] or [0-1] | ImageNet mean/std |
/// | Input Type | Usually UInt8 | Usually Float32 |
/// | Shape | [1, 224, 224, 3] | [1, 3, 224, 224] |
struct ExecuTorchClassificationView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dog", "car", "lion"]
    
    @State private var selectedImage: String = "dog"
    @State private var loadedImage: UIImage?
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Classification'"
    @State private var topPredictions: [(label: String, confidence: Float)] = []
    @State private var inferenceTime: Double = 0
    
    var body: some View {
        List {
            // MARK: - Model Info Section
            Section {
                HStack {
                    Image(systemName: "bolt.fill")
                        .font(.title)
                        .foregroundColor(.orange)
                    VStack(alignment: .leading) {
                        Text("MobileNet V2")
                            .font(.headline)
                        Text("Float32 • NCHW • ImageNet Norm")
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
                                                    .stroke(selectedImage == imageName ? Color.orange : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 70, height: 70)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption2)
                                        .foregroundColor(selectedImage == imageName ? .orange : .primary)
                                }
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("executorch-image-\(imageName)")
                        }
                    }
                    .padding(.vertical, 4)
                }
                
                // Selected Image Preview
                if let image = loadedImage {
                    HStack {
                        Spacer()
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        Spacer()
                    }
                }
            } header: {
                Text("Input Image")
            }
            
            // MARK: - Run Inference Section
            Section {
                Button {
                    Task {
                        await runClassification()
                    }
                } label: {
                    HStack {
                        Spacer()
                        if isRunning {
                            ProgressView()
                                .padding(.trailing, 8)
                        }
                        Text(isRunning ? "Running..." : "Run Classification")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .disabled(isRunning || loadedImage == nil)
                .accessibilityIdentifier("executorch-run-classification-button")
            }
            
            // MARK: - Results Section
            if !topPredictions.isEmpty {
                Section {
                    ForEach(Array(topPredictions.enumerated()), id: \.offset) { index, prediction in
                        ClassificationPredictionRow(
                            rank: index + 1,
                            label: prediction.label,
                            confidence: prediction.confidence
                        )
                        .accessibilityIdentifier("executorch-prediction-row-\(index)")
                    }
                } header: {
                    Label("Classification Results", systemImage: "chart.bar.fill")
                        .accessibilityIdentifier("executorch-classification-results-header")
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
                            .foregroundColor(.orange)
                            .accessibilityIdentifier("executorch-inference-time")
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
            
            // MARK: - Code Example Section
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("SwiftPixelUtils Usage:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("""
                    let input = try await PixelExtractor.getModelInput(
                        source: .uiImage(image),
                        framework: .execuTorch,
                        width: 224, height: 224
                    )
                    var tensorInput = input.floatTensorInput()!
                    let tensor = Tensor<Float>(
                        &tensorInput.array, 
                        shape: tensorInput.shape
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
        .navigationTitle("ExecuTorch Classification")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadSelectedImage()
        }
    }
    
    private func loadSelectedImage() {
        loadedImage = loadBundleImage(named: selectedImage)
        topPredictions = []
        inferenceTime = 0
        resultText = "Tap 'Run Classification' to process the image"
    }
    
    // MARK: - Classification Inference
    private func runClassification() async {
        guard let image = loadedImage else {
            resultText = "❌ No image selected"
            return
        }
        
        isRunning = true
        topPredictions = []
        resultText = "Preprocessing image..."
        
        do {
            // Step 1: Preprocess image using SwiftPixelUtils
            // .execuTorch gives us Float32 with ImageNet normalization in NCHW layout
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: .execuTorch,  // NCHW, ImageNet normalization, Float32
                width: 224,
                height: 224
            )
            
            let preprocessTime = modelInput.processingTimeMs
            resultText = "Loading model..."
            
            // Step 2: Load ExecuTorch model (.pte file)
            var modelPath: String? = Bundle.main.path(forResource: "mobilenet_v2_fp32_executorch", ofType: "pte", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "mobilenet_v2_fp32_executorch", ofType: "pte")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ Model not found. Add mobilenet_v2_fp32_executorch.pte to Resources."
                isRunning = false
                return
            }
            
            guard FileManager.default.fileExists(atPath: finalModelPath) else {
                resultText = "❌ Model file not found in app bundle."
                isRunning = false
                return
            }
            
            let module = Module(filePath: finalModelPath)
            
            // Load the forward method
            do {
                try module.load("forward")
            } catch {
                resultText = "❌ Failed to load model: \(error.localizedDescription)"
                isRunning = false
                return
            }
            
            // Step 3: Create ExecuTorch tensor from SwiftPixelUtils output
            // floatTensorInput() returns (array, shape) ready for Tensor initializer
            guard var tensorInput = modelInput.floatTensorInput() else {
                resultText = "❌ Failed to convert model input to tensor"
                isRunning = false
                return
            }
            
            let inputTensor = Tensor<Float>(&tensorInput.array, shape: tensorInput.shape)
            
            // Step 4: Run inference
            let startInference = CFAbsoluteTimeGetCurrent()
            let outputTensor = try Tensor<Float>(module.forward(inputTensor))
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInference) * 1000
            
            // Step 5: Process output using SwiftPixelUtils ClassificationOutput
            let outputArray = Array(outputTensor.scalars())
            
            let classificationResult = try ClassificationOutput.process(
                floatOutput: outputArray,
                topK: 5,
                labels: .imagenet(hasBackgroundClass: false)  // Standard ImageNet 1000 classes
            )
            
            await MainActor.run {
                topPredictions = classificationResult.predictions.map { ($0.label, $0.confidence) }
                resultText = """
                ✅ Classification complete!
                Model: MobileNet V2 (ExecuTorch)
                Input: \(modelInput.width)×\(modelInput.height) • \(modelInput.shape)
                Preprocess: \(String(format: "%.2f", preprocessTime)) ms
                Inference: \(String(format: "%.2f", inferenceTime)) ms
                """
            }
            
        } catch {
            resultText = "❌ Error: \(error.localizedDescription)"
        }
        
        isRunning = false
    }
}

// MARK: - Preview
// Using ClassificationPredictionRow from UIHelpers.swift

#Preview {
    NavigationStack {
        ExecuTorchClassificationView()
    }
}
