//
//  ExecuTorchInferenceView.swift
//  SwiftPixelUtilsExampleApp
//
//  ExecuTorch inference demo using SwiftPixelUtils for preprocessing
//

import SwiftUI
import SwiftPixelUtils
import ExecuTorch

struct ExecuTorchInferenceView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dog", "car", "lion"]
    
    @State private var selectedImage: String = "dog"
    @State private var loadedImage: UIImage?
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Inference'"
    @State private var topPredictions: [(label: String, confidence: Float)] = []
    @State private var inferenceTime: Double = 0
    
    /// Load an image from the Resources folder in the bundle
    private func loadBundleImage(named name: String) -> UIImage? {
        // Try loading from bundle's Resources folder
        if let path = Bundle.main.path(forResource: name, ofType: "jpg", inDirectory: "Resources") {
            return UIImage(contentsOfFile: path)
        }
        // Fallback to standard bundle resource
        if let path = Bundle.main.path(forResource: name, ofType: "jpg") {
            return UIImage(contentsOfFile: path)
        }
        // Fallback to asset catalog
        return UIImage(named: name)
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Image Selection
                GroupBox("Select Image") {
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
                                            .frame(width: 80, height: 80)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 8)
                                                    .stroke(selectedImage == imageName ? Color.orange : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 80, height: 80)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption)
                                        .foregroundColor(selectedImage == imageName ? .orange : .primary)
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.vertical, 8)
                }
                
                // Selected Image Preview
                if let image = loadedImage {
                    GroupBox("Selected Image") {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }
                
                // Run Inference Button
                Button {
                    Task {
                        await runInference()
                    }
                } label: {
                    HStack {
                        if isRunning {
                            ProgressView()
                                .tint(.white)
                        }
                        Text(isRunning ? "Running..." : "Run Inference")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(isRunning || loadedImage == nil)
                
                // Results
                if !topPredictions.isEmpty {
                    GroupBox {
                        VStack(spacing: 0) {
                            ForEach(Array(topPredictions.enumerated()), id: \.offset) { index, prediction in
                                ExecuTorchPredictionRow(
                                    rank: index + 1,
                                    label: prediction.label,
                                    confidence: prediction.confidence,
                                    isTop: index == 0
                                )
                                
                                if index < topPredictions.count - 1 {
                                    Divider()
                                        .padding(.vertical, 8)
                                }
                            }
                        }
                        .padding(.vertical, 8)
                    } label: {
                        Label("Top Predictions", systemImage: "chart.bar.fill")
                            .font(.headline)
                    }
                    
                    GroupBox("Performance") {
                        HStack {
                            Text("Inference Time:")
                            Spacer()
                            Text(String(format: "%.2f ms", inferenceTime))
                                .fontWeight(.medium)
                        }
                    }
                }
                
                // Status/Error Text
                GroupBox("Status") {
                    Text(resultText)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .onAppear {
            loadSelectedImage()
        }
    }
    
    private func loadSelectedImage() {
        loadedImage = loadBundleImage(named: selectedImage)
        topPredictions = []
        resultText = "Tap 'Run Inference' to classify the image"
    }
    
    private func runInference() async {
        guard let image = loadedImage else {
            resultText = "❌ No image selected"
            return
        }
        
        isRunning = true
        resultText = "Preprocessing image..."
        topPredictions = []
        
        do {
            // Step 1: Preprocess image using SwiftPixelUtils simplified API
            // ExecuTorch uses NCHW layout with ImageNet normalization (PyTorch convention)
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: .execuTorch,  // NCHW, ImageNet normalization, Float32
                width: 224,
                height: 224
            )
            
            let preprocessTime = modelInput.processingTimeMs
            resultText = "Preprocessing: \(String(format: "%.2f", preprocessTime)) ms\nLoading model..."
            
            // Step 2: Load ExecuTorch model (.pte file) from Resources folder
            var modelPath: String? = Bundle.main.path(forResource: "mobilenet_v2_fp32_executorch", ofType: "pte", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "mobilenet_v2_fp32_executorch", ofType: "pte")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ Model not found. Please add mobilenet_v2_fp32_executorch.pte to the app bundle."
                isRunning = false
                return
            }
            
            // Step 3: Load ExecuTorch module and run inference
            // Verify file actually exists at path
            guard FileManager.default.fileExists(atPath: finalModelPath) else {
                resultText = "❌ Model file not found in app bundle."
                isRunning = false
                return
            }
            
            let module = Module(filePath: finalModelPath)
            
            // Explicitly load the forward method first
            do {
                try module.load("forward")
            } catch {
                resultText = "❌ Failed to load model: \(error.localizedDescription)"
                isRunning = false
                return
            }
            
            // Get tensor input directly from package (includes array + shape)
            guard var tensorInput = modelInput.floatTensorInput() else {
                throw NSError(
                    domain: "ExecuTorch",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to convert model input to tensor input"]
                )
            }
            
            // Create ExecuTorch tensor directly using package-provided array and shape
            let inputTensor = Tensor<Float>(&tensorInput.array, shape: tensorInput.shape)
            
            // Step 4: Run inference
            let startInference = CFAbsoluteTimeGetCurrent()
            
            let outputTensor = try Tensor<Float>(module.forward(inputTensor))
            
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInference) * 1000
            
            // Step 5: Get output as Float array and process using SwiftPixelUtils
            let outputArray = Array(outputTensor.scalars())
            
            // Step 6: Process classification output using package API (accepts [Float] directly)
            let classificationResult = try ClassificationOutput.process(
                floatOutput: outputArray,
                topK: 5,
                labels: .imagenet(hasBackgroundClass: false)  // Standard ImageNet 1000 classes
            )
            
            await MainActor.run {
                topPredictions = classificationResult.predictions.map { ($0.label, $0.confidence) }
                
                resultText = """
                ✅ Inference complete!
                
                Model: MobileNet V2 (ExecuTorch)
                Input: \(modelInput.width)×\(modelInput.height) RGB (\(modelInput.dataType))
                Shape: \(modelInput.shape)
                Preprocessing: \(String(format: "%.2f", preprocessTime)) ms
                Inference: \(String(format: "%.2f", inferenceTime)) ms
                Output Processing: \(String(format: "%.2f", classificationResult.processingTimeMs)) ms
                """
            }
            
        } catch {
            resultText = "❌ Error: \(error.localizedDescription)"
        }
        
        isRunning = false
    }
}

// MARK: - Prediction Row Component
struct ExecuTorchPredictionRow: View {
    let rank: Int
    let label: String
    let confidence: Float
    let isTop: Bool
    
    private var rankColor: Color {
        switch rank {
        case 1: return .orange
        case 2: return .gray
        case 3: return .brown
        default: return .secondary
        }
    }
    
    private var confidenceColor: Color {
        if confidence > 0.5 { return .green }
        if confidence > 0.2 { return .orange }
        if confidence > 0.1 { return .yellow }
        return .secondary
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                // Rank badge
                ZStack {
                    Circle()
                        .fill(isTop ? rankColor.opacity(0.2) : Color.gray.opacity(0.1))
                        .frame(width: 32, height: 32)
                    Text("\(rank)")
                        .font(.system(.subheadline, design: .rounded, weight: .bold))
                        .foregroundColor(isTop ? rankColor : .secondary)
                }
                
                // Label
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.system(.body, weight: isTop ? .semibold : .medium))
                        .foregroundColor(.primary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                // Confidence percentage
                Text(formatConfidence(confidence))
                    .font(.system(.title3, design: .rounded, weight: .bold))
                    .foregroundColor(confidenceColor)
            }
            
            // Confidence bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.15))
                    
                    // Filled portion
                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            LinearGradient(
                                colors: [confidenceColor.opacity(0.8), confidenceColor],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * CGFloat(min(confidence, 1.0)))
                }
            }
            .frame(height: 8)
        }
    }
    
    private func formatConfidence(_ value: Float) -> String {
        if value >= 0.01 {
            return String(format: "%.1f%%", value * 100)
        } else if value >= 0.001 {
            return String(format: "%.2f%%", value * 100)
        } else {
            return "<0.1%"
        }
    }
}

#Preview {
    NavigationStack {
        ExecuTorchInferenceView()
    }
}
