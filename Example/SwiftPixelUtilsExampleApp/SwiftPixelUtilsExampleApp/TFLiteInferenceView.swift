//
//  TFLiteInferenceView.swift
//  SwiftPixelUtilsExampleApp
//
//  TensorFlow Lite inference demo using SwiftPixelUtils for preprocessing
//

import SwiftUI
import SwiftPixelUtils
import TensorFlowLite

struct TFLiteInferenceView: View {
    // Sample images from Resources folder
    private let sampleImages = ["banana", "car", "elephant"]
    
    @State private var selectedImage: String = "banana"
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
                                                    .stroke(selectedImage == imageName ? Color.blue : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 80, height: 80)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption)
                                        .foregroundColor(selectedImage == imageName ? .blue : .primary)
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
                .disabled(isRunning || loadedImage == nil)
                
                // Results
                if !topPredictions.isEmpty {
                    GroupBox {
                        VStack(spacing: 0) {
                            ForEach(Array(topPredictions.enumerated()), id: \.offset) { index, prediction in
                                PredictionRow(
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
        .navigationTitle("TFLite Inference")
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
            // Step 1: Preprocess image using SwiftPixelUtils
            // MobileNet v2 quantized expects:
            // - 224x224 RGB
            // - UInt8 values (0-255) - raw pixel values for quantized model
            let startPreprocess = CFAbsoluteTimeGetCurrent()
            
            let preprocessResult = try await PixelExtractor.getPixelData(
                source: .uiImage(image),
                options: PixelDataOptions(
                    colorFormat: .rgb,
                    resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                    normalization: .raw,  // Keep as 0-255 for quantized model
                    dataLayout: .nhwc,    // TensorFlow convention [1, 224, 224, 3]
                    outputFormat: .float32Array
                )
            )
            
            // Convert float (0-255) to UInt8 for quantized model input
            let uint8Data = preprocessResult.data.map { UInt8(min(255, max(0, $0))) }
            let inputData = Data(uint8Data)
            
            let preprocessTime = (CFAbsoluteTimeGetCurrent() - startPreprocess) * 1000
            resultText = "Preprocessing: \(String(format: "%.2f", preprocessTime)) ms\nLoading model..."
            
            // Step 2: Load TFLite model - try multiple locations
            var modelPath: String? = Bundle.main.path(forResource: "mobilenet_v2_1.0_224_quant", ofType: "tflite", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "mobilenet_v2_1.0_224_quant", ofType: "tflite")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ Model file not found in bundle. Please add mobilenet_v2_1.0_224_quant.tflite to the Resources folder."
                isRunning = false
                return
            }
            
            let interpreter = try Interpreter(modelPath: finalModelPath)
            try interpreter.allocateTensors()
            
            // Step 3: Run inference
            let startInference = CFAbsoluteTimeGetCurrent()
            
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInference) * 1000
            
            // Step 4: Get output tensor and process results
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Output is UInt8 quantized probabilities
            let outputArray = [UInt8](outputData)
            
            // Dequantize output and apply softmax to get probabilities
            let logits: [Float]
            if let quantParams = outputTensor.quantizationParameters {
                logits = outputArray.map { Float(Int($0) - quantParams.zeroPoint) * quantParams.scale }
            } else {
                logits = outputArray.map { Float($0) }
            }
            
            // Apply softmax to convert logits to probabilities
            let probabilities = softmax(logits)
            
            // Step 5: Get top predictions using SwiftPixelUtils LabelDatabase
            // Note: The model has 1001 classes (background + 1000 ImageNet classes)
            let topK = getTopK(probabilities: probabilities, k: 5)
            
            await MainActor.run {
                topPredictions = topK.map { (index, confidence) in
                    // Index 0 is background, so ImageNet classes start at 1
                    let label = LabelDatabase.getLabel(index > 0 ? index - 1 : index, dataset: .imagenet) ?? "Unknown (\(index))"
                    return (label: label, confidence: confidence)
                }
                
                resultText = """
                ✅ Inference complete!
                
                Model: MobileNet V2 (Quantized)
                Input: \(preprocessResult.width)×\(preprocessResult.height) RGB
                Output: \(outputArray.count) classes
                Preprocessing: \(String(format: "%.2f", preprocessTime)) ms
                Inference: \(String(format: "%.2f", inferenceTime)) ms
                """
            }
            
        } catch {
            resultText = "❌ Error: \(error.localizedDescription)"
        }
        
        isRunning = false
    }
    
    /// Get top-K predictions from probability array
    private func getTopK(probabilities: [Float], k: Int) -> [(index: Int, confidence: Float)] {
        let indexed = probabilities.enumerated().map { (index: $0.offset, confidence: $0.element) }
        let sorted = indexed.sorted { $0.confidence > $1.confidence }
        return Array(sorted.prefix(k))
    }
    
    /// Apply softmax to convert logits to probabilities
    private func softmax(_ logits: [Float]) -> [Float] {
        // Find max for numerical stability
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
}

// MARK: - Prediction Row Component
struct PredictionRow: View {
    let rank: Int
    let label: String
    let confidence: Float
    let isTop: Bool
    
    private var rankColor: Color {
        switch rank {
        case 1: return .yellow
        case 2: return .gray
        case 3: return .orange
        default: return .secondary
        }
    }
    
    private var confidenceColor: Color {
        if confidence > 0.5 { return .green }
        if confidence > 0.2 { return .blue }
        if confidence > 0.1 { return .orange }
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
        TFLiteInferenceView()
    }
}
