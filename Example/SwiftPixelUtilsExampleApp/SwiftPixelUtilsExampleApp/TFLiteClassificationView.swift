//
//  TFLiteClassificationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Image Classification using TensorFlow Lite with SwiftPixelUtils preprocessing
//
//  This demonstrates how to use SwiftPixelUtils for preprocessing images
//  for TensorFlow Lite models (quantized UInt8 input).
//

import SwiftUI
import SwiftPixelUtils
import TensorFlowLite

/// # TFLite Image Classification
///
/// This view demonstrates image classification using TensorFlow Lite with a
/// quantized MobileNet V2 model.
///
/// ## Model Details
/// - **Architecture**: MobileNet V2
/// - **Input**: 224×224×3 RGB (UInt8, 0-255)
/// - **Output**: 1001 classes (ImageNet + background)
/// - **Quantization**: Full integer (INT8)
///
/// ## SwiftPixelUtils Integration
///
/// ```swift
/// // Preprocess image for TFLite quantized model
/// let modelInput = try await PixelExtractor.getModelInput(
///     source: .uiImage(image),
///     framework: .tfliteQuantized,  // UInt8, NHWC, raw [0-255]
///     width: 224,
///     height: 224
/// )
///
/// // Use the raw Data directly with TFLite interpreter
/// try interpreter.copy(modelInput.data, toInputAt: 0)
/// ```
///
/// ## Post-processing with ClassificationOutput
///
/// ```swift
/// let result = try ClassificationOutput.process(
///     outputData: outputTensor.data,
///     quantization: .uint8(scale: scale, zeroPoint: zeroPoint),
///     topK: 5,
///     labels: .imagenet(hasBackgroundClass: true)
/// )
/// ```
struct TFLiteClassificationView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dog", "car", "lion"]
    
    @State private var selectedImage: String = "dog"
    @State private var loadedImage: UIImage?
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Classification'"
    @State private var topPredictions: [(label: String, confidence: Float)] = []
    @State private var inferenceTime: Double = 0
    
    /// Load an image from the Resources folder in the bundle
    private func loadBundleImage(named name: String) -> UIImage? {
        if let path = Bundle.main.path(forResource: name, ofType: "jpg", inDirectory: "Resources") {
            return UIImage(contentsOfFile: path)
        }
        if let path = Bundle.main.path(forResource: name, ofType: "jpg") {
            return UIImage(contentsOfFile: path)
        }
        return UIImage(named: name)
    }
    
    var body: some View {
        List {
            // MARK: - Model Info Section
            Section {
                HStack {
                    Image(systemName: "t.square.fill")
                        .font(.title)
                        .foregroundColor(.blue)
                    VStack(alignment: .leading) {
                        Text("MobileNet V2")
                            .font(.headline)
                        Text("Quantized INT8 • 224×224 • ImageNet")
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
                                                    .stroke(selectedImage == imageName ? Color.blue : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 70, height: 70)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption2)
                                        .foregroundColor(selectedImage == imageName ? .blue : .primary)
                                }
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("tflite-image-\(imageName)")
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
                .accessibilityIdentifier("tflite-run-classification-button")
            }
            
            // MARK: - Results Section
            if !topPredictions.isEmpty {
                Section {
                    ForEach(Array(topPredictions.enumerated()), id: \.offset) { index, prediction in
                        TFLitePredictionRow(
                            rank: index + 1,
                            label: prediction.label,
                            confidence: prediction.confidence,
                            isTop: index == 0
                        )
                        .accessibilityIdentifier("tflite-prediction-row-\(index)")
                    }
                } header: {
                    Label("Classification Results", systemImage: "chart.bar.fill")
                        .accessibilityIdentifier("tflite-classification-results-header")
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
                            .foregroundColor(.blue)
                            .accessibilityIdentifier("tflite-inference-time")
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
                        framework: .tfliteQuantized,
                        width: 224, height: 224
                    )
                    try interpreter.copy(input.data, toInputAt: 0)
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
        .navigationTitle("TFLite Classification")
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
            // .tfliteQuantized gives us UInt8 [0-255] in NHWC layout
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: .tfliteQuantized,
                width: 224,
                height: 224
            )
            
            let inputData = modelInput.data
            let preprocessTime = modelInput.processingTimeMs
            
            resultText = "Loading model..."
            
            // Step 2: Load TFLite model
            var modelPath: String? = Bundle.main.path(forResource: "mobilenet_v2_quant_tflite", ofType: "tflite", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "mobilenet_v2_quant_tflite", ofType: "tflite")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ Model not found. Add mobilenet_v2_quant_tflite.tflite to Resources."
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
            
            // Step 4: Process output using SwiftPixelUtils ClassificationOutput
            let outputTensor = try interpreter.output(at: 0)
            let quantParams = outputTensor.quantizationParameters
            
            let classificationResult = try ClassificationOutput.process(
                outputData: outputTensor.data,
                quantization: quantParams != nil
                    ? .uint8(scale: quantParams!.scale, zeroPoint: quantParams!.zeroPoint)
                    : .none,
                topK: 5,
                labels: .imagenet(hasBackgroundClass: true)
            )
            
            await MainActor.run {
                topPredictions = classificationResult.predictions.map { ($0.label, $0.confidence) }
                resultText = """
                ✅ Classification complete!
                Model: MobileNet V2 (Quantized)
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

// MARK: - Prediction Row Component
struct TFLitePredictionRow: View {
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
                ZStack {
                    Circle()
                        .fill(isTop ? rankColor.opacity(0.2) : Color.gray.opacity(0.1))
                        .frame(width: 32, height: 32)
                    Text("\(rank)")
                        .font(.system(.subheadline, design: .rounded, weight: .bold))
                        .foregroundColor(isTop ? rankColor : .secondary)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(.system(.body, weight: isTop ? .semibold : .medium))
                        .foregroundColor(.primary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                Text(formatConfidence(confidence))
                    .font(.system(.title3, design: .rounded, weight: .bold))
                    .foregroundColor(confidenceColor)
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.15))
                    
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
        TFLiteClassificationView()
    }
}
