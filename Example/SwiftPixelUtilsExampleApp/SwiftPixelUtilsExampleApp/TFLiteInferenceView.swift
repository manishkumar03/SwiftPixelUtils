//
//  TFLiteInferenceView.swift
//  SwiftPixelUtilsExampleApp
//
//  TensorFlow Lite inference demo using SwiftPixelUtils for preprocessing
//

import SwiftUI
import SwiftPixelUtils
import TensorFlowLite

// MARK: - Task Type Enum
enum InferenceTask: String, CaseIterable {
    case classification = "Image Classification"
    case detection = "Object Detection"
    
    var icon: String {
        switch self {
        case .classification: return "photo.badge.checkmark"
        case .detection: return "viewfinder.rectangular"
        }
    }
    
    var modelName: String {
        switch self {
        case .classification: return "mobilenet_v2_quant_tflite"
        case .detection: return "yolov5s_fp16_tflite"
        }
    }
    
    var inputSize: Int {
        switch self {
        case .classification: return 224
        case .detection: return 320  // YOLOv5s from neso613 repo uses 320x320
        }
    }
}

struct TFLiteInferenceView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dog", "car", "lion"]
    
    @State private var selectedImage: String = "dog"
    @State private var loadedImage: UIImage?
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Inference'"
    @State private var topPredictions: [(label: String, confidence: Float)] = []
    @State private var detections: [ObjectDetection] = []
    @State private var inferenceTime: Double = 0
    @State private var selectedTask: InferenceTask = .classification
    
    var body: some View {
        List {
            // MARK: - Task Selection Section
            Section {
                ForEach(InferenceTask.allCases, id: \.self) { task in
                    Button {
                        selectedTask = task
                        clearResults()
                    } label: {
                        HStack {
                            Image(systemName: task.icon)
                                .font(.title2)
                                .foregroundColor(selectedTask == task ? .blue : .secondary)
                                .frame(width: 32)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text(task.rawValue)
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                Text(task == .classification ? "MobileNet V2" : "YOLOv5s")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            if selectedTask == task {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.blue)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            } header: {
                Text("Task")
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
                        await runInference()
                    }
                } label: {
                    HStack {
                        Spacer()
                        if isRunning {
                            ProgressView()
                                .padding(.trailing, 8)
                        }
                        Text(isRunning ? "Running..." : "Run \(selectedTask.rawValue)")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .disabled(isRunning || loadedImage == nil)
            }
            
            // MARK: - Results Section
            if selectedTask == .classification && !topPredictions.isEmpty {
                Section {
                    ForEach(Array(topPredictions.enumerated()), id: \.offset) { index, prediction in
                        ClassificationPredictionRow(
                            rank: index + 1,
                            label: prediction.label,
                            confidence: prediction.confidence
                        )
                    }
                } header: {
                    Label("Classification Results", systemImage: "chart.bar.fill")
                }
            }
            
            if selectedTask == .detection && !detections.isEmpty {
                Section {
                    ForEach(Array(detections.enumerated()), id: \.offset) { index, detection in
                        DetectionResultRow(
                            rank: index + 1,
                            label: detection.label,
                            confidence: detection.confidence,
                            box: detection.pixelBoundingBox.map { [Float($0.minX), Float($0.minY), Float($0.width), Float($0.height)] }
                        )
                    }
                } header: {
                    Label("Detected Objects (\(detections.count))", systemImage: "viewfinder.rectangular")
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
        }
        .listStyle(.insetGrouped)
        .onAppear {
            loadSelectedImage()
        }
    }
    
    private func clearResults() {
        topPredictions = []
        detections = []
        inferenceTime = 0
        resultText = "Tap 'Run \(selectedTask.rawValue)' to process the image"
    }
    
    private func loadSelectedImage() {
        loadedImage = loadBundleImage(named: selectedImage)
        clearResults()
    }
    
    private func runInference() async {
        guard let image = loadedImage else {
            resultText = "❌ No image selected"
            return
        }
        
        isRunning = true
        topPredictions = []
        detections = []
        
        switch selectedTask {
        case .classification:
            await runClassification(image: image)
        case .detection:
            await runDetection(image: image)
        }
        
        isRunning = false
    }
    
    // MARK: - Classification Inference
    private func runClassification(image: UIImage) async {
        resultText = "Preprocessing image..."
        
        do {
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: .tfliteQuantized,
                width: 224,
                height: 224
            )
            
            let inputData = modelInput.data
            let preprocessTime = modelInput.processingTimeMs
            
            resultText = "Loading model..."
            
            var modelPath: String? = Bundle.main.path(forResource: "mobilenet_v2_quant_tflite", ofType: "tflite", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "mobilenet_v2_quant_tflite", ofType: "tflite")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ Model not found"
                return
            }
            
            let interpreter = try Interpreter(modelPath: finalModelPath)
            try interpreter.allocateTensors()
            
            let startInference = CFAbsoluteTimeGetCurrent()
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInference) * 1000
            
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
    }
    
    // MARK: - Detection Inference
    private func runDetection(image: UIImage) async {
        resultText = "Loading YOLOv5 model..."
        
        do {
            var modelPath: String? = Bundle.main.path(forResource: "yolov5s_fp16_tflite", ofType: "tflite", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "yolov5s_fp16_tflite", ofType: "tflite")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ YOLOv5 model not found. Add yolov5s_fp16_tflite.tflite to Resources."
                return
            }
            
            var options = Interpreter.Options()
            options.threadCount = 4
            let interpreter = try Interpreter(modelPath: finalModelPath, options: options)
            try interpreter.allocateTensors()
            
            // Check input tensor type to determine the right preprocessing
            let inputTensor = try interpreter.input(at: 0)
            let inputType = inputTensor.dataType
            let expectedBytes = inputTensor.data.count
            
            // Get actual dimensions from model shape [batch, height, width, channels]
            let inputShape = inputTensor.shape
            let modelHeight = inputShape.dimensions[1]
            let modelWidth = inputShape.dimensions[2]
            let modelChannels = inputShape.dimensions[3]
            
            print("Model input - type: \(inputType), shape: \(inputShape), bytes: \(expectedBytes)")
            print("Model dimensions - \(modelWidth)x\(modelHeight)x\(modelChannels)")
            
            // Calculate expected sizes based on ACTUAL model dimensions
            let pixelCount = modelWidth * modelHeight * modelChannels
            let uint8Size = pixelCount * 1      // 1 byte per value
            let float32Size = pixelCount * 4    // 4 bytes per value
            
            print("Expected sizes - UInt8: \(uint8Size), Float32: \(float32Size)")
            
            resultText = "Preprocessing... (expects \(expectedBytes) bytes)"
            
            // Determine framework based on actual byte count AND data type
            let framework: MLFramework
            if expectedBytes == uint8Size && inputType == .uInt8 {
                framework = .tfliteQuantized  // Model expects UInt8 [0-255]
                print("Detected UInt8 input")
            } else if expectedBytes == float32Size || inputType == .float32 {
                framework = .tfliteFloat      // Model expects Float32 [0-1]
                print("Detected Float32 input")
            } else {
                // Default based on data type
                framework = (inputType == .uInt8) ? .tfliteQuantized : .tfliteFloat
                print("Fallback detection based on dataType: \(framework)")
            }
            
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: framework,
                width: modelWidth,
                height: modelHeight
            )
            
            let inputData = modelInput.data
            let preprocessTime = modelInput.processingTimeMs
            
            print("Preprocessed input - bytes: \(inputData.count), expected: \(expectedBytes)")
            
            // Verify sizes match
            guard inputData.count == expectedBytes else {
                resultText = """
                ❌ Input size mismatch!
                Got: \(inputData.count) bytes
                Expected: \(expectedBytes) bytes
                Model type: \(inputType)
                Framework: \(framework)
                Model size: \(modelWidth)x\(modelHeight)
                
                The model may need different preprocessing.
                """
                return
            }
            
            resultText = "Running inference... (shape: \(inputShape), type: \(inputType))"
            
            let startInference = CFAbsoluteTimeGetCurrent()
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInference) * 1000
            
            let outputTensor = try interpreter.output(at: 0)
            let outputShape = outputTensor.shape
            
            let outputData = outputTensor.data
            let floatOutput = outputData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float.self))
            }
            
            let detectionResult = try DetectionOutput.process(
                floatOutput: floatOutput,
                format: .yolov5(numClasses: 80),
                confidenceThreshold: 0.25,
                iouThreshold: 0.45,
                maxDetections: 20,
                labels: .coco,
                imageSize: CGSize(width: image.size.width, height: image.size.height),
                modelInputSize: CGSize(width: Double(modelWidth), height: Double(modelHeight))
            )
            
            await MainActor.run {
                detections = detectionResult.detections
                resultText = """
                ✅ Detection complete!
                Model: YOLOv5s (FP16)
                Output shape: \(outputShape)
                Raw detections: \(detectionResult.rawDetectionCount)
                After NMS: \(detectionResult.detections.count) objects
                Preprocess: \(String(format: "%.2f", preprocessTime)) ms
                Inference: \(String(format: "%.2f", inferenceTime)) ms
                """
            }
            
        } catch {
            resultText = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Preview
// Using ClassificationPredictionRow and DetectionResultRow from UIHelpers.swift

#Preview {
    NavigationStack {
        TFLiteInferenceView()
    }
}
