//
//  YOLODetectionView.swift
//  SwiftPixelUtilsExampleApp
//
//  Object Detection using YOLOv5 with SwiftPixelUtils preprocessing
//
//  This demonstrates how to use SwiftPixelUtils for preprocessing images
//  for YOLO object detection models and post-processing with DetectionOutput.
//

import SwiftUI
import SwiftPixelUtils
import TensorFlowLite

/// # YOLOv5 Object Detection
///
/// This view demonstrates object detection using YOLOv5 with TensorFlow Lite
/// and SwiftPixelUtils for preprocessing and post-processing.
///
/// ## Model Details
/// - **Architecture**: YOLOv5s (small)
/// - **Input**: 320×320×3 RGB (Float32, [0-1] normalized)
/// - **Output**: 80 COCO classes
/// - **Quantization**: FP16 (Float16 weights, Float32 inference)
///
/// ## SwiftPixelUtils Integration
///
/// ### Preprocessing
/// ```swift
/// // Preprocess with automatic model detection
/// let modelInput = try await PixelExtractor.getModelInput(
///     source: .uiImage(image),
///     framework: .tfliteFloat,  // Float32 [0-1] in NHWC
///     width: modelWidth,
///     height: modelHeight
/// )
/// ```
///
/// ### Post-processing with DetectionOutput
/// ```swift
/// let result = try DetectionOutput.process(
///     floatOutput: outputArray,
///     format: .yolov5(numClasses: 80),
///     confidenceThreshold: 0.25,
///     iouThreshold: 0.45,
///     maxDetections: 20,
///     labels: .coco,
///     imageSize: originalImageSize,
///     modelInputSize: CGSize(width: 320, height: 320)
/// )
/// ```
///
/// ## YOLO Output Format
///
/// YOLOv5 output is a tensor of shape `[1, N, 85]` where:
/// - N = number of anchor predictions (e.g., 25200 for 320×320)
/// - 85 = 4 (bbox) + 1 (objectness) + 80 (class scores)
///
/// The bounding box format is `[cx, cy, w, h]` (center-based, normalized).
struct YOLODetectionView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dogBoat", "street", "twoBuses"]
    
    @State private var selectedImage: String = "dog"
    @State private var loadedImage: UIImage?
    @State private var annotatedImage: UIImage?  // Image with bounding boxes drawn
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Detection'"
    @State private var detections: [ObjectDetection] = []
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
                    Image(systemName: "viewfinder.rectangular")
                        .font(.title)
                        .foregroundColor(.green)
                    VStack(alignment: .leading) {
                        Text("YOLOv5s")
                            .font(.headline)
                        Text("FP16 • 320×320 • 80 COCO Classes")
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
                                                    .stroke(selectedImage == imageName ? Color.green : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 70, height: 70)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption2)
                                        .foregroundColor(selectedImage == imageName ? .green : .primary)
                                }
                            }
                            .buttonStyle(.plain)
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
                        await runDetection()
                    }
                } label: {
                    HStack {
                        Spacer()
                        if isRunning {
                            ProgressView()
                                .padding(.trailing, 8)
                        }
                        Text(isRunning ? "Running..." : "Run Detection")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .disabled(isRunning || loadedImage == nil)
            }
            
            // MARK: - Results Section
            if !detections.isEmpty {
                // Annotated image with bounding boxes
                if let annotated = annotatedImage {
                    Section {
                        HStack {
                            Spacer()
                            Image(uiImage: annotated)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            Spacer()
                        }
                    } header: {
                        Label("Detection Results", systemImage: "viewfinder.rectangular")
                    }
                }
                
                // Detection list
                Section {
                    ForEach(Array(detections.enumerated()), id: \.offset) { index, detection in
                        YOLODetectionRow(detection: detection, rank: index + 1)
                    }
                } header: {
                    Label("Detected Objects (\(detections.count))", systemImage: "list.bullet.rectangle")
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
                            .foregroundColor(.green)
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
                    Text("SwiftPixelUtils Post-processing:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("""
                    let result = try DetectionOutput.process(
                        floatOutput: outputArray,
                        format: .yolov5(numClasses: 80),
                        confidenceThreshold: 0.25,
                        iouThreshold: 0.45,
                        labels: .coco,
                        imageSize: image.size,
                        modelInputSize: CGSize(width: 320, height: 320)
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
        .navigationTitle("YOLO Detection")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadSelectedImage()
        }
    }
    
    private func loadSelectedImage() {
        loadedImage = loadBundleImage(named: selectedImage)
        annotatedImage = nil
        detections = []
        inferenceTime = 0
        resultText = "Tap 'Run Detection' to process the image"
    }
    
    // MARK: - Detection Inference
    private func runDetection() async {
        guard let image = loadedImage else {
            resultText = "❌ No image selected"
            return
        }
        
        isRunning = true
        detections = []
        resultText = "Loading YOLOv5 model..."
        
        do {
            // Step 1: Load TFLite model
            var modelPath: String? = Bundle.main.path(forResource: "yolov5s_fp16_tflite", ofType: "tflite", inDirectory: "Resources")
            if modelPath == nil {
                modelPath = Bundle.main.path(forResource: "yolov5s_fp16_tflite", ofType: "tflite")
            }
            
            guard let finalModelPath = modelPath else {
                resultText = "❌ YOLOv5 model not found. Add yolov5s_fp16_tflite.tflite to Resources."
                isRunning = false
                return
            }
            
            var options = Interpreter.Options()
            options.threadCount = 4
            let interpreter = try Interpreter(modelPath: finalModelPath, options: options)
            try interpreter.allocateTensors()
            
            // Step 2: Get model input specifications
            let inputTensor = try interpreter.input(at: 0)
            let inputType = inputTensor.dataType
            let expectedBytes = inputTensor.data.count
            
            // Get actual dimensions from model shape [batch, height, width, channels]
            let inputShape = inputTensor.shape
            let modelHeight = inputShape.dimensions[1]
            let modelWidth = inputShape.dimensions[2]
            let modelChannels = inputShape.dimensions[3]
            
            print("Model input - type: \(inputType), shape: \(inputShape), bytes: \(expectedBytes)")
            
            // Calculate expected sizes based on actual model dimensions
            let pixelCount = modelWidth * modelHeight * modelChannels
            let uint8Size = pixelCount * 1
            let float32Size = pixelCount * 4
            
            resultText = "Preprocessing... (\(modelWidth)×\(modelHeight))"
            
            // Determine framework based on byte count and data type
            let framework: MLFramework
            if expectedBytes == uint8Size && inputType == .uInt8 {
                framework = .tfliteQuantized
            } else if expectedBytes == float32Size || inputType == .float32 {
                framework = .tfliteFloat
            } else {
                framework = (inputType == .uInt8) ? .tfliteQuantized : .tfliteFloat
            }
            
            // Step 3: Preprocess image using SwiftPixelUtils
            // ┌─────────────────────────────────────────────────────────────────────┐
            // │ RESIZE STRATEGY FOR YOLO                                            │
            // ├─────────────────────────────────────────────────────────────────────┤
            // │ We use .stretch here which directly maps image coordinates to model │
            // │ input coordinates without preserving aspect ratio.                  │
            // │                                                                     │
            // │ Alternative strategies:                                             │
            // │ • .letterbox - Preserves aspect ratio with gray padding (114,114,114)│
            // │   Requires accounting for padding offset when mapping coords back   │
            // │ • .cover - Crops to fill, may lose objects near edges               │
            // │ • .contain - Fits within bounds with padding                        │
            // │                                                                     │
            // │ With .stretch, coordinate mapping is straightforward:               │
            // │   normalized_coord * image_dimension = pixel_coord                  │
            // └─────────────────────────────────────────────────────────────────────┘
            let modelInput = try await PixelExtractor.getModelInput(
                source: .uiImage(image),
                framework: framework,
                width: modelWidth,
                height: modelHeight,
                resizeStrategy: .stretch
            )
            
            let inputData = modelInput.data
            let preprocessTime = modelInput.processingTimeMs
            
            guard inputData.count == expectedBytes else {
                resultText = """
                ❌ Input size mismatch!
                Got: \(inputData.count) bytes
                Expected: \(expectedBytes) bytes
                """
                isRunning = false
                return
            }
            
            // Step 4: Run inference
            resultText = "Running inference..."
            
            let startInference = CFAbsoluteTimeGetCurrent()
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInference) * 1000
            
            // Step 5: Get output and convert to float array
            // ┌─────────────────────────────────────────────────────────────────────┐
            // │ YOLO OUTPUT FORMAT                                                  │
            // ├─────────────────────────────────────────────────────────────────────┤
            // │ YOLOv5 TFLite output shape: [1, N, 85]                              │
            // │ Where 85 = 4 (cx, cy, w, h) + 1 (objectness) + 80 (COCO classes)   │
            // │                                                                     │
            // │ This particular model outputs NORMALIZED coordinates (0-1 range),  │
            // │ not pixel coordinates (0-modelInputSize). This is important for    │
            // │ the coordinate transformation in Step 7.                           │
            // └─────────────────────────────────────────────────────────────────────┘
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            let floatOutput = outputData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float.self))
            }
            
            // Step 6: Post-process using SwiftPixelUtils DetectionOutput
            // ┌─────────────────────────────────────────────────────────────────────┐
            // │ DETECTION POST-PROCESSING                                           │
            // ├─────────────────────────────────────────────────────────────────────┤
            // │ DetectionOutput.process() handles:                                  │
            // │ • Parsing YOLOv5 output format [N, 85] (4 box + 1 obj + 80 classes)│
            // │ • Converting center-width-height to corner coordinates              │
            // │ • Confidence filtering (objectness × class_score > threshold)       │
            // │ • Non-Maximum Suppression (NMS) to remove overlapping boxes         │
            // │ • Label mapping using COCO 80-class vocabulary                      │
            // │                                                                     │
            // │ IMPORTANT: This TFLite YOLOv5 model outputs NORMALIZED coordinates │
            // │ (0-1 range), so we use outputCoordinateSpace: .normalized to tell  │
            // │ the package to NOT divide by modelInputSize.                       │
            // └─────────────────────────────────────────────────────────────────────┘
            let detectionResult = try DetectionOutput.process(
                floatOutput: floatOutput,
                format: .yolov5(numClasses: 80),
                confidenceThreshold: 0.25,
                iouThreshold: 0.45,
                maxDetections: 20,
                labels: .coco,
                imageSize: CGSize(width: image.size.width, height: image.size.height),
                modelInputSize: CGSize(width: Double(modelWidth), height: Double(modelHeight)),
                outputCoordinateSpace: .normalized  // TFLite YOLO outputs 0-1 normalized coords
            )
            
            // Step 7: Draw bounding boxes on the image
            var finalAnnotatedImage: UIImage? = nil
            if !detectionResult.detections.isEmpty {
                // ┌─────────────────────────────────────────────────────────────────────┐
                // │ CONVERT DETECTIONS TO DRAWABLE BOXES                               │
                // ├─────────────────────────────────────────────────────────────────────┤
                // │ DetectionResult.toDrawableBoxes() provides a one-line conversion   │
                // │ from ObjectDetection array to DrawableBox array:                   │
                // │ • Uses pixelBoundingBox (pre-calculated from imageSize in process) │
                // │ • Converts to [x1, y1, x2, y2] corner format                       │
                // │ • Applies DetectionColorPalette for per-class colors               │
                // │ • Returns array ready for Drawing.drawBoxes()                      │
                // └─────────────────────────────────────────────────────────────────────┘
                let boxes = detectionResult.toDrawableBoxes(imageSize: image.size)
                
                // Step 8: Draw boxes using SwiftPixelUtils Drawing API
                // ┌─────────────────────────────────────────────────────────────────────┐
                // │ VISUALIZATION WITH Drawing.drawBoxes()                             │
                // ├─────────────────────────────────────────────────────────────────────┤
                // │ Drawing.drawBoxes() renders bounding boxes on the image using      │
                // │ CoreGraphics. Features:                                            │
                // │ • Box coordinates in [x1, y1, x2, y2] corner format                │
                // │ • Per-box colors based on class index                              │
                // │ • Labels with confidence scores (rendered via CoreText)            │
                // │ • Configurable line width and label styling                        │
                // │                                                                     │
                // │ Note: UIImage must be created with original scale and orientation  │
                // │ to prevent display distortion on Retina screens.                   │
                // └─────────────────────────────────────────────────────────────────────┘
                do {
                    let drawingResult = try Drawing.drawBoxes(
                        on: .uiImage(image),
                        boxes: boxes,
                        options: BoxDrawingOptions(
                            lineWidth: 4.0,
                            drawLabels: true,
                            drawScores: true,
                            defaultColor: (255, 0, 0, 255)
                        )
                    )
                    
                    // Preserve original image scale and orientation for correct display
                    finalAnnotatedImage = UIImage(
                        cgImage: drawingResult.cgImage,
                        scale: image.scale,
                        orientation: image.imageOrientation
                    )
                } catch {
                    print("Failed to draw boxes: \(error)")
                }
            }
            
            let drawingStatus = finalAnnotatedImage != nil ? "Boxes drawn ✓" : "No boxes drawn"
            
            await MainActor.run {
                detections = detectionResult.detections
                annotatedImage = finalAnnotatedImage
                resultText = """
                ✅ Detection complete!
                Model: YOLOv5s (FP16)
                Input: \(modelWidth)×\(modelHeight) RGB
                Raw detections: \(detectionResult.rawDetectionCount)
                After NMS: \(detectionResult.detections.count) objects
                Preprocess: \(String(format: "%.2f", preprocessTime)) ms
                Inference: \(String(format: "%.2f", inferenceTime)) ms
                \(drawingStatus)
                """
            }
            
        } catch {
            resultText = "❌ Error: \(error.localizedDescription)"
        }
        
        isRunning = false
    }
}

// MARK: - Detection Row Component
struct YOLODetectionRow: View {
    let detection: ObjectDetection
    let rank: Int
    
    private var confidenceColor: Color {
        if detection.confidence > 0.7 { return .green }
        if detection.confidence > 0.5 { return .blue }
        if detection.confidence > 0.3 { return .orange }
        return .red
    }
    
    var body: some View {
        HStack(spacing: 12) {
            // Rank badge
            ZStack {
                Circle()
                    .fill(confidenceColor.opacity(0.2))
                    .frame(width: 32, height: 32)
                Text("\(rank)")
                    .font(.system(.subheadline, design: .rounded, weight: .bold))
                    .foregroundColor(confidenceColor)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(detection.label)
                    .font(.headline)
                
                // Bounding box info (use pixelBoundingBox if available, otherwise boundingBox)
                if let pixelBox = detection.pixelBoundingBox {
                    Text(String(format: "Box: (%.0f, %.0f) %.0f×%.0f", pixelBox.minX, pixelBox.minY, pixelBox.width, pixelBox.height))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                } else {
                    let box = detection.boundingBox
                    Text(String(format: "Box: (%.2f, %.2f) %.2f×%.2f", box.minX, box.minY, box.width, box.height))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            Text(String(format: "%.0f%%", detection.confidence * 100))
                .font(.system(.title3, design: .rounded, weight: .bold))
                .foregroundColor(confidenceColor)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    NavigationStack {
        YOLODetectionView()
    }
}
