//
//  DepthEstimationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Depth Estimation using Depth Anything with SwiftPixelUtils
//
//  This demonstrates how to use SwiftPixelUtils for preprocessing images
//  for depth estimation models and post-processing with DepthEstimationOutput.
//

import SwiftUI
import SwiftPixelUtils
import CoreML
import Vision

/// # Depth Anything Depth Estimation
///
/// This view demonstrates monocular depth estimation using Apple's Depth Anything
/// Core ML model with SwiftPixelUtils for post-processing and visualization.
///
/// ## Model Details
/// - **Architecture**: Depth Anything Small (DINOv2 + DPT decoder)
/// - **Input**: 518×518×3 RGB (Float32, normalized)
/// - **Output**: 518×518 depth map (relative inverse depth)
/// - **Quantization**: F16P6 (6-bit palettized, ~18MB)
///
/// ## SwiftPixelUtils Integration
///
/// ### Preprocessing
/// ```swift
/// let pixelBuffer = try await PixelExtractor.createPixelBuffer(
///     source: .uiImage(image),
///     width: 518,
///     height: 518,
///     pixelFormat: .bgra8
/// )
/// ```
///
/// ### Post-processing with DepthEstimationOutput
/// ```swift
/// let result = try DepthEstimationOutput.process(
///     multiArray: modelOutput,
///     modelType: .depthAnything,
///     originalWidth: image.width,
///     originalHeight: image.height
/// )
///
/// // Visualize with colormap
/// let coloredImage = result.toColoredImage(colormap: .viridis)
/// ```
struct DepthEstimationView: View {
    // Sample images from Resources folder
    private let sampleImages = ["dog", "car", "street", "lion"]
    
    @State private var selectedImage: String = "street"
    @State private var loadedImage: UIImage?
    @State private var depthImage: UIImage?
    @State private var isRunning = false
    @State private var resultText = "Select an image and tap 'Run Depth Estimation'"
    @State private var inferenceTime: Double = 0
    @State private var depthStats: DepthStatistics?
    @State private var selectedColormap: ColormapOption = .viridis
    @State private var showOverlay: Bool = false
    @State private var overlayAlpha: Double = 0.5
    
    // Model constants
    private let modelWidth = 518
    private let modelHeight = 518
    
    /// Available colormaps for visualization
    enum ColormapOption: String, CaseIterable {
        case viridis = "Viridis"
        case plasma = "Plasma"
        case inferno = "Inferno"
        case magma = "Magma"
        case turbo = "Turbo"
        case grayscale = "Grayscale"
        
        var colormap: DepthColormap {
            switch self {
            case .viridis: return .viridis
            case .plasma: return .plasma
            case .inferno: return .inferno
            case .magma: return .magma
            case .turbo: return .turbo
            case .grayscale: return .grayscale
            }
        }
    }
    
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
                    Image(systemName: "cube.transparent")
                        .font(.title)
                        .foregroundColor(.cyan)
                    VStack(alignment: .leading) {
                        Text("Depth Anything Small")
                            .font(.headline)
                        Text("F16P6 • 518×518 • Relative Depth")
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
                                                    .stroke(selectedImage == imageName ? Color.cyan : Color.clear, lineWidth: 3)
                                            )
                                    } else {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.3))
                                            .frame(width: 70, height: 70)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(imageName.capitalized)
                                        .font(.caption2)
                                        .foregroundColor(selectedImage == imageName ? .cyan : .primary)
                                }
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("depth-image-\(imageName)")
                        }
                    }
                    .padding(.vertical, 4)
                }
                
                // Main Output Preview - Show depth output if available, otherwise original
                HStack {
                    Spacer()
                    if let depth = depthImage {
                        if showOverlay, let original = loadedImage {
                            // Overlay mode
                            ZStack {
                                Image(uiImage: original)
                                    .resizable()
                                    .scaledToFit()
                                Image(uiImage: depth)
                                    .resizable()
                                    .scaledToFit()
                                    .opacity(overlayAlpha)
                            }
                            .frame(maxHeight: 250)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        } else {
                            // Depth only
                            Image(uiImage: depth)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 250)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                    } else if let image = loadedImage {
                        // Original image before inference
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 250)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    Spacer()
                }
                
                // Overlay controls (only show when depth output is available)
                if depthImage != nil {
                    Toggle("Overlay on Original", isOn: $showOverlay)
                        .accessibilityIdentifier("depth-overlay-toggle")
                    
                    if showOverlay {
                        HStack {
                            Text("Opacity")
                                .font(.caption)
                            Slider(value: $overlayAlpha, in: 0.1...0.9)
                                .accessibilityIdentifier("depth-opacity-slider")
                        }
                    }
                }
            } header: {
                Text(depthImage != nil ? "Depth Output" : "Input Image")
            }
            
            // MARK: - Colormap Selection
            Section {
                Picker("Colormap", selection: $selectedColormap) {
                    ForEach(ColormapOption.allCases, id: \.self) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("depth-colormap-picker")
                .onChange(of: selectedColormap) { _, _ in
                    // Re-run if we have results
                    if depthImage != nil {
                        Task {
                            await runDepthEstimation()
                        }
                    }
                }
            } header: {
                Text("Visualization")
            }
            
            // MARK: - Run Inference Section
            Section {
                Button {
                    Task {
                        await runDepthEstimation()
                    }
                } label: {
                    HStack {
                        Spacer()
                        if isRunning {
                            ProgressView()
                                .padding(.trailing, 8)
                        }
                        Text(isRunning ? "Running..." : "Run Depth Estimation")
                            .fontWeight(.semibold)
                        Spacer()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.cyan)
                .disabled(loadedImage == nil || isRunning)
                .accessibilityIdentifier("depth-run-button")
                
                // Timing (show when depth result is available)
                if depthImage != nil {
                    HStack {
                        Image(systemName: "clock")
                            .foregroundColor(.secondary)
                        Text("Inference Time")
                        Spacer()
                        Text(String(format: "%.1f ms", inferenceTime))
                            .foregroundColor(.cyan)
                            .fontWeight(.medium)
                    }
                    .accessibilityIdentifier("depth-inference-time")
                }
            }
            
            // Statistics Section
            if let stats = depthStats {
                Section {
                    StatRow(label: "Min Depth", value: String(format: "%.2f", stats.min))
                    StatRow(label: "Max Depth", value: String(format: "%.2f", stats.max))
                    StatRow(label: "Mean", value: String(format: "%.2f", stats.mean))
                    StatRow(label: "Std Dev", value: String(format: "%.2f", stats.stdDev))
                    StatRow(label: "Median", value: String(format: "%.2f", stats.median))
                } header: {
                    Text("Depth Statistics")
                }
            }
            
            // MARK: - Error/Status Section
            if !resultText.isEmpty && depthImage == nil {
                Section {
                    Text(resultText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .accessibilityIdentifier("depth-status-text")
                } header: {
                    Text("Status")
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Depth Estimation")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadSelectedImage()
        }
    }
    
    // MARK: - Image Loading
    
    private func loadSelectedImage() {
        loadedImage = loadBundleImage(named: selectedImage)
        // Reset results when changing image
        depthImage = nil
        depthStats = nil
        resultText = "Tap 'Run Depth Estimation' to analyze"
    }
    
    // MARK: - Depth Estimation
    
    private func runDepthEstimation() async {
        guard let inputImage = loadedImage else {
            resultText = "No image selected"
            return
        }
        
        isRunning = true
        resultText = "Running depth estimation..."
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Load the Core ML model - try multiple paths since Xcode may bundle differently
            let modelURL = findModelURL()
            
            guard let url = modelURL else {
                throw PixelUtilsError.processingFailed("Could not find DepthAnythingSmallF16P6 model in bundle")
            }
            
            // If it's an mlpackage, compile it first
            if url.pathExtension == "mlpackage" {
                let compiledURL = try await MLModel.compileModel(at: url)
                try await runInference(modelURL: compiledURL, inputImage: inputImage, startTime: startTime)
            } else {
                try await runInference(modelURL: url, inputImage: inputImage, startTime: startTime)
            }
            
        } catch {
            resultText = "Error: \(error.localizedDescription)"
            depthImage = nil
            depthStats = nil
        }
        
        isRunning = false
    }
    
    /// Searches for the depth model in various bundle locations
    private func findModelURL() -> URL? {
        let modelName = "DepthAnythingSmallF16P6"
        
        // Try compiled model first (faster)
        if let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
            return url
        }
        
        // Try mlpackage without subdirectory
        if let url = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") {
            return url
        }
        
        // Try with Resources subdirectory
        if let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Resources") {
            return url
        }
        if let url = Bundle.main.url(forResource: modelName, withExtension: "mlpackage", subdirectory: "Resources") {
            return url
        }
        
        // Try finding by path
        if let path = Bundle.main.path(forResource: modelName, ofType: "mlpackage") {
            return URL(fileURLWithPath: path)
        }
        if let path = Bundle.main.path(forResource: modelName, ofType: "mlmodelc") {
            return URL(fileURLWithPath: path)
        }
        
        return nil
    }
    
    private func runInference(modelURL: URL, inputImage: UIImage, startTime: CFAbsoluteTime) async throws {
        // Configure model - use CPU only on simulator since MPS Graph isn't available
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .cpuAndNeuralEngine
        #endif
        
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        
        guard let cgImage = inputImage.cgImage else {
            throw PixelUtilsError.processingFailed("Could not get CGImage from UIImage")
        }
        
        // Use Vision framework to handle image constraints properly
        // The model expects: shorter dimension = 518, larger dimension = multiple of 14
        let visionModel = try VNCoreMLModel(for: model)
        
        // Track inference time
        var inferenceStart: CFAbsoluteTime = 0
        var inferenceEnd: CFAbsoluteTime = 0
        var resultMultiArray: MLMultiArray?
        var resultPixelBuffer: CVPixelBuffer?
        var inferenceError: Error?
        
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            inferenceEnd = CFAbsoluteTimeGetCurrent()
            
            if let error = error {
                inferenceError = error
                return
            }
            
            guard let results = request.results, !results.isEmpty else {
                inferenceError = PixelUtilsError.processingFailed("No results from Vision request")
                return
            }
            
            // The model outputs a grayscale image, which Vision returns as VNPixelBufferObservation
            if let pixelBufferObservation = results.first as? VNPixelBufferObservation {
                resultPixelBuffer = pixelBufferObservation.pixelBuffer
                return
            }
            
            // Also handle VNCoreMLFeatureValueObservation for models that output MultiArray
            if let featureObservation = results.first as? VNCoreMLFeatureValueObservation {
                if let multiArray = featureObservation.featureValue.multiArrayValue {
                    resultMultiArray = multiArray
                    return
                }
                if let pixelBuffer = featureObservation.featureValue.imageBufferValue {
                    resultPixelBuffer = pixelBuffer
                    return
                }
            }
            
            // Debug: print what we actually got
            let resultTypes = results.map { String(describing: type(of: $0)) }
            inferenceError = PixelUtilsError.processingFailed("Unexpected result types: \(resultTypes.joined(separator: ", "))")
        }
        
        // Configure request to center crop to match model's expected aspect ratio
        request.imageCropAndScaleOption = .scaleFill
        
        // Create handler and perform request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        inferenceStart = CFAbsoluteTimeGetCurrent()
        try handler.perform([request])
        
        // Check for errors
        if let error = inferenceError {
            throw error
        }
        
        inferenceTime = (inferenceEnd - inferenceStart) * 1000
        
        // Process based on output type
        if let pixelBuffer = resultPixelBuffer {
            try await processDepthPixelBuffer(
                pixelBuffer: pixelBuffer,
                originalWidth: cgImage.width,
                originalHeight: cgImage.height
            )
        } else if let multiArray = resultMultiArray {
            try await processDepthOutput(
                multiArray: multiArray,
                originalWidth: cgImage.width,
                originalHeight: cgImage.height
            )
        } else {
            throw PixelUtilsError.processingFailed("No depth output from model")
        }
    }
    
    private func processDepthPixelBuffer(
        pixelBuffer: CVPixelBuffer,
        originalWidth: Int,
        originalHeight: Int
    ) async throws {
        // Convert pixel buffer to MLMultiArray for processing with DepthEstimationOutput
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw PixelUtilsError.processingFailed("Could not get pixel buffer base address")
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        // Create MLMultiArray from pixel buffer
        let multiArray = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        
        let floatPtr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: width * height)
        
        // Handle different pixel formats
        switch pixelFormat {
        case kCVPixelFormatType_OneComponent8:
            // 8-bit grayscale
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                for x in 0..<width {
                    let value = ptr[y * bytesPerRow + x]
                    floatPtr[y * width + x] = Float32(value) / 255.0
                }
            }
            
        case kCVPixelFormatType_OneComponent16Half:
            // 16-bit float (Float16)
            let ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
            let stride = bytesPerRow / 2
            for y in 0..<height {
                for x in 0..<width {
                    let halfValue = ptr[y * stride + x]
                    floatPtr[y * width + x] = float16ToFloat32(halfValue)
                }
            }
            
        case kCVPixelFormatType_OneComponent32Float:
            // 32-bit float
            let ptr = baseAddress.assumingMemoryBound(to: Float32.self)
            let stride = bytesPerRow / 4
            for y in 0..<height {
                for x in 0..<width {
                    floatPtr[y * width + x] = ptr[y * stride + x]
                }
            }
            
        default:
            throw PixelUtilsError.processingFailed("Unsupported pixel format: \(pixelFormat)")
        }
        
        // Now process with DepthEstimationOutput
        try await processDepthOutput(
            multiArray: multiArray,
            originalWidth: originalWidth,
            originalHeight: originalHeight
        )
    }
    
    /// Convert Float16 (stored as UInt16) to Float32
    private func float16ToFloat32(_ half: UInt16) -> Float32 {
        let sign = (half & 0x8000) >> 15
        let exponent = (half & 0x7C00) >> 10
        let fraction = half & 0x03FF
        
        if exponent == 0 {
            if fraction == 0 {
                return sign == 0 ? 0.0 : -0.0
            }
            // Denormalized
            let f = Float32(fraction) / 1024.0
            return (sign == 0 ? 1.0 : -1.0) * f * pow(2.0, -14.0)
        } else if exponent == 31 {
            if fraction == 0 {
                return sign == 0 ? Float32.infinity : -Float32.infinity
            }
            return Float32.nan
        }
        
        let f = 1.0 + Float32(fraction) / 1024.0
        let e = Float32(Int(exponent) - 15)
        return (sign == 0 ? 1.0 : -1.0) * f * pow(2.0, e)
    }
    
    private func processDepthOutput(
        multiArray: MLMultiArray,
        originalWidth: Int,
        originalHeight: Int
    ) async throws {
        // Process with SwiftPixelUtils DepthEstimationOutput
        let result = try DepthEstimationOutput.process(
            multiArray: multiArray,
            modelType: .depthAnything,
            originalWidth: originalWidth,
            originalHeight: originalHeight
        )
        
        // Resize to original dimensions for better visualization
        let resizedResult = result.resizedToOriginal()
        
        // Get colored depth image
        guard let coloredCGImage = resizedResult.toColoredImage(
            colormap: selectedColormap.colormap,
            invert: true // Closer objects appear in hot colors
        ) else {
            throw PixelUtilsError.processingFailed("Could not create colored depth image")
        }
        
        // Update UI on main thread
        await MainActor.run {
            depthImage = UIImage(cgImage: coloredCGImage)
            depthStats = resizedResult.statistics
            resultText = ""
        }
    }
}

// MARK: - Helper Views

struct StatRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
                .monospacedDigit()
        }
    }
}

#Preview {
    NavigationStack {
        DepthEstimationView()
    }
}
