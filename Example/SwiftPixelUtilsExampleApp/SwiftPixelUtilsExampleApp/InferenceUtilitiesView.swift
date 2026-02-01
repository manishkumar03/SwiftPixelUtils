//
//  InferenceUtilitiesView.swift
//  SwiftPixelUtilsExampleApp
//
//  Demonstrates inference post-processing utilities
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Inference Utilities View
struct InferenceUtilitiesView: View {
    var body: some View {
        List {
            Section("Activation Functions") {
                NavigationLink(destination: ActivationFunctionsView()) {
                    Label("Softmax & Sigmoid", systemImage: "function")
                }
            }
            
            Section("Top-K Extraction") {
                NavigationLink(destination: TopKExtractionView()) {
                    Label("Top-K Extraction", systemImage: "chart.bar.fill")
                }
            }
            
            Section("NMS Variations") {
                NavigationLink(destination: NMSVariationsView()) {
                    Label("Soft-NMS & Batched NMS", systemImage: "square.on.square.dashed")
                }
            }
            
            Section("Confidence Filtering") {
                NavigationLink(destination: ConfidenceFilteringView()) {
                    Label("Confidence Filtering", systemImage: "slider.horizontal.3")
                }
            }
            
            Section("Mask Utilities") {
                NavigationLink(destination: MaskUtilitiesView()) {
                    Label("Mask Resize & Threshold", systemImage: "square.split.diagonal.2x2")
                }
            }
            
            Section("CoreML Integration") {
                NavigationLink(destination: CoreMLConversionView()) {
                    Label("MLMultiArray Conversion", systemImage: "cpu")
                }
            }
        }
        .navigationTitle("Inference Utilities")
    }
}

// MARK: - Activation Functions Demo
struct ActivationFunctionsView: View {
    @State private var logitInput = "-1.0, 2.0, 0.5, -0.3"
    @State private var softmaxResult = ""
    @State private var sigmoidResult = ""
    @State private var processingTime = ""
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Input Logits") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Enter comma-separated logit values:")
                            .font(.caption)
                        
                        TextField("Logits", text: $logitInput)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                    }
                }
                
                GroupBox("Softmax") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Converts logits to probability distribution (sum = 1)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Button("Apply Softmax") {
                            applySoftmax()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        if !softmaxResult.isEmpty {
                            Text(softmaxResult)
                                .font(.system(.caption, design: .monospaced))
                                .padding(8)
                                .background(Color.gray.opacity(0.1))
                                .cornerRadius(8)
                        }
                    }
                }
                
                GroupBox("Sigmoid") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Converts each logit independently to (0, 1) range")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Button("Apply Sigmoid") {
                            applySigmoid()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        if !sigmoidResult.isEmpty {
                            Text(sigmoidResult)
                                .font(.system(.caption, design: .monospaced))
                                .padding(8)
                                .background(Color.gray.opacity(0.1))
                                .cornerRadius(8)
                        }
                    }
                }
                
                GroupBox("Comparison") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("• **Softmax**: Multi-class classification (outputs sum to 1)")
                        Text("• **Sigmoid**: Multi-label classification (each output independent)")
                    }
                    .font(.caption)
                }
                
                if !processingTime.isEmpty {
                    Text(processingTime)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
        }
        .navigationTitle("Activation Functions")
    }
    
    func applySoftmax() {
        let logits = parseLogits(logitInput)
        guard !logits.isEmpty else {
            softmaxResult = "Invalid input"
            return
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        let probs = ActivationFunctions.softmax(logits)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        
        softmaxResult = formatProbabilities(probs)
        let sum = probs.reduce(0, +)
        softmaxResult += "\n\nSum: \(String(format: "%.4f", sum)) (should be ~1.0)"
        processingTime = String(format: "Processing: %.3f ms", elapsed)
    }
    
    func applySigmoid() {
        let logits = parseLogits(logitInput)
        guard !logits.isEmpty else {
            sigmoidResult = "Invalid input"
            return
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        let probs = ActivationFunctions.sigmoid(logits)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        
        sigmoidResult = formatProbabilities(probs)
        let sum = probs.reduce(0, +)
        sigmoidResult += "\n\nSum: \(String(format: "%.4f", sum)) (varies)"
        processingTime = String(format: "Processing: %.3f ms", elapsed)
    }
    
    func parseLogits(_ input: String) -> [Float] {
        return input.split(separator: ",")
            .compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
    }
    
    func formatProbabilities(_ probs: [Float]) -> String {
        return probs.enumerated()
            .map { "[\($0.offset)]: \(String(format: "%.4f", $0.element)) (\(String(format: "%.1f%%", $0.element * 100)))" }
            .joined(separator: "\n")
    }
}

// MARK: - Top-K Extraction Demo
struct TopKExtractionView: View {
    @State private var result = ""
    @State private var k = 5
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Configuration") {
                    Stepper("K = \(k)", value: $k, in: 1...10)
                }
                
                GroupBox("Top-K Extraction") {
                    VStack(spacing: 12) {
                        Button("Extract Top-K (Raw Scores)") {
                            extractTopK()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Extract Top-K (With Softmax)") {
                            extractTopKWithSoftmax()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Argmax / Argmin") {
                    HStack(spacing: 12) {
                        Button("Find Argmax") {
                            findArgmax()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Find Argmin") {
                            findArgmin()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                if !result.isEmpty {
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Top-K Extraction")
    }
    
    func extractTopK() {
        // Simulate classifier output with 10 classes
        let scores: [Float] = [0.05, 0.82, 0.03, 0.25, 0.91, 0.12, 0.08, 0.67, 0.45, 0.33]
        
        let topK = TopKExtractor.extractTopK(values: scores, k: k)
        
        result = "Input scores: \(scores.map { String(format: "%.2f", $0) }.joined(separator: ", "))\n\n"
        result += "Top-\(k) results:\n"
        for (idx, val) in zip(topK.indices, topK.values) {
            result += "  Index \(idx): \(String(format: "%.4f", val))\n"
        }
        result += "\nProcessing: \(String(format: "%.3f ms", topK.processingTimeMs))"
    }
    
    func extractTopKWithSoftmax() {
        // Simulate raw logits
        let logits: [Float] = [-2.5, 3.2, -1.0, 1.5, 4.0, 0.3, -0.5, 2.8, 1.8, 1.0]
        
        let topK = TopKExtractor.extractTopKWithSoftmax(logits: logits, k: k)
        
        result = "Input logits: \(logits.map { String(format: "%.1f", $0) }.joined(separator: ", "))\n\n"
        result += "Top-\(k) probabilities (after softmax):\n"
        for (idx, val) in zip(topK.indices, topK.values) {
            result += "  Index \(idx): \(String(format: "%.4f", val)) (\(String(format: "%.1f%%", val * 100)))\n"
        }
        result += "\nProcessing: \(String(format: "%.3f ms", topK.processingTimeMs))"
    }
    
    func findArgmax() {
        let scores: [Float] = [0.15, 0.72, 0.08, 0.95, 0.43]
        if let (idx, val) = TopKExtractor.argmax(scores) {
            result = "Scores: \(scores)\n\nArgmax: Index \(idx), Value \(String(format: "%.4f", val))"
        }
    }
    
    func findArgmin() {
        let scores: [Float] = [0.15, 0.72, 0.08, 0.95, 0.43]
        if let (idx, val) = TopKExtractor.argmin(scores) {
            result = "Scores: \(scores)\n\nArgmin: Index \(idx), Value \(String(format: "%.4f", val))"
        }
    }
}

// MARK: - NMS Variations Demo
struct NMSVariationsView: View {
    @State private var result = ""
    @State private var iouThreshold: Float = 0.5
    @State private var scoreThreshold: Float = 0.3
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Configuration") {
                    VStack(spacing: 12) {
                        HStack {
                            Text("IoU Threshold: \(String(format: "%.2f", iouThreshold))")
                            Slider(value: $iouThreshold, in: 0.1...0.9)
                        }
                        
                        HStack {
                            Text("Score Threshold: \(String(format: "%.2f", scoreThreshold))")
                            Slider(value: $scoreThreshold, in: 0.1...0.9)
                        }
                    }
                }
                
                GroupBox("NMS Methods") {
                    VStack(spacing: 12) {
                        Button("Standard NMS") {
                            runStandardNMS()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Soft-NMS (Linear)") {
                            runSoftNMSLinear()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Soft-NMS (Gaussian)") {
                            runSoftNMSGaussian()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Class-Agnostic NMS") {
                            runClassAgnosticNMS()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Batch Operations") {
                    Button("Batched NMS (3 Images)") {
                        runBatchedNMS()
                    }
                    .buttonStyle(.bordered)
                }
                
                if !result.isEmpty {
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("NMS Variations")
    }
    
    func createTestDetections() -> [Detection] {
        // Create overlapping detections to test NMS
        return [
            Detection(box: [10, 10, 50, 50], score: 0.9, classIndex: 0, label: "person"),
            Detection(box: [12, 12, 52, 52], score: 0.85, classIndex: 0, label: "person"), // Overlaps with first
            Detection(box: [15, 15, 55, 55], score: 0.8, classIndex: 0, label: "person"),  // Overlaps with both
            Detection(box: [100, 100, 150, 150], score: 0.75, classIndex: 1, label: "car"),
            Detection(box: [105, 105, 155, 155], score: 0.7, classIndex: 1, label: "car"), // Overlaps
            Detection(box: [200, 200, 250, 250], score: 0.6, classIndex: 0, label: "person"), // Isolated
        ]
    }
    
    func formatDetections(_ detections: [Detection]) -> String {
        return detections.map { d in
            "[\(d.label ?? "class\(d.classIndex)")] score: \(String(format: "%.3f", d.score)), box: [\(d.box.map { String(format: "%.0f", $0) }.joined(separator: ", "))]"
        }.joined(separator: "\n")
    }
    
    func runStandardNMS() {
        let detections = createTestDetections()
        let filtered = NMSVariants.perClassNMS(
            detections: detections,
            iouThreshold: iouThreshold,
            scoreThreshold: scoreThreshold
        )
        
        result = "Input: \(detections.count) detections\n\n"
        result += formatDetections(detections)
        result += "\n\n--- Standard NMS ---\n"
        result += "Output: \(filtered.count) detections\n\n"
        result += formatDetections(filtered)
    }
    
    func runSoftNMSLinear() {
        let detections = createTestDetections()
        let filtered = NMSVariants.softNMS(
            detections: detections,
            iouThreshold: iouThreshold,
            scoreThreshold: scoreThreshold,
            mode: .linear
        )
        
        result = "Input: \(detections.count) detections\n\n"
        result += "--- Soft-NMS (Linear) ---\n"
        result += "Note: Scores are reduced rather than boxes removed\n\n"
        result += "Output: \(filtered.count) detections\n\n"
        result += formatDetections(filtered)
    }
    
    func runSoftNMSGaussian() {
        let detections = createTestDetections()
        let filtered = NMSVariants.softNMS(
            detections: detections,
            iouThreshold: iouThreshold,
            scoreThreshold: scoreThreshold,
            mode: .gaussian(sigma: 0.5)
        )
        
        result = "Input: \(detections.count) detections\n\n"
        result += "--- Soft-NMS (Gaussian, σ=0.5) ---\n\n"
        result += "Output: \(filtered.count) detections\n\n"
        result += formatDetections(filtered)
    }
    
    func runClassAgnosticNMS() {
        let detections = createTestDetections()
        let filtered = NMSVariants.classAgnosticNMS(
            detections: detections,
            iouThreshold: iouThreshold,
            scoreThreshold: scoreThreshold
        )
        
        result = "Input: \(detections.count) detections\n\n"
        result += "--- Class-Agnostic NMS ---\n"
        result += "All classes treated equally\n\n"
        result += "Output: \(filtered.count) detections\n\n"
        result += formatDetections(filtered)
    }
    
    func runBatchedNMS() {
        let batch = [
            createTestDetections(),
            [
                Detection(box: [20, 20, 80, 80], score: 0.95, classIndex: 0),
                Detection(box: [25, 25, 85, 85], score: 0.88, classIndex: 0),
            ],
            [
                Detection(box: [50, 50, 100, 100], score: 0.7, classIndex: 2),
            ]
        ]
        
        let filteredBatch = NMSVariants.batchedNMS(
            batchDetections: batch,
            iouThreshold: iouThreshold,
            scoreThreshold: scoreThreshold,
            maxDetectionsPerImage: 10
        )
        
        result = "--- Batched NMS ---\n"
        result += "Input: \(batch.count) images\n\n"
        for (i, detections) in filteredBatch.enumerated() {
            result += "Image \(i + 1): \(detections.count) detections\n"
        }
    }
}

// MARK: - Confidence Filtering Demo
struct ConfidenceFilteringView: View {
    @State private var result = ""
    @State private var threshold: Float = 0.5
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Configuration") {
                    HStack {
                        Text("Threshold: \(String(format: "%.2f", threshold))")
                        Slider(value: $threshold, in: 0.1...0.9)
                    }
                }
                
                GroupBox("Filtering Methods") {
                    VStack(spacing: 12) {
                        Button("Simple Threshold") {
                            simpleFilter()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Class-Specific Thresholds") {
                            classSpecificFilter()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Ratio-Based (Keep Top 50%)") {
                            ratioBasedFilter()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                if !result.isEmpty {
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Confidence Filtering")
    }
    
    func createTestDetections() -> [Detection] {
        return [
            Detection(box: [10, 10, 50, 50], score: 0.95, classIndex: 0, label: "person"),
            Detection(box: [60, 60, 100, 100], score: 0.45, classIndex: 0, label: "person"),
            Detection(box: [110, 110, 150, 150], score: 0.72, classIndex: 1, label: "car"),
            Detection(box: [160, 160, 200, 200], score: 0.35, classIndex: 1, label: "car"),
            Detection(box: [210, 210, 250, 250], score: 0.88, classIndex: 2, label: "dog"),
            Detection(box: [260, 260, 300, 300], score: 0.22, classIndex: 2, label: "dog"),
        ]
    }
    
    func simpleFilter() {
        let detections = createTestDetections()
        let filtered = ConfidenceFilter.filter(detections: detections, minConfidence: threshold)
        
        result = "Input: \(detections.count) detections\n"
        result += "Threshold: \(String(format: "%.2f", threshold))\n\n"
        result += "Filtered: \(filtered.count) detections\n\n"
        for d in filtered {
            result += "[\(d.label ?? "?")] \(String(format: "%.2f", d.score))\n"
        }
    }
    
    func classSpecificFilter() {
        let detections = createTestDetections()
        
        // Different thresholds per class
        let thresholds: [Int: Float] = [
            0: 0.7,  // person - high threshold
            1: 0.5,  // car - medium
            2: 0.3   // dog - low (small objects harder to detect)
        ]
        
        let filtered = ConfidenceFilter.filterWithClassThresholds(
            detections: detections,
            thresholds: thresholds,
            defaultThreshold: 0.5
        )
        
        result = "Input: \(detections.count) detections\n"
        result += "Class thresholds:\n"
        result += "  person: 0.70\n"
        result += "  car: 0.50\n"
        result += "  dog: 0.30\n\n"
        result += "Filtered: \(filtered.count) detections\n\n"
        for d in filtered {
            result += "[\(d.label ?? "?")] \(String(format: "%.2f", d.score))\n"
        }
    }
    
    func ratioBasedFilter() {
        let detections = createTestDetections()
        let filtered = ConfidenceFilter.filterByRatio(
            detections: detections,
            keepRatio: 0.5,
            minKeep: 1,
            maxKeep: 10
        )
        
        result = "Input: \(detections.count) detections\n"
        result += "Keep ratio: 50%\n\n"
        result += "Filtered: \(filtered.count) detections (sorted by score)\n\n"
        for d in filtered {
            result += "[\(d.label ?? "?")] \(String(format: "%.2f", d.score))\n"
        }
    }
}

// MARK: - Mask Utilities Demo
struct MaskUtilitiesView: View {
    @State private var result = ""
    @State private var threshold: Float = 0.5
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Mask Resizing") {
                    VStack(spacing: 12) {
                        Button("Resize (Nearest Neighbor)") {
                            resizeNearest()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Resize (Bilinear)") {
                            resizeBilinear()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Thresholding") {
                    VStack(spacing: 12) {
                        HStack {
                            Text("Threshold: \(String(format: "%.2f", threshold))")
                            Slider(value: $threshold, in: 0.1...0.9)
                        }
                        
                        Button("Apply Threshold") {
                            applyThreshold()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Argmax Mask") {
                    Button("Argmax per Pixel") {
                        computeArgmax()
                    }
                    .buttonStyle(.bordered)
                }
                
                GroupBox("Mask Metrics") {
                    HStack(spacing: 12) {
                        Button("Compute Area") {
                            computeArea()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Compute IoU") {
                            computeMaskIoU()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                if !result.isEmpty {
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Mask Utilities")
    }
    
    func resizeNearest() {
        // Create a simple 4x4 mask
        let mask: [Float] = [
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.5, 0.5
        ]
        
        let resized = MaskUtilities.resizeMask(
            mask: mask,
            sourceWidth: 4,
            sourceHeight: 4,
            targetWidth: 8,
            targetHeight: 8
        )
        
        result = "Original 4×4:\n"
        result += formatMask(mask, width: 4)
        result += "\n\nResized 8×8 (Nearest Neighbor):\n"
        result += formatMask(resized.mask, width: 8)
        result += "\n\nProcessing: \(String(format: "%.3f ms", resized.processingTimeMs))"
    }
    
    func resizeBilinear() {
        let mask: [Float] = [
            1.0, 0.0,
            0.0, 1.0
        ]
        
        let resized = MaskUtilities.resizeMaskBilinear(
            mask: mask,
            sourceWidth: 2,
            sourceHeight: 2,
            targetWidth: 4,
            targetHeight: 4
        )
        
        result = "Original 2×2:\n"
        result += formatMask(mask, width: 2)
        result += "\n\nResized 4×4 (Bilinear - smooth edges):\n"
        result += formatMask(resized.mask, width: 4)
        result += "\n\nProcessing: \(String(format: "%.3f ms", resized.processingTimeMs))"
    }
    
    func applyThreshold() {
        let probMask: [Float] = [0.2, 0.6, 0.8, 0.3, 0.9, 0.4, 0.7, 0.1, 0.5]
        let binary = MaskUtilities.threshold(mask: probMask, threshold: threshold)
        
        result = "Probability mask:\n"
        result += probMask.map { String(format: "%.1f", $0) }.joined(separator: " ")
        result += "\n\nThreshold: \(String(format: "%.2f", threshold))\n\n"
        result += "Binary mask:\n"
        result += binary.map { String(format: "%.0f", $0) }.joined(separator: " ")
    }
    
    func computeArgmax() {
        // Simulate 3-class segmentation output (3 x 2 x 2)
        // CHW layout: [class0_values..., class1_values..., class2_values...]
        let tensor: [Float] = [
            // Class 0 (2x2)
            0.2, 0.1,
            0.8, 0.3,
            // Class 1 (2x2)
            0.7, 0.2,
            0.1, 0.6,
            // Class 2 (2x2)
            0.1, 0.7,
            0.1, 0.1
        ]
        
        let argmaxMask = MaskUtilities.argmaxMask(
            tensor: tensor,
            shape: [3, 2, 2],
            layout: .chw
        )
        
        result = "Segmentation output (3 classes, 2×2):\n"
        result += "Class 0: [0.2, 0.1, 0.8, 0.3]\n"
        result += "Class 1: [0.7, 0.2, 0.1, 0.6]\n"
        result += "Class 2: [0.1, 0.7, 0.1, 0.1]\n\n"
        result += "Argmax mask (per-pixel class):\n"
        result += argmaxMask.map { String($0) }.joined(separator: " ")
    }
    
    func computeArea() {
        let mask: [Float] = [1, 1, 0, 1, 0, 0, 1, 1, 1]
        let area = MaskUtilities.computeArea(mask: mask)
        
        result = "Mask: \(mask.map { String(format: "%.0f", $0) }.joined(separator: " "))\n\n"
        result += "Area (non-zero pixels): \(area)"
    }
    
    func computeMaskIoU() {
        let mask1: [Float] = [1, 1, 0, 0, 1, 1, 0, 0, 0]
        let mask2: [Float] = [0, 1, 1, 0, 1, 1, 0, 0, 0]
        
        let iou = MaskUtilities.maskIoU(mask1: mask1, mask2: mask2)
        
        result = "Mask 1: \(mask1.map { String(format: "%.0f", $0) }.joined(separator: " "))\n"
        result += "Mask 2: \(mask2.map { String(format: "%.0f", $0) }.joined(separator: " "))\n\n"
        result += "Intersection over Union: \(String(format: "%.4f", iou))"
    }
    
    func formatMask(_ mask: [Float], width: Int) -> String {
        var lines: [String] = []
        for row in 0..<(mask.count / width) {
            let start = row * width
            let end = start + width
            let rowValues = mask[start..<end].map { String(format: "%.1f", $0) }
            lines.append(rowValues.joined(separator: " "))
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - CoreML Conversion Demo
struct CoreMLConversionView: View {
    @State private var result = ""
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("MLMultiArray Conversion") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Convert preprocessed pixel data to MLMultiArray for CoreML inference.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Button("Create MLMultiArray") {
                            createMLMultiArray()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Convert From MLMultiArray") {
                            convertFromMLMultiArray()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Usage Example") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("""
                        // Preprocess image
                        let result = try await PixelExtractor.getPixelData(
                            source: .cgImage(image),
                            options: ModelPresets.mobileNetV3.options
                        )
                        
                        // Convert to MLMultiArray
                        let inputArray = try CoreMLConversion.toMLMultiArray(
                            from: result
                        )
                        
                        // Use with CoreML model
                        let prediction = try model.prediction(input: inputArray)
                        """)
                        .font(.system(.caption2, design: .monospaced))
                    }
                }
                
                if !result.isEmpty {
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
            .padding()
        }
        .navigationTitle("CoreML Conversion")
    }
    
    func createMLMultiArray() {
        #if canImport(CoreML)
        // Simulate preprocessed data (1x3x4x4 tensor)
        let data: [Float] = Array(repeating: 0.5, count: 1 * 3 * 4 * 4)
        let shape = [1, 3, 4, 4]
        
        do {
            let multiArray = try CoreMLConversion.toMLMultiArray(data: data, shape: shape)
            
            result = "Created MLMultiArray:\n"
            result += "  Shape: \(multiArray.shape)\n"
            result += "  Data type: \(multiArray.dataType)\n"
            result += "  Count: \(multiArray.count)\n"
            result += "\n✅ Ready for CoreML inference!"
        } catch {
            result = "Error: \(error.localizedDescription)"
        }
        #else
        result = "CoreML not available on this platform"
        #endif
    }
    
    func convertFromMLMultiArray() {
        #if canImport(CoreML)
        do {
            // Create a sample MLMultiArray (simulating model output)
            let multiArray = try MLMultiArray(shape: [1, 10], dataType: .float32)
            
            // Fill with sample logits
            for i in 0..<10 {
                multiArray[i] = NSNumber(value: Float.random(in: -2...2))
            }
            
            // Convert back to Float array
            let (data, shape) = CoreMLConversion.fromMLMultiArray(multiArray)
            
            // Apply softmax for probabilities
            let probs = CoreMLConversion.toProbabilities(multiArray, applySoftmax: true)
            
            result = "MLMultiArray output:\n"
            result += "  Shape: \(shape)\n"
            result += "  Data: \(data.prefix(5).map { String(format: "%.2f", $0) }.joined(separator: ", "))...\n\n"
            result += "After softmax:\n"
            result += probs.enumerated().map { "  [\($0)]: \(String(format: "%.3f", $1))" }.joined(separator: "\n")
        } catch {
            result = "Error: \(error.localizedDescription)"
        }
        #else
        result = "CoreML not available on this platform"
        #endif
    }
}

// MARK: - CoreML Import
#if canImport(CoreML)
import CoreML
#endif

#Preview {
    NavigationView {
        InferenceUtilitiesView()
    }
}
