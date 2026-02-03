import SwiftUI
import SwiftPixelUtils

// MARK: - Image Download Helper

/// Downloads image data from a remote URL.
/// Use this to download images before passing to SwiftPixelUtils functions.
func downloadImageData(from urlString: String) async throws -> Data {
    guard let url = URL(string: urlString) else {
        throw URLError(.badURL)
    }
    let (data, response) = try await URLSession.shared.data(from: url)
    guard let httpResponse = response as? HTTPURLResponse,
          (200...299).contains(httpResponse.statusCode) else {
        throw URLError(.badServerResponse)
    }
    return data
}

/// Downloads image data from a remote URL and returns it as an ImageSource.
/// Use this helper to convert remote URLs to ImageSource.data() for SwiftPixelUtils functions.
func downloadImageSource(from urlString: String) async throws -> ImageSource {
    let data = try await downloadImageData(from: urlString)
    return .data(data)
}

/// Downloads multiple images from remote URLs and returns them as ImageSources.
func downloadImageSources(from urlStrings: [String]) async throws -> [ImageSource] {
    try await withThrowingTaskGroup(of: (Int, ImageSource).self) { group in
        for (index, urlString) in urlStrings.enumerated() {
            group.addTask {
                let source = try await downloadImageSource(from: urlString)
                return (index, source)
            }
        }
        
        var results = [(Int, ImageSource)]()
        for try await result in group {
            results.append(result)
        }
        
        // Sort by original index to maintain order
        return results.sorted { $0.0 < $1.0 }.map { $0.1 }
    }
}

// MARK: - Bundle Image Loading Helper

/// Loads an image from the app bundle, checking both Resources directory and root bundle
func loadBundleImage(named name: String) -> UIImage? {
    // Try Resources directory with .jpg extension
    if let path = Bundle.main.path(forResource: name, ofType: "jpg", inDirectory: "Resources") {
        return UIImage(contentsOfFile: path)
    }
    // Try Resources directory with .png extension
    if let path = Bundle.main.path(forResource: name, ofType: "png", inDirectory: "Resources") {
        return UIImage(contentsOfFile: path)
    }
    // Try root bundle with .jpg extension
    if let path = Bundle.main.path(forResource: name, ofType: "jpg") {
        return UIImage(contentsOfFile: path)
    }
    // Try root bundle with .png extension
    if let path = Bundle.main.path(forResource: name, ofType: "png") {
        return UIImage(contentsOfFile: path)
    }
    // Try asset catalog
    return UIImage(named: name)
}

// MARK: - Confidence Formatting

/// Formats a confidence value (0-1) as a percentage string with appropriate precision
func formatConfidence(_ value: Float) -> String {
    if value >= 0.01 {
        return String(format: "%.1f%%", value * 100)
    } else if value >= 0.001 {
        return String(format: "%.2f%%", value * 100)
    } else {
        return "<0.1%"
    }
}

/// Returns an appropriate color for a confidence level
func confidenceColor(for confidence: Float) -> Color {
    if confidence > 0.7 { return .green }
    if confidence > 0.5 { return .blue }
    if confidence > 0.3 { return .orange }
    return .red
}

/// Returns an appropriate color for classification confidence (different thresholds)
func classificationConfidenceColor(for confidence: Float) -> Color {
    if confidence > 0.5 { return .green }
    if confidence > 0.2 { return .blue }
    if confidence > 0.1 { return .orange }
    return .secondary
}

// MARK: - Shared UI Components

/// A reusable row view for displaying classification predictions
struct ClassificationPredictionRow: View {
    let rank: Int
    let label: String
    let confidence: Float
    let showProgressBar: Bool
    
    init(rank: Int, label: String, confidence: Float, showProgressBar: Bool = true) {
        self.rank = rank
        self.label = label
        self.confidence = confidence
        self.showProgressBar = showProgressBar
    }
    
    var body: some View {
        HStack(spacing: 12) {
            // Rank badge
            ZStack {
                Circle()
                    .fill(rankColor.opacity(0.2))
                    .frame(width: 28, height: 28)
                Text("\(rank)")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(rankColor)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                // Label
                Text(label)
                    .font(.system(size: 15, weight: .medium))
                    .lineLimit(1)
                
                if showProgressBar {
                    // Progress bar
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 6)
                                .cornerRadius(3)
                            
                            Rectangle()
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [rankColor.opacity(0.7), rankColor]),
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: geometry.size.width * CGFloat(min(confidence, 1.0)), height: 6)
                                .cornerRadius(3)
                        }
                    }
                    .frame(height: 6)
                }
            }
            
            Spacer()
            
            // Confidence percentage
            Text(formatConfidence(confidence))
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundColor(classificationConfidenceColor(for: confidence))
        }
        .padding(.vertical, 6)
    }
    
    private var rankColor: Color {
        switch rank {
        case 1: return .yellow
        case 2: return .gray
        case 3: return .orange
        default: return .blue
        }
    }
}

/// A reusable row view for displaying object detections
struct DetectionResultRow: View {
    let rank: Int
    let label: String
    let confidence: Float
    let box: [Float]?
    
    init(rank: Int, label: String, confidence: Float, box: [Float]? = nil) {
        self.rank = rank
        self.label = label
        self.confidence = confidence
        self.box = box
    }
    
    var body: some View {
        HStack(spacing: 12) {
            // Rank badge
            ZStack {
                Circle()
                    .fill(confidenceColor(for: confidence).opacity(0.2))
                    .frame(width: 28, height: 28)
                Text("\(rank)")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(confidenceColor(for: confidence))
            }
            
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.system(size: 15, weight: .medium))
                
                if let box = box {
                    Text(formatBox(box))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            Text(formatConfidence(confidence))
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundColor(confidenceColor(for: confidence))
        }
        .padding(.vertical, 4)
    }
    
    private func formatBox(_ box: [Float]) -> String {
        let formatted = box.map { String(format: "%.0f", $0) }.joined(separator: ", ")
        return "[\(formatted)]"
    }
}

// MARK: - Processing Time Display

/// Formats processing time in milliseconds
func formatProcessingTime(_ ms: Double) -> String {
    if ms < 1 {
        return String(format: "%.2f ms", ms)
    } else if ms < 100 {
        return String(format: "%.1f ms", ms)
    } else {
        return String(format: "%.0f ms", ms)
    }
}

/// A view that displays processing time with an icon
struct ProcessingTimeView: View {
    let preprocessingMs: Double?
    let inferenceMs: Double?
    let postprocessingMs: Double?
    
    var body: some View {
        HStack(spacing: 16) {
            if let preprocessing = preprocessingMs {
                timeLabel("Preprocess", preprocessing)
            }
            if let inference = inferenceMs {
                timeLabel("Inference", inference)
            }
            if let postprocessing = postprocessingMs {
                timeLabel("Postprocess", postprocessing)
            }
        }
        .font(.caption)
        .foregroundColor(.secondary)
    }
    
    private func timeLabel(_ label: String, _ ms: Double) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
            Text(formatProcessingTime(ms))
                .font(.system(size: 12, weight: .medium, design: .monospaced))
        }
    }
}

// MARK: - Result Card

/// A card-style container for displaying results
struct ResultCard<Content: View>: View {
    let title: String
    let systemImage: String
    let content: Content
    
    init(title: String, systemImage: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: systemImage)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
            }
            
            content
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}
