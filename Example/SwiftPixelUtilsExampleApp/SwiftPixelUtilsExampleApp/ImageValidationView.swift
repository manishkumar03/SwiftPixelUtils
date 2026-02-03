import SwiftUI
import SwiftPixelUtils

struct ImageValidationView: View {
    @State private var result = "Tap to validate images"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Image Validation") {
                    VStack(spacing: 12) {
                        Button("Validate Size Constraints") {
                            Task { await validateSizeConstraints() }
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Validate Format") {
                            Task { await validateFormat() }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Validation")
    }
    
    func validateSizeConstraints() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Get metadata first to display dimensions
            let metadata = try ImageAnalyzer.getMetadata(source: source)
            
            let options = ValidationOptions(
                minWidth: 100,
                minHeight: 100,
                maxWidth: 4000,
                maxHeight: 4000
            )
            let validation = try ImageAnalyzer.validate(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Image Validation
            Valid: \(validation.isValid)
            Width: \(metadata.width) (min: 100, max: 4000)
            Height: \(metadata.height) (min: 100, max: 4000)
            
            Issues: \(validation.issues.isEmpty ? "None" : validation.issues.joined(separator: ", "))
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func validateFormat() async {
        do {
            let imageData = try await downloadImageData(from: sampleImageURL)
            let source = ImageSource.data(imageData)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Get metadata first
            let metadata = try ImageAnalyzer.getMetadata(source: source)
            
            // Use default validation options (just checking image can be loaded)
            let options = ValidationOptions()
            let validation = try ImageAnalyzer.validate(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Format Validation
            Valid: \(validation.isValid)
            Dimensions: \(metadata.width)x\(metadata.height)
            Color Space: \(metadata.colorSpace)
            Has Alpha: \(metadata.hasAlpha)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
