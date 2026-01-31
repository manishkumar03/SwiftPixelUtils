import SwiftUI
import SwiftPixelUtils

struct DrawingView: View {
    @State private var result = "Tap to test drawing operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Box Drawing") {
                    VStack(spacing: 12) {
                        Button("Draw Detection Boxes") {
                            Task { await drawBoxes() }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                
                GroupBox("Keypoints") {
                    Button("Draw Keypoints") {
                        Task { await drawKeypoints() }
                    }
                    .buttonStyle(.bordered)
                }
                
                GroupBox("Overlays") {
                    VStack(spacing: 12) {
                        Button("Overlay Heatmap") {
                            Task { await overlayHeatmap() }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Overlay Segmentation Mask") {
                            Task { await overlayMask() }
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
        .navigationTitle("Drawing")
    }
    
    func drawBoxes() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let boxes = [
                DrawableBox(
                    box: [50, 50, 250, 200],  // [x1, y1, x2, y2]
                    label: "cat", score: 0.95,
                    color: (r: 255, g: 0, b: 0, a: 255)
                ),
                DrawableBox(
                    box: [300, 100, 450, 300],  // [x1, y1, x2, y2]
                    label: "object", score: 0.87,
                    color: (r: 0, g: 255, b: 0, a: 255)
                )
            ]
            
            let options = BoxDrawingOptions(lineWidth: 3, fontSize: 14, drawLabels: true, drawScores: true)
            let drawn = try await Drawing.drawBoxes(on: source, boxes: boxes, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Boxes Drawn
            Boxes: \(boxes.count)
            Output size: \(drawn.width)x\(drawn.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func drawKeypoints() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let keypoints = [
                DrawableKeypoint(x: 100, y: 100, confidence: 0.9, color: (r: 255, g: 0, b: 0, a: 255)),
                DrawableKeypoint(x: 80, y: 80, confidence: 0.85, color: (r: 0, g: 255, b: 0, a: 255)),
                DrawableKeypoint(x: 120, y: 80, confidence: 0.88, color: (r: 0, g: 0, b: 255, a: 255))
            ]
            
            let options = KeypointDrawingOptions(pointRadius: 5)
            let drawn = try await Drawing.drawKeypoints(on: source, keypoints: keypoints, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Keypoints Drawn
            Keypoints: \(keypoints.count)
            Output size: \(drawn.width)x\(drawn.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func overlayHeatmap() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Create a simple gradient heatmap (10x10)
            var heatmap: [Float] = []
            for y in 0..<10 {
                for x in 0..<10 {
                    let value = Float(x + y) / 18.0  // Values from 0 to 1
                    heatmap.append(value)
                }
            }
            
            let options = HeatmapOverlayOptions(
                alpha: 0.5,
                colorScheme: .jet,
                heatmapWidth: 10,
                heatmapHeight: 10
            )
            let overlaid = try await Drawing.overlayHeatmap(on: source, heatmap: heatmap, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Heatmap Overlaid
            Heatmap size: 10x10
            Alpha: 0.5
            Color scheme: Jet
            Output size: \(overlaid.width)x\(overlaid.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func overlayMask() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Create a simple segmentation mask (8x8)
            // Values > 0.5 will be colored with the mask
            var mask: [Float] = []
            for y in 0..<8 {
                for x in 0..<8 {
                    // Create a circular mask in the center
                    let cx = Float(x) - 3.5
                    let cy = Float(y) - 3.5
                    let dist = sqrt(cx * cx + cy * cy)
                    mask.append(dist < 3.0 ? 0.8 : 0.1)
                }
            }
            
            let options = MaskOverlayOptions(
                alpha: 0.6,
                color: (r: 0, g: 255, b: 0),  // Green mask
                maskWidth: 8,
                maskHeight: 8,
                threshold: 0.5
            )
            let overlaid = try Drawing.overlayMask(on: source, mask: mask, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Segmentation Mask Overlaid
            Mask size: 8x8
            Threshold: 0.5
            Alpha: 0.6
            Color: Green
            Output size: \(overlaid.width)x\(overlaid.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
