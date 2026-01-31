//
//  LetterboxView.swift
//  SwiftPixelUtilsExampleApp
//
//  Letterbox operations demo view
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Letterbox Demo
struct LetterboxView: View {
    @State private var result = "Tap to test letterbox operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Letterbox Padding") {
                    VStack(spacing: 12) {
                        Button("Letterbox to 640x640 (YOLO)") {
                            Task { await applyLetterbox(width: 640, height: 640) }
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Letterbox to 416x416") {
                            Task { await applyLetterbox(width: 416, height: 416) }
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Letterbox to 320x320") {
                            Task { await applyLetterbox(width: 320, height: 320) }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Reverse Transform") {
                    VStack(spacing: 12) {
                        Button("Test Single Box Transform") {
                            testReverseLetterbox()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Test Multiple Boxes Transform") {
                            testReverseMultipleBoxes()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Test Detection Transform") {
                            testReverseDetections()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("Combined Operations") {
                    Button("Letterbox + Extract Pixels") {
                        Task { await testApplyAndExtract() }
                    }
                    .buttonStyle(.bordered)
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
        }
        .navigationTitle("Letterbox")
    }
    
    func applyLetterbox(width: Int, height: Int) async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = LetterboxOptions(
                targetWidth: width,
                targetHeight: height,
                fillColor: (114, 114, 114)  // YOLO gray
            )
            let letterboxed = try await Letterbox.apply(to: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Letterbox Applied
            Target: \(width)x\(height)
            Fill Color: RGB(114, 114, 114)
            
            Transform Info:
            Scale: \(String(format: "%.4f", letterboxed.scale))
            Offset: (\(String(format: "%.1f", letterboxed.offsetX)), \(String(format: "%.1f", letterboxed.offsetY)))
            Original: \(letterboxed.originalWidth)x\(letterboxed.originalHeight)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testReverseLetterbox() {
        // Simulate a detection in letterboxed space
        let detectionBox: [Double] = [100, 150, 200, 250]  // xyxy in 640x640 space
        
        // Simulated letterbox info (as if original was 1920x1080)
        let scale = 0.333
        let offsetX = 0.0
        let offsetY = 120.0
        
        let originalBox = Letterbox.reverseTransformBox(
            box: detectionBox,
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY
        )
        
        result = """
        ✅ Reverse Letterbox Transform
        
        Detection in letterbox space (640x640):
        [\(detectionBox.map { String(format: "%.1f", $0) }.joined(separator: ", "))]
        
        Letterbox Info:
        Scale: \(scale)
        Offset: (\(offsetX), \(offsetY))
        
        Transformed to original space:
        [\(originalBox.map { String(format: "%.1f", $0) }.joined(separator: ", "))]
        """
    }
    
    func testReverseMultipleBoxes() {
        // Multiple detections in letterboxed space
        let boxes: [[Double]] = [
            [100, 150, 200, 250],
            [300, 200, 400, 350],
            [50, 50, 150, 100]
        ]
        
        let scale = 0.5
        let offsetX = 80.0
        let offsetY = 0.0
        
        let originalBoxes = Letterbox.reverseTransformBoxes(
            boxes: boxes,
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY
        )
        
        let boxStrings = originalBoxes.map { box in
            "[\(box.map { String(format: "%.1f", $0) }.joined(separator: ", "))]"
        }.joined(separator: "\n")
        
        result = """
        ✅ Reverse Multiple Boxes Transform
        
        \(boxes.count) detections in letterbox space
        Scale: \(scale), Offset: (\(offsetX), \(offsetY))
        
        Transformed boxes:
        \(boxStrings)
        """
    }
    
    func testReverseDetections() {
        // Simulated YOLO detections with confidence scores
        let detections = [
            Detection(box: [100, 150, 200, 250], score: 0.95, classIndex: 0),
            Detection(box: [300, 200, 400, 350], score: 0.87, classIndex: 1)
        ]
        
        let scale = 0.5
        let offsetX = 80.0
        let offsetY = 0.0
        
        let originalDetections = Letterbox.reverseTransformDetections(
            detections: detections,
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY
        )
        
        let detStrings = originalDetections.map { det in
            "Class \(det.classIndex): [\(det.box.map { String(format: "%.1f", $0) }.joined(separator: ", "))] score=\(String(format: "%.2f", det.score))"
        }.joined(separator: "\n")
        
        result = """
        ✅ Reverse Detection Transform
        
        \(detections.count) detections transformed
        Scale: \(scale), Offset: (\(offsetX), \(offsetY))
        
        Original space detections:
        \(detStrings)
        """
    }
    
    func testApplyAndExtract() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let letterboxOptions = LetterboxOptions(
                targetWidth: 640,
                targetHeight: 640,
                fillColor: (114, 114, 114)
            )
            let pixelOptions = PixelDataOptions(
                colorFormat: .rgb,
                normalization: .scale,
                dataLayout: .chw
            )
            
            let result = try await Letterbox.applyAndExtract(
                from: source,
                letterboxOptions: letterboxOptions,
                pixelOptions: pixelOptions
            )
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            self.result = """
            ✅ Letterbox + Extract
            Output: \(result.pixels.width)x\(result.pixels.height)
            Channels: \(result.pixels.channels)
            Layout: CHW
            Data count: \(result.pixels.data.count)
            
            Transform Info:
            Scale: \(String(format: "%.4f", result.letterbox.scale))
            Offset: (\(String(format: "%.1f", result.letterbox.offsetX)), \(String(format: "%.1f", result.letterbox.offsetY)))
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            self.result = "❌ Error: \(error.localizedDescription)"
        }
    }
}
