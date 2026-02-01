//
//  BoundingBoxView.swift
//  SwiftPixelUtilsExampleApp
//
//  Bounding box utilities demo tab
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Bounding Box Demo
struct BoundingBoxView: View {
    @State private var result = "Tap to test bounding box utilities"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Format Conversion") {
                        VStack(spacing: 12) {
                            Button("cxcywh → xyxy") {
                                testFormatConversion(from: .cxcywh, to: .xyxy)
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("bbox-convert-cxcywh-xyxy")
                            
                            Button("xyxy → xywh") {
                                testFormatConversion(from: .xyxy, to: .xywh)
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("bbox-convert-xyxy-xywh")
                            
                            Button("xywh → cxcywh") {
                                testFormatConversion(from: .xywh, to: .cxcywh)
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("bbox-convert-xywh-cxcywh")
                        }
                    }
                    
                    GroupBox("Box Operations") {
                        VStack(spacing: 12) {
                            Button("Calculate IoU") {
                                testIoU()
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("bbox-calculate-iou")
                            
                            Button("Scale Boxes") {
                                testScaling()
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("bbox-scale-boxes")
                            
                            Button("Clip Boxes") {
                                testClipping()
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("bbox-clip-boxes")
                            
                            Button("Non-Max Suppression") {
                                testNMS()
                            }
                            .buttonStyle(.borderedProminent)
                            .accessibilityIdentifier("bbox-nms")
                        }
                    }
                    
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .accessibilityIdentifier("bbox-result-text")
                    }
                }
                .padding()
            }
            .navigationTitle("Bounding Boxes")
        }
    }
    
    func testFormatConversion(from: BoxFormat, to: BoxFormat) {
        let inputBox: [Float] = [320, 240, 100, 80] // Example box
        let converted = BoundingBox.convertFormat([inputBox], from: from, to: to)
        
        result = """
        ✅ Format Conversion
        From: \(from) → To: \(to)
        Input: \(inputBox)
        Output: \(converted.first ?? [])
        """
    }
    
    func testIoU() {
        let box1: [Float] = [100, 100, 200, 200]
        let box2: [Float] = [150, 150, 250, 250]
        let iou = BoundingBox.calculateIoU(box1, box2, format: .xyxy)
        
        result = """
        ✅ IoU Calculation
        Box 1: \(box1)
        Box 2: \(box2)
        IoU: \(String(format: "%.4f", iou))
        Overlap: \(String(format: "%.1f", iou * 100))%
        """
    }
    
    func testScaling() {
        let boxes: [[Float]] = [[100, 100, 200, 200]]
        let scaled = BoundingBox.scale(
            boxes,
            from: CGSize(width: 640, height: 640),
            to: CGSize(width: 1920, height: 1080),
            format: .xyxy
        )
        
        result = """
        ✅ Box Scaling
        From: 640x640
        To: 1920x1080
        Input: \(boxes.first ?? [])
        Output: \(scaled.first ?? [])
        """
    }
    
    func testClipping() {
        let boxes: [[Float]] = [[-10, 50, 700, 500]]
        let clipped = BoundingBox.clip(boxes, imageSize: CGSize(width: 640, height: 480), format: .xyxy)
        
        result = """
        ✅ Box Clipping
        Image: 640x480
        Input: \(boxes.first ?? [])
        Clipped: \(clipped.first ?? [])
        """
    }
    
    func testNMS() {
        let detections: [Detection] = [
            Detection(box: [100, 100, 200, 200], score: 0.9, classIndex: 0),
            Detection(box: [110, 110, 210, 210], score: 0.8, classIndex: 0),
            Detection(box: [300, 300, 400, 400], score: 0.7, classIndex: 1),
            Detection(box: [105, 105, 205, 205], score: 0.85, classIndex: 0),
        ]
        
        let filtered = BoundingBox.nonMaxSuppression(
            detections: detections,
            iouThreshold: 0.5,
            scoreThreshold: 0.3
        )
        
        result = """
        ✅ Non-Max Suppression
        Input detections: \(detections.count)
        IoU threshold: 0.5
        Score threshold: 0.3
        Output detections: \(filtered.count)
        
        Kept boxes:
        \(filtered.map { "- Class \($0.classIndex): score \(String(format: "%.2f", $0.score))" }.joined(separator: "\n"))
        """
    }
}
