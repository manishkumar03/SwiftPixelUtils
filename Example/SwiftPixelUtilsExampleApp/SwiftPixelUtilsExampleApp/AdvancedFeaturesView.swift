//
//  AdvancedFeaturesView.swift
//  SwiftPixelUtilsExampleApp
//
//  Additional demo views for advanced SwiftPixelUtils features
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Tensor Operations Demo
struct TensorOperationsView: View {
    @State private var result = "Tap to test tensor operations"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Channel Operations") {
                        VStack(spacing: 12) {
                            Button("Extract Red Channel") {
                                Task { await extractChannel(index: 0, name: "Red") }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Extract Green Channel") {
                                Task { await extractChannel(index: 1, name: "Green") }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Extract Blue Channel") {
                                Task { await extractChannel(index: 2, name: "Blue") }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Shape Operations") {
                        VStack(spacing: 12) {
                            Button("Permute HWC → CHW") {
                                testPermute()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Squeeze") {
                                testSqueeze()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Unsqueeze (Add Batch)") {
                                testUnsqueeze()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Reshape") {
                                testReshape()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Patch Extraction") {
                        Button("Extract 64x64 Patch") {
                            Task { await extractPatch() }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding()
            }
            .navigationTitle("Tensor Ops")
        }
    }
    
    func extractChannel(index: Int, name: String) async {
        do {
            // Create sample RGB data (3x3 image)
            let data: [Float] = [
                // R, G, B for each pixel (HWC format)
                1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,
                0.5, 0.5, 0.0,  0.0, 0.5, 0.5,  0.5, 0.0, 0.5,
                1.0, 1.0, 1.0,  0.0, 0.0, 0.0,  0.5, 0.5, 0.5
            ]
            
            let channel = try TensorOperations.extractChannel(
                data: data,
                width: 3,
                height: 3,
                channels: 3,
                channelIndex: index,
                dataLayout: .hwc
            )
            
            result = """
            ✅ Channel Extraction
            Extracted: \(name) channel (index \(index))
            Input shape: [3, 3, 3] (HWC)
            Output size: \(channel.count)
            Output values: \(channel.map { String(format: "%.2f", $0) }.joined(separator: ", "))
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testPermute() {
        do {
            // Create sample data with shape [2, 3, 4] (H=2, W=3, C=4)
            let data: [Float] = Array(0..<24).map { Float($0) }
            
            let permuted = try TensorOperations.permute(
                data: data,
                shape: [2, 3, 4],
                order: [2, 0, 1]  // HWC -> CHW
            )
            
            result = """
            ✅ Permutation
            Input shape: [2, 3, 4] (HWC)
            Order: [2, 0, 1]
            Output shape: \(permuted.shape) (CHW)
            First 8 values: \(permuted.data.prefix(8).map { String(format: "%.0f", $0) }.joined(separator: ", "))
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testSqueeze() {
        // Squeeze is a pure shape operation (no throwing)
        let inputShape = [1, 2, 3]
        let squeezedShape = TensorOperations.squeeze(shape: inputShape, dims: [0])
        
        result = """
        ✅ Squeeze
        Input shape: \(inputShape)
        Squeeze dim: 0
        Output shape: \(squeezedShape)
        """
    }
    
    func testUnsqueeze() {
        // Unsqueeze is a pure shape operation (no throwing)
        let inputShape = [2, 3]
        let unsqueezedShape = TensorOperations.unsqueeze(shape: inputShape, dim: 0)
        
        result = """
        ✅ Unsqueeze
        Input shape: \(inputShape)
        Unsqueeze dim: 0
        Output shape: \(unsqueezedShape)
        """
    }
    
    func testReshape() {
        do {
            let data: [Float] = Array(0..<24).map { Float($0) }
            
            let reshaped = try TensorOperations.reshape(
                data: data,
                fromShape: [2, 3, 4],
                toShape: [4, 6]
            )
            
            result = """
            ✅ Reshape
            Input shape: [2, 3, 4]
            Output shape: \(reshaped.shape)
            Total elements: \(reshaped.data.count) (unchanged)
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func extractPatch() async {
        do {
            let source = ImageSource.url(URL(string: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg")!)
            
            let pixelResult = try await PixelExtractor.getPixelData(
                source: source,
                options: PixelDataOptions(
                    colorFormat: .rgb,
                    resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                    dataLayout: .hwc
                )
            )
            
            let patchOptions = PatchOptions(x: 50, y: 50, width: 64, height: 64)
            let patch = try TensorOperations.extractPatch(
                data: pixelResult.data,
                width: pixelResult.width,
                height: pixelResult.height,
                channels: pixelResult.channels,
                patchOptions: patchOptions,
                dataLayout: .hwc
            )
            
            result = """
            ✅ Patch Extraction
            Input: \(pixelResult.width)x\(pixelResult.height)
            Patch position: (50, 50)
            Patch size: \(patch.width)x\(patch.height)
            Channels: \(patch.channels)
            Data points: \(patch.data.count)
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Letterbox Demo
struct LetterboxView: View {
    @State private var result = "Tap to test letterbox operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
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
                        Button("Test Coordinate Transform") {
                            testReverseLetterbox()
                        }
                        .buttonStyle(.borderedProminent)
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
}

// MARK: - Quantization Demo
struct QuantizationView: View {
    @State private var result = "Tap to test quantization"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Quantization Types") {
                        VStack(spacing: 12) {
                            Button("Float → UInt8") {
                                testQuantization(dtype: .uint8)
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Float → Int8") {
                                testQuantization(dtype: .int8)
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Float → Int16") {
                                testQuantization(dtype: .int16)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Round Trip") {
                        Button("Quantize → Dequantize") {
                            testRoundTrip()
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    GroupBox("Calibration") {
                        Button("Calibrate Parameters") {
                            testCalibration()
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    GroupBox("Result") {
                        Text(result)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding()
            }
            .navigationTitle("Quantization")
        }
    }
    
    func testQuantization(dtype: QuantizationDType) {
        // Sample normalized float data (ImageNet-like)
        let floatData: [Float] = [-2.1, -1.0, -0.5, 0.0, 0.5, 1.0, 2.1]
        
        do {
            // First calibrate to get optimal params
            let params = Quantizer.calibrate(data: floatData, dtype: dtype)
            
            let options = QuantizationOptions(
                mode: .perTensor,
                dtype: dtype,
                scale: [params.scale],
                zeroPoint: [params.zeroPoint]
            )
            let quantized = try Quantizer.quantize(data: floatData, options: options)
            
            var dataStr = ""
            switch dtype {
            case .int8:
                dataStr = quantized.int8Data?.map { String($0) }.joined(separator: ", ") ?? "N/A"
            case .uint8:
                dataStr = quantized.uint8Data?.map { String($0) }.joined(separator: ", ") ?? "N/A"
            case .int16:
                dataStr = quantized.int16Data?.map { String($0) }.joined(separator: ", ") ?? "N/A"
            }
            
            result = """
            ✅ Quantization
            Input (Float32): \(floatData.map { String(format: "%.2f", $0) }.joined(separator: ", "))
            Output (\(dtype)): \(dataStr)
            
            Parameters:
            Scale: \(String(format: "%.6f", quantized.scale.first ?? 0))
            Zero Point: \(quantized.zeroPoint.first ?? 0)
            Mode: \(quantized.mode)
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testRoundTrip() {
        let original: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        do {
            // Calibrate
            let params = Quantizer.calibrate(data: original, dtype: .uint8)
            
            // Quantize
            let options = QuantizationOptions(
                mode: .perTensor,
                dtype: .uint8,
                scale: [params.scale],
                zeroPoint: [params.zeroPoint]
            )
            let quantized = try Quantizer.quantize(data: original, options: options)
            
            // Dequantize
            let restored = try Quantizer.dequantize(
                uint8Data: quantized.uint8Data,
                scale: quantized.scale,
                zeroPoint: quantized.zeroPoint,
                mode: .perTensor
            )
            
            // Calculate error
            let errors = zip(original, restored).map { abs($0 - $1) }
            let maxError = errors.max() ?? 0
            let avgError = errors.reduce(0, +) / Float(errors.count)
            
            result = """
            ✅ Round Trip (Float → UInt8 → Float)
            
            Original: \(original.map { String(format: "%.3f", $0) }.joined(separator: ", "))
            Quantized: \(quantized.uint8Data?.map { String($0) }.joined(separator: ", ") ?? "N/A")
            Restored: \(restored.map { String(format: "%.3f", $0) }.joined(separator: ", "))
            
            Scale: \(String(format: "%.6f", quantized.scale.first ?? 0))
            Zero Point: \(quantized.zeroPoint.first ?? 0)
            
            Max Error: \(String(format: "%.6f", maxError))
            Avg Error: \(String(format: "%.6f", avgError))
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testCalibration() {
        // Data with different ranges
        let data: [Float] = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        let params = Quantizer.calibrate(data: data, dtype: .int8)
        
        result = """
        ✅ Calibration
        
        Input data range: [\(String(format: "%.2f", data.min()!)), \(String(format: "%.2f", data.max()!))]
        
        Calculated for Int8:
        Scale: \(String(format: "%.6f", params.scale))
        Zero Point: \(params.zeroPoint)
        
        Quantized range would be: [-128, 127]
        """
    }
}

// MARK: - Multi-Crop Demo
struct MultiCropView: View {
    @State private var result = "Tap to test multi-crop operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Standard Crops") {
                        VStack(spacing: 12) {
                            Button("Five Crop (4 corners + center)") {
                                Task { await testFiveCrop() }
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Button("Ten Crop (5 + flips)") {
                                Task { await testTenCrop() }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Grid Extraction") {
                        VStack(spacing: 12) {
                            Button("Extract 3x3 Grid") {
                                Task { await testGridExtraction() }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    GroupBox("Random Crop") {
                        Button("5 Random 100x100 Crops") {
                            Task { await testRandomCrop() }
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
            .navigationTitle("Multi-Crop")
        }
    }
    
    func testFiveCrop() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CropOptions(width: 224, height: 224)
            let crops = try await MultiCropOperations.fiveCrop(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Five Crop
            Crop size: 224x224
            Crops extracted: \(crops.crops.count)
            
            Positions:
            \(crops.positions.enumerated().map { i, pos in
                "• Crop \(i + 1): (\(pos.x), \(pos.y))"
            }.joined(separator: "\n"))
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testTenCrop() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CropOptions(width: 224, height: 224)
            let crops = try await MultiCropOperations.tenCrop(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Ten Crop
            Crop size: 224x224
            Crops extracted: \(crops.crops.count)
            (5 original + 5 horizontal flips)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testGridExtraction() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = GridOptions(columns: 3, rows: 3)
            let grid = try await MultiCropOperations.extractGrid(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Grid Extraction
            Grid: \(grid.columns)x\(grid.rows) = \(grid.patches.count) patches
            Patch dimensions: \(grid.patches.first?.pixelData.width ?? 0)x\(grid.patches.first?.pixelData.height ?? 0)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testRandomCrop() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = RandomCropOptions(width: 100, height: 100, count: 5, seed: 42)
            let crops = try await MultiCropOperations.randomCrop(from: source, options: options)
            
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Random Crop
            Crop size: 100x100
            Count: \(crops.crops.count)
            Seed: 42 (reproducible)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Drawing/Visualization Demo
struct DrawingView: View {
    @State private var result = "Tap to test drawing operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
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
}

// MARK: - TensorToImage Demo
struct TensorToImageView: View {
    @State private var result = "Tap to test tensor to image conversion"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Convert Tensor to Image") {
                        VStack(spacing: 12) {
                            Button("RGB Tensor → Image") {
                                Task { await convertRGBTensor() }
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Button("Grayscale Tensor → Image") {
                                Task { await convertGrayscaleTensor() }
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
            .navigationTitle("Tensor → Image")
        }
    }
    
    func convertRGBTensor() async {
        do {
            // Create a simple 4x4 RGB gradient tensor (HWC format)
            var tensor: [Float] = []
            for y in 0..<4 {
                for x in 0..<4 {
                    let r = Float(x) / 3.0
                    let g = Float(y) / 3.0
                    let b = Float(x + y) / 6.0
                    tensor.append(contentsOf: [r, g, b])
                }
            }
            
            let start = CFAbsoluteTimeGetCurrent()
            let options = TensorToImageOptions(
                channels: 3,
                dataLayout: .hwc,
                denormalize: false
            )
            let converted = try await TensorToImage.convert(data: tensor, width: 4, height: 4, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ RGB Tensor Converted
            Input: 4x4x3 tensor (HWC)
            Output size: \(converted.width)x\(converted.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func convertGrayscaleTensor() async {
        do {
            // Create a simple 8x8 grayscale gradient tensor
            var tensor: [Float] = []
            for y in 0..<8 {
                for x in 0..<8 {
                    let value = Float(x + y) / 14.0
                    tensor.append(value)
                }
            }
            
            let start = CFAbsoluteTimeGetCurrent()
            let options = TensorToImageOptions(
                channels: 1,
                dataLayout: .hwc,
                denormalize: false
            )
            let converted = try await TensorToImage.convert(data: tensor, width: 8, height: 8, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Grayscale Tensor Converted
            Input: 8x8x1 tensor
            Output size: \(converted.width)x\(converted.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Tensor Validation Demo
struct TensorValidationView: View {
    @State private var result = "Tap to validate tensors"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Validation") {
                        VStack(spacing: 12) {
                            Button("Validate ImageNet Tensor") {
                                testImageNetValidation()
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Button("Validate Custom Tensor") {
                                testCustomValidation()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Calculate Statistics") {
                                testStatistics()
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
    }
    
    func testImageNetValidation() {
        // Create ImageNet-normalized tensor (should be in range [-2.5, 2.5] approximately)
        let validTensor: [Float] = Array(repeating: 0.5, count: 224 * 224 * 3)
        
        let validation = TensorValidation.validateImageNetTensor(data: validTensor, width: 224, height: 224)
        
        result = """
        ✅ ImageNet Tensor Validation
        Valid: \(validation.isValid)
        Size: \(validTensor.count) elements
        
        Statistics:
        Min: \(String(format: "%.4f", validation.statistics?.min ?? 0))
        Max: \(String(format: "%.4f", validation.statistics?.max ?? 0))
        Mean: \(String(format: "%.4f", validation.statistics?.mean ?? 0))
        
        Errors: \(validation.errors.isEmpty ? "None" : validation.errors.joined(separator: ", "))
        """
    }
    
    func testCustomValidation() {
        let tensor: [Float] = [0.1, 0.5, 0.9, 1.2, -0.1, 0.5]  // Some out of [0,1] range
        
        let spec = TensorSpec(
            shape: [2, 3],
            minValue: 0.0,
            maxValue: 1.0,
            checkNaN: true,
            checkInf: true
        )
        
        let validation = TensorValidation.validate(data: tensor, shape: [2, 3], spec: spec)
        
        result = """
        ✅ Custom Tensor Validation
        Valid: \(validation.isValid)
        Size: \(tensor.count) elements
        
        Statistics:
        Min: \(String(format: "%.4f", validation.statistics?.min ?? 0))
        Max: \(String(format: "%.4f", validation.statistics?.max ?? 0))
        
        Errors: \(validation.errors.isEmpty ? "None" : validation.errors.joined(separator: ", "))
        """
    }
    
    func testStatistics() {
        let data: [Float] = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        let stats = TensorValidation.calculateStatistics(data)
        
        result = """
        ✅ Tensor Statistics
        Data: \(data.map { String(format: "%.1f", $0) }.joined(separator: ", "))
        
        Min: \(String(format: "%.4f", stats.min))
        Max: \(String(format: "%.4f", stats.max))
        Mean: \(String(format: "%.4f", stats.mean))
        Std: \(String(format: "%.4f", stats.std))
        NaN Count: \(stats.nanCount)
        Inf Count: \(stats.infCount)
        """
    }
}

// MARK: - Batch Operations Demo
struct BatchOperationsView: View {
    @State private var result = "Tap to test batch operations"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Batch Pixel Extraction") {
                        Button("Batch Get Pixel Data") {
                            Task { await testBatchPixelExtraction() }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    GroupBox("Batch Assembly") {
                        VStack(spacing: 12) {
                            Button("Assemble Batch Tensor") {
                                Task { await testBatchAssembly() }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("TensorOperations.assembleBatch") {
                                Task { await testTensorBatchAssembly() }
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
            .navigationTitle("Batch Ops")
        }
    }
    
    func testBatchPixelExtraction() async {
        do {
            let sources = [
                ImageSource.url(URL(string: sampleImageURL)!),
                ImageSource.url(URL(string: sampleImageURL)!)
            ]
            
            let start = CFAbsoluteTimeGetCurrent()
            let options = PixelDataOptions(
                colorFormat: .rgb,
                resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                dataLayout: .chw
            )
            
            let results = try await PixelExtractor.batchGetPixelData(sources: sources, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Batch Pixel Extraction
            Images processed: \(results.count)
            Each image: \(results.first?.width ?? 0)x\(results.first?.height ?? 0)
            Channels: \(results.first?.channels ?? 0)
            Layout: CHW
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testBatchAssembly() async {
        do {
            let sources = [
                ImageSource.url(URL(string: sampleImageURL)!),
                ImageSource.url(URL(string: sampleImageURL)!)
            ]
            
            let options = PixelDataOptions(
                colorFormat: .rgb,
                resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                dataLayout: .chw
            )
            
            let start = CFAbsoluteTimeGetCurrent()
            let pixelResults = try await PixelExtractor.batchGetPixelData(sources: sources, options: options)
            let batch = try BatchAssembly.assemble(pixelResults)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Batch Assembly
            Batch size: \(batch.batchSize)
            Shape: \(batch.shape)
            Total elements: \(batch.data.count)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testTensorBatchAssembly() async {
        do {
            let sources = [
                ImageSource.url(URL(string: sampleImageURL)!),
                ImageSource.url(URL(string: sampleImageURL)!)
            ]
            
            let options = PixelDataOptions(
                colorFormat: .rgb,
                resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
                dataLayout: .chw
            )
            
            let start = CFAbsoluteTimeGetCurrent()
            let pixelResults = try await PixelExtractor.batchGetPixelData(sources: sources, options: options)
            let batch = try TensorOperations.assembleBatch(results: pixelResults, layout: .nchw)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ TensorOperations.assembleBatch
            Batch size: \(batch.batchSize)
            Shape: \(batch.shape)
            Layout: \(batch.layout)
            Total elements: \(batch.data.count)
            
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Cutout Demo
struct CutoutView: View {
    @State private var result = "Tap to test cutout augmentation"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    GroupBox("Cutout Augmentation") {
                        VStack(spacing: 12) {
                            Button("Single Cutout") {
                                Task { await testSingleCutout() }
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Button("Multiple Cutouts") {
                                Task { await testMultipleCutouts() }
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Random Cutouts") {
                                Task { await testRandomCutouts() }
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
            .navigationTitle("Cutout")
        }
    }
    
    func testSingleCutout() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CutoutOptions(
                numCutouts: 1,
                minSize: 0.05,
                maxSize: 0.10,
                fillMode: .constant,
                fillValue: [0, 0, 0]  // Black fill
            )
            let cutout = try await ImageAugmentor.cutout(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Single Cutout Applied
            Cutouts: 1
            Area ratio: 5-10%
            Fill mode: Constant (black)
            Output size: \(cutout.size.width)x\(cutout.size.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testMultipleCutouts() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CutoutOptions(
                numCutouts: 3,
                minSize: 0.03,
                maxSize: 0.08,
                fillMode: .constant,
                fillValue: [128, 128, 128]  // Gray fill
            )
            let cutout = try await ImageAugmentor.cutout(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Multiple Cutouts Applied
            Number of cutouts: 3
            Area ratio: 3-8%
            Fill mode: Constant (gray)
            Output size: \(cutout.size.width)x\(cutout.size.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func testRandomCutouts() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            let options = CutoutOptions(
                numCutouts: 5,
                minSize: 0.02,
                maxSize: 0.05,
                fillMode: .random,
                seed: 42
            )
            let cutout = try await ImageAugmentor.cutout(source: source, options: options)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            result = """
            ✅ Random Cutouts Applied
            Number of cutouts: 5
            Area ratio: 2-5%
            Fill mode: Random colors
            Seed: 42 (reproducible)
            Output size: \(cutout.size.width)x\(cutout.size.height)
            Processing Time: \(String(format: "%.2f", time))ms
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Image Validation Demo
struct ImageValidationView: View {
    @State private var result = "Tap to validate images"
    
    private let sampleImageURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    var body: some View {
        NavigationView {
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
    }
    
    func validateSizeConstraints() async {
        do {
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Get metadata first to display dimensions
            let metadata = try await ImageAnalyzer.getMetadata(source: source)
            
            let options = ValidationOptions(
                minWidth: 100,
                minHeight: 100,
                maxWidth: 4000,
                maxHeight: 4000
            )
            let validation = try await ImageAnalyzer.validate(source: source, options: options)
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
            let source = ImageSource.url(URL(string: sampleImageURL)!)
            let start = CFAbsoluteTimeGetCurrent()
            
            // Get metadata first
            let metadata = try await ImageAnalyzer.getMetadata(source: source)
            
            // Use default validation options (just checking image can be loaded)
            let options = ValidationOptions()
            let validation = try await ImageAnalyzer.validate(source: source, options: options)
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