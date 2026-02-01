//
//  QuantizationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Quantization demo view - demonstrates per-tensor and per-channel quantization
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Quantization Demo
struct QuantizationView: View {
    @State private var result = "Tap to test quantization"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Per-Tensor Quantization
                GroupBox("Per-Tensor Quantization") {
                    VStack(spacing: 12) {
                        Button("Float → UInt8") {
                            testQuantization(dtype: .uint8)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-per-tensor-uint8")
                        
                        Button("Float → Int8") {
                            testQuantization(dtype: .int8)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-per-tensor-int8")
                        
                        Button("Float → Int16") {
                            testQuantization(dtype: .int16)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-per-tensor-int16")
                    }
                }
                
                // Per-Channel Quantization
                GroupBox("Per-Channel Quantization") {
                    VStack(spacing: 12) {
                        Button("RGB → Int8 (CHW)") {
                            testPerChannelQuantization(dtype: .int8, layout: .chw)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-per-channel-int8-chw")
                        
                        Button("RGB → UInt8 (HWC)") {
                            testPerChannelQuantization(dtype: .uint8, layout: .hwc)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-per-channel-uint8-hwc")
                        
                        Button("Compare Per-Tensor vs Per-Channel") {
                            compareQuantizationModes()
                        }
                        .buttonStyle(.borderedProminent)
                        .accessibilityIdentifier("quant-compare-modes")
                    }
                }
                
                GroupBox("Round Trip") {
                    VStack(spacing: 12) {
                        Button("Per-Tensor Round Trip") {
                            testRoundTrip()
                        }
                        .buttonStyle(.borderedProminent)
                        .accessibilityIdentifier("quant-round-trip-per-tensor")
                        
                        Button("Per-Channel Round Trip") {
                            testPerChannelRoundTrip()
                        }
                        .buttonStyle(.borderedProminent)
                        .accessibilityIdentifier("quant-round-trip-per-channel")
                    }
                }
                
                GroupBox("Calibration") {
                    VStack(spacing: 12) {
                        Button("Per-Tensor Calibration") {
                            testCalibration()
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-calibrate-per-tensor")
                        
                        Button("Per-Channel Calibration") {
                            testPerChannelCalibration()
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("quant-calibrate-per-channel")
                    }
                }
                
                GroupBox("Result") {
                    Text(result)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityIdentifier("quant-result-text")
                }
            }
            .padding()
        }
        .navigationTitle("Quantization")
    }
    
    // MARK: - Per-Tensor Tests
    
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
            ✅ Per-Tensor Quantization
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
    
    // MARK: - Per-Channel Tests
    
    enum DataLayout {
        case chw, hwc
    }
    
    func testPerChannelQuantization(dtype: QuantizationDType, layout: DataLayout) {
        // Simulate 3-channel RGB data with different value ranges per channel
        // This demonstrates why per-channel is more accurate
        let numChannels = 3
        let spatialSize = 4  // 2x2 image for simplicity
        
        // Create data with deliberately different ranges per channel:
        // R: [-0.5, 0.5], G: [-1.0, 1.0], B: [-2.0, 2.0]
        var floatData: [Float]
        
        if layout == .chw {
            // CHW: all R, then all G, then all B
            let rChannel: [Float] = [-0.5, -0.2, 0.2, 0.5]   // Range: 1.0
            let gChannel: [Float] = [-1.0, -0.3, 0.3, 1.0]   // Range: 2.0
            let bChannel: [Float] = [-2.0, -0.5, 0.5, 2.0]   // Range: 4.0
            floatData = rChannel + gChannel + bChannel
        } else {
            // HWC: interleaved RGBRGBRGB...
            floatData = [
                -0.5, -1.0, -2.0,  // Pixel 0: R, G, B
                -0.2, -0.3, -0.5,  // Pixel 1
                 0.2,  0.3,  0.5,  // Pixel 2
                 0.5,  1.0,  2.0   // Pixel 3
            ]
        }
        
        do {
            // Calibrate per-channel
            let params = Quantizer.calibratePerChannel(
                data: floatData,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: layout == .chw ? 0 : 2,
                dtype: dtype
            )
            
            let options = QuantizationOptions(
                mode: .perChannel,
                dtype: dtype,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: layout == .chw ? 0 : 2,
                numChannels: numChannels,
                spatialSize: spatialSize
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
            ✅ Per-Channel Quantization (\(layout == .chw ? "CHW" : "HWC"))
            
            Input (\(numChannels) channels, \(spatialSize) pixels each):
            \(floatData.map { String(format: "%.2f", $0) }.joined(separator: ", "))
            
            Output (\(dtype)):
            \(dataStr)
            
            Per-Channel Parameters:
            R: scale=\(String(format: "%.4f", params.scales[0])), zp=\(params.zeroPoints[0])
            G: scale=\(String(format: "%.4f", params.scales[1])), zp=\(params.zeroPoints[1])
            B: scale=\(String(format: "%.4f", params.scales[2])), zp=\(params.zeroPoints[2])
            
            Value Ranges Detected:
            R: [\(String(format: "%.2f", params.minValues[0])), \(String(format: "%.2f", params.maxValues[0]))]
            G: [\(String(format: "%.2f", params.minValues[1])), \(String(format: "%.2f", params.maxValues[1]))]
            B: [\(String(format: "%.2f", params.minValues[2])), \(String(format: "%.2f", params.maxValues[2]))]
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    func compareQuantizationModes() {
        // Create data where per-channel should clearly outperform per-tensor
        let numChannels = 3
        let spatialSize = 4
        
        // CHW layout with very different ranges
        let rChannel: [Float] = [0.0, 0.1, 0.2, 0.3]      // Range: 0.3 (small)
        let gChannel: [Float] = [-1.0, -0.3, 0.3, 1.0]    // Range: 2.0 (medium)
        let bChannel: [Float] = [-10.0, -5.0, 5.0, 10.0]  // Range: 20.0 (large!)
        let floatData = rChannel + gChannel + bChannel
        
        do {
            // Per-Tensor quantization
            let tensorParams = Quantizer.calibrate(data: floatData, dtype: .int8)
            let tensorOptions = QuantizationOptions(
                mode: .perTensor,
                dtype: .int8,
                scale: [tensorParams.scale],
                zeroPoint: [tensorParams.zeroPoint]
            )
            let tensorQuantized = try Quantizer.quantize(data: floatData, options: tensorOptions)
            let tensorRestored = try Quantizer.dequantize(
                int8Data: tensorQuantized.int8Data,
                scale: tensorQuantized.scale,
                zeroPoint: tensorQuantized.zeroPoint,
                mode: .perTensor
            )
            
            // Per-Channel quantization
            let channelParams = Quantizer.calibratePerChannel(
                data: floatData,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: 0,
                dtype: .int8
            )
            let channelOptions = QuantizationOptions(
                mode: .perChannel,
                dtype: .int8,
                scale: channelParams.scales,
                zeroPoint: channelParams.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
            let channelQuantized = try Quantizer.quantize(data: floatData, options: channelOptions)
            let channelRestored = try Quantizer.dequantize(
                int8Data: channelQuantized.int8Data,
                scale: channelQuantized.scale,
                zeroPoint: channelQuantized.zeroPoint,
                mode: .perChannel,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: 0
            )
            
            // Calculate errors
            let tensorErrors = zip(floatData, tensorRestored).map { abs($0 - $1) }
            let channelErrors = zip(floatData, channelRestored).map { abs($0 - $1) }
            
            let tensorMaxError = tensorErrors.max() ?? 0
            let tensorAvgError = tensorErrors.reduce(0, +) / Float(tensorErrors.count)
            let channelMaxError = channelErrors.max() ?? 0
            let channelAvgError = channelErrors.reduce(0, +) / Float(channelErrors.count)
            
            // Calculate per-channel errors for R channel (which has small range)
            let rTensorErrors = zip(rChannel, Array(tensorRestored[0..<spatialSize])).map { abs($0 - $1) }
            let rChannelErrors = zip(rChannel, Array(channelRestored[0..<spatialSize])).map { abs($0 - $1) }
            
            result = """
            ✅ Per-Tensor vs Per-Channel Comparison
            
            Data with varying channel ranges:
            R: [0.0, 0.3] (small range)
            G: [-1.0, 1.0] (medium range)
            B: [-10.0, 10.0] (large range)
            
            ═══════════════════════════════════════
            Per-TENSOR Results:
            Scale: \(String(format: "%.6f", tensorParams.scale))
            Max Error: \(String(format: "%.4f", tensorMaxError))
            Avg Error: \(String(format: "%.4f", tensorAvgError))
            R-channel Max Error: \(String(format: "%.4f", rTensorErrors.max() ?? 0))
            
            ═══════════════════════════════════════
            Per-CHANNEL Results:
            R Scale: \(String(format: "%.6f", channelParams.scales[0]))
            G Scale: \(String(format: "%.6f", channelParams.scales[1]))
            B Scale: \(String(format: "%.6f", channelParams.scales[2]))
            Max Error: \(String(format: "%.4f", channelMaxError))
            Avg Error: \(String(format: "%.4f", channelAvgError))
            R-channel Max Error: \(String(format: "%.4f", rChannelErrors.max() ?? 0))
            
            ═══════════════════════════════════════
            🎯 Per-channel is \(String(format: "%.1f", tensorAvgError / max(channelAvgError, 0.0001)))x more accurate!
            
            The R channel benefits most because per-tensor
            uses B's large range, wasting precision on R.
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Round Trip Tests
    
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
            ✅ Per-Tensor Round Trip (Float → UInt8 → Float)
            
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
    
    func testPerChannelRoundTrip() {
        let numChannels = 3
        let spatialSize = 4
        
        // CHW layout with different ranges
        let rChannel: [Float] = [-0.5, -0.2, 0.2, 0.5]
        let gChannel: [Float] = [-1.0, -0.3, 0.3, 1.0]
        let bChannel: [Float] = [-2.0, -0.5, 0.5, 2.0]
        let original = rChannel + gChannel + bChannel
        
        do {
            // Calibrate per-channel
            let params = Quantizer.calibratePerChannel(
                data: original,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: 0,
                dtype: .int8
            )
            
            // Quantize
            let options = QuantizationOptions(
                mode: .perChannel,
                dtype: .int8,
                scale: params.scales,
                zeroPoint: params.zeroPoints,
                channelAxis: 0,
                numChannels: numChannels,
                spatialSize: spatialSize
            )
            let quantized = try Quantizer.quantize(data: original, options: options)
            
            // Dequantize
            let restored = try Quantizer.dequantize(
                int8Data: quantized.int8Data,
                scale: quantized.scale,
                zeroPoint: quantized.zeroPoint,
                mode: .perChannel,
                numChannels: numChannels,
                spatialSize: spatialSize,
                channelAxis: 0
            )
            
            // Calculate per-channel errors
            let rErrors = zip(rChannel, Array(restored[0..<spatialSize])).map { abs($0 - $1) }
            let gErrors = zip(gChannel, Array(restored[spatialSize..<2*spatialSize])).map { abs($0 - $1) }
            let bErrors = zip(bChannel, Array(restored[2*spatialSize..<3*spatialSize])).map { abs($0 - $1) }
            
            result = """
            ✅ Per-Channel Round Trip (Float → Int8 → Float)
            
            R Channel:
              Original: \(rChannel.map { String(format: "%.2f", $0) }.joined(separator: ", "))
              Restored: \(Array(restored[0..<spatialSize]).map { String(format: "%.2f", $0) }.joined(separator: ", "))
              Max Error: \(String(format: "%.4f", rErrors.max() ?? 0))
            
            G Channel:
              Original: \(gChannel.map { String(format: "%.2f", $0) }.joined(separator: ", "))
              Restored: \(Array(restored[spatialSize..<2*spatialSize]).map { String(format: "%.2f", $0) }.joined(separator: ", "))
              Max Error: \(String(format: "%.4f", gErrors.max() ?? 0))
            
            B Channel:
              Original: \(bChannel.map { String(format: "%.2f", $0) }.joined(separator: ", "))
              Restored: \(Array(restored[2*spatialSize..<3*spatialSize]).map { String(format: "%.2f", $0) }.joined(separator: ", "))
              Max Error: \(String(format: "%.4f", bErrors.max() ?? 0))
            
            Quantized Int8: \(quantized.int8Data?.map { String($0) }.joined(separator: ", ") ?? "N/A")
            """
        } catch {
            result = "❌ Error: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Calibration Tests
    
    func testCalibration() {
        // Data with different ranges
        let data: [Float] = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        let params = Quantizer.calibrate(data: data, dtype: .int8)
        
        result = """
        ✅ Per-Tensor Calibration
        
        Input data range: [\(String(format: "%.2f", data.min()!)), \(String(format: "%.2f", data.max()!))]
        
        Calculated for Int8:
        Scale: \(String(format: "%.6f", params.scale))
        Zero Point: \(params.zeroPoint)
        
        Quantized range would be: [-128, 127]
        """
    }
    
    func testPerChannelCalibration() {
        let numChannels = 3
        let spatialSize = 4
        
        // CHW layout with deliberately different ranges
        let rChannel: [Float] = [0.0, 0.1, 0.2, 0.3]       // Small range
        let gChannel: [Float] = [-1.0, -0.3, 0.3, 1.0]     // Medium range
        let bChannel: [Float] = [-5.0, -2.0, 2.0, 5.0]     // Large range
        let data = rChannel + gChannel + bChannel
        
        let params = Quantizer.calibratePerChannel(
            data: data,
            numChannels: numChannels,
            spatialSize: spatialSize,
            channelAxis: 0,
            dtype: .int8
        )
        
        result = """
        ✅ Per-Channel Calibration
        
        Channel Value Ranges:
        R: [\(String(format: "%.2f", params.minValues[0])), \(String(format: "%.2f", params.maxValues[0]))] (range: \(String(format: "%.2f", params.maxValues[0] - params.minValues[0])))
        G: [\(String(format: "%.2f", params.minValues[1])), \(String(format: "%.2f", params.maxValues[1]))] (range: \(String(format: "%.2f", params.maxValues[1] - params.minValues[1])))
        B: [\(String(format: "%.2f", params.minValues[2])), \(String(format: "%.2f", params.maxValues[2]))] (range: \(String(format: "%.2f", params.maxValues[2] - params.minValues[2])))
        
        Calculated Parameters (Int8):
        R: scale=\(String(format: "%.6f", params.scales[0])), zeroPoint=\(params.zeroPoints[0])
        G: scale=\(String(format: "%.6f", params.scales[1])), zeroPoint=\(params.zeroPoints[1])
        B: scale=\(String(format: "%.6f", params.scales[2])), zeroPoint=\(params.zeroPoints[2])
        
        Note: Each channel gets its own scale, preserving
        precision for channels with smaller value ranges.
        """
    }
}
