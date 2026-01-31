//
//  QuantizationView.swift
//  SwiftPixelUtilsExampleApp
//
//  Quantization demo view
//

import SwiftUI
import SwiftPixelUtils

// MARK: - Quantization Demo
struct QuantizationView: View {
    @State private var result = "Tap to test quantization"
    
    var body: some View {
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
