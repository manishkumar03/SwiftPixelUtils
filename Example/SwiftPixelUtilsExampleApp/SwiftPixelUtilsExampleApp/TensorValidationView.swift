import SwiftUI
import SwiftPixelUtils

struct TensorValidationView: View {
    @State private var result = "Tap to validate tensors"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                GroupBox("Validation") {
                    VStack(spacing: 12) {
                        Button("Validate ImageNet Tensor") {
                            testImageNetValidation()
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Validate TensorFlow Tensor") {
                            testTensorFlowValidation()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Validate Scaled [0,1] Tensor") {
                            testScaledValidation()
                        }
                        .buttonStyle(.bordered)
                        
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
    
    func testTensorFlowValidation() {
        // TensorFlow uses [-1, 1] range normalization
        let tensor: [Float] = Array(repeating: 0.0, count: 224 * 224 * 3)
        
        let validation = TensorValidation.validateTensorFlowTensor(data: tensor, width: 224, height: 224)
        
        result = """
        ✅ TensorFlow Tensor Validation
        Valid: \(validation.isValid)
        Size: \(tensor.count) elements
        Expected range: [-1, 1]
        
        Statistics:
        Min: \(String(format: "%.4f", validation.statistics?.min ?? 0))
        Max: \(String(format: "%.4f", validation.statistics?.max ?? 0))
        Mean: \(String(format: "%.4f", validation.statistics?.mean ?? 0))
        
        Errors: \(validation.errors.isEmpty ? "None" : validation.errors.joined(separator: ", "))
        """
    }
    
    func testScaledValidation() {
        // Test scaled [0, 1] range validation
        let validTensor: [Float] = [0.1, 0.5, 0.8, 0.95, 0.2, 0.6]
        let invalidTensor: [Float] = [0.1, 0.5, 1.2, -0.1, 0.2, 0.6]  // out of range
        
        let validResult = TensorValidation.validateScaledTensor(data: validTensor)
        let invalidResult = TensorValidation.validateScaledTensor(data: invalidTensor)
        
        result = """
        ✅ Scaled [0,1] Tensor Validation
        
        Valid tensor [0.1, 0.5, 0.8, 0.95, 0.2, 0.6]:
          Valid: \(validResult.isValid)
          Errors: \(validResult.errors.isEmpty ? "None" : validResult.errors.joined(separator: ", "))
        
        Invalid tensor [0.1, 0.5, 1.2, -0.1, 0.2, 0.6]:
          Valid: \(invalidResult.isValid)
          Errors: \(invalidResult.errors.isEmpty ? "None" : invalidResult.errors.joined(separator: ", "))
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
