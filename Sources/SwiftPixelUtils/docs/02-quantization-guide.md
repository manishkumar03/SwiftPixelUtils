# Quantization for Mobile ML

A comprehensive reference for model quantization theory, implementation, and usage patterns for efficient ML inference on Apple platforms.

## Table of Contents

- [Introduction](#introduction)
- [What is Quantization?](#what-is-quantization)
  - [The Basic Concept](#the-basic-concept)
  - [Precision vs Efficiency Tradeoff](#precision-vs-efficiency-tradeoff)
- [Why Quantize?](#why-quantize)
  - [Model Size Reduction](#model-size-reduction)
  - [Inference Speed](#inference-speed)
  - [Memory Bandwidth](#memory-bandwidth)
  - [Power Consumption](#power-consumption)
  - [Hardware Acceleration](#hardware-acceleration)
- [Quantization Theory](#quantization-theory)
  - [The Quantization Formula](#the-quantization-formula)
  - [Scale Factor](#scale-factor)
  - [Zero Point](#zero-point)
  - [Symmetric vs Asymmetric Quantization](#symmetric-vs-asymmetric-quantization)
  - [Per-Tensor vs Per-Channel Quantization](#per-tensor-vs-per-channel-quantization)
  - [Dynamic Range Analysis](#dynamic-range-analysis)
- [Quantization Schemes](#quantization-schemes)
  - [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
  - [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
  - [Dynamic Quantization](#dynamic-quantization)
  - [Mixed-Precision Quantization](#mixed-precision-quantization)
- [Data Types for Quantization](#data-types-for-quantization)
  - [INT8 Quantization](#int8-quantization)
  - [UINT8 Quantization](#uint8-quantization)
  - [INT4 Quantization](#int4-quantization)
  - [FP16 (Float16)](#fp16-float16)
  - [BF16 (BFloat16)](#bf16-bfloat16)
- [Framework-Specific Quantization](#framework-specific-quantization)
  - [TensorFlow Lite](#tensorflow-lite)
  - [PyTorch / ExecuTorch](#pytorch--executorch)
  - [Core ML](#core-ml)
  - [ONNX](#onnx)
- [Input Preprocessing for Quantized Models](#input-preprocessing-for-quantized-models)
  - [Understanding Input Quantization Parameters](#understanding-input-quantization-parameters)
  - [Correct Preprocessing Pipeline](#correct-preprocessing-pipeline)
  - [Common Mistakes](#common-mistakes)
- [Output Dequantization](#output-dequantization)
  - [Converting Quantized Output to Float](#converting-quantized-output-to-float)
  - [Handling Classification Output](#handling-classification-output)
  - [Handling Detection Output](#handling-detection-output)
- [Accuracy Considerations](#accuracy-considerations)
  - [Expected Accuracy Drop](#expected-accuracy-drop)
  - [When Quantization Works Well](#when-quantization-works-well)
  - [When to Be Careful](#when-to-be-careful)
  - [Measuring Quantization Impact](#measuring-quantization-impact)
- [SwiftPixelUtils Quantization API](#swiftpixelutils-quantization-api)
- [Practical Examples](#practical-examples)
- [Troubleshooting](#troubleshooting)
- [Mathematical Foundations](#mathematical-foundations)

---

## Introduction

Quantization is one of the most important techniques for deploying machine learning models on mobile and edge devices. By reducing the numerical precision of model weights and activations, we can achieve significant reductions in model size, memory usage, and inference latency while maintaining acceptable accuracy.

This guide covers everything you need to know about quantization for ML inference on Apple platforms, from theory to practical implementation.

---

## What is Quantization?

### The Basic Concept

Quantization is the process of mapping continuous or high-precision values to a discrete, lower-precision representation.

```
Float32 (32 bits, ~7 decimal digits precision):
3.14159265358979...

    ↓ Quantize

Int8 (8 bits, 256 discrete values):
3

The float value 3.14159... gets mapped to the nearest integer.
```

In the context of neural networks, quantization typically refers to converting:
- **Weights:** The learned parameters of the model
- **Activations:** The intermediate outputs of each layer
- **Inputs/Outputs:** The model's input and output tensors

### Visual Representation

```
Floating Point Weights:              Quantized Weights (Int8):
┌──────────────────────────┐         ┌──────────────────────────┐
│  0.0234   -0.1562        │         │    3        -20          │
│  0.7891    0.0012        │   →     │   101         0          │
│ -0.4450    0.5673        │         │   -57        73          │
└──────────────────────────┘         └──────────────────────────┘
      Float32 (4 bytes each)               Int8 (1 byte each)
      Memory: 24 bytes                     Memory: 6 bytes
                                           4× smaller!
```

### Precision vs Efficiency Tradeoff

| Precision | Bits | Unique Values | Relative Size | Typical Accuracy Drop |
|-----------|------|---------------|---------------|----------------------|
| Float32 | 32 | 4.3 billion | 1.0× | Baseline |
| Float16 | 16 | 65,536 | 0.5× | <0.1% |
| Int8 | 8 | 256 | 0.25× | 0.5-2% |
| Int4 | 4 | 16 | 0.125× | 1-5% |
| Binary | 1 | 2 | 0.03× | 5-20% |

---

## Why Quantize?

### Model Size Reduction

Quantization directly reduces model size by using fewer bits per parameter.

```
Model: MobileNetV2 (~3.5M parameters)

Float32: 3.5M × 4 bytes = 14.0 MB
Float16: 3.5M × 2 bytes =  7.0 MB (2× smaller)
Int8:    3.5M × 1 byte  =  3.5 MB (4× smaller)
Int4:    3.5M × 0.5 bytes = 1.75 MB (8× smaller)
```

**Benefits:**
- Faster app downloads and updates
- Less storage on device
- Fits larger models in memory
- Smaller app bundle size

### Inference Speed

Lower precision means faster computation:

```
Float32 multiplication:  ~4 cycles
Int8 multiplication:     ~1 cycle

Additionally:
- More operations per SIMD register
- Better cache utilization
- Specialized hardware instructions
```

**Typical speedup: 1.5× to 4× depending on hardware and model**

### Memory Bandwidth

Memory bandwidth is often the bottleneck in ML inference:

```
Reading Float32 weights:  Need to transfer 4 bytes per weight
Reading Int8 weights:     Need to transfer 1 byte per weight

Same memory bandwidth → 4× more weights transferred → 4× faster
```

### Power Consumption

Integer operations use significantly less energy:

```
Energy per operation (approximate):
┌────────────────────┬────────────┐
│ Operation          │ Energy     │
├────────────────────┼────────────┤
│ Float32 multiply   │   ~30 pJ   │
│ Float16 multiply   │   ~3 pJ    │
│ Int8 multiply      │   ~0.2 pJ  │
└────────────────────┴────────────┘

Int8 is ~150× more energy efficient than Float32!
```

**Critical for:**
- Mobile devices (battery life)
- Always-on ML features
- Edge deployments

### Hardware Acceleration

Modern processors have specialized hardware for quantized operations:

**Apple Neural Engine (ANE):**
- Optimized for Int8 and Int16 operations
- 16× int8 operations per cycle vs 4× float16
- Available on A11 Bionic and later

**CPU SIMD:**
- AVX-512 VNNI: 4× int8 operations in one instruction
- ARM NEON: Efficient int8 dot products

---

## Quantization Theory

### The Quantization Formula

The fundamental equations for affine quantization:

**Quantize (float → int):**
$$q = \text{round}\left(\frac{r}{s}\right) + z$$

**Dequantize (int → float):**
$$r = s \cdot (q - z)$$

Where:
- $r$ = real (float) value
- $q$ = quantized (int) value
- $s$ = scale factor
- $z$ = zero point

### Scale Factor

The scale factor determines the step size between quantized values.

```
scale = (float_max - float_min) / (quant_max - quant_min)
```

**Example:**
```
Mapping float range [0, 1] to uint8 range [0, 255]:

scale = (1.0 - 0.0) / (255 - 0)
scale = 1/255 ≈ 0.00392157

Each integer step represents a change of 0.00392157 in float space.
```

**Smaller scale = finer granularity but smaller representable range**

### Zero Point

The zero point is the integer value that represents zero in the float domain.

```
zero_point = round(quant_min - float_min / scale)
```

**Why it matters:**
- Ensures exact representation of 0.0
- Critical for ReLU (many zeros in network)
- Affects multiplication efficiency

**Examples:**

| Float Range | Int Range | Scale | Zero Point |
|-------------|-----------|-------|------------|
| [0, 1] | [0, 255] (uint8) | 1/255 | 0 |
| [-1, 1] | [-128, 127] (int8) | 2/255 | 0 |
| [0, 1] | [-128, 127] (int8) | 1/255 | -128 |
| [-0.5, 1.5] | [0, 255] (uint8) | 2/255 | 64 |

### Symmetric vs Asymmetric Quantization

**Symmetric Quantization:**
- Zero point is fixed at 0 (or midpoint)
- Scale is symmetric around zero
- Faster computation (no zero point subtraction)

```
Symmetric int8: range [-127, 127], zero_point = 0
scale = max(|float_min|, |float_max|) / 127
```

**Asymmetric Quantization:**
- Zero point can be any value in quantization range
- Can represent asymmetric distributions better
- Slightly more computation needed

```
Asymmetric int8: range [-128, 127], zero_point varies
Better utilization when float values are mostly positive (like ReLU outputs)
```

### Per-Tensor vs Per-Channel Quantization

**Per-Tensor Quantization:**
- Single scale and zero_point for entire tensor
- Simpler, less metadata
- May lose precision if value ranges vary significantly

```
Entire weight tensor uses:
  scale = 0.0234
  zero_point = 0
```

**Per-Channel Quantization:**
- Different scale and zero_point for each channel
- Better accuracy (handles varying ranges)
- More metadata, slightly complex implementation

```
Channel 0: scale = 0.0234, zero_point = 0
Channel 1: scale = 0.0198, zero_point = -2
Channel 2: scale = 0.0312, zero_point = 1
...
```

**Per-channel is generally preferred for weights, per-tensor for activations.**

### Dynamic Range Analysis

To determine optimal quantization parameters, we need to analyze the range of values:

**For Weights (static):**
- Analyze weight tensor once during model conversion
- Use min/max values or percentile clipping (e.g., 99.9th percentile)

**For Activations (calibration):**
- Run representative input samples through the model
- Record activation statistics for each layer
- Determine scale/zero_point from collected statistics

```
Calibration methods:
1. MinMax: Use observed min/max values
2. Entropy: Minimize KL divergence between float and quantized distributions
3. MSE: Minimize mean squared error
4. Percentile: Use Nth percentile to clip outliers
```

---

## Quantization Schemes

### Post-Training Quantization (PTQ)

Quantize a pre-trained floating-point model without retraining.

```
Trained Float Model
        ↓
    Analyze weights
        ↓
    Calibrate activations
    (run representative data)
        ↓
    Quantized Model
```

**Advantages:**
- Simple and fast
- No training infrastructure needed
- Works with any pre-trained model

**Disadvantages:**
- May have accuracy loss (typically 1-3%)
- Sensitive to calibration data quality

**Best for:**
- Quick deployment
- Models with good quantization tolerance
- When training data is unavailable

**TensorFlow Lite PTQ:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Full integer quantization
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Calibration dataset
def representative_dataset():
    for sample in calibration_samples:
        yield [sample]
converter.representative_dataset = representative_dataset

quantized_model = converter.convert()
```

### Quantization-Aware Training (QAT)

Simulate quantization during training so the model learns to be robust.

```
Training Loop:
    Forward Pass:
        weights → fake_quantize → compute
        activations → fake_quantize → next layer
    
    Backward Pass:
        Use straight-through estimator
        (gradients flow through fake quantization)
    
Final Model:
    Export with real quantization
```

**Advantages:**
- Minimal accuracy loss (often <1%)
- Model learns to tolerate quantization noise
- Better for sensitive applications

**Disadvantages:**
- Requires retraining
- More complex training pipeline
- Need access to training data

**Best for:**
- Production deployments
- Accuracy-critical applications
- Models with significant PTQ accuracy drop

**PyTorch QAT:**
```python
import torch.quantization as quant

# Prepare model for QAT
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)

# Train with fake quantization
for epoch in range(num_epochs):
    train(model, train_loader)

# Convert to quantized model
model.eval()
quantized_model = quant.convert(model, inplace=False)
```

### Dynamic Quantization

Weights are quantized ahead of time, activations quantized dynamically at runtime.

```
Model loading:
    Weights: Float32 → Int8 (stored as int8)

Runtime:
    Load int8 weights → dequantize to float
    Compute in float
    No activation quantization
```

**Advantages:**
- Simple to apply
- No calibration needed
- Good for memory-bound models

**Disadvantages:**
- Less speedup than full quantization
- Weights still dequantized for computation

**Best for:**
- Large models where memory is the bottleneck
- When calibration data is unavailable
- Quick experiments

### Mixed-Precision Quantization

Different layers use different precision levels.

```
Layer sensitivity analysis:
    Conv1: Low sensitivity  → Int8
    Conv2: High sensitivity → Int16
    FC1:   Medium           → Int8
    Output: Critical        → Float32
```

**Automatic mixed-precision tools:**
- TensorFlow Model Optimization Toolkit
- NVIDIA TensorRT
- Apple Neural Engine (automatic)

---

## Data Types for Quantization

### INT8 Quantization

**Signed 8-bit integer:** Range [-128, 127]

```
Bit pattern: [sign][7 value bits]
Range: -2^7 to 2^7 - 1 = -128 to 127
Values: 256 unique values
```

**Common uses:**
- Weights (symmetric around 0)
- Activations after normalization

**Advantages:**
- Symmetric range for weights
- Good hardware support
- Standard in most frameworks

### UINT8 Quantization

**Unsigned 8-bit integer:** Range [0, 255]

```
Bit pattern: [8 value bits]
Range: 0 to 2^8 - 1 = 0 to 255
Values: 256 unique values
```

**Common uses:**
- Activations after ReLU (all non-negative)
- Pixel values
- TFLite quantized model inputs

**Advantages:**
- Natural for pixel values
- Full range for positive values

### INT4 Quantization

**4-bit integer:** Range [-8, 7] or [0, 15]

```
Bit pattern: [4 bits]
Values: 16 unique values
Two values per byte (packed)
```

**Use cases:**
- Large language models (LLMs)
- Extreme compression needs
- When accuracy drop is acceptable

**Challenges:**
- Significant accuracy impact
- Complex packing/unpacking
- Limited hardware support

### FP16 (Float16)

**IEEE 754 half-precision:** Range ±65504

```
Bit pattern: [1 sign][5 exponent][10 mantissa]
Precision: ~3 decimal digits
Special values: ±Inf, NaN, denormals
```

**Advantages:**
- Native GPU support
- No calibration needed
- Very small accuracy drop

**Disadvantages:**
- Only 2× size reduction
- Overflow risk with large values
- Less efficient than int8

### BF16 (BFloat16)

**Brain floating point:** Same range as float32

```
Bit pattern: [1 sign][8 exponent][7 mantissa]
Range: Same as Float32 (±3.4×10^38)
Precision: ~2 decimal digits
```

**Advantages:**
- No overflow (same exponent as float32)
- Trivial conversion from float32
- Good for training

**Disadvantages:**
- Less precision than FP16
- Limited hardware support (newer chips only)

---

## Framework-Specific Quantization

### TensorFlow Lite

**Quantization modes:**

**1. Dynamic Range Quantization:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**2. Full Integer Quantization (int8 weights and activations):**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset
```

**3. Float16 Quantization:**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

**Reading TFLite quantization parameters in Swift:**
```swift
let interpreter = try Interpreter(modelPath: modelPath)
try interpreter.allocateTensors()

let inputTensor = try interpreter.input(at: 0)

if let quantParams = inputTensor.quantizationParameters {
    print("Scale: \(quantParams.scale)")
    print("Zero Point: \(quantParams.zeroPoint)")
} else {
    print("Model input is not quantized")
}
```

### PyTorch / ExecuTorch

**Post-Training Static Quantization:**
```python
import torch.quantization as quant

# Prepare model
model.eval()
model.qconfig = quant.get_default_qconfig('qnnpack')  # Mobile-optimized
quant.prepare(model, inplace=True)

# Calibrate
with torch.no_grad():
    for batch in calibration_data:
        model(batch)

# Convert
quantized_model = quant.convert(model, inplace=False)

# Export for mobile
torch.jit.save(torch.jit.script(quantized_model), "quantized_model.pt")
```

**ExecuTorch Quantization:**
```python
from executorch.backends.xnnpack.quantizer import XNNPACKQuantizer
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

quantizer = XNNPACKQuantizer()
prepared = prepare_pt2e(model, quantizer)

# Calibrate
with torch.no_grad():
    for batch in calibration_data:
        prepared(batch)

quantized = convert_pt2e(prepared)
```

### Core ML

**Core ML Tools Quantization:**
```python
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# Load model
model = ct.models.MLModel("model.mlpackage")

# Quantize to 8-bit weights
quantized_model = quantization_utils.quantize_weights(model, nbits=8)
quantized_model.save("model_quantized.mlpackage")
```

**Linear Quantization (palettization):**
```python
# Use lookup table instead of direct values
config = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8"
)
quantized = ct.optimize.coreml.linear_quantize_weights(
    model, config
)
```

### ONNX

**ONNX Runtime Quantization:**
```python
from onnxruntime.quantization import quantize_dynamic, quantize_static

# Dynamic quantization
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8
)

# Static quantization (needs calibration)
quantize_static(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ
)
```

---

## Input Preprocessing for Quantized Models

### Understanding Input Quantization Parameters

Quantized models expect quantized inputs. You need to:
1. Normalize/preprocess as usual (float operations)
2. Quantize the preprocessed data using model's input parameters

**Reading model's expected input parameters:**
```swift
// TFLite
let inputTensor = try interpreter.input(at: 0)
let quantParams = inputTensor.quantizationParameters!

print("Input scale: \(quantParams.scale)")
print("Input zero point: \(quantParams.zeroPoint)")
print("Input dtype: \(inputTensor.dataType)")  // uint8 or int8
```

**Common configurations:**

| Model Type | Input Range | Scale | Zero Point | Data Type |
|------------|-------------|-------|------------|-----------|
| TFLite Quantized (typical) | [0, 255] | 1.0 | 0 | UInt8 |
| TFLite Quantized ([0,1] normalized) | [0, 1] | 1/255 | 0 | UInt8 |
| TFLite Int8 | [-1, 1] | 2/255 | 0 | Int8 |
| Custom | Varies | Varies | Varies | Varies |

### Correct Preprocessing Pipeline

**For TFLite quantized model expecting [0, 255] input:**
```swift
// Model expects raw pixel values as uint8
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteQuantized,  // Returns [0,255] UInt8
    width: 224,
    height: 224
)

// Copy uint8 data directly
try interpreter.copy(Data(input.dataUInt8), toInputAt: 0)
```

**For TFLite quantized model expecting normalized then quantized:**
```swift
// Model was trained with normalized inputs, then quantized
// Input scale and zero point encode the normalization

let inputTensor = try interpreter.input(at: 0)
let scale = inputTensor.quantizationParameters!.scale
let zeroPoint = inputTensor.quantizationParameters!.zeroPoint

// Get normalized float data
let floatData = try await PixelExtractor.getPixelData(
    source: .uiImage(image),
    options: PixelDataOptions(
        targetSize: CGSize(width: 224, height: 224),
        normalization: .zeroToOne  // Match model's training
    )
)

// Quantize using model's parameters
let quantized = Quantizer.quantize(
    floatData.data,
    to: .uint8,
    scale: scale,
    zeroPoint: zeroPoint
)

try interpreter.copy(Data(quantized), toInputAt: 0)
```

### Common Mistakes

**1. Feeding float data to quantized model:**
```swift
// WRONG - Model expects uint8!
let floatData: [Float] = ...
try interpreter.copy(Data(bytes: floatData, count: floatData.count * 4), toInputAt: 0)

// RIGHT - Convert to uint8 first
let uint8Data: [UInt8] = ...
try interpreter.copy(Data(uint8Data), toInputAt: 0)
```

**2. Ignoring quantization parameters:**
```swift
// WRONG - Assuming no scaling needed
let raw = image.pixelData  // [0-255]
try interpreter.copy(Data(raw), toInputAt: 0)

// RIGHT - Apply model's quantization parameters
let normalized = raw.map { Float($0) / 255.0 }
let quantized = Quantizer.quantize(normalized, to: .uint8, scale: scale, zeroPoint: zeroPoint)
```

**3. Using wrong quantization parameters:**
```swift
// WRONG - Hardcoded values
let quantized = raw.map { UInt8(clamping: Int($0 * 255)) }

// RIGHT - Use model's actual parameters
let scale = inputTensor.quantizationParameters!.scale
let zeroPoint = inputTensor.quantizationParameters!.zeroPoint
let quantized = Quantizer.quantize(floatData, scale: scale, zeroPoint: zeroPoint)
```

---

## Output Dequantization

### Converting Quantized Output to Float

Model outputs quantized values that need dequantization for interpretation.

```swift
// Get output tensor and its quantization parameters
let outputTensor = try interpreter.output(at: 0)
let scale = outputTensor.quantizationParameters!.scale
let zeroPoint = outputTensor.quantizationParameters!.zeroPoint

// Read quantized output
let quantizedOutput: [UInt8] = outputTensor.data.withUnsafeBytes { Array($0) }

// Dequantize
let floatOutput = Quantizer.dequantize(
    quantizedOutput,
    from: .uint8,
    scale: scale,
    zeroPoint: zeroPoint
)
// floatOutput is now [Float] ready for postprocessing
```

### Handling Classification Output

```swift
func classifyQuantized(interpreter: Interpreter) throws -> [(label: String, confidence: Float)] {
    // Get quantized output
    let outputTensor = try interpreter.output(at: 0)
    let quantParams = outputTensor.quantizationParameters!
    
    let quantizedOutput: [UInt8] = outputTensor.data.withUnsafeBytes { Array($0) }
    
    // Dequantize to get probabilities
    let probabilities = Quantizer.dequantize(
        quantizedOutput,
        from: .uint8,
        scale: quantParams.scale,
        zeroPoint: quantParams.zeroPoint
    )
    
    // Process as normal
    let result = try ClassificationOutput.process(
        floatOutput: probabilities,
        labels: .imagenet,
        topK: 5
    )
    
    return result.predictions.map { ($0.label, $0.confidence) }
}
```

### Handling Detection Output

```swift
func detectQuantized(interpreter: Interpreter, imageSize: CGSize) throws -> DetectionResult {
    // Get quantized output
    let outputTensor = try interpreter.output(at: 0)
    let quantParams = outputTensor.quantizationParameters!
    
    // Shape: [1, num_detections, 85] for YOLO
    let quantizedOutput: [UInt8] = outputTensor.data.withUnsafeBytes { Array($0) }
    
    // Dequantize
    let floatOutput = Quantizer.dequantize(
        quantizedOutput,
        from: .uint8,
        scale: quantParams.scale,
        zeroPoint: quantParams.zeroPoint
    )
    
    // Process detections
    return try DetectionOutput.process(
        floatOutput: floatOutput,
        format: .yolov5(numClasses: 80),
        labels: .coco,
        imageSize: imageSize
    )
}
```

---

## Accuracy Considerations

### Expected Accuracy Drop

Typical accuracy impact by quantization scheme:

| Scheme | Weights | Activations | Accuracy Drop |
|--------|---------|-------------|---------------|
| FP16 | Float16 | Float16 | <0.1% |
| Dynamic | Int8 | Float32 | <0.5% |
| Static PTQ | Int8 | Int8 | 0.5-2% |
| QAT | Int8 | Int8 | <0.5% |
| Int4 | Int4 | Int8 | 2-5% |

### When Quantization Works Well

**Good candidates for quantization:**
- Classification models (ResNet, EfficientNet, MobileNet)
- Object detection models (YOLO, SSD)
- Models with batch normalization
- Models with ReLU activations (bounded range)
- Models trained with data augmentation (more robust)

**Characteristics that help:**
- Large number of parameters (more redundancy)
- Well-conditioned weight distributions
- Not many outliers in activations

### When to Be Careful

**Challenging cases:**
- Regression tasks (continuous outputs)
- Models with very small/narrow layers
- Models with large dynamic range in weights
- Generative models (GANs, diffusion)
- Models with attention mechanisms (sometimes)

**High-stakes applications:**
- Medical diagnosis
- Safety-critical systems
- Financial decisions

**Always validate on your specific use case!**

### Measuring Quantization Impact

```swift
// Compare float vs quantized predictions
func measureQuantizationImpact(
    floatModel: Interpreter,
    quantizedModel: Interpreter,
    testImages: [UIImage]
) async throws -> QuantizationMetrics {
    var floatPredictions: [[Float]] = []
    var quantizedPredictions: [[Float]] = []
    
    for image in testImages {
        // Run float model
        let floatInput = try await prepareFloatInput(image)
        let floatOutput = try runInference(floatModel, input: floatInput)
        floatPredictions.append(floatOutput)
        
        // Run quantized model
        let quantInput = try await prepareQuantizedInput(image)
        let quantOutput = try runQuantizedInference(quantizedModel, input: quantInput)
        quantizedPredictions.append(quantOutput)
    }
    
    return QuantizationMetrics(
        mse: computeMSE(floatPredictions, quantizedPredictions),
        topKAgreement: computeTopKAgreement(floatPredictions, quantizedPredictions, k: 5),
        maxAbsDiff: computeMaxAbsDiff(floatPredictions, quantizedPredictions)
    )
}
```

---

## SwiftPixelUtils Quantization API

### Quantizer Class

```swift
// Quantize float array to integer type
let quantized = Quantizer.quantize(
    floatData,           // [Float]
    to: .uint8,          // Target type: .uint8, .int8
    scale: 0.00392157,   // From model's quantization parameters
    zeroPoint: 0         // From model's quantization parameters
)

// Dequantize integer array back to float
let dequantized = Quantizer.dequantize(
    quantizedData,       // [UInt8] or [Int8]
    from: .uint8,        // Source type
    scale: scale,
    zeroPoint: zeroPoint
)
```

### Per-Channel Quantization

```swift
// Per-channel quantization (different scale/zp per channel)
let perChannelQuantized = Quantizer.quantizePerChannel(
    floatData,                        // [Float]
    to: .int8,
    scales: [0.0039, 0.0041, 0.0038], // Per RGB channel
    zeroPoints: [0, 0, 0],
    channelAxis: 2,                    // Channel dimension index
    shape: [224, 224, 3]              // Tensor shape
)
```

### Convenience Methods

```swift
// One-line quantized model input
let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .tfliteQuantized,  // Automatic uint8 output
    width: 224,
    height: 224
)

// Access quantized data
let uint8Data: [UInt8] = input.dataUInt8
```

### Custom Quantization Configuration

```swift
let customConfig = QuantizationConfig(
    inputType: .uint8,
    inputScale: 0.00392157,
    inputZeroPoint: 0,
    outputType: .uint8,
    outputScale: 0.00784314,
    outputZeroPoint: 128
)

let input = try await PixelExtractor.getModelInput(
    source: .uiImage(image),
    framework: .custom(quantization: customConfig),
    width: 224,
    height: 224
)
```

---

## Practical Examples

### Example 1: MobileNet Quantized Classification

```swift
func classifyWithQuantizedMobileNet(_ image: UIImage) async throws -> ClassificationResult {
    // 1. Load quantized model
    let interpreter = try Interpreter(modelPath: "mobilenet_v2_quantized.tflite")
    try interpreter.allocateTensors()
    
    // 2. Check input requirements
    let inputTensor = try interpreter.input(at: 0)
    print("Input type: \(inputTensor.dataType)")  // Should be uint8
    print("Input shape: \(inputTensor.shape)")    // [1, 224, 224, 3]
    
    // 3. Preprocess to uint8
    let input = try await PixelExtractor.getModelInput(
        source: .uiImage(image),
        framework: .tfliteQuantized,
        width: 224,
        height: 224
    )
    
    // 4. Run inference
    try interpreter.copy(Data(input.dataUInt8), toInputAt: 0)
    try interpreter.invoke()
    
    // 5. Get quantized output
    let outputTensor = try interpreter.output(at: 0)
    let quantParams = outputTensor.quantizationParameters!
    let quantizedOutput: [UInt8] = outputTensor.data.withUnsafeBytes { Array($0) }
    
    // 6. Dequantize
    let floatOutput = Quantizer.dequantize(
        quantizedOutput,
        from: .uint8,
        scale: quantParams.scale,
        zeroPoint: quantParams.zeroPoint
    )
    
    // 7. Process results
    return try ClassificationOutput.process(
        floatOutput: floatOutput,
        labels: .imagenet,
        topK: 5
    )
}
```

### Example 2: YOLO Quantized Detection

```swift
func detectWithQuantizedYOLO(_ image: UIImage) async throws -> DetectionResult {
    let interpreter = try Interpreter(modelPath: "yolov5s_quantized.tflite")
    try interpreter.allocateTensors()
    
    // Letterbox preprocessing
    let letterboxed = try Letterbox.apply(
        to: image,
        targetSize: CGSize(width: 640, height: 640),
        color: (114, 114, 114)
    )
    
    // Get quantized input
    let inputTensor = try interpreter.input(at: 0)
    let inputScale = inputTensor.quantizationParameters?.scale ?? (1.0/255.0)
    let inputZeroPoint = inputTensor.quantizationParameters?.zeroPoint ?? 0
    
    // Prepare input (normalize to [0,1] then quantize)
    let floatData = try await PixelExtractor.getPixelData(
        source: .cgImage(letterboxed.image),
        options: PixelDataOptions(
            targetSize: CGSize(width: 640, height: 640),
            normalization: .zeroToOne,
            dataLayout: .nhwc
        )
    )
    
    let quantizedInput = Quantizer.quantize(
        floatData.data,
        to: .uint8,
        scale: inputScale,
        zeroPoint: inputZeroPoint
    )
    
    // Run inference
    try interpreter.copy(Data(quantizedInput), toInputAt: 0)
    try interpreter.invoke()
    
    // Dequantize output
    let outputTensor = try interpreter.output(at: 0)
    let outputScale = outputTensor.quantizationParameters!.scale
    let outputZeroPoint = outputTensor.quantizationParameters!.zeroPoint
    let quantizedOutput: [UInt8] = outputTensor.data.withUnsafeBytes { Array($0) }
    
    let floatOutput = Quantizer.dequantize(
        quantizedOutput,
        from: .uint8,
        scale: outputScale,
        zeroPoint: outputZeroPoint
    )
    
    // Process detections
    var result = try DetectionOutput.process(
        floatOutput: floatOutput,
        format: .yolov5(numClasses: 80),
        labels: .coco,
        imageSize: CGSize(width: 640, height: 640)
    )
    
    // Reverse letterbox transformation
    result = Letterbox.reverseTransform(
        detections: result,
        letterboxInfo: letterboxed.info
    )
    
    return result
}
```

---

## Troubleshooting

### Problem: Very Wrong Predictions

**Possible causes:**
1. Using wrong input type (float vs uint8)
2. Wrong quantization parameters
3. Missing normalization step

**Debug steps:**
```swift
// 1. Check input tensor type
let inputTensor = try interpreter.input(at: 0)
print("Expected type: \(inputTensor.dataType)")

// 2. Check quantization parameters
if let params = inputTensor.quantizationParameters {
    print("Scale: \(params.scale), ZP: \(params.zeroPoint)")
} else {
    print("Not quantized - use float input")
}

// 3. Verify input data range
print("Input data range: \(inputData.min()!) - \(inputData.max()!)")
```

### Problem: Overflow or Clipping

**Symptom:** Output values are all at min/max

**Cause:** Values outside quantization range

**Solution:**
```swift
// Clip values before quantization
let clipped = floatData.map { max(floatMin, min(floatMax, $0)) }
let quantized = Quantizer.quantize(clipped, ...)
```

### Problem: Accuracy Much Worse Than Expected

**Possible causes:**
1. Per-channel vs per-tensor mismatch
2. Calibration data not representative
3. Model not suitable for quantization

**Debug:**
```swift
// Compare intermediate layer outputs
func compareActivations(floatModel: Model, quantModel: Model, input: Data) {
    // Get activations from both models
    // Compare statistics (mean, std, histogram)
}
```

---

## Mathematical Foundations

### Quantization Error Analysis

**Quantization error:**
$$e = r - s \cdot (q - z)$$

**Expected error (uniform quantization):**
$$E[e^2] = \frac{s^2}{12}$$

**Signal-to-quantization-noise ratio (SQNR):**
$$SQNR = 10 \log_{10}\left(\frac{E[r^2]}{E[e^2]}\right) \approx 6.02n + 1.76 \text{ dB}$$

Where $n$ is the number of bits.

### Optimal Scale Selection

**Min-max scaling:**
$$s = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

**MSE-optimal scaling:**
$$s^* = \arg\min_s \sum_i (r_i - s \cdot q_i)^2$$

### Gradient Computation (QAT)

**Straight-through estimator (STE):**
$$\frac{\partial L}{\partial r} = \frac{\partial L}{\partial q} \cdot \frac{\partial q}{\partial r} \approx \frac{\partial L}{\partial q} \cdot 1$$

Gradient flows through quantization as if it were identity.

---

## Next Steps

- [Classification Output](03-classification-output.md) - Process classifier outputs
- [Detection Output](04-detection-output.md) - YOLO postprocessing
- [Segmentation Output](05-segmentation-output.md) - Semantic segmentation
