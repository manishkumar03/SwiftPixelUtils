# Image Classification: Theory and Implementation

A comprehensive reference for image classification concepts, neural network architectures, output processing, and implementation patterns using SwiftPixelUtils.

## Table of Contents

- [Introduction](#introduction)
- [What is Image Classification?](#what-is-image-classification)
  - [Task Definition](#task-definition)
  - [Applications](#applications)
  - [Single-Label vs Multi-Label](#single-label-vs-multi-label)
- [How Classification Networks Work](#how-classification-networks-work)
  - [The Complete Pipeline](#the-complete-pipeline)
  - [Feature Extraction](#feature-extraction)
  - [Feature Hierarchy](#feature-hierarchy)
  - [The Classification Head](#the-classification-head)
- [Understanding Model Outputs](#understanding-model-outputs)
  - [Raw Logits](#raw-logits)
  - [Softmax Probabilities](#softmax-probabilities)
  - [Log-Softmax](#log-softmax)
  - [Sigmoid for Multi-Label](#sigmoid-for-multi-label)
  - [Output Format by Framework](#output-format-by-framework)
- [Softmax Deep Dive](#softmax-deep-dive)
  - [The Softmax Function](#the-softmax-function)
  - [Numerical Stability](#numerical-stability)
  - [Temperature Scaling](#temperature-scaling)
  - [Softmax Properties](#softmax-properties)
- [Top-K Predictions](#top-k-predictions)
  - [Why Top-K?](#why-top-k)
  - [Efficient Top-K Algorithms](#efficient-top-k-algorithms)
  - [Confidence Thresholds](#confidence-thresholds)
- [Common Architectures](#common-architectures)
  - [MobileNet Family](#mobilenet-family)
  - [EfficientNet Family](#efficientnet-family)
  - [ResNet Family](#resnet-family)
  - [Vision Transformers (ViT)](#vision-transformers-vit)
  - [ConvNeXt](#convnext)
  - [Architecture Comparison](#architecture-comparison)
- [Label Databases](#label-databases)
  - [ImageNet-1K](#imagenet-1k)
  - [ImageNet-21K](#imagenet-21k)
  - [CIFAR-10 and CIFAR-100](#cifar-10-and-cifar-100)
  - [Places365](#places365)
  - [Custom Labels](#custom-labels)
- [Transfer Learning](#transfer-learning)
  - [Feature Extraction](#feature-extraction-1)
  - [Fine-Tuning](#fine-tuning)
  - [When to Use What](#when-to-use-what)
- [Confidence Calibration](#confidence-calibration)
  - [The Overconfidence Problem](#the-overconfidence-problem)
  - [Temperature Scaling for Calibration](#temperature-scaling-for-calibration)
  - [Expected Calibration Error](#expected-calibration-error)
- [SwiftPixelUtils Classification API](#swiftpixelutils-classification-api)
  - [Basic Usage](#basic-usage)
  - [Output Formats](#output-formats)
  - [Advanced Options](#advanced-options)
  - [ClassificationResult Properties](#classificationresult-properties)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Mathematical Foundations](#mathematical-foundations)

---

## Introduction

Image classification is the foundational task of computer vision, where a model assigns one or more labels to an entire image from a predefined set of categories. This guide covers everything from the theory behind classification networks to practical implementation using SwiftPixelUtils.

---

## What is Image Classification?

### Task Definition

Given an input image, predict which class (or classes) it belongs to from a predefined set.

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────────┐
│                 │       │                 │       │                     │
│   🐕 Image      │   →   │  Neural Network │   →   │  "Golden Retriever" │
│                 │       │                 │       │      (97.3%)        │
│                 │       │                 │       │                     │
└─────────────────┘       └─────────────────┘       └─────────────────────┘
```

**Formal Definition:**
- Input: Image $x \in \mathbb{R}^{H \times W \times C}$
- Output: Class label $y \in \{1, 2, ..., K\}$ or probability distribution $p(y|x)$
- Model: $f_\theta: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^K$

### Applications

| Domain | Example Classes | Use Case |
|--------|-----------------|----------|
| **Photo Organization** | cat, dog, car, person, beach, mountain | Automatic album sorting, search |
| **Medical Imaging** | benign, malignant, uncertain | Cancer screening, diagnosis |
| **Quality Control** | pass, fail, defect_type_A, defect_type_B | Manufacturing inspection |
| **Content Moderation** | safe, explicit, violence, spam | Social media filtering |
| **Species Identification** | 1000+ bird/plant species | Nature apps, biodiversity |
| **Food Recognition** | 100+ food types | Calorie tracking, dietary apps |
| **Document Classification** | invoice, receipt, form, letter | Document processing |
| **Retail** | product categories | Inventory, visual search |
| **Agriculture** | healthy, diseased, pest_damage | Crop monitoring |
| **Fashion** | clothing categories, styles | Recommendation, tagging |

### Single-Label vs Multi-Label

**Single-Label Classification:**
- Each image belongs to exactly one class
- Use softmax activation (outputs sum to 1)
- Example: ImageNet (one dominant object)

**Multi-Label Classification:**
- Each image can have multiple labels
- Use sigmoid activation (independent probabilities)
- Example: Movie genres, image tags

```
Single-label:     Multi-label:
P(dog) = 0.9      P(outdoor) = 0.95
P(cat) = 0.1      P(sunny) = 0.87
Sum = 1.0         P(people) = 0.72
                  P(beach) = 0.45
                  (independent probabilities)
```

---

## How Classification Networks Work

### The Complete Pipeline

```
Input Image (224×224×3 RGB)
         │
         ▼
┌────────────────────────────────────┐
│       Convolutional Backbone       │
│  ┌──────────────────────────────┐  │
│  │ Conv → BN → ReLU → Pool      │  │  Spatial: 224 → 112
│  │ (extract edges, colors)      │  │  Channels: 3 → 64
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Conv blocks (textures)       │  │  Spatial: 112 → 56
│  │                              │  │  Channels: 64 → 128
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Conv blocks (patterns)       │  │  Spatial: 56 → 28
│  │                              │  │  Channels: 128 → 256
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Conv blocks (parts)          │  │  Spatial: 28 → 14
│  │                              │  │  Channels: 256 → 512
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Conv blocks (objects)        │  │  Spatial: 14 → 7
│  │                              │  │  Channels: 512 → 1024
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
         │
         ▼ Feature map: 7×7×1024
┌────────────────────────────────────┐
│     Global Average Pooling         │
│     7×7×1024 → 1×1×1024            │
└────────────────────────────────────┘
         │
         ▼ Feature vector: 1024
┌────────────────────────────────────┐
│     Classification Head            │
│     Linear: 1024 → 1000            │
└────────────────────────────────────┘
         │
         ▼ Logits: 1000
┌────────────────────────────────────┐
│     Softmax                        │
└────────────────────────────────────┘
         │
         ▼ Probabilities: 1000
   [0.002, 0.001, ..., 0.973, ...]
```

### Feature Extraction

The convolutional backbone transforms raw pixels into increasingly abstract features:

```
Layer 1 output:
┌─────────────────┐
│ Responds to     │  Edges, gradients, colors
│ local patterns  │  Receptive field: 3×3 to 7×7 pixels
└─────────────────┘

Layer 3 output:
┌─────────────────┐
│ Responds to     │  Textures, corners, curves
│ texture patches │  Receptive field: ~30×30 pixels
└─────────────────┘

Layer 5 output:
┌─────────────────┐
│ Responds to     │  Parts: eyes, wheels, leaves
│ object parts    │  Receptive field: ~100×100 pixels
└─────────────────┘

Final layer output:
┌─────────────────┐
│ Responds to     │  Whole objects, scenes
│ whole objects   │  Receptive field: entire image
└─────────────────┘
```

### Feature Hierarchy

```
       Abstraction Level
            ↑
            │  ┌────────────────────┐
    High    │  │ "Dog", "Car"       │  Semantic concepts
            │  └────────────────────┘
            │  ┌────────────────────┐
            │  │ Face, Wheel        │  Object parts
            │  └────────────────────┘
            │  ┌────────────────────┐
            │  │ Fur texture, Metal │  Textures, materials
            │  └────────────────────┘
            │  ┌────────────────────┐
    Low     │  │ Edges, Gradients   │  Low-level features
            │  └────────────────────┘
            └───────────────────────→
                Network Depth
```

### The Classification Head

Converts spatial features to class scores:

```
Feature Map: H×W×C
      │
      ▼
Global Average Pooling (GAP):
      pool over H×W → 1×1×C
      │
      ▼
Flatten: 1×1×C → C
      │
      ▼
Fully Connected: C → num_classes
      │
      ▼
Logits: [score_0, score_1, ..., score_K-1]
```

**Why Global Average Pooling?**
- Reduces parameters (no FC on spatial dimensions)
- Translation invariance
- Less overfitting
- Works with variable input sizes

---

## Understanding Model Outputs

### Raw Logits

The model's final layer outputs "logits" – unbounded real numbers representing evidence for each class.

```swift
// Example: 1000-class ImageNet model output
let logits: [Float] = [
     2.1,    // class 0: tench
    -0.5,    // class 1: goldfish
     4.8,    // class 2: great white shark
    -2.3,    // class 3: tiger shark
     ...     // 996 more values
   -1.2      // class 999: toilet tissue
]
```

**Properties of logits:**
- Can be any real number (positive or negative)
- Higher value = stronger evidence for that class
- No upper or lower bound
- Don't sum to any particular value
- Not directly interpretable as probabilities

### Softmax Probabilities

Softmax transforms logits into a valid probability distribution:

```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

Example:
logits = [2.1, -0.5, 4.8, -2.3]

exp(logits) = [8.17, 0.61, 121.51, 0.10]
sum = 130.39

softmax = [0.063, 0.005, 0.932, 0.001]
```

**Properties of softmax output:**
- All values in range (0, 1)
- Values sum to exactly 1.0
- Relative ordering preserved
- Interpretable as confidences

### Log-Softmax

Some models output log-softmax for numerical stability during training:

```
log_softmax(z_i) = z_i - log(Σ_j exp(z_j))

To convert back:
probability = exp(log_probability)
```

**Why log-softmax?**
- More stable gradient computation
- Works better with negative log-likelihood loss
- Common in PyTorch models

### Sigmoid for Multi-Label

For multi-label classification, use sigmoid instead of softmax:

```
sigmoid(z_i) = 1 / (1 + exp(-z_i))

Each output is independent:
P(outdoor) = sigmoid(logit_outdoor) = 0.95
P(sunny) = sigmoid(logit_sunny) = 0.87
P(beach) = sigmoid(logit_beach) = 0.45
```

### Output Format by Framework

| Framework/Model | Typical Output | Notes |
|-----------------|----------------|-------|
| TFLite (.tflite) | Probabilities | Softmax usually included |
| CoreML (.mlpackage) | Probabilities | With class labels dict |
| PyTorch (.pt) | Logits | Apply softmax manually |
| ONNX (.onnx) | Varies | Check model export |
| Keras (.h5) | Usually probabilities | Depends on final layer |

```swift
// SwiftPixelUtils handles both automatically
let result = try ClassificationOutput.process(
    outputData: modelOutput,
    labels: .imagenet,
    outputFormat: .auto  // Detects logits vs probabilities
)
```

---

## Softmax Deep Dive

### The Softmax Function

**Mathematical definition:**
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Intuition:**
- Exponential emphasizes differences (larger logits get much larger probabilities)
- Normalization ensures valid probability distribution

**Gradient:**
$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j)$$

### Numerical Stability

Naive softmax implementation can overflow:

```swift
// WRONG - can overflow
func naiveSoftmax(_ logits: [Float]) -> [Float] {
    let exps = logits.map { exp($0) }  // exp(1000) = Inf!
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}
```

**Stable implementation** (subtract max before exp):

```swift
// CORRECT - numerically stable
func stableSoftmax(_ logits: [Float]) -> [Float] {
    let maxLogit = logits.max()!
    let exps = logits.map { exp($0 - maxLogit) }  // Max exp is 1
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}
```

**Why it works:**
$$\text{softmax}(z - c)_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}e^{-c}}{\sum_j e^{z_j}e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The constant $c$ cancels out.

### Temperature Scaling

Temperature controls "sharpness" of the distribution:

$$\text{softmax}(z/T)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T < 1 | Sharper (more confident) | Knowledge distillation |
| T = 1 | Standard | Normal inference |
| T > 1 | Softer (less confident) | Calibration, exploration |
| T → 0 | argmax (one-hot) | Hard predictions |
| T → ∞ | Uniform | Maximum entropy |

**Visual example:**
```
Logits: [2.0, 1.0, 0.5]

T=0.5:  [0.84, 0.12, 0.04]  ← Very confident
T=1.0:  [0.66, 0.24, 0.10]  ← Standard
T=2.0:  [0.46, 0.32, 0.22]  ← Less confident
T=10:   [0.36, 0.33, 0.31]  ← Nearly uniform
```

```swift
// Apply temperature
func softmaxWithTemperature(_ logits: [Float], temperature: Float) -> [Float] {
    let scaled = logits.map { $0 / temperature }
    return stableSoftmax(scaled)
}
```

### Softmax Properties

1. **Monotonic:** If $z_i > z_j$, then $\text{softmax}(z)_i > \text{softmax}(z)_j$
2. **Translation invariant:** $\text{softmax}(z + c) = \text{softmax}(z)$
3. **Range:** Each output in $(0, 1)$
4. **Sum:** $\sum_i \text{softmax}(z)_i = 1$
5. **Limit:** $\lim_{z_i \to \infty} \text{softmax}(z)_i = 1$

---

## Top-K Predictions

### Why Top-K?

Instead of all 1000 classes, report the K most likely:

```swift
// Top-5 predictions for an image
1. Golden Retriever: 94.2%
2. Labrador Retriever: 3.1%
3. Cocker Spaniel: 1.2%
4. Irish Setter: 0.8%
5. Brittany: 0.4%
```

**Reasons to use Top-K:**
1. **Ambiguity:** Multiple classes may be reasonable
2. **Fine-grained:** Similar classes (dog breeds)
3. **Uncertainty:** Gap between top predictions indicates confidence
4. **Fallback:** If #1 is wrong, #2-5 might be right
5. **User experience:** Show alternatives

### Efficient Top-K Algorithms

**Naive approach (O(n log n)):**
```swift
// Sort all, take first K
let sorted = probabilities.enumerated().sorted { $0.1 > $1.1 }
let topK = sorted.prefix(5)
```

**Heap-based (O(n log k)):**
```swift
// Maintain min-heap of size K
var heap = MinHeap<(index: Int, prob: Float)>(capacity: k)
for (i, p) in probabilities.enumerated() {
    if heap.count < k || p > heap.min!.prob {
        heap.insertOrReplace((i, p))
    }
}
```

**Partial sort (O(n + k log k)):**
```swift
// Use Accelerate framework
import Accelerate
var indices = [vDSP_Length](repeating: 0, count: k)
var values = [Float](repeating: 0, count: k)
vDSP_vsorti(probabilities, &indices, &values, vDSP_Length(k), 1)
```

### Confidence Thresholds

Filter predictions below a threshold:

```swift
// Only show confident predictions
let confident = result.predictions.filter { $0.confidence > 0.05 }

// Interpretation:
// - High threshold (>0.5): Only very confident predictions
// - Medium threshold (0.1-0.3): Reasonably confident
// - Low threshold (<0.05): Include uncertain predictions
```

**Confidence gap analysis:**
```swift
let topTwo = result.predictions.prefix(2)
let gap = topTwo[0].confidence - topTwo[1].confidence

if gap > 0.5 {
    print("High confidence: definitely \(topTwo[0].label)")
} else if gap > 0.2 {
    print("Likely \(topTwo[0].label), possibly \(topTwo[1].label)")
} else {
    print("Uncertain between \(topTwo[0].label) and \(topTwo[1].label)")
}
```

---

## Common Architectures

### MobileNet Family

**Purpose:** Efficient inference on mobile/edge devices

**Key innovation:** Depthwise separable convolutions

```
Standard Convolution:
Input: H×W×C_in
Kernel: K×K×C_in×C_out
Params: K² × C_in × C_out

Depthwise Separable:
1. Depthwise: K×K×1 per channel = K² × C_in params
2. Pointwise: 1×1×C_in×C_out params
Total: K² × C_in + C_in × C_out

Reduction: ~8-9× fewer parameters for K=3
```

**MobileNetV1 (2017):**
- Introduced depthwise separable convolutions
- Width multiplier α for model scaling

**MobileNetV2 (2018):**
- Inverted residual blocks
- Linear bottlenecks
- Better accuracy/efficiency

**MobileNetV3 (2019):**
- Neural architecture search (NAS)
- Hard-swish activation
- Squeeze-and-excitation

| Model | Params | Top-1 Acc | Latency (iPhone) |
|-------|--------|-----------|------------------|
| MobileNetV2 1.0 | 3.4M | 72.0% | ~10ms |
| MobileNetV3-Small | 2.5M | 67.4% | ~6ms |
| MobileNetV3-Large | 5.4M | 75.2% | ~15ms |

**Best for:** Real-time mobile inference, resource-constrained devices

### EfficientNet Family

**Purpose:** Optimal accuracy/efficiency tradeoff

**Key innovation:** Compound scaling

```
Scaling dimensions:
- Width (w): More channels per layer
- Depth (d): More layers
- Resolution (r): Larger input images

EfficientNet compound scaling:
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

Where α × β² × γ² ≈ 2 (FLOPS roughly double)

EfficientNet-B0 baseline: φ = 0, α=β=γ=1
EfficientNet-B7: φ = 7
```

| Model | Params | Top-1 Acc | Input Size |
|-------|--------|-----------|------------|
| EfficientNet-B0 | 5.3M | 77.1% | 224 |
| EfficientNet-B1 | 7.8M | 79.1% | 240 |
| EfficientNet-B2 | 9.2M | 80.1% | 260 |
| EfficientNet-B3 | 12M | 81.6% | 300 |
| EfficientNet-B4 | 19M | 82.9% | 380 |
| EfficientNet-B5 | 30M | 83.6% | 456 |
| EfficientNet-B6 | 43M | 84.0% | 528 |
| EfficientNet-B7 | 66M | 84.3% | 600 |

**EfficientNetV2 (2021):**
- Progressive training (start small, increase resolution)
- Fused-MBConv blocks
- Faster training, better accuracy

**Best for:** When accuracy matters, scalable models

### ResNet Family

**Purpose:** Enable training of very deep networks

**Key innovation:** Residual connections (skip connections)

```
Standard block:      Residual block:
x → Conv → out      x → Conv → + → out
                         ↓       ↑
                         └───────┘
                       (skip connection)

Mathematically:
H(x) = F(x) + x

The network learns the residual F(x) = H(x) - x
which is easier than learning H(x) directly.
```

**Why it works:**
- Gradient can flow directly through skip connections
- Enables training of 100+ layer networks
- Easier to learn identity mapping if needed

| Model | Params | Top-1 Acc | Notes |
|-------|--------|-----------|-------|
| ResNet-18 | 11.7M | 69.8% | Good for fine-tuning |
| ResNet-34 | 21.8M | 73.3% | |
| ResNet-50 | 25.6M | 76.1% | Most popular |
| ResNet-101 | 44.5M | 77.4% | |
| ResNet-152 | 60.2M | 78.3% | |

**Best for:** Transfer learning base, well-understood architecture

### Vision Transformers (ViT)

**Purpose:** Apply transformer architecture to images

**Key innovation:** Treat image patches as tokens

```
Image (224×224)
       ↓
Split into 14×14 patches of 16×16
       ↓
196 patches
       ↓
Linear projection: 16×16×3 → 768
       ↓
Add position embeddings
       ↓
[CLS] token + 196 patch tokens
       ↓
┌─────────────────────────────────┐
│   Transformer Encoder ×L       │
│   (Self-attention + FFN)       │
└─────────────────────────────────┘
       ↓
[CLS] token output
       ↓
Classification head
```

| Model | Params | Top-1 Acc | Patch Size |
|-------|--------|-----------|------------|
| ViT-B/16 | 86M | 81.8% | 16×16 |
| ViT-L/16 | 307M | 85.2% | 16×16 |
| ViT-H/14 | 632M | 88.6% | 14×14 |

**Variants:**
- DeiT: Data-efficient training
- Swin: Hierarchical with shifted windows
- BEiT: BERT-style pre-training

**Best for:** Large-scale pre-training, when data is abundant

### ConvNeXt

**Purpose:** Modernize ConvNets to compete with ViT

**Key innovations:**
- Larger kernels (7×7)
- Fewer activations/normalizations
- GELU activation
- LayerNorm instead of BatchNorm

| Model | Params | Top-1 Acc |
|-------|--------|-----------|
| ConvNeXt-Tiny | 29M | 82.1% |
| ConvNeXt-Small | 50M | 83.1% |
| ConvNeXt-Base | 89M | 83.8% |
| ConvNeXt-Large | 198M | 84.3% |

**Best for:** When you want ConvNet simplicity with ViT-level accuracy

### Architecture Comparison

| Architecture | Params | Top-1 | Mobile | Server | Notes |
|--------------|--------|-------|--------|--------|-------|
| MobileNetV3-Large | 5.4M | 75.2% | ✓✓✓ | ✓ | Best for mobile |
| EfficientNet-B0 | 5.3M | 77.1% | ✓✓ | ✓ | Efficient & accurate |
| ResNet-50 | 25.6M | 76.1% | ✓ | ✓✓ | Transfer learning |
| EfficientNet-B4 | 19M | 82.9% | ✓ | ✓✓ | Balance |
| ViT-B/16 | 86M | 81.8% | - | ✓✓ | Large data |
| ConvNeXt-Base | 89M | 83.8% | - | ✓✓ | Modern ConvNet |

---

## Label Databases

### ImageNet-1K

**The most important classification benchmark.**

- **Classes:** 1,000 fine-grained categories
- **Training:** ~1.28 million images
- **Validation:** 50,000 images (50 per class)
- **Source:** ILSVRC (ImageNet Large Scale Visual Recognition Challenge)

**Category breakdown:**
| Category | Num Classes | Examples |
|----------|-------------|----------|
| Dogs | 120 | Golden Retriever, Poodle, Beagle |
| Birds | 59 | Robin, Flamingo, Owl |
| Insects | 27 | Butterfly, Bee, Ant |
| Cats | 8 | Persian, Siamese, Tabby |
| Fish | 13 | Goldfish, Shark |
| Vehicles | 40+ | Sports car, Bicycle, Airplane |
| Food | 20+ | Pizza, Ice cream |
| Household | 100+ | Laptop, Cup, Chair |

**Notable class indices:**
| Index | Label |
|-------|-------|
| 0 | tench (a fish) |
| 1 | goldfish |
| 207 | golden retriever |
| 281 | tabby cat |
| 285 | Egyptian cat |
| 386 | African elephant |
| 409 | analog clock |
| 508 | computer keyboard |
| 701 | parachute |
| 920 | traffic light |
| 999 | toilet tissue |

```swift
let labels = LabelDatabase.getAllLabels(for: .imagenet)
// labels[0] = "tench"
// labels[207] = "golden retriever"
```

### ImageNet-21K

- **Classes:** 21,841 categories
- **Images:** ~14 million
- **Used for:** Pre-training before fine-tuning

### CIFAR-10 and CIFAR-100

**CIFAR-10:**
```
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```
- 10 classes, 60,000 32×32 images
- 6,000 images per class

**CIFAR-100:**
- 100 fine-grained classes
- 20 superclasses
- 600 images per class

### Places365

**Scene recognition dataset:**
- 365 scene categories
- ~1.8 million images
- Categories: kitchen, bedroom, forest, office, beach, etc.

```swift
let labels = LabelDatabase.getAllLabels(for: .places365)
// ["abbey", "airport_terminal", "alley", ...]
```

### Custom Labels

```swift
// From array
let labels: [String] = ["cat", "dog", "bird", "other"]
let result = try ClassificationOutput.process(
    outputData: output,
    labels: .custom(labels)
)

// From file
let labelsText = try String(contentsOfFile: "labels.txt")
let labels = labelsText.components(separatedBy: .newlines)
    .filter { !$0.isEmpty }

// Validate count matches model
guard labels.count == numClasses else {
    throw Error.labelCountMismatch
}
```

---

## Transfer Learning

### Feature Extraction

Use pre-trained network as fixed feature extractor:

```
┌────────────────────────────────────────────┐
│ Pre-trained Backbone (frozen)              │
│ ImageNet features → Good general features  │
└────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────┐
│ New Classification Head (trained)          │
│ FC: features → your_num_classes            │
└────────────────────────────────────────────┘
```

**Advantages:**
- Fast (only train small head)
- Works with small datasets
- No risk of destroying pre-trained features

**Best for:** Small datasets, quick prototyping

### Fine-Tuning

Train the entire network (or later layers) on your data:

```
┌────────────────────────────────────────────┐
│ Early layers (frozen or low learning rate) │
│ Generic features: edges, textures          │
└────────────────────────────────────────────┘
                    │
┌────────────────────────────────────────────┐
│ Later layers (higher learning rate)        │
│ Task-specific features                     │
└────────────────────────────────────────────┘
                    │
┌────────────────────────────────────────────┐
│ Classification Head (highest learning rate)│
└────────────────────────────────────────────┘
```

**Advantages:**
- Better accuracy
- Adapts features to your domain

**Best for:** Larger datasets, domain-specific tasks

### When to Use What

| Your Dataset | Domain Similarity | Approach |
|--------------|-------------------|----------|
| Very small (<1000) | Similar to ImageNet | Feature extraction |
| Small (1k-10k) | Similar | Fine-tune last few layers |
| Small | Different | Feature extraction + augmentation |
| Medium (10k-100k) | Similar | Full fine-tuning |
| Medium | Different | Fine-tune from scratch or use different pre-training |
| Large (>100k) | Any | Train from scratch or fine-tune |

---

## Confidence Calibration

### The Overconfidence Problem

Modern neural networks are often **overconfident**:

```
Model says: "99% confident this is a dog"
Reality: Only 80% of 99%-confident predictions are correct

Expected Calibration:
If model says 90% confidence → should be right 90% of the time
```

**Reliability diagram:**
```
Accuracy ↑
   100% │                    ◆
        │               ◆ ◆ 
        │          ◆ ◆     
        │     ◆ ◆        Ideal: accuracy = confidence
        │◆ ◆              
        └────────────────→ Confidence
         0%            100%
         
Typical overconfident model:
Accuracy ↑
   100% │            ◆ ◆ ◆  ← High confidence, not 100% accurate
        │       ◆ ◆        
        │  ◆ ◆             
        │◆                  
        └────────────────→ Confidence
```

### Temperature Scaling for Calibration

Find optimal temperature T on validation set:

```python
# Pseudo-code for temperature calibration
def find_temperature(logits, labels):
    T = 1.0  # Start with T=1
    # Minimize negative log likelihood on validation set
    T = optimize(lambda T: nll_loss(softmax(logits/T), labels))
    return T  # Usually T > 1 (softer predictions)
```

**Typical calibration temperatures:** 1.5 - 3.0

```swift
// Apply calibrated temperature at inference
let calibratedProbs = softmaxWithTemperature(logits, temperature: 2.1)
```

### Expected Calibration Error

Quantify calibration quality:

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

Where $B_m$ are bins of predictions grouped by confidence.

**Interpretation:**
- ECE = 0: Perfect calibration
- ECE = 0.05: 5% average calibration error
- ECE > 0.1: Poorly calibrated

---

## SwiftPixelUtils Classification API

### Basic Usage

```swift
import SwiftPixelUtils

// Process classification output
let result = try ClassificationOutput.process(
    outputData: modelOutput,    // Data from model
    labels: .imagenet,          // Label set
    topK: 5                     // Return top 5 predictions
)

// Access results
print("Top prediction: \(result.topPrediction.label)")
print("Confidence: \(result.topPrediction.confidence * 100)%")

for prediction in result.predictions {
    print("\(prediction.label): \(prediction.confidence)")
}
```

### Output Formats

```swift
// Auto-detect (recommended)
ClassificationOutput.process(
    outputData: output,
    labels: .imagenet,
    outputFormat: .auto
)

// Explicit: probabilities (softmax already applied)
ClassificationOutput.process(
    floatOutput: probabilities,  // [Float]
    labels: .imagenet,
    outputFormat: .probabilities
)

// Explicit: logits (need softmax)
ClassificationOutput.process(
    floatOutput: logits,
    labels: .imagenet,
    outputFormat: .logits
)

// Explicit: log-probabilities
ClassificationOutput.process(
    floatOutput: logProbs,
    labels: .imagenet,
    outputFormat: .logProbabilities
)
```

### Advanced Options

```swift
let result = try ClassificationOutput.process(
    outputData: output,
    labels: .imagenet,
    topK: 10,
    confidenceThreshold: 0.01,      // Filter below 1%
    temperature: 1.0,               // Calibration temperature
    outputFormat: .auto
)
```

### ClassificationResult Properties

```swift
let result: ClassificationResult

// Top prediction
result.topPrediction.classIndex    // 207
result.topPrediction.label         // "golden retriever"
result.topPrediction.confidence    // 0.942

// All predictions (sorted by confidence)
result.predictions                 // [Prediction]
result.predictions[0].classIndex   // Same as topPrediction
result.predictions[0].label
result.predictions[0].confidence

// Metadata
result.numClasses                  // 1000
result.processingTimeMs            // 2.3

// Raw data (if needed)
result.allProbabilities            // [Float] - all 1000 probabilities
```

---

## Complete Implementation Examples

### Example 1: Basic Classification with TFLite

```swift
import SwiftPixelUtils
import TensorFlowLite

func classifyImage(_ image: UIImage) async throws -> ClassificationResult {
    // 1. Initialize interpreter
    let interpreter = try Interpreter(modelPath: "mobilenet_v2.tflite")
    try interpreter.allocateTensors()
    
    // 2. Get model input shape
    let inputTensor = try interpreter.input(at: 0)
    let inputShape = inputTensor.shape
    let height = inputShape.dimensions[1]
    let width = inputShape.dimensions[2]
    
    // 3. Preprocess image
    let input = try await PixelExtractor.getModelInput(
        source: .uiImage(image),
        framework: .tfliteFloat,
        width: width,
        height: height
    )
    
    // 4. Run inference
    try interpreter.copy(input.data, toInputAt: 0)
    try interpreter.invoke()
    
    // 5. Get output
    let outputTensor = try interpreter.output(at: 0)
    
    // 6. Process results
    return try ClassificationOutput.process(
        outputData: outputTensor.data,
        labels: .imagenet,
        topK: 5
    )
}

// Usage
Task {
    do {
        let result = try await classifyImage(myImage)
        print("Prediction: \(result.topPrediction.label)")
        print("Confidence: \(result.topPrediction.confidence * 100)%")
    } catch {
        print("Error: \(error)")
    }
}
```

### Example 2: Multi-Model Ensemble

```swift
func ensembleClassify(_ image: UIImage) async throws -> ClassificationResult {
    // Run multiple models
    async let result1 = classifyWithMobileNet(image)
    async let result2 = classifyWithEfficientNet(image)
    async let result3 = classifyWithResNet(image)
    
    let results = try await [result1, result2, result3]
    
    // Average probabilities
    var avgProbs = [Float](repeating: 0, count: 1000)
    for result in results {
        for (i, p) in result.allProbabilities.enumerated() {
            avgProbs[i] += p / Float(results.count)
        }
    }
    
    // Process averaged probabilities
    return try ClassificationOutput.process(
        floatOutput: avgProbs,
        labels: .imagenet,
        outputFormat: .probabilities,
        topK: 5
    )
}
```

### Example 3: Confidence-Based Decision Making

```swift
func classifyWithConfidenceHandling(_ image: UIImage) async throws -> String {
    let result = try await classifyImage(image)
    
    let top = result.topPrediction
    let second = result.predictions[1]
    let gap = top.confidence - second.confidence
    
    if top.confidence > 0.9 {
        return "Definitely \(top.label)"
    } else if top.confidence > 0.7 && gap > 0.3 {
        return "Probably \(top.label)"
    } else if gap < 0.1 {
        return "Could be \(top.label) or \(second.label)"
    } else {
        return "Uncertain, best guess: \(top.label)"
    }
}
```

---

## Performance Optimization

### Batch Processing

```swift
// Process multiple images efficiently
func classifyBatch(_ images: [UIImage]) async throws -> [ClassificationResult] {
    // Prepare all inputs
    let inputs = try await withThrowingTaskGroup(of: (Int, ModelInput).self) { group in
        for (i, image) in images.enumerated() {
            group.addTask {
                let input = try await PixelExtractor.getModelInput(
                    source: .uiImage(image),
                    framework: .tfliteFloat,
                    width: 224,
                    height: 224
                )
                return (i, input)
            }
        }
        
        var results = [(Int, ModelInput)]()
        for try await result in group {
            results.append(result)
        }
        return results.sorted { $0.0 < $1.0 }.map { $0.1 }
    }
    
    // Run inference (ideally batched if model supports)
    var results: [ClassificationResult] = []
    for input in inputs {
        // ... inference ...
        let result = try ClassificationOutput.process(...)
        results.append(result)
    }
    
    return results
}
```

### Model Caching

```swift
class ClassificationService {
    static let shared = ClassificationService()
    
    private var interpreter: Interpreter?
    private let lock = NSLock()
    
    func getInterpreter() throws -> Interpreter {
        lock.lock()
        defer { lock.unlock() }
        
        if let interpreter = interpreter {
            return interpreter
        }
        
        let newInterpreter = try Interpreter(modelPath: modelPath)
        try newInterpreter.allocateTensors()
        interpreter = newInterpreter
        return newInterpreter
    }
}
```

---

## Best Practices

### 1. Match Preprocessing to Model Training

```swift
// Know your model's expected preprocessing
// PyTorch ImageNet: ImageNet normalization, RGB
// TFLite: Usually [0,1], RGB
// Check model documentation!
```

### 2. Validate Output Shape

```swift
let outputTensor = try interpreter.output(at: 0)
let expectedShape = [1, 1000]

guard outputTensor.shape.dimensions == expectedShape else {
    throw ClassificationError.unexpectedShape(
        expected: expectedShape,
        got: outputTensor.shape.dimensions
    )
}
```

### 3. Handle Edge Cases

```swift
// Empty or corrupt images
guard let cgImage = image.cgImage else {
    throw ClassificationError.invalidImage
}

// Very small images
let minSize = 32
guard cgImage.width >= minSize && cgImage.height >= minSize else {
    throw ClassificationError.imageTooSmall
}
```

### 4. Use Appropriate Confidence Thresholds

```swift
// Task-specific thresholds
let threshold: Float = switch task {
case .productCatalog: 0.7    // High confidence for automation
case .photoSuggestion: 0.3   // Lower for suggestions
case .medicalScreening: 0.1  // Catch everything, review later
}
```

---

## Troubleshooting

### Problem: All Predictions Have Similar Confidence

**Possible causes:**
- Wrong normalization (model expects different range)
- Model not loading correctly
- Input image is unusual for the model

**Debug:**
```swift
// Check logit range
print("Logits range: \(logits.min()!) to \(logits.max()!)")
// Should have some variation, not all ~0
```

### Problem: Predictions Are Wrong

**Checklist:**
1. ✓ Color format (RGB vs BGR)?
2. ✓ Normalization ([0,1] vs ImageNet)?
3. ✓ Data type (Float32 vs UInt8)?
4. ✓ Layout (HWC vs CHW)?
5. ✓ Labels match model?

### Problem: Confidence Is Always Very High

**Cause:** Model is overconfident

**Solution:** Apply temperature scaling
```swift
let calibrated = softmaxWithTemperature(logits, temperature: 2.0)
```

---

## Mathematical Foundations

### Cross-Entropy Loss

Training loss for classification:

$$L = -\sum_{i=1}^{K} y_i \log(p_i) = -\log(p_c)$$

Where $y$ is one-hot label and $c$ is correct class.

### Information Theory

**Entropy** of prediction:
$$H(p) = -\sum_i p_i \log(p_i)$$

- Low entropy: Confident (one class dominates)
- High entropy: Uncertain (uniform distribution)

**KL Divergence** between distributions:
$$D_{KL}(p || q) = \sum_i p_i \log\frac{p_i}{q_i}$$

### Evaluation Metrics

**Top-1 Accuracy:**
$$\text{Top-1} = \frac{\text{correct predictions}}{\text{total predictions}}$$

**Top-5 Accuracy:**
$$\text{Top-5} = \frac{\text{correct class in top 5}}{\text{total predictions}}$$

---

## Next Steps

- [Object Detection](04-detection-output.md) - YOLO and bounding boxes
- [Semantic Segmentation](05-segmentation-output.md) - Pixel-wise classification
- [Quantization](02-quantization-guide.md) - Optimize for mobile
