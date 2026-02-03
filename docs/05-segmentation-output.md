# Semantic Segmentation: Theory and Implementation

A comprehensive reference for semantic segmentation concepts, DeepLabV3 architecture, output processing, and implementation patterns using SwiftPixelUtils.

## Table of Contents

- [Introduction](#introduction)
- [What is Semantic Segmentation?](#what-is-semantic-segmentation)
  - [Task Definition](#task-definition)
  - [Segmentation vs Detection vs Classification](#segmentation-vs-detection-vs-classification)
  - [Types of Segmentation](#types-of-segmentation)
  - [Applications](#applications)
- [How Segmentation Networks Work](#how-segmentation-networks-work)
  - [Encoder-Decoder Architecture](#encoder-decoder-architecture)
  - [The Resolution Challenge](#the-resolution-challenge)
  - [Skip Connections](#skip-connections)
  - [Feature Aggregation](#feature-aggregation)
- [DeepLab Architecture Family](#deeplab-architecture-family)
  - [DeepLabV1: Dilated Convolutions](#deeplabv1-dilated-convolutions)
  - [DeepLabV2: ASPP](#deeplabv2-aspp)
  - [DeepLabV3: Improved ASPP](#deeplabv3-improved-aspp)
  - [DeepLabV3+: Decoder](#deeplabv3-decoder)
- [Key Techniques](#key-techniques)
  - [Dilated/Atrous Convolutions](#dilatedatrous-convolutions)
  - [ASPP: Atrous Spatial Pyramid Pooling](#aspp-atrous-spatial-pyramid-pooling)
  - [Depthwise Separable Convolutions](#depthwise-separable-convolutions)
  - [Output Stride](#output-stride)
- [Understanding Segmentation Output](#understanding-segmentation-output)
  - [Raw Output Format](#raw-output-format)
  - [Logits vs Probabilities](#logits-vs-probabilities)
  - [Per-Pixel Predictions](#per-pixel-predictions)
  - [Output Resolution](#output-resolution)
- [Post-Processing Techniques](#post-processing-techniques)
  - [Argmax for Class Assignment](#argmax-for-class-assignment)
  - [Upsampling Methods](#upsampling-methods)
  - [CRF Refinement](#crf-refinement)
  - [Boundary Refinement](#boundary-refinement)
- [Popular Segmentation Models](#popular-segmentation-models)
  - [FCN (Fully Convolutional Networks)](#fcn-fully-convolutional-networks)
  - [U-Net](#u-net)
  - [DeepLabV3+](#deeplabv3-1)
  - [PSPNet](#pspnet)
  - [SegFormer (Transformer-based)](#segformer-transformer-based)
  - [Model Comparison](#model-comparison)
- [Label Databases for Segmentation](#label-databases-for-segmentation)
  - [Pascal VOC (21 Classes)](#pascal-voc-21-classes)
  - [ADE20K (150 Classes)](#ade20k-150-classes)
  - [Cityscapes (19 Classes)](#cityscapes-19-classes)
  - [COCO-Stuff](#coco-stuff)
  - [Custom Datasets](#custom-datasets)
- [Color Maps and Visualization](#color-maps-and-visualization)
  - [Standard Color Palettes](#standard-color-palettes)
  - [Creating Color Maps](#creating-color-maps)
  - [Overlay Techniques](#overlay-techniques)
- [SwiftPixelUtils Segmentation API](#swiftpixelutils-segmentation-api)
  - [Basic Usage](#basic-usage)
  - [Configuration Options](#configuration-options)
  - [Segmentation Result Structure](#segmentation-result-structure)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Advanced Topics](#advanced-topics)
  - [Multi-Scale Inference](#multi-scale-inference)
  - [Sliding Window for Large Images](#sliding-window-for-large-images)
  - [Instance Segmentation](#instance-segmentation)
  - [Panoptic Segmentation](#panoptic-segmentation)
- [Performance Optimization](#performance-optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [Mathematical Foundations](#mathematical-foundations)

---

## Introduction

Semantic segmentation assigns a class label to every pixel in an image, enabling detailed scene understanding at the finest granularity. This guide covers segmentation theory, DeepLabV3 architecture details, and practical implementation with SwiftPixelUtils.

---

## What is Semantic Segmentation?

### Task Definition

Classify each pixel in an image into one of K predefined categories:

```
Input Image                    Segmentation Output
┌─────────────────────┐        ┌─────────────────────┐
│     Sky             │        │  ████████████████   │  Sky (blue)
│  ════════════════   │        │  ════════════════   │
│    🌳🏠🌳          │   →    │  🟢🟫🟢            │  Tree/Building
│ ═══════════════════ │        │  ░░░░░░░░░░░░░░░░░  │  Road (gray)
│   🚗    🧑         │        │  🔴    🟡           │  Car/Person
│░░░░░░░░░░░░░░░░░░░░░│        │░░░░░░░░░░░░░░░░░░░░░│
└─────────────────────┘        └─────────────────────┘

Every pixel labeled: sky, tree, building, road, car, person, etc.
```

**Formal Definition:**
- Input: Image $x \in \mathbb{R}^{H \times W \times 3}$
- Output: Label map $y \in \{0, 1, ..., K-1\}^{H \times W}$
- Model: $f_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{H \times W \times K}$

Each pixel location $(i, j)$ gets a probability distribution over $K$ classes.

### Segmentation vs Detection vs Classification

```
Classification:     Detection:           Segmentation:
┌────────────┐      ┌────────────┐       ┌────────────┐
│            │      │ ┌──────┐   │       │████████████│
│   🐕       │      │ │  🐕   │       │██🐕██████  │
│            │      │ └──────┘   │       │████████████│
│            │      │            │       │░░░░░░░░░░░░│
└────────────┘      └────────────┘       └────────────┘
Output: "dog"       Output: box +        Output: pixel-wise
                    "dog" label          class map

Granularity: Image → Object → Pixel
```

| Task | Output | Localization | Level |
|------|--------|--------------|-------|
| Classification | Single label | None | Image |
| Detection | Bounding boxes | Box-level | Object |
| Semantic Segmentation | Pixel labels | Pixel-level | Pixel |
| Instance Segmentation | Pixel labels + instances | Pixel + instance | Object |
| Panoptic Segmentation | All above combined | Full scene | Scene |

### Types of Segmentation

**Semantic Segmentation:**
- All pixels of same class labeled identically
- Cannot distinguish between instances
- "All cars are red, all people are blue"

```
Two cars:
┌────────────────────┐
│   🔴🔴🔴  🔴🔴🔴   │  Both cars = same color
│   car     car      │  (same class)
└────────────────────┘
```

**Instance Segmentation:**
- Distinguishes individual object instances
- "Car #1 is red, Car #2 is green"

```
Two cars:
┌────────────────────┐
│   🔴🔴🔴  🟢🟢🟢   │  Different colors
│   car#1   car#2    │  (different instances)
└────────────────────┘
```

**Panoptic Segmentation:**
- Combines semantic + instance
- "Things" (countable) get instance IDs
- "Stuff" (uncountable) only semantic labels

### Applications

| Domain | Classes | Use Case |
|--------|---------|----------|
| **Autonomous Driving** | road, sidewalk, car, pedestrian, sign, sky | Scene understanding, navigation |
| **Medical Imaging** | organ, tumor, tissue types | Diagnosis, surgical planning |
| **Satellite Imagery** | building, road, vegetation, water | Urban planning, environmental monitoring |
| **Photo Editing** | person, background, hair, skin | Background removal, portrait mode |
| **Augmented Reality** | floor, wall, furniture, person | Scene understanding, object placement |
| **Robotics** | obstacles, free space, objects | Navigation, manipulation |
| **Agriculture** | crop, weed, soil, disease | Precision farming |
| **Fashion** | clothing items, body parts | Virtual try-on |

---

## How Segmentation Networks Work

### Encoder-Decoder Architecture

The dominant paradigm for segmentation:

```
          ENCODER                              DECODER
    (Downsample/Extract)                    (Upsample/Reconstruct)
    
Input: H×W×3
    │
    ▼
┌──────────┐
│  Conv    │ → H×W×64
│  Pool    │ → H/2×W/2×64
└──────────┘
    │
    ▼
┌──────────┐
│  Conv    │ → H/2×W/2×128
│  Pool    │ → H/4×W/4×128
└──────────┘
    │
    ▼
┌──────────┐
│  Conv    │ → H/4×W/4×256
│  Pool    │ → H/8×W/8×256
└──────────┘
    │
    ▼                               
┌──────────┐                    ┌──────────┐
│ Bottleneck│ → H/16×W/16×512   │ Upsample │ → H/8×W/8×256
└──────────┘                    └──────────┘
                                    │
                                    ▼
                                ┌──────────┐
                                │ Upsample │ → H/4×W/4×128
                                └──────────┘
                                    │
                                    ▼
                                ┌──────────┐
                                │ Upsample │ → H/2×W/2×64
                                └──────────┘
                                    │
                                    ▼
                                ┌──────────┐
                                │ Upsample │ → H×W×K (classes)
                                └──────────┘
```

### The Resolution Challenge

**Problem:** Classification networks progressively downsample, losing spatial detail.

```
Input: 512×512
After pooling layers:
  → 256×256 → 128×128 → 64×64 → 32×32 → 16×16

Feature map at 16×16 has great semantic info
but lost 32× spatial resolution!

For segmentation, need full resolution output.
```

**Solutions:**
1. **Upsampling/Deconvolution:** Learn to upsample
2. **Skip connections:** Bring back early features
3. **Dilated convolutions:** Avoid downsampling
4. **Feature pyramid:** Multi-scale fusion

### Skip Connections

Bring back high-resolution features from encoder:

```
Encoder          Skip          Decoder
                Connection
┌──────┐                      ┌──────┐
│H×W×64│─────────────────────→│Concat│→ H×W×(64+64)
└──────┘                      └──────┘
   │                              ↑
   ▼                              │
┌──────┐                      ┌──────┐
│H/2×W/2│────────────────────→│Concat│
└──────┘                      └──────┘
   │                              ↑
   ▼                              │
   ...    (bottleneck)    ...→ Upsample
```

**Why skip connections help:**
- Early layers have fine spatial detail
- Deep layers have rich semantic info
- Combining both → accurate boundaries + correct classes

### Feature Aggregation

Multiple ways to combine multi-scale features:

```
Addition:      Concatenation:     Attention:
F_low          [F_low, F_high]    Attention(F_low, F_high)
  +                  │                    │
F_high              Conv                 Weighted
  │                  │                   Fusion
  ▼                  ▼                    ▼
Combined         Combined             Combined
```

---

## DeepLab Architecture Family

### DeepLabV1: Dilated Convolutions

**Key insight:** Use dilated (atrous) convolutions to maintain spatial resolution.

```
Standard Conv (3×3):        Dilated Conv (3×3, rate=2):
                            
■ ■ ■                       ■ □ ■ □ ■
■ ■ ■    receptive         □ □ □ □ □
■ ■ ■    field = 3×3       ■ □ ■ □ ■    receptive
                            □ □ □ □ □    field = 5×5
                            ■ □ ■ □ ■
                            
Same params, larger receptive field!
```

### DeepLabV2: ASPP

**Atrous Spatial Pyramid Pooling:** Capture multi-scale context.

```
                     Input Features
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    │          │           │           │          │
    ▼          ▼           ▼           ▼          ▼
 Conv 1×1   Dilated     Dilated     Dilated    GAP
            rate=6      rate=12     rate=18
    │          │           │           │          │
    └──────────┴───────────┴───────────┴──────────┘
                           │
                       Concatenate
                           │
                       Conv 1×1
                           │
                       Output
```

### DeepLabV3: Improved ASPP

Improvements over V2:
- Batch normalization after each conv
- Image-level features (global average pooling branch)
- Wider range of dilation rates

```
DeepLabV3 ASPP:
┌─────────────────────────────────────────────────┐
│                Input Features                   │
│                     │                           │
│  ┌────┬────┬────┬────┬────┐                     │
│  │1×1 │r=6 │r=12│r=18│GAP │                     │
│  │conv│conv│conv│conv│→1×1│                     │
│  │+BN │+BN │+BN │+BN │+BN │                     │
│  └────┴────┴────┴────┴────┘                     │
│       │    │    │    │    │                     │
│       └────┴────┴────┴────┘                     │
│                │                                │
│           Concatenate (256×5 = 1280 channels)   │
│                │                                │
│            Conv 1×1, BN, ReLU                   │
│                │                                │
│           Output (256 channels)                 │
└─────────────────────────────────────────────────┘
```

### DeepLabV3+: Decoder

Added a simple but effective decoder:

```
DeepLabV3+ Architecture:

                  ENCODER                           DECODER
                     
Input ──→ Backbone (ResNet/Xception)
              │
              ├──→ Low-level features ────────────┐
              │    (early in backbone)            │
              │                                   │
              └──→ ASPP ──→ 1×1 Conv ──→ Upsample 4× ──→ Concat
                                                          │
                                                    Conv 3×3
                                                          │
                                                    Upsample 4×
                                                          │
                                                    Output (H×W×K)
```

**Decoder benefits:**
- Sharper boundaries
- Better small object segmentation
- Simple and efficient

---

## Key Techniques

### Dilated/Atrous Convolutions

Increase receptive field without reducing resolution:

```
Dilation rate = 1 (standard):    Dilation rate = 2:
                                 
┌─┬─┬─┐                          ┌─┬─┬─┬─┬─┐
│■│■│■│  3×3 kernel              │■│ │■│ │■│  3×3 kernel
├─┼─┼─┤  3×3 receptive           ├─┼─┼─┼─┼─┤  5×5 receptive
│■│■│■│  field                   │ │ │ │ │ │  field
├─┼─┼─┤                          ├─┼─┼─┼─┼─┤
│■│■│■│                          │■│ │■│ │■│
└─┴─┴─┘                          ├─┼─┼─┼─┼─┤
                                 │ │ │ │ │ │
                                 ├─┼─┼─┼─┼─┤
                                 │■│ │■│ │■│
                                 └─┴─┴─┴─┴─┘
```

**Mathematical definition:**
$$y[i] = \sum_k x[i + r \cdot k] \cdot w[k]$$

Where $r$ is the dilation rate.

**Effective receptive field:**
$$RF_{effective} = (k - 1) \times r + 1$$

For 3×3 kernel: rate=1 → 3×3, rate=2 → 5×5, rate=3 → 7×7

### ASPP: Atrous Spatial Pyramid Pooling

Capture objects at multiple scales:

```
Why multiple rates?

Small dilation (r=6):   Medium (r=12):    Large (r=18):
Captures fine detail    Medium context    Large context

    🔍                     🔍                 🔍
  Small objects        Medium objects     Large objects
  Local patterns       Regional info      Global context
```

**Rates for different output strides:**
| Output Stride | Dilation Rates | Notes |
|---------------|----------------|-------|
| 8 | 12, 24, 36 | More context |
| 16 | 6, 12, 18 | Standard |
| 32 | 3, 6, 9 | Less context |

### Depthwise Separable Convolutions

Used in efficient variants (MobileNet backbone):

```
Standard Conv:                  Depthwise Separable:
                                
Input C_in → Output C_out       Input C_in
                                    │
K×K×C_in×C_out params               ▼
                                Depthwise: K×K×1×C_in
                                (spatial filtering per channel)
                                    │
                                    ▼
                                Pointwise: 1×1×C_in×C_out
                                (channel mixing)
                                
                                Total: K²×C_in + C_in×C_out
                                Savings: ~8-9× for K=3
```

### Output Stride

Ratio of input to output spatial resolution:

```
Output Stride = 16:
Input: 512×512 → Output: 32×32

Output Stride = 8:
Input: 512×512 → Output: 64×64 (better detail, more compute)

Lower output stride = higher resolution = better boundaries
                    = more computation = more memory
```

**Trade-off:**
| Output Stride | Resolution | Speed | Accuracy |
|---------------|------------|-------|----------|
| 32 | Low | Fast | Lower |
| 16 | Medium | Medium | Good |
| 8 | High | Slow | Best |

---

## Understanding Segmentation Output

### Raw Output Format

Segmentation models output a tensor with per-pixel class logits:

```
Output shape: [batch, height, width, num_classes]
Example: [1, 513, 513, 21] for Pascal VOC

At each pixel (i, j):
output[0, i, j, :] = [logit_0, logit_1, ..., logit_20]
                      └─── 21 class scores ───────┘

The class with highest logit wins:
predicted_class[i, j] = argmax(output[0, i, j, :])
```

### Logits vs Probabilities

**Raw logits:**
- Unbounded real numbers
- Need softmax for probabilities
- More numerically stable for loss computation

**Probabilities (after softmax):**
- Values in [0, 1]
- Sum to 1 at each pixel
- Interpretable as confidence

```swift
// Convert logits to probabilities
func pixelSoftmax(_ logits: [Float]) -> [Float] {
    let maxLogit = logits.max()!
    let exps = logits.map { exp($0 - maxLogit) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}
```

### Per-Pixel Predictions

Each pixel is an independent classification:

```
Pixel at (100, 200):
Logits: [2.1, -0.5, 8.3, -1.2, ..., 0.4]  (21 values)
                    ↑
              Highest = class 2 (probably "person")
              
Softmax: [0.01, 0.01, 0.95, 0.01, ..., 0.01]
                       ↑
              95% confident it's "person"
```

### Output Resolution

Many models output at lower resolution:

```
Input: 512×512
Model output: 32×32 (output stride 16)
              or 64×64 (output stride 8)

Need to upsample to original resolution for final mask.
```

---

## Post-Processing Techniques

### Argmax for Class Assignment

Convert logits/probabilities to class labels:

```swift
func createMask(from output: [Float], 
                height: Int, width: Int, 
                numClasses: Int) -> [UInt8] {
    var mask = [UInt8](repeating: 0, count: height * width)
    
    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * numClasses
            
            // Find class with highest score
            var maxClass = 0
            var maxScore = output[offset]
            
            for c in 1..<numClasses {
                if output[offset + c] > maxScore {
                    maxScore = output[offset + c]
                    maxClass = c
                }
            }
            
            mask[y * width + x] = UInt8(maxClass)
        }
    }
    
    return mask
}
```

### Upsampling Methods

**Bilinear interpolation:**
```
┌───┬───┐          ┌─┬─┬─┬─┐
│ A │ B │   2×     │A│•│•│B│   • = interpolated
├───┼───┤  ────→   ├─┼─┼─┼─┤
│ C │ D │          │•│•│•│•│
└───┴───┘          ├─┼─┼─┼─┤
                   │C│•│•│D│
                   └─┴─┴─┴─┘
```

**Nearest neighbor:**
```
┌───┬───┐          ┌─┬─┬─┬─┐
│ A │ B │   2×     │A│A│B│B│   Just replicate
├───┼───┤  ────→   ├─┼─┼─┼─┤
│ C │ D │          │A│A│B│B│
└───┴───┘          ├─┼─┼─┼─┤
                   │C│C│D│D│
                   ├─┼─┼─┼─┤
                   │C│C│D│D│
                   └─┴─┴─┴─┘
```

**When to use which:**
| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| Nearest | Fastest | Blocky edges | Mask indices |
| Bilinear | Fast | Smooth | Probabilities |
| Learned (transposed conv) | Slow | Best | Training |

### CRF Refinement

**Conditional Random Field** post-processing:

```
Before CRF:              After CRF:
┌────────────────┐       ┌────────────────┐
│ ▓▓▓░░░░░░░░░░░ │       │ ████░░░░░░░░░░ │
│ ▓▓▓▓░░░░░░░░░░ │       │ ████░░░░░░░░░░ │
│ ░▓▓▓▓░░░░░░░░░ │  →    │ █████░░░░░░░░░ │
│ ░░▓▓▓▓▓░░░░░░░ │       │ ░█████░░░░░░░░ │
│ ░░░░▓▓░░░░░░░░ │       │ ░░█████░░░░░░░ │
└────────────────┘       └────────────────┘
Noisy boundaries         Clean, aligned to edges
```

**CRF encourages:**
- Nearby pixels with similar color → same class
- Respecting image edges
- Smooth regions

### Boundary Refinement

Techniques for sharper boundaries:

```swift
// Morphological operations
func refineMask(_ mask: [UInt8], width: Int, height: Int) -> [UInt8] {
    // 1. Remove small holes (closing)
    let closed = morphologicalClose(mask, kernelSize: 3)
    
    // 2. Remove small islands (opening)
    let opened = morphologicalOpen(closed, kernelSize: 3)
    
    // 3. Smooth boundaries with median filter
    let smoothed = medianFilter(opened, kernelSize: 3)
    
    return smoothed
}
```

---

## Popular Segmentation Models

### FCN (Fully Convolutional Networks)

**Pioneer of deep learning segmentation (2015):**

```
┌─────────────────────────────────────────┐
│ VGG-style backbone (all FC → Conv)      │
│                                         │
│ Input: H×W                              │
│   ↓                                     │
│ Pool1 → Pool2 → Pool3 → Pool4 → Pool5   │
│ H/2     H/4     H/8     H/16    H/32    │
│                                         │
│ Upsample (deconv/bilinear) → H×W        │
│                                         │
│ Skip connections from pool3, pool4      │
└─────────────────────────────────────────┘

Variants:
FCN-32s: Only from pool5 (coarse)
FCN-16s: pool5 + pool4 (better)
FCN-8s:  pool5 + pool4 + pool3 (best)
```

SwiftPixelUtils preset:
```swift
let result = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.fcn
)
// 512×512, contain, ImageNet normalization, NCHW
```

### U-Net

**Symmetric encoder-decoder with dense skip connections:**

```
Encoder           Decoder
    │               │
┌───┴───┐       ┌───┴───┐
│ 64×64 │←──────│ 64×64 │   Skip connections
│       │  copy │       │   at every level
└───┬───┘       └───┬───┘
    │               │
┌───┴───┐       ┌───┴───┐
│ 128   │←──────│ 128   │
└───┬───┘       └───┬───┘
    │               │
┌───┴───┐       ┌───┴───┐
│ 256   │←──────│ 256   │
└───┬───┘       └───┬───┘
    │               │
    └──→ Bottleneck ←──┘
         512 channels
```

**Strengths:**
- Excellent for biomedical images
- Works with small datasets
- Very good boundary details

SwiftPixelUtils presets:
```swift
// Standard 512×512 UNet
let result = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.unet
)

// Fast 256×256 variant
let fast = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.unet_256
)

// High-res 1024×1024 variant
let highRes = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.unet_1024
)
// All use contain resize, scale normalization, NCHW
```

### DeepLabV3+

See [DeepLab Architecture Family](#deeplab-architecture-family) above.

**Summary:**
- ASPP for multi-scale context
- Simple effective decoder
- State-of-art on multiple benchmarks

SwiftPixelUtils presets:
```swift
// Standard 513×513 DeepLab
let result = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.deeplab  // or deeplabv3, deeplabv3_plus
)

// Higher resolution variants
let highRes = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.deeplab_769  // or deeplab_1025
)
// All use contain resize, ImageNet normalization, NCHW
```

### PSPNet

**Pyramid Scene Parsing Network:**

```
┌─────────────────────────────────────────────────┐
│              Pyramid Pooling Module             │
│                                                 │
│    Input Features                               │
│         │                                       │
│  ┌──────┼──────┬──────┬──────┐                  │
│  │      │      │      │      │                  │
│  ▼      ▼      ▼      ▼      ▼                  │
│ 1×1   2×2    3×3    6×6    Original             │
│ pool  pool   pool   pool   features             │
│  │      │      │      │      │                  │
│  ▼      ▼      ▼      ▼      │                  │
│ Conv   Conv   Conv   Conv    │                  │
│  │      │      │      │      │                  │
│  └──────┴──────┴──────┴──────┘                  │
│                │                                │
│          Upsample & Concat                      │
│                │                                │
│            Conv → Output                        │
└─────────────────────────────────────────────────┘
```

**Key idea:** Global context at multiple scales through pooling pyramid.

SwiftPixelUtils preset:
```swift
let result = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.pspnet
)
// 473×473, contain, ImageNet normalization, NCHW
```

### SegFormer (Transformer-based)

**Modern transformer approach:**

```
┌─────────────────────────────────────────────────┐
│              Hierarchical Transformer           │
│                                                 │
│ Image → Patch Embed → Transformer Block ×N      │
│                            │                    │
│                     Multi-scale features        │
│                      ↙    ↓    ↘                │
│                   1/4   1/8   1/16  1/32        │
│                      ↘    ↓    ↙                │
│                    MLP Decoder                  │
│                         │                       │
│                    Upsample                     │
│                         │                       │
│                     Output                      │
└─────────────────────────────────────────────────┘
```

**Advantages:**
- No positional encoding needed
- Efficient MLP decoder
- Strong performance

| Variant | mIoU (ADE20K) | Parameters |
|---------|---------------|------------|
| SegFormer-B0 | 37.4 | 3.8M |
| SegFormer-B1 | 42.2 | 13.7M |
| SegFormer-B2 | 46.5 | 27.4M |
| SegFormer-B3 | 49.4 | 47.3M |
| SegFormer-B4 | 50.3 | 64.1M |
| SegFormer-B5 | 51.0 | 84.7M |

SwiftPixelUtils presets:
```swift
let result = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.segformer  // or segformer_b0, segformer_b5
)
// 512×512, contain, ImageNet normalization, NCHW
```

### Mask2Former

**Universal segmentation with masked attention:**

Mask2Former unifies semantic, instance, and panoptic segmentation with a single architecture:
- Masked attention for efficient training
- Multi-scale deformable attention
- Query-based mask prediction

| Backbone | mIoU (ADE20K) | PQ (COCO) | Parameters |
|----------|---------------|-----------|------------|
| ResNet-50 | 47.2 | 51.9 | 44M |
| Swin-T | 47.7 | 52.1 | 47M |
| Swin-L | 56.1 | 57.8 | 216M |

SwiftPixelUtils presets:
```swift
let result = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.mask2former  // or mask2former_swin_t, mask2former_swin_l
)
// 512×512, contain, ImageNet normalization, NCHW
```

### SAM / SAM2 (Segment Anything Model)

**Promptable segmentation with zero-shot generalization:**

SAM can segment any object given a point, box, or text prompt. SAM2 extends this to video with streaming memory.

| Model | Description | Parameters |
|-------|-------------|------------|
| SAM ViT-H | Original SAM, high quality | 636M |
| SAM2-T | Tiny, 6× faster than SAM | 38.9M |
| SAM2-S | Small | 46M |
| SAM2-B+ | Base Plus | 80.8M |
| SAM2-L | Large, best quality | 224.4M |

SAM2 improvements over SAM:
- 6× faster image segmentation
- Video object tracking with memory
- Occlusion handling

SwiftPixelUtils presets:
```swift
// SAM (original)
let sam = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.sam
)

// SAM2 variants
let sam2 = try PixelExtractor.getPixelData(
    source: .cgImage(image),
    options: ModelPresets.sam2  // or sam2_t, sam2_s, sam2_b_plus, sam2_l
)
// All use 1024×1024, contain, ImageNet normalization, NCHW
```

### Model Comparison

| Model | mIoU (VOC) | mIoU (ADE20K) | Params | Best For |
|-------|------------|---------------|--------|----------|
| FCN-8s | 62.2 | 29.4 | 134M | Baseline |
| U-Net | 66.0 | - | 31M | Medical, small data |
| PSPNet | 82.6 | 43.3 | 65M | Scene parsing |
| DeepLabV3+ | 87.8 | 45.7 | 54M | General purpose |
| SegFormer-B4 | 84.0 | 50.3 | 64M | Modern, efficient |
| Mask2Former-Swin-L | - | 56.1 | 216M | Universal segmentation |
| SAM2-L | - | - | 224M | Zero-shot, promptable |

---

## Label Databases for Segmentation

### Pascal VOC (21 Classes)

Standard benchmark with 21 classes (including background):

```swift
let vocLabels = [
    "background",    // 0 - black
    "aeroplane",     // 1
    "bicycle",       // 2
    "bird",          // 3
    "boat",          // 4
    "bottle",        // 5
    "bus",           // 6
    "car",           // 7
    "cat",           // 8
    "chair",         // 9
    "cow",           // 10
    "diningtable",   // 11
    "dog",           // 12
    "horse",         // 13
    "motorbike",     // 14
    "person",        // 15
    "pottedplant",   // 16
    "sheep",         // 17
    "sofa",          // 18
    "train",         // 19
    "tvmonitor"      // 20
]
```

### ADE20K (150 Classes)

Large-scale scene parsing:

```swift
// Selected categories from ADE20K
let ade20kSample = [
    "wall", "building", "sky", "floor", "tree",
    "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair",
    // ... 130 more classes
]
```

### Cityscapes (19 Classes)

Autonomous driving focus:

```swift
let cityscapesLabels = [
    // Flat
    "road", "sidewalk",
    // Human
    "person", "rider",
    // Vehicle
    "car", "truck", "bus", "train", "motorcycle", "bicycle",
    // Construction
    "building", "wall", "fence",
    // Object
    "pole", "traffic light", "traffic sign",
    // Nature
    "vegetation", "terrain",
    // Sky
    "sky"
]
```

### COCO-Stuff

COCO with stuff classes (171 total):
- 80 thing classes (from COCO detection)
- 91 stuff classes (sky, grass, road, etc.)

### Custom Datasets

```swift
// Define custom segmentation labels
let customLabels = ["background", "defect_type_a", "defect_type_b", "good"]

let result = try SegmentationOutput.process(
    output: modelOutput,
    labels: .custom(customLabels),
    colorMap: myColorMap
)
```

---

## Color Maps and Visualization

### Standard Color Palettes

**Pascal VOC color map:**

```swift
let vocColorMap: [(UInt8, UInt8, UInt8)] = [
    (0, 0, 0),       // 0: background - black
    (128, 0, 0),     // 1: aeroplane - maroon
    (0, 128, 0),     // 2: bicycle - green
    (128, 128, 0),   // 3: bird - olive
    (0, 0, 128),     // 4: boat - navy
    (128, 0, 128),   // 5: bottle - purple
    (0, 128, 128),   // 6: bus - teal
    (128, 128, 128), // 7: car - gray
    (64, 0, 0),      // 8: cat - dark red
    (192, 0, 0),     // 9: chair - red
    (64, 128, 0),    // 10: cow - olive green
    (192, 128, 0),   // 11: diningtable - orange
    (64, 0, 128),    // 12: dog - purple
    (192, 0, 128),   // 13: horse - pink
    (64, 128, 128),  // 14: motorbike - teal
    (192, 128, 128), // 15: person - light pink
    (0, 64, 0),      // 16: pottedplant - dark green
    (128, 64, 0),    // 17: sheep - brown
    (0, 192, 0),     // 18: sofa - lime
    (128, 192, 0),   // 19: train - yellow-green
    (0, 64, 128)     // 20: tvmonitor - blue
]
```

### Creating Color Maps

```swift
// Generate distinct colors programmatically
func generateColorMap(numClasses: Int) -> [(UInt8, UInt8, UInt8)] {
    var colors: [(UInt8, UInt8, UInt8)] = []
    
    for i in 0..<numClasses {
        // Use HSV for perceptually distinct colors
        let hue = Float(i) / Float(numClasses)
        let saturation: Float = 0.8
        let value: Float = 0.9
        
        let rgb = hsvToRgb(h: hue, s: saturation, v: value)
        colors.append(rgb)
    }
    
    // Make background black
    colors[0] = (0, 0, 0)
    
    return colors
}
```

### Overlay Techniques

```swift
// Blend segmentation mask with original image
func createOverlay(
    image: UIImage,
    mask: [UInt8],
    colorMap: [(UInt8, UInt8, UInt8)],
    alpha: Float = 0.5
) -> UIImage {
    let width = Int(image.size.width)
    let height = Int(image.size.height)
    
    // Get image pixels
    var imagePixels = getPixels(from: image)
    
    // Blend with mask colors
    for i in 0..<(width * height) {
        let classId = Int(mask[i])
        let (r, g, b) = colorMap[classId]
        
        let idx = i * 4
        imagePixels[idx + 0] = UInt8(Float(imagePixels[idx + 0]) * (1 - alpha) + Float(r) * alpha)
        imagePixels[idx + 1] = UInt8(Float(imagePixels[idx + 1]) * (1 - alpha) + Float(g) * alpha)
        imagePixels[idx + 2] = UInt8(Float(imagePixels[idx + 2]) * (1 - alpha) + Float(b) * alpha)
    }
    
    return createImage(from: imagePixels, width: width, height: height)
}
```

---

## SwiftPixelUtils Segmentation API

### Basic Usage

```swift
import SwiftPixelUtils

// Process segmentation output
let result = try SegmentationOutput.process(
    outputData: modelOutput,
    outputWidth: 513,
    outputHeight: 513,
    originalWidth: Int(image.size.width),
    originalHeight: Int(image.size.height),
    labels: .pascalVOC,
    outputFormat: .logits
)

// Get class mask
let mask = result.classMask  // [UInt8] with class indices

// Get colored visualization
let coloredMask = result.coloredMask  // [UInt8] RGBA data

// Get per-class statistics
for stat in result.classStatistics {
    print("\(stat.label): \(stat.pixelCount) pixels (\(stat.percentage)%)")
}
```

### Configuration Options

```swift
let config = SegmentationConfig(
    // Output processing
    outputFormat: .logits,           // or .probabilities
    upsampleMethod: .bilinear,       // or .nearest
    
    // Filtering
    minPixelsPerClass: 100,          // Ignore tiny segments
    backgroundClass: 0,              // Index of background
    
    // Visualization
    overlayAlpha: 0.5,
    colorMap: .pascalVOC             // or .custom([...])
)

let result = try SegmentationOutput.process(
    outputData: output,
    labels: .pascalVOC,
    config: config
)
```

### Segmentation Result Structure

```swift
struct SegmentationResult {
    // Primary outputs
    let classMask: [UInt8]           // H×W class indices
    let confidenceMask: [Float]       // H×W max confidence
    let coloredMask: [UInt8]         // H×W×4 RGBA visualization
    
    // Dimensions
    let width: Int
    let height: Int
    
    // Statistics
    let classStatistics: [ClassStatistic]
    let detectedClasses: [Int]        // Classes with pixels > threshold
    
    // Optional
    let probabilities: [Float]?       // H×W×C full probabilities
}

struct ClassStatistic {
    let classIndex: Int
    let label: String
    let pixelCount: Int
    let percentage: Float
    let boundingBox: CGRect?          // Bounding box of class region
}
```

---

## Complete Implementation Examples

### Example 1: DeepLabV3 with TFLite

```swift
import SwiftPixelUtils
import TensorFlowLite

class SemanticSegmentor {
    private let interpreter: Interpreter
    private let inputWidth = 513
    private let inputHeight = 513
    
    init(modelPath: String) throws {
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }
    
    func segment(image: UIImage) throws -> SegmentationResult {
        // 1. Preprocess - DeepLabV3 expects specific normalization (synchronous)
        let input = try PixelExtractor.getModelInput(
            source: .uiImage(image),
            framework: .tfliteFloat,
            width: inputWidth,
            height: inputHeight,
            normalization: .custom(mean: [127.5, 127.5, 127.5], 
                                   std: [127.5, 127.5, 127.5])  // [-1, 1]
        )
        
        // 2. Run inference
        try interpreter.copy(input.data, toInputAt: 0)
        try interpreter.invoke()
        
        // 3. Get output
        let outputTensor = try interpreter.output(at: 0)
        
        // 4. Process segmentation
        return try SegmentationOutput.process(
            outputData: outputTensor.data,
            outputWidth: inputWidth,
            outputHeight: inputHeight,
            originalWidth: Int(image.size.width),
            originalHeight: Int(image.size.height),
            labels: .pascalVOC
        )
    }
}

// Usage
let segmentor = try SemanticSegmentor(modelPath: "deeplabv3.tflite")
let result = try segmentor.segment(image: myImage)

print("Detected classes:")
for stat in result.classStatistics where stat.pixelCount > 0 {
    print("  \(stat.label): \(String(format: "%.1f", stat.percentage))%")
}
```

### Example 2: Person Segmentation (Portrait Mode)

```swift
class PortraitSegmentor {
    private let segmentor: SemanticSegmentor
    private let personClassIndex = 15  // Pascal VOC person class
    
    func extractPerson(from image: UIImage) throws -> UIImage {
        let result = try segmentor.segment(image: image)
        
        // Create binary mask for person only
        let personMask = result.classMask.map { $0 == personClassIndex ? UInt8(255) : UInt8(0) }
        
        // Apply mask to original image
        return applyMask(image: image, mask: personMask)
    }
    
    func blurBackground(image: UIImage) throws -> UIImage {
        let result = try segmentor.segment(image: image)
        
        // Create person mask
        let personMask = result.classMask.map { Float($0 == personClassIndex ? 1.0 : 0.0) }
        
        // Blur original image
        let blurred = applyGaussianBlur(image: image, radius: 25)
        
        // Composite: person from original, rest from blurred
        return composite(
            foreground: image,
            background: blurred,
            mask: personMask
        )
    }
}
```

### Example 3: Scene Analysis

```swift
func analyzeScene(image: UIImage) throws -> SceneAnalysis {
    let result = try segmentor.segment(image: image)
    
    // Determine dominant scene type
    let outdoor = ["sky", "tree", "grass", "road", "building"]
    let indoor = ["wall", "floor", "ceiling", "furniture"]
    
    var outdoorPixels = 0
    var indoorPixels = 0
    
    for stat in result.classStatistics {
        if outdoor.contains(stat.label) {
            outdoorPixels += stat.pixelCount
        } else if indoor.contains(stat.label) {
            indoorPixels += stat.pixelCount
        }
    }
    
    let sceneType: SceneType = outdoorPixels > indoorPixels ? .outdoor : .indoor
    
    // Find main subjects
    let subjects = result.classStatistics
        .filter { !["background", "sky", "wall", "floor"].contains($0.label) }
        .filter { $0.percentage > 1.0 }
        .sorted { $0.percentage > $1.percentage }
        .prefix(5)
    
    return SceneAnalysis(
        sceneType: sceneType,
        mainSubjects: Array(subjects),
        segmentationResult: result
    )
}
```

---

## Advanced Topics

### Multi-Scale Inference

Run model at multiple scales and merge:

```swift
func multiScaleSegment(image: UIImage) throws -> SegmentationResult {
    let scales: [Float] = [0.5, 0.75, 1.0, 1.25, 1.5]
    var allProbs: [[Float]] = []
    
    for scale in scales {
        let scaledSize = CGSize(
            width: image.size.width * CGFloat(scale),
            height: image.size.height * CGFloat(scale)
        )
        let scaled = image.scaled(to: scaledSize)
        
        let result = try segmentWithProbs(scaled)
        
        // Resize probs back to original size
        let resized = resizeProbabilities(result.probabilities, 
                                          to: image.size)
        allProbs.append(resized)
    }
    
    // Average probabilities across scales
    var mergedProbs = [Float](repeating: 0, 
                              count: allProbs[0].count)
    for probs in allProbs {
        for i in 0..<probs.count {
            mergedProbs[i] += probs[i] / Float(scales.count)
        }
    }
    
    return createResult(from: mergedProbs)
}
```

### Sliding Window for Large Images

```swift
func segmentLargeImage(
    image: UIImage,
    windowSize: Int = 513,
    overlap: Int = 128
) throws -> SegmentationResult {
    let stride = windowSize - overlap
    let width = Int(image.size.width)
    let height = Int(image.size.height)
    
    // Initialize accumulator
    var sumProbs = [Float](repeating: 0, 
                           count: width * height * numClasses)
    var counts = [Float](repeating: 0, 
                         count: width * height)
    
    // Process each window
    for y in stride(from: 0, to: height, by: stride) {
        for x in stride(from: 0, to: width, by: stride) {
            // Extract window
            let window = extractWindow(image, x: x, y: y, 
                                       size: windowSize)
            
            // Segment window
            let result = try segment(window)
            
            // Accumulate results
            accumulateResults(result, into: &sumProbs, 
                            counts: &counts, 
                            offsetX: x, offsetY: y)
        }
    }
    
    // Average overlapping regions
    let finalProbs = divideByCount(sumProbs, counts: counts)
    
    return createResult(from: finalProbs)
}
```

### Instance Segmentation

Combine detection + segmentation:

```swift
struct InstanceSegmentation {
    let classId: Int
    let instanceId: Int
    let confidence: Float
    let boundingBox: CGRect
    let mask: [Bool]  // Binary mask for this instance
}

// Using Mask R-CNN or similar
func instanceSegment(image: UIImage) throws -> [InstanceSegmentation] {
    // 1. Detect objects
    let detections = try detect(image)
    
    // 2. For each detection, predict mask
    var instances: [InstanceSegmentation] = []
    
    for (i, detection) in detections.enumerated() {
        let crop = cropRegion(image, box: detection.boundingBox)
        let mask = try segmentInstance(crop)
        
        instances.append(InstanceSegmentation(
            classId: detection.classIndex,
            instanceId: i,
            confidence: detection.confidence,
            boundingBox: detection.boundingBox,
            mask: mask
        ))
    }
    
    return instances
}
```

### Panoptic Segmentation

Unified scene understanding:

```swift
struct PanopticSegmentation {
    // Semantic mask for "stuff" (sky, road, etc.)
    let stuffMask: [UInt8]
    
    // Instance masks for "things" (cars, people, etc.)
    let instances: [InstanceSegmentation]
    
    // Combined panoptic mask (unique ID per instance/stuff class)
    let panopticMask: [Int]
}
```

---

## Performance Optimization

### Memory Optimization

```swift
// Process in tiles to reduce memory
func memoryEfficientSegment(image: UIImage) async throws -> SegmentationResult {
    let tileSize = 256
    let numClasses = 21
    
    // Allocate output once
    let width = Int(image.size.width)
    let height = Int(image.size.height)
    var outputMask = [UInt8](repeating: 0, count: width * height)
    
    // Process tiles
    for tileY in 0..<(height / tileSize + 1) {
        for tileX in 0..<(width / tileSize + 1) {
            autoreleasepool {
                // Process single tile
                let tile = extractTile(...)
                let result = segment(tile)
                copyTileToOutput(result, into: &outputMask)
            }
        }
    }
    
    return createResult(from: outputMask)
}
```

### Model Selection by Device

| Device | Recommended Model | Input Size | Notes |
|--------|-------------------|------------|-------|
| iPhone 12+ | DeepLabV3+ MobileNet | 513×513 | Good balance |
| iPhone 11 | DeepLabV3 MobileNet | 321×321 | Faster, smaller |
| iPad Pro | DeepLabV3+ ResNet | 513×513 | Best quality |
| Apple Watch | Not recommended | - | Too resource intensive |

---

## Evaluation Metrics

### Mean IoU (mIoU)

Primary segmentation metric:

$$mIoU = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FP_k + FN_k}$$

Where $TP$, $FP$, $FN$ are true positives, false positives, false negatives for class $k$.

### Pixel Accuracy

$$Accuracy = \frac{\sum_k TP_k}{\sum_k (TP_k + FN_k)}$$

### Frequency-Weighted IoU

Weight by class frequency:

$$fwIoU = \frac{1}{\sum_k n_k} \sum_k \frac{n_k \cdot IoU_k}{1}$$

---

## Troubleshooting

### Problem: Blocky/Pixelated Output

**Cause:** Low output stride, nearest neighbor upsampling

**Solutions:**
- Use bilinear upsampling
- Use model with output stride 8
- Apply CRF post-processing

### Problem: Wrong Classes Detected

**Checklist:**
1. ✓ Correct preprocessing (normalization)?
2. ✓ Labels match model training?
3. ✓ Output format (logits vs probs)?
4. ✓ Channel ordering?

### Problem: Slow Inference

**Optimizations:**
- Reduce input size
- Use quantized model
- Use MobileNet backbone
- Process on GPU (Metal)

---

## Mathematical Foundations

### Cross-Entropy Loss for Segmentation

$$L = -\frac{1}{HW} \sum_{i,j} \sum_k y_{ijk} \log(p_{ijk})$$

Where $(i,j)$ indexes pixels and $k$ indexes classes.

### Dice Loss

$$L_{Dice} = 1 - \frac{2 \sum_{i,j} p_{ij} \cdot y_{ij}}{\sum_{i,j} p_{ij}^2 + \sum_{i,j} y_{ij}^2}$$

Better for class imbalance.

### IoU Loss

$$L_{IoU} = 1 - \frac{\sum_{i,j} p_{ij} \cdot y_{ij}}{\sum_{i,j} p_{ij} + \sum_{i,j} y_{ij} - \sum_{i,j} p_{ij} \cdot y_{ij}}$$

