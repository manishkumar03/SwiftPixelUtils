//
//  ModelPresets.swift
//  SwiftPixelUtils
//
//  Pre-configured settings for common ML models
//

import Foundation
import CoreGraphics

/// Pre-configured preprocessing settings for popular ML models.
///
/// ## Overview
///
/// Different ML models expect different input preprocessing. Using incorrect settings
/// causes silent accuracy degradation or outright failures. These presets match the
/// exact preprocessing used during model training.
///
/// ## Usage
///
/// ```swift
/// // For YOLO object detection
/// let result = try await PixelExtractor.getPixelData(
///     from: .cgImage(image),
///     options: ModelPresets.yolo
/// )
///
/// // For CLIP embeddings
/// let result = try await PixelExtractor.getPixelData(
///     from: .cgImage(image),
///     options: ModelPresets.clip
/// )
/// ```
///
/// ## Why Settings Differ
///
/// | Model Family | Input Size | Normalization | Layout | Reason |
/// |-------------|------------|---------------|--------|--------|
/// | YOLO | 640×640 | Scale [0,1] | NCHW | Trained on COCO with letterboxing |
/// | MobileNet | 224×224 | ImageNet | NHWC | TensorFlow convention |
/// | ResNet | 224×224 | ImageNet | NCHW | PyTorch convention |
/// | CLIP | 224×224 | Custom | NCHW | OpenAI's LAION statistics |
/// | SAM | 1024×1024 | ImageNet | NCHW | High-res segmentation |
/// | ExecuTorch | varies | varies | NCHW | PyTorch-exported models |
///
/// ## ExecuTorch Compatibility
///
/// All presets with NCHW layout work directly with ExecuTorch models exported from PyTorch.
/// For quantized ExecuTorch models, use the output with ``Quantizer`` to convert to Int8.
///
/// ## Creating Custom Presets
///
/// For models not listed, check the model's documentation for:
/// 1. **Input size**: Usually in model architecture or config
/// 2. **Normalization**: Mean/std values from training code
/// 3. **Data layout**: NCHW (PyTorch) vs NHWC (TensorFlow)
/// 4. **Channel order**: RGB (most) vs BGR (OpenCV-based)
public enum ModelPresets {
    
    // MARK: - YOLO Models
    
    /// YOLO (You Only Look Once) object detection preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 640×640**: YOLO's default input size, balances speed and accuracy.
    ///   Larger sizes (1280) improve small object detection but are slower.
    ///
    /// - **Letterbox resize**: Preserves aspect ratio by adding gray padding (114, 114, 114).
    ///   Critical for accurate bounding box predictions. Stretching would distort
    ///   learned aspect ratio priors.
    ///
    /// - **Scale normalization [0, 1]**: YOLO uses simple 0-1 scaling, not ImageNet stats.
    ///   This matches the original Darknet training code.
    ///
    /// - **NCHW layout**: YOLO implementations (Ultralytics, ONNX exports) use
    ///   channels-first format from PyTorch training.
    ///
    /// ## Why Letterbox?
    ///
    /// ```
    /// Original (16:9):        Letterboxed (1:1):
    /// ┌────────────────┐      ┌────────────────┐
    /// │                │      │░░░░░░░░░░░░░░░░│
    /// │     image      │  →   │     image      │
    /// │                │      │░░░░░░░░░░░░░░░░│
    /// └────────────────┘      └────────────────┘
    ///
    /// Gray padding (114) chosen to minimize edge artifacts at detection boundaries.
    /// ```
    ///
    /// ## Post-Processing Note
    ///
    /// Remember to scale bounding box outputs back to original image coordinates:
    /// ```swift
    /// let boxes = BoundingBox.scale(rawBoxes, from: CGSize(width: 640, height: 640), to: originalSize, format: .cxcywh)
    /// ```
    public static let yolo = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 640, height: 640, strategy: .letterbox),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// YOLOv8 uses the same preprocessing as YOLO base.
    public static let yolov8 = yolo
    
    // MARK: - MobileNet Models
    
    /// MobileNet family preprocessing (v1, v2, v3).
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ImageNet classification input size.
    ///   MobileNet was designed for efficient inference at this resolution.
    ///
    /// - **Cover resize**: Crops to fill the target, maximizing image content.
    ///   Classification assumes the subject is roughly centered.
    ///
    /// - **ImageNet normalization**: MobileNet trained on ImageNet uses the standard
    ///   mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    ///
    /// - **NHWC layout**: TensorFlow convention (MobileNet originated at Google).
    ///   Use NCHW if using PyTorch ports.
    ///
    /// ## Performance Notes
    ///
    /// MobileNet uses depthwise separable convolutions for efficiency:
    /// - v1: ~4.2M parameters, ~569M MACs
    /// - v2: ~3.4M parameters, ~300M MACs (inverted residuals)
    /// - v3: ~5.4M parameters, ~219M MACs (neural architecture search)
    public static let mobilenet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nhwc,
        outputFormat: .float32Array
    )
    
    /// MobileNetV2 uses the same preprocessing as MobileNet base.
    public static let mobilenet_v2 = mobilenet
    
    /// MobileNetV3 uses the same preprocessing as MobileNet base.
    public static let mobilenet_v3 = mobilenet
    
    // MARK: - EfficientNet
    
    /// EfficientNet preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: B0 variant size. Larger variants use bigger inputs:
    ///   - B1: 240×240, B2: 260×260, B3: 300×300, B4: 380×380, etc.
    ///
    /// - **ImageNet normalization**: Standard ImageNet statistics.
    ///
    /// - **NHWC layout**: TensorFlow convention (EfficientNet is from Google Brain).
    ///
    /// ## Scaling Note
    ///
    /// EfficientNet uses compound scaling (depth, width, resolution together).
    /// For larger variants, modify the resize dimensions accordingly.
    public static let efficientnet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nhwc,
        outputFormat: .float32Array
    )
    
    // MARK: - ResNet Models
    
    /// ResNet family preprocessing (18, 34, 50, 101, 152).
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ImageNet classification input size.
    ///
    /// - **ImageNet normalization**: ResNet trained on ImageNet uses the standard
    ///   mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    ///
    /// - **NCHW layout**: PyTorch convention (ResNet reference implementation).
    ///
    /// ## Architecture Note
    ///
    /// ResNet's skip connections enable training very deep networks without
    /// vanishing gradients. The identity mappings allow gradients to flow
    /// directly through the network during backpropagation.
    public static let resnet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// ResNet-50 uses the same preprocessing as ResNet base.
    public static let resnet50 = resnet
    
    // MARK: - Vision Transformer
    
    /// Vision Transformer (ViT) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ViT-B/16 input size (16×16 patches = 14×14 grid).
    ///   ViT-L uses 224×224 or 384×384 depending on variant.
    ///
    /// - **ImageNet normalization**: ViT models trained on ImageNet-21k or ImageNet-1k
    ///   use standard statistics.
    ///
    /// - **NCHW layout**: PyTorch/JAX convention for most ViT implementations.
    ///
    /// ## Patch Embedding
    ///
    /// ViT divides the image into fixed-size patches (typically 16×16 or 32×32),
    /// then linearly embeds each patch. A 224×224 image with 16×16 patches yields
    /// a sequence of 196 tokens (14 × 14), plus a [CLS] token for classification.
    public static let vit = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - CLIP
    
    /// CLIP (Contrastive Language-Image Pre-training) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: CLIP ViT-B/32 and ViT-B/16 input size.
    ///   CLIP ViT-L/14 uses 224×224, ViT-L/14@336px uses 336×336.
    ///
    /// - **Custom normalization**: CLIP uses statistics from LAION-400M dataset,
    ///   not ImageNet. These values differ slightly:
    ///   - Mean: [0.48145466, 0.4578275, 0.40821073]
    ///   - Std:  [0.26862954, 0.26130258, 0.27577711]
    ///
    /// - **NCHW layout**: PyTorch convention (OpenAI's implementation).
    ///
    /// ## Why Different Statistics?
    ///
    /// CLIP was trained on 400M image-text pairs from the internet, which has
    /// different color distributions than ImageNet's curated 1.2M images.
    /// The LAION dataset includes more diverse content (memes, photos, art),
    /// resulting in slightly different mean/std values.
    ///
    /// ## Zero-Shot Classification
    ///
    /// CLIP enables zero-shot classification by comparing image embeddings
    /// with text embeddings of class descriptions. No task-specific training required.
    public static let clip = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: Normalization(
            preset: .custom,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711]
        ),
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - SAM (Segment Anything Model)
    
    /// SAM (Segment Anything Model) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 1024×1024**: SAM uses high resolution for precise segmentation.
    ///   The image encoder (ViT-H) can handle this large input.
    ///
    /// - **Contain resize**: Preserves aspect ratio without cropping, important
    ///   for segmentation where all image content matters.
    ///
    /// - **ImageNet normalization**: SAM uses standard ImageNet statistics.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Architecture Note
    ///
    /// SAM's image encoder runs once per image, then the mask decoder can
    /// generate multiple masks for different prompts (points, boxes, text).
    /// The high input resolution enables fine-grained boundary predictions.
    public static let sam = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 1024, height: 1024, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - DINO
    
    /// DINO (Self-Distillation with No Labels) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 224×224**: Standard ViT input size for DINO.
    ///
    /// - **ImageNet normalization**: DINO uses standard ImageNet statistics.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Self-Supervised Learning
    ///
    /// DINO learns visual features without labels by having a student network
    /// match the output of a teacher network (momentum-updated copy).
    /// The learned features are remarkably good for k-NN classification,
    /// object discovery, and semantic segmentation.
    public static let dino = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 224, height: 224, strategy: .cover),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - DETR
    
    /// DETR (DEtection TRansformer) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 800×800**: DETR uses larger input for object detection.
    ///   The original paper uses 800 max dimension with aspect ratio preserved.
    ///
    /// - **Contain resize**: Preserves aspect ratio, important for accurate
    ///   bounding box predictions.
    ///
    /// - **ImageNet normalization**: DETR uses standard ImageNet statistics
    ///   (backbone is typically ResNet-50 pretrained on ImageNet).
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Transformer Detection
    ///
    /// DETR treats detection as a set prediction problem, using a transformer
    /// encoder-decoder to directly output a fixed set of predictions.
    /// Hungarian matching during training assigns predictions to ground truth.
    public static let detr = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 800, height: 800, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - RT-DETR (Real-Time DETR)
    
    /// RT-DETR (Real-Time Detection Transformer) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 640×640**: RT-DETR uses 640×640 by default for real-time performance.
    ///   Supports multi-scale: 640 (fast), 800 (balanced), 1024 (accurate).
    ///
    /// - **Letterbox resize**: Like YOLO, preserves aspect ratio with gray padding.
    ///   Critical for accurate localization without aspect ratio distortion.
    ///
    /// - **Scale normalization [0, 1]**: RT-DETR uses simple 0-1 scaling.
    ///
    /// - **NCHW layout**: PyTorch convention (Baidu PaddleDetection origin).
    ///
    /// ## Why RT-DETR?
    ///
    /// RT-DETR achieves real-time performance (30+ FPS) while maintaining
    /// transformer-based detection accuracy. Key innovations:
    /// - Efficient hybrid encoder (CNN + transformer)
    /// - IoU-aware query selection
    /// - Decoupled intra-scale and cross-scale feature interaction
    ///
    /// ## Variants
    ///
    /// - RT-DETR-L: 32M params, 53.0 AP on COCO
    /// - RT-DETR-X: 67M params, 54.8 AP on COCO
    public static let rtdetr = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 640, height: 640, strategy: .letterbox),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// RT-DETR-L (Large variant) - same preprocessing as base.
    public static let rtdetr_l = rtdetr
    
    /// RT-DETR-X (Extra-large variant) - same preprocessing as base.
    public static let rtdetr_x = rtdetr
    
    // MARK: - YOLOv10
    
    /// YOLOv10 preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 640×640**: Standard YOLO input size for balanced speed/accuracy.
    ///
    /// - **Letterbox resize**: Preserves aspect ratio with gray (114, 114, 114) padding.
    ///
    /// - **Scale normalization [0, 1]**: Simple 0-1 scaling like other YOLO variants.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## YOLOv10 Innovations
    ///
    /// YOLOv10 introduces NMS-free training with consistent dual assignments:
    /// - One-to-one head for inference (no NMS needed)
    /// - One-to-many head for training (better supervision)
    /// - Efficiency-accuracy driven model design
    /// - Holistic efficiency-accuracy optimization
    ///
    /// ## Variants
    ///
    /// - YOLOv10-N: 2.3M params, 38.5 AP (nano)
    /// - YOLOv10-S: 7.2M params, 46.3 AP (small)
    /// - YOLOv10-M: 15.4M params, 51.1 AP (medium)
    /// - YOLOv10-B: 19.1M params, 52.5 AP (balanced)
    /// - YOLOv10-L: 24.4M params, 53.2 AP (large)
    /// - YOLOv10-X: 29.5M params, 54.4 AP (extra-large)
    public static let yolov10 = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 640, height: 640, strategy: .letterbox),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// YOLOv10-N (Nano) - same preprocessing.
    public static let yolov10_n = yolov10
    
    /// YOLOv10-S (Small) - same preprocessing.
    public static let yolov10_s = yolov10
    
    /// YOLOv10-M (Medium) - same preprocessing.
    public static let yolov10_m = yolov10
    
    /// YOLOv10-L (Large) - same preprocessing.
    public static let yolov10_l = yolov10
    
    /// YOLOv10-X (Extra-large) - same preprocessing.
    public static let yolov10_x = yolov10
    
    // MARK: - YOLOv9
    
    /// YOLOv9 preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// Same preprocessing as other YOLO variants. YOLOv9 introduces:
    /// - Programmable Gradient Information (PGI)
    /// - Generalized Efficient Layer Aggregation Network (GELAN)
    ///
    /// ## Variants
    ///
    /// - YOLOv9-T: Tiny
    /// - YOLOv9-S: Small
    /// - YOLOv9-M: Medium
    /// - YOLOv9-C: Compact (51.4 AP)
    /// - YOLOv9-E: Extended (55.6 AP)
    public static let yolov9 = yolo
    
    // MARK: - SAM2 (Segment Anything Model 2)
    
    /// SAM2 (Segment Anything Model 2) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 1024×1024**: High resolution for precise segmentation.
    ///   SAM2 maintains the same input resolution as SAM for compatibility.
    ///
    /// - **Contain resize**: Preserves aspect ratio, pads if necessary.
    ///   Full image context is important for segmentation.
    ///
    /// - **ImageNet normalization**: Standard ImageNet statistics.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## SAM2 Improvements
    ///
    /// SAM2 extends SAM to video with streaming memory:
    /// - 6× faster than SAM on images
    /// - Memory attention for video object tracking
    /// - Occlusion handling with memory bank
    /// - Promptable visual segmentation across frames
    ///
    /// ## Model Sizes
    ///
    /// - SAM2-T: Tiny (38.9M params)
    /// - SAM2-S: Small (46M params)
    /// - SAM2-B+: Base Plus (80.8M params)
    /// - SAM2-L: Large (224.4M params)
    public static let sam2 = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 1024, height: 1024, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// SAM2 Tiny variant - same preprocessing.
    public static let sam2_t = sam2
    
    /// SAM2 Small variant - same preprocessing.
    public static let sam2_s = sam2
    
    /// SAM2 Base Plus variant - same preprocessing.
    public static let sam2_b_plus = sam2
    
    /// SAM2 Large variant - same preprocessing.
    public static let sam2_l = sam2
    
    // MARK: - Mask2Former
    
    /// Mask2Former preprocessing for universal image segmentation.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 512×512**: Default size balancing quality and speed.
    ///   Can use 640×640 or 1024×1024 for higher accuracy.
    ///
    /// - **Contain resize**: Preserves aspect ratio for accurate segmentation.
    ///
    /// - **ImageNet normalization**: Backbone pretrained on ImageNet.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Universal Segmentation
    ///
    /// Mask2Former unifies semantic, instance, and panoptic segmentation:
    /// - Masked attention for efficient training
    /// - Multi-scale high-resolution features
    /// - Query-based mask prediction
    ///
    /// ## Backbones
    ///
    /// - ResNet-50: 44M params
    /// - ResNet-101: 63M params
    /// - Swin-T: 47M params
    /// - Swin-L: 216M params (best accuracy)
    public static let mask2former = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 512, height: 512, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// Mask2Former with Swin-T backbone - same preprocessing.
    public static let mask2former_swin_t = mask2former
    
    /// Mask2Former with Swin-L backbone - same preprocessing.
    public static let mask2former_swin_l = mask2former
    
    // MARK: - UNet
    
    /// UNet preprocessing for semantic segmentation.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 512×512**: Common UNet input size. Original paper used 572×572
    ///   for biomedical images. Many implementations use powers of 2.
    ///
    /// - **Contain resize**: Preserves aspect ratio. For medical imaging,
    ///   stretching could distort anatomical structures.
    ///
    /// - **Scale normalization [0, 1]**: Simple 0-1 scaling is common.
    ///   Medical imaging may use dataset-specific normalization.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Architecture
    ///
    /// ```
    /// Encoder (Contracting)        Decoder (Expanding)
    /// ┌─────────────────┐         ┌─────────────────┐
    /// │   64 filters    │ ──────► │   64 filters    │
    /// ├─────────────────┤         ├─────────────────┤
    /// │   128 filters   │ ──────► │   128 filters   │
    /// ├─────────────────┤         ├─────────────────┤
    /// │   256 filters   │ ──────► │   256 filters   │
    /// ├─────────────────┤         ├─────────────────┤
    /// │   512 filters   │ ──────► │   512 filters   │
    /// └─────────────────┘         └─────────────────┘
    ///            └──── Bottleneck (1024) ────┘
    /// ```
    ///
    /// Skip connections concatenate encoder features to decoder for precise localization.
    public static let unet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 512, height: 512, strategy: .contain),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// UNet with 256×256 input - faster inference.
    public static let unet_256 = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 256, height: 256, strategy: .contain),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// UNet with 1024×1024 input - higher resolution segmentation.
    public static let unet_1024 = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 1024, height: 1024, strategy: .contain),
        normalization: .scale,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - DeepLab
    
    /// DeepLabV3/DeepLabV3+ preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 513×513**: DeepLab uses odd sizes (513, 769, 1025) due to
    ///   atrous convolution stride alignment. 513 = 16×32 + 1.
    ///
    /// - **Contain resize**: Preserves aspect ratio for accurate segmentation.
    ///
    /// - **ImageNet normalization**: Backbone (ResNet, Xception) pretrained on ImageNet.
    ///
    /// - **NCHW layout**: PyTorch convention. NHWC for TensorFlow implementations.
    ///
    /// ## Atrous Spatial Pyramid Pooling (ASPP)
    ///
    /// DeepLab's key innovation captures multi-scale context:
    /// ```
    /// ┌─────────────────────────────────────┐
    /// │            ASPP Module              │
    /// │  ┌────┐ ┌────┐ ┌────┐ ┌────┐        │
    /// │  │ 1×1│ │ 3×3│ │ 3×3│ │ 3×3│ Pool   │
    /// │  │rate│ │ r=6│ │r=12│ │r=18│        │
    /// │  │ =1 │ │    │ │    │ │    │        │
    /// │  └────┘ └────┘ └────┘ └────┘        │
    /// │           └── Concat ──┘            │
    /// └─────────────────────────────────────┘
    /// ```
    ///
    /// ## Variants
    ///
    /// - DeepLabV3: ASPP on top of ResNet
    /// - DeepLabV3+: Adds decoder module for sharper boundaries
    public static let deeplab = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 513, height: 513, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// DeepLabV3 - same preprocessing as base.
    public static let deeplabv3 = deeplab
    
    /// DeepLabV3+ - same preprocessing as base.
    public static let deeplabv3_plus = deeplab
    
    /// DeepLab with 769×769 input - higher resolution.
    public static let deeplab_769 = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 769, height: 769, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// DeepLab with 1025×1025 input - highest resolution.
    public static let deeplab_1025 = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 1025, height: 1025, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - SegFormer
    
    /// SegFormer preprocessing for semantic segmentation.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 512×512**: Default SegFormer input size.
    ///
    /// - **Contain resize**: Preserves aspect ratio.
    ///
    /// - **ImageNet normalization**: Standard ImageNet statistics.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Transformer Segmentation
    ///
    /// SegFormer combines hierarchical transformers with lightweight MLP decoders:
    /// - Mix-FFN: 3×3 convolutions in feed-forward network
    /// - Efficient self-attention without positional encoding
    /// - Multi-scale features without complex decoders
    ///
    /// ## Variants
    ///
    /// - SegFormer-B0: 3.8M params, 37.4 mIoU (ADE20K)
    /// - SegFormer-B1: 13.7M params, 42.2 mIoU
    /// - SegFormer-B2: 27.4M params, 46.5 mIoU
    /// - SegFormer-B3: 47.3M params, 49.4 mIoU
    /// - SegFormer-B4: 64.1M params, 50.3 mIoU
    /// - SegFormer-B5: 84.7M params, 51.0 mIoU
    public static let segformer = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 512, height: 512, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    /// SegFormer-B0 (smallest) - same preprocessing.
    public static let segformer_b0 = segformer
    
    /// SegFormer-B5 (largest) - same preprocessing.
    public static let segformer_b5 = segformer
    
    // MARK: - FCN (Fully Convolutional Network)
    
    /// FCN preprocessing for semantic segmentation.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 512×512**: Common FCN input size.
    ///
    /// - **Contain resize**: Preserves aspect ratio.
    ///
    /// - **ImageNet normalization**: VGG/ResNet backbone pretrained on ImageNet.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Architecture Variants
    ///
    /// - FCN-32s: Single upsampling (coarse)
    /// - FCN-16s: Skip from pool4 (medium)
    /// - FCN-8s: Skips from pool3 and pool4 (fine)
    ///
    /// FCN pioneered dense prediction with fully convolutional architectures.
    public static let fcn = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 512, height: 512, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
    
    // MARK: - PSPNet
    
    /// PSPNet (Pyramid Scene Parsing Network) preprocessing.
    ///
    /// ## Configuration Rationale
    ///
    /// - **Size 473×473**: PSPNet uses this size for scene parsing.
    ///   Also common: 713×713 for higher resolution.
    ///
    /// - **Contain resize**: Preserves aspect ratio.
    ///
    /// - **ImageNet normalization**: ResNet backbone pretrained on ImageNet.
    ///
    /// - **NCHW layout**: PyTorch convention.
    ///
    /// ## Pyramid Pooling Module
    ///
    /// ```
    /// ┌─────────────────────────────────────┐
    /// │     Pyramid Pooling Module          │
    /// │  ┌───┐ ┌───┐ ┌───┐ ┌───┐          │
    /// │  │1×1│ │2×2│ │3×3│ │6×6│          │
    /// │  │bin│ │bin│ │bin│ │bin│          │
    /// │  └───┘ └───┘ └───┘ └───┘          │
    /// │        └── Upsample & Concat ──┘   │
    /// └─────────────────────────────────────┘
    /// ```
    ///
    /// Captures global context at multiple scales.
    public static let pspnet = PixelDataOptions(
        colorFormat: .rgb,
        resize: ResizeOptions(width: 473, height: 473, strategy: .contain),
        normalization: .imagenet,
        dataLayout: .nchw,
        outputFormat: .float32Array
    )
}
