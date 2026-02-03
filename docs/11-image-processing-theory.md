# Image Processing Theory: Core Reference

A focused theory reference covering the foundations behind image preprocessing, augmentation, and visualization. Use this as a conceptual companion to the feature‑specific guides.

## Table of Contents

- [Image Processing Theory: Core Reference](#image-processing-theory-core-reference)
  - [Table of Contents](#table-of-contents)
  - [Sampling and Quantization](#sampling-and-quantization)
  - [Filtering and Convolution](#filtering-and-convolution)
  - [Optics, PSF, and MTF](#optics-psf-and-mtf)
  - [Frequency Domain Intuition](#frequency-domain-intuition)
  - [Aliasing and Anti‑Aliasing](#aliasing-and-antialiasing)
  - [Noise Models](#noise-models)
  - [Dynamic Range and Tone Mapping](#dynamic-range-and-tone-mapping)
  - [Color and Perception](#color-and-perception)
  - [Edges, Gradients, and Feature Detectors](#edges-gradients-and-feature-detectors)
  - [Morphological Operations](#morphological-operations)
  - [Histogram Operations](#histogram-operations)
  - [Geometric Transforms](#geometric-transforms)
  - [Compression Artifacts](#compression-artifacts)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Decision Guide: What to Review](#decision-guide-what-to-review)

---

## Sampling and Quantization

A digital image samples a continuous scene onto a discrete grid. Sampling determines **where** we measure; quantization determines **how precisely** we store values.

- **Sampling rate** must satisfy the Nyquist criterion to avoid aliasing.
- **Quantization** introduces error bounded by half a quantization step.

Uniform quantization error (for step size $\Delta$):
$$
\varepsilon \sim \mathcal{U}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right), \quad \mathbb{E}[\varepsilon]=0, \quad \text{Var}(\varepsilon)=\frac{\Delta^2}{12}
$$

---

## Filtering and Convolution

Most image operations are linear, shift‑invariant filters applied via convolution.

2D convolution (discrete):
$$
(I * K)(x,y)=\sum_{i=-m}^{m}\sum_{j=-n}^{n} I(x-i,y-j)K(i,j)
$$

- **Low‑pass filters** smooth noise but blur edges.
- **High‑pass filters** sharpen edges but amplify noise.
- **Separable kernels** (e.g., Gaussian) reduce cost from $O(k^2)$ to $O(2k)$.

---

## Optics, PSF, and MTF

Real cameras blur images due to optics and sensor integration. This blur is modeled by the **Point Spread Function (PSF)**, which describes how a point light source spreads on the sensor.

- **PSF** in spatial domain ↔ **Optical Transfer Function (OTF)** in frequency domain.
- The **Modulation Transfer Function (MTF)** is the magnitude of the OTF and summarizes how contrast is preserved at different spatial frequencies.

**Why it matters:**
- Small objects can vanish if the PSF is large.
- Super‑resolution and deblurring attempt to invert PSF effects.

---

---

## Frequency Domain Intuition

The Fourier transform decomposes an image into spatial frequencies:

- **Low frequencies** = smooth regions, illumination.
- **High frequencies** = edges, texture.

Filtering in frequency space corresponds to convolution in spatial space (Convolution Theorem).

---

## Aliasing and Anti‑Aliasing

Aliasing occurs when high‑frequency content is sampled too coarsely. It appears as jagged edges or moiré patterns.

**Anti‑aliasing** uses low‑pass filtering before downsampling:

1. Blur (reduce high frequencies)
2. Downsample

This is why high‑quality resize uses prefiltering.

---

## Noise Models

Common noise types:

- **Gaussian**: sensor/thermal noise, modeled as $\mathcal{N}(0,\sigma^2)$.
- **Poisson**: photon shot noise, variance proportional to signal.
- **Salt‑and‑pepper**: impulse noise (dead pixels/compression).

Signal‑to‑Noise Ratio (SNR):
$$
\text{SNR (dB)}=10\log_{10}\left(\frac{\sigma_{signal}^2}{\sigma_{noise}^2}\right)
$$

---

## Dynamic Range and Tone Mapping

**Dynamic range** is the ratio between the darkest and brightest representable intensities. Sensors often capture more range than displays can show.

- **HDR** merges multiple exposures or uses high‑bit sensors.
- **Tone mapping** compresses HDR into displayable range while preserving contrast.

Common tone‑mapping curve (log):
$$
I_{out} = \frac{\log(1 + \alpha I_{in})}{\log(1 + \alpha)}
$$

Use tone mapping before ML only if the model was trained on tone‑mapped images.

---

## Color and Perception

Human vision is more sensitive to **luminance** than **chrominance**, enabling YUV/YCbCr subsampling. Perceptual uniform spaces (CIELAB) help quantify color differences.

Color difference (Delta‑E, simplified):
$$
\Delta E_{ab} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}
$$

---

## Edges, Gradients, and Feature Detectors

Edges are locations of rapid intensity change and often correspond to object boundaries.

- **Gradient magnitude**: $|\nabla I| = \sqrt{I_x^2 + I_y^2}$
- **Sobel/Scharr** filters estimate gradients.
- **Canny** adds non‑maximum suppression and hysteresis for clean edges.

Feature detectors (Harris, FAST, DoG) rely on gradient structure. When inputs are overly blurred or aliased, edge‑based features degrade.

---

## Morphological Operations

Morphology modifies shapes using a structuring element:

- **Erosion** removes small bright regions.
- **Dilation** expands bright regions.
- **Opening** (erosion→dilation) removes small noise.
- **Closing** (dilation→erosion) fills small holes.

These are useful for post‑processing segmentation masks.

---

## Histogram Operations

Histograms summarize intensity distributions.

- **Histogram equalization** improves contrast globally.
- **CLAHE** improves contrast locally while limiting noise amplification.

Use histogram operations cautiously; they can alter model‑expected statistics.

---

## Geometric Transforms

Rigid and affine transforms are represented with matrices:

$$
\begin{bmatrix}x'\\y'\\1\end{bmatrix}=
\begin{bmatrix}
 a & b & t_x \\
 c & d & t_y \\
 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}x\\y\\1\end{bmatrix}
$$

- **Rotation** preserves lengths/angles.
- **Shear** preserves parallel lines.
- **Perspective** models pinhole cameras.

Interpolation (nearest, bilinear, bicubic) determines how resampled pixels are computed.

---

## Compression Artifacts

Lossy compression (JPEG, H.264) introduces blockiness, ringing, and color shifts.

- **Blocking artifacts** occur at 8×8 boundaries.
- **Ringing** appears near sharp edges.

Compression artifacts can confuse detectors; if your inputs are heavily compressed, consider light denoising or train with similar compression levels.

---

## Evaluation Metrics

Common metrics to evaluate image processing effects:

- **PSNR** (Peak Signal‑to‑Noise Ratio):
$$
\text{PSNR}=10\log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

- **SSIM** (Structural Similarity) evaluates luminance, contrast, and structure.

These metrics help quantify preprocessing fidelity or augmentation severity.

---

## Decision Guide: What to Review

- **Blurry results** → review filtering and aliasing.
- **Color shifts** → review color and perception.
- **Jagged edges after resize** → review anti‑aliasing and interpolation.
- **Noisy outputs** → review noise models and denoising trade‑offs.
- **Geometric distortions** → review transforms and coordinate systems.
