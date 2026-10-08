# 📖 NeuroNetWorks: Comprehensive User Manual

Welcome to **NeuroNetWorks**, an interactive desktop deep learning educational laboratory built with **PyTorch**, **NumPy**, and **PyQt6**. 

This manual is designed as an interactive user guide to help you navigate, experiment with, and master every feature of the NeuroNetWorks application. Whether you are taking your first steps in machine learning or analyzing complex neural network dynamics, this guide provides hands-on walkthroughs, interface explanations, and self-learning experiments for every module.

---

## 📑 Table of Contents
1. [System Setup & Launch Options](#1-system-setup--launch-options)
2. [Global User Interface Overview](#2-global-user-interface-overview)
3. [Module 1: MNIST Handwriting Digit Recognizer & Vision Inspector](#3-module-1-mnist-handwriting-digit-recognizer--vision-inspector)
4. [Auxiliary Neural Activity Inspector (Pop-Out / Multi-Monitor)](#4-auxiliary-neural-activity-inspector-pop-out--multi-monitor)
5. [Module 2: 2D Decision Boundary Playground](#5-module-2-2d-decision-boundary-playground)
6. [Module 3: Autoencoder Latent Space Explorer & Denoising](#6-module-3-autoencoder-latent-space-explorer--denoising)
7. [Module 4: Universal Function Approximator (1D Regression)](#7-module-4-universal-function-approximator-1d-regression)
8. [Module 5: Real-Time Signal & Audio Waveform Denoising (1D-CNN)](#8-module-5-real-time-signal--audio-waveform-denoising-1d-cnn)
9. [Module 6: Reinforcement Learning Cart-Pole Balancer (DQN)](#9-module-6-reinforcement-learning-cart-pole-balancer-dqn)
10. [Troubleshooting & Performance Tips](#10-troubleshooting--performance-tips)

---

## 1. System Setup & Launch Options

### 1.1 Requirements
* **Operating System:** Linux (X11 or Wayland), macOS, or Windows
* **Python Environment:** Python 3.10+ (tested up to Python 3.14)
* **Core Libraries:** `torch`, `torchvision`, `numpy`, `PyQt6`, `scipy`

### 1.2 Starting the Application
From the project root directory, you can launch the application using the launcher script:
```bash
./launch.sh
```

Or invoke Python directly using the project virtual environment:
```bash
.venv/bin/python app.py
```

### 1.3 Command-Line Arguments & Shortcuts
NeuroNetWorks provides dedicated CLI arguments for quick access and automated verification:

| Command | Action |
| :--- | :--- |
| `./launch.sh --test` | Runs the automated self-test across all 6 modules and exits. |
| `./launch.sh --module 0` | Boots directly into Module 1: MNIST Digit Recognizer. |
| `./launch.sh --module 1` | Boots directly into Module 2: 2D Decision Boundary Playground. |
| `./launch.sh --module 2` | Boots directly into Module 3: Autoencoder Latent Space Explorer. |
| `./launch.sh --module 3` | Boots directly into Module 4: Universal Function Approximator. |
| `./launch.sh --module 4` | Boots directly into Module 5: Real-Time Signal Denoising. |
| `./launch.sh --module 5` | Boots directly into Module 6: Reinforcement Learning Cart-Pole Balancer. |

---

## 2. Global User Interface Overview

The interface follows a **Cyber-Lab** aesthetic designed for clarity, high contrast, and responsive real-time feedback.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🧠 NEURONETWORKS   Cutting-Edge Deep Learning Suite  [🔍 Inspector] [⚡ CPU]│  Top Header
├───────────────┬─────────────────────────────────────────────────────────────┤
│ ✍️ MNIST CNN  │                                                             │
│ 🌐 Boundaries │                                                             │
│ 🌌 Autoencoder│                   Active Module Workspace                   │  Main Area
│ 📈 Approx 1D  │             (Interactive Canvases, Sliders,                 │
│ 📡 Signal CNN │              Probability Bars, Real-Time Plots)             │
│ 🕹️ CartPole RL│                                                             │
├───────────────┴─────────────────────────────────────────────────────────────┤
│ Ready. Active Module: MNIST Handwriting Recognizer  | PyTorch v2.x | PyQt6  │  Status Bar
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Elements:
1. **Top Header Bar:**
   * **Application Title & Subtitle:** Confirms current version and environment status.
   * **🔍 Neural Activity Inspector Button:** Opens or focuses the detachable multi-monitor neural inspector window.
   * **⚡ Hardware Accelerator Badge:** Displays the active compute device (e.g., `CUDA: NVIDIA RTX ...`, `Apple Silicon MPS`, or `Intel / CPU (N threads)`).
2. **Left Navigation Sidebar:**
   * Displays the 6 core modules with clear icons and descriptions.
   * Clicking any button immediately switches the central viewport without interrupting background processes.
3. **Bottom Status Ribbon:**
   * Shows currently loaded PyTorch and NumPy versions alongside system status messages.

---

## 3. Module 1: MNIST Handwriting Digit Recognizer & Vision Inspector

Module 1 demonstrates computer vision classification using deep 2D Convolutional Neural Networks and Multi-Layer Perceptrons on handwritten digits.

```
┌───────────────────────────────┬─────────────────────────────────────────────┐
│ INTERACTIVE DRAWING PAD       │ PYTORCH PREDICTION & CONFIDENCE             │
│ ┌───────────────────────────┐ │ [WINNING: 7]   [CONF: 99.4%]  [H: 0.08 bits]│
│ │                           │ ├─────────────────────────────────────────────┤
│ │         ✍️ 7              │ │ Active Architecture: [Deep CNN ▼]          │
│ │                           │ │ Digit 0: ░░░░░░░░░░░░░░░░░░░░░░░░░  0.0%    │
│ └───────────────────────────┘ │ ...                                         │
│ [🗑️ Clear] [☑ Centroid Align] │ Digit 7: █████████████████████████ 99.4%    │
│ Brush Stroke Width: [═══●═══] │ Digit 8: ░░░░░░░░░░░░░░░░░░░░░░░░░  0.1%    │
│ Test Samples: [0][1]..[9][🎲] │ Digit 9: ░░░░░░░░░░░░░░░░░░░░░░░░░  0.4%    │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

### 3.1 Controls & Usage
* **Drawing Canvas (280×280):** Click and drag the left mouse button to sketch any single digit (0 through 9).
* **Clear Button (`🗑️ Clear`):** Wipes the canvas back to pure black background.
* **Centroid Alignment Checkbox (`☑ Centroid Alignment`):**
  * When checked, the drawing pad automatically calculates the center-of-mass $(\bar{x}, \bar{y})$ of your strokes and shifts the digit directly to the center of the $28\times 28$ input grid.
  * *Why this matters:* The standard MNIST training dataset is strictly normalized by center of mass. Centroid alignment drastically improves recognition accuracy for off-center sketches.
* **Brush Stroke Width Slider:** Adjusts drawing thickness between 10px and 34px. A stroke width of ~20px best mirrors the anti-aliased pens used in the original MNIST collection.
* **Authentic Test Digits Carousel (`[0]` to `[9]` & `[🎲]`):**
  * Click any digit button (`0`–`9`) to immediately load a real handwritten sample from the 1,000-sample test set.
  * Click the dice icon (`🎲`) to pick a random test digit.
* **Active Architecture Selector:**
  * **Deep CNN:** 2-layer Convolutional Network with Max Pooling and Dense readout.
  * **MLP Baseline:** Dense fully connected network (`784 → 128 → 64 → 10`).

### 3.2 Reading the Output Telemetry
* **WINNING CLASS:** The argmax predicted digit (0 through 9).
* **CONFIDENCE (%):** Softmax posterior probability $P(\text{class} = d \mid \mathbf{x})$.
* **UNCERTAINTY (Entropy in bits):**
  * Shannon Entropy: $H(p) = -\sum_{i=0}^9 p_i \log_2(p_i)$.
  * A decisive prediction yields entropy close to **0.00 bits**.
  * An ambiguous drawing (e.g., a cross between a 4 and a 9) produces higher entropy (up to **3.32 bits** for a completely uniform distribution).

### 3.3 Educational Self-Study Experiments
> [!TIP]
> **Experiment 1: The Centroid Alignment Effect**
> Draw a small digit `1` in the far top-right corner with Centroid Alignment **unchecked**. Notice how the network might misclassify it or show high uncertainty. Now **check** Centroid Alignment and watch the prediction instantly lock onto `1` with high confidence.
>
> **Experiment 2: CNN vs. MLP Spatial Robustness**
> Draw a standard digit `8`. Switch the architecture dropdown to **MLP Baseline**. Notice how MLPs are much more sensitive to subtle stroke translations than CNNs because MLPs lack convolutional weight sharing.
>
> **Experiment 3: Creating Ambiguity to Maximize Entropy**
> Draw a loop that looks half like a `0` and half like a `6`, or a `3` that looks like an `8`. Observe the Entropy badge jump above 1.5 bits while both candidate probability bars light up.

---

## 4. Auxiliary Neural Activity Inspector (Pop-Out / Multi-Monitor)

The **Auxiliary Neural Activity Inspector** is a dedicated window designed for deep visual introspection into the inner workings of the convolutional neural network.

You can launch it anytime by:
* Clicking the **🔍 Neural Activity Inspector** button in the top header.
* Clicking the **🔍 Open Neural Activity Inspector** button inside Module 1.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE ARCHITECTURE & LIVE PROPAGATION                                    │
│ [Input] ───> [Conv1] ───> [Pool1] ───> [Conv2] ───> [Pool2] ───> [Dense FC] │
│ (Glowing signal pulse animates along layer connections upon forward pass)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ [🔬 Feature Maps]  |  [🎯 Autograd Saliency Map]  |  [⚡ Dense Heatmap]      │
│                                                                             │
│  Layer 1 Conv2D (16 Filters @ 14×14)     Layer 2 Conv2D (32 Filters @ 7×7)  │
│  [#0] [#1] [#2] [#3] [#4] [#5]...        [#0] [#1] [#2] [#3] [#4] [#5]...   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Tab 1: Convolutional Feature Maps
* **Layer 1 Feature Maps (16 filters @ 14×14):**
  * Shows the feature representations after `Conv2d(1, 16)` followed by `ReLU` and `MaxPool2d(2, 2)`.
  * *Observation:* Notice that Layer 1 filters act as basic edge detectors (horizontal lines, vertical strokes, diagonal contours).
* **Layer 2 Feature Maps (32 filters @ 7×7):**
  * Shows the activations after `Conv2d(16, 32)` followed by `ReLU` and `MaxPool2d(2, 2)`.
  * *Observation:* These filters combine Layer 1 edges into higher-level structural motifs: loops, intersections, corners, and stroke terminations.

### 4.2 Tab 2: Autograd Saliency Map (Explainable AI)
* Displays side-by-side:
  1. **Input Image (28×28):** Your raw or normalized input digit.
  2. **Attribution Gradients ($\frac{\partial y}{\partial x}$):** A high-contrast gradient heatmap showing pixel attribution.
* **How to interpret:**
  * Pixels glowing in bright cyan/yellow/white represent stroke regions that exerted the strongest positive gradient influence on the network's classification.
  * Dark areas were effectively ignored by the receptive fields.
  * This confirms whether the model classifies the digit based on true structural strokes rather than accidental background noise.

### 4.3 Tab 3: Dense FC1 Neuron Activation Heatmap
* An $8 \times 16$ grid displaying the activation levels of all **128 neurons** in the first dense layer (`fc1`).
* Demonstrates **activation sparsity** post-ReLU: notice that only a small, specific subset of neurons fires for any given digit class, illustrating distributed sparse coding.

---

## 5. Module 2: 2D Decision Boundary Playground

Module 2 allows you to explore non-linear classification, neural representations, and decision manifold geometries in a 2D coordinate plane.

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ INTERACTIVE 2D DECISION MANIFOLD│ TRAINING TELEMETRY                        │
│ ┌─────────────────────────────┐ │ [EPOCH: 142]   [LOSS: 0.041] [ACC: 98.5%] │
│ │  • (Cyan: Class 0)          │ ├───────────────────────────────────────────┤
│ │                             │ │ Benchmark Pattern: [Interlocking Spirals▼]│
│ │         Manifold            │ │ Noise Jitter: [═════●═════]  0.10         │
│ │      Probability            │ ├───────────────────────────────────────────┤
│ │        Contour              │ │ Hidden Layers: [2 Hidden Layers ▼]        │
│ │                             │ │ Neurons per Layer: [════●════]  16        │
│ │   • (Purple: Class 1)       │ │ Activation: [ReLU ▼]                      │
│ └─────────────────────────────┘ │ Learning Rate: [════●════]  0.03          │
│ Left Click: Add Cyan (Cls 0)    │                                           │
│ Right Click: Add Purple (Cls 1) │ [▶ Start Training]  [⏭ Step]  [↺ Reset]  │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 5.1 Controls & Setup
* **Canvas Interaction:**
  * **Left Mouse Click:** Adds a new **Class 0 (Cyan)** coordinate point.
  * **Right Mouse Click:** Adds a new **Class 1 (Purple)** coordinate point.
  * You can place custom points while training is running and observe the decision boundary reshape in real time!
* **Playback Controls:**
  * **▶ Start Training / ⏸ Pause:** Starts or pauses the live PyTorch optimization loop running at ~40 FPS.
  * **⏭ Step:** Executes a single gradient descent epoch.
  * **↺ Reset:** Re-initializes model weights using PyTorch default initializers (Kaiming uniform) and resets the training state.
* **Dataset Selection:**
  * **Interlocking Spirals:** The classic non-linear classification benchmark.
  * **Two Moons:** Two interlocking crescent arcs.
  * **Concentric Circles:** Radial symmetry testing inner vs. outer rings.
  * **4-Quadrant XOR:** The canonical non-linearly separable problem.
  * **Gaussian Clusters:** Two distinct Gaussian point clouds.
* **Architecture Configuration:**
  * **Hidden Layers:** Choose between 1, 2, or 3 hidden layers.
  * **Neurons per Layer:** Slide from 4 up to 64 neurons.
  * **Activation Function:** Select **ReLU**, **Tanh**, **Sigmoid**, **GELU**, or **LeakyReLU**.
  * **Learning Rate Slider:** Adjust learning rate between 0.005 and 0.10.

### 5.2 Educational Self-Study Experiments
> [!TIP]
> **Experiment 1: The Single-Layer XOR Failure**
> Select **4-Quadrant XOR**. Set Hidden Layers to **1 Hidden Layer** and Neurons to **4**. Start training. Observe how the boundary struggles with a single linear cut. Now increase to **2 Hidden Layers** with **16 neurons** and watch the network easily solve the XOR problem.
>
> **Experiment 2: Activation Geometry: ReLU vs. Tanh**
> Select **Interlocking Spirals**. Train with **ReLU** and look at the contour: it consists of sharp, polygonal, piecewise linear partitions. Reset and switch the activation to **Tanh** or **GELU**: observe how the boundary becomes a set of smooth, continuous curved equipotential lines.
>
> **Experiment 3: Live Adversarial Points**
> While the model is happily achieving 99% accuracy on the Concentric Circles, right-click to place several Class 1 points deep inside the Class 0 center region. Watch the decision surface bend and carve out an isolated classification island!

---

## 6. Module 3: Autoencoder Latent Space Explorer & Denoising

Module 3 demonstrates unsupervised representation learning, continuous manifold exploration, and image restoration using an Autoencoder with a **2D Latent Bottleneck** $(z_1, z_2)$.

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ 2D LATENT SPACE MANIFOLD (z₁, z₂)│ GENERATIVE DECODER OUTPUT (28×28)         │
│ ┌─────────────────────────────┐ │ ┌───────────────────┐ [Z₁: -1.45]         │
│ │   Digit 0 (Red)             │ │ │                   │                     │
│ │   Digit 1 (Orange)          │ │ │       Decoded     │ [Z₂: +2.10]         │
│ │   ...                       │ │ │        Digit      │                     │
│ │   Digit 7 (Pink)            │ │ │                   │                     │
│ │          ⊕ (Cursor)         │ │ └───────────────────┘                     │
│ └─────────────────────────────┘ ├───────────────────────────────────────────┤
│ Drag crosshair cursor to decode │ CONTINUOUS MANIFOLD INTERPOLATION         │
│ synthetic digits in real time   │ From: [Digit 3 ▼]   To: [Digit 8 ▼]       │
│                                 │ Alpha (t): [════════●════════]  0.50      │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 6.1 Sub-Tab 1: 2D Latent Bottleneck Manifold
* **Latent Space Canvas:** Displays 1,000 test digits projected into 2D Cartesian coordinates $(z_1, z_2)$, color-coded by digit class (0–9).
* **Interactive Crosshair Cursor:**
  * Click or drag anywhere in the 2D plane.
  * The coordinates $(z_1, z_2)$ are immediately passed through the decoder $G(z_1, z_2)$, synthesizing a clean 28×28 digit preview in real time!
* **Manifold Interpolation (Morphing Slider):**
  * Select a source digit (e.g., `Digit 3`) and a target digit (e.g., `Digit 8`).
  * Drag the **Alpha ($t$)** slider from $0.0$ to $1.0$.
  * Watch the digit smoothly morph across the latent space manifold, revealing the topological transitions between characters.

### 6.2 Sub-Tab 2: Denoising & Reconstruction Lab
* **Test Sample Selector:** Pick any digit sample from the offline test collection.
* **Noise Modes:**
  * **Gaussian White Noise:** Injects zero-mean Gaussian perturbation $\mathcal{N}(0, \sigma^2)$.
  * **Salt & Pepper Noise:** Randomly flips pixels to pure black (0.0) or pure white (1.0).
  * **Occlusion Patch Mask:** Blots out an entire rectangular patch, simulating severe occlusions.
* **Visual Comparison Display:**
  1. **Clean Original:** Pristine uncorrupted digit.
  2. **Corrupted Input:** Noisy digit passed to the encoder.
  3. **Neural Reconstruction:** The denoised output produced by the autoencoder.
  4. **Residual Error Map:** Pixel-wise absolute difference $|x_{\text{recon}} - x_{\text{clean}}|$ rendered as a thermal heatmap.
* **Telemetry Metrics:**
  * **INPUT PSNR (dB):** Peak Signal-to-Noise Ratio of the corrupted image.
  * **RECONSTRUCTED PSNR (dB):** PSNR of the autoencoder's output.
  * **PSNR RECOVERY GAIN (dB):** Net fidelity improvement (e.g., $+8.4\text{ dB}$).

### 6.3 Educational Self-Study Experiments
> [!TIP]
> **Experiment 1: Wandering into Latent Voids**
> Drag the crosshair cursor to regions of the 2D plane where no sample points exist (e.g., $(z_1, z_2) = (5, 5)$). Observe what the decoder generates. You will see surreal, hybrid digit shapes, illustrating that the network has learned a continuous mathematical manifold rather than a finite lookup table.
>
> **Experiment 2: Occlusion Inpainting**
> Switch to the Denoising tab. Select **Occlusion Patch Mask** with Corruption Intensity set to **0.50**. Notice how the autoencoder reconstructs the missing part of the loop or stroke based on the remaining global structure.

---

## 7. Module 4: Universal Function Approximator (1D Regression)

Module 4 presents a live visual demonstration of the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991).

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ 1D REGRESSION APPROXIMATION     │ CONVERGENCE TELEMETRY                     │
│ ┌─────────────────────────────┐ │ [STEPS: 250]  [MSE: 0.0012]  [R²: 99.2%]  │
│ │ --- Target Ground Truth     │ ├───────────────────────────────────────────┤
│ │ ━  Neural Output f(x)       │ │ Target Curve: [Harmonic Superposition ▼]  │
│ │ ┈  Hidden Basis Neurons     │ ├───────────────────────────────────────────┤
│ │                             │ │ Hidden Depth: [2 Hidden Layers ▼]         │
│ │             /\     /\       │ │ Neurons per Layer: [════●════]  32        │
│ │   \  /\    /  \   /  \      │ │ Basis Function: [Tanh (Smooth) ▼]         │
│ │    \/  \__/    \_/          │ │ Learning Rate: [════●════]  0.02          │
│ └─────────────────────────────┘ │                                           │
│  Cyan Line: Combined Prediction │ [▶ Start Training]  [⏭ Step]  [↺ Reset]  │
│  Purple Dashed: Basis Neurons   │                                           │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 7.1 Controls & Setup
* **Target Benchmark Curves:**
  * **Sine Wave:** Classic harmonic curve $\sin(2\pi x)$.
  * **Harmonic Superposition:** Multi-frequency wave $\sin(2\pi x) + 0.5\cos(6\pi x)$.
  * **Damped Oscillator:** Decaying sinusoid $e^{-1.2x}\sin(4\pi x)$.
  * **Runge's Function:** Challenging rational curve $\frac{1}{1 + 25x^2}$ known for polynomial oscillation problems.
  * **Discontinuous Square / Step Wave:** Non-smooth step function testing Gibbs phenomenon.
  * **Sawtooth Wave:** Linear ramp with instantaneous reset discontinuities.
* **Network Parameters:**
  * **Hidden Depth:** 1 Layer (pure mathematical theorem form), 2, 3, or 4 hidden layers.
  * **Neurons per Layer:** Slide between 4 and 128 neurons.
  * **Basis Function:**
    * **Tanh:** Smooth sigmoidal basis functions.
    * **ReLU:** Piecewise linear hinges.
    * **GELU:** Smooth Gaussian-gated curves.
    * **SIREN:** Sine periodic activations $\sin(\omega_0 x)$.
    * **Sigmoid:** Classic logistic basis curves.
* **Basis Neuron Visualizer:**
  * Look closely at the background of the canvas! The faint **purple dashed curves** are the individual outputs $w_i \cdot \sigma(v_i x + b_i)$ of each hidden neuron.
  * The solid **cyan line** is their weighted sum plus bias.

### 7.2 Educational Self-Study Experiments
> [!TIP]
> **Experiment 1: Witnessing the Basis Functions**
> Set Hidden Depth to **1 Hidden Layer (Theorem Pure Form)** and Neurons to **6**. Select **ReLU**. Start training on **Sine Wave**. Observe how the cyan prediction is visibly composed of 6 straight segments connected by hinges! Now switch to **Tanh** and see the segments become 6 smooth S-curves.
>
> **Experiment 2: SIREN Periodic Activations**
> Select **Harmonic Superposition**. Choose **SIREN (Periodic Sine)** as the activation function. Notice how quickly the sine basis functions fit harmonic oscillations compared to standard ReLU.
>
> **Experiment 3: The Challenge of Discontinuities**
> Select **Discontinuous Square / Step Wave**. Watch how continuous activations (Tanh/GELU) round off the sharp vertical cliffs, illustrating why continuous neural networks struggle with sharp discontinuities.

---

## 8. Module 5: Real-Time Signal & Audio Waveform Denoising (1D-CNN)

Module 5 demonstrates high-throughput 1D temporal convolutional neural filtering on real-time streaming physical and biological waveforms.

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ DUAL-DOMAIN WAVEFORM & SPECTRAL │ SIGNAL-TO-NOISE RATIO (SNR) TELEMETRY     │
│ ┌─────────────────────────────┐ │ [IN: 7.2 dB]  [OUT: 21.8 dB] [GAIN: +14.6]│
│ │ TIME-DOMAIN OSCILLOSCOPE    │ ├───────────────────────────────────────────┤
│ │ ━ Noisy Input (Red)         │ │ Waveform: [ECG Cardiac Pulse ▼]           │
│ │ --- Clean Ground Truth (Grn)│ │ Carrier Frequency: [════●════]  5.0 Hz    │
│ │ ━ 1D-CNN Denoised (Cyan)    │ ├───────────────────────────────────────────┤
│ ├─────────────────────────────┤ │ Noise Mode: [Gaussian White Noise ▼]      │
│ │ FFT FREQUENCY SPECTRUM      │ │ Noise Amplitude (σ): [════●════]  0.35    │
│ │ █ Red: Noisy Floor          │ │                                           │
│ │ █ Cyan: Denoised Harmonics  │ │ [⏸ Pause Stream]                          │
│ └─────────────────────────────┘ └───────────────────────────────────────────┘
```

### 8.1 Controls & Setup
* **Streaming Toggle (`▶ Resume / ⏸ Pause Stream`):** Starts or freezes the scrolling oscilloscope stream.
* **Waveform Generator:**
  * **Acoustic Harmonics / Chord:** Harmonic chord superposition.
  * **ECG Electrocardiogram Pulse:** Realistic cardiac heartbeat cycle modeling P-wave, Q-depression, R-peak, S-depression, and T-wave.
  * **Linear Chirp Sweep (FM):** Dynamic frequency-swept carrier wave.
  * **Band-Limited Square Wave:** Multi-harmonic alternating plateau wave.
* **Carrier Frequency Slider:** Controls signal fundamental frequency from 1.0 Hz to 15.0 Hz.
* **Noise Distortion Generator:**
  * **Gaussian White Noise:** Broadband Gaussian noise floor.
  * **Impulsive Spikes & Clicks:** Sparse high-amplitude pops and audio clicks.
  * **60Hz Powerline AC Hum:** Realistic electrical interference hum.
* **Noise Amplitude Slider ($\sigma$):** Adjusts corruption intensity from 0.05 up to 0.80.

### 8.2 Telemetry & Dual-Domain Analysis
* **Time-Domain Oscilloscope:**
  * **Red:** Raw corrupted input signal.
  * **Green (Dashed):** Clean ground-truth signal.
  * **Cyan:** Real-time reconstructed output from the 1D-CNN.
* **FFT Frequency Spectrum:**
  * Displays the discrete frequency bins ($0$ to $F_s/2$).
  * The red bars represent the high-frequency noise floor spread across all frequencies.
  * The cyan bars show how the 1D-CNN suppresses the noise floor while preserving the primary harmonic peaks.
* **SNR Telemetry Badges:**
  * **INPUT SNR:** Signal-to-noise ratio before neural filtering.
  * **OUTPUT SNR:** Signal-to-noise ratio of the reconstructed output.
  * **NET RECOVERY GAIN:** Quantitative improvement in decibels (dB).

### 8.3 Educational Self-Study Experiments
> [!TIP]
> **Experiment 1: Preserving Sharp ECG QRS Peaks**
> Select **ECG Electrocardiogram Pulse** and inject **Gaussian White Noise** at $\sigma = 0.40$. Traditional low-pass filters blur out the sharp R-peak when removing noise. Observe how the 1D-CNN preserves the sharp spike while suppressing the noise baseline.
>
> **Experiment 2: Eliminating 60Hz Powerline Hum**
> Switch noise mode to **60Hz Powerline AC Hum**. Check the FFT spectrum at the bottom: observe the spike at the 60Hz bin, and watch how the neural filter attenuates this frequency.

---

## 9. Module 6: Reinforcement Learning Cart-Pole Balancer (DQN)

Module 6 simulates an inverted pendulum mounted on a movable cart governed by Newtonian physics and balanced by a Deep Q-Network (DQN) agent.

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ NEWTONIAN DYNAMICS SIMULATOR    │ EPISODIC PERFORMANCE                      │
│ ┌─────────────────────────────┐ │ [SCORE: 480 steps] [RECORD: 500] [ε: 0.01]│
│ │            O (Tip Ball)     │ ├───────────────────────────────────────────┤
│ │            |                │ │ DEEP Q-NETWORK ACTION VALUES Q(s, a)      │
│ │            | (Pole)         │ │ Q(s, Push Left):  [ 18.42 ]               │
│ │         [Cart]              │ │ Q(s, Push Right): [ 24.81 ] ──> (Chosen)  │
│ │ ───o──────[====]──────o───  │ │ Position: -0.12m   Tilt: +1.4°            │
│ └─────────────────────────────┘ ├───────────────────────────────────────────┤
│ PHYSICAL NUDGE:                 │ Policy Mode: [Pre-trained Master Policy ▼]│
│ [⬅ Push Left]   [Push Right ➡]  │ Planetary Gravity: [════●════] 9.8 m/s²   │
│ [↺ Reset Env]                   │ Pole Length: [════●════] 0.50 m           │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 9.1 Controls & Setup
* **Policy Execution Modes:**
  * **Pre-trained Master Policy:** Fully trained DQN model with greedy evaluation ($\epsilon = 0.01$) that balances the pole reliably.
  * **Live Training from Scratch:** Starts with random exploration ($\epsilon = 1.0$), filling the experience replay buffer and updating Q-network weights live in the background.
  * **Manual Human Control:** Lets you test how challenging inverted pendulum balancing is without a neural controller!
* **Interactive Physical Nudges (`⬅ Push Left` / `Push Right ➡`):**
  * Applies an instantaneous impulsive force kick ($\pm 16.0\text{ N}$) to test the agent's dynamic recovery capabilities.
* **Planetary Physics Sliders:**
  * **Planetary Gravity:** Test balancing on the **Moon** ($1.62\text{ m/s}^2$), **Earth** ($9.8\text{ m/s}^2$), or **Jupiter** ($24.8\text{ m/s}^2$).
  * **Pole Length:** Adjust between $0.20\text{ m}$ (fast, difficult balancing) and $1.20\text{ m}$ (slow, sluggish dynamics).

### 9.2 Reading the Telemetry & Q-Values
* **Q(s, Push Left) & Q(s, Push Right):**
  * Shows the expected discounted cumulative future rewards for taking each discrete action from the current physical state.
  * The action with the higher Q-value is selected by the greedy policy.
* **State HUD:**
  * **Position ($x$):** Distance in meters from track center (failure limit: $\pm 2.4\text{ m}$).
  * **Tilt ($\theta$):** Angular deviation in degrees from vertical upright (failure limit: $\pm 12^\circ$).
* **Color-Coded Pole Indicator:**
  * **Green:** Safe vertical balance ($|\theta| < 5^\circ$).
  * **Amber:** Moderate tilt ($5^\circ \le |\theta| < 10^\circ$).
  * **Red:** Critical tipping hazard ($|\theta| \ge 10^\circ$).

### 9.3 Educational Self-Study Experiments
> [!TIP]
> **Experiment 1: Interactive Disturbance Recovery**
> Run the **Pre-trained Master Policy**. While the cart is centered, rapidly click **⬅ Push Left** two or three times. Observe how the Q-value gauges swap values immediately, commanding hard right pushes to slide the cart under the tipping pole to catch it!
>
> **Experiment 2: Changing Planetary Gravity**
> Slide Planetary Gravity to **Jupiter (24.8 m/s²)**. Notice that the pendulum falls much faster, requiring rapid, decisive control corrections. Then slide to **Moon (1.62 m/s²)** and notice the slow, floaty swing.
>
> **Experiment 3: Watch Live Reinforcement Learning from Scratch**
> Switch Policy Mode to **Live Training from Scratch**. Watch the first 10 episodes fail within 10–20 steps as the cart explores randomly. Around episode 30–50, observe how the score begins climbing as the experience replay buffer accumulates successful trajectories.

---

## 10. Troubleshooting & Performance Tips

### 10.1 UI Responsiveness & Threading
* NeuroNetWorks automatically configures thread counts via `core.device.configure_torch_threads()`, preserving at least one core for the PyQt6 GUI event loop to ensure smooth 60 FPS rendering.
* If you experience input lag on older dual-core systems, set the environment variable:
  ```bash
  OMP_NUM_THREADS=2 ./launch.sh
  ```

### 10.2 Retraining Offline Weights
If any `.pth` checkpoint in `models_weights/` is deleted or you wish to retrain all models from scratch:
```bash
.venv/bin/python -m core.trainer
```
This script will retrain the MNIST CNN, the 2D Autoencoder, the 1D Signal Denoiser, and the Cart-Pole DQN agent in ~45 seconds.

### 10.3 Offscreen / Headless Verification
To verify the application in headless continuous integration or without a physical display:
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python app.py --test
```
You should see: `ALL NEURONETWORKS TEST CHECKS PASSED PERFECTLY!`.

---

*Enjoy your learning journey with **NeuroNetWorks**! For complete mathematical proofs, neural network architectures, and training loss equations, consult the companion [TECHNICAL_DOCUMENTATION.md](file:///home/cyrus/Projects/Python_AI/NeuroNetWorks/TECHNICAL_DOCUMENTATION.md).*
