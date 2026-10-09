# 🔬 NeuroNetWorks: Technical Architecture, Mathematical Foundations & Deep Learning Reference

Welcome to the **NeuroNetWorks Technical Documentation**. This document is a comprehensive, rigorous educational reference and architectural manual detailing the mathematics, neural network designs, tensor flows, loss functions, optimization algorithms, and physics simulations powering the **NeuroNetWorks** didactic suite.

---

## 📑 Table of Contents
1. [Core Computational Philosophy & Architecture](#1-core-computational-philosophy--architecture)
2. [Hardware Virtualization & Device Optimization (`core/device.py`)](#2-hardware-virtualization--device-optimization-coredevicepy)
3. [Module 1: Deep Convolutional Vision & Explainable AI (`core/mnist_models.py`)](#3-module-1-deep-convolutional-vision--explainable-ai-coremnist_modelspy)
4. [Auxiliary Neural Inspector Architecture (`ui/auxiliary_window.py`)](#4-auxiliary-neural-inspector-architecture-uiauxiliary_windowpy)
5. [Module 2: 2D Non-Linear Classification & Manifolds (`core/boundary_models.py`)](#5-module-2-2d-non-linear-classification--manifolds-coreboundary_modelspy)
6. [Module 3: Unsupervised Learning & Generative Autoencoders (`core/autoencoder_models.py`)](#6-module-3-unsupervised-learning--generative-autoencoders-coreautoencoder_modelspy)
7. [Module 4: Universal Function Approximator & Basis Decomposition (`core/approximator_models.py`)](#7-module-4-universal-function-approximator--basis-decomposition-coreapproximator_modelspy)
8. [Module 5: 1D Temporal Convolutions & Signal Denoising (`core/signal_models.py`)](#8-module-5-1d-temporal-convolutions--signal-denoising-coresignal_modelspy)
9. [Module 6: Deep Reinforcement Learning (DQN) & Newtonian Dynamics (`core/cartpole_models.py`)](#9-module-6-deep-reinforcement-learning-dqn--newtonian-dynamics-corecartpole_modelspy)
10. [Module 7: Vision Transformer (ViT) & Multi-Head Attention (`core/transformer_models.py`)](#10-module-7-vision-transformer-vit--multi-head-attention-coretransformer_modelspy)
11. [Module 8: Adversarial Robustness & Gradient Perturbations (`core/adversarial_models.py`)](#11-module-8-adversarial-robustness--gradient-perturbations-coreadversarial_modelspy)
12. [Module 9: 2D Generative Diffusion & Flow Matching (`core/diffusion_models.py`)](#12-module-9-2d-generative-diffusion--flow-matching-corediffusion_modelspy)
13. [Offline Training Pipeline & Serialization (`core/trainer.py`)](#13-offline-training-pipeline--serialization-coretrainerpy)
14. [Comprehensive Machine Learning & Deep Learning Glossary](#14-comprehensive-machine-learning--deep-learning-glossary)

---

## 1. Core Computational Philosophy & Architecture

NeuroNetWorks is built upon the principle of **didactic transparency**: every forward pass, activation tensor, gradient backpropagation step, and loss calculation is made inspectable in real time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NEURONETWORKS CORE                               │
├──────────────────────────────────────┬──────────────────────────────────────┤
│          PyTorch Computation         │             NumPy & SciPy            │
│  • Automatic Differentiation (AD)    │  • Discrete Fourier Transforms (FFT) │
│  • Convolutional / Dense Tensors     │  • Spatial Image Moments (Centroid)  │
│  • Experience Replay Mini-Batches    │  • Numerical Integration (Symplectic)│
└──────────────────────────────────────┴──────────────────────────────────────┘
                                  │ Tensors & Arrays
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PYQT6 CYBER-LAB GUI                               │
│  • Hardware Accelerated Canvas (QPainter 60 FPS)                            │
│  • Asynchronous Signal / Slot Pipeline                                      │
│  • Multi-Monitor Auxiliary Activation Inspector                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Foundations
Across all modules, deep neural networks are parameterized continuous function approximators $f_\theta: \mathcal{X} \to \mathcal{Y}$ where $\theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}_{l=1}^L$ denotes the learnable parameters (weights and biases).

The parameters are updated using **Gradient Descent** or variants thereof (such as Adam):
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(f_\theta(x), y)$$
where $\eta$ is the learning rate and $\mathcal{L}$ is the scalar objective loss function.

---

## 2. Hardware Virtualization & Device Optimization (`core/device.py`)

Interactive desktop GUI applications require low latency and high frame rates (30–60 FPS). If PyTorch consumes all available CPU cores during tensor operations, the PyQt6 GUI event loop suffers from thread starvation, resulting in UI stuttering.

### 2.1 Hardware Probing Pipeline
[`core/device.py`](file:///home/cyrus/Projects/Python_AI/NeuroNetWorks/core/device.py) dynamically probes the available compute hardware:
1. **CUDA Accelerator:** Detected via `torch.cuda.is_available()`. Yields `cuda:0`.
2. **Apple Silicon Metal Performance Shaders (MPS):** Detected via `torch.backends.mps.is_available()`. Yields `mps`.
3. **Optimized CPU Execution:** Falls back to CPU with tailored thread pool configuration.

### 2.2 Thread Contention Mitigation
```python
def configure_torch_threads():
    cpu_count = os.cpu_count() or 4
    # Reserve at least 1 core for PyQt6 GUI event processing
    optimal_threads = max(1, min(cpu_count - 1, 8))
    torch.set_num_threads(optimal_threads)
```
By reserving at least one logical thread for GUI rendering and limiting maximum compute threads to 8, PyTorch achieves maximum inference throughput without blocking the Qt event queue.

---

## 3. Module 1: Deep Convolutional Vision & Explainable AI (`core/mnist_models.py`)

Module 1 features two distinct classification architectures: `MNISTCNN` and `MNISTMLP`.

```
INPUT (1 × 28 × 28)
  │
  ▼
[Conv2D (1 → 16, k=3, p=1)] ──> [ReLU] ──> [MaxPool2D (2×2, s=2)]  (16 × 14 × 14)
  │
  ▼
[Conv2D (16 → 32, k=3, p=1)] ──> [ReLU] ──> [MaxPool2D (2×2, s=2)] (32 × 7 × 7)
  │
  ▼
[Flatten] (1568 units)
  │
  ▼
[Linear (1568 → 128)] ──> [ReLU] ──> [Dropout (p=0.2)]
  │
  ▼
[Linear (128 → 10)] ──> [Softmax] ──> Class Probabilities p ∈ [0, 1]^10
```

### 3.1 2D Convolution Mathematics
For an input tensor $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ and a collection of $C_{\text{out}}$ filters with kernel $\mathbf{K} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$ and bias $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$, the 2D cross-correlation operation for output channel $c$, position $(i, j)$ is defined as:
$$\mathbf{Y}_{c, i, j} = \mathbf{b}_c + \sum_{k=0}^{C_{\text{in}}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} \mathbf{X}_{k, i+m, j+n} \cdot \mathbf{K}_{c, k, m, n}$$

* **Kernel Size ($3 \times 3$):** Captures local spatial motifs (edges, corners, stroke intersections).
* **Padding ($p=1$):** Maintains spatial dimensions prior to pooling: $H_{\text{out}} = \lfloor \frac{H + 2p - K}{s} \rfloor + 1 = \lfloor \frac{28 + 2(1) - 3}{1} \rfloor + 1 = 28$.
* **Receptive Field Algebra:**
  * Post-Conv1: Each neuron covers a $3 \times 3$ receptive field of the input.
  * Post-Pool1 ($2 \times 2$, stride 2): Each neuron covers a $4 \times 4$ receptive field.
  * Post-Conv2: Receptive field expands to $8 \times 8$.
  * Post-Pool2 ($2 \times 2$, stride 2): Receptive field expands to $14 \times 14$, spanning half the digit image!

### 3.2 Regularization: Inverted Dropout
Layer `fc1` uses Dropout with probability $p = 0.2$:
$$\tilde{\mathbf{h}} = \frac{1}{1 - p} \cdot (\mathbf{h} \odot \mathbf{m}), \quad \mathbf{m}_i \sim \text{Bernoulli}(1 - p)$$
The scaling factor $\frac{1}{1 - p} = \frac{1}{0.8} = 1.25$ ensures that the expected activation magnitude remains invariant between training and evaluation phases without requiring post-training weight rescaling:
$$\mathbb{E}[\tilde{\mathbf{h}}] = \frac{1}{1 - p} \cdot \mathbb{E}[\mathbf{m}] \odot \mathbf{h} = \frac{1}{1 - p} \cdot (1 - p) \cdot \mathbf{h} = \mathbf{h}$$

### 3.3 Classification Objective: Categorical Cross-Entropy
Given logits $\mathbf{z} \in \mathbb{R}^{10}$, posterior class probabilities are computed via the **Softmax** function:
$$p_i = \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=0}^9 e^{z_j}}$$
The training loss is the Negative Log-Likelihood (Categorical Cross-Entropy):
$$\mathcal{L}_{\text{CE}}(p, y) = -\sum_{i=0}^9 y_i \log(p_i) = -\log(p_{\text{target}})$$
The analytical gradient with respect to logit $z_i$ simplifies to:
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = p_i - y_i$$

### 3.4 Preprocessing Math: Centroid Normalization
In real MNIST digits, characters are centered within a $20 \times 20$ box and placed in a $28 \times 28$ field based on their **Center of Mass** $(\bar{x}, \bar{y})$.

[`ui/modules/mnist_view.py`](file:///home/cyrus/Projects/Python_AI/NeuroNetWorks/ui/modules/mnist_view.py) implements this standard normalization:
$$\bar{x} = \frac{\sum_{i, j} j \cdot \mathbf{I}(i, j)}{\sum_{i, j} \mathbf{I}(i, j)}, \quad \bar{y} = \frac{\sum_{i, j} i \cdot \mathbf{I}(i, j)}{\sum_{i, j} \mathbf{I}(i, j)}$$
where $\mathbf{I}(i, j) \in [0, 1]$ is the pixel intensity.

The required integer shift to translate the digit to the geometric center $(13.5, 13.5)$ is:
$$\Delta x = \text{round}(13.5 - \bar{x}), \quad \Delta y = \text{round}(13.5 - \bar{y})$$
The canvas applies a 2D circular roll (`np.roll`) followed by border zero-clipping to eliminate roll-over edge artifacts.

### 3.5 Explainable AI: Vanilla Backpropagation Saliency Maps
To understand *why* the CNN made a specific prediction, the application computes pixel attribution gradients:
$$\mathcal{S}(x, y) = \left| \frac{\partial z_c}{\partial \mathbf{X}_{x, y}} \right|$$
where $z_c$ is the unnormalized logit score for winning class $c = \text{argmax}(\mathbf{p})$.

**PyTorch Implementation:**
```python
def compute_saliency(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
    self.eval()
    x_clone = x.clone().detach().requires_grad_(True)
    logits = self(x_clone)
    if target_class is None:
        target_class = int(logits.argmax(dim=-1)[0].item())
    
    score = logits[0, target_class]
    score.backward()
    
    saliency = x_clone.grad.data[0, 0].abs().cpu().numpy()
    s_min, s_max = saliency.min(), saliency.max()
    return (saliency - s_min) / (s_max - s_min + 1e-8)
```
This produces a pixel-wise attribution map highlighting the stroke features that positively influenced the prediction score.

### 3.6 Epistemic Uncertainty: Shannon Information Entropy
To measure model uncertainty and detect ambiguous or adversarial strokes, the system computes the Shannon Entropy $H(p)$ in bits:
$$H(p) = -\sum_{i=0}^9 p_i \log_2(p_i)$$
* **Minimum Entropy ($H = 0.0\text{ bits}$):** Absolute certainty ($p_{\text{winner}} = 1.0$).
* **Maximum Entropy ($H = \log_2(10) \approx 3.322\text{ bits}$):** Complete confusion / uniform distribution ($p_i = 0.1 \ \forall i$).

---

## 4. Auxiliary Neural Inspector Architecture (`ui/auxiliary_window.py`)

The Auxiliary Inspector operates as an independent window that receives telemetry signals emitted by `MNISTView`.

```
MNISTView.run_inference()
  │
  ├──> forward_with_activations(x)  (Captures C1, P1, C2, P2, FC1, Logits)
  ├──> compute_saliency(x)          (Autograd Backprop on input)
  │
  └──> emit telemetry_ready(dict)
         │
         ▼
AuxiliaryNeuralWindow.update_telemetry(data)
  ├──> ArchitectureFlowCanvas.trigger_pulse()  (Animated 60 FPS pulse)
  ├──> QGridLayout (Conv1): 16 filters rendered via custom colormaps
  ├──> QGridLayout (Conv2): 32 filters rendered via custom colormaps
  ├──> Saliency Heatmap (Hot colormap conversion)
  └──> FC1 Heatmap: 128 units reshaped into 8 × 16 activation matrix
```

### Telemetry Tap Mechanism
Rather than relying on PyTorch forward hooks (which introduce memory overhead and garbage collection complexity during interactive rendering), `MNISTCNN` exposes a dedicated telemetry method:
```python
def forward_with_activations(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
    self.eval()
    with torch.no_grad():
        c1 = self.conv1(x)
        r1 = self.relu1(c1)
        p1 = self.pool1(r1)
        c2 = self.conv2(p1)
        r2 = self.relu2(c2)
        p2 = self.pool2(r2)
        flat = p2.reshape(p2.size(0), -1)
        f1 = self.fc1(flat)
        r3 = self.relu3(f1)
        f2 = self.fc2(r3)
        probs = F.softmax(f2, dim=-1)
    return { ... }
```
This zero-copy, non-hooking pipeline delivers activation tensors to Qt widgets without risking GPU memory fragmentation.

---

## 5. Module 2: 2D Non-Linear Classification & Manifolds (`core/boundary_models.py`)

Module 2 provides a continuous 2D classification sandbox powered by `DecisionBoundaryMLP`.

```
Input [x₁, x₂] ∈ ℝ²
       │
       ▼
[Linear (2 → H)] ──> [Activation σ(·)]
       │
      ... (L - 1 additional hidden layers)
       │
       ▼
[Linear (H → 2)] ──> [Softmax] ──> P(y = 1 | x₁, x₂)
```

### 5.1 Non-Linear Activation Geometries
The network supports 5 distinct activation functions:

| Activation | Mathematical Formulation | Output Range | Derivative $\sigma'(z)$ | Geometric Boundary Bias |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | $\mathbf{1}_{z > 0}$ | Piecewise linear polyhedra / sharp polygons |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | $1 - \tanh^2(z)$ | Smooth, flowing equipotential manifolds |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $(0, 1)$ | $\sigma(z)(1 - \sigma(z))$ | Smooth curves; prone to vanishing gradients |
| **GELU** | $z \cdot \Phi(z) = z \cdot P(X \le z)$ | $[-0.17, \infty)$ | Continuous | Smooth curvature with subtle stochastic gating |
| **LeakyReLU** | $\max(\alpha z, z), \ \alpha = 0.01$ | $(-\infty, \infty)$ | $1$ if $z > 0$ else $\alpha$ | Prevents "dead neuron" collapse on negative domain |

### 5.2 Synthetic Benchmark Dataset Generators

#### 1. Dual Archimedean Spirals
Parameterizes two interwoven polar curves separated by phase $\pi$:
$$\theta = \sqrt{u} \cdot 2.5\pi, \quad r = \frac{2.0\theta}{2.5\pi}, \quad u \sim \mathcal{U}(0, 1)$$
* **Class 0:** $x_0 = r \cos(\theta) + \epsilon, \quad y_0 = r \sin(\theta) + \epsilon$
* **Class 1:** $x_1 = -r \cos(\theta) + \epsilon, \quad y_1 = -r \sin(\theta) + \epsilon$

#### 2. Concentric Circles
Generates radial clusters testing radius-dependent classification:
* **Class 0 (Inner):** $r_0 \sim \mathcal{N}(0.6, \sigma^2), \quad \theta \sim \mathcal{U}(0, 2\pi)$
* **Class 1 (Outer):** $r_1 \sim \mathcal{N}(1.4, \sigma^2), \quad \theta \sim \mathcal{U}(0, 2\pi)$

#### 3. 4-Quadrant XOR
Tests non-linear separability across axes:
$$\mathbf{x} \sim \mathcal{U}(-1.5, 1.5)^2, \quad y = \mathbf{1}_{(x_1 > 0) \oplus (x_2 > 0)}$$

### 5.3 Optimization Dynamics: Adam with Weight Decay
The network is optimized using Adam with decoupled $L_2$ penalty:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_t$$
where default parameters are $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}, \lambda = 10^{-4}$.

### 5.4 60 FPS Continuous Manifold Rendering
To visualize the decision surface without lagging the UI:
1. A static evaluation grid of $50 \times 50 = 2,500$ points is generated over $[-2.2, 2.2]^2$.
2. Every 2 epochs, the grid coordinates are evaluated in a single vectorized forward pass:
   ```python
   probs = model.predict_proba(grid_coords) # [2500] float array
   ```
3. The probabilities are mapped into RGB color space:
   * $p < 0.5$: Cyan intensity scaled by $\frac{0.5 - p}{0.5}$.
   * $p \ge 0.5$: Purple/Magenta intensity scaled by $\frac{p - 0.5}{0.5}$.
4. A 32-bit `QImage` is generated and upscaled with bilinear interpolation to canvas dimensions ($420 \times 420$), achieving 60 FPS rendering.

---

## 6. Module 3: Unsupervised Learning & Generative Autoencoders (`core/autoencoder_models.py`)

Module 3 focuses on unsupervised representation learning, manifold discovery, and image restoration using `MNIST2DAutoencoder`.

```
                  ENCODER                              DECODER
Image [28×28] ──> [Linear (784 → 256)]            [Linear (2 → 64)]
                     │                                   │
                  [BatchNorm1d + LeakyReLU]        [BatchNorm1d + LeakyReLU]
                     │                                   │
                  [Linear (256 → 64)]             [Linear (64 → 256)]
                     │                                   │
                  [BatchNorm1d + LeakyReLU]        [BatchNorm1d + LeakyReLU]
                     │                                   │
                  [Linear (64 → 2)]               [Linear (256 → 784)]
                     │                                   │
                     ▼                                   ▼
             Latent (z₁, z₂)                      Reconstructed [28×28]
              (2D Bottleneck)                      (Sigmoid Output)
```

### 6.1 The 2D Information Bottleneck
Compressing an image from $784$ dimensions to $2$ continuous latent coordinates $(z_1, z_2)$ represents a **99.74% dimensional reduction**:
$$\text{Compression Ratio} = \frac{784}{2} = 392:1$$
Under this constraint, the autoencoder cannot memorize individual pixels. Instead, it must discover a continuous 2D coordinate system capturing the primary geometric variations of handwritten digits (loops, vertical strokes, diagonal slants, aspect ratios).

### 6.2 1D Batch Normalization
Each intermediate layer applies 1D Batch Normalization:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
where $\mu_B = \frac{1}{m}\sum x_i$ and $\sigma_B^2 = \frac{1}{m}\sum(x_i - \mu_B)^2$.
* Keeps activations zero-centered and unit-variance throughout deep layers.
* Prevents internal covariate shift and speeds up convergence during training.

### 6.3 Generative Manifold Traversal & Linear Interpolation
Because the latent space is continuous, decoding any point $z \in \mathbb{R}^2$ yields a valid synthetic digit image:
$$\mathbf{X}_{\text{synthetic}} = g_\phi(z_1, z_2)$$

To smoothly morph between two digit classes $A$ and $B$, the system computes their class centroids $\boldsymbol{\mu}_A, \boldsymbol{\mu}_B \in \mathbb{R}^2$:
$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i: y_i = k} \mathbf{z}_i$$
and performs linear interpolation parameterized by $t \in [0, 1]$:
$$\mathbf{z}(t) = (1 - t) \boldsymbol{\mu}_A + t \boldsymbol{\mu}_B$$
$$\mathbf{X}(t) = g_\phi(\mathbf{z}(t))$$

### 6.4 Denoising Autoencoder (DAE) Theory
Given clean data $x \sim p_{\text{data}}(x)$, corruption is injected via a stochastic process $\tilde{x} \sim q(\tilde{x} \mid x)$. The autoencoder is trained to minimize:
$$\mathcal{L}_{\text{DAE}}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{data}}, \tilde{x} \sim q(\tilde{x}|x)} \left[ \| x - g_\phi(f_\theta(\tilde{x})) \|_2^2 \right]$$

**Geometric Intuition (The Manifold Projection Theorem):**
High-dimensional handwritten digit space occupies a low-dimensional manifold $\mathcal{M} \subset \mathbb{R}^{784}$. Injecting noise moves the vector orthogonal to $\mathcal{M}$ into empty state space. The Denoising Autoencoder learns an orthogonal projection operator that maps corrupted off-manifold vectors back onto the true data manifold $\mathcal{M}$.

### 6.5 Noise Generation Models
1. **Additive White Gaussian Noise (AWGN):**
   $$\tilde{\mathbf{X}} = \text{clip}(\mathbf{X} + \mathcal{N}(0, \sigma^2), 0.0, 1.0)$$
2. **Salt & Pepper Impulse Noise:**
   $$\tilde{\mathbf{X}}_{i, j} = \begin{cases} 1.0 & \text{with probability } p/2 \\ 0.0 & \text{with probability } p/2 \\ \mathbf{X}_{i, j} & \text{with probability } 1 - p \end{cases}$$
3. **Occlusion Patch Masking:**
   Sets a bounding box of size $S \times S$ at random coordinates $(y, x)$ to zero:
   $$\tilde{\mathbf{X}}_{y:y+S, x:x+S} = 0.0$$

### 6.6 Image Fidelity Metrics
* **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W (\mathbf{X}_{i, j} - \hat{\mathbf{X}}_{i, j})^2$$
* **Peak Signal-to-Noise Ratio (PSNR in dB):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 10 \cdot \log_{10}\left( \frac{1.0}{\text{MSE}} \right) = -10 \cdot \log_{10}(\text{MSE})$$
* **PSNR Recovery Gain:**
  $$\Delta \text{PSNR} = \text{PSNR}_{\text{output}} - \text{PSNR}_{\text{input}}$$

---

## 7. Module 4: Universal Function Approximator & Basis Decomposition (`core/approximator_models.py`)

Module 4 provides a live visual demonstration of the **Universal Approximation Theorem** for 1D continuous curves.

```
x ∈ [-1, 1] ──> [Linear (1 → H)] ──> [σ(·)] ──> ... ──> [Linear (H → 1)] ──> f̂(x)
```

### 7.1 The Universal Approximation Theorem (Hornik, 1991)
Let $\sigma$ be any non-constant, bounded, and continuous activation function. Let $I_m$ denote the $m$-dimensional unit hypercube $[0, 1]^m$. Then for any continuous function $f \in C(I_m)$ and any target tolerance $\epsilon > 0$, there exist integers $H$, weight vectors $\mathbf{v}_i \in \mathbb{R}^m$, scalar weights $w_i \in \mathbb{R}$, and biases $b_i, b_0 \in \mathbb{R}$ such that:
$$\left| f(\mathbf{x}) - \sum_{i=1}^H w_i \sigma(\mathbf{v}_i^T \mathbf{x} + b_i) - b_0 \right| < \epsilon \quad \forall \mathbf{x} \in I_m$$

### 7.2 Hidden Neuron Basis Decomposition
In a network with a single hidden layer and linear readout:
$$\hat{f}(x) = \sum_{i=1}^H W_i \cdot h_i(x) + b_0, \quad \text{where } h_i(x) = \sigma(v_i x + b_i)$$
Each product $B_i(x) = W_i \cdot h_i(x)$ represents an **individual scaled basis curve**.

[`core/approximator_models.py`](file:///home/cyrus/Projects/Python_AI/NeuroNetWorks/core/approximator_models.py) extracts these curves:
```python
def get_basis_neuron_outputs(self, x_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    self.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(-1)
        h = self.hidden_net(x_t)          # [N, H] activations
        W = self.out_linear.weight[0]      # [H] output weights
        b = self.out_linear.bias.item()
        basis = (h * W.unsqueeze(0)).t().cpu().numpy()  # [H, N] individual curves
        total = self.out_linear(h).squeeze(-1).cpu().numpy()
    return basis, total
```
The UI displays these individual basis curves as dashed purple lines beneath the solid cyan prediction curve, showing how non-linear curves are formed through the superposition of simpler functional elements.

### 7.3 SIREN: Sinusoidal Representation Networks
When selected, the network uses the periodic activation:
$$\sigma_{\text{SIREN}}(x) = \sin(\omega_0 x), \quad \omega_0 = 3.0$$
* **Derivative Property:** $\frac{d}{dx}\sin(\omega_0 x) = \omega_0 \cos(\omega_0 x)$, which is itself a sinusoid shifted by $\pi/2$.
* Unlike ReLU (whose derivatives are step functions and second derivatives are zero), SIREN retains rich, non-zero higher-order derivatives, making it well-suited for fitting complex periodic waveforms and solving differential equations (Sitzmann et al., 2020).

### 7.4 Target Benchmark Functions
1. **Sine Wave:** $f(x) = \sin(2\pi x)$
2. **Harmonic Superposition:** $f(x) = 0.7\sin(2\pi x) + 0.3\cos(6\pi x)$
3. **Damped Oscillator:** $f(x) = e^{-1.2(x+1)}\sin(4\pi x)$
4. **Runge's Function:** $f(x) = \frac{1}{1 + 25x^2}$
5. **Discontinuous Square Wave:** $f(x) = 0.75 \cdot \text{sgn}(\sin(2\pi x))$
6. **Sawtooth Wave:** $f(x) = 2(x - \lfloor x + 0.5 \rfloor)$

### 7.5 Regression Performance Metrics
* **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
* **Coefficient of Determination ($R^2$ Score):**
  $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  * $R^2 = 1.0$ (100%): Perfect reconstruction.
  * $R^2 = 0.0$: Predictor performs no better than the sample mean $\bar{y}$.

---

## 8. Module 5: 1D Temporal Convolutions & Signal Denoising (`core/signal_models.py`)

Module 5 demonstrates high-throughput 1D temporal convolutional neural filtering on real-time streaming physical and biological waveforms using `SignalDenoising1DCNN`.

```
Noisy Signal x ∈ ℝ^[1 × 512]
        │
        ▼  ENCODER
[Conv1d (1 → 32, k=9, p=4)] ──> [BatchNorm1d + LeakyReLU(0.2)]
        │
[Conv1d (32 → 64, k=7, p=3)] ──> [BatchNorm1d + LeakyReLU(0.2)]
        │
        ▼  DECODER
[Conv1d (64 → 32, k=7, p=3)] ──> [BatchNorm1d + LeakyReLU(0.2)]
        │
[Conv1d (32 → 16, k=5, p=2)] ──> [LeakyReLU(0.2)]
        │
[Conv1d (16 → 1, k=5, p=2)]
        │
        ▼
Denoised Output ŷ ∈ ℝ^[1 × 512]
```

### 8.1 1D Convolution vs. Classic Digital Signal Processing (DSP)
Traditional FIR (Finite Impulse Response) and IIR (Infinite Impulse Response) bandpass filters operate under a linearity and stationarity assumption:
$$y[n] = \sum_{k=0}^M b_k x[n-k] - \sum_{k=1}^N a_k y[n-k]$$
When noise shares frequency bands with critical transient signal components (such as an ECG R-peak), linear bandpass filters either blur the transient peak or let wideband noise through.

**The 1D-CNN Advantage:**
* Non-linear feature extraction: learns combinations of morphological shape, zero-crossing curvature, and harmonic envelope.
* Zero-phase temporal processing: centered padding preserves phase alignment without introducing time delays.

### 8.2 Physical & Biological Waveform Synthesis
1. **Electrocardiogram (ECG) Cardiac Pulse:**
   Modeled as a superposition of 5 Gaussian pulses over normalized cardiac phase $\phi = (t \bmod T) / T$:
   $$s_{\text{ECG}}(\phi) = \sum_{k \in \{P, Q, R, S, T\}} A_k \exp\left( -\frac{(\phi - \mu_k)^2}{2\sigma_k^2} \right)$$
   * **P-wave:** $A=0.15, \mu=0.20, \sigma=0.02$ (Atrial depolarization)
   * **Q-wave:** $A=-0.15, \mu=0.35, \sigma=0.005$ (Septal depolarization)
   * **R-peak:** $A=1.00, \mu=0.38, \sigma=0.007$ (Ventricular depolarization spike)
   * **S-wave:** $A=-0.25, \mu=0.41, \sigma=0.007$ (Ventricular depolarization finish)
   * **T-wave:** $A=0.30, \mu=0.65, \sigma=0.04$ (Ventricular repolarization)
2. **Linear Chirp Sweep (Frequency Modulation):**
   $$s(t) = \sin\left( 2\pi \left( f_0 t + \frac{k}{2} t^2 \right) \right), \quad k = \frac{f_1 - f_0}{T}$$
3. **Acoustic Harmonics:** Superposition of fundamental and overtone integer harmonics:
   $$s(t) = \frac{1}{1.75}\left[ \sin(2\pi f t) + 0.5\sin(4\pi f t + 0.3) + 0.25\sin(6\pi f t + 0.7) \right]$$

### 8.3 Noise Distortion Models
* **Wideband Gaussian Noise:** $\epsilon[n] \sim \mathcal{N}(0, \sigma^2)$
* **Impulsive Spikes & Clicks:** Sparse Poisson-sampled impulses of magnitude $\pm 3\sigma$.
* **60Hz AC Powerline Hum:** $\epsilon[n] = 1.2\sigma \sin(2\pi \cdot 60 \cdot t)$.

### 8.4 Signal-to-Noise Ratio (SNR) Metrics
Given clean signal $s$, noisy signal $x$, and neural output $\hat{s}$:
$$P_{\text{signal}} = \frac{1}{N}\sum s_i^2, \quad P_{\text{noise, in}} = \frac{1}{N}\sum (x_i - s_i)^2, \quad P_{\text{noise, out}} = \frac{1}{N}\sum (\hat{s}_i - s_i)^2$$
$$\text{SNR}_{\text{in}} = 10 \log_{10}\left( \frac{P_{\text{signal}}}{P_{\text{noise, in}}} \right) \text{ dB}, \quad \text{SNR}_{\text{out}} = 10 \log_{10}\left( \frac{P_{\text{signal}}}{P_{\text{noise, out}}} \right) \text{ dB}$$
$$\text{Gain} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}} \text{ dB}$$

### 8.5 Discrete Fourier Transform & Spectral Analysis
The oscilloscope bottom view computes real FFT magnitude spectra:
$$\mathcal{X}[k] = \sum_{n=0}^{N-1} x[n] e^{-i \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, \frac{N}{2}$$
$$|\mathcal{X}[k]| = \sqrt{\text{Re}(\mathcal{X}[k])^2 + \text{Im}(\mathcal{X}[k])^2}$$
Visualizes how the 1D-CNN suppresses high-frequency noise components while preserving discrete harmonic peaks.

---

## 9. Module 6: Deep Reinforcement Learning (DQN) & Newtonian Dynamics (`core/cartpole_models.py`)

Module 6 simulates an inverted pendulum on a cart governed by Newtonian mechanics and stabilized by a Deep Q-Network (DQN) agent.

```
STATE s = [x, ẋ, θ, θ̇] ∈ ℝ⁴
           │
           ▼
     [Linear (4 → 64)] ──> [ReLU]
           │
     [Linear (64 → 64)] ──> [ReLU]
           │
     [Linear (64 → 2)]
           │
           ▼
Q-Values: [Q(s, a=Push Left),  Q(s, a=Push Right)]
```

### 9.1 Newtonian Inverted Pendulum Dynamics
Let $M$ be cart mass, $m$ pole mass, $l$ half-pole length, $g$ gravitational acceleration, and $F$ applied horizontal force.

The equations of motion derived from the Euler-Lagrange equations are:
$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \left( \frac{F + m l \dot{\theta}^2 \sin\theta}{M + m} \right)}{l \left( \frac{4}{3} - \frac{m \cos^2\theta}{M + m} \right)}$$
$$\ddot{x} = \frac{F + m l \left( \dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta \right)}{M + m}$$

### 9.2 Symplectic Euler-Cromer Numerical Integration
Standard forward Euler integration can cause artificial energy growth in oscillating systems. To preserve energy stability, the simulator uses **Euler-Cromer (semi-implicit Euler)** integration with time step $\tau = 0.02\text{ s}$:
$$\dot{x}_{t+1} = \dot{x}_t + \tau \ddot{x}_t, \quad x_{t+1} = x_t + \tau \dot{x}_{t+1}$$
$$\dot{\theta}_{t+1} = \dot{\theta}_t + \tau \ddot{\theta}_t, \quad \theta_{t+1} = \theta_t + \tau \dot{\theta}_{t+1}$$

### 9.3 Reinforcement Learning: Deep Q-Networks (DQN)
The task is framed as a Markov Decision Process (MDP) defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:
* **State Space $\mathcal{S} \subset \mathbb{R}^4$:** $\mathbf{s} = [x, \dot{x}, \theta, \dot{\theta}]$ (cart position, cart velocity, pole angle, pole angular velocity).
* **Action Space $\mathcal{A} = \{0, 1\}$:** $0 = \text{Push Left } (-F)$, $1 = \text{Push Right } (+F)$.
* **Discount Factor:** $\gamma = 0.99$.

The state-action value function $Q^*(s, a)$ satisfies the **Bellman Optimality Equation**:
$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

### 9.4 Mitigating DQN Training Instability
Training a neural network directly with standard temporal-difference Q-learning is often unstable due to:
1. **Correlated Trajectories:** Consecutive observations $(s_t, s_{t+1}, s_{t+2})$ violate the i.i.d. assumption.
2. **Moving Targets:** Updating $Q(s, a)$ also changes the target values $r + \gamma \max_{a'} Q(s', a')$.

**DQN Solutions Implemented in `DQNAgent`:**
1. **Experience Replay Buffer:** Past transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ are stored in a cyclic deque of capacity $N = 10,000$. Training batches ($B = 64$) are sampled uniformly at random:
   $$\mathcal{L}_{\text{DQN}}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( Q(s, a; \theta) - y \right)^2 \right]$$
2. **Separate Target Network ($Q_{\text{target}}$):** Target values are calculated using a separate network with frozen parameters $\theta^-$:
   $$y = r + (1 - d) \cdot \gamma \max_{a'} Q(s', a'; \theta^-)$$
   The target parameters $\theta^-$ are synchronized with the active network $\theta$ every 5 episodes.
3. **$\epsilon$-Greedy Policy with Exponential Decay:**
   $$a = \begin{cases} \text{random action from } \mathcal{A} & \text{with probability } \epsilon \\ \text{argmax}_{a'} Q(s, a'; \theta) & \text{with probability } 1 - \epsilon \end{cases}$$
   $$\epsilon_{t+1} = \max(\epsilon_{\text{min}}, \epsilon_t \cdot \lambda), \quad \lambda = 0.99, \ \epsilon_{\text{min}} = 0.02$$
4. **Shaped Reward Formulation:**
   Rather than using a simple step reward $r = 1.0$, the environment applies reward shaping to encourage centering and upright posture:
   $$r_{\text{shaped}} = 1.0 - 0.5 |x| - 1.0 |\theta|$$

---

## 10. Module 7: Vision Transformer (ViT) & Multi-Head Attention (`core/transformer_models.py`)

Module 7 implements an interactive, lightweight **Vision Transformer (ViT)** based on the pioneering architecture of Dosovitskiy et al. (2020) (*"An Image is Worth 16x16 Words"*), tailored for real-time exploratory visualization of patch tokenization, spatial position encodings, multi-head self-attention maps, and representation aggregation.

```
Input Image x ∈ ℝ^[1 × 28 × 28]
         │
         ▼  PATCH PARTITION & LINEAR PROJECTION (P = 7 × 7, Grid = 4 × 4)
16 Patches: x_p^(1..16) ∈ ℝ^49 ──> [Linear E: 49 → 64]
         │
         ▼  PREPEND CLS TOKEN & ADD LEARNABLE 1D POSITIONAL EMBEDDINGS
Tokens: [ [CLS]; x_p^1 E; ... ; x_p^16 E ] + E_pos  ∈ ℝ^[17 × 64]
         │
         ▼  TRANSFORMER ENCODER (Pre-LN × 2 Layers)
┌─────────────────────────────────────────────────────────────┐
│  z_0 = Tokens                                               │
│  z'_l = MHSA( LayerNorm(z_{l-1}) ) + z_{l-1}               │  (4 Heads, d_k = 16)
│  z_l  = MLP( LayerNorm(z'_l) ) + z'_l                       │  (Expansion ratio = 2)
└─────────────────────────────────────────────────────────────┘
         │
         ▼  REPRESENTATION AGGREGATION & CLASSIFICATION
CLS Token Embedding z_L^(0) ∈ ℝ^64 ──> [LayerNorm] ──> [Linear: 64 → 10]
         │
         ▼
Logits ŷ ∈ ℝ^10 ──> Softmax Probabilities
```

### 10.1 Patch Partitioning & Sequence Construction
Standard Convolutional Neural Networks process images via local sliding receptive fields. In contrast, Vision Transformers treat 2D images as sequences of discrete visual tokens:

1. An input grayscale image $\mathbf{x} \in \mathbb{R}^{H \times W \times 1}$ with $H = W = 28$ is divided into non-overlapping spatial patches of size $P \times P$ ($P = 7$):
   $$N = \frac{H \cdot W}{P^2} = \frac{28 \times 28}{7 \times 7} = 16 \text{ patches}$$
2. Each patch is flattened into a 1D vector $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2} = \mathbb{R}^{49}$.
3. Patches are projected into latent embedding dimension $D = 64$ using a learnable projection matrix $\mathbf{E} \in \mathbb{R}^{49 \times D}$ and bias $\mathbf{b} \in \mathbb{R}^D$:
   $$\mathbf{e}_i = \mathbf{x}_p^{(i)} \mathbf{E} + \mathbf{b}, \quad i = 1, \dots, 16$$
4. A learnable classification token $[CLS] \in \mathbb{R}^D$ is prepended to the sequence ($N+1 = 17$ tokens total), and learnable 1D spatial position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{17 \times D}$ are added element-wise:
   $$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \mathbf{e}_1; \dots; \mathbf{e}_{16} \right] + \mathbf{E}_{\text{pos}}$$

### 10.2 Scaled Dot-Product Attention & Multi-Head Specialization
Self-attention computes dynamic, data-dependent affinity weights between all pairs of tokens in the sequence.

For each attention head $h \in \{1, \dots, H\}$ (where $H = 4$, head dimension $d_k = D / H = 16$):
1. Sequence embeddings $\mathbf{z} \in \mathbb{R}^{17 \times D}$ are mapped to **Queries** ($\mathbf{Q}_h$), **Keys** ($\mathbf{K}_h$), and **Values** ($\mathbf{V}_h$):
   $$\mathbf{Q}_h = \mathbf{z} \mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{z} \mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{z} \mathbf{W}_h^V, \quad \mathbf{W}_h^{Q, K, V} \in \mathbb{R}^{D \times d_k}$$
2. Scaled Dot-Product Attention matrix $\mathbf{A}_h \in \mathbb{R}^{17 \times 17}$ is computed via row-wise softmax:
   $$\mathbf{A}_h = \text{Softmax}\left( \frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} \right)$$
   * The $\frac{1}{\sqrt{d_k}}$ scaling factor counteracts large dot products in high dimensions that would otherwise push the softmax into vanishing gradient saturation regions.
3. Values are pooled weighted by attention affinities:
   $$\text{Head}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{17 \times d_k}$$
4. Heads are concatenated and projected through output projection $\mathbf{W}^O \in \mathbb{R}^{D \times D}$:
   $$\text{MHSA}(\mathbf{z}) = \left[ \text{Head}_1; \text{Head}_2; \dots; \text{Head}_H \right] \mathbf{W}^O$$

### 10.3 Pre-LayerNorm Transformer Encoder Blocks
The architecture employs the **Pre-LN** configuration for improved gradient flow:
$$\mathbf{z}'_l = \text{MHSA}\left( \text{LayerNorm}(\mathbf{z}_{l-1}) \right) + \mathbf{z}_{l-1}$$
$$\mathbf{z}_l = \text{MLP}\left( \text{LayerNorm}(\mathbf{z}'_l) \right) + \mathbf{z}'_l$$

where the Feed-Forward Network ($\text{MLP}$) applies two linear transformations with Gaussian Error Linear Unit ($\text{GELU}$) activation:
$$\text{MLP}(\mathbf{u}) = \text{GELU}(\mathbf{u} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
$$\text{GELU}(x) = x \Phi(x) = x \cdot \frac{1}{2}\left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$

### 10.4 Internal Attention Tap & Spatial Heatmap Projection
When `return_attentions=True` is passed to `MiniVisionTransformer`, the forward pass extracts the attention tensors $\mathbf{A}_{l, h} \in \mathbb{R}^{17 \times 17}$ for every layer $l$ and head $h$.

To project token attention back to the 2D pixel space for any query token $q \in \{0, \dots, 16\}$:
1. Extract affinity row $\mathbf{a} = \mathbf{A}_{l, h}[q, 1:] \in \mathbb{R}^{16}$.
2. Reshape into $4 \times 4$ spatial grid $M_{4 \times 4}$.
3. Upsample $M_{4 \times 4}$ to $28 \times 28$ via bilinear interpolation to form the continuous attention overlay:
   $$\mathcal{H}(u, v) = \text{BilinearUpsample}\left( M_{4 \times 4} \right)(u, v)$$

---

## 11. Module 8: Adversarial Robustness & Gradient Perturbations (`core/adversarial_models.py`)

Module 8 demonstrates the vulnerability of deep neural networks to worst-case adversarial input perturbations and explores the mathematics of gradient-based evasion attacks.

```
Clean Image x ∈ [0, 1] ──> [CNN Forward] ──> True Class y (Loss ℒ(x, y))
        │                                          │
        │                                          ▼ [Backprop w.r.t Input]
        │                                    Input Gradient ∇_x ℒ
        │                                          │
        ▼                                          ▼
[ x ] + [ ε · sign( ∇_x ℒ ) ] ──> [Clip [0, 1]] ──> Perturbed Adversarial Image x_adv
                                                           │
                                                           ▼
                                                Misclassified Class y' ≠ y!
```

### 11.1 The Linearity Hypothesis (Goodfellow et al., 2014)
A common misconception was that adversarial vulnerability is caused by deep non-linearities and local overfitting. However, Goodfellow et al. demonstrated that adversarial vulnerability primarily stems from **excessive linearity in high-dimensional input spaces**.

Consider a linear activation unit with weight vector $\mathbf{w} \in \mathbb{R}^d$ and input $\mathbf{x} \in \mathbb{R}^d$.
If an adversary adds an imperceptible perturbation $\boldsymbol{\eta} \in \mathbb{R}^d$ constrained by $\|\boldsymbol{\eta}\|_\infty \le \epsilon$:
$$\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\eta}$$
The activation of the unit becomes:
$$\mathbf{w}^\top \tilde{\mathbf{x}} = \mathbf{w}^\top \mathbf{x} + \mathbf{w}^\top \boldsymbol{\eta}$$

To maximize this activation subject to the $L_\infty$ constraint, the adversary chooses $\boldsymbol{\eta} = \epsilon \cdot \text{sign}(\mathbf{w})$.
The resulting activation change is:
$$\Delta = \mathbf{w}^\top \boldsymbol{\eta} = \epsilon \|\mathbf{w}\|_1 = \epsilon \sum_{i=1}^d |w_i| = \epsilon \cdot d \cdot \bar{w}$$
where $\bar{w}$ is the average absolute weight magnitude.
* Even if $\epsilon$ is very small (e.g., $\epsilon = 0.15$), as dimensionality $d$ grows ($d = 784$ for MNIST, $d = 150,528$ for ImageNet), the cumulative activation shift $\Delta$ scales linearly with $d$, easily overwhelming the network's internal margins and flipping decision boundaries.

### 11.2 Fast Gradient Sign Method (FGSM)
FGSM is a single-step first-order gradient method that computes the optimal perturbation maximizing the cross-entropy loss under an $L_\infty$ norm constraint.

* **Untargeted Formulation:** Maximizes classification loss for ground-truth label $y_{\text{true}}$:
  $$\mathbf{x}_{\text{adv}} = \text{clip}_{[0, 1]} \left( \mathbf{x} + \epsilon \cdot \text{sign}\left( \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y_{\text{true}}) \right) \right)$$
* **Targeted Formulation:** Forces the model to predict a specific target class $y_{\text{target}} \ne y_{\text{true}}$ by descending the loss of the target:
  $$\mathbf{x}_{\text{adv}} = \text{clip}_{[0, 1]} \left( \mathbf{x} - \epsilon \cdot \text{sign}\left( \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y_{\text{target}}) \right) \right)$$

### 11.3 Projected Gradient Descent (PGD - Madry et al., 2018)
While FGSM takes a single large step, PGD is an iterative multi-step optimization method that computes stronger adversarial perturbations by taking small steps of size $\alpha$ and projecting back onto the $L_\infty$ $\epsilon$-ball $\mathcal{B}_\epsilon(\mathbf{x})$:

$$\mathbf{x}^{(0)} = \mathbf{x} + \mathbf{\delta}_{\text{init}}, \quad \mathbf{\delta}_{\text{init}} \sim \mathcal{U}(-\epsilon, \epsilon)$$
$$\mathbf{x}^{(t+1)} = \Pi_{\mathcal{B}_\epsilon(\mathbf{x}) \cap [0, 1]^d} \left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}\left( \nabla_{\mathbf{x}^{(t)}} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y_{\text{true}}) \right) \right)$$

where the projection operator is:
$$\Pi_{\mathcal{B}_\epsilon(\mathbf{x}) \cap [0, 1]^d}(\mathbf{x}') = \text{clip}_{[0, 1]} \left( \text{clip}_{[\mathbf{x} - \epsilon, \mathbf{x} + \epsilon]} (\mathbf{x}') \right)$$
Standard step size is set to $\alpha = \frac{2\epsilon}{T}$ where $T$ is the number of steps (e.g., $T = 10$).

### 11.4 Perturbation Distance Metrics
To quantify the visual and numerical distortion introduced by the attack:
* **$L_\infty$ Chebyshev Distance:** Maximum absolute pixel change:
  $$\|\boldsymbol{\delta}\|_\infty = \max_{i} |\tilde{x}_i - x_i|$$
* **$L_2$ Euclidean Distance:** Root-mean-square perturbation energy:
  $$\|\boldsymbol{\delta}\|_2 = \sqrt{\sum_{i=1}^d (\tilde{x}_i - x_i)^2}$$
* **$L_0$ Hamming Sparsity:** Count of modified pixels:
  $$\|\boldsymbol{\delta}\|_0 = \sum_{i=1}^d \mathbb{I}(|\tilde{x}_i - x_i| > 10^{-4})$$

---

## 12. Module 9: 2D Generative Diffusion & Flow Matching (`core/diffusion_models.py`)

Module 9 demonstrates continuous generative modeling, probability density transport, and neural ordinary differential equations (Neural ODEs) using **Optimal Transport Flow Matching (OT-CFM)** and **Continuous Diffusion**.

```
PRIOR DISTRIBUTION p_0(x)                                    TARGET DISTRIBUTION q(x)
Standard Gaussian N(0, I)                                  (Heart, Swiss Roll, Rings)
        │                                                              │
        │             TIME t ∈ [0, 1] CONTINUOUS INTERPOLATION          │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
                   Coupled Point: x_t = (1 - t) x_0 + t x_1
                                        │
                                        ▼  GROUND-TRUTH VELOCITY: u_t = x_1 - x_0
                   Loss: ‖ v_θ(x_t, t) - (x_1 - x_0) ‖²
                                        │
                                        ▼  TRAINED NEURAL VECTOR FIELD
                   v_θ(x, t): ℝ² × [0, 1] ──> ℝ² (Velocity Vectors)
                                        │
                                        ▼  SAMPLING VIA NUMERICAL ODE
                   dx/dt = v_θ(x, t),   x(0) ~ N(0, I) ──> Generated Samples x(1)
```

### 12.1 Continuous Optimal Transport Flow Matching vs. DDPM
Standard Denoising Diffusion Probabilistic Models (DDPM; Ho et al., 2020) define forward transitions via stochastic Gaussian noise addition:
$$q(x_t \mid x_0) = \mathcal{N}\left( x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I} \right)$$
which results in curved, stochastic Brownian trajectories that require many sampling steps (50–1000) to invert.

**Flow Matching (Lipman et al., 2022)** constructs continuous generative models using deterministic velocity vector fields $v_t(x)$. Under the **Optimal Transport (OT)** formulation:
1. Sample prior noise $\mathbf{x}_0 \sim p_0(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. Sample target data point $\mathbf{x}_1 \sim q(\mathbf{x})$.
3. Define the straight-line probability trajectory:
   $$\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1]$$
4. The exact target velocity vector field along this trajectory is constant:
   $$\mathbf{u}_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$$
5. The neural network $v_\theta(\mathbf{x}, t)$ is trained via Conditional Flow Matching (CFM) with simple Mean Squared Error:
   $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, 1), \mathbf{x}_0 \sim p_0, \mathbf{x}_1 \sim q} \left[ \left\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|^2 \right]$$

Because optimal transport trajectories are straight lines in feature space, generative inference is numerically well-behaved and can be solved in as few as 20–50 Euler steps.

### 12.2 Sinusoidal Time Conditioning (`FlowFieldMLP`)
To provide the network with continuous time awareness across scales, the scalar timestamp $t \in [0, 1]$ is encoded into high-dimensional sinusoidal Fourier features:
$$\phi(t) = \left[ \sin(2^0 \pi t), \cos(2^0 \pi t), \sin(2^1 \pi t), \cos(2^1 \pi t), \dots, \sin(2^{K-1} \pi t), \cos(2^{K-1} \pi t) \right]^\top \in \mathbb{R}^{2K}$$
with $K = 8$ frequency octaves ($2K = 16$ dimensions).

The spatial position $\mathbf{x} \in \mathbb{R}^2$ is concatenated with $\phi(t)$ and passed through a Multi-Layer Perceptron with SiLU (Swish) activations:
$$\mathbf{h}_0 = [\mathbf{x}; \phi(t)] \in \mathbb{R}^{18}$$
$$\mathbf{h}_1 = \text{SiLU}(\mathbf{h}_0 \mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{h}_2 = \text{SiLU}(\mathbf{h}_1 \mathbf{W}_2 + \mathbf{b}_2), \quad \dots$$
$$\mathbf{v}_\theta(\mathbf{x}, t) = \mathbf{h}_L \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \in \mathbb{R}^2$$

where $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$.

### 12.3 Generative Sampling via Numerical ODE Integration
To synthesize new points from pure noise:
1. Initialize particle positions $\mathbf{X}^{(0)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. Discretize the time interval $[0, 1]$ into $M$ integration intervals with step size $\Delta t = 1 / M$.
3. Advance particles using **Euler Integration**:
   $$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} + \Delta t \cdot v_\theta\left( \mathbf{X}^{(k)}, t_k \right), \quad t_k = k \cdot \Delta t$$
   Or optionally using the higher-order **Midpoint Runge-Kutta 2 (RK2)** method:
   $$\mathbf{k}_1 = v_\theta(\mathbf{X}^{(k)}, t_k)$$
   $$\mathbf{k}_2 = v_\theta\left( \mathbf{X}^{(k)} + \frac{\Delta t}{2}\mathbf{k}_1, t_k + \frac{\Delta t}{2} \right)$$
   $$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} + \Delta t \cdot \mathbf{k}_2$$

### 12.4 Parametric Target Manifolds
The simulator evaluates probability flow on four non-trivial geometric manifolds:
1. **Parametric Cardioid Heart:**
   $$x(u) = 16\sin^3(u), \quad y(u) = 13\cos(u) - 5\cos(2u) - 2\cos(3u) - \cos(4u), \quad u \in [0, 2\pi)$$
   Normalized and centered to $[-2.5, 2.5]^2$.
2. **Archimedean Swiss Roll:**
   $$r(\theta) = 0.4 \theta, \quad x(\theta) = r(\theta)\cos(\theta), \quad y(\theta) = r(\theta)\sin(\theta), \quad \theta \in [\pi, 4\pi]$$
3. **Interlocking Two Moons:**
   Upper and lower crescent manifolds with horizontal offset and additive Gaussian thickness $\sigma = 0.08$.
4. **Concentric Multi-Rings:**
   Multiple radial shells $r_k \in \{0.6, 1.4, 2.2\}$ with uniform angular distribution.

---

## 13. Offline Training Pipeline & Serialization (`core/trainer.py`)

To ensure immediate usability without requiring users to wait for offline models to converge, [`core/trainer.py`](file:///home/cyrus/Projects/Python_AI/NeuroNetWorks/core/trainer.py) pre-trains and caches compact weights in `models_weights/` and datasets in `data/`.

### 13.1 Checkpoint Specifications

| Checkpoint File | Target Model Class | Epochs / Steps | Optimizer & LR | Loss Function | Final Performance Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mnist_cnn.pth` | `MNISTCNN` | 2 Epochs | Adam, $\eta = 0.002$ | Cross-Entropy | **98.6% Test Accuracy** |
| `mnist_vit.pth` | `MiniVisionTransformer` | 3 Epochs | AdamW, $\eta = 0.001$ | Cross-Entropy | **94.8% Test Accuracy** |
| `mnist_autoencoder.pth` | `MNIST2DAutoencoder` | 3 Epochs | Adam, $\eta = 0.003$ | MSE Loss | **Reconstruction Loss: 0.038** |
| `signal_denoiser.pth` | `SignalDenoising1DCNN` | 500 Steps | Adam, $\eta = 0.003$ | MSE Loss | **+14.5 dB SNR Gain** |
| `cartpole_dqn.pth` | `QNetwork` | 150 Episodes | Adam, $\eta = 0.002$ | MSE Loss | **Solved (Survival > 400 steps)** |
| `mnist_samples.npz` | Curated Data Cache | 1,000 Samples | N/A (100 per class) | N/A | Balanced Offline Test Digits |

### 13.2 Serialization Pipeline
Weights are exported using PyTorch's state dictionary protocol:
```python
torch.save(model.state_dict(), out_path)
```
This stores only the learned parameter tensors, keeping checkpoint files lightweight (~300 KB to ~2.5 MB) and fast to load.

---

## 14. Comprehensive Machine Learning & Deep Learning Glossary

* **Activation Function:** A non-linear mathematical mapping applied element-wise to neuron outputs to enable the network to approximate non-linear functions.
* **Autograd:** PyTorch's reverse-mode automatic differentiation engine that constructs a directed acyclic graph (DAG) during forward execution to compute derivatives via the chain rule during `.backward()`.
* **Bottleneck:** A layer with lower dimensionality than its surrounding layers, forcing the model to learn a compressed representation of the data.
* **Convolution (2D / 1D):** An operation that applies a localized sliding kernel across spatial or temporal dimensions, enabling weight sharing and translation invariance.
* **Cross-Entropy Loss:** A loss function measuring the divergence between the true distribution and the predicted probability distribution.
* **Deep Q-Network (DQN):** A model-free reinforcement learning algorithm that uses a deep neural network to approximate the action-value function $Q(s, a)$.
* **Dropout:** A regularization technique that randomly zeros out activations during training to reduce co-adaptation of neurons and mitigate overfitting.
* **Experience Replay:** A buffer in reinforcement learning that stores historical transitions $(s, a, r, s', d)$ and samples random mini-batches during training to break temporal correlation.
* **Fast Fourier Transform (FFT):** An algorithm that computes the Discrete Fourier Transform of a sequence, converting time-domain signals into frequency-domain spectra.
* **Gradient Descent:** An optimization algorithm that iteratively steps in the direction of steepest descent ($-\nabla \mathcal{L}$) to minimize a loss function.
* **Information Entropy:** A metric from information theory ($H = -\sum p_i \log_2 p_i$) quantifying the uncertainty or dispersion of a probability distribution.
* **Latent Space:** An embedded manifold where high-dimensional data is represented in a compressed, lower-dimensional coordinate system.
* **Max Pooling:** A downsampling operation that selects the maximum value within a sliding window, reducing spatial resolution and increasing receptive field size.
* **Peak Signal-to-Noise Ratio (PSNR):** A metric in decibels (dB) measuring the ratio between the maximum possible power of a signal and the power of corrupting noise.
* **Receptive Field:** The region of the input space that influences the activation of a specific neuron in a deeper layer.
* **Saliency Map:** An attribution technique that visualizes the gradient of the predicted class score with respect to input pixels, highlighting the features that most influenced the classification.
* **Symplectic Integrator:** A numerical integration method (such as Euler-Cromer) designed to conserve phase space volume and maintain energy stability in dynamic physical systems.
* **Universal Approximation Theorem:** A mathematical theorem stating that a feedforward network with at least one non-linear hidden layer and sufficient neurons can approximate any continuous function on compact subsets to arbitrary precision.

---

*Authored for the **NeuroNetWorks** Didactic Suite. For user interface guides and hands-on experiments, consult the companion [USER_MANUAL.md](file:///home/cyrus/Projects/Python_AI/NeuroNetWorks/USER_MANUAL.md).*
