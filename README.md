# 🧠 NeuroCanvas × TBP.Monty

### Unconstrained Multidimensional Neocortical Heterarchy BCI, Pure Phase-Coherence Synthesizer (Zero-Power), 89.5 Hz Ripple Causal DAGs, 2D Working Memory Geometry, and Real-Time Closed-Loop Diffusion

[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1016/j.neuron.2014.03.013](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2014.03.013-orange.svg)](https://doi.org/10.1016/j.neuron.2014.03.013)
[![DOI:10.1038/nature08573](https://img.shields.io/badge/DOI-10.1038%2Fnature08573-blue.svg)](https://doi.org/10.1038/nature08573)
[![arXiv:2507.05888](https://img.shields.io/badge/arXiv-2507.05888-b31b1b.svg)](https://arxiv.org/abs/2507.05888)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-yellow.svg)](https://doi.org/10.1088/1741-2552/aacfe4)

---

## 📑 Table of Contents
1. [Executive Summary & Paradigm Shift](#1-executive-summary--paradigm-shift)
2. [Biophysical Foundations: Pure Phase-Coherence & TBT 2.0](#2-biophysical-foundations-pure-phase-coherence--tbt-20)
   - 2.1 [Single-Device Scale (26 mm Footprint = 2,000–4,200 Macrocolumns)](#21-single-device-scale-26-mm-footprint--20004200-macrocolumns)
   - 2.2 [The Zero-Power Principle: Why Scalar FFT Power is Obsolete](#22-the-zero-power-principle-why-scalar-fft-power-is-obsolete)
   - 2.3 [Universal Column Engine vs. Regional Generative Modality](#23-universal-column-engine-vs-regional-generative-modality)
3. [Multi-Scale Oscillatory Hierarchy & 2D Neural Geometry](#3-multi-scale-oscillatory-hierarchy--2d-neural-geometry)
   - 3.1 [Continuous PAC Chronometer ($K_\theta = f_\theta / f_\delta$)](#31-continuous-pac-chronometer-k_\theta--f_\theta--f_\delta)
   - 3.2 [Two-Phase Theta Chronology & Past–Future Vector ($r_y$)](#32-two-phase-theta-chronology--pastfuture-vector-r_y)
   - 3.3 [2D Orthogonal SWM Rank Manifold (Fan 2024 / Chen 2024)](#33-2d-orthogonal-swm-rank-manifold-fan-2024--chen-2024)
   - 3.4 [89.5 Hz Cortical Ripple Causal DAG & Gram-Schmidt Tree (Dickey 2022)](#34-895-hz-cortical-ripple-causal-dag--gram-schmidt-tree-dickey-2022)
   - 3.5 [Genuine Cosine Delta Stability & World Seal Unlocking](#35-genuine-cosine-delta-stability--world-seal-unlocking)
4. [Anti-Blur Latent Conditioning & Generative Dynamics](#4-anti-blur-latent-conditioning--generative-dynamics)
   - 4.1 [Token-Wise Norm Calibration (Cross-Attention Sharpening)](#41-token-wise-norm-calibration-cross-attention-sharpening)
   - 4.2 [Elimination of the Recursive Pixel-Blur Loop (`cv2.addWeighted`)](#42-elimination-of-the-recursive-pixel-blur-loop-cv2addweighted)
   - 4.3 [SVD Tangent Bundle Affordance Operator (No 2D Pixel Warping)](#43-svd-tangent-bundle-affordance-operator-no-2d-pixel-warping)
5. [Decoupled Multi-Service Microarchitecture](#5-decoupled-multi-service-microarchitecture)
   - 5.1 [Structural Decoupling (`neuro_genesis_engine` vs `neuro_hud` vs runner)](#51-structural-decoupling-neuro_genesis_engine-vs-neuro_hud-vs-runner)
   - 5.2 [Dynamic Hot-Switching of Device Roles at Runtime (Zero Restarts)](#52-dynamic-hot-switching-of-device-roles-at-runtime-zero-restarts)
   - 5.3 [Headless Mode & Server Isolation](#53-headless-mode--server-isolation)
6. [Docker Compose Production Deployment](#6-docker-compose-production-deployment)
7. [Interactive Keybindings & HUD Instrumentation](#6-interactive-keybindings--hud-instrumentation)
8. [CLI Configuration Reference](#7-cli-configuration-reference)
9. [Comprehensive Scientific Bibliography & DOIs](#8-comprehensive-scientific-bibliography--dois)

---

## 1. Executive Summary & Paradigm Shift

Traditional Brain-Computer Interfaces (BCIs) reduce neurophysiology to 1D scalar band power or discrete classification states. **NeuroCanvas** treats the cerebral cortex as an **endogenous generative simulation engine**, interfacing continuous mesoscopic electrophysiology directly with latent diffusion models and Thousand Brains neocortical heterarchies (*Hawkins, Leadholm, Clay, 2025/2026*).

### Fundamental Axioms:
1. **The Human Brain is the World Model:** We do not execute heavy, latency-plagued external AI world models to simulate reality. The 16 billion neurons of the human neocortex compute state transitions in real time. Latent diffusion acts as an **inverted artificial retina**—a high-dimensional projector converting cortical semantic coordinates into photons on screen.
2. **Zero 2D Pixel Warping:** The brain does not shift the visual field via affine canvas transformations. Action is an **affordance-based generative transition** (*Gibson, 1979; Friston, 2010*). When an action is taken, the object itself undergoes physical/semantic metamorphosis via the tangent bundle of its concept manifold.
3. **Pure Phase Coherence (Zero Power):** All computation is mediated through volume-conduction-free corrected imaginary Phase Locking Value ($ci\text{PLV}$), directed phase lags, and Kuramoto synchronization order parameters. Scalar FFT power is completely eliminated.
4. **Complete Decoupling:** The mathematical engine (`neuro_genesis_engine.py`), visual instrumentation (`neuro_hud.py`), and operational runner (`neuro_open_latent_genesis_live.py`) are strictly decoupled through a flat telemetry data contract. Modifying one file never breaks the others.

---

## 2. Biophysical Foundations: Pure Phase-Coherence & TBT 2.0

### 2.1 Single-Device Scale (26 mm Footprint = 2,000–4,200 Macrocolumns)
A single FreeEEG16 sensor head has a diameter of $26\text{ mm}$ ($\text{Area} = \pi \times 13^2 \approx 531\text{ mm}^2$). 
* According to Vernon Mountcastle (1957, 1997) and Jeff Hawkins (2017), a cortical macrocolumn has a diameter of $300\text{--}600\ \mu\text{m}$ (area $\sim 0.12\text{--}0.28\text{ mm}^2$).
* Beneath a single 26 mm sensor lie **between 1,900 and 4,200 macrocolumns**, containing **over 250,000 minicolumns** and **more than 50 million neurons** (at $\sim 100,000\text{ neurons/mm}^2$).
* 16 concentric electrodes measure **120 unique bipolar dipoles**, capturing the phase-coherence manifold across this cellular population.

```
                    SINGLE 26mm SENSOR (531 mm²)
 ┌─────────────────────────────────────────────────────────────────┐
 │  • 16 Concentric Electrodes -> 120 Bipolar Dipoles              │
 │  • 1,900 to 4,200 Cortical Macrocolumns (Mountcastle / Hawkins)  │
 │  • >250,000 Minicolumns across Layers 1 to 6                    │
 │  • >50,000,000 Biological Neurons                               │
 │                                                                 │
 │  COMPLETE SELF-SUFFICIENCY: A single sensor independently       │
 │  decodes delta-theta slots, 89.5 Hz ripple causal DAGs,         │
 │  2D rank manifolds, and drives closed-loop diffusion!           │
 └─────────────────────────────────────────────────────────────────┘
```

**Corollary:** A single sensor anywhere on the cranium is a self-sufficient computational cluster capable of executing the full canonical microcircuit.

### 2.2 The Zero-Power Principle: Why Scalar FFT Power is Obsolete
Conventional single-channel EEGs rely on scalar band power ($\mu\text{V}^2/\text{Hz}$), which collapses spatial phase relationships, conflates volume conduction with neural communication, and is susceptible to muscular artifacts.
NeuroCanvas operates entirely on **Phase Geometry**:
1. **$ci\text{PLV}$ (Corrected Imaginary Phase Locking Value; *Bruña et al., 2018*):** Eliminates zero-lag volume conduction artifacts.
2. **Kuramoto Order Parameter ($R \in [0, 1]$):** Quantifies phase synchronization across the 120 dipoles without amplitude bias.
3. **Asymmetric Phase Lags ($\Delta \phi \neq 0$):** Establishes directed causal flow ("who leads whom").

### 2.3 Universal Column Engine vs. Regional Generative Modality
Following Mountcastle's principle of cortical uniformity, the neocortex implements an identical computational microcircuit everywhere. We formalize this through a two-layer architecture:

#### Layer A: Universal Column Engine (`UniversalColumnProcessor`)
Runs identically in **any** connected physical region:
* Calculates the $\theta/\delta$ PAC chronometer ($K_\theta = f_\theta / f_\delta \implies S(t) \in [4, 16]$ slots).
* Projects 120 dipoles into the 768-D cortical column basis ($\mathbf{W}_{\text{phys}} \in \mathbb{R}^{768 \times 120}$).
* Extracts 89.5 Hz Dickey ripple causal DAGs ($ci\text{PLV}$).
* Builds the recursive Gram-Schmidt tree ($A \supset B$ vs $A \parallel B$) across its own slots.
* Evaluates cycle-to-cycle cosine stability and charges/discharges World Seals.

#### Layer B: Regional Generative Modality (`HeterarchicalWorldSynthesizer`)
Maps the universal tree into the specific generative degrees of freedom of diffusion:
* **`AFz` (rmPFC / dACC):** Macro-Scene Habitat & Relational Base ($T_{\text{root}}$) on slow Delta ($0.5\text{--}1.5\text{ Hz}$).
* **`F3` (Left dlPFC):** Symbolic Syntax, Local Ranks ($L_1\text{--}L_3$), and Token-Level Cross-Attention binding.
* **`F4` (Right dlPFC):** Atmospheric Context, Global Ranks ($G_1\text{--}G_3$), Channel-Wise Latent Dispersion, and Denoising `strength`.
* **`FCz` (pre-SMA / SMA):** 16-D SVD Tangent Affordance Operator (active verbs / physical transformations).
* **`Fpz` (BA 10):** Epistemic Horizon, Continuous Hopfield Energy Landscape, and Paradigm Shifts.

---

## 3. Multi-Scale Oscillatory Hierarchy & 2D Neural Geometry

```
  FREQUENCY SCALE       BAND            REGIONAL SPECIALIZATION  GENERATIVE FUNCTION
 ────────────────────────────────────────────────────────────────────────────────────────────────
  Slow Delta            0.5 – 1.5 Hz    AFz / rmPFC              Macro-Scene & Relational Canvas
  Theta Carrier         4.0 – 8.0 Hz    Frontal Cortex           Working Memory Slot Parsing
  Infragranular Beta    15.0 – 30.0 Hz  F3 / F4 (Deep Layers)    Cross-Attention Token Lock (Gating)
  Superficial Gamma     30.0 – 65.0 Hz  L2/3 Supragranular       Active Feature Emission / Affordances
  Dickey Ripples        70.0 – 100 Hz   Transcortical            Causal Directed DAG (A ⊃ B vs A ∥ B)
  Maximum Gamma Limit   up to 100.0 Hz  Tunable (--gamma-max)    Full High-Frequency Tracking
```

### 3.1 Continuous PAC Chronometer ($K_\theta = f_\theta / f_\delta$)
Working memory capacity is dynamic (*Lisman & Jensen, 2013; Axmacher et al., 2010*):
$$K_\theta(t) = \text{clamp}\left( \text{round}\left( \frac{f_\theta(t)}{f_\delta(t)} \right), 2, 8 \right), \quad S(t) = 2 \times K_\theta(t) \in [4, 16] \text{ active phase sectors}$$
The engine resamples the continuous 120-dipole state into exactly $S(t)$ temporal sectors via GPU 1D adaptive average pooling (`torch.nn.functional.adaptive_avg_pool1d`) without host–device synchronization stalls.

### 3.2 Two-Phase Theta Chronology & Past–Future Vector ($r_y$)
Following *Colgin et al. (Nature, 2009)* and *Bieri et al. (Neuron, 2014)*:
* **Slot 0 (Early Descending Theta Phase, $\sim 257^\circ$, Slow Gamma):** Encodes the **Past / Memory Anchor ($\mathbf{z}_{\text{past}}$)**.
* **Slot 31 (Theta Trough, $\sim 329^\circ$, Fast Gamma):** Encodes the **Future / Lookahead Intention ($\mathbf{z}_{\text{future}}$)**.
* **Temporal Bias ($r_y$):** Evaluates the directional momentum of thought:
  $$r_y = \frac{\|\mathbf{z}_{\text{future}}\| - \|\mathbf{z}_{\text{past}}\|}{\|\mathbf{z}_{\text{future}}\| + \|\mathbf{z}_{\text{past}}\| + \epsilon} \in [-1.0, +1.0]$$
  Modulates transformation velocity without spatial dislocation.

### 3.3 2D Orthogonal SWM Rank Manifold (Fan 2024 / Chen 2024)
As demonstrated by *Fan et al. (Nature Human Behaviour, 2024)* and *Chen et al. (Neuron, 2024)*, sequence working memory in primate and human prefrontal cortex is organized along two orthogonal geometric axes:
* **Axis $X$ (Local Rank $L$):** Position of a feature/item within a chunk ($L_1, L_2, L_3$).
* **Axis $Y$ (Global Rank $G$):** Position of a chunk within the macro-sequence ($G_1, G_2, G_3$).

The 120 dipoles project onto this 2D plane:
$$x_{\text{local}} = \text{clamp}\left( \frac{1}{40} \sum_{d=1}^{120} \mathbf{\Psi}(d) \cdot \Delta x_d, -1.0, 1.0 \right), \quad y_{\text{global}} = \text{clamp}\left( \frac{1}{40} \sum_{d=1}^{120} \mathbf{\Psi}(d) \cdot \Delta y_d, -1.0, 1.0 \right)$$
The coordinate $(x_L, y_G)$ glides smoothly across the 2D working memory manifold, directly balancing token-level attention.

### 3.4 89.5 Hz Cortical Ripple Causal DAG & Gram-Schmidt Tree (Dickey 2022)
Volume-conduction-free directed phase locking across all 120 electrode pairs is computed at 89.5 Hz (*Dickey et al., PNAS 2022; Bruña et al., J Neural Eng 2018*):
$$ci\text{PLV}_{i, j} = \frac{\frac{1}{T} \sum_{t=1}^T \Im \left( z_i(t) z_j^*(t) \right)}{\sqrt{1 - \left( \frac{1}{T} \sum_{t=1}^T \Re \left( z_i(t) z_j^*(t) \right) \right)^2}}$$

The pairwise causal lead matrix between temporal slots is calculated on GPU:
$$\mathbf{D}_{i, j} = \frac{1}{120} \sum_{d=1}^{120} \left( \mathbf{Rip}_i(d) - \mathbf{Rip}_j(d) \right)$$
* $\mathbf{D}_{A, B} \ge +0.03 \implies A \supset B$ ($A$ is parent/container of $B$). Project into orthogonal complement via Gram-Schmidt:
  $$\mathbf{T}_{B}^{\perp} = \mathbf{T}_B - \frac{\langle \mathbf{T}_B, \mathbf{T}_{\text{accum}} \rangle}{\|\mathbf{T}_{\text{accum}}\|^2 + \epsilon} \mathbf{T}_{\text{accum}}$$
* $\mathbf{D}_{A, B} < 0.03 \implies A \parallel B$ ($A$ and $B$ are co-equal peers). Integrated via hemispheric latent partitioning.
* **Collinearity Rejection Guard:** If $\|\mathbf{T}^{\perp}\| \le 0.05 \|\mathbf{T}\|$, machine-precision noise is rejected, preventing single-frame visual glitches.

### 3.5 Genuine Cosine Delta Stability & World Seal Unlocking
World Seals do not rely on dummy counters. Stability is evaluated by comparing the 768-D slot vector between cycle $t$ and cycle $t-1$:
$$\text{drift\_cos}_s = \frac{\langle \mathbf{z}_{\text{slot}}(s, t), \mathbf{z}_{\text{slot}}(s, t-1) \rangle}{\|\mathbf{z}_{\text{slot}}(s, t)\| \|\mathbf{z}_{\text{slot}}(s, t-1)\|}$$
$$\text{stability}_s = (0.5 \cdot \text{drift\_cos}_s + 0.5) \times (0.2 + 0.8 \cdot R_{\text{kuramoto}})$$
* **Locking:** When $\text{stability}_s \ge \text{--seal-thresh}$ for `--seal-cycles` consecutive delta periods, the slot crystallizes into an immutable World Seal.
* **Unlocking:** When mental focus shifts and $\text{stability}_s < \text{--seal-thresh}$, the charge decays at the rate of `--decay-cycles`. When charge drops below 20%, **the seal dissolves and unlocks automatically**. Setting `--seal-thresh 1.0` disables sealing completely.

---

## 4. Anti-Blur Latent Conditioning & Generative Dynamics

### 4.1 Token-Wise Norm Calibration (Cross-Attention Sharpening)
In Stable Diffusion / LCM, text token vectors in `last_hidden_state` exhibit authentic norms of **$28.0$ to $36.0$**. Forcing an arbitrary fixed norm (e.g. $15.0$) halves the logits in Cross-Attention, which is mathematically equivalent to doubling the softmax temperature:
$$\text{Attention}(Q, K) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right)$$
This flattens attention distributions across the canvas, destroying fine details and turning landscapes into flat watercolor mud.

**The Fix:**
$$\mathbf{T}_{\text{calibrated}} = \frac{\mathbf{T}}{\|\mathbf{T}\|_2 + \epsilon} \odot \|\mathbf{T}_{\text{clean CLIP}}\|_2$$
Furthermore, 120-dipole neural injections target **only semantic tokens ($1 \dots S$)**, preserving structural anchor tokens ($0$: `<|startoftext|>` and trailing padding), keeping Cross-Attention razor-sharp.

### 4.2 Elimination of the Recursive Pixel-Blur Loop (`cv2.addWeighted`)
In recurrent img2img, applying $20\%$ pixel-space blending (`cv2.addWeighted(old, 0.2, new, 0.8)`) acts as an infinite-impulse-response (IIR) spatial low-pass filter. Combined with repeated VAE encode/decode cycles, the canvas degrades into total blur within 15 frames.

**The Fix:** The diffusion output `resp` is passed directly to the next iteration without pixel-level temporal low-pass blending. Visual sharpness is preserved across hours of continuous streaming.

### 4.3 SVD Tangent Bundle Affordance Operator (No 2D Pixel Warping)
`FCz` does not execute `cv2.warpAffine` camera panning. For the active concept, the engine computes the local SVD covariance over its $K$-nearest semantic neighbors in the 50,000-word CLIP manifold:
$$\mathbf{\Sigma} = \sum_{k=1}^K (\mathbf{w}_k - \mathbf{z}_{\text{root}})(\mathbf{w}_k - \mathbf{z}_{\text{root}})^T = \mathbf{U} \mathbf{S} \mathbf{V}^T$$
The top-16 right-singular vectors $\mathbf{V}_{:16} \in \mathbb{R}^{16 \times 768}$ span the **natural tangent space of physical affordances** (e.g. for water: flow, wave, freeze; for stone: crack, erode, crumble). 
The 120 dipoles of `FCz` project onto this tangent basis:
$$\Delta \mathbf{z} = \left( \mathbf{z}_{\text{FCz}} \cdot \mathbf{V}_{:16}^T \right) \mathbf{V}_{:16} \in \mathbb{R}^{768}$$
The object transforms according to its own natural physical degrees of freedom.

---

## 5. Decoupled Multi-Service Microarchitecture

```
                            PRODUCTION RUNTIME TOPOLOGY
                            
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                           CLIENT WEB BROWSER                                │
   │   • Live Canvas View: <img src="http://localhost:8080/stream.mjpg">          │
   │   • Direct Bluetooth: Web Bluetooth API (Service: 4fafc201-1fb5...)         │
   │   • Ingestion: Batched POST /api/eeg_push (16-channel 24-bit packets)       │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ HTTP / MJPEG (Port 8080)
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                     1. web_cloud_gateway.py (Port 8080)                     │
   │   • Web UI Host & MJPEG Video Streamer                                      │
   │   • Dynamic LSL Router: creates outlets 'User1_AFz', 'User2_F3' on demand   │
   │   • IPC Server on Port 6002 (key: canvas): receives rendered BGR frames     │
   └───────────────────▲───────────────────────────────────────┬─────────────────┘
                       │                                       │
     Rendered JPEG     │ Direct Memory IPC (Port 6002)         │ Native LabStreamingLayer
     Frame Push        │                                       │ (LSL Multicast Streams)
                       │                                       ▼
   ┌───────────────────┴─────────────────────────────────────────────────────────┐
   │                2. neuro_open_latent_genesis_live.py                         │
   │   • Orchestrator & Runner: pulls LSL frames via HeterarchicalBrainEngine    │
   │   • Delegates 100% of GPU math to neuro_genesis_engine.py                   │
   │   • Delegates 100% of UI/HUD to neuro_hud.py (unless --headless)            │
   │   • Direct IPC Client to Gateway (Port 6002, key: canvas)                   │
   │   • Direct IPC Client to Brain Server (Port 6000, key: brain)               │
   └───────────────────┬───────────────────────────────────────┬─────────────────┘
                       │                                       │
                       ▼                                       ▼
   ┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
   │      neuro_genesis_engine.py         │  │           neuro_hud.py            │
   │  • UniversalColumnProcessor (All Dev)│  │  • Pure Decoupled HUD            │
   │  • Hopfield CANN & SVD Affordances   │  │  • Reads flat telemetry: dict    │
   │  • True Cosine Delta World Seals     │  │  • 4D Gyroscope & 2D SWM Puck    │
   │  • Returns flat telemetry: dict      │  │  • Interactive Device Role Cycle │
   └──────────────────────────────────────┘  └───────────────────────────────────┘
                       │ Latent Tensors & Prompts
                       │ Direct Memory IPC (Port 6000)
                       ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                     3. brain_server.py (Port 6000)                          │
   │   • Dedicated U-Net Diffusion Worker (LCM / SD-Turbo / SDXL)                │
   │   • Single-pass GPU execution, returns raw rendered RGB arrays              │
   └─────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Structural Decoupling
* **`neuro_genesis_engine.py`:** Contains zero Pygame or GUI code. Outputs a single flat `telemetry: dict`.
* **`neuro_hud.py`:** Contains zero domain-specific class dependencies. Consumes `telemetry: dict` via safe `.get()` calls. Modifying the engine will never produce a `NameError` or `KeyError` in the HUD.
* **`neuro_open_latent_genesis_live.py`:** An immutable ~140-line coordinator pipeline.

### 5.2 Dynamic Hot-Switching of Device Roles at Runtime (Zero Restarts)
No need to restart the Python process to test different electrode placements:
* Press **`TAB`** in the HUD window to cycle active hardware devices (`Dev 0`, `Dev 1`, etc.).
* Press **`1`** $\to$ Instantly assign device to **`AFz`** (Macro-Scene & Delta Tree).
* Press **`2`** $\to$ Instantly assign device to **`F3`** (2D SWM Ranks & Syntax Locking).
* Press **`3`** $\to$ Instantly assign device to **`F4`** (Atmospheric Context & Entropy).
* Press **`4`** $\to$ Instantly assign device to **`FCz`** (4D Affordance Operator).
* Press **`5`** $\to$ Instantly assign device to **`Fpz`** (Epistemic Horizon & Paradigm Shift).
* Press **`0`** $\to$ Instantly **disable** the device (its regional widget dims to `[OFFLINE]`).

### 5.3 Headless Mode & Server Isolation
When executed with `--headless`:
* `neuro_hud.py` and `pygame` are **never imported or loaded into memory**.
* Runs in pure headless Docker containers or remote GPU servers with zero X11/Wayland dependencies.
* Rendered frames stream directly via IPC (Port 6002) to `web_cloud_gateway.py` for browser viewing at `http://localhost:8080/stream.mjpg`.

---

## 6. Docker Compose Production Deployment

The entire system is orchestrated via Docker Compose with full NVIDIA GPU passthrough, host networking (mandatory for LSL multicast resolution), and **Hugging Face model volume caching** so models are downloaded once and never redownloaded on container restarts.

### 1. `docker-compose.yml`

```yaml
services:
  # 1. Diffusion U-Net Worker (Port 6000)
  brain-server:
    build: .
    container_name: neuro_brain_server
    command: python3 brain_server.py --mode lcm
    network_mode: host
    ipc: host
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  # 2. Web UI, Web Bluetooth & Stream Gateway (Port 8080 & IPC 6002)
  web-gateway:
    build: .
    container_name: neuro_web_gateway
    command: python3 web_cloud_gateway.py 8080
    network_mode: host
    ipc: host
    depends_on:
      - brain-server
    restart: unless-stopped

  # 3. Headless BCI Genesis Core
  neuro-canvas:
    build: .
    container_name: neuro_canvas_engine
    command: >
      python3 neuro_open_latent_genesis_live.py
      --headless
      --lore world_lore.txt
      --steps 2
      --strength-low 0.65
      --burst-strength 0.85
    network_mode: host
    ipc: host
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - brain-server
      - web-gateway
    restart: unless-stopped
```

### 2. Launching with Docker Compose

```bash
# Build and launch all three microservices in detached mode
docker compose up --build -d

# Follow generation logs and LSL routing
docker compose logs -f neuro-canvas

# Open the Web UI in your browser:
# http://localhost:8080/
```

---

## 7. Interactive Keybindings & HUD Instrumentation

```
  KEY               ACTION                                NEUROCOMPUTATIONAL FUNCTION
 ──────────────────────────────────────────────────────────────────────────────────────────────────
  TAB               Cycle Device Selection                Selects Dev 0, Dev 1, etc., for role assignment
  1                 Assign to AFz                         Instantiates Macro-Scene & Delta Tree
  2                 Assign to F3                          Instantiates 2D SWM Rank Matrix & Syntax Lock
  3                 Assign to F4                          Instantiates Contextual Entropy & Denoising Modulation
  4                 Assign to FCz                         Instantiates 4D Affordance Gyroscope
  5                 Assign to Fpz                         Instantiates Epistemic Horizon & Hopfield Basin
  0                 Disable Device                        Sets device to Offline (stops faking data)
  H                 Toggle Minimal UI                     Switches between Full Diagnostic HUD and Clean Canvas
  R                 Reset World Seals                     Dissolves all active seals and clears chrono-tunnel
  C                 Flush Canvas                          Flushes img2img recurrent buffer
  ESC               Safe Shutdown                         Terminates threads and releases IPC sockets cleanly
```

### Visual Panel Layout:
1. **Top Bar (Fpz):** Displays active paradigm status, 50k Hopfield attractor name, and continuous epistemic tension meter.
2. **Left Panel (Primary Active Slots):** Displays the $S(t)$ delta-theta sectors of whichever physical region is active, with individual stability bars.
3. **Right-Top Panel (89.5 Hz Causal Ripple Graph):** Displays the live force-directed DAG of directed phase leads ($\Lambda$).
4. **Bottom Panel (Multi-Regional Diagnostic Suite):**
   * **`AFz` Tunnel (x=30):** Multi-ring Delta spiral chrono-tunnel.
   * **`F3/F4` 2D SWM Manifold (x=350):** 2D rank pad showing continuous $(x_L, y_G)$ puck motion + F4 entropy iris.
   * **`FCz` Gyroscope (x=750):** 4D radar showing translation $[l_x, l_y]$, torsion $r_x$, and lookahead $r_y$.
   * **Active Lore Palette (x=1050):** Clean concept cards with zero text collisions.

---

## 8. CLI Configuration Reference

```bash
# Minimal single-sensor test (F3 isolated, 100 Hz Gamma, no faking)
python3 neuro_open_latent_genesis_live.py \
  --users "User1:F3=0" \
  --concepts "mountains,rivers" \
  --gamma-max 100.0 \
  --seal-thresh 0.68 \
  --steps 2

# Full 5-region heterarchical suite
python3 neuro_open_latent_genesis_live.py \
  --users "User1:AFz=0,F3=1,F4=2,FCz=3,Fpz=4" \
  --lore world_lore.txt \
  --gamma-max 100.0 \
  --seal-thresh 0.72 \
  --decay-cycles 2 \
  --steps 2

# Production headless server (Docker / Cloud GPU)
python3 neuro_open_latent_genesis_live.py \
  --headless \
  --lore world_lore.txt \
  --gamma-max 100.0 \
  --steps 2
```

### Complete Parameter Index:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--gamma-max` | `float` | `100.0` | Maximum frequency cutoff for the cortical gamma band in Hz. |
| `--seal-thresh` | `float` | `0.68` | Cosine stability threshold required to charge a World Seal (`1.0` disables sealing). |
| `--seal-cycles` | `int` | `4` | Number of consecutive stable delta cycles required to lock a seal. |
| `--decay-cycles`| `int` | `2` | Number of unstable delta cycles required to dissolve/unlock a seal. |
| `--refresh-cycles`| `int` | `2` | Buffer length for cycle-to-cycle cosine stability comparison. |
| `--lore`, `--concepts-file` | `str` | `None` | Path to `.txt` or `.json` world lore file. |
| `--concepts` | `str` | `None` | Comma-separated list of concept strings. |
| `--users` | `str` | `None` | Montage mapping string (e.g. `'User1:F3=0'` or `'User1:AFz=0,FCz=1'`). |
| `--mode` | `str` | `lcm` | Pipeline mode (`lcm`, `turbo`, `sdxl-turbo`, `sdxl`). |
| `--speed` | `str` | `fast` | Speed profile (`fast` = $448 \times 336$, `quality` = $512 \times 384$). |
| `--steps` | `int` | `2` | Denoising inference steps per frame. |
| `--strength-low` | `float` | `0.55` | Baseline img2img mutation strength. |
| `--strength-high`| `float` | `0.75` | Ceiling of img2img mutation strength under high F4 entropy. |
| `--burst-strength`| `float`| `0.75` | Mutation strength during concept switches or Fpz paradigm shifts. |
| `--burst-duration`| `float`| `0.0` | Duration of concept switch burst in seconds. |
| `--headless` | `flag` | `False` | Disables Pygame window for headless server / Docker environments. |
| `--minimal-ui` | `flag` | `False` | Starts in full-canvas mode without diagnostic HUD widgets. |
| `--sps` | `int` | `250` | EEG sampling rate (`250` or `500` Hz). |
| `--no-taesd` | `flag` | `False` | Disables Tiny Autoencoder (TAESD) and uses standard SD 1.5 VAE. |
| `--no-color` | `flag` | `False` | Bypasses color constancy post-processing surgery. |

---

## 9. Comprehensive Scientific Bibliography & DOIs

1. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory. *Nature Human Behaviour*, 8, 2150–2163. [DOI: 10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
2. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** Flexible control of sequence working memory in the macaque frontal cortex. *Neuron*, 112(20), 3502–3514. [DOI: 10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
3. **Dickey, C. W., et al. (2022).** Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall. *PNAS*, 119(28), e2107797119. [DOI: 10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
4. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy. *arXiv:2507.05888v2 [q-bio.NC]*.
5. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** Working Memory 2.0. *Neuron*, 100(2), 463–475. [DOI: 10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
6. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory. *PNAS*, 115(5), 1117–1122. [DOI: 10.1073/pnas.1717766115](https://doi.org/10.1073/pnas.1717766115)
7. **Colgin, L. L., et al. (2009).** Frequency of gamma oscillations routes flow of information in the hippocampus. *Nature*, 462(7271), 353–357. [DOI: 10.1038/nature08573](https://doi.org/10.1038/nature08573)
8. **Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** Slow and fast gamma rhythms coordinate different spatial coding modes in hippocampal place cells. *Neuron*, 82(3), 670–681. [DOI: 10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
9. **Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** Cortical tracking of hierarchical linguistic structures in connected speech. *Nature Neuroscience*, 19(1), 158–164. [DOI: 10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
10. **Bruña, R., Maestú, F., & Pereda, E. (2018).** Phase Locking Value revisited: teaching new tricks to an old dog. *Journal of Neural Engineering*, 15(5), 056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
11. **Churchland, M. M., & Shenoy, K. V. (2024).** Preparatory activity and the expansive null-space. *Nature Reviews Neuroscience*, 25, 213–236. [DOI: 10.1038/s41583-024-00796-z](https://doi.org/10.1038/s41583-024-00796-z)
12. **Zimnik, A. J., & Churchland, M. M. (2021).** Independent generation of sequence elements by motor cortex. *Nature Neuroscience*, 24, 412–424. [DOI: 10.1038/s41593-021-00798-5](https://doi.org/10.1038/s41593-021-00798-5)
13. **Weber, J., et al. (2023).** Subspace partitioning in the human prefrontal cortex resolves cognitive interference. *PNAS*, 120(31), e2220523120. [DOI: 10.1073/pnas.2220523120](https://doi.org/10.1073/pnas.2220523120)
14. **Flesch, T., et al. (2022).** Orthogonal representations for robust context-dependent task performance in brains and neural networks. *Neuron*, 110(7), 1258–1270. [DOI: 10.1016/j.neuron.2022.01.005](https://doi.org/10.1016/j.neuron.2022.01.005)
15. **Panichello, M. F., & Buschman, T. J. (2021).** Shared mechanisms underlie the control of working memory and attention. *Nature*, 592, 601–605. [DOI: 10.1038/s41586-021-03390-w](https://doi.org/10.1038/s41586-021-03390-w)
16. **Badre, D., et al. (2021).** The dimensionality of neural representations for control. *Current Opinion in Behavioral Sciences*, 38, 20–28. [DOI: 10.1016/j.cobeha.2020.07.002](https://doi.org/10.1016/j.cobeha.2020.07.002)
17. **Koechlin, E., & Hyafil, A. (2007).** Anterior prefrontal function and the limits of human decision-making. *Science*, 318(5850), 594–598. [DOI: 10.1126/science.1142995](https://doi.org/10.1126/science.1142995)
18. **Boorman, E. D., et al. (2009).** How green is the grass on the other side? Frontopolar cortex and evidence for alternatives. *Neuron*, 62(5), 733–743. [DOI: 10.1016/j.neuron.2009.05.014](https://doi.org/10.1016/j.neuron.2009.05.014)
19. **Fries, P. (2015).** Rhythms for cognition: communication through coherence. *Neuron*, 88(1), 220–235. [DOI: 10.1016/j.neuron.2015.09.034](https://doi.org/10.1016/j.neuron.2015.09.034)
20. **Lisman, J. E., & Jensen, O. (2013).** The theta-gamma neural code. *Neuron*, 77(6), 1002–1016. [DOI: 10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
