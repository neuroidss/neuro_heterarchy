# 🧠 NeuroCanvas × TBP.Monty

### 100% VRAM-Resident Neocortical SDR Heterarchy BCI, Amodal Working Memory CUDA-IPC Bus, Pure Phase-Coherence Synthesizer ($ci\text{PLV}$), 89.5 Hz Ripple Causal DAGs, and Bit-for-Bit Deterministic Real-Time Diffusion & MusicGen

[![Deploy All-in-One on RunPod](https://img.shields.io/badge/RunPod_Hub-Deploy_All--in--One_(Vision_+_Music)-7c3aed?logo=runpod&style=for-the-badge)](https://console.runpod.io/hub/template/lwrm1qmm9z?ref=2n2ruooc)
[![Deploy Vision on RunPod](https://img.shields.io/badge/RunPod_Hub-Deploy_Diffusion_Only-0284c7?logo=runpod&style=for-the-badge)](https://console.runpod.io/hub/template/tcmwh5826v?ref=2n2ruooc)
[![Deploy MusicGen on RunPod](https://img.shields.io/badge/RunPod_Hub-Deploy_MusicGen_Only-059669?logo=runpod&style=for-the-badge)](https://console.runpod.io/hub/template/6nara33osx?ref=2n2ruooc)

[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1016/j.neuron.2014.03.013](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2014.03.013-orange.svg)](https://doi.org/10.1016/j.neuron.2014.03.013)
[![DOI:10.1038/nature08573](https://img.shields.io/badge/DOI-10.1038%2Fnature08573-blue.svg)](https://doi.org/10.1038/nature08573)
[![arXiv:2507.05888](https://img.shields.io/badge/arXiv-2507.05888-b31b1b.svg)](https://arxiv.org/abs/2507.05888)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-yellow.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.1038/nn.4186](https://img.shields.io/badge/DOI-10.1038%2Fnn.4186-teal.svg)](https://doi.org/10.1038/nn.4186)

---

## 🚀 1-Click Cloud Deployment (RunPod Hub Templates)

Deploy your private, isolated GPU studio directly inside RunPod with **zero external proxies, zero API key sharing, and native HTTPS (`:8080`)**:

| Template Name | Docker Image | Modalities Included | 1-Click Launch Link |
| :--- | :--- | :--- | :--- |
| **`neuro_heterarchy_all`** | `neuroidss/neuro_heterarchy_all:latest` | **Unified Vision + MusicGen** (Shared VRAM Working Memory Bus) | [**🚀 Launch All-in-One**](https://console.runpod.io/hub/template/lwrm1qmm9z?ref=2n2ruooc) |
| **`neuro_heterarchy_diffusion`** | `neuroidss/neuro_heterarchy_diffusion:latest` | **Vision Only** (LCM / SD-Turbo / SDXL-Turbo) | [**🎨 Launch Diffusion**](https://console.runpod.io/hub/template/tcmwh5826v?ref=2n2ruooc) |
| **`neuro_heterarchy_musicgen`** | `neuroidss/neuro_heterarchy_musicgen:latest` | **MusicGen Only** (32 kHz Continuous Diagonal Stream) | [**🎵 Launch MusicGen**](https://console.runpod.io/hub/template/6nara33osx?ref=2n2ruooc) |

---

## 📑 Table of Contents
1. [System Overview & Scientific Paradigm Shift](#1-system-overview--scientific-paradigm-shift)
2. [100% VRAM-Resident Amodal Cognitive Engine & Cross-Python CUDA IPC](#2-100-vram-resident-amodal-cognitive-engine--cross-python-cuda-ipc)
   - 2.1 [Zero-CPU Compute Law & Scaling from 16 (120 Dipoles) to 1,024 Channels (523,776 Dipoles)](#21-zero-cpu-compute-law--scaling-from-16-120-dipoles-to-1024-channels-523776-dipoles)
   - 2.2 [Amodal Working Memory Bus (`HardwareVRAMBus`) via Native PyTorch `_share_cuda_`](#22-amodal-working-memory-bus-hardwarevrambus-via-native-pytorch-_share_cuda_)
   - 2.3 [100% Bit-for-Bit Deterministic EEG $\to$ Video & Audio Homomorphism (`SHA-256` Verified)](#23-100-bit-for-bit-deterministic-eeg--video--audio-homomorphism-sha-256-verified)
3. [Biophysical Foundations & Neocortical Circuit Theory](#3-biophysical-foundations--neocortical-circuit-theory)
   - 3.1 [Macrocolumnar Scale: FreeEEG16 (120 Dipoles) & FreeEEG24 (276 Dipoles) on 26 mm](#31-macrocolumnar-scale-freeeeg16-120-dipoles--freeeeg24-276-dipoles-on-26-mm)
   - 3.2 [The Zero-Power Principle: Volume-Conduction-Free Phase Coherence ($ci\text{PLV}$)](#32-the-zero-power-principle-volume-conduction-free-phase-coherence-ciplv)
   - 3.3 [Inter-Region Monty SDR Heterarchy (L3➔L4 Anomaly Up, L6a➔L1 Prediction Down)](#33-inter-region-monty-sdr-heterarchy-l3l4-anomaly-up-l6al1-prediction-down)
   - 3.4 [Lossless Tensor Homomorphism & Multi-Architecture Vocabulary Bridge](#34-lossless-tensor-homomorphism--multi-architecture-vocabulary-bridge)
   - 3.5 [Pure Semantic Actions via Intrinsic SVD Affordances (Zero Pixel Shifting)](#35-pure-semantic-actions-via-intrinsic-svd-affordances-zero-pixel-shifting)
   - 3.6 [89.5 Hz Cortical Ripple Causal DAGs ($A \supset B$ vs $A \parallel B$)](#36-895-hz-cortical-ripple-causal-dags-a--b-vs-a--b)
4. [Zero-Buffer Real-Time Generative Engines (Vision & Audio)](#4-zero-buffer-real-time-generative-engines-vision--audio)
   - 4.1 [Continuous Diagonal Stream MusicGen: Fused Codebook Kernels, Auto-Min Block ($0.02\text{ s}$) & Server-Side 32 kHz Clock](#41-continuous-diagonal-stream-musicgen-fused-codebook-kernels-auto-min-block-002-s--server-side-32-khz-clock)
   - 4.2 [Zero-Buffer Server-Push Diffusion: 100% GPU Color Surgery, Auto-Steps & Latest-Frame RAM Mailbox](#42-zero-buffer-server-push-diffusion-100-gpu-color-surgery-auto-steps--latest-frame-ram-mailbox)
   - 4.3 [On-the-Fly Model Hot-Swapping (LCM / SD-Turbo / SDXL-Turbo & MusicGen Small / Medium / Melody / Large)](#43-on-the-fly-model-hot-swapping-lcm--sd-turbo--sdxl-turbo--musicgen-small--medium--melody--large)
5. [The 4-Stage Cortical Calibration Ladder (`FCz` ➔ `F3/F4` ➔ `AFz` ➔ `Fpz`)](#5-the-4-stage-cortical-calibration-ladder-fcz--f3f4--afz--fpz)
6. [Hardware Signal Chain, Auto-Reconnecting LSL Recorder & Local Docker Runner](#6-hardware-signal-chain-auto-reconnecting-lsl-recorder--local-docker-runner)
7. [REST API & AI Agent Integration](#7-rest-api--ai-agent-integration)
8. [Comprehensive Scientific Bibliography & DOIs](#8-comprehensive-scientific-bibliography--dois)

---

## 1. System Overview & Scientific Paradigm Shift

Conventional Brain-Computer Interfaces (BCIs) reduce high-dimensional cortical neurodynamics to discrete classification labels, heuristic `if/else` thresholds, or scalar 1D band-power estimates ($\mu\text{V}^2/\text{Hz}$). In contemplative and neurofeedback applications, traditional systems focus almost exclusively on **Stage 1 (Sensory Dissolution)** via alpha-band amplitude coherence—a state of top-down sensory inhibition that frequently leads to drowsiness rather than active mental synthesis.

**NeuroCanvas × TBP.Monty** is engineered for **Stage 2 (Generation Stage / Waking Lucid Synthesis)**, grounded in the **Thousand Brains Theory 2.0** [[1](#ref-1), [2](#ref-2)], **Working Memory 2.0** [[4](#ref-4), [5](#ref-5)], and empirical multi-scale cortical neurogeometry [[6](#ref-6), [7](#ref-7), [8](#ref-8), [9](#ref-9)]:

1. **No Telepathy Myth — Closed-Loop Cortical Entrainment:** The system does not claim to "read pre-existing thoughts." Instead, it provides a **100% deterministic, full-dimensional mathematical homomorphism** that translates $32 \times P$ cortical phase trajectories into continuous semantic transformations on screen and in audio. Through zero-buffer closed-loop visual and auditory feedback, the neocortex rapidly adapts its own phase dynamics to steer the generative manifold.
2. **Strict Anti-Pixel-Shift, Anti-`if/else` & Anti-Dimensionality-Reduction Law:** Mechanical 2D pixel manipulations (`warpAffine`), arbitrary `if/else` signal gates, and dimensionality-collapsing heuristics are strictly prohibited. Every visual and musical transition emerges purely from **full-rank latent tensor transformations** within the active concept manifold.
3. **True Multimodal Heterarchy (No Master–Slave Modality Subordination):** Vision does not command audition, and audition does not command vision. Instead, a unified **Amodal Working Memory Engine (`neuro_amodal_engine.py`)** computes the cortical state once in GPU VRAM, and both the visual (`CLIP [77×768]`) and auditory (`T5 [64×768]`) generators project that shared amodal state into their respective sensory domains.

---

## 2. 100% VRAM-Resident Amodal Cognitive Engine & Cross-Python CUDA IPC

### 2.1 Zero-CPU Compute Law & Scaling from 16 (120 Dipoles) to 1,024 Channels (523,776 Dipoles)

To scale seamlessly from **16 channels** ($\binom{16}{2} = 120$ dipoles) and **24 channels** ($\binom{24}{2} = 276$ dipoles) up to **1,024-channel uHD EEG grids** ($\binom{1024}{2} = 523,776$ dipoles $\times 32$ phase slots $= 16,760,832$ phase modes per frame) [[23](#ref-23)], **all computations after raw LSL sample ingestion are executed 100% inside GPU VRAM (`neuro_amodal_engine.py`)**:

* **Direct LSL-to-VRAM Spectral Frontend (`DirectGPUSpectralFrontend`):** Raw 16-channel ADC samples enter `raw_buf_gpu` directly. 50/100 Hz notch filtering, 1,024-point slow FFT frequency tracking, 32-band gamma and 89.5 Hz ripple Hilbert transforms, $ci\text{PLV}$ dipole matrices, 4,096-column Monty `topk` competitions, SVD affordances, and `GPUSealEngine` slot seals run entirely on GPU tensor cores with **zero `.cpu().numpy()` roundtrips**.
* **Isomorphic 768-D Cortical Projection:** Batched dipole geometry (`build_scalable_dipole_geometry_gpu`) projects $P$ dipoles into the $768\text{-D}$ cortical column space via a single tensor-core matrix multiplication $\mathbf{Z}_{32 \times 768} = \mathbf{\Psi}_{32 \times P}\mathbf{\Pi}_{P \times 768}$, preventing VRAM explosion even at $P = 523,776$.

### 2.2 Amodal Working Memory Bus (`HardwareVRAMBus`) via Native PyTorch `_share_cuda_`

```text
 ═══════════════════════════════════════════════════════════════════════════
                 SHARED GPU VRAM (NATIVE PYTORCH _share_cuda_ IPC)
 ═══════════════════════════════════════════════════════════════════════════

  [RAW LSL EEG STREAM] ──► DirectGPUSpectralFrontend (raw_buf_gpu)
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. AMODAL COGNITIVE ENGINE (neuro_amodal_engine.py — 100% GPU VRAM)    │
  │                                                                        │
  │ • 32×P ciPLV & 89.5 Hz Dickey Ripples (Zero CPU copies)                │
  │ • 5 Monty SDR Columns (Fpz, AFz, F3, F4, FCz) × 4,096 minicolumns      │
  │ • 2D SWM Geometry (x_local, y_global, Kuramoto R) & 4D Kinematics      │
  │ • Gram-Schmidt Compositional Ripple Tree (A ⊃ B / A ∥ B)               │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                 HardwareVRAMBus (UntypedStorage._share_cuda_)
                 [77×768 Amodal Manifold | 16×768 Slots | 16×16 DAG]
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │ (Zero-Copy VRAM Pointer)                                  │ (Zero-Copy VRAM Pointer)
        ▼                                                           ▼
  ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
  │ 2. VISION RUNNER (Python 3.10)    │       │ 3. MUSICGEN RUNNER (Python 3.9)   │
  │ • Maps shared VRAM tensor         │       │ • Maps the SAME shared VRAM tensor│
  │ • Projects to CLIP [1 × 77 × 768] │       │ • Lockstep execution with server  │
  │ • 100% GPU Color Surgery & Blend  │       │ • Projects to T5 [1 × 64 × 768]   │
  └───────────────────────────────────┘       └───────────────────────────────────┘
 ═══════════════════════════════════════════════════════════════════════════
```

* **Cross-Python Zero-Copy VRAM Sharing:** `HardwareVRAMBus` exports the physical GPU storage handle via PyTorch's native C++ allocator (`UntypedStorage._share_cuda_()` and `UntypedStorage._new_shared_cuda()`). Both the Python 3.10 Diffusion process and the Python 3.9 MusicGen process map the **exact same physical VRAM address** without `pickle` incompatibilities and with **0 bytes transferred over PCIe or CPU RAM**.
* **Auto-Host Election:** Whichever runner starts first automatically initializes the LSL-to-VRAM cognitive engine and publishes the CUDA IPC handle; subsequent runners attach as zero-compute VRAM readers.

### 2.3 100% Bit-for-Bit Deterministic EEG $\to$ Video & Audio Homomorphism (`SHA-256` Verified)

By default (`deterministic=True`), **all pseudo-random number generators (`torch.randn`, `.uniform_()`) are eliminated from the generative loop** so that feeding the exact same EEG recording (`eeg_raw.npz`) always reproduces the **exact same video and audio bit-for-bit**:

1. **Deterministic Full-Rank Token Selection in MusicGen (`musicgen_server.py`):** Instead of drawing random uniform noise `u.uniform_()` for Gumbel-Max sampling, each of the $K=4$ EnCodec codebooks at autoregressive step `offset` extracts its own $k_{\text{eff}}$-dimensional slice directly from the full-dimensional EEG cross-attention tensor $\mathbf{C}_{\text{EEG}} \in \mathbb{R}^{B \times 64 \times 1024}$:

$$
\mathbf{U}_{B \times 4 \times k_{\text{eff}}} = \sigma\Big(\text{reshape}\big(\mathbf{C}_{\text{EEG}}\big[:, \, \text{offset} \bmod 64, \, :4 k_{\text{eff}}\big], \, [B, 4, k_{\text{eff}}]\big)\Big), \qquad \mathbf{g} = \mathbf{L}_{\text{top}} - \log\left(-\log(\mathbf{U})\right)
$$

2. **Deterministic U-Net Latent Evolution (`render_logic.py`):** Locks `torch.backends.cudnn.deterministic = True` and binds a deterministic `torch.Generator(device="cuda").manual_seed(0)` on every step, turning the diffusion U-Net into a pure mathematical function $\mathbf{I}_{t+1} = \mathcal{D}(\mathbf{I}_t, \mathbf{E}_{\text{EEG}}(t))$.
3. **Cryptographic Verification (`neuro_deterministic_reproducer.py`):** Replays any recorded `eeg_raw.npz` twice through the full `[16×256] ➔ [32×120] ➔ [77×768] / [64×768]` pipeline and verifies `SHA-256` hash equality for both `.mp4` and `.wav` outputs. *(Classic stochastic sampling remains available at any time via the `--stochastic` CLI flag).*

---

## 3. Biophysical Foundations & Neocortical Circuit Theory

### 3.1 Macrocolumnar Scale: FreeEEG16 (120 Dipoles) & FreeEEG24 (276 Dipoles) on 26 mm

The physical interface uses the **FreeEEG16-alpha2** (16 active channels + local `AINREF` + `GND`, $5.38\text{ mm}$ pitch) and **FreeEEG24-alpha** (24 active channels + local `AINREF` + `GND` across three synchronized `ADS131M08` ADCs, $4.20\text{ mm}$ pitch) circular sensor arrays ($26.0\text{ mm}$ diameter, $\text{Area} \approx 531\text{ mm}^2$):

* Following Vernon Mountcastle [[3](#ref-3)] and Hawkins et al. [[1](#ref-1), [2](#ref-2)], a cortical macrocolumn spans $300\text{--}600\ \mu\text{m}$ in diameter. Underneath a single $26\text{ mm}$ sensor reside **1,900 to 4,200 macrocolumns** ($>50,000,000$ neurons).
* **16 channels** yield $\binom{16}{2} = 120$ unique bipolar dipoles ($32 \times 120 = 3,840$ continuous phase modes per frame).
* **24 channels** yield $\binom{24}{2} = 276$ unique bipolar dipoles ($32 \times 276 = 8,832$ continuous phase modes per frame), partitioning each $26\text{ mm}$ patch into Micro ($4.2\text{--}9.5\text{ mm}$), Meso ($9.5\text{--}16.5\text{ mm}$), and Macro ($16.5\text{--}26.0\text{ mm}$) spatial scales.

### 3.2 The Zero-Power Principle: Volume-Conduction-Free Phase Coherence ($ci\text{PLV}$)

All analytic signals are normalized to unit modulus ($z_i(t) = \exp(i\varphi_i(t))$) to compute the **Corrected Imaginary Phase-Locking Value ($ci\text{PLV}$)** [[10](#ref-10), [11](#ref-11)]:

$$
ci\text{PLV}_{i,j} = \frac{\frac{1}{T} \sum_{t=1}^{T} \Im\left(z_i(t) z_j^*(t)\right)}{\sqrt{1 - \left(\frac{1}{T} \sum_{t=1}^{T} \Re\left(z_i(t) z_j^*(t)\right)\right)^2}}
$$

### 3.3 Inter-Region Monty SDR Heterarchy (L3➔L4 Anomaly Up, L6a➔L1 Prediction Down)

Grounded in *Hawkins et al. (2017; 2025)* [[1](#ref-1), [2](#ref-2)] and *Ding et al. (2016)* [[8](#ref-8)], the engine maintains five coupled `AmodalMontyColumnGPU` columns with nested temporal receptive windows ($\tau_{\text{FCz}} = 0.04\text{ s}$, $\tau_{\text{F3/F4}} = 0.35\text{ s}$, $\tau_{\text{AFz}} = 1.20\text{ s}$, $\tau_{\text{Fpz}} = 5.00\text{ s}$):

* **Top-Down Apical Prediction (`L6a -> L1`):** $\widehat{\mathbf{s}}^{(r)} = \mathbf{W}_{\text{topdown}}^{(r+1 \to r)} \mathbf{s}_{\text{L2/3}}^{(r+1)}$
* **Bottom-Up Unpredicted Anomaly (`L3 -> L4`):** $\mathbf{a}_{\text{up}}^{(r)} = \text{ReLU}\left(\mathbf{s}_{\text{inst}}^{(r)} - \widehat{\mathbf{s}}^{(r)}\right)$

### 3.4 Lossless Tensor Homomorphism & Multi-Architecture Vocabulary Bridge

| Cortical Slice | Region & Mathematical Tensor ($N \times 768$) |
| :--- | :--- |
| **`Tokens [1..4]`** | **`Fpz` (Epoch Wheel):** Macro-dipole horizon & smooth crossfade across rule sets. |
| **`Tokens [5..20]`** | **`AFz` (Compositional DAG $A \supset B$):** 16 Delta-Theta sectors orthogonalized via Gram-Schmidt over 89.5 Hz ripples. |
| **`Tokens [21..36]`** | **`F3` (Structural ON-Subspace):** $16 \times 768$ positive phase-lag modes $\text{ReLU}(+\mathbf{\Psi}) \mathbf{\Pi}_{P \to 768}$. |
| **`Tokens [37..52]`** | **`F4` (Contextual OFF-Subspace):** $16 \times 768$ negative modes $\text{ReLU}(-\mathbf{\Psi})$ + Temporal Asymmetry $\mathbf{\Delta}_{\text{temp}}$ [[12](#ref-12), [13](#ref-13)]. |
| **`Tokens [53..68]`** | **`FCz` (Intrinsic Semantic Actions):** $16 \times 768$ trajectory integrated along the SVD tangent basis $\mathbf{V}_{\text{aff}}$. |
| **`Tokens [69..75]`** | **Global L2/3 Monty SDR Consensus:** $7 \times 768$ projection of the 4,096-column HTM state. |

### 3.5 Pure Semantic Actions via Intrinsic SVD Affordances (Zero Pixel Shifting)

An action on a concept $\mathbf{C}_0$ is projected onto its orthonormal tangent basis $\mathbf{V}_{\text{aff}} \in \mathbb{R}^{16 \times 768}$ extracted via Singular Value Decomposition (SVD) of its internal token manifold and active lore neighbors:

$$
\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}^T = \text{SVD}\left(\mathbf{M}_{\text{combined}}\right), \qquad \mathbf{V}_{\text{aff}} = \mathbf{V}_{1:16}
$$

### 3.6 89.5 Hz Cortical Ripple Causal DAGs ($A \supset B$ vs $A \parallel B$)

Following *Dickey et al. (2022, PNAS)* [[9](#ref-9)], phase-lead matrices in the $70\text{--}100\text{ Hz}$ human ripple band ($\sim 89.5\text{ Hz}$) govern hierarchical nesting via vectorized GPU Gram-Schmidt orthogonalization:

$$
\mathbf{E}_B^{\perp} = \mathbf{E}_B - \frac{\langle \mathbf{E}_B, \mathbf{E}_A \rangle}{\left\Vert \mathbf{E}_A \right\Vert_2^2 + \epsilon} \mathbf{E}_A
$$

---

## 4. Zero-Buffer Real-Time Generative Engines (Vision & Audio)

### 4.1 Continuous Diagonal Stream MusicGen: Fused Codebook Kernels, Auto-Min Block ($0.02\text{ s}$) & Server-Side 32 kHz Clock

* **Ultra-Optimized Transformer Kernels (`prepare_lm_fast_weights`):** Fuses all 4 codebook embedding tables into a single batched `F.embedding` call (`_fused_emb_weight`), pre-caches contiguous cross-attention `Q/K/V` weights, and eliminates redundant `LowerTriangularMask` allocations, achieving **`17.7 ms/step` (`1.09x` RTF)** even on an RTX 3060.
* **Lockstep Cognitive-to-Audio Pipeline (`neuro_heterarchy_musicgen.py`):** Executes the working memory projection synchronously right before each `generate_step` call so that 0% of GPU compute is stolen while the 24-layer MusicGen transformer is running, while automatically updating working memory up to **50 times per second ($0.02\text{ s}$ blocks)** on RTX 4090/5090.
* **Server-Side 32,000 Hz Playout Clock (`server_realtime_audio_clock`):** The engine buffer belongs strictly to the server and is drained at exact real-time speed ($40\text{ ms}$ ticks) regardless of whether browser audio is muted or active. The browser (`static/app.js`) performs zero audio slicing or time-warping.

### 4.2 Zero-Buffer Server-Push Diffusion: 100% GPU Color Surgery, Auto-Steps & Latest-Frame RAM Mailbox

* **100% GPU Color Surgery & Frame Blending (`apply_color_surgery_and_blend_gpu`):** Performs TAESD green-drift correction and `world_context` temporal blending directly on GPU tensors (`torch.float32` on `cuda`), eliminating CPU OpenCV bottlenecks.
* **Continuous Server-Push + Latest-Frame RAM Mailbox:** `web_cloud_gateway.py` pushes frames over `/stream.mjpg` with `TCP_NODELAY`, while `static/app.js` overwrites stale frames in a single-slot RAM mailbox (`0` buffered frames).

### 4.3 On-the-Fly Model Hot-Swapping

* **Vision Models:** `SimianLuo/LCM_Dreamshaper_v7` (SD 1.5), `stabilityai/sd-turbo` (SD 2.1), `stabilityai/sdxl-turbo` (SDXL 1.0).
* **Audio Models:** `facebook/musicgen-small` (300M), `facebook/musicgen-medium` (1.5B), `facebook/musicgen-melody` (1.5B), `facebook/musicgen-large` (3.3B).

---

## 5. The 4-Stage Cortical Calibration Ladder (`FCz` ➔ `F3/F4` ➔ `AFz` ➔ `Fpz`)

* **Stage 1 (`FCz` Semantic Action Focus — 20s Calibration):** Clamps `Fpz`, `AFz`, `F3`, and `F4` while the user steers `FCz` along the 16 intrinsic SVD affordance axes $\mathbf{V}_{\text{aff}}$.
* **Stage 2 (`+F3/F4` Structural Form & Contextual Style):** Unlocks the $16 \times 768$ ON-structural (`F3`) and OFF-contextual (`F4`) subspaces.
* **Stage 3 (`+AFz` Compositional Ripple Trees $A \supset B$):** Unlocks full multi-concept Gram-Schmidt composition across Delta-Theta sectors.
* **Stage 4 (`+Fpz` Full Heterarchy & Epoch Wheel):** Unlocks all 5 regions.

---

## 6. Hardware Signal Chain, Auto-Reconnecting LSL Recorder & Local Docker Runner

* **Auto-Reconnecting Multimodal Session Recorder (`neuro_recorder.py`):** Continuously polls `resolve_streams()` every $0.5\text{ s}$ with a $360\text{ s}$ LSL buffer (`max_buflen=360`), capturing 16-channel `eeg_raw.npz`, 1:1 hardware-locked 30 FPS `video_timeline.mp4`, 32 kHz `music_timeline.wav`, and `events.csv`.
* **Local All-in-One Docker Testing (`docker-compose.all.yml` / `run_local_all.sh`):** Runs the unified `neuroidss/neuro_heterarchy_all:latest` container locally with host IPC and shared GPU VRAM.
* **Deterministic Verification (`neuro_deterministic_reproducer.py`):**
  ```bash
  python3 neuro_deterministic_reproducer.py --eeg sessions/latest/eeg_raw.npz --steps 8
  ```

---

## 7. REST API & AI Agent Integration

* `GET  /api/world/concepts` & `GET /api/music/concepts` — List active visual and musical concepts.
* `POST /api/world/concepts/add` & `POST /api/world/concepts/remove` — Dynamically add/remove visual concepts.
* `POST /api/music/concepts/add` & `POST /api/music/concepts/remove` — Dynamically add/remove musical themes.
* `POST /api/diffusion/config` — Hot-swap diffusion model (`mode`), `resolution`, `use_taesd`, `steps`, `world_context`, and `ema_smooth`.
* `POST /api/music_config` — Hot-swap MusicGen model (`model`), `target_block` (`0` = Auto-Min), `target_context`, and `target_buffer_count`.
* `POST /api/recording/start` & `POST /api/recording/stop` — Start/stop 1:1 multimodal dataset recording.

---

## 8. Comprehensive Scientific Bibliography & DOIs

1. <a id="ref-1"></a>**Hawkins, J., Leadholm, N., & Clay, V. (2025).** *Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain (The Thousand Brains Theory 2.0).* arXiv:2507.05888 [q-bio.NC]. [https://doi.org/10.48550/arXiv.2507.05888](https://doi.org/10.48550/arXiv.2507.05888)
2. <a id="ref-2"></a>**Hawkins, J., Ahmad, S., & Cui, Y. (2017).** *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World.* Frontiers in Neural Circuits, 11:81. [https://doi.org/10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)
3. <a id="ref-3"></a>**Mountcastle, V. B. (1997).** *The columnar organization of the neocortex.* Brain, 120(4), 701–722. [https://doi.org/10.1093/brain/120.4.701](https://doi.org/10.1093/brain/120.4.701)
4. <a id="ref-4"></a>**Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [https://doi.org/10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
5. <a id="ref-5"></a>**Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** *Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory.* Proceedings of the National Academy of Sciences (PNAS), 115(5), 1117–1122. [https://doi.org/10.1073/pnas.1717766115](https://doi.org/10.1073/pnas.1717766115)
6. <a id="ref-6"></a>**Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [https://doi.org/10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
7. <a id="ref-7"></a>**Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [https://doi.org/10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
8. <a id="ref-8"></a>**Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** *Cortical tracking of hierarchical linguistic structures in connected speech.* Nature Neuroscience, 19(1), 158–164. [https://doi.org/10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
9. <a id="ref-9"></a>**Dickey, C. W., Verzhbinsky, I. A., Jiang, X., Rosen, B. Q., Kajfez, S., Stedelin, B., Shih, J. J., Ben-Haim, S., Raslan, A. M., Eskandar, E. N., Gonzalez-Martinez, J., Cash, S. S., & Halgren, E. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* Proceedings of the National Academy of Sciences (PNAS), 119(28), e2107797119. [https://doi.org/10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
10. <a id="ref-10"></a>**Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011. [https://doi.org/10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
11. <a id="ref-11"></a>**Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).** *Identifying true brain interaction from EEG data using the imaginary part of coherency.* Clinical Neurophysiology, 115(10), 2292–2307. [https://doi.org/10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
12. <a id="ref-12"></a>**Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., Moser, M.-B., & Moser, E. I. (2009).** *Frequency of gamma oscillations routes flow of information in the hippocampus.* Nature, 462(7271), 353–357. [https://doi.org/10.1038/nature08573](https://doi.org/10.1038/nature08573)
13. <a id="ref-13"></a>**Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** *Slow and fast gamma rhythms coordinate different spatial coding modes in hippocampal place cells.* Neuron, 82(3), 670–681. [https://doi.org/10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
14. <a id="ref-14"></a>**Lisman, J. E., & Jensen, O. (2013).** *The theta-gamma neural code.* Neuron, 77(6), 1002–1016. [https://doi.org/10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
15. <a id="ref-15"></a>**Fries, P. (2015).** *Rhythms for cognition: communication through coherence.* Neuron, 88(1), 220–235. [https://doi.org/10.1016/j.neuron.2015.09.034](https://doi.org/10.1016/j.neuron.2015.09.034)
16. <a id="ref-16"></a>**Churchland, M. M., & Shenoy, K. V. (2024).** *Preparatory activity and the expansive null-space.* Nature Reviews Neuroscience, 25, 213–236. [https://doi.org/10.1038/s41583-024-00796-z](https://doi.org/10.1038/s41583-024-00796-z)
17. <a id="ref-17"></a>**Zimnik, A. J., & Churchland, M. M. (2021).** *Independent generation of sequence elements by motor cortex.* Nature Neuroscience, 24, 412–424. [https://doi.org/10.1038/s41593-021-00798-5](https://doi.org/10.1038/s41593-021-00798-5)
18. <a id="ref-18"></a>**Weber, J., Iwama, G., Solbakk, A.-K., Blenkmann, A. O., Larsson, P. G., Ivanovic, J., Knight, R. T., Endestad, T., & Helfrich, R. (2023).** *Subspace partitioning in the human prefrontal cortex resolves cognitive interference.* Proceedings of the National Academy of Sciences (PNAS), 120(28), e2220523120. [https://doi.org/10.1073/pnas.2220523120](https://doi.org/10.1073/pnas.2220523120)
19. <a id="ref-19"></a>**Flesch, T., Juechems, K., Dumbalska, T., Saxe, A., & Summerfield, C. (2022).** *Orthogonal representations for robust context-dependent task performance in brains and neural networks.* Neuron, 110(7), 1258–1270. [https://doi.org/10.1016/j.neuron.2022.01.005](https://doi.org/10.1016/j.neuron.2022.01.005)
20. <a id="ref-20"></a>**Panichello, M. F., & Buschman, T. J. (2021).** *Shared mechanisms underlie the control of working memory and attention.* Nature, 592, 601–605. [https://doi.org/10.1038/s41586-021-03390-w](https://doi.org/10.1038/s41586-021-03390-w)
21. <a id="ref-21"></a>**Koechlin, E., & Hyafil, A. (2007).** *Anterior prefrontal function and the limits of human decision-making.* Science, 318(5850), 594–598. [https://doi.org/10.1126/science.1142995](https://doi.org/10.1126/science.1142995)
22. <a id="ref-22"></a>**Boorman, E. D., Behrens, T. E. J., Woolrich, M. W., & Rushworth, M. F. S. (2009).** *How green is the grass on the other side? Frontopolar cortex and evidence for the reliability of alternative courses of action.* Neuron, 62(5), 733–743. [https://doi.org/10.1016/j.neuron.2009.05.014](https://doi.org/10.1016/j.neuron.2009.05.014)
23. <a id="ref-23"></a>**Lee, H. S., Schreiner, L., Jo, S.-H., Sieghartsleitner, S., Jordan, M., Pretl, H., Guger, C., & Park, H.-S. (2022).** *Individual finger movement decoding using a novel ultra-high-density electroencephalography-based brain-computer interface system.* Frontiers in Neuroscience, 16:1009878. [https://doi.org/10.3389/fnins.2022.1009878](https://doi.org/10.3389/fnins.2022.1009878)
24. <a id="ref-24"></a>**Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y., & Défossez, A. (2023).** *Simple and Controllable Music Generation (MusicGen).* Advances in Neural Information Processing Systems (NeurIPS), 36. [https://doi.org/10.48550/arXiv.2306.05284](https://doi.org/10.48550/arXiv.2306.05284)
25. <a id="ref-25"></a>**Luo, S., Tan, Y., Huang, L., Li, J., & Zhao, H. (2023).** *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference.* arXiv:2310.04378 [cs.CV]. [https://doi.org/10.48550/arXiv.2310.04378](https://doi.org/10.48550/arXiv.2310.04378)

