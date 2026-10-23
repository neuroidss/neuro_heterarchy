# 🧠 NeuroCanvas × TBP.Monty

### Full-Dimensional Neocortical SDR Heterarchy BCI, Pure Phase-Coherence Synthesizer ($ci\text{PLV}$), 89.5 Hz Ripple Causal DAGs, Intrinsic SVD Semantic Affordances, and Zero-Buffer Real-Time Closed-Loop Diffusion & Audio

[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1016/j.neuron.2014.03.013](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2014.03.013-orange.svg)](https://doi.org/10.1016/j.neuron.2014.03.013)
[![DOI:10.1038/nature08573](https://img.shields.io/badge/DOI-10.1038%2Fnature08573-blue.svg)](https://doi.org/10.1038/nature08573)
[![arXiv:2507.05888](https://img.shields.io/badge/arXiv-2507.05888-b31b1b.svg)](https://arxiv.org/abs/2507.05888)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-yellow.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.1038/nn.4186](https://img.shields.io/badge/DOI-10.1038%2Fnn.4186-teal.svg)](https://doi.org/10.1038/nn.4186)
[![Deploy to Cloudflare Workers](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/dmitry-sukhoruchkin/runpod-cors-proxy)

---

## 📑 Table of Contents
1. [System Overview & Scientific Paradigm Shift](#1-system-overview--scientific-paradigm-shift)
2. [Biophysical Foundations & Neocortical Circuit Theory](#2-biophysical-foundations--neocortical-circuit-theory)
   - 2.1 [Macrocolumnar Scale: FreeEEG16 (120 Dipoles) & FreeEEG24 (276 Dipoles) on 26 mm](#21-macrocolumnar-scale-freeeeg16-120-dipoles--freeeeg24-276-dipoles-on-26-mm)
   - 2.2 [The Zero-Power Principle: Volume-Conduction-Free Phase Coherence ($ci\text{PLV}$)](#22-the-zero-power-principle-volume-conduction-free-phase-coherence-ciplv)
   - 2.3 [Inter-Region Monty SDR Heterarchy (L3➔L4 Anomaly Up, L6a➔L1 Prediction Down)](#23-inter-region-monty-sdr-heterarchy-l3l4-anomaly-up-l6al1-prediction-down)
   - 2.4 [Lossless Tensor Homomorphism & Multi-Architecture Vocabulary Bridge](#24-lossless-tensor-homomorphism--multi-architecture-vocabulary-bridge)
   - 2.5 [Pure Semantic Actions via Intrinsic SVD Affordances (Zero Pixel Shifting)](#25-pure-semantic-actions-via-intrinsic-svd-affordances-zero-pixel-shifting)
   - 2.6 [89.5 Hz Cortical Ripple Causal DAGs ($A \supset B$ vs $A \parallel B$)](#26-895-hz-cortical-ripple-causal-dags-a--b-vs-a--b)
   - 2.7 [Lyapunov-Stable NLMS Learning, Persistence Boost & Smooth Epoch Wheel](#27-lyapunov-stable-nlms-learning-persistence-boost--smooth-epoch-wheel)
3. [Zero-Buffer Real-Time Generative Engines (Vision & Audio)](#3-zero-buffer-real-time-generative-engines-vision--audio)
   - 3.1 [Continuous Diagonal Stream MusicGen: Auto-Minimum Block ($0.02\text{ s}$) & Adaptive Slowdown](#31-continuous-diagonal-stream-musicgen-auto-minimum-block-002-s--adaptive-slowdown)
   - 3.2 [Zero-Buffer Server-Push Diffusion: Latest-Frame RAM Mailbox, Auto-Steps & TAESD Link](#32-zero-buffer-server-push-diffusion-latest-frame-ram-mailbox-auto-steps--taesd-link)
   - 3.3 [On-the-Fly Model Hot-Swapping (LCM / SD-Turbo / SDXL-Turbo & MusicGen Small / Medium / Melody / Large)](#33-on-the-fly-model-hot-swapping-lcm--sd-turbo--sdxl-turbo--musicgen-small--medium--melody--large)
4. [The 4-Stage Cortical Calibration Ladder (`FCz` ➔ `F3/F4` ➔ `AFz` ➔ `Fpz`)](#4-the-4-stage-cortical-calibration-ladder-fcz--f3f4--afz--fpz)
5. [Hardware Signal Chain & Transport Integrity](#5-hardware-signal-chain--transport-integrity)
6. [Serverless GitHub Pages Controller & Cloud Deployment](#6-serverless-github-pages-controller--cloud-deployment)
7. [Multimodal 1:1 Synchronized Session Recorder](#7-multimodal-11-synchronized-session-recorder)
8. [REST API & AI Agent Integration](#8-rest-api--ai-agent-integration)
9. [Comprehensive Scientific Bibliography & DOIs](#9-comprehensive-scientific-bibliography--dois)

---

## 1. System Overview & Scientific Paradigm Shift

Conventional Brain-Computer Interfaces (BCIs) reduce high-dimensional cortical neurodynamics to discrete classification labels or scalar 1D band-power estimates ($\mu\text{V}^2/\text{Hz}$). In contemplative and neurofeedback applications, traditional systems focus almost exclusively on **Stage 1 (Sensory Dissolution)** via alpha-band amplitude coherence—a state of top-down sensory inhibition that frequently leads to drowsiness rather than active mental synthesis.

**NeuroCanvas × TBP.Monty** is engineered for **Stage 2 (Generation Stage / Waking Lucid Synthesis)**, grounded in the **Thousand Brains Theory 2.0** [[1](#ref-1), [2](#ref-2)], **Working Memory 2.0** [[4](#ref-4), [5](#ref-5)], and empirical multi-scale cortical neurogeometry [[6](#ref-6), [7](#ref-7), [8](#ref-8), [9](#ref-9)]:

1. **No Telepathy Myth — Closed-Loop Cortical Entrainment:** The system does not claim to "read pre-existing thoughts." Instead, it provides a **100% deterministic, high-dimensional mathematical mirror** that translates $32 \times P$ cortical phase trajectories into continuous semantic transformations on screen and in audio. Through zero-buffer closed-loop visual and auditory feedback, the neocortex rapidly adapts its own phase dynamics to steer the generative manifold (converging in ~20 seconds on sensorimotor loops).
2. **Strict Anti-Pixel-Shift & Anti-Imposed-Prompt Law:** Mechanical 2D pixel manipulations (`warpAffine`, artificial zoom/pan/rotation of rendered bitmaps) and hardcoded text injections are strictly prohibited. Every visual and musical transition emerges purely from **latent semantic transformations** within the active concept manifold.
3. **Always-On Virtual–Real Cortical Sheet:** All five canonical prefrontal/motor regions (`FCz`, `F3`, `F4`, `AFz`, `Fpz`) run continuously on the GPU as coupled **Monty SDR Columns (`PredictiveColumnMonty`)**. Whether **0, 1, 2, or 16 physical sensors** are connected, every cortical level remains active, deterministic, and responsive.

---

## 2. Biophysical Foundations & Neocortical Circuit Theory

### 2.1 Macrocolumnar Scale: FreeEEG16 (120 Dipoles) & FreeEEG24 (276 Dipoles) on 26 mm

The physical interface uses the **FreeEEG16-alpha2** (16 active channels + local `AINREF` + `GND`, $5.38\text{ mm}$ pitch) and **FreeEEG24-alpha** (24 active channels + local `AINREF` + `GND` across three synchronized `ADS131M08` ADCs, $4.20\text{ mm}$ pitch) circular sensor arrays ($26.0\text{ mm}$ diameter, $\text{Area} \approx 531\text{ mm}^2$):

* Following Vernon Mountcastle [[3](#ref-3)] and Hawkins et al. [[1](#ref-1), [2](#ref-2)], a cortical macrocolumn spans $300\text{--}600\ \mu\text{m}$ in diameter. Underneath a single $26\text{ mm}$ sensor reside **1,900 to 4,200 macrocolumns** ($>50,000,000$ neurons).
* As demonstrated in ultra-high-density EEG studies (*Lee et al., 2022* [[23](#ref-23)]), reducing inter-electrode spacing below $8.6\text{ mm}$ preserves $>56\%$ independent variance ($1 - R^2$) between neighboring sites.
* **16 channels** yield $\binom{16}{2} = 120$ unique bipolar dipoles ($32 \times 120 = 3,840$ continuous phase modes per frame).
* **24 channels** yield $\binom{24}{2} = 276$ unique bipolar dipoles ($32 \times 276 = 8,832$ continuous phase modes per frame), partitioning each $26\text{ mm}$ patch into three spatial scales:
  * **Micro-dipoles ($4.2\text{--}9.5\text{ mm}$):** Local fast gamma/ripple derivatives ("strokes / immediate actions").
  * **Meso-dipoles ($9.5\text{--}16.5\text{ mm}$):** Intermediate structural chunks ("letters / words / forms").
  * **Macro-dipoles ($16.5\text{--}26.0\text{ mm}$):** Long-axis phase gradients ("sentences / overarching rules").

### 2.2 The Zero-Power Principle: Volume-Conduction-Free Phase Coherence ($ci\text{PLV}$)

Scalar spectral power is vulnerable to impedance drift and EMG artifacts. NeuroCanvas normalizes all analytic signals to unit modulus ($z_i(t) = \exp(i\varphi_i(t))$) and computes the **Corrected Imaginary Phase-Locking Value ($ci\text{PLV}$)** [[10](#ref-10), [11](#ref-11)]:

$$
ci\text{PLV}_{i,j} = \frac{\frac{1}{T} \sum_{t=1}^{T} \Im\left(z_i(t) z_j^*(t)\right)}{\sqrt{1 - \left(\frac{1}{T} \sum_{t=1}^{T} \Re\left(z_i(t) z_j^*(t)\right)\right)^2}}
$$

Instantaneous zero-lag volume conduction falls strictly on the real axis ($\Im = 0$) and vanishes identically.

### 2.3 Inter-Region Monty SDR Heterarchy (L3➔L4 Anomaly Up, L6a➔L1 Prediction Down)

Grounded in *Hawkins et al. (2017; 2025)* [[1](#ref-1), [2](#ref-2)] and *Ding et al. (2016)* [[8](#ref-8)], the engine maintains five coupled `PredictiveColumnMonty` columns with nested temporal receptive windows ($\tau_r$):

```text
==========================================================================================
       INTER-REGION MONTY SDR HETERARCHY (BIDIRECTIONAL SDR PREDICTION & ANOMALY)
==========================================================================================

  [LEVEL 4: Fpz (Meta-Rules / Smooth Epoch Wheel)] ── τ = 5.00 s (Macro-Horizon)
       ▲  │
       │  │ L6a->L1 Top-Down SDR Prediction (W_topdown_sdr · s_Fpz)
       │  ▼
  [LEVEL 3: AFz (Compositional Ripple DAG A ⊃ B)]  ── τ = 1.20 s (Delta Scale ~1 Hz)
       ▲  │
       │  │ L6a->L1 Top-Down SDR Prediction (W_topdown_sdr · s_AFz)
       │  ▼
  [LEVEL 2: F3 (ON-Form) ◄─L2/3─► F4 (OFF-Context)] ─ τ = 0.35 s (Theta Scale ~4 Hz)
       ▲  │
       │  │ L6a->L1 Top-Down SDR Prediction (W_topdown_sdr · s_F3/F4)
       │  ▼
  [LEVEL 1: FCz (Intrinsic SVD Semantic Actions)]  ── τ = 0.04 s (Gamma Scale, 60 FPS)
==========================================================================================
```

* **Dual-Channel (ON/OFF) L4 HTM Column:** Splits bipolar phase matrices $\mathbf{\Psi} \in \mathbb{R}^{32 \times P}$ into $\mathbf{x}^+ = \text{ReLU}(\mathbf{\Psi})$ and $\mathbf{x}^- = \text{ReLU}(-\mathbf{\Psi})$, activating $k = 80$ winners out of $4,096$ minicolumns without collapsing $360^\circ$ phase directionality.
* **Top-Down Apical Prediction (`L6a -> L1`):** Slower region $r+1$ predicts the expected SDR of faster region $r$:

$$
\widehat{\mathbf{s}}^{(r)} = \mathbf{W}_{\text{topdown}}^{(r+1 \to r)} \mathbf{s}_{\text{L2/3}}^{(r+1)}
$$

* **Bottom-Up Unpredicted Anomaly (`L3 -> L4`):** Activity in region $r$ that violates the top-down prediction generates an upward SDR anomaly:

$$
\mathbf{a}_{\text{up}}^{(r)} = \text{ReLU}\left(\mathbf{s}_{\text{inst}}^{(r)} - \widehat{\mathbf{s}}^{(r)}\right)
$$

### 2.4 Lossless Tensor Homomorphism & Multi-Architecture Vocabulary Bridge

Instead of collapsing 32 gamma phase slots and 120 dipoles into 2D scalars, the engine projects the full cortical state directly into the $77 \times 768$ CLIP conditioning tensor ($\mathbf{E}_{768} \in \mathbb{R}^{77 \times 768}$) and $64 \times 768$ T5 conditioning tensor ($\mathbf{E}_{\text{T5}} \in \mathbb{R}^{64 \times 768}$):

* **768-D Sagitta Tensor ($\mathbf{\Sigma}_{\text{sagitta}} \in \mathbb{R}^{16 \times 768}$):** Orthogonal deviation of intermediate theta-phase slots from the downbeat-to-upbeat chord $\mathbf{Z}_{31} - \mathbf{Z}_0$ in 768-D space.
* **768-D Temporal Asymmetry Tensor ($\mathbf{\Delta}_{\text{temp}} = \mathbf{Z}_{16:32} - \mathbf{Z}_{0:16} \in \mathbb{R}^{16 \times 768}$):** Prospective fast-gamma future minus retrospective slow-gamma past (*Colgin et al., 2009* [[12](#ref-12)]; *Bieri et al., 2014* [[13](#ref-13)]).
* **Least-Squares BPE Vocabulary Isomorphism (`LCM` $\leftrightarrow$ `SD-Turbo` $\leftrightarrow$ `SDXL-Turbo`):** Because `CLIP-ViT-L` ($768\text{-D}$), `OpenCLIP-ViT-H` ($1024\text{-D}$ in `SD-Turbo`), and `OpenCLIP-ViT-bigG` ($1280\text{-D}$ in `SDXL-Turbo`) share the exact 49,408 BPE token vocabulary $\mathbf{V}_{\text{BPE}}$, `NeuroRender` solves the linear bridge across token embedding tables at model load time:

$$
\mathbf{W}_{768 \to D} = \arg\min_{\mathbf{W}} \left\Vert \mathbf{V}_{768} \mathbf{W} - \mathbf{V}_{D} \right\Vert_F^2
$$

For `SDXL-Turbo`, `NeuroRender` concatenates $\mathbf{E}_{2048} = \left[\mathbf{E}_{768} \,\Vert\, \mathbf{E}_{768}\mathbf{W}_{768 \to 1280}\right] \in \mathbb{R}^{1 \times 77 \times 2048}$ along with the projected pooled vector $\mathbf{p}_{1280} \in \mathbb{R}^{1 \times 1280}$, enabling instant switching between SD 1.5, SD 2.1, and SDXL architectures without altering the 60 FPS cortical engine.

| CLIP Token Slice | Cortical Source & Mathematical Tensor ($N \times 768$) |
| :--- | :--- |
| **`Token [0]`** | `<startoftext>` BOS norm anchor. |
| **`Tokens [1..4]`** | **`Fpz` (Epoch Wheel):** Smooth sinusoidal crossfade between adjacent rule sets driven by upward surprise. |
| **`Tokens [5..20]`** | **`AFz` (Compositional DAG $A \supset B$):** 16 Delta-Theta sectors orthogonalized via Gram-Schmidt over 89.5 Hz ripples. |
| **`Tokens [21..36]`** | **`F3` (Structural ON-Subspace):** $16 \times 768$ positive phase-lag modes $\text{ReLU}(+\mathbf{\Psi}) \mathbf{\Pi}_{P \to 768}$. |
| **`Tokens [37..52]`** | **`F4` (Contextual OFF-Subspace):** $16 \times 768$ negative modes $\text{ReLU}(-\mathbf{\Psi})$ + 768-D Temporal Asymmetry $\mathbf{\Delta}_{\text{temp}}$. |
| **`Tokens [53..68]`** | **`FCz` (Intrinsic Semantic Actions):** $16 \times 768$ trajectory integrated along the SVD tangent basis $\mathbf{V}_{\text{aff}}$. |
| **`Tokens [69..75]`** | **Global L2/3 Monty SDR Consensus:** $7 \times 768$ projection of the 4096-column HTM state. |
| **`Token [76]`** | `<endoftext>` EOS summary vector. |

### 2.5 Pure Semantic Actions via Intrinsic SVD Affordances (Zero Pixel Shifting)

An action on a concept $\mathbf{C}_0$ is defined strictly by the local geometry of semantic space around $\mathbf{C}_0$:

1. **Intrinsic Token Manifold (Works even for $N=1$ concept):** The centered internal token matrix $\widetilde{\mathbf{E}}_{C_0} = \mathbf{E}_{C_0}[1:76] - \boldsymbol{\mu}_{C_0} \in \mathbb{R}^{75 \times 768}$ encodes the internal semantic degrees of freedom of that concept.
2. **Neighborhood Relational Manifold ($N \ge 2$ concepts):** Difference vectors to nearest neighbors in the active lore $\Delta \mathbf{C}_k = \mathbf{C}_k - \mathbf{C}_0$ define relational transition axes.
3. **Orthonormal Action Basis ($\mathbf{V}_{\text{aff}} \in \mathbb{R}^{16 \times 768}$):** Extracted via Singular Value Decomposition (SVD) of the combined manifold:

$$
\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}^T = \text{SVD}\left(\mathbf{M}_{\text{combined}}\right), \qquad \mathbf{V}_{\text{aff}} = \mathbf{V}_{1:16}
$$

### 2.6 89.5 Hz Cortical Ripple Causal DAGs ($A \supset B$ vs $A \parallel B$)

Following *Dickey et al. (2022, PNAS)* [[9](#ref-9)], phase-lead matrices in the $70\text{--}100\text{ Hz}$ human ripple band ($\sim 89.5\text{ Hz}$) determine hierarchical nesting across slots:

* **Parent–Child Composition ($A \supset B$):** When slot $A$ phase-leads slot $B$, $B$ is projected into the orthogonal complement of $A$ via Gram-Schmidt:

$$
\mathbf{E}_B^{\perp} = \mathbf{E}_B - \frac{\langle \mathbf{E}_B, \mathbf{E}_A \rangle}{\left\Vert \mathbf{E}_A \right\Vert_2^2 + \epsilon} \mathbf{E}_A
$$

* **Co-Equal Peers ($A \parallel B$):** When phase-lead is below threshold, $A$ and $B$ partition orthogonal half-spaces.

### 2.7 Lyapunov-Stable NLMS Learning, Persistence Boost & Smooth Epoch Wheel

Each `PredictiveColumnMonty` reconstructs its $32 \times P$ phase field $\widehat{\mathbf{\Psi}} = \mathbf{W}_{\text{out}} \mathbf{s}_{\text{L2/3}}$ using contractive Normalized Least Mean Squares (NLMS):

$$
\eta_{\text{stable}} = \frac{\alpha \left(1 + 0.5 \cdot \text{surprise}\right)}{\left\Vert \mathbf{s}_{\text{L2/3}} \right\Vert_2^2 + \epsilon}, \qquad \mathbf{W}_{\text{out}} \leftarrow 0.9995\,\mathbf{W}_{\text{out}} + \eta_{\text{stable}}\left(\mathbf{e} \otimes \mathbf{s}_{\text{L2/3}}\right)
$$

---

## 3. Zero-Buffer Real-Time Generative Engines (Vision & Audio)

### 3.1 Continuous Diagonal Stream MusicGen: Auto-Minimum Block ($0.02\text{ s}$) & Adaptive Slowdown

* **Auto-Minimum Block Mode (`target_block = 0` Default):** When `target_block` is set to `0`, `musicgen_server.py` dynamically shrinks the generation block down to the physical 1-token quantum of the 50 Hz EnCodec RVQ (**$1\text{ frame} = 0.02\text{ s} = 20\text{ ms} = 640\text{ samples}$**), burning 100% of available GPU compute on maximum real-time responsiveness while holding the buffer strictly at `1..target_buffer_count` blocks. If the buffer empties (`0` blocks), the server automatically increases block size to amortize context prefill overhead.
* **Adaptive Playback Slowdown for Mid-Range GPUs (`play_speed` $\le 1.0\times$):** When running a large context (e.g., $3.6\text{ s}$) and a fixed block (e.g., $1.2\text{ s}$) on a slower GPU (such as RTX 3060) where $\text{RTF} < 1.0$, the server smoothly time-stretches the generated audio prior to the 640-sample GPU crossfader:

$$
N_{\text{stretched}} = 640 + \text{round}\left(\frac{N_{\text{new}} - 640}{\text{play\_speed}}\right), \qquad \text{play\_speed} = \text{clip}\left(0.95 \cdot \text{RTF},\, 0.35,\, 1.00\right)
$$

* **Strict Buffer Block Cap:** `web_cloud_gateway.py` paces HTTP delivery to the browser (`ahead <= 0.15s`) and enforces `len(audio_pcm_queue) <= target_buffer_count + 1`, preventing runaway buffer accumulation.

### 3.2 Zero-Buffer Server-Push Diffusion: Latest-Frame RAM Mailbox, Auto-Steps & TAESD Link

* **Continuous Server-Push + Latest-Frame RAM Mailbox (Up to 60 FPS, 0 Buffered Frames):**
  * `web_cloud_gateway.py` uses a `threading.Condition` (`jpeg_cond.notify_all()`) and `TCP_NODELAY` to push each new JPEG frame down `/stream.mjpg` the microsecond it finishes rendering, draining any intermediate IPC frames (`while conn.poll(0):`).
  * `static/app.js` reads the continuous push stream via `ReadableStream`, extracts complete JPEG frames (`0xFFD8`..`0xFFD9`), and writes only the newest frame into a single-slot RAM mailbox (`latestJpegBytes`), decoding via zero-copy `Blob URL` in ~1.5 ms without ever sending per-frame HTTP requests.
* **Auto-Steps (`steps = 0` Default):** Automatically scales `num_inference_steps` between $2$ and $6$ based on measured GPU frame time (`gen_ms`).
* **Automatic `TAESD` $\leftrightarrow$ `Color Surgery` Coupling:** Enabling `TAESD` automatically enables `apply_color_surgery` (correcting TAESD green drift); switching to `Full VAE` automatically disables `apply_color_surgery` to preserve native VAE color fidelity.
* **World Context Continuity ($w_{\text{ctx}} \in [0.0, 0.60]$):** Controls frame-to-frame visual inertia in `img2img` feedback.

### 3.3 On-the-Fly Model Hot-Swapping (LCM / SD-Turbo / SDXL-Turbo & MusicGen Small / Medium / Melody / Large)

Both generative servers support live model switching from the Web Studio UI without restarting the container:
* **Vision Models:** `SimianLuo/LCM_Dreamshaper_v7` (SD 1.5), `stabilityai/sd-turbo` (SD 2.1), `stabilityai/sdxl-turbo` (SDXL 1.0).
* **Audio Models:** `facebook/musicgen-small` (300M), `facebook/musicgen-medium` (1.5B), `facebook/musicgen-melody` (1.5B), `facebook/musicgen-large` (3.3B).

---

## 4. The 4-Stage Cortical Calibration Ladder (`FCz` ➔ `F3/F4` ➔ `AFz` ➔ `Fpz`)

* **Stage 1 (`FCz` Semantic Action Focus — 20s Calibration):** Clamps `Fpz`, `AFz`, `F3`, and `F4` to an anchor concept while the user steers `FCz` along the 16 intrinsic SVD affordance axes $\mathbf{V}_{\text{aff}}$.
* **Stage 2 (`+F3/F4` Structural Form & Contextual Style):** Unlocks the $16 \times 768$ ON-structural (`F3`) and OFF-contextual (`F4`) subspaces.
* **Stage 3 (`+AFz` Compositional Ripple Trees $A \supset B$):** Unlocks full multi-concept Gram-Schmidt composition across Delta-Theta sectors.
* **Stage 4 (`+Fpz` Full Heterarchy & Epoch Wheel):** Unlocks all 5 regions, including anomaly-driven smooth crossfading across lore epochs.

---

## 5. Hardware Signal Chain & Transport Integrity

* **ADS131M08 Programmable Gain Amplifier with Readback Verification:** Hardware PGA registers (`0x04`, `0x05`) support `4x`, `8x`, `16x`, and `32x` gain with closed-loop readback confirmation:

$$
\text{uvScale} = \frac{1.2}{\text{Gain} \times 8,388,607} \times 10^6\ \mu\text{V}/\text{count}
$$

* **Real-Time 24-Bit Rail Saturation Detection:** Flags any channel where $\vert\text{count}\vert \ge 8,300,000$ ($>99\%$ full-scale rail) on both client and GPU daemon.

---

## 6. Serverless GitHub Pages Controller & Cloud Deployment

* **Zero-Backend Static Web Controller (`index.html`):** Hosted directly on GitHub Pages, communicating with the RunPod GraphQL API through a user-owned 15-line Cloudflare Worker CORS proxy (`localStorage` key storage, zero third-party key exposure).
* **Automatic Environment Routing:** Automatically selects `/opt/venv_musicgen/bin/python3` in the unified `all` image or native `python3` in single-purpose images, mounting `/workspace/hf_cache` for persistent model weights.

---

## 7. Multimodal 1:1 Synchronized Session Recorder

* **Hardware-Locked 1:1 CFR Video & Audio Timeline (`neuro_recorder.py`):** Duplicates variable-framerate diffusion frames across exact 30 FPS ticks locked to `pylsl.local_clock()` alongside 32 kHz `music_timeline.wav` and 16-channel `eeg_raw.npz`, verified via `analyze_session.py`.

---

## 8. REST API & AI Agent Integration

* `GET  /api/world/concepts` & `GET /api/music/concepts` — List active visual and musical concepts.
* `POST /api/world/concepts/add` & `POST /api/world/concepts/remove` — Dynamically add/remove visual concepts.
* `POST /api/music/concepts/add` & `POST /api/music/concepts/remove` — Dynamically add/remove musical themes.
* `POST /api/diffusion/config` — Hot-swap diffusion model (`mode`), `resolution`, `use_taesd`, `steps`, `world_context`, and `ema_smooth`.
* `POST /api/music_config` — Hot-swap MusicGen model (`model`), `target_block` (`0` = Auto-Min), `target_context`, and `target_buffer_count`.
* `POST /api/preset/ladder` — Switch calibration ladder stage `1..4`.
* `POST /api/recording/start` & `POST /api/recording/stop` — Start/stop 1:1 multimodal dataset recording.

---

## 9. Comprehensive Scientific Bibliography & DOIs

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

