# 🧠 NeuroCanvas × TBP.Monty

### Full-Dimensional Neocortical SDR Heterarchy BCI, Pure Phase-Coherence Synthesizer ($ci\text{PLV}$), 89.5 Hz Ripple Causal DAGs, Intrinsic SVD Semantic Affordances, and Real-Time Closed-Loop Diffusion & Audio

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

## 📑 Table of Contents
1. [System Overview & Scientific Paradigm Shift](#1-system-overview--scientific-paradigm-shift)
2. [Biophysical Foundations & Neocortical Circuit Theory](#2-biophysical-foundations--neocortical-circuit-theory)
   - 2.1 [Macrocolumnar Scale: FreeEEG16 (120 Dipoles) & FreeEEG24 (276 Dipoles) on 26 mm](#21-macrocolumnar-scale-freeeeg16-120-dipoles--freeeeg24-276-dipoles-on-26-mm)
   - 2.2 [The Zero-Power Principle: Volume-Conduction-Free Phase Coherence ($ci\text{PLV}$)](#22-the-zero-power-principle-volume-conduction-free-phase-coherence-ciplv)
   - 2.3 [Inter-Region Monty SDR Heterarchy (L3➔L4 Anomaly Up, L6a➔L1 Prediction Down)](#23-inter-region-monty-sdr-heterarchy-l3l4-anomaly-up-l6al1-prediction-down)
   - 2.4 [Lossless Tensor Homomorphism: $[32 \times P] \to [77 \times 768]$ (Zero 2D Bottleneck)](#24-lossless-tensor-homomorphism-32--p--77--768-zero-2d-bottleneck)
   - 2.5 [Pure Semantic Actions via Intrinsic SVD Affordances (Zero Pixel Shifting)](#25-pure-semantic-actions-via-intrinsic-svd-affordances-zero-pixel-shifting)
   - 2.6 [89.5 Hz Cortical Ripple Causal DAGs ($A \supset B$ vs $A \parallel B$)](#26-895-hz-cortical-ripple-causal-dags-a--b-vs-a--b)
   - 2.7 [Lyapunov-Stable NLMS Learning, Persistence Boost & Smooth Epoch Wheel](#27-lyapunov-stable-nlms-learning-persistence-boost--smooth-epoch-wheel)
3. [The 4-Stage Cortical Calibration Ladder (`FCz` ➔ `F3/F4` ➔ `AFz` ➔ `Fpz`)](#3-the-4-stage-cortical-calibration-ladder-fcz--f3f4--afz--fpz)
4. [Hardware Signal Chain & Transport Integrity](#4-hardware-signal-chain--transport-integrity)
   - 4.1 [ADS131M08 Programmable Gain Amplifier with Hardware Readback Verification](#41-ads131m08-programmable-gain-amplifier-with-hardware-readback-verification)
   - 4.2 [Real-Time 24-Bit Hardware Rail Saturation & Clipping Detection](#42-real-time-24-bit-hardware-rail-saturation--clipping-detection)
5. [Decoupled Cloud Architecture & Orange Pi OpenStack Orchestrator](#5-decoupled-cloud-architecture--orange-pi-openstack-orchestrator)
   - 5.1 [Orange Pi HTTPS Cloud Controller (`orangepi_cloud_controller.py`)](#51-orange-pi-https-cloud-controller-orangepi_cloud_controllerpy)
   - 5.2 [Cloud Cost Optimization Lifecycle (90 RUB/h ➔ Free Shelve / 2 RUB/h SSD ➔ 0 RUB/h)](#52-cloud-cost-optimization-lifecycle-90-rubh--free-shelve--2-rubh-ssd--0-rubh)
   - 5.3 [Zero-Build GitHub Hot-Reload & Persistent `/opt/hf_cache`](#53-zero-build-github-hot-reload--persistent-opthf_cache)
6. [Multimodal 1:1 Synchronized Session Recorder & Session Manager](#6-multimodal-11-synchronized-session-recorder--session-manager)
7. [REST API & AI Agent Integration (OpenAPI 3.0)](#7-rest-api--ai-agent-integration-openapi-30)
8. [Deployment & Operations](#8-deployment--operations)
9. [Comprehensive Scientific Bibliography & DOIs](#9-comprehensive-scientific-bibliography--dois)

---

## 1. System Overview & Scientific Paradigm Shift

Conventional Brain-Computer Interfaces (BCIs) reduce high-dimensional cortical neurodynamics to discrete classification labels or scalar 1D band-power estimates ($\mu\text{V}^2/\text{Hz}$). In meditative and contemplative applications, traditional systems focus almost exclusively on **Stage 1 (Sensory Dissolution)** via alpha-band amplitude coherence—a state of top-down sensory inhibition that frequently leads to drowsiness rather than active mental synthesis.

**NeuroCanvas × TBP.Monty** is engineered for **Stage 2 (Generation Stage / Waking Lucid Dream / Illusory Body Synthesis)**, grounded in the **Thousand Brains Theory 2.0** [1, 2], **Working Memory 2.0** [4, 5], and empirical multi-scale cortical neurogeometry [6, 7, 8, 9]:

1. **No Telepathy Myth — Closed-Loop Cortical Entrainment:** The system does not claim to "read pre-existing thoughts." Instead, it provides a **100% deterministic, high-dimensional mathematical mirror** that translates $32 \times P$ cortical phase trajectories into semantic transformations on screen and in audio. Through closed-loop visual and auditory feedback, the neocortex rapidly adapts its own phase dynamics to steer the generative manifold (converging in ~20 seconds on sensorimotor loops).
2. **Strict Anti-Pixel-Shift & Anti-Imposed-Prompt Law:** Mechanical 2D pixel manipulations (`warpAffine`, artificial zoom/pan/rotation of rendered bitmaps) and hardcoded text injections are strictly prohibited. Every visual and musical change emerges purely from **latent semantic transformations** within the active concept manifold.
3. **Always-On Virtual–Real Cortical Sheet:** All five canonical prefrontal/motor regions (`FCz`, `F3`, `F4`, `AFz`, `Fpz`) run continuously on the GPU as coupled **Monty SDR Columns (`PredictiveColumnMonty`)**. Whether **0, 1, 2, or 64 physical sensors** are connected, every cortical level remains active, deterministic, and responsive.

---

## 2. Biophysical Foundations & Neocortical Circuit Theory

### 2.1 Macrocolumnar Scale: FreeEEG16 (120 Dipoles) & FreeEEG24 (276 Dipoles) on 26 mm
The physical interface uses the **FreeEEG16-alpha2** (16 active channels + local `AINREF` + `GND`, 5.38 mm pitch) and **FreeEEG24-alpha** (24 active channels + local `AINREF` + `GND` across three synchronized `ADS131M08` ADCs, 4.20 mm pitch) circular sensor arrays ($26.0\text{ mm}$ diameter, $\text{Area} \approx 531\text{ mm}^2$):
* Following Vernon Mountcastle [3] and Hawkins et al. [1, 2], a cortical macrocolumn spans $300\text{--}600\ \mu\text{m}$ in diameter. Underneath a single 26 mm sensor reside **1,900 to 4,200 macrocolumns** (>50,000,000 neurons).
* As demonstrated in ultra-high-density EEG studies (*Lee et al., 2022* [23]), reducing inter-electrode spacing below $8.6\text{ mm}$ preserves $>56\%$ independent variance ($1 - R^2$) between neighboring sites.
* **16 channels** yield $\binom{16}{2} = \mathbf{120\text{ unique bipolar dipoles}}$ ($32 \times 120 = 3,840$ continuous phase modes per frame).
* **24 channels** yield $\binom{24}{2} = \mathbf{276\text{ unique bipolar dipoles}}$ ($32 \times 276 = 8,832$ continuous phase modes per frame), partitioning each 26 mm patch into three spatial scales:
  * **Micro-dipoles ($4.2\text{--}9.5\text{ mm}$):** Local fast gamma/ripple derivatives ("strokes / immediate actions").
  * **Meso-dipoles ($9.5\text{--}16.5\text{ mm}$):** Intermediate structural chunks ("letters / words / forms").
  * **Macro-dipoles ($16.5\text{--}26.0\text{ mm}$):** Long-axis phase gradients ("sentences / overarching rules").

### 2.2 The Zero-Power Principle: Volume-Conduction-Free Phase Coherence ($ci\text{PLV}$)
Scalar spectral power is vulnerable to impedance drift and EMG artifacts. NeuroCanvas normalizes all analytic signals to unit modulus ($z_i(t) = e^{i\varphi_i(t)}$) and computes the **Corrected Imaginary Phase-Locking Value ($ci\text{PLV}$)** [10, 11]:

$$ci\text{PLV}_{i, j} = \frac{\frac{1}{T} \sum_{t=1}^T \Im \left( z_i(t) z_j^*(t) \right)}{\sqrt{1 - \left( \frac{1}{T} \sum_{t=1}^T \Re \left( z_i(t) z_j^*(t) \right) \right)^2}}$$

Instantaneous zero-lag volume conduction falls strictly on the real axis ($\Im = 0$) and vanishes identically.

### 2.3 Inter-Region Monty SDR Heterarchy (L3➔L4 Anomaly Up, L6a➔L1 Prediction Down)
Grounded in *Hawkins et al. (2017; 2026)* [1, 2] and *Ding et al. (2016)* [8], the engine maintains five coupled `PredictiveColumnMonty` columns with nested temporal receptive windows ($\tau_r$):

```
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

* **Dual-Channel (ON/OFF) L4 HTM Column:** Splits bipolar phase matrices $\mathbf{\Psi} \in \mathbb{R}^{32 \times P}$ into $\mathbf{x}^+ = \text{ReLU}(\mathbf{\Psi})$ and $\mathbf{x}^- = \text{ReLU}(-\mathbf{\Psi})$, activating $k=80$ winners out of $4,096$ minicolumns without collapsing $360^\circ$ phase directionality.
* **Top-Down Apical Prediction (`L6a -> L1`):** Slower region $r+1$ predicts the expected SDR of faster region $r$: $\widehat{\mathbf{s}}^{(r)} = \mathbf{W}_{\text{topdown}}^{(r+1 \to r)} \mathbf{s}_{\text{L2/3}}^{(r+1)}$.
* **Bottom-Up Unpredicted Anomaly (`L3 -> L4`):** Movements in region $r$ that violate the top-down prediction generate an upward SDR anomaly $\mathbf{a}_{\text{up}}^{(r)} = \text{ReLU}(\mathbf{s}_{\text{inst}}^{(r)} - \widehat{\mathbf{s}}^{(r)})$, driving synaptic updates and epoch transitions in region $r+1$.

### 2.4 Lossless Tensor Homomorphism: $[32 \times P] \to [77 \times 768]$ (Zero 2D Bottleneck)
Instead of collapsing 32 gamma phase slots and 120/276 dipoles into 2D scalars, the engine projects the full cortical state directly into the $77 \times 768$ CLIP conditioning tensor ($\mathbf{C}_{\text{CLIP}} \in \mathbb{R}^{77 \times 768}$):
* **768-D Sagitta Tensor ($\mathbf{\Sigma}_{\text{sagitta}} \in \mathbb{R}^{16 \times 768}$):** Orthogonal deviation of intermediate theta-phase slots from the downbeat-to-upbeat chord $\mathbf{Z}_{31} - \mathbf{Z}_0$ in 768-D space.
* **768-D Temporal Asymmetry Tensor ($\mathbf{\Delta}_{\text{temp}} = \mathbf{Z}_{16:32} - \mathbf{Z}_{0:16} \in \mathbb{R}^{16 \times 768}$):** Prospective fast-gamma future minus retrospective slow-gamma past (*Colgin, 2009* [12]; *Bieri et al., 2014* [13]).

| CLIP Token Slice | Cortical Source & Mathematical Tensor ($[N \times 768]$) |
| :--- | :--- |
| **`Token [0]`** | `<\|startoftext\|>` BOS norm anchor. |
| **`Tokens [1..4]`** | **`Fpz` (Epoch Wheel):** Smooth sinusoidal crossfade between adjacent rule sets driven by upward surprise. |
| **`Tokens [5..20]`** | **`AFz` (Compositional DAG $A \supset B$):** 16 Delta-Theta sectors orthogonalized via Gram-Schmidt over 89.5 Hz ripples. |
| **`Tokens [21..36]`** | **`F3` (Structural ON-Subspace):** $16 \times 768$ positive phase-lag modes $\text{ReLU}(+\mathbf{\Psi}) \cdot \mathbf{\Pi}_{P \to 768}$. |
| **`Tokens [37..52]`** | **`F4` (Contextual OFF-Subspace):** $16 \times 768$ negative modes $\text{ReLU}(-\mathbf{\Psi})$ + 768-D Temporal Asymmetry $\mathbf{\Delta}_{\text{temp}}$. |
| **`Tokens [53..68]`** | **`FCz` (Intrinsic Semantic Actions):** $16 \times 768$ trajectory integrated along the SVD tangent basis $\mathbf{V}_{\text{aff}}$ of the active concept. |
| **`Tokens [69..75]`** | **Global L2/3 Monty SDR Consensus:** $7 \times 768$ projection of the 4096-column HTM state. |
| **`Token [76]`** | `<\|endoftext\|>` EOS summary vector. |

### 2.5 Pure Semantic Actions via Intrinsic SVD Affordances (Zero Pixel Shifting)
An action on a concept $\mathbf{C}_0$ is defined strictly by the local geometry of semantic space around $\mathbf{C}_0$:
1. **Intrinsic Token Manifold (Works even for $N=1$ concept):** The centered internal token matrix $\widetilde{\mathbf{E}}_{C_0} = \mathbf{E}_{C_0}[1:76] - \boldsymbol{\mu}_{C_0} \in \mathbb{R}^{75 \times 768}$ encodes the internal semantic degrees of freedom of that concept in CLIP's transformer.
2. **Neighborhood Relational Manifold ($N \ge 2$ concepts):** Difference vectors to nearest neighbors in the active lore $\Delta \mathbf{C}_k = \mathbf{C}_k - \mathbf{C}_0$ define relational transition axes.
3. **Orthonormal Action Basis ($\mathbf{V}_{\text{aff}} \in \mathbb{R}^{16 \times 768}$):** Extracted via Singular Value Decomposition (SVD) of the combined manifold. `FCz` projects its 768-D sagitta and gamma wave onto $\mathbf{V}_{\text{aff}}$, deforming the concept along its own natural affordances and executing a geodesic step to a neighboring concept when sustained `persistence` crosses the cosine Voronoi boundary.

### 2.6 89.5 Hz Cortical Ripple Causal DAGs ($A \supset B$ vs $A \parallel B$)
Following *Dickey et al. (2022, PNAS)* [9], phase-lead matrices in the $70\text{--}100\text{ Hz}$ human ripple band ($\sim 89.5\text{ Hz}$) determine hierarchical nesting across slots:
* **Parent–Child Composition ($A \supset B$):** When slot $A$ phase-leads slot $B$, $B$ is projected into the orthogonal complement of $A$ via Gram-Schmidt, embedding $B$ as a detail inside container $A$.
* **Co-Equal Peers ($A \parallel B$):** When phase-lead is below threshold, $A$ and $B$ partition orthogonal half-spaces.

### 2.7 Lyapunov-Stable NLMS Learning, Persistence Boost & Smooth Epoch Wheel
Each `PredictiveColumnMonty` reconstructs its $32 \times P$ phase field $\widehat{\mathbf{\Psi}} = \mathbf{W}_{\text{out}} \mathbf{s}_{\text{L2/3}}$ using contractive Normalized Least Mean Squares (NLMS):

$$\eta_{\text{stable}} = \frac{\alpha (1 + 0.5 \cdot \text{surprise})}{\|\mathbf{s}_{\text{L2/3}}\|^2 + \epsilon}, \qquad \mathbf{W}_{\text{out}} \leftarrow 0.9995 \mathbf{W}_{\text{out}} + \eta_{\text{stable}} (\mathbf{e} \otimes \mathbf{s}_{\text{L2/3}})$$

* **Persistence Boost (20-Second Convergence):** When the 768-D phase chord maintains directional consistency ($\cos\theta > 0.75$), `persistence` ramps from $0.0 \to 1.0$, multiplying semantic step velocity by $1.0 + 4.0 \cdot \text{persistence}$ and smoothly opening `dyn_strength` from `strength_low` ($0.50$) to `burst_strength` ($0.85$).
* **Anomaly-Driven Epoch Wheel (`Fpz`):** When the user's cortical pattern remains within the learned subspace ($\mathcal{L}_{\text{MSE}} \le \bar{\mathcal{L}}$), `surprise` is $0$ and the epoch is locked. When the user intentionally breaks the pattern ($\mathcal{L}_{\text{MSE}} \gg \bar{\mathcal{L}}$), `surprise` smoothly advances the Epoch Wheel $\Phi_{\text{epoch}}$.

---

## 3. The 4-Stage Cortical Calibration Ladder (`FCz` ➔ `F3/F4` ➔ `AFz` ➔ `Fpz`)

Controlled via `POST /api/preset/ladder` or the Web Studio dropdown, allowing users to master control progressively from fast sensorimotor loops to macro-epistemic rules:

* **Stage 1 (`FCz` Semantic Action Focus — 20s Calibration):** Clamps `Fpz`, `AFz`, `F3`, and `F4` to a single predictable anchor concept from the lore. The user steers `FCz` along the 16 intrinsic SVD affordance axes $\mathbf{V}_{\text{aff}}$ with immediate `persistence` feedback.
* **Stage 2 (`+F3/F4` Structural Form & Contextual Style):** Unlocks the $16 \times 768$ ON-structural (`F3`) and OFF-contextual (`F4`) subspaces while keeping `AFz`/`Fpz` anchored.
* **Stage 3 (`+AFz` Compositional Ripple Trees $A \supset B$):** Unlocks full multi-concept Gram-Schmidt composition across Delta-Theta sectors.
* **Stage 4 (`+Fpz` Full Heterarchy & Epoch Wheel):** Unlocks all 5 regions, including anomaly-driven smooth crossfading across lore epochs.

---

## 4. Hardware Signal Chain & Transport Integrity

### 4.1 ADS131M08 Programmable Gain Amplifier with Hardware Readback Verification
* **Default Firmware Gain:** `4x` (`0x22`), configurable on the fly to `8x` (`0x33`), `16x` (`0x44`), or `32x` (`0x55`).
* **Conversion Factor:** $\text{uvScale} = \frac{1.2}{\text{Gain} \times 8\,388\,607} \times 10^6\ \mu\text{V}/\text{count}$.
* **Iterative Readback Loop:** Every PGA register write (`0x04`, `0x05`) is verified by reading back the register notifications from the ADC until hardware confirmation matches.

### 4.2 Real-Time 24-Bit Hardware Rail Saturation & Clipping Detection
* Monitors raw 24-bit integer samples against $|\text{count}| \ge 8\,300\,000$ ($>99\%$ full-scale rail) on both client and GPU daemon, highlighting saturated electrodes in real time.

---

## 5. Decoupled Cloud Architecture & Orange Pi OpenStack Orchestrator

### 5.1 Orange Pi HTTPS Cloud Controller (`orangepi_cloud_controller.py`)
Designed for gallery and studio deployments with low-latency links (~8 ms) to **OpenStack Zed** GPU clouds (`immers.cloud`):
* Runs a local HTTPS server (`https://0.0.0.0:5000`) on an Orange Pi in the gallery.
* **Keystone v3 Auto-Discovery:** Automatically discovers `OS_PROJECT_ID`, the direct external network (`immers`), and fast `SSD` Cinder volume types (`ssd_replicated_hv_multiattach`) using only `OS_USERNAME` and `OS_PASSWORD`.
* **Permanent Projector Kiosk (`/projector`):** The gallery projector stays pointed at `https://localhost:5000/projector` on the Orange Pi and automatically switches to the live `stream.mjpg` feed whenever a cloud RTX 4090 VM wakes up.

### 5.2 Cloud Cost Optimization Lifecycle (90 RUB/h ➔ Free Shelve / 2 RUB/h SSD ➔ 0 RUB/h)
1. **🚀 Start / Wake Up RTX 4090:** Boots from an existing `SHELVED_OFFLOADED` state, or performs a 40-second **Warm Boot** from the persistent SSD volume (`delete_on_termination=False`), or performs a fresh **Cold Init**.
2. **❄️ Free Pause (`Shelve`):** Archives recorded sessions to the Orange Pi, flushes disk caches (`sync`), and calls `conn.compute.shelve_server()`, detaching the RTX 4090 and stopping GPU billing (`0 RUB/h` for GPU).
3. **⏸️ Delete VM, Keep SSD Volume:** Deletes the compute instance while preserving the SSD volume with `/opt/hf_cache` (~2 RUB/h).
4. **🌙 End of Day (`Full Teardown` into 0.00 RUB/h):** Harvests all recorded sessions to the Orange Pi and deletes all VMs, Cinder volumes, and Floating IPs.

### 5.3 Zero-Build GitHub Hot-Reload & Persistent `/opt/hf_cache`
* Uses `docker-compose.yml` (`--no-build`) with direct host bind-mounts (`/opt/neuro_heterarchy:/app` and `/opt/hf_cache:/root/.cache/huggingface`).
* Model weights (LCM, CLIP, TAESD, MusicGen) are stored on the persistent SSD volume in `/opt/hf_cache` and are never re-downloaded after pause/resume.
* Clicking **«🔄 Update Code from GitHub»** executes `git fetch && git reset --hard origin/HEAD` and restarts the containers in **3 seconds** over SSH without rebuilding Docker images.

---

## 6. Multimodal 1:1 Synchronized Session Recorder & Session Manager

* **Hardware-Locked 1:1 CFR Video Timeline (`neuro_recorder.py`):** Duplicates variable-framerate diffusion frames across exact 30 FPS ticks locked to `pylsl.local_clock()`, guaranteeing $<15\text{ ms}$ alignment between `video_timeline.mp4` and 16-channel `eeg_raw.npz`.
* **Built-In Session Manager on Orange Pi:** Downloads each session as a compact `.tar.gz` archive (`video_timeline.mp4`, `eeg_raw.npz`, `events.csv`, `manifest.json` — automatically excluding raw `frames/*.jpg`), runs `analyze_session.py` to generate `verified_sync_alignment.png`, and provides an in-browser video + EEG alignment viewer that remains accessible even after the cloud GPU is deleted.

---

## 7. REST API & AI Agent Integration (OpenAPI 3.0)

Full OpenAPI specification is served at `https://<HOST>:8080/openapi.json`:
* `GET  /api/world/concepts` — List active visual concepts.
* `POST /api/world/concepts/set` — Atomically replace the active concept pool (`{"concepts": [...]}`).
* `POST /api/world/concepts/add` — Add a visual concept (`{"concept": "..."}`).
* `POST /api/world/concepts/remove` — Remove a visual concept (`{"concept": "..."}`).
* `POST /api/preset/ladder` — Switch calibration ladder stage `1..4` (`{"stage": 1, "anchor": "..."}`).
* `POST /api/region/config` — Configure individual region parameters (`FCz`, `F3`, `F4`, `AFz`, `Fpz`).
* `GET  /api/world/telemetry` — Query dominant thought, 2D SWM coordinates, `monty_loss`, `persistence`, and `surprise`.
* `GET  /api/region/telemetry` — Query per-region Monty SDR metrics across all 5 regions.
* `POST /api/recording/start` & `POST /api/recording/stop` — Start/stop 1:1 multimodal dataset recording.

---

## 8. Deployment & Operations

### 1. Orange Pi / Local Cloud Controller Deployment
```bash
export OS_USERNAME="your_immers_cloud_login"
export OS_PASSWORD="your_immers_cloud_password"
python3 orangepi_cloud_controller.py
# Open https://localhost:5000 (Operator Dashboard)
# Open https://localhost:5000/projector (Fullscreen Projector View)
```

### 2. Autonomous BLE-to-Cloud Bridge on Orange Pi
```bash
python3 orangepi_ble_bridge.py --role AFz --gain 4
```

### 3. Direct Docker Compose Deployment on GPU Host
```bash
docker compose -f docker-compose.yml up -d --no-build
```

---

## 9. Comprehensive Scientific Bibliography & DOIs

1. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** *The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy.* arXiv:2507.05888v2 [q-bio.NC].
2. **Hawkins, J., Ahmad, S., & Cui, Y. (2017).** *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World.* Frontiers in Neural Circuits, 11:81. [DOI: 10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)
3. **Mountcastle, V. B. (1997).** *The columnar organization of the neocortex.* Brain, 120(4), 701–722. [DOI: 10.1093/brain/120.4.701](https://doi.org/10.1093/brain/120.4.701)
4. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [DOI: 10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
5. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** *Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory.* PNAS, 115(5), 1117–1122. [DOI: 10.1073/pnas.1717766115](https://doi.org/10.1073/pnas.1717766115)
6. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [DOI: 10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
7. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [DOI: 10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
8. **Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** *Cortical tracking of hierarchical linguistic structures in connected speech.* Nature Neuroscience, 19(1), 158–164. [DOI: 10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
9. **Dickey, C. W., et al. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* PNAS, 119(28), e2107797119. [DOI: 10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
10. **Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
11. **Nolte, G., et al. (2004).** *Identifying true brain interaction from EEG data using the imaginary part of coherency.* Clinical Neurophysiology, 115(10), 2292–2307. [DOI: 10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
12. **Colgin, L. L., et al. (2009).** *Frequency of gamma oscillations routes flow of information in the hippocampus.* Nature, 462(7271), 353–357. [DOI: 10.1038/nature08573](https://doi.org/10.1038/nature08573)
13. **Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** *Slow and fast gamma rhythms coordinate different spatial coding modes in hippocampal place cells.* Neuron, 82(3), 670–681. [DOI: 10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
14. **Lisman, J. E., & Jensen, O. (2013).** *The theta-gamma neural code.* Neuron, 77(6), 1002–1016. [DOI: 10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
15. **Fries, P. (2015).** *Rhythms for cognition: communication through coherence.* Neuron, 88(1), 220–235. [DOI: 10.1016/j.neuron.2015.09.034](https://doi.org/10.1016/j.neuron.2015.09.034)
16. **Churchland, M. M., & Shenoy, K. V. (2024).** *Preparatory activity and the expansive null-space.* Nature Reviews Neuroscience, 25, 213–236. [DOI: 10.1038/s41583-024-00796-z](https://doi.org/10.1038/s41583-024-00796-z)
17. **Zimnik, A. J., & Churchland, M. M. (2021).** *Independent generation of sequence elements by motor cortex.* Nature Neuroscience, 24, 412–424. [DOI: 10.1038/s41593-021-00798-5](https://doi.org/10.1038/s41593-021-00798-5)
18. **Weber, J., et al. (2023).** *Subspace partitioning in the human prefrontal cortex resolves cognitive interference.* PNAS, 120(31), e2220523120. [DOI: 10.1073/pnas.2220523120](https://doi.org/10.1073/pnas.2220523120)
19. **Flesch, T., et al. (2022).** *Orthogonal representations for robust context-dependent task performance in brains and neural networks.* Neuron, 110(7), 1258–1270. [DOI: 10.1016/j.neuron.2022.01.005](https://doi.org/10.1016/j.neuron.2022.01.005)
20. **Panichello, M. F., & Buschman, T. J. (2021).** *Shared mechanisms underlie the control of working memory and attention.* Nature, 592, 601–605. [DOI: 10.1038/s41586-021-03390-w](https://doi.org/10.1038/s41586-021-03390-w)
21. **Koechlin, E., & Hyafil, A. (2007).** *Anterior prefrontal function and the limits of human decision-making.* Science, 318(5850), 594–598. [DOI: 10.1126/science.1142995](https://doi.org/10.1126/science.1142995)
22. **Boorman, E. D., et al. (2009).** *How green is the grass on the other side? Frontopolar cortex and evidence for alternatives.* Neuron, 62(5), 733–743. [DOI: 10.1016/j.neuron.2009.05.014](https://doi.org/10.1016/j.neuron.2009.05.014)
23. **Lee, H. S., Schreiner, L., Jo, S.-H., Sieghartsleitner, S., Jordan, M., Pretl, H., Guger, C., & Park, H.-S. (2022).** *Individual finger movement decoding using a novel ultra-high-density electroencephalography-based brain-computer interface system.* Frontiers in Neuroscience, 16:1009878. [DOI: 10.3389/fnins.2022.1009878](https://doi.org/10.3389/fnins.2022.1009878)

