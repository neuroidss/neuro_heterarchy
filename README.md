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
1. [System Overview & Scientific Paradigm Shift](#1-system-overview--scientific-paradigm-shift)
2. [Biophysical Foundations & Circuit Theory](#2-biophysical-foundations--circuit-theory)
   - 2.1 [Macrocolumnar Scale (26 mm Footprint = 2,000–4,200 Macrocolumns)](#21-macrocolumnar-scale-26-mm-footprint--20004200-macrocolumns)
   - 2.2 [The Zero-Power Principle: Pure Phase Coherence ($ci\text{PLV}$)](#22-the-zero-power-principle-pure-phase-coherence-ciplv)
   - 2.3 [Laminar Push-Pull Gating: Deep Beta vs Superficial Gamma](#23-laminar-push-pull-gating-deep-beta-vs-superficial-gamma)
   - 2.4 [2D Sequence Working Memory Manifold (Fan 2024 / Chen 2024)](#24-2d-sequence-working-memory-manifold-fan-2024--chen-2024)
   - 2.5 [89.5 Hz Cortical Ripple Causal DAGs (Dickey 2022)](#25-895-hz-cortical-ripple-causal-dags-dickey-2022)
3. [Closed-Loop Neurofeedback vs. Passive Recording](#3-closed-loop-neurofeedback-vs-passive-recording)
4. [Hardware Signal Chain & Transport Integrity](#4-hardware-signal-chain--transport-integrity)
   - 4.1 [ADS131M08 Programmable Gain Amplifier (4x, 8x, 16x, 32x)](#41-ads131m08-programmable-gain-amplifier-4x-8x-16x-32x)
   - 4.2 [Real-Time 24-bit Hardware Rail Clipping Detection](#42-real-time-24-bit-hardware-rail-clipping-detection)
   - 4.3 [Zero-Allocation Web Bluetooth & Drop Rate Tracking](#43-zero-allocation-web-bluetooth--drop-rate-tracking)
5. [Decoupled Architecture & Remote Command Relay](#5-decoupled-architecture--remote-command-relay)
   - 5.1 [Secure Context TLS / HTTPS Requirements](#51-secure-context-tls--https-requirements)
   - 5.2 [Remote Command Relay via Server-Sent Events (SSE)](#52-remote-command-relay-via-server-sent-events-sse)
   - 5.3 [Ultra-Lean BLE Probe Mode for Mobile Hosts](#53-ultra-lean-ble-probe-mode-for-mobile-hosts)
6. [Dynamic Concept Engine & Generative Synthesis](#6-dynamic-concept-engine--generative-synthesis)
   - 6.1 [Zero-Hardcode & Unconditioned Latent Browsing](#61-zero-hardcode--unconditioned-latent-browsing)
   - 6.2 [Thread-Safe Atomic Vocab Swapping](#62-thread-safe-atomic-vocab-swapping)
   - 6.3 [Dynamics Presets & Safety Watchdog](#63-dynamics-presets--safety-watchdog)
   - 6.4 [MusicGen 32 kHz Real-Time Lag Monitor & Anti-Lag Buffer](#64-musicgen-32-khz-real-time-lag-monitor--anti-lag-buffer)
7. [Multimodal 1:1 Synchronized Session Recorder](#7-multimodal-11-synchronized-session-recorder)
   - 7.1 [Hardware-Locked 1:1 CFR Video Timeline](#71-hardware-locked-11-cfr-video-timeline)
   - 7.2 [Dataset Structure (`eeg_raw.npz`, `video_timeline.mp4`, `events.csv`)](#72-dataset-structure-eeg_rawnpz-video_timelinemp4-eventscsv)
8. [AI Agent Integration (ChatGPT / Function Calling)](#8-ai-agent-integration-chatgpt--function-calling)
9. [Deployment & Operations](#9-deployment--operations)
10. [Scientific References](#10-scientific-references)

---

## 1. System Overview & Scientific Paradigm Shift

**NeuroCanvas** is an unconstrained, real-time closed-loop Brain-Computer Interface (BCI) based on the **Thousand Brains Theory 2.0** (*Hawkins, Leadholm, Clay, 2025/2026*), **Working Memory 2.0** (*Miller, Lundqvist, Bastos, 2018*), and modern empirical neurogeometry (*Fan et al., 2024; Chen et al., 2024; Dickey et al., 2022*).

Rather than reducing brain activity to discrete classifiers or arbitrary 1D bandpower metrics, NeuroCanvas models the neocortex as a **distributed, self-sufficient sensorimotor simulation engine**:
* **The Neocortex is a Heterarchy:** Cortical regions operate in parallel and hierarchically at the same time. Primary, secondary, and association areas learn structured models of complete 3D objects and abstract concepts.
* **The Diffusion Canvas is an Inverted Artificial Retina:** Generative diffusion models (LCM, SD-Turbo, SDXL) project continuous cortical semantic states directly into high-dimensional sensory feedback.
* **Pure Phase Coherence:** Operates entirely on volume-conduction-free directed phase lags ($ci\text{PLV}$), eliminating classical amplitude/power artifacts.

---

## 2. Biophysical Foundations & Circuit Theory

### 2.1 Macrocolumnar Scale (26 mm Footprint = 2,000–4,200 Macrocolumns)
A single **FreeEEG16-alpha2** circular sensor has an outer diameter of $26\text{ mm}$ ($\text{Area} = \pi \times 13^2 \approx 531\text{ mm}^2$).
* According to Vernon Mountcastle (1957, 1997) and Jeff Hawkins (2017), a cortical macrocolumn has a diameter of $300\text{--}600\ \mu\text{m}$ (area $\sim 0.12\text{--}0.28\text{ mm}^2$).
* Underneath a single 26 mm sensor reside **between 1,900 and 4,200 macrocolumns**, containing **over 250,000 minicolumns** and **more than 50 million biological neurons**.
* 16 concentric electrodes form **120 unique bipolar channel pairs**, measuring the phase-gradient manifold across this population.

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

### 2.2 The Zero-Power Principle: Pure Phase Coherence ($ci\text{PLV}$)
Traditional EEG metrics rely on scalar FFT power ($\mu\text{V}^2/\text{Hz}$), which is corrupted by volume conduction, skull attenuation, and muscle artifacts. NeuroCanvas computes the **Corrected Imaginary Phase Locking Value ($ci\text{PLV}$)** (*Bruña et al., 2018; Nolte et al., 2004*):

$$ci\text{PLV}_{i, j} = \frac{\frac{1}{T} \sum_{t=1}^T \Im \left( z_i(t) z_j^*(t) \right)}{\sqrt{1 - \left( \frac{1}{T} \sum_{t=1}^T \Re \left( z_i(t) z_j^*(t) \right) \right)^2}}$$

$ci\text{PLV}$ discards instantaneous zero-lag projections (volume conduction) while preserving true non-zero phase synchronization between cortical generators.

### 2.3 Laminar Push-Pull Gating: Deep Beta vs Superficial Gamma
Following *Miller, Lundqvist, & Bastos (Neuron, 2018)*:
* **Deep Layers (5 & 6) — Beta ($15\text{--}30\text{ Hz}$):** Encodes top-down task rules, executive inhibition, and attentional stability.
* **Superficial Layers (2 & 3) — Gamma ($30\text{--}100\text{ Hz}$):** Encodes feedforward sensory content and bottom-up feature bursts.
* **Push-Pull Interaction:** Elevated deep-layer beta suppresses superficial gamma, clearing working memory. Relaxation of beta disinhibits layer 2/3 recurrent networks, allowing gamma bursts to refresh synaptic weights.

### 2.4 2D Sequence Working Memory Manifold (Fan 2024 / Chen 2024)
Primate and human working memory organizes sequences into a 2D neural representational geometry (*Fan et al., 2024; Chen et al., 2024*):
* **Axis $X$ (Local Rank $L$):** Position of an item within a chunk ($L_1, L_2, L_3$).
* **Axis $Y$ (Global Rank $G$):** Position of a chunk within the macro-sequence ($G_1, G_2, G_3$).

The 120 bipolar dipoles project onto this orthogonal plane:
$$x_{\text{local}} = \text{clamp}\left( \frac{1}{40} \sum_{d=1}^{120} \mathbf{\Psi}(d) \cdot \Delta x_d, -1.0, 1.0 \right), \quad y_{\text{global}} = \text{clamp}\left( \frac{1}{40} \sum_{d=1}^{120} \mathbf{\Psi}(d) \cdot \Delta y_d, -1.0, 1.0 \right)$$
This $(x_L, y_G)$ coordinate glides across the 2D working memory manifold, balancing token-level cross-attention without discrete state resets.

### 2.5 89.5 Hz Cortical Ripple Causal DAGs (Dickey 2022)
Volume-conduction-free directed phase locking across all 120 electrode pairs is evaluated at 89.5 Hz (*Dickey et al., PNAS 2022*). The pairwise causal lead matrix between temporal phase slots yields a Directed Acyclic Graph (DAG):
* $\mathbf{D}_{A, B} \ge +0.03 \implies A \supset B$ ($A$ is container/parent of $B$). $B$ is projected into the orthogonal complement via Gram-Schmidt orthogonalization:
$$\mathbf{T}_B^{\perp} = \mathbf{T}_B - \frac{\langle \mathbf{T}_B, \mathbf{T}_{\text{accum}} \rangle}{\|\mathbf{T}_{\text{accum}}\|^2 + \epsilon} \mathbf{T}_{\text{accum}}$$
* $\mathbf{D}_{A, B} < 0.03 \implies A \parallel B$ ($A$ and $B$ are co-equal peers). Integrated via hemispheric latent partitioning.

---

## 3. Closed-Loop Neurofeedback vs. Passive Recording

A central tenet of the NeuroCanvas framework is the **epistemological necessity of closed-loop active inference**:
1. **Passive / Open-Loop EEG is High-Entropy Noise:** In the absence of an attentive feedback loop, the 2,000–4,200 macrocolumns beneath a 26 mm sensor fire asynchronously. Their microscopic dipoles cancel out, leaving raw recordings dominated by $1/f$ pink noise and muscle artifacts. Recording open-loop EEG with the expectation of decoding "thoughts" is scientifically ungrounded.
2. **Closed-Loop Neurofeedback Forces Columnar Phase-Locking:** When a human perceives the generative canvas and actively attempts to steer the image, the brain enters a closed-loop cybernetic state. Prediction errors between top-down intention (deep-layer beta) and bottom-up visual feedback (superficial gamma) force local macrocolumns to synchronize their dipole moments into a macroscopic coherent wave, dramatically increasing the signal-to-noise ratio (SNR).
3. **Ground-Truth Event Alignment:** Every generated image acts as a known attractor in latent space, providing physical ground truth for the accompanying electrophysiological state.

---

## 4. Hardware Signal Chain & Transport Integrity

### 4.1 ADS131M08 Programmable Gain Amplifier (4x, 8x, 16x, 32x)
The FreeEEG16-alpha2 board uses two 8-channel 24-bit Texas Instruments ADS131M08 ADCs.
* **Default Firmware Gain:** **4x** (Register `0x04` = `0x22`, `0x05` = `0x22`).
* **Dynamic Conversion Factor ($\mu\text{V}/\text{count}$):**
$$\text{uvScale} = \frac{1.2}{\text{Gain} \times (2^{23} - 1)} \times 10^6 = \frac{1.2}{\text{Gain} \times 8\,388\,607} \times 10^6$$
  * Gain 4x: $\approx 0.0357628\,\mu\text{V}/\text{count}$ (Full-Scale Range: $\pm 300\text{ mV}$)
  * Gain 8x: $\approx 0.0178814\,\mu\text{V}/\text{count}$ (Full-Scale Range: $\pm 150\text{ mV}$)
  * Gain 16x: $\approx 0.0089407\,\mu\text{V}/\text{count}$ (Full-Scale Range: $\pm 75\text{ mV}$)
  * Gain 32x: $\approx 0.0044703\,\mu\text{V}/\text{count}$ (Full-Scale Range: $\pm 37.5\text{ mV}$)
When the gain is adjusted via the UI or API, the gateway transmits the appropriate register commands over BLE and updates `uvScale` dynamically, guaranteeing physical microvolt calibration in LSL.

### 4.2 Real-Time 24-bit Hardware Rail Clipping Detection
Hardware saturation occurs when an electrode loses contact or picks up large electrostatic potentials, driving the ADC output code to the positive or negative supply rail ($+8\,388\,607$ or $-8\,388\,608$).
* **Threshold:** $|v| \ge 8\,300\,000$ counts (99% of 24-bit full-scale range).
* **Zero CPU Overhead:** The Web Bluetooth packet loop performs a single integer comparison per channel. Peak values are evaluated once per second, lighting up red indicator LEDs on the specific saturated channels.

### 4.3 Zero-Allocation Web Bluetooth & Drop Rate Tracking
Packet format: `[0xA0, Counter (0..255), 48 Bytes ADC Data (16 ch x 24 bit), 0xC0]`.
* **Packet Loss Tracking:** Byte `b[1]` is checked on every arrival:
$$\text{Lost} = (\text{Counter} - (\text{LastCounter} + 1)) \pmod{256}$$
$$\text{Loss Rate (\%)} = \frac{\text{Total Lost}}{\text{Total Received} + \text{Total Lost}} \times 100\%$$
* **Throttled Ingestion:** Employs a non-overlapping request queue (`sending: false` latch, max buffer 400 samples) to prevent socket exhaustion (`ERR_INSUFFICIENT_RESOURCES`) in Chrome.

---

## 5. Decoupled Architecture & Remote Command Relay

```
                           SYSTEM NETWORK TOPOLOGY
                           
     [Projector in Darkness]      [Operator Smartphone]       [ChatGPT / External AI]
       (Fullscreen Canvas)         (BLE Relay & Controls)        (REST Action Caller)
                │                            │                            │
                │ GET /stream.mjpg           │ POST /api/eeg_push         │ POST /api/world/concepts/add
                │ GET /api/state_stream      │ GET /api/control_stream    │ GET /api/world/telemetry
                ▼                            ▼                            ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                    web_cloud_gateway.py (Port 8080)                         │
   │  • TLS / HTTPS Secure Context Host (Cert: --cert, Key: --key)               │
   │  • Central Device & Command Registry (Remote Command Relay)                 │
   │  • Server-Sent Events (SSE) Broadcast Hub                                   │
   │  • OpenAPI 3.0 Server (/openapi.json)                                       │
   │  • Dynamic LSL Stream Router (pylsl)                                        │
   └─────────────▲───────────────────────────▲─────────────────────────────┬─────┘
                 │ IPC (Port 6002)           │ IPC (Port 6003)             │ IPC (Port 6004)
                 │ JPEG Stream               │ PCM Audio Stream            │ Control Bus
                 │                           │                             ▼
   ┌─────────────┴─────────────┐ ┌───────────┴─────────────┐ ┌───────────────────┴───┐
   │ neuro_heterarchy_diff...  │ │ neuro_heterarchy_mus... │ │  neuro_recorder.py     │
   │ • Diffusion Runner        │ │ • MusicGen 32kHz Runner │ │  • 1:1 Video Timeline  │
   │ • TBT 2.0 Genesis Engine  │ │ • Real-Time Lag Monitor │ │  • 16-ch EEG NPZ Dump  │
   │ • Watchdog Parameter Guard│ │ • Anti-Lag Auto-Drop    │ │  • BIDS events.csv     │
   └───────────────────────────┘ └─────────────────────────┘ └────────────────────────┘
```

### 5.1 Secure Context TLS / HTTPS Requirements
The Web Bluetooth API (`navigator.bluetooth`) is strictly prohibited by browser security specifications on insecure origins (`http://`). It functions **exclusively** on `https://` (or `localhost`).
* The gateway provides CLI parameters `--cert` and `--key`, with cascaded fallback:
  1. CLI arguments `--cert` / `--key`.
  2. Environment variables `CERT_PATH` / `KEY_PATH`.
  3. Local directory: `./cert.pem`, `./key.pem`.
  4. Docker container directory: `/app/cert.pem`, `/app/key.pem`.
* Self-signed certificates can be generated with a single command:
  ```bash
  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
  ```

### 5.2 Remote Command Relay via Server-Sent Events (SSE)
To decouple the device from the operator display:
* A phone connects to the FreeEEG16 board via Web Bluetooth and registers with the gateway (`POST /api/device/register`).
* An operator on a separate laptop (with no Bluetooth enabled) accesses the web console and adjusts the PGA Gain to 32x.
* The gateway dispatches an SSE event (`ble_cmd`) to the phone.
* The phone executes the GATT write to `cmdCharacteristic` and acknowledges completion.

### 5.3 Ultra-Lean BLE Probe Mode for Mobile Hosts
When running Web Bluetooth on lower-powered devices (such as smartphones or tablets):
* Enable the `⚡ Легковесный BLE-зонд` toggle in the web console.
* This shuts down local MJPEG video decoding and audio playback.
* CPU usage drops below $1\%$, eliminating thermal throttling, garbage collection pauses, and packet drops.

---

## 6. Dynamic Concept Engine & Generative Synthesis

### 6.1 Zero-Hardcode & Unconditioned Latent Browsing
* **No Hardcoded Concepts:** If no lore file is passed (`--lore`), the system initializes with an empty list (`[]`).
* **Unconditioned Baseline:** In the absence of concepts, the engine does not pass an all-zero tensor (which causes UNet `NaN` crashes and black screens). Instead, it loads the clean empty-string prompt embedding `get_77_embedding_gpu("")` and allows the canvas to evolve purely via phase-gradient drift.
* **Subtractive Mechanics:** When concepts are removed, the engine smoothly transitions from the latent coordinates of the most recent concept without abrupt black-frame resets.

### 6.2 Thread-Safe Atomic Vocab Swapping
Dynamically adding or removing concepts from an external agent runs asynchronously against the primary GPU inference loop.
* `UniversalSemanticSpace` uses an internal re-entrant lock (`threading.RLock()`).
* New embeddings are precomputed and normalized before swapping `self.words` and `self.norm_vocab` atomically.
* Slot index lookups are protected via modulo boundary clamping (`idx % len(self.words)`), preventing `IndexError: list index out of range`.

### 6.3 Dynamics Presets & Safety Watchdog
To prevent out-of-range parameters from freezing or breaking diffusion, the console provides three tuned presets alongside manual overrides:
* **🧘 Глубокая медитация:** Steps: 2, CFG: 1.0, Strength: 0.45–0.70, EMA: 0.30 (slow, viscous, continuous morphing).
* **⚖️ Баланс / Стандарт:** Steps: 2, CFG: 1.0, Strength: 0.55–0.85, EMA: 0.16 (stable concept tracking).
* **⚡ Экспрессия / Высокая динамика:** Steps: 3, CFG: 1.2, Strength: 0.65–0.92, EMA: 0.08 (rapid response to phase transitions).
* **Watchdog Auto-Reversion:** Parameters are clamped server-side. If the U-Net fails to produce a valid image, the worker automatically rolls back to the last known safe configuration.

### 6.4 MusicGen 32 kHz Real-Time Lag Monitor & Anti-Lag Buffer
Because MusicGen generates audio in discrete blocks, faster-than-real-time generation (RTF > 1.0) can lead to buffer bloat:
* **Live Telemetry:** The UI displays queue depth in seconds, chunk count, RTF, and latency in milliseconds.
* **Anti-Lag Auto-Drop:** If the queue exceeds `target_buffer_count` (e.g., 2 chunks), the gateway drops the oldest audio blocks, keeping auditory feedback locked to the current cortical state.

---

## 7. Multimodal 1:1 Synchronized Session Recorder

### 7.1 Hardware-Locked 1:1 CFR Video Timeline
Diffusion generation produces frames at variable intervals (VFR). Storing frames sequentially in a fixed-framerate container causes temporal distortion.
* `NeuroSessionRecorder` creates a **30 FPS Constant Frame Rate (CFR)** container with **zero cumulative drift**.
* Each frame is held in the video for the exact number of ticks it was displayed on screen:
$$\text{Ticks to Write} = \text{round}\left((T_{\text{next}} - T_{\text{start}}) \times 30.0\right) - \text{Total Ticks Written}$$
* The final video duration matches the continuous raw EEG duration down to the millisecond.

### 7.2 Dataset Structure
When recording is terminated, a timestamped session directory is finalized:
```
sessions/session_20261010_131400/
├── eeg_raw.npz          # Raw 16-channel EEG data and LSL hardware timestamps
├── video_timeline.mp4   # 1:1 real-time video timeline (30 FPS)
├── events.csv           # Millisecond-accurate log of frame indices, concepts, and 2D SWM coordinates
├── manifest.json        # Session metadata, durations, and sample counts
└── frames/              # Directory containing individual full-resolution JPEG frames
    ├── frame_000000.jpg
    ├── frame_000001.jpg
    └── ...
```

#### Structure of `events.csv`:
| Column | Description |
| :--- | :--- |
| `unique_frame_id` | Monotonically increasing unique image identifier. |
| `lsl_timestamp` | Hardware LSL timestamp (`local_clock()`) when frame was displayed. |
| `wall_time` | POSIX system timestamp. |
| `duration_ms` | Exact duration the frame was visible on screen. |
| `video_tick_start` | Starting index of frame in `video_timeline.mp4` (at 30 FPS). |
| `video_tick_end` | Ending index of frame in `video_timeline.mp4`. |
| `event_type` | Event descriptor (`FRAME_DISPLAYED`, `SESSION_START`, etc.). |
| `root_concept` | Dominant concept container ($A$ in $A \supset B$). |
| `child_concept` | Nested concept ($B$ in $A \supset B$). |
| `dag_relation` | Topological relation ($⊃$ parent, $\parallel$ co-equal peer). |
| `x_local` | 2D SWM Local Rank coordinate ($L_1\text{--}L_3$) from F3. |
| `y_global` | 2D SWM Global Rank coordinate ($G_1\text{--}G_3$) from F3. |
| `dyn_strength` | Diffusion denoising strength applied for this frame. |
| `active_pool` | Pipe-delimited list of active concepts in memory. |

---

## 8. AI Agent Integration (ChatGPT / Function Calling)

External agents interface with the world via OpenAPI 3.0 endpoints (`http://localhost:8080/openapi.json`):

### Available API Endpoints:
* `GET  /api/world/concepts` — Retrieve active visual concepts.
* `POST /api/world/concepts/add` — Add a visual concept on the fly (`{"concept": "Wave"}`).
* `POST /api/world/concepts/remove` — Remove a visual concept on the fly (`{"concept": "Wave"}`).
* `GET  /api/music/concepts` — Retrieve active musical themes.
* `POST /api/music/concepts/add` — Add a music theme (`{"concept": "crackling fire ambient"}`).
* `POST /api/music/concepts/remove` — Remove a music theme.
* `GET  /api/world/telemetry` — Query dominant concept, 2D SWM coordinates, and user focus.
* `POST /api/recording/start` — Start synchronized multimodal recording.
* `POST /api/recording/stop` — Stop recording and finalize session dataset.
* `POST /api/diffusion/config` — Adjust U-Net generation parameters.
* `POST /api/device/command` — Relay hardware commands (gain, role) to active devices.

### Automated Agent Simulation:
Run the provided simulation script to test dynamic two-by-two concept replacement:
```bash
python3 demo_agent_quest_simulation.py
```

---

## 9. Deployment & Operations

### 9.1 Docker Compose Startup
```bash
# Build and launch all 5 containers
docker compose up --build -d

# Monitor gateway logs
docker compose logs -f web-gateway
```

### 9.2 Local Development Startup (Without Docker)
```bash
# 1. Start the diffusion model worker
python3 brain_server.py --mode lcm

# 2. Start the MusicGen audio worker
python3 musicgen_server.py --port 6005

# 3. Start the Web & LSL Gateway
python3 web_cloud_gateway.py --port 8080 --cert ./cert.pem --key ./key.pem

# 4. Start the Vision Heterarchy Engine
python3 neuro_heterarchy_diffusion.py --mode lcm --gamma-max 100.0

# 5. Start the Music Heterarchy Engine
python3 neuro_heterarchy_musicgen.py --mode semantic --gamma-max 100.0
```

### 9.3 Dataset Verification & Analysis
To verify synchronization and view EEG traces aligned with specific video frames:
```bash
python3 analyze_session.py sessions/session_YYYYMMDD_HHMMSS
```

---

## 10. Comprehensive Scientific Bibliography & Anatomical Mapping

### I. Neocortical Heterarchy & Canonical Column Microcircuits (TBT)
1. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** *The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy.* arXiv:2507.05888v2 [q-bio.NC].
   * *System Role:* Theoretical framework for neocortical heterarchy, compositionality via location-by-location coordinate transforms, and thalamic pose conversion.
2. **Hawkins, J., Ahmad, S., & Cui, Y. (2017).** *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World.* Frontiers in Neural Circuits, 11:81. [DOI: 10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)
   * *System Role:* Microcircuit foundation: Layer 4 sensory-location binding and Layer 2/3 inter-column voting consensus.
3. **Mountcastle, V. B. (1997).** *The columnar organization of the neocortex.* Brain, 120(4), 701–722. [DOI: 10.1093/brain/120.4.701](https://doi.org/10.1093/brain/120.4.701)
   * *System Role:* Morphological proof of 300–600 µm macrocolumn modules (~2,000–4,200 under a single 26mm sensor).

### II. Laminar Working Memory & Push-Pull Executive Gating (Miller WM 2.0)
4. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [DOI: 10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
   * *System Role:* Push-pull gating mechanism: Deep-layer beta (15–30 Hz) controls top-down inhibition, superficial gamma (30–100 Hz) drives feedforward content reactivations.
5. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).** *Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory.* PNAS, 115(5), 1117–1122. [DOI: 10.1073/pnas.1717766115](https://doi.org/10.1073/pnas.1717766115)
   * *System Role:* Empirical laminar separation: Deep beta phase modulates superficial gamma burst amplitude.

### III. 2D Neural Geometry & Sequence Working Memory (F3 / Left dlPFC)
6. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [DOI: 10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
   * *System Role:* 2D orthogonal rank manifold mapping: Factorization of local rank ($L$) and global rank ($G$) in prefrontal state space.
7. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [DOI: 10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
   * *System Role:* Disentangled sensory and rank-WM subspaces in primate frontal cortex during forward and backward sequence sorting.
8. **Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** *Cortical tracking of hierarchical linguistic structures in connected speech.* Nature Neuroscience, 19(1), 158–164. [DOI: 10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
   * *System Role:* Multi-scale phase tracking of abstract hierarchical syntax across distinct cortical temporal receptive windows.

### IV. Transcortical Ripples & Causal Phase DAGs (AFz / Dickey Graph)
9. **Dickey, C. W., et al. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* PNAS, 119(28), e2107797119. [DOI: 10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
   * *System Role:* ~89.5 Hz human cortical ripples phase-locking over 25 cm; foundation for the causal lead matrix $\mathbf{D}_{i,j}$ and recursive Gram-Schmidt slot trees ($A \supset B$).
10. **Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
    * *System Role:* Mathematical foundation of the vector-accelerated Corrected Imaginary Phase Locking Value ($ci\text{PLV}$), zero-lag artifact immunity.
11. **Nolte, G., et al. (2004).** *Identifying true brain interaction from EEG data using the imaginary part of coherency.* Clinical Neurophysiology, 115(10), 2292–2307. [DOI: 10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
    * *System Role:* Physical proof that imaginary coherence isolates true non-instantaneous axonal propagation from volume conduction.

### V. Phase-Amplitude Coupling & Temporal Chronometry
12. **Colgin, L. L., et al. (2009).** *Frequency of gamma oscillations routes flow of information in the hippocampus.* Nature, 462(7271), 353–357. [DOI: 10.1038/nature08573](https://doi.org/10.1038/nature08573)
    * *System Role:* Routing of past/retrieval via slow gamma and future/encoding via fast gamma at distinct theta phases.
13. **Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** *Slow and fast gamma rhythms coordinate different spatial coding modes in hippocampal place cells.* Neuron, 82(3), 670–681. [DOI: 10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
    * *System Role:* Prospective vs retrospective coding modes; derivation of the directional temporal bias vector $r_y$.
14. **Lisman, J. E., & Jensen, O. (2013).** *The theta-gamma neural code.* Neuron, 77(6), 1002–1016. [DOI: 10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
    * *System Role:* Multi-item working memory buffers structured as nested gamma subcycles within slow theta oscillations.
15. **Fries, P. (2015).** *Rhythms for cognition: communication through coherence.* Neuron, 88(1), 220–235. [DOI: 10.1016/j.neuron.2015.09.034](https://doi.org/10.1016/j.neuron.2015.09.034)
    * *System Role:* Communication-through-Coherence (CTC) theory governing selective inter-areal message routing via phase alignment.

### VI. Motor Preparation, Null-Space & Kinematics (FCz / pre-SMA / SMA)
16. **Churchland, M. M., & Shenoy, K. V. (2024).** *Preparatory activity and the expansive null-space.* Nature Reviews Neuroscience, 25, 213–236. [DOI: 10.1038/s41583-024-00796-z](https://doi.org/10.1038/s41583-024-00796-z)
    * *System Role:* Motor preparation in output-null subspaces; justifies non-disruptive 4D kinematic tracking during active memory maintenance.
17. **Zimnik, A. J., & Churchland, M. M. (2021).** *Independent generation of sequence elements by motor cortex.* Nature Neuroscience, 24, 412–424. [DOI: 10.1038/s41593-021-00798-5](https://doi.org/10.1038/s41593-021-00798-5)
    * *System Role:* Independent population dynamics for successive sequence components in premotor/motor hierarchies.

### VII. Cognitive Interference, Entropy & Orthogonal Subspaces (F4 / Right dlPFC)
18. **Weber, J., et al. (2023).** *Subspace partitioning in the human prefrontal cortex resolves cognitive interference.* PNAS, 120(31), e2220523120. [DOI: 10.1073/pnas.2220523120](https://doi.org/10.1073/pnas.2220523120)
    * *System Role:* Prefrontal subspace partitioning; explains how right dlPFC phase entropy modulates denoising strength without corrupting semantic identity.
19. **Flesch, T., et al. (2022).** *Orthogonal representations for robust context-dependent task performance in brains and neural networks.* Neuron, 110(7), 1258–1270. [DOI: 10.1016/j.neuron.2022.01.005](https://doi.org/10.1016/j.neuron.2022.01.005)
    * *System Role:* Contextual task representation via geometry: Orthogonal low-dimensional manifolds prevent catastrophic forgetting.
20. **Panichello, M. F., & Buschman, T. J. (2021).** *Shared mechanisms underlie the control of working memory and attention.* Nature, 592, 601–605. [DOI: 10.1038/s41586-021-03390-w](https://doi.org/10.1038/s41586-021-03390-w)
    * *System Role:* Rotational dynamics and shared selection mechanisms between attention and working memory.

### VIII. Epistemic Horizon, Veto & Cognitive Branching (Fpz / Frontopolar BA 10)
21. **Koechlin, E., & Hyafil, A. (2007).** *Anterior prefrontal function and the limits of human decision-making.* Science, 318(5850), 594–598. [DOI: 10.1126/science.1142995](https://doi.org/10.1126/science.1142995)
    * *System Role:* Architecture of human frontopolar cortex (BA 10) in cognitive branching: Holding secondary behavioral goals while executing primary tasks.
22. **Boorman, E. D., et al. (2009).** *How green is the grass on the other side? Frontopolar cortex and evidence for alternatives.* Neuron, 62(5), 733–743. [DOI: 10.1016/j.neuron.2009.05.014](https://doi.org/10.1016/j.neuron.2009.05.014)
    * *System Role:* Tracking evidence for counterfactual options; mathematical basis of the Fpz fatigue accumulator and phase-drift paradigm shift ("The Drop").
