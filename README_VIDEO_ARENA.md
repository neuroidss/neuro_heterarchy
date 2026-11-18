# 🌌 EXOCORTEX 3D-DiT: CONTINUOUS REAL-TIME NEUROFEEDBACK WORLD ENGINE, HYBRID S³ + T5 CONCEPT MANIFOLD & GRADUAL WORKING MEMORY SYNTHESIS

> **100% Continuous Real-Time 3D-DiT Neurofeedback Simulator (`LTX-Video`), Zero Turn-Based Blocking (All Cortical Regions Active on Every Denoising Sub-Step), On-Manifold Shared VAE Denoising Projection (*HaCohen et al., 2024*), Orthonormal 19,200-D Working Memory Eigenmode Resonance, Gradual Multi-Agent Working Memory Mixer (`FreeEEG16` × 24-Channel Gamepad/Agent Synthesizer × `TBP.Monty`), and Intransitive Autocatalytic Dynamics ($\delta = 4.6692016$)**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-FP8_%7C_INT8_%7C_NF4_%7C_BF16_Fast_Stream-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![DOI:10.1007/BF02478259](https://img.shields.io/badge/DOI-10.1007%2FBF02478259-00b4d8.svg)](https://doi.org/10.1007/BF02478259)
[![DOI:10.1007/BF01020332](https://img.shields.io/badge/DOI-10.1007%2FBF01020332-ff9f1c.svg)](https://doi.org/10.1007/BF01020332)
[![DOI:10.1016/0022-5193(86)90047-9](https://img.shields.io/badge/DOI-10.1016%2F0022--5193(86)90047--9-2ec4b6.svg)](https://doi.org/10.1016/0022-5193(86)90047-9)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-yellow.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016/j.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![arXiv:2507.05888](https://img.shields.io/badge/arXiv-2507.05888-b31b1b.svg)](https://arxiv.org/abs/2507.05888)
[![arXiv:2501.00103](https://img.shields.io/badge/arXiv-2501.00103-ff69b4.svg)](https://arxiv.org/abs/2501.00103)

---

## 📑 Table of Contents
1. [Core Design Law: Continuous Real-Time Neurofeedback (Zero Turn-Based Windows, Zero User Blocking)](#1-core-design-law-continuous-real-time-neurofeedback-zero-turn-based-windows-zero-user-blocking)
2. [Unified Working Memory Field (`[5 × 5 × 32 × 120]` = 96,000-D): How EEG, Gamepad, Monty & AI Agents Coexist](#2-unified-working-memory-field-5--5--32--120--96000-d-how-eeg-gamepad-monty--ai-agents-coexist)
3. [Mathematical Fixes for 100% Visual Stability & Instant Concept Responsiveness](#3-mathematical-fixes-for-100-visual-stability--instant-concept-responsiveness)
4. [24-Channel Gamepad/Keyboard Working Memory Synthesizer & Co-Catalytic Superposition (Diagonals & Multi-Region)](#4-24-channel-gamepadkeyboard-working-memory-synthesizer--co-catalytic-superposition-diagonals--multi-region)
5. [Intransitive Mutable Catalysts & Free Latent Drift When Concept Pools Are Empty](#5-intransitive-mutable-catalysts--free-latent-drift-when-concept-pools-are-empty)
6. [Architecture & Module Map](#6-architecture--module-map)
7. [Quick Start & CLI Reference](#7-quick-start--cli-reference)
8. [Comprehensive Scientific Bibliography & DOIs](#8-comprehensive-scientific-bibliography--dois)

---

## 1. Core Design Law: Continuous Real-Time Neurofeedback (Zero Turn-Based Windows, Zero User Blocking)

**Exocortex 3D-DiT** is a **real-time neurofeedback world simulator**, not a turn-based strategy game. Three strict architectural invariants govern user and agent control:

1. **Zero User Blocking Across Stages:**
   No cortical region (`FCz`, `AFz`, `F3`, `F4`, `Fpz`) is ever locked into a passive `"HOLD"` state that ignores user input. Every region continuously projects its active concept(s) and working memory wave into the 3D-DiT scene on every frame. Catalytic stages modulate how regions interact and transmute one another, but **never block or mute the user's active region**.
2. **Intra-Step Sub-150ms Real-Time Control (`_live_pos_emb_fn` on Every Euler Step):**
   A 3D-DiT denoising window of 6 Euler steps takes ~900–1100 ms. Waiting until the next window to read working memory or user input introduces a 1-second lag and drops brief inputs. Instead:
   * Both **concept selection (`select_concept_full_rank_524288d`)** and **geodesic T5 steering (`steer_semantic_probe_full_rank`)** are re-evaluated **inside `_live_pos_emb_fn(step_i, total_steps)` on every single Euler denoising step $i \in \{0 \dots S-1\}$** (~150 ms resolution), immediately steering the 3D-DiT velocity field $\mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{E}_{\text{T5}}(t_i))$ mid-flight!
   * **Synaptic Calcium Trace Latch (`_latched_gamepad_channels`):** Following *Mongillo, Barak, & Tsodyks (Science 2008)* [[14]](#ref-14), transient control impulses are held in a decaying synaptic trace so no short keypress, stick flick, or gamma burst is ever missed between frames.
3. **Canonical Pool Ordering Across All Stages:**
   Each entity's concept pool `allowed_pool` maintains a **strictly invariant canonical index order (`#1, #2, #3, #4`)** across all stages (`stages[0..N]`), ensuring 1-to-1 consistency between Web Studio concept chips, working memory eigenmodes $\boldsymbol{\Phi}_{r, k}$, and gamepad bindings.

---

## 2. Unified Working Memory Field (`[5 × 5 × 32 × 120]` = 96,000-D): How EEG, Gamepad, Monty & AI Agents Coexist

In **Exocortex 3D-DiT**, **no controller, gamepad, or AI agent is allowed to hardcode prompts or bypass Working Memory**. Every control source is an **Agent that writes waves into the shared 96,000-D Working Memory Tensor** $\mathbf{W} \in \mathbb{R}^{5 \times 5 \times 32 \times 120}$ (`5 Regions × 5 Oscillatory Bands × 32 Causal Time-Frequency Slots × 120 Dipoles`):

```
 ═════════════════════════════════════════════════════════════════════════════
   UNIFIED 96,000-D WORKING MEMORY BUS: GRADUAL MULTI-AGENT SUPERPOSITION
 ═════════════════════════════════════════════════════════════════════════════

  [1. Biological EEG (FreeEEG16 BLE/LSL)] ──(eeg_wm_gain ∈ [0..1])──┐
  [2. 24-Ch Gamepad / Keyboard Synthesizer] ──(gamepad_wm_mix ∈ [0..1])┼─► ┌───────────────────────────────┐
  [3. Self-Learning Monty Columns (TBT 2.0)] ─────────────────────────┤   │ SHARED 96,000-D WM FIELD      │
  [4. External AI Agents (Local Heuristic / Cloud VLM)] ──────────────┘   │ W_r ∈ R^{5 × 32 × 120} (VRAM) │
                                                                          └───────────────┬───────────────┘
                                                                                          │
                          ┌───────────────────────────────────────────────────────────────┴───────────────┐
                          ▼ (Orthonormal Eigenmode Resonance: <W_r, Φ_{r,k}>)                             ▼
  ┌───────────────────────────────────────────────────────┐               ┌───────────────────────────────┐
  │ PURE WORKING MEMORY CONCEPT DECODER & T5 NAVIGATOR    │               │ COHERENT S³ WAVE INTEGRATOR   │
  │ • Projects W_r onto K Orthonormal Modes Φ_{r,k}       │               │ • [1, 128, F, 1, 1] Phase     │
  │ • Co-Catalytic Superposition of Simultaneously Active │               │ • Quadratic Context Vibration │
  │   Modes (e.g. 50% UP + 50% LEFT -> Diagonal Vector!)  │               │ • On-Manifold VAE Projection  │
  │ • QR-Tangent Rotation strictly inside T5 Manifold     │               │   (Zero Grey Washout / Fog)   │
  └───────────────────────────────────────────────────────┘               └───────────────────────────────┘
 ═════════════════════════════════════════════════════════════════════════════
```

### Gradual Non-Binary Mixing (`eeg_wm_gain` × `gamepad_wm_mix`)
Instead of a brittle binary switch that can shock the latent space on transition, the system uses two continuous sliders in the Web Studio (`:8080`):
* **`eeg_wm_gain` $\gamma_{\text{EEG}} \in [0.00, 1.00]$ (Default `1.00`):** Scales the biological `FreeEEG16` (or background phase clock) contribution. Setting `eeg_wm_gain = 0.00` silences background drift completely so you can test **100% isolated Gamepad / AI Agent control**.
* **`gamepad_wm_mix` $\gamma_{\text{GP}} \in [0.00, 1.00]$ (Default `1.00`):** Scales the Gamepad/Agent Working Memory Synthesizer.
  * When `gamepad_wm_mix = 0.00`, the gamepad synthesizer is **100% disabled** (zero effect on the system).
  * When `gamepad_wm_mix > 0.00` and the gamepad is **released** ($\|\mathbf{a}_{\text{GP}}\| = 0$), the synthesized wave is zero ($\mathbf{W}_r^{\text{GP}} = 0$), so **100% unmodified EEG Working Memory** drives the world!
  * As soon as any stick half-axis or button is pressed with intensity $a \in (0, 1]$, the synthesized eigenmode wave $\mathbf{W}_r^{\text{GP}}$ smoothly blends into $\mathbf{W}_r$ on the sphere.

---

## 3. Mathematical Fixes for 100% Visual Stability & Instant Concept Responsiveness

| Issue / Symptom | Root Cause | Exact Mathematical Solution Implemented |
| :--- | :--- | :--- |
| **1. Rainbow blotches at `00:00` in 3D-DiT** | Passing `per_token_timesteps` to `FlowMatchEulerDiscreteScheduler.step(noise_pred, ...)` without negating `-noise_pred` flipped `dt = current_sigma - next_sigma > 0`, adding noise instead of denoising. | Step generated frames `latents[:, :, n_cond:]` via standard `scheduler.step(noise_pred[:, :, n_cond:], t, latents[:, :, n_cond:])` (`dt = sigma_next - sigma < 0`) with `shift_terminal = None`. |
| **2. Progressive grey washout over time** | Iterative `tone_map_latents` compression + raw transformer latent feedback bypassing the Denoising VAE Decoder (`decode_t = 0.05`). | Re-encode `clean_video_frames[-1]` (after Denoising VAE `decode_t = 0.05`, `decode_noise_scale = 0.0`) via `encode_frame_to_latent` + gentle AdaIN (`factor = 0.15`) to the genesis reference. |
| **3. Concept selection stuck on `s0` (`#1`)** | 1) Zero-mean averaging `wm_flat[k::K].mean() ~ 0`.<br>2) `stages[1..3]` had permuted `allowed_pool` lists in JSON.<br>3) High cosine similarity of `catalyst_field` in T5 overwhelmed WM scores.<br>4) Arrow keys were bound to `RS_*` (`AFz`) while viewing `FCZ`. | 1) Orthonormal 19,200-D eigenmodes $\langle \boldsymbol{\Phi}_{r,k}, \boldsymbol{\Phi}_{r,m}\rangle = \delta_{km}$ recover exact mode weights $\alpha_k = \langle \mathbf{W}_r, \boldsymbol{\Phi}_{r,k}\rangle$.<br>2) Canonical `stages[0]` pool ordering across all stages.<br>3) Direct positive-resonance allocation $p_k \propto \max(0, s_k - \tau)$.<br>4) Keyboard `WASD` & `Arrow Keys` dynamically route to the currently selected region in the Web Studio! |

---

## 4. 24-Channel Gamepad/Keyboard Working Memory Synthesizer & Co-Catalytic Superposition (Diagonals & Multi-Region)

Every concept $k \in \{0 \dots K-1\}$ in cortical region $r \in \{\text{FCz}, \text{AFz}, \text{F3}, \text{F4}, \text{Fpz}\}$ is assigned a strictly orthonormal 19,200-D standing wave eigenmode $\boldsymbol{\Phi}_{r, k} \in \mathbb{R}^{5 \times 32 \times 120}$ via QR decomposition (`get_canonical_concept_wm_basis_Kx5x32x120`).

* **24 Bindable Hardware Channels (`ALL_GAMEPAD_CHANNELS`):**
  * **Left Stick Half-Axes:** `LS_UP`, `LS_DOWN`, `LS_LEFT`, `LS_RIGHT`
  * **Right Stick Half-Axes:** `RS_UP`, `RS_DOWN`, `RS_LEFT`, `RS_RIGHT`
  * **Analog Triggers & Bumpers:** `LT`, `RT`, `LB`, `RB`
  * **D-Pad:** `DPAD_UP`, `DPAD_DOWN`, `DPAD_LEFT`, `DPAD_RIGHT`
  * **Face & System Buttons:** `BTN_A`, `BTN_B`, `BTN_X`, `BTN_Y`, `L3`, `R3`, `BACK`, `START`
* **1-Click Interactive Binding (`🎯` Button on Every Concept Chip):**
  Click `🎯` next to any concept in the Web Studio and push any stick direction (or hold two axes simultaneously for a diagonal combo like `LS_UP+LS_LEFT`) or press any gamepad button — the browser captures the channel and binds it to that concept's working memory eigenmode $\boldsymbol{\Phi}_{r, k}$.
* **Simultaneous Multi-Concept & Multi-Region Superposition (Co-Catalysis):**
  When multiple channels are active simultaneously (e.g., pushing the stick diagonally `LS_UP = 0.71` + `LS_LEFT = 0.71` in `FCz` while holding `BTN_A = 1.0` in `F3`), the synthesizer excites the linear superposition of those standing waves in Working Memory:
  $$\mathbf{W}_r^{\text{GP}} = \sum_{k=0}^{K-1} a_{r, k} \, \boldsymbol{\Phi}_{r, k}$$
  The Working Memory decoder projects $\mathbf{W}_r$ onto $\{\boldsymbol{\Phi}_{r, k}\}$, recovers $p_{\text{UP}} = 50\%$ and $p_{\text{LEFT}} = 50\%$, highlights **both** active chips with `★` in the Web UI, and computes the geodesic spherical superposition of both T5 embeddings!

---

## 5. Intransitive Mutable Catalysts & Free Latent Drift When Concept Pools Are Empty

* **Concepts as Mutable Catalysts:** In each region, the selected concept(s) act as a **Catalyst** that drives state transitions in downstream regions (`e0 -> e1 -> e2 -> e3 -> e0`) while simultaneously contributing directly to the active visual scene.
* **Free Latent Drift (`🧹 Clear Region` / Empty Pool `[]`):** Clicking `🧹 Очистить регион (Свободный дрейф)` removes all text concepts from that region, setting its T5 anchor to `""` (`empty_embed`) so the 96,000-D working memory wave steers the 3D-DiT $S^3$ phase manifold in pure aconceptual mode.

---

## 6. Architecture & Module Map

| File | Role in the Heterarchy |
| :--- | :--- |
| **`neuro_battle_server.py`** | Immortal LTX-Video Weights Server (`localhost:6008`) with resident 4-bit NF4 T5-XXL, 3D-DiT (`fp8`/`int8`/`nf4`/`bf16`), and 3D-VAE (`encode_frame` + `decode_latents`). |
| **`neuro_ltx_backend.py`** | `LTXModelBackend` & `RemoteLTXModelBackend` with `shift_terminal=None`, `decode_t=0.05`, `decode_noise_scale=0.0`, and `encode_frame_to_latent`. |
| **`neuro_controller.py`** | `SymmetricBotController` + `get_canonical_concept_wm_basis_Kx5x32x120` + `synthesize_region_gamepad_wm_5x32x120`: orthonormal 19,200-D WM eigenmode synthesizer and QR-tangent T5 navigator. |
| **`neuro_solver_stepper.py`** | `LatentFlowWindowStepper`: hybrid I2V + $S^3$ integrator with `[1, 128, F, 1, 1]` travelling waves, quadratic timestep vibration on context frames, and standard negative-`dt` Euler stepping. |
| **`neuro_engine.py`** | `UniversalLatentAutobattlerEngine`: continuous real-time multi-region orchestrator with synaptic input latch (`_latched_gamepad_channels`), canonical pool ordering, and On-Manifold VAE projection. |
| **`neuro_run.py`** | Unified launcher with 60 Hz `SharedAmodalCognitiveEngine` BLE/LSL $\to$ VRAM bus thread, `SmoothRealtimeVideoPacer` (`:6002`), and Web Studio control bridge (`:6007`). |
| **`web_cloud_gateway.py` & `static/app.js`** | Unified HTTPS Web Studio (`:8080`) with Gradual WM Mixer (`EEG WM Gain` × `Gamepad WM Mix`), 24-channel HTML5 Gamepad poller, and `🎯` 1-click concept binding. |

---

## 7. Quick Start & CLI Reference

```bash
# Terminal 1: Launch Immortal Weights Server in VRAM (T5-XXL resident in VRAM)
python3 neuro_battle_server.py --port 6008 --quant fp8

# Terminal 2: Launch Unified Web Studio Gateway on :8080
python3 web_cloud_gateway.py --port 8080 --cert cert.pem --key key.pem --video-lore configs/mode_03_semantic_maze_gm.json

# Terminal 3: Launch Continuous Real-Time 3D-DiT Engine
# Use --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0 to test pure isolated Gamepad/Keyboard WM synthesis,
# or --eeg-wm-gain 1.0 --gamepad-wm-mix 1.0 for automatic gradual mixing!
python3 neuro_run.py --mode maze --context-latents 1 --step-latents 2 --steps 6 --cfg 3.2 --stochastic 0.0 --wm-radius 0.50 --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0
```

---

## 8. Comprehensive Scientific Bibliography & DOIs

1. <a id="ref-1"></a>**McCulloch, W. S. (1945).** *A heterarchy of values determined by the topology of nervous nets.* The Bulletin of Mathematical Biophysics, 7(2), 89–93. [https://doi.org/10.1007/BF02478259](https://doi.org/10.1007/BF02478259)
2. <a id="ref-2"></a>**Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** *The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy.* arXiv:2507.05888. [https://doi.org/10.48550/arXiv.2507.05888](https://doi.org/10.48550/arXiv.2507.05888)
3. <a id="ref-3"></a>**Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [https://doi.org/10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
4. <a id="ref-4"></a>**Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [https://doi.org/10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
5. <a id="ref-5"></a>**Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [https://doi.org/10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
6. <a id="ref-6"></a>**Feigenbaum, M. J. (1978).** *Quantitative universality for a class of nonlinear transformations.* Journal of Statistical Physics, 19(1), 25–52. [https://doi.org/10.1007/BF01020332](https://doi.org/10.1007/BF01020332)
7. <a id="ref-7"></a>**May, R. M. (1976).** *Simple mathematical models with very complicated dynamics.* Nature, 261(5560), 459–467. [https://doi.org/10.1038/261459a0](https://doi.org/10.1038/261459a0)
8. <a id="ref-8"></a>**Kauffman, S. A. (1986).** *Autocatalytic sets of proteins.* Journal of Theoretical Biology, 119(1), 1–24. [https://doi.org/10.1016/0022-5193(86)90047-9](https://doi.org/10.1016/0022-5193(86)90047-9)
9. <a id="ref-9"></a>**Soloveichik, D., Seelig, G., & Winfree, E. (2010).** *DNA as a universal substrate for chemical kinetics.* Proceedings of the National Academy of Sciences (PNAS), 107(12), 5393–5398. [https://doi.org/10.1073/pnas.0909380107](https://doi.org/10.1073/pnas.0909380107)
10. <a id="ref-10"></a>**Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011. [https://doi.org/10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
11. <a id="ref-11"></a>**Dickey, C. W., et al. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* PNAS, 119(28), e2107797119. [https://doi.org/10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
12. <a id="ref-12"></a>**HaCohen, Y., et al. (2024).** *LTX-Video: Realtime Video Latent Diffusion.* arXiv:2501.00103. [https://doi.org/10.48550/arXiv.2501.00103](https://doi.org/10.48550/arXiv.2501.00103)
13. <a id="ref-13"></a>**Yang, C., et al. (2025).** *Reasoning via Video: The First Evaluation of Video Models’ Reasoning Abilities through Maze-Solving Tasks (VR-Bench).* arXiv:2511.15065. [https://doi.org/10.48550/arXiv.2511.15065](https://doi.org/10.48550/arXiv.2511.15065)
14. <a id="ref-14"></a>**Mongillo, G., Barak, O., & Tsodyks, M. (2008).** *Synaptic Theory of Working Memory.* Science, 319(5869), 1543–1546. [https://doi.org/10.1126/science.1150769](https://doi.org/10.1126/science.1150769)

