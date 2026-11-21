# 🌌 EXOCORTEX 3D-DiT: CONTINUOUS REAL-TIME NEUROFEEDBACK WORLD ENGINE, PURE-TENSOR T5 MANIFOLD & RECTIFIED-FLOW 3D-DiT ORACLE

> **100% Continuous Real-Time 3D-DiT World Engine (`LTX-Video 0.9.8-2B-Distilled` NF4 / FP8 & `0.9.0-Base`), 1-Step Batched Rectified-Flow 3D-DiT World-Model Oracle (Full `[1, 128, 4096]` T5 Embeddings, Zero Random Projections), Pure-Tensor Conditioning (Zero Hardcoded Words, Zero Runtime String Concatenation), Shared 96,000-D Working Memory Bus (`FreeEEG16` × 24-Channel Gamepad/Agent Synthesizer × `TBP.Monty`), and Intransitive Catalytic Dynamics**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NF4_%7C_FP8_%7C_INT8_%7C_BF16_Fast_Stream-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![DOI:10.1007/BF02478259](https://img.shields.io/badge/DOI-10.1007%2FBF02478259-00b4d8.svg)](https://doi.org/10.1007/BF02478259)
[![DOI:10.1007/BF01020332](https://img.shields.io/badge/DOI-10.1007%2FBF01020332-ff9f1c.svg)](https://doi.org/10.1007/BF01020332)
[![DOI:10.1016/0022-5193(86)90047-9](https://img.shields.io/badge/DOI-10.1016%2F0022--5193(86)90047--9-2ec4b6.svg)](https://doi.org/10.1016/0022-5193(86)90047-9)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-yellow.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1038/s41562-024-02047-8](https://img.shields.io/badge/DOI-10.1038%2Fs41562--024--02047--8-blue.svg)](https://doi.org/10.1038/s41562-024-02047-8)
[![DOI:10.1016/j.neuron.2024.07.024](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2024.07.024-red.svg)](https://doi.org/10.1016/j.neuron.2024.07.024)
[![DOI:10.1073/pnas.2107797119](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2107797119-green.svg)](https://doi.org/10.1073/pnas.2107797119)
[![arXiv:2507.05888](https://img.shields.io/badge/arXiv-2507.05888-b31b1b.svg)](https://arxiv.org/abs/2507.05888)
[![arXiv:2501.00103](https://img.shields.io/badge/arXiv-2501.00103-ff69b4.svg)](https://arxiv.org/abs/2501.00103)

---

## 📑 Table of Contents
1. [Core Design Laws: Pure-Tensor Engine, Neurofeedback-First & Zero Hardcoded Words](#1-core-design-laws-pure-tensor-engine-neurofeedback-first--zero-hardcoded-words)
2. [Unified Working Memory Bus (`[5 × 5 × 32 × 120]`): EEG, Gamepad & NPC Agents on Equal Footing](#2-unified-working-memory-bus-5--5--32--120-eeg-gamepad--npc-agents-on-equal-footing)
3. [1-Step Batched Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)](#3-1-step-batched-rectified-flow-3d-dit-oracle-neuro_refereepy)
4. [LTX-Video 0.9.8-2B-Distilled In-Place 4-Bit NF4 & Stability Fixes](#4-ltx-video-098-2b-distilled-in-place-4-bit-nf4--stability-fixes)
5. [24-Channel Gamepad/Keyboard WM Synthesizer & Zero-Input Equilibrium](#5-24-channel-gamepadkeyboard-wm-synthesizer--zero-input-equilibrium)
6. [Architecture & Module Map](#6-architecture--module-map)
7. [Quick Start & CLI Reference](#7-quick-start--cli-reference)
8. [Current Status & Next-Session Handoff Notes](#8-current-status--next-session-handoff-notes)
9. [Comprehensive Scientific Bibliography & DOIs](#9-comprehensive-scientific-bibliography--dois)

---

## 1. Core Design Laws: Pure-Tensor Engine, Neurofeedback-First & Zero Hardcoded Words

**Exocortex 3D-DiT** is a **language-agnostic, lore-agnostic continuous real-time world engine** built around four strict invariants:

1. **Zero Hardcoded Words or Prompts in Engine Code:**
   All Python engine modules (`neuro_controller.py`, `neuro_engine.py`, `neuro_referee.py`, `neuro_solver_stepper.py`, `neuro_ltx_backend.py`) are 100% aconceptual mathematical operators. No language-specific prompts, directions, or magic state strings are ever injected into the model. All world semantics come strictly from external JSON manifests (`configs/*.json`) and are encoded once into GPU tensors at warmup.
2. **100% Pure-Tensor Loop (Zero String Concatenation at Runtime):**
   `neuro_engine.py` composes `pos_emb` directly in `[1, 128, 4096]` embedding space via spherical normalization/superposition of pre-encoded `steered_probe_pairs` (or `passive_pair` at zero-input equilibrium), preserving continuous diagonal and multi-region superpositions without ever re-concatenating text strings.
3. **Neurofeedback-First Universal Agent Architecture:**
   Every participant in the world — human EEG (`FreeEEG16`), human gamepad/keyboard emulator, or autonomous NPC — operates through the exact same Working Memory field $\mathbf{W} \in \mathbb{R}^{5 \times 5 \times 32 \times 120}$ and catalytic competition loop. No classic game commands (`if btn: move()`) bypass the Working Memory bus.
4. **Continuous Intra-Step Steering & Zero User Blocking:**
   No cortical region is ever locked into a passive `"HOLD"` state. Geodesic T5 steering (`_live_pos_emb_fn`) runs on every Euler denoising step, and transient impulses are latched via a synaptic trace (`_latched_gamepad_channels`).

---

## 2. Unified Working Memory Bus (`[5 × 5 × 32 × 120]`): EEG, Gamepad & NPC Agents on Equal Footing

```
 ═════════════════════════════════════════════════════════════════════════════
   UNIFIED WORKING MEMORY BUS: GRADUAL MULTI-AGENT SUPERPOSITION
 ═════════════════════════════════════════════════════════════════════════════

  [1. Biological EEG (FreeEEG16 BLE/LSL)] ──(eeg_wm_gain ∈ [0..1])──┐
  [2. 24-Ch Gamepad / Keyboard Synthesizer] ──(gamepad_wm_mix ∈ [0..1])┼─► ┌───────────────────────────────┐
  [3. Autonomous NPC / Monty Agents] ─────────────────────────────────┘   │ SHARED WM FIELD (VRAM IPC)    │
                                                                          │ W_r ∈ R^{5 × 32 × 120}        │
                                                                          └───────────────┬───────────────┘
                                                                                          │
            ┌─────────────────────────────────────────────────────────────────────────────┴──────────────────┐
            ▼ (Orthonormal Mode Projection + Pure-Tensor Spherical Superposition)                            ▼
  ┌──────────────────────────────────────────────────────┐                ┌────────────────────────────────────────┐
  │ PURE-TENSOR T5 CONDITIONING & S³ STEPPER             │                │ 1-STEP RECTIFIED-FLOW 3D-DiT ORACLE    │
  │ • LTX-Video 0.9.8-2B-Distilled (In-Place 4-bit NF4)  │                │ • Evaluates P(z_new | z_prev, H_k)     │
  │ • Official t² anchor jitter (image_cond_noise=0.15)  │───────────────►│ • Full [M, 128, 4096] T5 Hypotheses    │
  │ • Causal Multi-Frame VAE Encode + last_frame_fix     │  unpacked_z0   │ • Shared ε + Motion-Weighted t-stat    │
  └──────────────────────────────────────────────────────┘                └────────────────────────────────────────┘
 ═════════════════════════════════════════════════════════════════════════════
```

* **`eeg_wm_gain` $\in [0.00, 1.00]$:** Scales biological `FreeEEG16` (or background phase clock) contribution. Set to `0.00` for isolated Gamepad/Agent verification.
* **`gamepad_wm_mix` $\in [0.00, 1.00]$:** Scales the 24-channel Gamepad/Keyboard Working Memory Synthesizer. When `gamepad_wm_mix = 0.00` (or stick/keys released), zero gamepad wave is injected.

---

## 3. 1-Step Batched Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)

Instead of a static 2D image classifier (like CLIP, which cannot perceive temporal motion) or a random projection matrix, `CoordinateFreeHistoryReferee` uses **LTX-Video's own 3D-DiT** to judge the actual spatiotemporal transition $\mathbf{z}_{\text{prev}} \to \mathbf{z}_{\text{new}}$:

1. Keeps conditioning prefix frames `:n_cond` clean ($t_c = 0$) and noises only generated frames `n_cond:` at $\sigma_{\text{probe}}$ (`0.725` for `0.9.8-distilled`, `0.55` for `0.9.0-base`) with a **single shared noise tensor** $\boldsymbol{\epsilon}$ (Common Random Numbers) across all hypotheses $H_k \in \{H_{\text{ongoing}}, H_{\text{completed}}, H_{\text{failed}}\}$:
   $$\mathbf{z}_\sigma^{\text{new}} = (1 - \sigma)\mathbf{z}_0^{\text{new}} + \sigma\boldsymbol{\epsilon}, \qquad \mathbf{v}^* = \boldsymbol{\epsilon} - \mathbf{z}_0^{\text{new}}$$
2. Runs **one batched forward pass** of `LTXVideoTransformer3DModel` (`B = 2` or `3`) with **full `[B, 128, 4096]` T5 embeddings** (0 extra VRAM).
3. Computes per-token squared velocity error weighted by actual spatiotemporal motion $w_{\text{motion}} \propto \|\mathbf{z}_0^{\text{new}} - \mathbf{z}_0^{\text{prev}}\|_2$ and a scale-invariant paired $t$-statistic across tokens.
4. Produces unbiased probabilities (`50/50` at neutral, zero hardcoded positive bias, zero mechanical progress ratchet).

---

## 4. LTX-Video 0.9.8-2B-Distilled In-Place 4-Bit NF4 & Stability Fixes

| Issue / Symptom | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **1. `from_single_file` + `BitsAndBytesConfig` fails on `0.9.8-distilled`** | `diffusers` `from_single_file` fails with `_no_split_modules` and `bitsandbytes__*` missing keys when loading raw `.safetensors` into `Linear4bit`. | `quantize_model_inplace_bnb()` in `neuro_ltx_backend.py`: loads `ltxv-2b-0.9.8-distilled.safetensors` on CPU, replaces `nn.Linear` (`in/out >= 512`) layer-by-layer with `bnb.nn.Linear4bit` (NF4) / `Linear8bitLt`, and streams each layer directly to CUDA (~8.0–8.5 GB total VRAM with 4-bit T5). |
| **2. Pitch-black screen / vertical striping at 6–7 steps on `0.9.0-base`** | Double-shifted `linear_quadratic_sigmas(6)` + `mu` shift wasted 4/6 steps above $\sigma=0.97$ and jumped $-0.66 \to 0.0$ in one step. | Default to `ltxv-2b-0.9.8-distilled.safetensors` (matched 0.9.8 Transformer + 0.9.8 VAE + exact distilled sigma schedule `use_dynamic_shifting=False`), plus fixed uniform `np.linspace(1.0, min_sigma, steps)` for `--model base`. |
| **3. Past-frame throwback & boundary seams** | Feeding 8-frame latent into 1-frame slot `z[0]` + 3-frame `torch.roll` noise loop + non-causal decoder tail artifact. | Multi-frame causal `encode_frames_to_latent`, full-window decode from `z[0]` with `last_frame_fix=True`, monotonic S³ noise seeds, and `decode_noise_scale=0.025` for distilled VAE. |
| **4. Object frozen on `z[0]` in 1-frame I2V** | Zero noise on conditioning prefix `z[:n_cond]` glued objects to frame 0. | Added official `image_cond_noise_scale = 0.15` quadratic jitter (`+ 0.15 * noise * t_norm**2` from `pipeline_ltx_condition.py`) inside `neuro_solver_stepper.py`. |

---

## 5. 24-Channel Gamepad/Keyboard WM Synthesizer & Zero-Input Equilibrium

* **24 Bindable Channels (`ALL_GAMEPAD_CHANNELS`):** Left/Right stick half-axes (`LS_*`, `RS_*`), triggers/bumpers (`LT`, `RT`, `LB`, `RB`), D-Pad (`DPAD_*`), and buttons (`BTN_A/B/X/Y`, `L3`, `R3`, `BACK`, `START`).
* **Orthonormal 19,200-D Standing Wave Basis (`get_canonical_concept_wm_basis_Kx5x32x120`):** Simultaneous presses (e.g., diagonals `LS_UP + LS_LEFT`) excite linear superpositions $\mathbf{W}_r^{\text{GP}} = \sum_k a_{r,k}\boldsymbol{\Phi}_{r,k}$ and recover exact mode weights.
* **Zero-Input Equilibrium (Aconceptual):** When $\|\mathbf{W}_r\|_\infty < 10^{-4}$ (nothing pressed and `eeg_wm_gain = 0`), `select_concept_full_rank_524288d` returns empty `active_indices = []` and the spherical mean tensor of the pool, causing `neuro_engine.py` to condition on the manifest's `passive_pair` (`static_hold_desc`) instead of latching onto the last direction.

---

## 6. Architecture & Module Map

| File | Role |
| :--- | :--- |
| **`neuro_battle_server.py`** | Immortal LTX-Video Weights Server (`localhost:6008`) supporting `--model distilled\|base` and `--quant nf4\|fp8\|int8\|bf16`. |
| **`neuro_ltx_backend.py`** | `LTXModelBackend`, `RemoteLTXModelBackend`, `quantize_model_inplace_bnb` (in-place 4-bit NF4 for single-file `.safetensors`), distilled/base sigma schedules, and `encode_frames_to_latent`. |
| **`neuro_controller.py`** | `SymmetricBotController`, orthonormal 19,200-D WM eigenmode synthesizer, zero-input spherical equilibrium, and QR-tangent T5 navigator. |
| **`neuro_solver_stepper.py`** | `LatentFlowWindowStepper`: S³ wave integrator, automatic `CFG=1.0` for distilled, `image_cond_noise_scale=0.15` anchor jitter, and negative-`dt` Euler stepping. |
| **`neuro_referee.py`** | `CoordinateFreeHistoryReferee`: 1-step batched Rectified-Flow 3D-DiT Oracle over full `[M, 128, 4096]` T5 embeddings with motion-weighted paired $t$-stat. |
| **`neuro_engine.py`** | `UniversalLatentAutobattlerEngine`: pure-tensor multi-agent orchestrator (zero string concatenation in the loop), synaptic input latch, and multi-frame causal VAE feedback. |
| **`neuro_run.py`** | Unified launcher with 60 Hz `SharedAmodalCognitiveEngine` BLE/LSL $\to$ VRAM bus, `SmoothRealtimeVideoPacer` (`:6002`), and Web Studio bridge (`:6007`). |
| **`web_cloud_gateway.py` & `static/app.js`** | Unified HTTPS Web Studio (`:8080`) with Gradual WM Mixer, 24-channel HTML5 Gamepad poller, and `🎯` 1-click concept binding. |

---

## 7. Quick Start & CLI Reference

```bash
# Terminal 1: Launch Immortal Weights Server (LTX 0.9.8-2B-Distilled in 4-bit NF4)
python3 neuro_battle_server.py --port 6008 --model distilled --quant nf4 --allow-t5-offload

# Terminal 2: Launch Unified Web Studio Gateway on :8080
python3 web_cloud_gateway.py --port 8080 --cert cert.pem --key key.pem --video-lore configs/mode_05_wm_control_test_arena.json

# Terminal 3: Launch Continuous Real-Time 3D-DiT Engine
# Recommended: --steps 8 (full 8-sigma distilled schedule)
python3 neuro_run.py --mode ball --context-latents 1 --step-latents 2 --steps 8 --cfg 1.0 --stochastic 0.0 --wm-radius 0.35 --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0
```

---

## 8. Current Status & Next-Session Handoff Notes

### ✅ Verified Working (as of Nov 20, 2026)
* **`0.9.8-2B-Distilled` + In-Place NF4 (`neuro_ltx_backend.py`):** Loads cleanly into 12 GB VRAM (`RTX 3060`) alongside 4-bit T5-XXL. Produces **1+ minute of razor-sharp, artifact-free video** with **zero vertical striping and zero grey washout** (`00:00–01:09` verified).
* **Pure-Tensor Pipeline & Zero-Input Equilibrium:** `neuro_controller.py` and `neuro_engine.py` cleanly decode `up/down/left/right: 100%` on press and return to equal-weight equilibrium (`25%` each -> `passive_pair`) on release, with zero hardcoded words in Python.
* **1-Step Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`):** Runs in ~150 ms with 0 extra VRAM on full `[128, 4096]` T5 embeddings and honestly rejects mismatched / static scenes without fake progress inflation.

### ⚠️ Open Engineering Issues to Address Next Session
1. **Weak Directional Prompt Adherence in `mode_05` at `cfg=1.0` / `steps=7`:**
   * At `steps=7`, `build_distilled_sigmas(7)` dropped the 8th sigma (`0.4219`). Always test with **`--steps 8`**.
   * At `cfg=1.0` (no CFG, currently forced when `is_distilled=True` in `neuro_solver_stepper.py`), `0.9.8-distilled` preserves the sharp white sphere cleanly for 70s, **but weakly obeys directional text changes** on a featureless black background (no visual landmarks for optical flow).
   * **Next experiments to run:**
     1. Allow optional mild CFG (`--cfg 1.5..2.5`) on `distilled` in `neuro_solver_stepper.py` (remove `not is_distilled` hard override) to test whether CFG restores strong directional steering.
     2. Compare pure-tensor `0.78 * c_emb + 0.22 * a_emb` blending vs. pure `c_emb` (`e0` alone, without blending `e1`'s static zone description into `pos_emb`, since `e1_s0` says `"waiting for the ball"` and pulls the prompt toward stillness!).
     3. Test `0.9.8-distilled` + new RF Oracle on richer semantic/catalytic worlds (`--mode maze` / `mode_03_semantic_maze_gm.json` and `--mode ritual` / `mode_04_forest_neuro_ritual.json`) for the Crocus Expo demo (Nov 24–26).
2. **Real EEG vs. Gamepad Basis (`neuro_controller.py`):**
   * Current `get_canonical_concept_wm_basis_Kx5x32x120` uses synthetic sinusoids (100% exact for gamepad emulation, noisy for raw non-invasive `FreeEEG16` EEG). Next step for live EEG: replace fixed sinusoids with online-calibrated subject subspaces / physiological Gamma-vs-Beta gating (Miller et al., 2018; Chen et al., 2024) + EMG check on 70–100 Hz.
3. **Repo Hygiene:** Remove self-signed `key.pem` from git tracking and archive legacy duplicate video stack (`video_world_server.py` / `video_render_logic.py`) once `neuro_battle_server.py` + `neuro_engine.py` is finalized.

---

## 9. Comprehensive Scientific Bibliography & DOIs

1. <a id="ref-1"></a>**McCulloch, W. S. (1945).** *A heterarchy of values determined by the topology of nervous nets.* The Bulletin of Mathematical Biophysics, 7(2), 89–93. [https://doi.org/10.1007/BF02478259](https://doi.org/10.1007/BF02478259)
2. <a id="ref-2"></a>**Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** *The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy.* arXiv:2507.05888. [https://doi.org/10.48550/arXiv.2507.05888](https://doi.org/10.48550/arXiv.2507.05888)
3. <a id="ref-3"></a>**Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [https://doi.org/10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
4. <a id="ref-4"></a>**Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [https://doi.org/10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
5. <a id="ref-5"></a>**Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [https://doi.org/10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
6. <a id="ref-6"></a>**Feigenbaum, M. J. (1978).** *Quantitative universality for a class of nonlinear transformations.* Journal of Statistical Physics, 19(1), 25–52. [https://doi.org/10.1007/BF01020332](https://doi.org/10.1007/BF01020332)
7. <a id="ref-7"></a>**May, R. M. (1976).** *Simple mathematical models with very complicated dynamics.* Nature, 261(5560), 459–467. [https://doi.org/10.1038/261459a0](https://doi.org/10.1038/261459a0)
8. <a id="ref-8"></a>**Kauffman, S. A. (1986).** *Autocatalytic sets of proteins.* Journal of Theoretical Biology, 119(1), 1–24. [https://doi.org/10.1016/0022-5193(86)90047-9](https://doi.org/10.1016/0022-5193(86)90047-9)
9. <a id="ref-9"></a>**Soloveichik, D., Seelig, G., & Winfree, E. (2010).** *DNA as a universal substrate for chemical kinetics.* PNAS, 107(12), 5393–5398. [https://doi.org/10.1073/pnas.0909380107](https://doi.org/10.1073/pnas.0909380107)
10. <a id="ref-10"></a>**Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011. [https://doi.org/10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
11. <a id="ref-11"></a>**Dickey, C. W., et al. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* PNAS, 119(28), e2107797119. [https://doi.org/10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
12. <a id="ref-12"></a>**HaCohen, Y., et al. (2024).** *LTX-Video: Realtime Video Latent Diffusion.* arXiv:2501.00103. [https://doi.org/10.48550/arXiv.2501.00103](https://doi.org/10.48550/arXiv.2501.00103)
13. <a id="ref-13"></a>**Yang, C., et al. (2025).** *Reasoning via Video: The First Evaluation of Video Models’ Reasoning Abilities through Maze-Solving Tasks (VR-Bench).* arXiv:2511.15065. [https://doi.org/10.48550/arXiv.2511.15065](https://doi.org/10.48550/arXiv.2511.15065)
14. <a id="ref-14"></a>**Mongillo, G., Barak, O., & Tsodyks, M. (2008).** *Synaptic Theory of Working Memory.* Science, 319(5869), 1543–1546. [https://doi.org/10.1126/science.1150769](https://doi.org/10.1126/science.1150769)

