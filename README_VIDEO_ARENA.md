# 🌌 EXOCORTEX 3D-DiT: CONTINUOUS REAL-TIME NEUROFEEDBACK WORLD ENGINE, MULTI-BRANCH 3D-DiT ORACLE & AUTONOMOUS WM AGENTS

> **100% Continuous Real-Time 3D-DiT World Engine (`LTX-Video 0.9.8-2B-Distilled` NF4 / FP8 & `0.9.0-Base`), Automatic `T5 ↔ VAE` VRAM Swap (Zero OOM on 12 GB GPUs), Multi-Branch Rectified-Flow 3D-DiT World-Model Oracle (Full `[M, 128, 4096]` T5 Embeddings), Universal Aconceptual `npc:` Agents (Zero Hardcoded Words), Pure-Tensor Conditioning, and Shared 96,000-D Working Memory Bus (`FreeEEG16` × 24-Channel Gamepad Synthesizer × `TBP.Monty`)**

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
1. [Core Design Laws: Zero Hardcoded Words, Pure-Tensor Engine & Dual Mode](#1-core-design-laws-zero-hardcoded-words-pure-tensor-engine--dual-mode)
2. [Unified Working Memory Bus (`[5 × 5 × 32 × 120]`) & Autonomous `npc:` Agents](#2-unified-working-memory-bus-5--5--32--120--autonomous-npc-agents)
3. [Multi-Branch Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)](#3-multi-branch-rectified-flow-3d-dit-oracle-neuro_refereepy)
4. [LTX-Video 0.9.8-2B-Distilled NF4 & `T5 ↔ VAE` Zero-OOM Architecture](#4-ltx-video-098-2b-distilled-nf4--t5--vae-zero-oom-architecture)
5. [Canonical Module Map & Ports](#5-canonical-module-map--ports)
6. [Quick Start & CLI Reference](#6-quick-start--cli-reference)
7. [Current Status & Next-Session Handoff Notes](#7-current-status--next-session-handoff-notes)
8. [Comprehensive Scientific Bibliography & DOIs](#8-comprehensive-scientific-bibliography--dois)

---

## 1. Core Design Laws: Zero Hardcoded Words, Pure-Tensor Engine & Dual Mode

**Exocortex 3D-DiT** is a **100% aconceptual, language-agnostic, lore-agnostic continuous real-time world engine** governed by four strict invariants:

1. **Zero Hardcoded Words in Python Engine Code:**
   All engine modules (`neuro_controller.py`, `neuro_engine.py`, `neuro_referee.py`, `neuro_solver_stepper.py`, `neuro_ltx_backend.py`) are pure mathematical tensor operators. No directions (`up/down`), game concepts (`enemy/duel/maze/hp`), or fallback prompts exist anywhere in Python. All semantics live exclusively in external JSON manifests (`configs/*.json`) or `.txt` lores (`lore/*.txt`).
2. **100% Pure-Tensor Runtime Loop (Zero String Concatenation):**
   All manifest concepts and hypotheses are encoded once into `[1, 128, 4096]` T5 tensors at warmup. During real-time generation, `neuro_engine.py` composes `pos_emb` solely via spherical superposition of **actively driven** entity tensors (falling back to `passive_pair` at zero-input equilibrium), with zero dilution from inactive/passive slots.
3. **Dual Mode in One Unified Stack:**
   * **Closed-Loop Catalytic World Mode (`.json` manifests with hypotheses):** Enables the Multi-Branch 3D-DiT RF Oracle, real-time `BRANCH_SWITCH` / `STAGE_DONE` transitions, and adaptive `npc:` feedback loops (movement, mazes, rituals, 1v1/NvN duels).
   * **Fast Open-Loop Pure WM Mode (`.txt` lores or 1-stage manifests without `completed_hypothesis`):** Automatically skips the 3D-DiT Oracle forward pass (~350 ms saved per step) for pure continuous 3D-DiT neuro-visual streaming (e.g., `lore/video_lore_psy.txt`).
4. **Custom Initial Scene Injection (`--init-image` & `/api/video/init_image`):**
   Any starting PNG/JPG (such as a 2D orthographic maze map) can be encoded directly into `z[0]` (`local_context_latents_5d`) from CLI or the Web Studio.

---

## 2. Unified Working Memory Bus (`[5 × 5 × 32 × 120]`) & Autonomous `npc:` Agents

```
 ═════════════════════════════════════════════════════════════════════════════
   UNIFIED WORKING MEMORY BUS: SYMMETRIC HUMAN + GAMEPAD + AUTONOMOUS NPC
 ═════════════════════════════════════════════════════════════════════════════

  [1. Biological EEG (FreeEEG16 BLE/LSL)] ──(eeg_wm_gain ∈ [0..1])──┐
  [2. 24-Ch Gamepad / Keyboard Synth] ──────(gamepad_wm_mix ∈ [0..1])┼─► ┌───────────────────────────────┐
  [3. Autonomous NPC Agents ("npc:AFz")] ──(Oracle Reward -> WM)────┘   │ SHARED WM FIELD (VRAM IPC)    │
                                                                        │ W_r ∈ R^{5 × 32 × 120}        │
                                                                        └───────────────┬───────────────┘
                                                                                        │
            ┌───────────────────────────────────────────────────────────────────────────┴────────────────────┐
            ▼ (Orthonormal 19,200-D Eigenmodes + Active-Slot Spherical Superposition)                        ▼
  ┌──────────────────────────────────────────────────────┐                ┌────────────────────────────────────────┐
  │ PURE-TENSOR T5 CONDITIONING & S³ STEPPER             │                │ MULTI-BRANCH 3D-DiT RF ORACLE          │
  │ • LTX-Video 0.9.8-2B-Distilled (In-Place 4-bit NF4)  │                │ • Scores [H_stay, H_1..H_M] on video   │
  │ • Full 8-sigma schedule + optional mild CFG (1.0-2.0)│───────────────►│ • 2-probe CRN ε + effective DoF t-stat │
  │ • Official t² anchor jitter (image_cond_noise=0.15)  │  unpacked_z0   │ • Drives BRANCH_SWITCH & NPC reward    │
  └──────────────────────────────────────────────────────┘                └────────────────────────────────────────┘
 ═════════════════════════════════════════════════════════════════════════════
```

* **Symmetric Autonomous NPC / Adversary Agents (`control_binding: "npc:AFz"`):**
  Every entity slot (`e0`, `e1`, ...) uses the exact same `SymmetricBotController` class and orthonormal 19,200-D eigenmode basis (`get_canonical_concept_wm_basis_Kx5x32x120`). Slots marked `"npc:<REGION>"` maintain a latent preference vector `npc_logits_K` over their `allowed_pool`, synthesize a 19,200-D WM wave with deterministic phase exploration, and update their policy via `observe_oracle_feedback(reward)` based on which branch actually prevailed in the generated video. Zero hardcoded "enemy" or "duel" logic exists in the engine.
* **Per-Slot Region Isolation:**
  `--entity-source auto` strictly preserves each slot's JSON `control_binding` (`eeg:FCz`, `eeg:AFz`, `npc:AFz`, etc.) without overwriting regions.

---

## 3. Multi-Branch Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)

Instead of a static 2D classifier or single-stage lock, `CoordinateFreeHistoryReferee` uses **LTX-Video's own 3D-DiT** to evaluate all competing world branches directly on the generated spatiotemporal transition $\mathbf{z}_{\text{prev}} \to \mathbf{z}_{\text{new}}$:

1. Keeps prefix frames `:n_cond` clean ($t_c = 0$) and noises generated frames `n_cond:` at $\sigma_{\text{probe}}$ (`0.725` for `0.9.8-distilled`, `0.55` for `0.9.0-base`).
2. Evaluates `[H_stay, H_branch_1, ..., H_branch_M]` over full `[1, 128, 4096]` T5 embeddings in VRAM-safe chunks (`max_batch=4`), averaged across `n_noise_probes=2` shared noise draws $\boldsymbol{\epsilon}$ (Common Random Numbers).
3. Computes motion-weighted paired $t$-statistics using effective spatial-temporal degrees of freedom (`n_eff = n_tok / 16.0`), eliminating neutral-scene probability jitter.
4. Produces live video-grounded `reaction_branch_scores` (`branches`), triggers seamless `BRANCH_SWITCH` when another branch clearly dominates in the video, and routes unbiased `±reward` to `npc:` controllers.

---

## 4. LTX-Video 0.9.8-2B-Distilled NF4 & `T5 ↔ VAE` Zero-OOM Architecture

| Component / Mechanism | Implementation (`neuro_ltx_backend.py` & `video_world_server.py`) |
| :--- | :--- |
| **In-Place 4-Bit NF4 Quantization** | `quantize_model_inplace_bnb()` loads `ltxv-2b-0.9.8-distilled.safetensors` on CPU and streams 284 `bnb.nn.Linear4bit` (NF4) layers onto CUDA (~1.6 GB VRAM for the 2B 3D-DiT). |
| **Automatic `T5 ↔ VAE` VRAM Swap** | `T5-XXL 4-bit` (`torch_dtype=bfloat16`) and `0.9.8 3D-VAE` are mutually exclusive in VRAM when `--allow-t5-offload` is active (default): during text encoding, `VAE` moves to CPU (~0.1s, peak `~7.0 GB` total); before any Transformer/VAE step, `T5` is evicted and `VAE` returns to CUDA (`3.27 GB` server VRAM). Zero OOM on startup and on live browser concept switches on 12 GB RTX 3060! |
| **3D-VAE Tiling + Causal Feedback** | `vae_tile_min_px=256` enabled automatically on `<=16 GB` GPUs; `decode_noise_scale=0.025`, `last_frame_fix=True`, multi-frame causal re-encode into `local_context_latents_5d`, and `image_cond_noise_scale=0.15` quadratic anchor jitter. |
| **Exact Distilled Sigma Schedule** | `build_distilled_sigmas(steps)` uses all 8 official sigmas `[1.0000 ... 0.4219]` at `--steps 8` and interpolates preserving both endpoints for any other step count. Optional mild CFG (`--cfg 1.5..2.0`) supported alongside fast `--cfg 1.0`. |

---

## 5. Canonical Module Map & Ports

All legacy duplicate video files (`video_render_logic.py`, `video_heterarchy_composer.py`, `neuro_video_maze_autotest.py`, `neuro_battle_server.py`, `neuro_run.py`) have been removed and unified into the canonical naming scheme matching `neuro_heterarchy_diffusion.py` and `neuro_heterarchy_musicgen.py`:

| File | Port / Authkey | Role |
| :--- | :--- | :--- |
| **`video_world_server.py`** | `localhost:6006` (`b"video_world"`) | Immortal LTX-Video weights RPC server (`0.9.8-distilled` NF4 + `T5 ↔ VAE` auto-swap). |
| **`neuro_heterarchy_video.py`** | `:6002` (frames) / `:6007` (ctrl) | Unified 3D-DiT runner supporting `.json` catalytic manifests, `.txt` pure-WM lores, `--init-image`, 60 Hz BLE/LSL→VRAM bus, `SmoothRealtimeVideoPacer`, and Web Studio bridge. |
| **`neuro_ltx_backend.py`** | — | `LTXModelBackend`, `RemoteLTXModelBackend`, in-place NF4/INT8 quantization, `T5 ↔ VAE` swap, distilled/base sigma schedules. |
| **`neuro_engine.py`** | — | `UniversalLatentAutobattlerEngine`: pure-tensor N-agent orchestrator, active-slot spherical composition, dual Catalytic / Pure-WM (Oracle-skip) modes, `set_init_image_bytes`, and NPC feedback routing. |
| **`neuro_controller.py`** | — | `SymmetricBotController`, orthonormal 19,200-D WM eigenmode synthesizer (EEG / 24-ch Gamepad / Autonomous `npc:` replicator policy), zero-input equilibrium. |
| **`neuro_solver_stepper.py`** | — | `LatentFlowWindowStepper`: S³ wave integrator, optional CFG on distilled, `t²` anchor jitter, AdaIN latent stabilization. |
| **`neuro_referee.py`** | — | `CoordinateFreeHistoryReferee`: chunked multi-branch 3D-DiT Rectified-Flow Oracle (`n_noise_probes=2`, effective DoF `t-stat`). |
| **`neuro_world.py`** | — | N-entity manifest dataclasses & JSON loader (`GameWorldManifest`, `StageSpec`, `StageEntitySlot`, `GenerativeQuestSpec`). |
| **`web_cloud_gateway.py` & `static/`** | `:8080` (HTTPS) | Unified Web Studio UI with FreeEEG16 Web-BLE, Gradual WM Mixer (`eeg_wm_gain` × `gamepad_wm_mix`), 24-ch Gamepad poller & `🎯` concept binding. |

---

## 6. Quick Start & CLI Reference

```bash
# Terminal 1: Immortal Video World Weights Server (port 6006, 0.9.8-distilled NF4, T5<->VAE swap)
./video_world_server.sh
# Equivalent:
# python3 video_world_server.py --port 6006 --model distilled --quant nf4 --vae-tile-min 256 --allow-t5-offload

# Terminal 2: Unified Web Studio Gateway (:8080)
python3 web_cloud_gateway.py --port 8080 --cert cert.pem --key key.pem --video-lore configs/mode_05_wm_control_test_arena.json

# Terminal 3A: 4-Way Movement / Catalytic Control Test (default ball mode, 8 distilled steps)
./neuro_heterarchy_video.sh
# Equivalent:
# python3 neuro_heterarchy_video.py --mode ball --video-port 6006 --steps 8 --cfg 1.0 --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0

# Terminal 3B: 1v1 Catalytic Duel vs Autonomous Adaptive NPC (e0=Human, e1=npc:AFz)
python3 neuro_heterarchy_video.py --mode duel --steps 8 --cfg 1.5 --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0

# Terminal 3C: Pure Open-Loop .txt Art Stream (automatically skips 3D-DiT Oracle)
python3 neuro_heterarchy_video.py --lore lore/video_lore_psy.txt --steps 8 --cfg 1.0 --eeg-wm-gain 1.0

# Terminal 3D: Custom Starting PNG Map (e.g. Real Maze init frame)
python3 neuro_heterarchy_video.py --manifest configs/mode_05_wm_control_test_arena.json --init-image maze_start.png --steps 8 --cfg 1.5
```

---

## 7. Current Status & Next-Session Handoff Notes

### ✅ Verified Working (as of Nov 21, 2026)
* **Zero OOM Across Full Lifecycle:** `0.9.8-2B-Distilled` NF4 + `T5 ↔ VAE` auto-swap runs stably in `3.27 GB` server VRAM during generation (`~1.8s/step` with multi-branch Oracle) and swaps cleanly without OOM when editing concepts or switching manifests live in the browser.
* **Clean Single-Region Control & Directional Response:** `--entity-source auto` bug fixed; pressing `LS_UP/DOWN/LEFT/RIGHT` activates exclusively `e0` (`100%`) while `e1..e4` remain neutral (`33%`), visibly steering the white sphere on the grid (`00:10–00:13` UP, `00:31–00:33` DOWN).
* **Stabilized Multi-Branch Oracle:** `P+` stays calm around `43–58%` at rest (no `3% ↔ 95%` noise spikes) and `branches` dynamically reflect real video trajectories with live `BRANCH_SWITCH` transitions.
* **Repo Cleanup Complete:** Legacy video stack removed; canonical `video_world_server.py` (`:6006`) + `neuro_heterarchy_video.py` unified.

### 🎯 Next Steps for Next Session
1. **Build the Real Maze World (Obstacles + Extraction + Optional NPC Guardian):**
   * Use `--init-image` (2D orthographic maze PNG) + a clean orthographic prompt (avoid `"rolls"` + `"grid lines"`, which triggers 3D bowling-lane perspective tilt; prefer `"flat 2D top-down orthographic white dot sliding on a flat 2D dark maze map"`).
   * Test `--cfg 1.5` vs `1.0` on `0.9.8-distilled` for crisp wall compliance and directional control.
2. **Test Catalytic 1v1 Duels (`configs/mode_02_catalytic_duel.json`) & Ritual (`configs/mode_04_forest_neuro_ritual.json`):**
   * Verify human (`e0=eeg:FCz`) vs autonomous adaptive NPC (`e1=npc:AFz`) and NPC-vs-NPC (`--entity-source "e0=npc:FCz,e1=npc:AFz"`).
3. **Live `FreeEEG16` Calibration (`neuro_controller.py`):**
   * Replace synthetic sinusoid basis with online-calibrated subject subspaces / physiological Gamma-vs-Beta gating (Miller et al., 2018; Chen et al., 2024) + EMG artifact check on 70–100 Hz.
4. **Minor Repo Hygiene:**
   * Keep `key.pem` / `cert.pem` out of git and LLM context packs; keep `neuro_genesis_engine.py` until `neuro_heterarchy_musicgen.py` and `neuro_deterministic_reproducer.py` drop its import.

---

## 8. Comprehensive Scientific Bibliography & DOIs

1. **McCulloch, W. S. (1945).** *A heterarchy of values determined by the topology of nervous nets.* Bull. Math. Biophys., 7(2), 89–93. [https://doi.org/10.1007/BF02478259](https://doi.org/10.1007/BF02478259)
2. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** *The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy.* arXiv:2507.05888. [https://doi.org/10.48550/arXiv.2507.05888](https://doi.org/10.48550/arXiv.2507.05888)
3. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [https://doi.org/10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
4. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [https://doi.org/10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
5. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [https://doi.org/10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
6. **Feigenbaum, M. J. (1978).** *Quantitative universality for a class of nonlinear transformations.* J. Stat. Phys., 19(1), 25–52. [https://doi.org/10.1007/BF01020332](https://doi.org/10.1007/BF01020332)
7. **Kauffman, S. A. (1986).** *Autocatalytic sets of proteins.* J. Theor. Biol., 119(1), 1–24. [https://doi.org/10.1016/0022-5193(86)90047-9](https://doi.org/10.1016/0022-5193(86)90047-9)
8. **Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* J. Neural Eng., 15(5), 056011. [https://doi.org/10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
9. **Dickey, C. W., et al. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* PNAS, 119(28), e2107797119. [https://doi.org/10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
10. **HaCohen, Y., et al. (2024).** *LTX-Video: Realtime Video Latent Diffusion.* arXiv:2501.00103. [https://doi.org/10.48550/arXiv.2501.00103](https://doi.org/10.48550/arXiv.2501.00103)
11. **Yang, C., et al. (2025).** *Reasoning via Video: The First Evaluation of Video Models’ Reasoning Abilities through Maze-Solving Tasks (VR-Bench).* arXiv:2511.15065. [https://doi.org/10.48550/arXiv.2511.15065](https://doi.org/10.48550/arXiv.2511.15065)
12. **Mongillo, G., Barak, O., & Tsodyks, M. (2008).** *Synaptic Theory of Working Memory.* Science, 319(5869), 1543–1546. [https://doi.org/10.1126/science.1150769](https://doi.org/10.1126/science.1150769)

