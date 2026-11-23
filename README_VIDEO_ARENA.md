# 🌌 EXOCORTEX 3D-DiT: CONTINUOUS REAL-TIME NEUROFEEDBACK WORLD ENGINE, DYNAMIC AGENT MANAGER & MULTI-BRANCH 3D-DiT ORACLE

> **100% Continuous Real-Time 3D-DiT World Engine (`LTX-Video 0.9.8-2B-Distilled` NF4 / FP8 & `0.9.0-Base`), Automatic `T5 ↔ VAE` VRAM Swap (Zero OOM on 12 GB GPUs), EOS-Cone Empty-Prompt Navigation, Multidimensional `T5-XXL` SVD Semantic Navigation (`96,000-D` Working Memory), Dynamic Multi-Agent Manager, Chronological Multi-Frame Context (`ctx=2` Smooth Continuity), and Multi-Branch Rectified-Flow 3D-DiT Oracle (`FreeEEG16` × 24-Ch Gamepad × `TBP.Monty`)**

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
[![arXiv:2605.07973](https://img.shields.io/badge/arXiv-2605.07973-blueviolet.svg)](https://arxiv.org/abs/2605.07973)
[![arXiv:2501.09982](https://img.shields.io/badge/arXiv-2501.09982-informational.svg)](https://arxiv.org/abs/2501.09982)
[![arXiv:2312.07537](https://img.shields.io/badge/arXiv-2312.07537-brightgreen.svg)](https://arxiv.org/abs/2312.07537)
[![arXiv:2502.03496](https://img.shields.io/badge/arXiv-2502.03496-orange.svg)](https://arxiv.org/abs/2502.03496)

---

## 📑 Table of Contents
1. [Core Design Laws: Zero Hardcoded Words, Pure-Tensor Engine & Dual Mode](#1-core-design-laws-zero-hardcoded-words-pure-tensor-engine--dual-mode)
2. [Dynamic Agent Manager, Active-Only Routing & Unified WM Bus](#2-dynamic-agent-manager-active-only-routing--unified-wm-bus)
3. [Full-Rank 96,000-D 3D-DiT Integration: EOS-Cone, Multidimensional SVD Semantics & Spectral 2D-DCT $S^3$](#3-full-rank-96000-d-3d-dit-integration-eos-cone-multidimensional-svd-semantics--spectral-2d-dct-s3)
4. [Chronological Multi-Frame Context (`ctx=1..3`) & Empirical Stability](#4-chronological-multi-frame-context-ctx13--empirical-stability)
5. [Multi-Branch Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)](#5-multi-branch-rectified-flow-3d-dit-oracle-neuro_refereepy)
6. [LTX-Video 0.9.8-2B-Distilled NF4 & `T5 ↔ VAE` Zero-OOM Architecture](#6-ltx-video-098-2b-distilled-nf4--t5--vae-zero-oom-architecture)
7. [Canonical Module Map & Ports](#7-canonical-module-map--ports)
8. [Quick Start & CLI Reference](#8-quick-start--cli-reference)
9. [Current Status & Next Feature Roadmap (Multi-View Shared World / TBT 2.0)](#9-current-status--next-feature-roadmap-multi-view-shared-world--tbt-20)
10. [Comprehensive Scientific Bibliography & DOIs](#10-comprehensive-scientific-bibliography--dois)

---

## 1. Core Design Laws: Zero Hardcoded Words, Pure-Tensor Engine & Dual Mode

**Exocortex 3D-DiT** is a **100% aconceptual, language-agnostic, lore-agnostic continuous real-time world engine** governed by four strict invariants:

1. **Zero Hardcoded Words in Python Engine Code:**
   All engine modules (`neuro_controller.py`, `neuro_engine.py`, `neuro_referee.py`, `neuro_solver_stepper.py`, `neuro_ltx_backend.py`) are pure mathematical tensor operators. No directions (`up/down`), game concepts (`enemy/duel/maze/hp`), or fallback prompts exist anywhere in Python. All semantics live exclusively in external JSON manifests (`configs/*.json`) or `.txt` lores (`lore/*.txt`), while the unconditional empty-prompt cone is extracted directly from `T5TokenizerFast`'s own vocabulary.
2. **100% Pure-Tensor Runtime Loop (Zero String Concatenation):**
   All active manifest concepts and hypotheses are encoded once into `[1, 128, 4096]` T5 tensors at warmup (only live `allowed_pool` IDs are warmed; stale deleted keys are purged). During real-time generation (including intra-step live updates across the 8 denoising steps), `neuro_engine.py` composes `pos_emb` solely via **Theta-Gamma slot multiplexing** (`multiplex_t5_concept_slots`) and **intra-manifold Kent-geodesic traversal** over active agent tensors.
3. **Dual Mode in One Unified Stack:**
   * **Closed-Loop Catalytic World Mode (`.json` manifests with `quest` hypotheses):** Enables the Multi-Branch 3D-DiT RF Oracle, real-time `BRANCH_SWITCH` / `STAGE_DONE` transitions, and adaptive `npc:` feedback loops (movement, mazes, rituals, 1v1/NvN duels).
   * **Fast Open-Loop Pure WM & Self-Organizing Ecology Mode (`.txt` lores or manifests without `quest`, e.g., `mode_06_black_smoker_ecosphere.json`):** Automatically skips the 3D-DiT Oracle forward pass (~350 ms saved per step) for pure continuous 3D-DiT neuro-visual streaming and velocity-coupled `npc:` ecosystems.
4. **Custom Initial Scene Injection (`--init-image` & `/api/video/init_image`):**
   Any starting PNG/JPG (such as a 2D orthographic maze map) can be encoded directly into `z[0]` (`local_context_latents_5d`) from CLI or the Web Studio and cleanly cancelled via `✕`.

---

## 2. Dynamic Agent Manager, Active-Only Routing & Unified WM Bus

```
 ═════════════════════════════════════════════════════════════════════════════
   UNIFIED WORKING MEMORY BUS: DYNAMIC AGENT MANAGER (ARBITRARY N AGENTS)
 ═════════════════════════════════════════════════════════════════════════════

  [1. Biological EEG (FreeEEG16 BLE/LSL)] ──(eeg_wm_gain ∈ [0..1])──┐
  [2. 24-Ch Gamepad / Keyboard Synth] ──────(gamepad_wm_mix ∈ [0..1])┼─► ┌───────────────────────────────┐
  [3. Autonomous NPC/Monty ("npc:AFz")] ────(monty_mode + Oracle)────┘   │ SHARED WM FIELD (VRAM IPC)    │
                                                                         │ W_r ∈ R^{5 × 32 × 120}        │
                                                                         └───────────────┬───────────────┘
                                                                                         │
   ┌─────────────────────────────────────────────────────────────────────────────────────┴────────────────────┐
   ▼ (Active-Only Agents → Multidimensional SVD Semantics + Spectral 2D-DCT S^3)                              ▼
  ┌──────────────────────────────────────────────────────┐                ┌────────────────────────────────────────┐
  │ FULL-RANK 96,000-D 3D-DiT STEPPER                    │                │ MULTI-BRANCH 3D-DiT RF ORACLE          │
  │ • LTX-Video 0.9.8-2B-Distilled (In-Place 4-bit NF4)  │                │ • Scores [H_stay, H_1..H_M] on video   │
  │ • Chronological Causal Context (ctx=2 smooth motion) │───────────────►│ • 2-probe CRN ε + effective DoF t-stat │
  │ • Dual Stochasticity (active vs unassigned regions)  │  unpacked_z0   │ • Drives BRANCH_SWITCH & NPC reward    │
  └──────────────────────────────────────────────────────┘                └────────────────────────────────────────┘
 ═════════════════════════════════════════════════════════════════════════════
```

* **Dynamic Agent Manager (Web Studio `:8080` → `/api/video/agent_manage`, `/api/video/agent_concepts`):**
  Works identically for simple flat `.txt` lores (e.g. `world_lore.txt` shared across all 5 cortical slots `e0..e4`, with `Общий пул` checked automatically so edits/deletions apply globally with zero phantom concepts) and complex multi-agent `.json` worlds (`mode_02`, `mode_04`, `mode_06`). Each agent card displays live status (`🟢 ACTIVE` / `⏸️ IDLE`), real concept count, live source tag (`EEG_VRAM:Fpz`, `NPC_WM:AFz`, `GP_WM`, `OFF`), binding dropdown (`eeg:FCz..Fpz`, `npc:FCz..Fpz`, `gamepad`, `off`), `+ АГЕНТ`, and `✕` remove.
* **Strict Active-Only Concept Routing (`is_slot_active_for_concepts`):**
  Matches 2D Diffusion and MusicGen: when `monty_mode == "off"`, **inactive regions and paused bots send ZERO concepts** to `multiplex_t5_concept_slots`. Connecting a BLE board to any cortical zone (e.g., `Fpz`) automatically lifts `eeg_wm_gain = 1.0` and activates strictly that zone. When `monty_mode` is set to `missing_only` or `full`, `npc:` bots and empty cortical regions engage symmetrically.
* **Dual Stochasticity (`--stochastic` vs. `--unassigned-stochastic`):**
  Active agents obey `#vStochastic`, while unassigned/empty cortical regions obey `#vUnassignedStochastic` inside `unfold_wm_spectral_3d_s3` while preserving exact $\mathcal{N}(0, I)$ noise variance at every voxel.

---

## 3. Full-Rank 96,000-D 3D-DiT Integration: EOS-Cone, Multidimensional SVD Semantics & Spectral 2D-DCT $S^3$

1. **Empty-Prompt (`""`) Unconditional `EOS`-Cone Navigation (`get_t5_output_manifold` + `synthesize_empty_t5_from_kent_manifold`):**
   * *Why it works:* `LTX-Video`'s nonlinear `caption_projection` MLP (`Linear(4096→2048)→GELU→Linear(2048→2048)`) is strictly calibrated to real `T5-XXL` sentence states and the unconditional null `EOS` token (`""`).
   * *Implementation:* `LTXModelBackend.get_t5_output_manifold` filters `T5TokenizerFast.get_vocab()` down to clean whole lowercase ASCII English words (`tok.startswith("▁")`, `isalpha()`, `islower()`, len 3–12, **zero hardcoded words**), encodes 4,096 words once (~1–2s on RTX 3060, cached to `t5_eos_manifold_128x4096.pt`), and extracts the exact covariance eigenbasis `(V_eos, sigma_eos)` on the **`EOS` token (`[:, 1, :]`)**.
   * At empty prompt, `synthesize_empty_t5_from_kent_manifold` preserves the native 1-token mask (`[True, False, ...]`) and rotates `empty_emb[0, 0]` within a safe norm-preserving tangent cone ($\theta \le 0.06\pi$) while all 96,000 WM modes steer the $\mathcal{N}(0, I)$ $S^3$ latent noise. Zero colored static and zero OCR letters at any `wm_radius`.
2. **True Multidimensional Semantic-Space Navigation (`select_concept_full_rank_524288d`):**
   * Extracts active content token vectors $\mathbf{c}_k \in \mathbb{R}^{4096}$ (excluding `PAD`/`EOS`) for all $K$ concepts in an agent's pool, computes centered SVD (`V_sem`), and embeds every concept at its true unit coordinates $\mathbf{p}_k \in S^{r-1}$ ($r = \min(K-1, 32)$) — **zero `#0` bias, zero 1D list-order artifacts**.
   * Semantically related concepts automatically cluster together on $S^{r-1}$ (e.g., `ocean + waves + sea foam`, `jungle + trees + vines`, `nebula + stars + galaxy`).
   * A region's 19,200-D Working Memory (`[5 bands × 32 causal slots × 120 dipoles]`) maps through a cached **5-band isotropic QR projector** (`_get_semantic_wm_projector`):
     * **Slow rhythms (`Delta + Theta`, `0.8–8.5 Hz`)** navigate macro semantic axes (global cluster selection — *Fan et al., 2024; Ding et al., 2016*).
     * **Fast rhythms (`Beta + Gamma + 89.5 Hz Ripples`, `13–100 Hz`)** navigate all $r$ axes (fine-grained intra-cluster details — *Miller et al., 2018; Colgin et al., 2009; Dickey et al., 2022*).
   * Top coherent cluster concepts (`prob >= 14%`, up to 3) are packed via `multiplex_t5_concept_slots` and locally steered via `steer_pool_t5_within_manifold`, with full gamepad precision & gradual co-piloting (`eeg_wm_gain` × `gamepad_wm_mix`).
3. **Spatially-Coherent 2D-DCT Spectral $S^3$ Noise (`unfold_wm_spectral_3d_s3`):**
   Grounded in *FreeInit* (*Wu et al., 2024*) [arXiv:2312.07537] and *FreqPrior* (*Yuan et al., 2025*) [arXiv:2502.03496]:
   * **`stochastic = 1.0` & `unassigned_stochastic = 1.0`:** Pure `torch.randn(shape)` and unperturbed `T5-XXL` prompts — native LTX-Video parity.
   * **`stochastic = 0.0`:** All `96,000` WM modes rotate 4 deterministic orthogonal Gaussian carriers $(B_1 \dots B_4)$ on $S^3$ across **128 latent channels $\times F_{\text{lat}}$ chronological frames $\times$ smooth 2D-DCT spatial cosine harmonics (`K_sp = 15`)** with advancing `win_seed = base_seed + _window_step_idx * 997`. Preserves exact $\mathcal{N}(0, I)$ statistics, zero `32×32` patch tearing, and zero standing-wave stripes.

---

## 4. Chronological Multi-Frame Context (`ctx=1..3`) & Empirical Stability

To preserve physical momentum and eliminate window-seam snaps without freezing motion:
* **Causal Chronological Prefix Encoding (`neuro_engine.py`):**
  Instead of duplicating the last latent frame, `neuro_engine.py` slices the exact chronological pixel tail from `recent_clean_frames_rgb` (`1` frame for `ctx=1`, `9` frames for `ctx=2`, `17` frames for `ctx=3`) and encodes it as a coherent causal clip via `encode_frames_to_latent`.
* **Clean Prefix Conditioning (`neuro_solver_stepper.py`):**
  Follows `LTXConditionPipeline`: all `n_cond` prefix latent frames are conditioned at `t = 0` with soft `t²` conditioning jitter (`image_cond_noise_scale = 0.08` for `n_cond > 1` on `sigma > 0.50`) so steering can change direction smoothly, while `scheduler.step` advances strictly on free future frames `[:, :, n_cond:]` with zero backward `lerp` pulls.
* **Verified Stability Profile (RTX 3060 12 GB, `0.9.8-2B-Distilled` NF4, `steps=8`, `step_latents=2`):**
  * **`context_latents = 1` (1 frame):** Fastest reactivity and sharp resets, slight velocity discontinuity across windows.
  * **`context_latents = 2` (9 frames = 0.5s motion memory) — RECOMMENDED DEFAULT FOR SMOOTH MOTION:** Verified **stable, sharp, and fluid** over long continuous rollouts (e.g., continuous dancing with stable background and virtually imperceptible window seams).
  * **`context_latents = 3` (17 frames = 1.0s memory):** Works dynamically without freezing, but begins to accumulate blur/softening over long autoregressive rollouts due to repeated 17-frame pixel-space 3D-VAE re-encoding at `1:192` compression. *(Future optimization: pass denoised latents directly for frames `1..n_cond-1` instead of re-encoding 17 pixels through VAE.)*

---

## 5. Multi-Branch Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)

`CoordinateFreeHistoryReferee` uses **LTX-Video's own 3D-DiT** to evaluate all competing world branches directly on the generated spatiotemporal transition $\mathbf{z}_{\text{prev}} \to \mathbf{z}_{\text{new}}$:
1. Keeps prefix frames `:n_cond` clean ($t_c = 0$) and noises generated frames `n_cond:` at $\sigma_{\text{probe}}$ (`0.725` for `0.9.8-distilled`, `0.55` for `0.9.0-base`).
2. Evaluates `[H_stay, H_branch_1, ..., H_branch_M]` over full `[1, 128, 4096]` T5 embeddings in VRAM-safe chunks (`max_batch=4`), averaged across `n_noise_probes=2` shared noise draws $\boldsymbol{\epsilon}$ (Common Random Numbers).
3. Computes motion-weighted paired $t$-statistics using effective spatial-temporal degrees of freedom (`n_eff = n_tok / 16.0`), triggers `BRANCH_SWITCH` with hysteresis (`t_win > 0.85`, `Δt > 0.65`), and routes unbiased `±reward` to active `npc:` controllers.

---

## 6. LTX-Video 0.9.8-2B-Distilled NF4 & `T5 ↔ VAE` Zero-OOM Architecture

| Component / Mechanism | Implementation (`neuro_ltx_backend.py` & `video_world_server.py`) |
| :--- | :--- |
| **In-Place 4-Bit NF4 Quantization** | `quantize_model_inplace_bnb()` streams 284 `bnb.nn.Linear4bit` (NF4) layers onto CUDA (~1.6 GB VRAM for the 2B 3D-DiT). |
| **Automatic `T5 ↔ VAE` VRAM Swap** | `T5-XXL 4-bit` and `0.9.8 3D-VAE` are mutually exclusive in VRAM when `--allow-t5-offload` is active (default): `3.27 GB` steady server VRAM, zero OOM during live WebUI concept edits. |
| **T5 EOS Manifold Caching** | `get_t5_output_manifold()` extracts & caches `t5_eos_manifold_128x4096.pt` in `QUANT_T5_DIR` at server startup (~1–2s) and serves it via `get_t5_manifold` RPC. |
| **Clean 3D-VAE Feedback Loop** | `vae_tile_min >= 512px` on `> 10 GB` GPUs (zero `256px` tile seams); clean `decode_noise_scale=0.0` + `adain=0.0` eliminates contrast washout. |
| **Exact Distilled Sigma Schedule** | `build_distilled_sigmas(steps)` uses all 8 official sigmas `[1.0000 ... 0.4219]` at `--steps 8`. |

---

## 7. Canonical Module Map & Ports

| File | Port / Authkey | Role |
| :--- | :--- | :--- |
| **`video_world_server.py`** | `localhost:6006` (`b"video_world"`) | Immortal LTX-Video weights RPC server (`0.9.8-distilled` NF4 + `T5 ↔ VAE` auto-swap + `get_t5_manifold` + `encode_frames`). |
| **`neuro_heterarchy_video.py`** | `:6002` (frames) / `:6007` (ctrl) | Unified 3D-DiT runner supporting `.json` manifests, `.txt` lores (shared flat or regional), `--init-image`, Dynamic Agent Manager IPC, 60 Hz BLE/LSL→VRAM bus, and `SmoothRealtimeVideoPacer`. |
| **`neuro_ltx_backend.py`** | — | `LTXModelBackend`, `RemoteLTXModelBackend`, in-place NF4/INT8 quantization, `T5 ↔ VAE` swap, `t5_eos_manifold_128x4096.pt` cache, causal multi-frame VAE encode/decode. |
| **`neuro_engine.py`** | — | `UniversalLatentAutobattlerEngine`: pure-tensor N-agent orchestrator, active-only evaluation (`_evaluate_agents_for_step`), chronological multi-frame context (`ctx=1..3`), live intra-step updates. |
| **`neuro_controller.py`** | — | `SymmetricBotController`, `is_slot_active_for_concepts`, multidimensional SVD semantic navigation + 5-band isotropic QR projectors, safe EOS-cone empty-prompt steering, `multiplex_t5_concept_slots`, 24-ch Gamepad & `npc:` policy. |
| **`neuro_solver_stepper.py`** | — | `LatentFlowWindowStepper`: 2D-DCT Spectral $S^3$ 96,000-D wave integrator with dual stochasticity (`stochastic` × `unassigned_stochastic`), clean multi-frame prefix Euler stepping. |
| **`neuro_referee.py`** | — | `CoordinateFreeHistoryReferee`: chunked multi-branch 3D-DiT Rectified-Flow Oracle (`n_noise_probes=2`, effective DoF `t-stat`). |
| **`neuro_world.py`** | — | N-entity manifest dataclasses & JSON loader (`GameWorldManifest`, `StageSpec`, `StageEntitySlot`, `GenerativeQuestSpec`). |
| **`web_cloud_gateway.py` & `static/`** | `:8080` (HTTPS) | Unified Web Studio UI with FreeEEG16 Web-BLE, Dynamic Agent Manager (`/api/video/agent_manage`, `/api/video/agent_concepts`), Gradual WM Mixer (`eeg_wm_gain` × `gamepad_wm_mix`), Dual Stochasticity sliders, and 24-ch Gamepad `🎯` binding. |

---

## 8. Quick Start & CLI Reference

```bash
# Terminal 1: Immortal Video World Weights Server (port 6006, 0.9.8-distilled NF4, T5<->VAE swap, no 256px seams)
python3 video_world_server.py --port 6006 --model distilled --quant nf4 --vae-tile-min 512 --allow-t5-offload

# Terminal 2: Unified Web Studio Gateway (:8080)
python3 web_cloud_gateway.py --port 8080 --cert cert.pem --key key.pem --video-lore lore/world_lore.txt

# Terminal 3A: Smooth Continuous 3D-DiT Stream (Recommended ctx=2, step=2, pure 96,000-D WM)
python3 neuro_heterarchy_video.py --lore lore/world_lore.txt --steps 8 --cfg 1.0 --stochastic 0.0 --unassigned-stochastic 0.0 --wm-radius 0.35 --context-latents 2 --step-latents 2

# Terminal 3B: Black Smoker Autocatalytic Ecosphere (Multi-Agent with Monty/NPC Bots)
python3 neuro_heterarchy_video.py --manifest configs/mode_06_black_smoker_ecosphere.json --monty-mode missing_only --steps 8 --cfg 1.0 --stochastic 0.0 --wm-radius 0.35 --context-latents 2 --step-latents 2

# Terminal 3C: 1v1 Catalytic Duel vs Autonomous Adaptive NPC (e0=Human, e1=npc:AFz)
python3 neuro_heterarchy_video.py --mode duel --monty-mode missing_only --steps 8 --cfg 1.5 --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0

# Terminal 3D: Custom Starting PNG Map (e.g. Real Maze init frame)
python3 neuro_heterarchy_video.py --manifest configs/mode_05_wm_control_test_arena.json --init-image maze_start.png --steps 8 --cfg 1.5
```

---

## 9. Current Status & Next Feature Roadmap (Multi-View Shared World / TBT 2.0)

### ✅ Verified Working (as of Nov 23, 2026)
* **Zero OOM Across Full Lifecycle:** `0.9.8-2B-Distilled` NF4 + `T5 ↔ VAE` auto-swap (`3.27 GB` server VRAM, `RTF ~1.0x`).
* **Empty Prompt (`""`) Fixed:** Safe `EOS`-cone Kent steering (`t5_eos_manifold_128x4096.pt`) + $\mathcal{N}(0, I)$ $S^3$ noise produces coherent unconditional scenes at any `wm_radius` with zero static and zero OCR letters.
* **Multidimensional Semantic Navigation & 100% WebUI Sync:** Isotropic 5-band QR projectors over real `T5-XXL` SVD concept coordinates eliminate `#0` bias; Dynamic Agent Manager (`e0..eN`) enforces active-only routing (`monty_mode=off` silences inactive agents) with exact 1:1 concept sync and live `#vUnassignedStochastic` control.
* **Smooth Multi-Frame Context (`ctx=2`):** Chronological 9-frame causal VAE prefix (`ctx=2`) delivers stable, smooth continuous motion with minimal seam flicker (note: `ctx=3` softens over long runs due to 17-frame VAE re-encoding; use `ctx=2` for best sharpness + continuity).

### 🚀 Next Feature Roadmap: Multi-User / Multi-View Shared World (TBT 2.0 + Multi-View World Model)
* Grounded in **Thousand Brains Theory 2.0** (*Hawkins, Leadholm, & Clay, 2026*, Sec. 3.1–3.2):
  1. **Shared Allocentric World State (`L3` Consensus):** All connected users (`User1..UserN`) and autonomous `npc:` agents inhabit and mutate a single shared allocentric world state / causal event graph via the Dynamic Agent Manager & `HardwareVRAMBus`.
  2. **Per-User Egocentric Viewpoint Relay (Thalamic Transform):** Each user receives their own viewpoint stream (separate camera angle / 1st-person avatar perspective / top-down overview) via per-view camera conditioning / `video_coords` / batched multi-view generation (`B = N_users`), with optional Cross-View Attention or shared latent anchor synchronization for geometric consistency across perspectives.
* *(Optional Future Polish)* **Latent-Pass `ctx=3`:** Reuse clean denoised latents directly for frames `1..n_cond-1` (pixel-encoding only frame `0` or using `LTXI2VLongMultiPromptPipeline` overlap crossfade) if 17-frame (`ctx=3`) memory without VAE blur is desired.

---

## 10. Comprehensive Scientific Bibliography & DOIs

1. **McCulloch, W. S. (1945).** *A heterarchy of values determined by the topology of nervous nets.* Bull. Math. Biophys., 7(2), 89–93. [https://doi.org/10.1007/BF02478259](https://doi.org/10.1007/BF02478259)
2. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** *The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy.* arXiv:2507.05888. [https://doi.org/10.48550/arXiv.2507.05888](https://doi.org/10.48550/arXiv.2507.05888)
3. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463–475. [https://doi.org/10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
4. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** *Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory.* Nature Human Behaviour, 8, 2150–2163. [https://doi.org/10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
5. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** *Flexible control of sequence working memory in the macaque frontal cortex.* Neuron, 112(20), 3502–3514. [https://doi.org/10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
6. **Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).** *Cortical tracking of hierarchical linguistic structures in connected speech.* Nature Neuroscience, 19(1), 158–164. [https://doi.org/10.1038/nn.4186](https://doi.org/10.1038/nn.4186)
7. **Colgin, L. L., et al. (2009).** *Frequency of gamma oscillations routes flow of information in the hippocampus.* Nature, 462, 353–357. [https://doi.org/10.1038/nature08573](https://doi.org/10.1038/nature08573)
8. **Bieri, K. W., Bobbitt, K. N., & Colgin, L. L. (2014).** *Slow and fast gamma rhythms coordinate different spatial coding modes in hippocampal place cells.* Neuron, 82(3), 670–681. [https://doi.org/10.1016/j.neuron.2014.03.013](https://doi.org/10.1016/j.neuron.2014.03.013)
9. **Feigenbaum, M. J. (1978).** *Quantitative universality for a class of nonlinear transformations.* J. Stat. Phys., 19(1), 25–52. [https://doi.org/10.1007/BF01020332](https://doi.org/10.1007/BF01020332)
10. **Kauffman, S. A. (1986).** *Autocatalytic sets of proteins.* J. Theor. Biol., 119(1), 1–24. [https://doi.org/10.1016/0022-5193(86)90047-9](https://doi.org/10.1016/0022-5193(86)90047-9)
11. **Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase Locking Value revisited: teaching new tricks to an old dog.* J. Neural Eng., 15(5), 056011. [https://doi.org/10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
12. **Dickey, C. W., et al. (2022).** *Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall.* PNAS, 119(28), e2107797119. [https://doi.org/10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
13. **HaCohen, Y., et al. (2024).** *LTX-Video: Realtime Video Latent Diffusion.* arXiv:2501.00103. [https://doi.org/10.48550/arXiv.2501.00103](https://doi.org/10.48550/arXiv.2501.00103)
14. **Yang, C., et al. (2025).** *Reasoning via Video: The First Evaluation of Video Models’ Reasoning Abilities through Maze-Solving Tasks (VR-Bench).* arXiv:2511.15065. [https://doi.org/10.48550/arXiv.2511.15065](https://doi.org/10.48550/arXiv.2511.15065)
15. **Mongillo, G., Barak, O., & Tsodyks, M. (2008).** *Synaptic Theory of Working Memory.* Science, 319(5869), 1543–1546. [https://doi.org/10.1126/science.1150769](https://doi.org/10.1126/science.1150769)
16. **Roy, A., Das Biswas, S., & Roy, K. (2026).** *HEART: Hyperspherical Embedding Alignment via Kent-Representation Traversal in Diffusion Models.* arXiv:2605.07973. [https://doi.org/10.48550/arXiv.2605.07973](https://doi.org/10.48550/arXiv.2605.07973)
17. **Cao, Y., et al. (2025).** *RichSpace: Enriching Text-to-Video Prompt Space via Text Embedding Interpolation.* arXiv:2501.09982. [https://doi.org/10.48550/arXiv.2501.09982](https://doi.org/10.48550/arXiv.2501.09982)
18. **Wu, T., Si, C., Jiang, Y., Huang, Z., & Liu, Z. (2024).** *FreeInit: Bridging Initialization Gap in Video Diffusion Models.* ECCV 2024 / arXiv:2312.07537. [https://doi.org/10.48550/arXiv.2312.07537](https://doi.org/10.48550/arXiv.2312.07537)
19. **Yuan, Y., et al. (2025).** *FreqPrior: Improving Video Diffusion Models with Frequency Filtering Gaussian Noise.* ICLR 2025 / arXiv:2502.03496. [https://doi.org/10.48550/arXiv.2502.03496](https://doi.org/10.48550/arXiv.2502.03496)

