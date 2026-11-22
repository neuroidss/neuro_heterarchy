# 🌌 EXOCORTEX 3D-DiT: CONTINUOUS REAL-TIME NEUROFEEDBACK WORLD ENGINE, MULTI-BRANCH 3D-DiT ORACLE & AUTONOMOUS WM AGENTS

> **100% Continuous Real-Time 3D-DiT World Engine (`LTX-Video 0.9.8-2B-Distilled` NF4 / FP8 & `0.9.0-Base`), Automatic `T5 ↔ VAE` VRAM Swap (Zero OOM on 12 GB GPUs), Kent-Manifold `T5-XXL` Steering & Theta-Gamma Slot Multiplexing, Spatially-Coherent Spectral 2D-DCT $S^3$ Noise (`96,000-D` Working Memory), Multi-Branch Rectified-Flow 3D-DiT Oracle, and Shared Working Memory Bus (`FreeEEG16` × 24-Ch Gamepad × `TBP.Monty`)**

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
2. [Unified Working Memory Bus (`[5 × 5 × 32 × 120]`) & Autonomous `npc:` Agents](#2-unified-working-memory-bus-5--5--32--120--autonomous-npc-agents)
3. [Full-Rank 96,000-D 3D-DiT Integration: Kent-T5 Manifold & Spectral 2D-DCT $S^3$ Noise](#3-full-rank-96000-d-3d-dit-integration-kent-t5-manifold--spectral-2d-dct-s3-noise)
4. [Multi-Branch Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)](#4-multi-branch-rectified-flow-3d-dit-oracle-neuro_refereepy)
5. [LTX-Video 0.9.8-2B-Distilled NF4 & `T5 ↔ VAE` Zero-OOM Architecture](#5-ltx-video-098-2b-distilled-nf4--t5--vae-zero-oom-architecture)
6. [Canonical Module Map & Ports](#6-canonical-module-map--ports)
7. [Quick Start & CLI Reference](#7-quick-start--cli-reference)
8. [Current Status, Remaining Pre-Release Fixes & Multi-View Roadmap](#8-current-status-remaining-pre-release-fixes--multi-view-roadmap)
9. [Comprehensive Scientific Bibliography & DOIs](#9-comprehensive-scientific-bibliography--dois)

---

## 1. Core Design Laws: Zero Hardcoded Words, Pure-Tensor Engine & Dual Mode

**Exocortex 3D-DiT** is a **100% aconceptual, language-agnostic, lore-agnostic continuous real-time world engine** governed by four strict invariants:

1. **Zero Hardcoded Words in Python Engine Code:**
   All engine modules (`neuro_controller.py`, `neuro_engine.py`, `neuro_referee.py`, `neuro_solver_stepper.py`, `neuro_ltx_backend.py`) are pure mathematical tensor operators. No directions (`up/down`), game concepts (`enemy/duel/maze/hp`), or fallback prompts exist anywhere in Python. All semantics live exclusively in external JSON manifests (`configs/*.json`, `mode_*.json`) or `.txt` lores (`lore/*.txt`), while the general semantic cone is extracted directly from `T5-XXL`'s own vocabulary.
2. **100% Pure-Tensor Runtime Loop (Zero String Concatenation):**
   All manifest concepts and hypotheses are encoded once into `[1, 128, 4096]` T5 tensors at warmup. During real-time generation (including intra-step live updates across the 8 denoising steps), `neuro_engine.py` composes `pos_emb` solely via **Theta-Gamma slot multiplexing** (`multiplex_t5_concept_slots`) and **intra-manifold Kent-geodesic traversal** over active entity tensors.
3. **Dual Mode in One Unified Stack:**
   * **Closed-Loop Catalytic World Mode (`.json` manifests with `quest` hypotheses):** Enables the Multi-Branch 3D-DiT RF Oracle, real-time `BRANCH_SWITCH` / `STAGE_DONE` transitions, and adaptive `npc:` feedback loops (movement, mazes, rituals, 1v1/NvN duels).
   * **Fast Open-Loop Pure WM & Self-Organizing Ecology Mode (`.txt` lores or manifests without `quest`, e.g., `mode_06_black_smoker_ecosphere.json`):** Automatically skips the 3D-DiT Oracle forward pass (~350 ms saved per step) for pure continuous 3D-DiT neuro-visual streaming and velocity-coupled `npc:` ecosystems.
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
  [3. Autonomous NPC Agents ("npc:AFz")] ──(Oracle + DiT Velocity)───┘   │ SHARED WM FIELD (VRAM IPC)    │
                                                                        │ W_r ∈ R^{5 × 32 × 120}        │
                                                                        └───────────────┬───────────────┘
                                                                                        │
            ┌───────────────────────────────────────────────────────────────────────────┴────────────────────┐
            ▼ (Theta-Gamma T5 Multiplexing + Kent Cone + 2D-DCT Spectral S^3)                                ▼
  ┌──────────────────────────────────────────────────────┐                ┌────────────────────────────────────────┐
  │ FULL-RANK 96,000-D 3D-DiT STEPPER                    │                │ MULTI-BRANCH 3D-DiT RF ORACLE          │
  │ • LTX-Video 0.9.8-2B-Distilled (In-Place 4-bit NF4)  │                │ • Scores [H_stay, H_1..H_M] on video   │
  │ • Spectral 2D-DCT x 128-Ch S^3 (0 patch tearing)     │───────────────►│ • 2-probe CRN ε + effective DoF t-stat │
  │ • Clean I2V anchor (decode_noise=0.0, tile>=512px)   │  unpacked_z0   │ • Drives BRANCH_SWITCH & NPC reward    │
  └──────────────────────────────────────────────────────┘                └────────────────────────────────────────┘
 ═════════════════════════════════════════════════════════════════════════════
```

* **Symmetric Autonomous NPC / Ecosystem Agents (`control_binding: "npc:<REGION>"`):**
  Every entity slot (`e0`, `e1`, ...) uses the exact same `SymmetricBotController` class and canonical 19,200-D eigenmode basis (`get_canonical_concept_wm_basis_Kx5x32x120`). Slots marked `"npc:<REGION>"` maintain latent preference logits `npc_logits_K` over their `allowed_pool`, synthesize a 19,200-D WM wave with deterministic phase exploration, and update via both Oracle reward (`observe_oracle_feedback`) and 3D-DiT velocity feedback (`inject_dit_velocity_feedback`).
* **Per-Slot Region Isolation:**
  `--entity-source auto` strictly preserves each slot's JSON `control_binding` (`eeg:FCz`, `npc:AFz`, `npc:F3`, `npc:F4`, etc.) without overwriting regions.

---

## 3. Full-Rank 96,000-D 3D-DiT Integration: Kent-T5 Manifold & Spectral 2D-DCT $S^3$ Noise

Unlike 2D UNet (`CLIP`, `8×8` patches) or `MusicGen` (`T5-base`, discrete EnCodec tokens), `LTX-Video` combines a **highly anisotropic `T5-XXL` `[128, 4096]` cross-attention manifold** with a **`1:8192` (`32×32×8`) `patch_size=1` latent space** where each `(h, w)` voxel encodes a full `32×32` pixel macro-block. To drive all **96,000 modes without dimensionality reduction** and without off-manifold collapse:

1. **Empirical `T5-XXL` Kent Manifold (`get_t5_output_manifold` + `t5_output_manifold_*.pt`):**
   Grounded in *HEART* (*Roy et al., 2026*) [arXiv:2605.07973], `T5-XXL` output hidden states occupy an anisotropic Kent cone around a non-zero mean $\boldsymbol{\mu}_{\text{T5}}$ with decaying singular spectrum $\boldsymbol{\sigma}_{\text{T5}}$. Rather than injecting random `4096-D` noise, `LTXModelBackend` extracts $(\boldsymbol{\mu}_{\text{T5}}, \mathbf{V}_{\text{T5}}, \boldsymbol{\sigma}_{\text{T5}}, r_{\text{T5}})$ directly from `T5-XXL` outputs and caches it on disk (`QUANT_T5_DIR`).
2. **Active-Token Intra-Manifold Steering & Theta-Gamma Multiplexing (`neuro_controller.py`):**
   * **Within a region (`steer_pool_t5_within_manifold`):** Masks out `PAD` and `</s>` tokens, computes SVD strictly on the active content tokens of the pool (`S > 1e-3 * S[0]`, supplemented by $\boldsymbol{\sigma}_{\text{T5}} \mathbf{V}_{\text{T5}}$ when $K=1$), and rotates along real semantic axes within a safe cone ($\theta \le 0.08\pi \cdot R_{\text{WM}}$) while preserving exact per-token `T5-XXL` norms (*RichSpace*, *Cao et al., 2025*).
   * **Across regions (`multiplex_t5_concept_slots`):** Packs all simultaneously active entities (`e0..e4`, e.g., vent plume + bacterial mat + tube worms + abyssal current in `mode_06_black_smoker_ecosphere.json`) into non-overlapping token slots in `[1, 128, 4096]` with 96,000-D WM gain modulation (`0.72..1.02`).
   * **Intra-step live updates (`live_wm_step_fn`):** Updates `curr_pos_emb` cleanly on the fly inside the 8 denoising steps without perturbing the ODE trajectory (`sigma > 0.94`).
3. **Spatially-Coherent 2D-DCT Spectral $S^3$ Noise (`unfold_wm_spectral_3d_s3`):**
   Grounded in *FreeInit* (*Wu et al., 2024*) [arXiv:2312.07537] and *FreqPrior* (*Yuan et al., 2025*) [arXiv:2502.03496]:
   * **`stochastic = 1.0` (Out-of-the-Box Parity):** Pure `torch.randn(shape)` and unperturbed `T5-XXL` prompts — identical to official LTX-Video demos.
   * **`stochastic = 0.0` (100% Working Memory Drive):** Zero random draws; all `96,000` WM modes `[5 × 5 × 32 × 120]` rotate 4 deterministic orthogonal Gaussian carriers $(B_1 \dots B_4)$ on $S^3$ across **128 latent channels $\times F_{\text{lat}}$ chronological frames $\times$ smooth 2D-DCT spatial cosine harmonics (`K_sp = 15`)** with advancing deterministic `win_seed = base_seed + _window_step_idx * 997`. Preserves exact $\mathcal{N}(0, I)$ statistics, zero `32×32` patch tearing, and zero standing-wave stripes.

---

## 4. Multi-Branch Rectified-Flow 3D-DiT Oracle (`neuro_referee.py`)

Instead of a static 2D classifier or single-stage lock, `CoordinateFreeHistoryReferee` uses **LTX-Video's own 3D-DiT** to evaluate all competing world branches directly on the generated spatiotemporal transition $\mathbf{z}_{\text{prev}} \to \mathbf{z}_{\text{new}}$:

1. Keeps prefix frames `:n_cond` clean ($t_c = 0$) and noises generated frames `n_cond:` at $\sigma_{\text{probe}}$ (`0.725` for `0.9.8-distilled`, `0.55` for `0.9.0-base`).
2. Evaluates `[H_stay, H_branch_1, ..., H_branch_M]` over full `[1, 128, 4096]` T5 embeddings in VRAM-safe chunks (`max_batch=4`), averaged across `n_noise_probes=2` shared noise draws $\boldsymbol{\epsilon}$ (Common Random Numbers).
3. Computes motion-weighted paired $t$-statistics using effective spatial-temporal degrees of freedom (`n_eff = n_tok / 16.0`), eliminating neutral-scene probability jitter.
4. Produces live video-grounded `reaction_branch_scores` (`branches`), triggers `BRANCH_SWITCH` with hysteresis (`t_win > 0.85`, `Δt > 0.65`), and routes unbiased `±reward` to `npc:` controllers.

---

## 5. LTX-Video 0.9.8-2B-Distilled NF4 & `T5 ↔ VAE` Zero-OOM Architecture

| Component / Mechanism | Implementation (`neuro_ltx_backend.py` & `video_world_server.py`) |
| :--- | :--- |
| **In-Place 4-Bit NF4 Quantization** | `quantize_model_inplace_bnb()` loads `ltxv-2b-0.9.8-distilled.safetensors` on CPU and streams 284 `bnb.nn.Linear4bit` (NF4) layers onto CUDA (~1.6 GB VRAM for the 2B 3D-DiT). |
| **Automatic `T5 ↔ VAE` VRAM Swap** | `T5-XXL 4-bit` (`bfloat16`) and `0.9.8 3D-VAE` are mutually exclusive in VRAM when `--allow-t5-offload` is active (default): during text encoding, `VAE` moves to CPU (~0.1s); before any Transformer/VAE step, `T5` is evicted and `VAE` returns to CUDA (`3.27 GB` server VRAM). |
| **T5 Kent Manifold Caching** | `get_t5_output_manifold()` pre-computes & caches `t5_output_manifold_128x4096.pt` in `QUANT_T5_DIR` at server startup and serves it via `get_t5_manifold` RPC. |
| **Clean 3D-VAE Feedback Loop** | `vae_tile_min >= 512px` on `> 10 GB` GPUs (prevents `256px` tile seams on `384×256` / `512×384`); clean `decode_noise_scale=0.0` + `adain=0.0` stops 300-step autoregressive contrast washout; `scheduler.step` on `latents[:, :, 1:]` matches `pipeline_ltx_image2video.py`. |
| **Exact Distilled Sigma Schedule** | `build_distilled_sigmas(steps)` uses all 8 official sigmas `[1.0000 ... 0.4219]` at `--steps 8` and interpolates preserving both endpoints for any other step count. |

---

## 6. Canonical Module Map & Ports

| File | Port / Authkey | Role |
| :--- | :--- | :--- |
| **`video_world_server.py`** | `localhost:6006` (`b"video_world"`) | Immortal LTX-Video weights RPC server (`0.9.8-distilled` NF4 + `T5 ↔ VAE` auto-swap + `get_t5_manifold`). |
| **`neuro_heterarchy_video.py`** | `:6002` (frames) / `:6007` (ctrl) | Unified 3D-DiT runner supporting `.json` manifests, `.txt` lores, `--init-image`, 60 Hz BLE/LSL→VRAM bus, `SmoothRealtimeVideoPacer`, and Web Studio bridge. |
| **`neuro_ltx_backend.py`** | — | `LTXModelBackend`, `RemoteLTXModelBackend`, in-place NF4/INT8 quantization, `T5 ↔ VAE` swap, T5 Kent manifold cache, distilled/base schedules. |
| **`neuro_engine.py`** | — | `UniversalLatentAutobattlerEngine`: pure-tensor N-agent orchestrator, multi-slot `multiplex_t5_concept_slots`, empty-prompt Kent synthesis, live intra-step updates, NPC feedback. |
| **`neuro_controller.py`** | — | `SymmetricBotController`, `steer_pool_t5_within_manifold`, `synthesize_empty_t5_from_kent_manifold`, `multiplex_t5_concept_slots`, 24-ch Gamepad & `npc:` policy. |
| **`neuro_solver_stepper.py`** | — | `LatentFlowWindowStepper`: 2D-DCT Spectral $S^3$ 96,000-D wave integrator (`stochastic=0.0` vs native `1.0`), clean I2V Euler step. |
| **`neuro_referee.py`** | — | `CoordinateFreeHistoryReferee`: chunked multi-branch 3D-DiT Rectified-Flow Oracle (`n_noise_probes=2`, effective DoF `t-stat`). |
| **`neuro_world.py`** | — | N-entity manifest dataclasses & JSON loader (`GameWorldManifest`, `StageSpec`, `StageEntitySlot`, `GenerativeQuestSpec`). |
| **`web_cloud_gateway.py` & `static/`** | `:8080` (HTTPS) | Unified Web Studio UI with FreeEEG16 Web-BLE, Gradual WM Mixer (`eeg_wm_gain` × `gamepad_wm_mix`), 24-ch Gamepad poller & `🎯` concept binding. |

---

## 7. Quick Start & CLI Reference

```bash
# Terminal 1: Immortal Video World Weights Server (port 6006, 0.9.8-distilled NF4, T5<->VAE swap, no 256px tile seams)
python3 video_world_server.py --port 6006 --model distilled --quant nf4 --vae-tile-min 512 --allow-t5-offload

# Terminal 2: Unified Web Studio Gateway (:8080)
python3 web_cloud_gateway.py --port 8080 --cert cert.pem --key key.pem --video-lore configs/mode_06_black_smoker_ecosphere.json

# Terminal 3A: Black Smoker Autocatalytic Ecosphere — 100% Full-Rank WM Drive (stochastic=0.0)
python3 neuro_heterarchy_video.py --manifest mode_06_black_smoker_ecosphere.json --steps 8 --cfg 1.0 --stochastic 0.0 --wm-radius 0.35 --context-latents 1 --step-latents 2

# Terminal 3B: Out-of-the-Box Native LTX-Video Demo Parity (stochastic=1.0)
python3 neuro_heterarchy_video.py --manifest mode_06_black_smoker_ecosphere.json --steps 8 --cfg 1.0 --stochastic 1.0 --context-latents 1 --step-latents 2

# Terminal 3C: 4-Way Movement / Catalytic Control Test (default ball mode)
./neuro_heterarchy_video.sh

# Terminal 3D: 1v1 Catalytic Duel vs Autonomous Adaptive NPC (e0=Human, e1=npc:AFz)
python3 neuro_heterarchy_video.py --mode duel --steps 8 --cfg 1.5 --eeg-wm-gain 0.0 --gamepad-wm-mix 1.0

# Terminal 3E: Custom Starting PNG Map (e.g. Real Maze init frame)
python3 neuro_heterarchy_video.py --manifest configs/mode_05_wm_control_test_arena.json --init-image maze_start.png --steps 8 --cfg 1.5
```

---

## 8. Current Status, Remaining Pre-Release Fixes & Multi-View Roadmap

### ✅ Verified Working (as of Nov 22, 2026)
* **Zero OOM Across Full Lifecycle:** `0.9.8-2B-Distilled` NF4 + `T5 ↔ VAE` auto-swap runs stably in `3.27 GB` server VRAM (`~1.5s/step`, `RTF ~1.0x`) and swaps cleanly on live browser edits.
* **Dynamics, Variability & Sharpness Restored:** Spectral 2D-DCT $S^3$ (`unfold_wm_spectral_3d_s3`) + clean `decode_noise_scale=0.0` + `vae_tile_min >= 512px` eliminated the `32×32` patch grid tearing, standing-wave vertical stripes, and 300-step VAE contrast washout.
* **Multi-Agent Composition Restored:** `multiplex_t5_concept_slots` packs all active entities (`e0..e4`) cleanly into `[1, 128, 4096]`.

### 🔧 4 Final Pre-Release Fixes (Next Session Checklist)
1. **Empty Prompt (`""`) Full-Vocabulary `T5-XXL` Manifold (`neuro_ltx_backend.py`):**
   * *Symptom:* Current `get_t5_output_manifold` sampled random subword IDs `[100..31900]` (which include punctuation/OCR fragments/numbers), causing LTX-Video to render printed letters/text at empty prompt with limited scene variety.
   * *Fix:* Filter `T5TokenizerFast.get_vocab()` down to real whole alphabetic English words (`tok.startswith("▁")`, ASCII alpha, len 3–12; ~10k–14k words, **zero hardcoded words in Python**), encode all of them once on RTX 3060 (~4s, cached to `t5_full_vocab_manifold_512.pt`), and extract a 512-axis Kent ellipsoid + 512 Farthest-Point concept attractors.
2. **Eliminate `#0` Item Bias ("Mountains" Lock) in `world_lore.txt` (`neuro_controller.py`):**
   * *Symptom:* `get_canonical_concept_wm_basis_Kx5x32x120` assigns `freq_s = k_i + 1`, giving item `#0` (`freq_s = 1`, e.g., mountains) massive energy advantage over items `5..29` under low-frequency EEG/synth signals, and winner-take-all ignores the rest of the pool.
   * *Fix:* Build an **isotropic frequency-balanced `QR` basis** across all $K$ concepts (equal expected energy per concept) and map the region's 19,200-D WM continuously across the full $K$-concept *RichSpace* simplex.
3. **Unfreeze `context_latents = 2, 3` (`neuro_engine.py` & `neuro_solver_stepper.py`):**
   * *Symptom:* At `context_latents >= 2`, `frame0_lat` and `extra_tail` both ended on $t_{\text{end}}$ (backward/duplicate overlap) while `sigma > 0.90` `lerp` pulled frame `1` back toward frame `0`, freezing motion.
   * *Fix:* Order context strictly chronologically `[past_{t-8}, ..., frame0_t]`, use `step_latents >= 3` with `LTXI2VLongMultiPromptPipeline`-style soft overlap (`mask = 0.5`, `t_per_token = t * mask`), and remove backward `lerp` pulls on steps `0..4`.
4. **Restore `vUnassignedStochastic` Slider & Arbitrary Multi-Bot Monty Control (`static/index.html`, `neuro_heterarchy_video.py`, `neuro_engine.py`):**
   * *Symptom:* `#vUnassignedStochastic` is hidden with `style="display:none;"` in `static/index.html`, and `monty_mode` / `unassigned_stochastic` are not wired to control `npc:` bots and empty cortical regions.
   * *Fix:* Un-hide `#vUnassignedStochastic` and connect `monty_mode` (`off | missing_only | full`) + `unassigned_stochastic` so any number of `npc:<REGION>` bots (driven by `AmodalMontyColumnGPU` or disabled cleanly when `monty_mode == "off"`) work symmetrically across all regions.

### 🚀 Post-Release Roadmap: Multi-User / Multi-View Shared World (TBT 2.0 + NVIDIA-Style World Model)
* Grounded in **TBT 2.0** (*Hawkins et al., 2026*, Sec. 3.1–3.2): maintain **one shared allocentric latent world state & L3 object consensus** across all users (`User1..UserN`), while each user's thalamic/camera relay applies an **egocentric viewpoint transform** (per-user `video_coords` / camera pose conditioning in batched `B = N_users` or multi-view attention) so multiple participants simultaneously perceive and act inside the same world from different perspectives.

---

## 9. Comprehensive Scientific Bibliography & DOIs

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
13. **Roy, A., Das Biswas, S., & Roy, K. (2026).** *HEART: Hyperspherical Embedding Alignment via Kent-Representation Traversal in Diffusion Models.* arXiv:2605.07973. [https://doi.org/10.48550/arXiv.2605.07973](https://doi.org/10.48550/arXiv.2605.07973)
14. **Cao, Y., et al. (2025).** *RichSpace: Enriching Text-to-Video Prompt Space via Text Embedding Interpolation.* arXiv:2501.09982. [https://doi.org/10.48550/arXiv.2501.09982](https://doi.org/10.48550/arXiv.2501.09982)
15. **Wu, T., Si, C., Jiang, Y., Huang, Z., & Liu, Z. (2024).** *FreeInit: Bridging Initialization Gap in Video Diffusion Models.* ECCV 2024 / arXiv:2312.07537. [https://doi.org/10.48550/arXiv.2312.07537](https://doi.org/10.48550/arXiv.2312.07537)
16. **Yuan, Y., et al. (2025).** *FreqPrior: Improving Video Diffusion Models with Frequency Filtering Gaussian Noise.* ICLR 2025 / arXiv:2502.03496. [https://doi.org/10.48550/arXiv.2502.03496](https://doi.org/10.48550/arXiv.2502.03496)

