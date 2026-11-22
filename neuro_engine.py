#!/usr/bin/env python3
"""
neuro_engine.py — Полноранговый 96,000-D 3D-DiT Движок (RichSpace + Kent-T5 + Spectral S^3)
=============================================================================================
- Мультиплексирует ВСЕ активные сущности (e0..e4) через multiplex_t5_concept_slots.
- При пустом промпте использует конус Кента T5-XXL (synthesize_empty_t5_from_kent_manifold).
- На лету внутри 8 шагов денойзинга обновляет T5-эмбеддинги от 60 Гц шины рабочей памяти и геймпада.
- Убирает 300-кратное мыло авторегрессии: чистый декод якоря (decode_noise_scale=0.0)
  и сохранение импульса при context_latents > 1.
=============================================================================================
"""

from __future__ import annotations
from collections import deque
import io
import os
import multiprocessing as mp
from multiprocessing.connection import Client
import threading
import time
import gc
import cv2
import numpy as np
from PIL import Image
import torch
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
mp.current_process().authkey = b"video_world"

from neuro_world import GameWorldManifest, StageEntitySlot
from neuro_controller import (
    SymmetricBotController,
    DEFAULT_REGION_BINDINGS,
    multiplex_t5_concept_slots,
    synthesize_empty_t5_from_kent_manifold,
)
from neuro_ltx_backend import LTXModelBackend, RemoteLTXModelBackend
from neuro_referee import (
    CoordinateFreeHistoryReferee,
    LatentIdentityFrameTelemetry,
    R_EDGE_OF_CHAOS,
)
from neuro_solver_stepper import LatentFlowWindowStepper

SLOT_MACRO_TO_MICRO_RANK = {4: 0, 1: 1, 2: 2, 3: 3, 0: 4}


class UniversalLatentAutobattlerEngine:

    def __init__(
        self,
        manifest: GameWorldManifest,
        width: int = 384,
        height: int = 256,
        prep_seconds: float = 1.0,
        combat_seconds: float = 1.0,
        game_mode: str = "strict_ko",
        max_overtime_chunks: int = 12,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        video_server_port: int = 6006,
        battle_server_port: int | None = None,
        force_local: bool = False,
        monochrome: bool = False,
        context_lat_frames: int = 1,
        step_lat_frames: int = 2,
        stochastic: float = 0.0,
        wm_radius: float = 0.35,
    ):
        self.manifest = manifest
        self.width = max(256, (width // 32) * 32)
        self.height = max(192, (height // 32) * 32)
        self.monochrome = bool(monochrome)

        self.game_mode = game_mode
        self.steps = max(2, int(steps))
        self.cfg_scale = float(cfg_scale)
        self.base_seed = int(seed)
        self.video_server_port = int(
            battle_server_port if battle_server_port is not None else video_server_port
        )
        self.force_local = bool(force_local)

        self.context_lat_frames = max(1, int(context_lat_frames))
        self.step_lat_frames = max(1, int(step_lat_frames))
        self.stochastic = float(np.clip(stochastic, 0.0, 1.0))
        self.wm_radius = float(np.clip(wm_radius, 0.0, 1.0))

        self.eeg_wm_gain: float = 1.0
        self.gamepad_wm_mix: float = 1.0
        self._gp_lock = threading.Lock()
        self.gamepad_channels: dict[str, float] = {}
        self._latched_gamepad_channels: dict[str, float] = {}
        self.concept_bindings_by_region: dict[str, list[str]] = {
            k: list(v) for k, v in DEFAULT_REGION_BINDINGS.items()
        }

        self._cmd_lock = threading.Lock()
        self._pending_manifest: tuple[GameWorldManifest, str] | None = None
        self._pending_sections_fn = None
        self._pending_init_image: bytes | None = None
        self._pending_reset_genesis: bool = False
        self._just_reset_visual_stream: bool = False

        self.curr_idx = 0
        self.block_idx = 0
        self.total_rendered_frames = 0
        self.maneuver_phase: int = 0
        self.quest_progress_pct: float = 0.0
        self.need_hard_genesis: bool = True

        self.active_entity_parties: list[list[str]] = []
        self.persistent_entity_states: dict[str, str] = {}
        self.last_active_pos_pair: tuple[torch.Tensor, torch.Tensor] | None = None
        self.entity_coherences: list[float] = []
        self.entity_transitions: list[float] = []

        self.active_concept_idx_by_region: dict[str, int] = {
            "Fpz": -1, "AFz": -1, "F3": -1, "F4": -1, "FCz": -1
        }
        self.active_concept_indices_by_region: dict[str, list[int]] = {
            "Fpz": [], "AFz": [], "F3": [], "F4": [], "FCz": []
        }
        self.region_choice_probs: dict[str, list[float]] = {
            "Fpz": [], "AFz": [], "F3": [], "F4": [], "FCz": []
        }

        self.entity_controllers: dict[str, SymmetricBotController] = {}
        self.source_mode_overrides: dict[str, str] = {}

        self.local_genesis_5d: torch.Tensor | None = None
        self.local_context_latents_5d: torch.Tensor | None = None
        self.recent_clean_frames_rgb: deque[np.ndarray] = deque(maxlen=33)

        self.backend = None
        if not self.force_local and self.video_server_port > 0:
            try:
                conn = Client(
                    ("localhost", self.video_server_port), authkey=b"video_world"
                )
                conn.send({"cmd": "ping"})
                ping_resp = conn.recv()
                if isinstance(ping_resp, dict) and ping_resp.get("status") == "ok":
                    self.backend = RemoteLTXModelBackend(
                        conn=conn, ping_info=ping_resp
                    )
                    print(
                        f"⚡ [ENGINE] Подключено к video_world_server (localhost:{self.video_server_port}, "
                        f"distilled={self.backend.is_distilled}). "
                        f"Конус Кента T5-XXL + 96,000-D Spectral S^3 активны!"
                    )
            except Exception:
                self.backend = None

        if self.backend is None:
            print("ℹ️ [ENGINE] Запуск локального LTX-Video в VRAM...")
            self.backend = LTXModelBackend()

        self.device = self.backend.device
        self.dtype = self.backend.dtype
        self.t5_manifold = self.backend.get_t5_output_manifold(max_rank=128)

        is_dist = bool(getattr(self.backend, "is_distilled", False))
        self.referee = CoordinateFreeHistoryReferee(
            self.backend.transformer,
            self.device,
            self.dtype,
            probe_sigma=0.725 if is_dist else 0.55,
            n_noise_probes=2,
            max_batch=4,
        )
        self.stepper = LatentFlowWindowStepper(
            backend=self.backend,
            referee=self.referee,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            guidance_rescale=0.7,
            image_cond_noise_scale=0.05,
        )

        self._warmup_manifest_semantics()

    def request_manifest_switch(self, new_manifest: GameWorldManifest, label: str = ""):
        with self._cmd_lock:
            self._pending_manifest = (new_manifest, label)
            self._pending_sections_fn = None

    def request_lore_sections_update(self, apply_fn):
        with self._cmd_lock:
            self._pending_sections_fn = apply_fn

    def request_init_image_bytes(self, img_bytes: bytes):
        with self._cmd_lock:
            self._pending_init_image = img_bytes

    def request_reset_genesis(self):
        with self._cmd_lock:
            self._pending_reset_genesis = True

    def consume_visual_reset_flag(self) -> bool:
        val = self._just_reset_visual_stream
        self._just_reset_visual_stream = False
        return val

    @torch.no_grad()
    def _apply_pending_commands(self):
        with self._cmd_lock:
            pend_manifest = self._pending_manifest
            pend_sec_fn = self._pending_sections_fn
            pend_img = self._pending_init_image
            pend_reset = self._pending_reset_genesis
            self._pending_manifest = None
            self._pending_sections_fn = None
            self._pending_init_image = None
            self._pending_reset_genesis = False

        if pend_manifest is not None:
            new_m, label = pend_manifest
            self.manifest = new_m
            self.curr_idx = 0
            self.maneuver_phase = 0
            self.quest_progress_pct = 0.0
            self.need_hard_genesis = True
            self.active_entity_parties = []
            self.persistent_entity_states.clear()
            self.last_active_pos_pair = None
            self.entity_controllers.clear()
            self.local_genesis_5d = None
            self.local_context_latents_5d = None
            self.recent_clean_frames_rgb.clear()
            self.stepper.smooth_wm_drift_5x5x32x120 = None
            self._warmup_manifest_semantics()
            self._just_reset_visual_stream = True
            print(
                f"🌍 [ENGINE WORLD SWITCH] Применён мир: {label or new_m.world_name}",
                flush=True,
            )

        if pend_sec_fn is not None:
            pend_sec_fn(self.manifest)
            self.active_entity_parties = []
            self.persistent_entity_states.clear()
            self.last_active_pos_pair = None
            self._warmup_manifest_semantics()
            print(
                "📝 [ENGINE LORE EDITOR] Пулы концептов 3D-DiT обновлены!",
                flush=True,
            )

        if pend_reset:
            self.need_hard_genesis = True
            self.maneuver_phase = 0
            self.quest_progress_pct = 0.0
            self.last_active_pos_pair = None
            self.local_genesis_5d = None
            self.local_context_latents_5d = None
            self.recent_clean_frames_rgb.clear()
            self.stepper.smooth_wm_drift_5x5x32x120 = None
            for ctrl in self.entity_controllers.values():
                ctrl.autonomous_wm_field_5x32x120 = None
            self._just_reset_visual_stream = True
            print("🔄 [ENGINE RESET] Сброс генезиса 3D-DiT выполнен!", flush=True)

        if pend_img:
            self.set_init_image_bytes(pend_img)
            self._just_reset_visual_stream = True

    @torch.no_grad()
    def set_init_image_bytes(self, img_bytes: bytes):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            rgb_np = np.array(pil_img, dtype=np.uint8)
            lat_5d = self.backend.encode_frame_to_latent(rgb_np)
            self.local_context_latents_5d = lat_5d.detach().clone()
            self.local_genesis_5d = lat_5d[:, :, :1].detach().clone()
            self.recent_clean_frames_rgb.clear()
            self.recent_clean_frames_rgb.append(rgb_np)
            self.need_hard_genesis = False
            self.maneuver_phase = 0
            print(f"🖼️ [ENGINE] Стартовая сцена закодирована в z[0] ({self.width}x{self.height})!", flush=True)
        except Exception as e:
            print(f"⚠️ [ENGINE Init Image Error]: {e}", flush=True)

    def set_gamepad_channels(self, channels: dict[str, float]):
        cleaned = {
            str(k).upper(): float(np.clip(float(v), 0.0, 1.0))
            for k, v in (channels or {}).items()
            if float(v) > 0.04
        }
        with self._gp_lock:
            self.gamepad_channels = cleaned
            for k, v in cleaned.items():
                self._latched_gamepad_channels[k] = max(
                    self._latched_gamepad_channels.get(k, 0.0), v
                )

    def _consume_step_gamepad_channels(self) -> dict[str, float]:
        with self._gp_lock:
            merged = dict(self._latched_gamepad_channels)
            for k, v in self.gamepad_channels.items():
                merged[k] = max(merged.get(k, 0.0), v)
            self._latched_gamepad_channels = dict(self.gamepad_channels)
            return merged

    def _peek_live_gamepad_channels(self) -> dict[str, float]:
        with self._gp_lock:
            merged = dict(self._latched_gamepad_channels)
            for k, v in self.gamepad_channels.items():
                merged[k] = max(merged.get(k, 0.0), v)
            return merged

    def set_concept_bindings(self, bindings_by_region: dict[str, list[str]]):
        for reg_k, b_list in (bindings_by_region or {}).items():
            r_up = str(reg_k).upper()
            if isinstance(b_list, list):
                self.concept_bindings_by_region[r_up] = [
                    str(x).strip().upper() for x in b_list
                ]

    def set_gamepad_input(
        self,
        lx: float = 0.0,
        ly: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        manual_idx: int = -1,
        target_slot: str = "e0",
        channels: dict[str, float] | None = None,
    ):
        if channels is not None:
            self.set_gamepad_channels(channels)
            return
        ch: dict[str, float] = {}
        if ly > 0.08:
            ch["LS_UP"] = float(ly)
        elif ly < -0.08:
            ch["LS_DOWN"] = float(-ly)
        if lx < -0.08:
            ch["LS_LEFT"] = float(-lx)
        elif lx > 0.08:
            ch["LS_RIGHT"] = float(lx)
        if ry > 0.08:
            ch["RS_UP"] = float(ry)
        elif ry < -0.08:
            ch["RS_DOWN"] = float(-ry)
        if rx < -0.08:
            ch["RS_LEFT"] = float(-rx)
        elif rx > 0.08:
            ch["RS_RIGHT"] = float(rx)
        self.set_gamepad_channels(ch)

    @torch.no_grad()
    def _warmup_manifest_semantics(self):
        t0 = time.perf_counter()
        prompts_to_warm = set()
        prompts_to_warm.add("")
        prompts_to_warm.add(self.manifest.negative_desc)
        if self.manifest.static_hold_desc:
            prompts_to_warm.add(self.manifest.static_hold_desc)
        if self.manifest.arena_context_desc:
            prompts_to_warm.add(self.manifest.arena_context_desc)

        for _, arch in self.manifest.archetypes.items():
            sentence = f"{arch.essence_desc} {arch.action_desc}".strip()
            if sentence and not sentence.endswith("."):
                sentence += "."
            probe_p = (
                f"{self.manifest.arena_context_desc}. {sentence}".strip()
                if self.manifest.arena_context_desc
                else sentence
            )
            prompts_to_warm.add(probe_p)

        for st in self.manifest.stages:
            prefix = f"{self.manifest.arena_context_desc}. " if self.manifest.arena_context_desc else ""
            if st.quest.ongoing_hypothesis:
                prompts_to_warm.add(f"{prefix}{st.quest.ongoing_hypothesis}".strip())
            if st.quest.completed_hypothesis:
                prompts_to_warm.add(f"{prefix}{st.quest.completed_hypothesis}".strip())
            if st.quest.failed_hypothesis:
                prompts_to_warm.add(f"{prefix}{st.quest.failed_hypothesis}".strip())

        print(
            f"🔥 [T5 WARMUP] Предвычисление {len(prompts_to_warm)} тензоров эмбеддингов...",
            flush=True,
        )
        for p in prompts_to_warm:
            self.backend.encode_semantics(p, max_len=128)
        dt = time.perf_counter() - t0
        print(f"✅ [T5 WARMUP] Готово за {dt:.1f}с!", flush=True)

        vram_after = self.backend.offload_t5_to_cpu()
        if vram_after:
            print(
                f"🧹 [T5 WARMUP] T5 выгружен из VRAM (VRAM сервера: {vram_after:.2f} GB).",
                flush=True,
            )

    def set_entity_source(self, slot_id: str, source_mode: str):
        clean_mode = str(source_mode).strip()
        if clean_mode.lower() == "auto":
            self.source_mode_overrides.pop(slot_id, None)
            return
        self.source_mode_overrides[slot_id] = clean_mode
        if slot_id in self.entity_controllers:
            self.entity_controllers[slot_id].source_mode = clean_mode

    def set_all_entities_source(self, source_mode: str):
        clean_mode = str(source_mode).strip()
        if clean_mode.lower() == "auto":
            self.source_mode_overrides.pop("*", None)
            return
        self.source_mode_overrides["*"] = clean_mode
        for ctrl in self.entity_controllers.values():
            ctrl.source_mode = clean_mode

    def _get_effective_binding(self, slot: StageEntitySlot, idx: int) -> str:
        slot_key = slot.slot_id or f"e{idx}"
        ov_slot = self.source_mode_overrides.get(slot_key)
        ov_all = self.source_mode_overrides.get("*")
        if ov_slot and ov_slot.lower() != "auto":
            return ov_slot
        if ov_all and ov_all.lower() != "auto":
            return ov_all
        return slot.control_binding

    def _get_controller(
        self,
        slot: StageEntitySlot,
        idx: int,
        step_gp_channels: dict[str, float] | None = None,
    ) -> SymmetricBotController:
        slot_key = slot.slot_id or f"e{idx}"
        binding = self._get_effective_binding(slot, idx)
        if slot_key not in self.entity_controllers:
            self.entity_controllers[slot_key] = SymmetricBotController(
                slot_key,
                self.manifest.archetypes,
                source_mode=binding,
                stochastic=self.stochastic,
                wm_radius=self.wm_radius,
            )
        ctrl = self.entity_controllers[slot_key]
        ctrl.source_mode = binding
        ctrl.stochastic = self.stochastic
        ctrl.wm_radius = self.wm_radius
        ctrl.eeg_wm_gain = self.eeg_wm_gain
        ctrl.gamepad_wm_mix = self.gamepad_wm_mix
        ctrl.gamepad_channels = (
            step_gp_channels if step_gp_channels is not None else self.gamepad_channels
        )
        ctrl.concept_bindings_by_region = self.concept_bindings_by_region
        ctrl.t5_manifold = self.t5_manifold
        ctrl.attempt_num = self.block_idx + 1
        return ctrl

    def _build_pool_candidates_payload(
        self, stage
    ) -> tuple[list[list[dict]], list[str]]:
        pool_payload: list[list[dict]] = []
        curr_states: list[str] = []
        st0 = self.manifest.stages[0] if self.manifest.stages else stage

        for idx_s, slot in enumerate(stage.entities):
            s_key = slot.slot_id or f"e{idx_s}"
            pool_ids = (
                list(st0.entities[idx_s].allowed_pool)
                if idx_s < len(st0.entities)
                else list(slot.allowed_pool)
            )
            if s_key not in self.persistent_entity_states and pool_ids:
                self.persistent_entity_states[s_key] = pool_ids[0]
            curr_states.append(
                self.persistent_entity_states.get(
                    s_key, pool_ids[0] if pool_ids else ""
                )
            )

            slot_candidates = []
            for aid in pool_ids:
                arch = self.manifest.archetypes.get(aid)
                if arch:
                    sentence = f"{arch.essence_desc} {arch.action_desc}".strip()
                    if sentence and not sentence.endswith("."):
                        sentence += "."
                    probe_p = (
                        f"{self.manifest.arena_context_desc}. {sentence}".strip()
                        if self.manifest.arena_context_desc
                        else sentence
                    )
                    slot_candidates.append(
                        {
                            "state_id": aid,
                            "display_name": arch.display_name,
                            "probe_prompt": probe_p,
                        }
                    )
            pool_payload.append(slot_candidates)
        return pool_payload, curr_states

    def _build_candidate_branches_payload(
        self,
    ) -> tuple[list[dict], list[tuple[torch.Tensor, torch.Tensor]]]:
        branches = []
        hyp_pairs = []
        prefix_ctx = (
            f"{self.manifest.arena_context_desc}. "
            if self.manifest.arena_context_desc
            else ""
        )
        for s_idx, st in enumerate(self.manifest.stages):
            if not st.quest.completed_hypothesis.strip():
                continue
            branches.append(
                {
                    "stage_idx": s_idx,
                    "global_stage": st.global_stage,
                    "driver_tag": st.quest.driver_tag,
                    "reaction_formula": st.quest.reaction_formula,
                    "active_slot_idx": st.quest.active_slot_idx,
                    "catalyst_slot_idx": st.quest.catalyst_slot_idx,
                }
            )
            h_txt = f"{prefix_ctx}{st.quest.completed_hypothesis}".strip()
            hyp_pairs.append(self.backend.encode_semantics(h_txt, max_len=128))
        return branches, hyp_pairs

    @torch.no_grad()
    def _compose_active_pos_pair(
        self,
        active_slots: list[int],
        steered_probe_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        slot_weights: list[float],
        full_wm_5x5x32x120: torch.Tensor,
        empty_pair: tuple[torch.Tensor, torch.Tensor],
        passive_pair: tuple[torch.Tensor, torch.Tensor],
        has_any_pool_items: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ordered_slots = sorted(
            set(active_slots),
            key=lambda s_i: SLOT_MACRO_TO_MICRO_RANK.get(s_i, 10 + s_i),
        )
        if not ordered_slots:
            if not has_any_pool_items and not self.manifest.static_hold_desc.strip():
                # Полностью пустой промпт -> свободная навигация внутри конуса Кента T5-XXL!
                return synthesize_empty_t5_from_kent_manifold(
                    empty_pair=empty_pair,
                    t5_manifold=self.t5_manifold,
                    wm_reg_5x32x120=full_wm_5x5x32x120.mean(dim=0),
                    wm_radius=self.wm_radius,
                    stochastic=self.stochastic,
                )
            if self.manifest.static_hold_desc.strip():
                return passive_pair[0], passive_pair[1]
            if self.last_active_pos_pair is not None:
                return self.last_active_pos_pair
            return passive_pair[0], passive_pair[1]

        if len(ordered_slots) == 1:
            pos_emb, pos_mask = steered_probe_pairs[ordered_slots[0]]
        else:
            pos_emb, pos_mask = multiplex_t5_concept_slots(
                embeds_list=[steered_probe_pairs[s_i][0] for s_i in ordered_slots],
                masks_list=[steered_probe_pairs[s_i][1] for s_i in ordered_slots],
                weights=[slot_weights[s_i] for s_i in ordered_slots],
                max_len=128,
            )
        self.last_active_pos_pair = (pos_emb.detach().clone(), pos_mask.detach().clone())
        return pos_emb, pos_mask

    @torch.no_grad()
    def execute_turn_step(
        self,
    ) -> tuple[
        list[np.ndarray],
        list[LatentIdentityFrameTelemetry],
        float,
        float,
        str,
        dict,
    ]:
        t0 = time.perf_counter()
        self._apply_pending_commands()
        if not self.manifest.stages:
            raise RuntimeError("Manifest has no stages")
        self.curr_idx = self.curr_idx % len(self.manifest.stages)
        stage = self.manifest.stages[self.curr_idx]
        n_ents = max(1, len(stage.entities))

        step_gp_channels = self._consume_step_gamepad_channels()
        entity_pool_candidates, entity_current_states = self._build_pool_candidates_payload(stage)
        has_any_pool_items = any(len(p) > 0 for p in entity_pool_candidates)

        if self.maneuver_phase == 0 or not self.active_entity_parties:
            self.active_entity_parties = [
                [item["state_id"] for item in pool] for pool in entity_pool_candidates
            ]
            self.entity_coherences = [100.0] * n_ents
            self.entity_transitions = [
                0.0 if i == stage.quest.active_slot_idx else 100.0
                for i in range(n_ents)
            ]
            self.quest_progress_pct = 0.0

        reset_stage_stream = self.need_hard_genesis
        self.need_hard_genesis = False
        if reset_stage_stream:
            self.recent_clean_frames_rgb.clear()
            self.local_context_latents_5d = None

        candidate_branch_specs, candidate_branch_hyp_pairs = (
            self._build_candidate_branches_payload()
        )
        has_catalytic_oracle = bool(
            stage.quest.completed_hypothesis.strip() or len(candidate_branch_specs) > 0
        )

        prefix_ctx = (
            f"{self.manifest.arena_context_desc}. "
            if self.manifest.arena_context_desc
            else ""
        )
        ongoing_hyp = f"{prefix_ctx}{stage.quest.ongoing_hypothesis}".strip()
        completed_hyp = f"{prefix_ctx}{stage.quest.completed_hypothesis}".strip()
        failed_hyp = (
            f"{prefix_ctx}{stage.quest.failed_hypothesis}".strip()
            if stage.quest.failed_hypothesis
            else ""
        )

        if self.stochastic <= 0.001:
            step_seed = (self.base_seed + stage.global_stage * 1009) % 2147483647
        else:
            step_seed = (
                self.base_seed
                + stage.global_stage * 1009
                + int(self.block_idx * 53)
            ) % 2147483647
        spin_seed = self.base_seed + stage.global_stage

        h_lat = self.height // self.backend.spatial_ratio
        w_lat = self.width // self.backend.spatial_ratio

        if reset_stage_stream or self.local_context_latents_5d is None:
            anchor_latent = None
            is_cont = False
            total_lat_frames = max(3, self.context_lat_frames + self.step_lat_frames)
        else:
            anchor_latent = self.local_context_latents_5d.clone()
            is_cont = True
            total_lat_frames = anchor_latent.shape[2] + self.step_lat_frames

        empty_pair = self.backend.encode_semantics("", max_len=128)
        neg_emb, neg_mask = self.backend.encode_semantics(self.manifest.negative_desc, max_len=128)
        passive_pair = (
            self.backend.encode_semantics(self.manifest.static_hold_desc, max_len=128)
            if self.manifest.static_hold_desc.strip()
            else (
                self.backend.encode_semantics(self.manifest.arena_context_desc, max_len=128)
                if self.manifest.arena_context_desc.strip()
                else empty_pair
            )
        )

        steered_probe_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
            empty_pair for _ in range(n_ents)
        ]
        entity_choice_distributions: list[dict[str, float]] = [
            {} for _ in range(n_ents)
        ]
        dynamic_state_labels: list[str] = ["" for _ in range(n_ents)]
        slot_weights: list[float] = [1.0 for _ in range(n_ents)]

        slot_to_reg_name = {0: "FCz", 1: "AFz", 2: "F3", 3: "F4", 4: "Fpz"}
        active_driven_slots: list[int] = []
        full_wm_5x5x32x120 = torch.zeros((5, 5, 32, 120), device=self.device, dtype=torch.float32)
        active_region_flags = [0.0, 0.0, 0.0, 0.0, 0.0]
        cached_slot_banks: list[tuple[list[str], torch.Tensor, torch.Tensor] | None] = []

        for idx_e in range(n_ents):
            slot = stage.entities[idx_e]
            s_key = slot.slot_id or f"e{idx_e}"
            ctrl = self._get_controller(slot, idx_e, step_gp_channels)
            pool_items = entity_pool_candidates[idx_e]
            cur_st_id = entity_current_states[idx_e]
            reg_ui_name = slot_to_reg_name.get(idx_e, "FCz")
            reg_bus_idx, _, _, _ = ctrl._get_region_specs()
            ctrl.pool_size_hint = max(1, len(pool_items))

            if not pool_items:
                cached_slot_banks.append(None)
                ctrl.sample_working_memory_5x5x32x120(self.device, sub_step_frac=0.0)
                if ctrl.latest_region_wm_5x32x120 is not None:
                    full_wm_5x5x32x120[reg_bus_idx] = ctrl.latest_region_wm_5x32x120
                    if float(ctrl.latest_region_wm_5x32x120.abs().max().item()) > 1e-4:
                        active_region_flags[reg_bus_idx] = 1.0
                steered_probe_pairs[idx_e] = empty_pair
                entity_choice_distributions[idx_e] = {}
                dynamic_state_labels[idx_e] = f"e{idx_e}:0"
                self.active_concept_idx_by_region[reg_ui_name] = -1
                self.active_concept_indices_by_region[reg_ui_name] = []
                self.region_choice_probs[reg_ui_name] = []
                continue

            pool_ids = [item["state_id"] for item in pool_items]
            pool_prompts = [item["probe_prompt"] for item in pool_items]

            enc_pairs = [
                self.backend.encode_semantics(p, max_len=128) for p in pool_prompts
            ]
            bank_K = torch.cat([ep[0] for ep in enc_pairs], dim=0)
            masks_K = torch.cat([ep[1] for ep in enc_pairs], dim=0)
            cached_slot_banks.append((pool_ids, bank_K, masks_K))

            start_id, target_id, steered_emb, steered_mask, probs_dict, active_indices = (
                ctrl.select_concept_full_rank_524288d(
                    pool_ids=pool_ids,
                    candidate_embeds_Kx128x4096=bank_K,
                    device=self.device,
                    current_state_id=cur_st_id,
                    catalyst_field_128x4096=None,
                    candidate_masks_Kx128=masks_K,
                )
            )

            if ctrl.latest_region_wm_5x32x120 is not None:
                full_wm_5x5x32x120[reg_bus_idx] = ctrl.latest_region_wm_5x32x120
                slot_weights[idx_e] = 0.75 + 0.25 * math.tanh(
                    float(torch.norm(ctrl.latest_region_wm_5x32x120).item()) * 0.05
                )

            if not active_indices:
                steered_probe_pairs[idx_e] = (steered_emb.to(self.dtype), steered_mask)
                entity_choice_distributions[idx_e] = probs_dict
                self.active_concept_idx_by_region[reg_ui_name] = -1
                self.active_concept_indices_by_region[reg_ui_name] = []
                self.region_choice_probs[reg_ui_name] = [
                    probs_dict.get(pid, 0.0) for pid in pool_ids
                ]
                dynamic_state_labels[idx_e] = f"e{idx_e}:0"
                continue

            top_idx = active_indices[0]
            steered_probe_pairs[idx_e] = (
                steered_emb.to(self.dtype),
                steered_mask,
            )
            entity_choice_distributions[idx_e] = probs_dict
            self.active_concept_idx_by_region[reg_ui_name] = int(top_idx)
            self.active_concept_indices_by_region[reg_ui_name] = [
                int(i) for i in active_indices
            ]
            self.region_choice_probs[reg_ui_name] = [
                probs_dict.get(pid, 0.0) for pid in pool_ids
            ]

            self.persistent_entity_states[s_key] = pool_ids[top_idx]
            active_driven_slots.append(idx_e)
            active_region_flags[reg_bus_idx] = 1.0
            if idx_e == stage.quest.active_slot_idx and start_id and start_id != target_id:
                dynamic_state_labels[idx_e] = f"e{idx_e}:{start_id}->{target_id}"
            else:
                dynamic_state_labels[idx_e] = f"e{idx_e}:{target_id}"

        pos_emb, pos_mask = self._compose_active_pos_pair(
            active_slots=active_driven_slots,
            steered_probe_pairs=steered_probe_pairs,
            slot_weights=slot_weights,
            full_wm_5x5x32x120=full_wm_5x5x32x120,
            empty_pair=empty_pair,
            passive_pair=passive_pair,
            has_any_pool_items=has_any_pool_items,
        )

        matchup_tag = " | ".join(dynamic_state_labels)
        walk_in_emb, walk_in_mask = pos_emb, pos_mask

        def _live_wm_step_fn(step_i: int, total_steps: int):
            if self.stochastic >= 0.999:
                return None
            frac = float(step_i) / float(max(1, total_steps))
            live_gp = self._peek_live_gamepad_channels()
            live_wm_5x5x32x120 = torch.zeros((5, 5, 32, 120), device=self.device, dtype=torch.float32)
            live_flags = [0.0, 0.0, 0.0, 0.0, 0.0]
            live_active_slots: list[int] = []
            live_pairs = list(steered_probe_pairs)
            live_weights = list(slot_weights)

            for idx_e in range(n_ents):
                slot_e = stage.entities[idx_e]
                ctrl_e = self._get_controller(slot_e, idx_e, live_gp)
                r_bus_idx, _, _, _ = ctrl_e._get_region_specs()
                bank_entry = cached_slot_banks[idx_e]
                if bank_entry is None:
                    ctrl_e.sample_working_memory_5x5x32x120(self.device, sub_step_frac=frac)
                    if ctrl_e.latest_region_wm_5x32x120 is not None:
                        live_wm_5x5x32x120[r_bus_idx] = ctrl_e.latest_region_wm_5x32x120
                        if float(ctrl_e.latest_region_wm_5x32x120.abs().max().item()) > 1e-4:
                            live_flags[r_bus_idx] = 1.0
                    continue

                p_ids, b_K, m_K = bank_entry
                _, _, st_emb, st_msk, _, act_idxs = ctrl_e.select_concept_full_rank_524288d(
                    pool_ids=p_ids,
                    candidate_embeds_Kx128x4096=b_K,
                    device=self.device,
                    current_state_id=entity_current_states[idx_e],
                    catalyst_field_128x4096=None,
                    candidate_masks_Kx128=m_K,
                )
                if ctrl_e.latest_region_wm_5x32x120 is not None:
                    live_wm_5x5x32x120[r_bus_idx] = ctrl_e.latest_region_wm_5x32x120
                    live_weights[idx_e] = 0.75 + 0.25 * math.tanh(
                        float(torch.norm(ctrl_e.latest_region_wm_5x32x120).item()) * 0.05
                    )
                if act_idxs:
                    live_active_slots.append(idx_e)
                    live_flags[r_bus_idx] = 1.0
                    live_pairs[idx_e] = (st_emb.to(self.dtype), st_msk)

            fresh_e, fresh_m = self._compose_active_pos_pair(
                active_slots=live_active_slots,
                steered_probe_pairs=live_pairs,
                slot_weights=live_weights,
                full_wm_5x5x32x120=live_wm_5x5x32x120,
                empty_pair=empty_pair,
                passive_pair=passive_pair,
                has_any_pool_items=has_any_pool_items,
            )
            return fresh_e, fresh_m, live_wm_5x5x32x120, live_flags

        gen = torch.Generator(device=self.device).manual_seed(step_seed)
        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        self.stepper.steps = self.steps
        self.stepper.cfg_scale = self.cfg_scale

        unpacked_z0, dom_ratio = self.stepper.run_window(
            anchor_latent_0=anchor_latent,
            genesis_ref_latent_5d=self.local_genesis_5d,
            walk_in_emb=walk_in_emb,
            walk_in_mask=walk_in_mask,
            pos_emb=pos_emb,
            pos_mask=pos_mask,
            neg_emb=neg_emb,
            neg_mask=neg_mask,
            entity_probes=steered_probe_pairs,
            passive_emb=passive_pair[0],
            passive_mask=passive_pair[1],
            total_lat_frames=total_lat_frames,
            h_lat=h_lat,
            w_lat=w_lat,
            gen=gen,
            is_continuation_block=is_cont,
            stochastic=self.stochastic,
            wm_radius=self.wm_radius,
            base_seed=spin_seed,
            wm_5x5x32x120=full_wm_5x5x32x120,
            active_region_flags=active_region_flags,
            live_wm_step_fn=_live_wm_step_fn,
        )

        if self.stepper.last_velocity_5d is not None:
            for idx_s, slot_s in enumerate(stage.entities):
                ctrl_s = self._get_controller(slot_s, idx_s, step_gp_channels)
                ctrl_s.inject_dit_velocity_feedback(self.stepper.last_velocity_5d)

        tsf = int(self.backend.temporal_ratio)
        # Чистый decode (decode_noise_scale=0.0) предотвращает накопление мыла за 300+ шагов:
        if not is_cont:
            self.local_genesis_5d = unpacked_z0[:, :, :1].detach().clone()
            clean_video_frames = self.backend.decode_latents_chunk(
                unpacked_z0,
                generator=gen,
                decode_t=0.05,
                decode_noise_scale=0.0,
                last_frame_fix=True,
            )
        else:
            decoded_chunk = self.backend.decode_latents_chunk(
                unpacked_z0,
                generator=gen,
                decode_t=0.05,
                decode_noise_scale=0.0,
                last_frame_fix=True,
            )
            n_new_px = self.step_lat_frames * tsf
            clean_video_frames = (
                decoded_chunk[-n_new_px:]
                if len(decoded_chunk) > n_new_px
                else decoded_chunk[1:]
            )

        if clean_video_frames:
            for fr in clean_video_frames:
                self.recent_clean_frames_rgb.append(fr)

            try:
                frame0_lat = self.backend.encode_frame_to_latent(clean_video_frames[-1])
            except Exception:
                frame0_lat = unpacked_z0[:, :, -1:].detach().clone()

            if self.context_lat_frames <= 1:
                self.local_context_latents_5d = frame0_lat
            else:
                # Кадр 0 — чистая causal-проекция последнего кадра, далее чистый латентный импульс:
                extra_tail = unpacked_z0[:, :, -(self.context_lat_frames - 1) :].detach().clone()
                self.local_context_latents_5d = torch.cat([frame0_lat, extra_tail], dim=2)
        else:
            self.local_context_latents_5d = unpacked_z0[:, :, -self.context_lat_frames :].detach().clone()

        if self.monochrome:
            clean_video_frames = [
                cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                for f in clean_video_frames
            ]

        dur_s = len(clean_video_frames) / 16.0

        if not has_catalytic_oracle:
            self.maneuver_phase += 1
            traj = [
                LatentIdentityFrameTelemetry(
                    frame_idx=f_i,
                    time_sec=round(f_i / 16.0, 2),
                    phase_label="PURE_WM_STREAM",
                    primary_coherence_pct=100.0,
                    quest_progress_pct=0.0,
                    quest_remaining_pct=100.0,
                    prob_completed_pct=50.0,
                    prob_ongoing_pct=50.0,
                    semantic_flux_pct=0.0,
                    matchup_tag=matchup_tag,
                    feigenbaum_r=R_EDGE_OF_CHAOS,
                    catalytic_resonance_pct=100.0,
                    chaos_regime="PURE_WM_FLOW",
                    entity_coherences=[100.0] * n_ents,
                    entity_works=[0.0] * n_ents,
                    entity_transition_pcts=[0.0] * n_ents,
                    entity_state_labels=dynamic_state_labels,
                    entity_choice_distributions=entity_choice_distributions,
                    reaction_branch_scores={},
                )
                for f_i in range(max(1, len(clean_video_frames)))
            ]
            decision_log = f"PURE_WM_STREAM #{self.maneuver_phase} | {matchup_tag}"
            high_dim_ctx_dict = {
                "reaction_branch_scores": {},
                "feigenbaum_r": R_EDGE_OF_CHAOS,
                "catalytic_resonance_pct": 100.0,
                "chaos_Window_regime": "PURE_WM_FLOW",
                "prob_completed_peak": 50.0,
                "prob_ongoing_mean": 50.0,
                "prediction_error_mse": 0.0,
                "rf_t_stat": 0.0,
                "entity_coherence_end": [100.0] * n_ents,
                "entity_work_done": [0.0] * n_ents,
                "transformer_physics_explanation": "SPECTRAL_96000D_S3 + KENT_T5_MANIFOLD",
                "entity_state_labels": dynamic_state_labels,
                "entity_choice_distributions": entity_choice_distributions,
            }
        else:
            ongoing_pair = self.backend.encode_semantics(ongoing_hyp, max_len=128)
            completed_pair = self.backend.encode_semantics(completed_hyp, max_len=128)
            failed_pair = (
                self.backend.encode_semantics(failed_hyp, max_len=128)
                if failed_hyp
                else None
            )
            n_cond_for_ref = (
                anchor_latent.shape[2]
                if (is_cont and anchor_latent is not None)
                else 0
            )

            traj, high_dim_ctx = self.referee.evaluate_generative_quest_and_entities(
                unpacked_latents_5d=unpacked_z0,
                n_cond_frames=n_cond_for_ref,
                origin_anchor_5d=(
                    anchor_latent if anchor_latent is not None else self.local_genesis_5d
                ),
                entity_probe_embeds_list=steered_probe_pairs,
                ongoing_hyp_embeds=ongoing_pair,
                completed_hyp_embeds=completed_pair,
                failed_hyp_embeds=failed_pair,
                passive_void_embeds=passive_pair,
                rope_scale=rope_scale,
                total_video_frames=len(clean_video_frames),
                matchup_tag=matchup_tag,
                dom_ratio=dom_ratio,
                gen=gen,
                init_entity_coherences=self.entity_coherences,
                init_quest_progress=self.quest_progress_pct,
                completion_gain=stage.quest.completion_gain,
                entropy_drain=stage.quest.entropy_drain,
                maneuver_phase=self.maneuver_phase,
                active_slot_idx=stage.quest.active_slot_idx,
                catalyst_slot_idx=stage.quest.catalyst_slot_idx,
                driver_tag=stage.quest.driver_tag,
                entity_state_labels=dynamic_state_labels,
                init_entity_transitions=self.entity_transitions,
                entity_choice_distributions=entity_choice_distributions,
                candidate_branch_specs=candidate_branch_specs,
                candidate_branch_hyp_pairs=candidate_branch_hyp_pairs,
                current_stage_idx=self.curr_idx,
                last_velocity_5d=self.stepper.last_velocity_5d,
            )

            win_stg_idx = high_dim_ctx.winning_branch_stage_idx
            win_stage_obj = (
                self.manifest.stages[win_stg_idx]
                if 0 <= win_stg_idx < len(self.manifest.stages)
                else stage
            )
            for idx_s, slot_s in enumerate(stage.entities):
                ctrl_s = self._get_controller(slot_s, idx_s, step_gp_channels)
                if ctrl_s.is_npc_binding():
                    if idx_s == win_stage_obj.quest.catalyst_slot_idx:
                        ctrl_s.observe_oracle_feedback(abs(high_dim_ctx.oracle_advantage) + 0.25)
                    else:
                        ctrl_s.observe_oracle_feedback(-abs(high_dim_ctx.oracle_advantage) - 0.15)

            fin = traj[-1]
            self.entity_coherences = list(fin.entity_coherences)
            self.entity_transitions = list(fin.entity_transition_pcts)
            self.quest_progress_pct = float(fin.quest_progress_pct)

            quest_completed = self.quest_progress_pct >= 95.0
            branch_switched = (
                not quest_completed
                and 0 <= win_stg_idx < len(self.manifest.stages)
                and win_stg_idx != self.curr_idx
                and high_dim_ctx.winning_branch_t_stat > 0.85
                and (high_dim_ctx.winning_branch_t_stat - high_dim_ctx.rf_t_stat) > 0.65
            )

            if branch_switched:
                self.curr_idx = win_stg_idx
                self.maneuver_phase = 1
                self.quest_progress_pct = min(
                    45.0, max(15.0, high_dim_ctx.winning_branch_t_stat * 15.0)
                )
                new_st = self.manifest.stages[self.curr_idx]
                decision_log = (
                    f"BRANCH_SWITCH -> S{new_st.global_stage} [{new_st.quest.driver_tag}] "
                    f"(t={high_dim_ctx.winning_branch_t_stat:+.2f})"
                )
            elif not quest_completed:
                self.maneuver_phase += 1
                decision_log = (
                    f"BLOCK #{self.maneuver_phase} [{stage.quest.driver_tag}] -> "
                    f"{stage.quest.reaction_formula} | dS={self.quest_progress_pct:.1f}% | "
                    f"t={high_dim_ctx.rf_t_stat:+.2f}"
                )
            else:
                self.maneuver_phase = 0
                for idx_s, slot_s in enumerate(stage.entities):
                    s_key = slot_s.slot_id or f"e{idx_s}"
                    if idx_s < len(high_dim_ctx.entity_state_labels):
                        lbl = high_dim_ctx.entity_state_labels[idx_s]
                        if "->" in lbl:
                            self.persistent_entity_states[s_key] = lbl.split("->", 1)[1].strip()
                        elif ":" in lbl:
                            val_s = lbl.split(":", 1)[1].strip()
                            if val_s and val_s != "0":
                                self.persistent_entity_states[s_key] = val_s

                if (
                    0 <= high_dim_ctx.chosen_next_stage_idx < len(self.manifest.stages)
                    and high_dim_ctx.chosen_next_stage_idx != self.curr_idx
                ):
                    next_idx = high_dim_ctx.chosen_next_stage_idx
                else:
                    next_idx = (self.curr_idx + 1) % len(self.manifest.stages)

                next_st = self.manifest.stages[next_idx]
                decision_log = (
                    f"STAGE_DONE [{stage.quest.reaction_formula}] -> Stage {next_st.global_stage}"
                )
                self.curr_idx = next_idx

            high_dim_ctx_dict = {
                "reaction_branch_scores": high_dim_ctx.reaction_branch_scores,
                "feigenbaum_r": high_dim_ctx.feigenbaum_r,
                "catalytic_resonance_pct": high_dim_ctx.catalytic_resonance_pct,
                "chaos_Window_regime": high_dim_ctx.chaos_Window_regime,
                "prob_completed_peak": high_dim_ctx.prob_completed_peak,
                "prob_ongoing_mean": high_dim_ctx.prob_ongoing_mean,
                "prediction_error_mse": high_dim_ctx.prediction_error_mse,
                "rf_t_stat": high_dim_ctx.rf_t_stat,
                "entity_coherence_end": high_dim_ctx.entity_coherence_end,
                "entity_work_done": high_dim_ctx.entity_work_done,
                "transformer_physics_explanation": high_dim_ctx.transformer_physics_explanation,
                "entity_state_labels": high_dim_ctx.entity_state_labels,
                "entity_choice_distributions": high_dim_ctx.entity_choice_distributions,
            }

        t_start = self.total_rendered_frames / 16.0
        self.total_rendered_frames += len(clean_video_frames)
        t_end = self.total_rendered_frames / 16.0
        self.block_idx += 1
        gen_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        fps = len(clean_video_frames) / max(0.001, gen_ms / 1000.0)

        per_entity_choice_debug = {}
        for idx_c, c_dist in enumerate(high_dim_ctx_dict["entity_choice_distributions"]):
            s_id = (
                stage.entities[idx_c].slot_id
                if idx_c < len(stage.entities)
                else f"e{idx_c}"
            )
            if c_dist:
                sorted_c = sorted(c_dist.items(), key=lambda x: x[1], reverse=True)
                per_entity_choice_debug[s_id] = {
                    "chosen": {sorted_c[0][0]: sorted_c[0][1]},
                    "rejected_alternatives": {k: v for k, v in sorted_c[1:]},
                }

        telemetry_record = {
            "step": self.block_idx,
            "t": f"{t_start:.1f}-{t_end:.1f}",
            "duration_sec": round(dur_s, 2),
            "global_stage": stage.global_stage,
            "stage_title": stage.stage_title,
            "reaction_formula": stage.quest.reaction_formula,
            "driver_tag": stage.quest.driver_tag,
            "active_entities": matchup_tag,
            "entity_state_transitions_pct": dict(
                zip(high_dim_ctx_dict["entity_state_labels"], self.entity_transitions)
            ),
            "wm_524288d_choice_debug": per_entity_choice_debug,
            "active_concept_idx": dict(self.active_concept_idx_by_region),
            "active_concept_indices": dict(self.active_concept_indices_by_region),
            "region_choice_probs": dict(self.region_choice_probs),
            "concept_bindings": dict(self.concept_bindings_by_region),
            "eeg_wm_gain": round(self.eeg_wm_gain, 2),
            "gamepad_wm_mix": round(self.gamepad_wm_mix, 2),
            "reaction_branch_competition_128d": high_dim_ctx_dict["reaction_branch_scores"],
            "latent_dominance_ratio": round(dom_ratio, 3),
            "feigenbaum_chaos_metrics": {
                "feigenbaum_r": round(high_dim_ctx_dict["feigenbaum_r"], 4),
                "feigenbaum_delta_const": 4.669201609,
                "catalytic_resonance_pct": round(high_dim_ctx_dict["catalytic_resonance_pct"], 1),
                "chaos_regime": high_dim_ctx_dict["chaos_Window_regime"],
            },
            "quest_oracle_metrics": {
                "quest_progress_pct": round(self.quest_progress_pct, 1),
                "prob_completed_peak_pct": round(high_dim_ctx_dict["prob_completed_peak"], 1),
                "prob_ongoing_mean_pct": round(high_dim_ctx_dict["prob_ongoing_mean"], 1),
                "prediction_error_mse": round(high_dim_ctx_dict["prediction_error_mse"], 5),
                "rf_t_stat": round(high_dim_ctx_dict["rf_t_stat"], 3),
                "entity_coherence_end": high_dim_ctx_dict["entity_coherence_end"],
                "entity_work_done": high_dim_ctx_dict["entity_work_done"],
            },
            "transformer_physics_explanation": high_dim_ctx_dict["transformer_physics_explanation"],
            "decision": decision_log,
            "prompt": matchup_tag,
        }

        if self.block_idx % 20 == 0:
            gc.collect()

        info_tag = f"[S{stage.global_stage} {stage.quest.driver_tag} | {dur_s:.1f}s] {decision_log}"
        return clean_video_frames, traj, gen_ms, fps, info_tag, telemetry_record
