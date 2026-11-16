#!/usr/bin/env python3
"""
neuro_engine.py — Гетерархический движок N агентов с On-Manifold проекцией через Denoising VAE
==============================================================================================
- Выполняет честную On-Manifold проекцию между окнами (Shared Diffusion Objective из статьи LTX-Video):
  финальный шаг денойзинга выполняется декодером 3D-VAE (decode_t = 0.05, decode_noise_scale = 0.0),
  после чего чистый последний кадр за ~10 мс кодируется обратно через `encode_frame_to_latent` и
  выравнивается по дисперсии с генезис-кадром (`adain_stabilize_latents`).
- Благодаря этому и исправленному знаку `dt < 0` в `LatentFlowWindowStepper` видео остаётся
  кристально четким и динамичным часами при любом `wm_radius` (включая 0.84)!
"""

from __future__ import annotations
import os
import multiprocessing as mp
from multiprocessing.connection import Client
import time
import gc
import cv2
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
mp.current_process().authkey = b"battle_ltx"

from neuro_world import GameWorldManifest, StageEntitySlot, compose_entity_semantics
from neuro_controller import SymmetricBotController
from neuro_ltx_backend import LTXModelBackend, RemoteLTXModelBackend
from neuro_referee import CoordinateFreeHistoryReferee, LatentIdentityFrameTelemetry
from neuro_solver_stepper import (
    LatentFlowWindowStepper,
    adain_stabilize_latents,
    imprint_allegiance_spin,
)


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
        steps: int = 6,
        cfg_scale: float = 3.0,
        seed: int = 42,
        battle_server_port: int = 6008,
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
        self.battle_server_port = int(battle_server_port)
        self.force_local = bool(force_local)

        self.context_lat_frames = max(1, int(context_lat_frames))
        self.step_lat_frames = max(1, int(step_lat_frames))
        self.stochastic = float(np.clip(stochastic, 0.0, 1.0))
        self.wm_radius = float(np.clip(wm_radius, 0.0, 1.0))

        self.curr_idx = 0
        self.block_idx = 0
        self.total_rendered_frames = 0
        self.maneuver_phase: int = 0
        self.quest_progress_pct: float = 0.0
        self.need_hard_genesis: bool = True

        self.active_entity_parties: list[list[str]] = []
        self.persistent_entity_states: dict[str, str] = {}
        self.entity_coherences: list[float] = []
        self.entity_transitions: list[float] = []

        self.entity_controllers: dict[str, SymmetricBotController] = {}
        self.source_mode_overrides: dict[str, str] = {}

        self.local_genesis_5d: torch.Tensor | None = None
        self.local_context_latents_5d: torch.Tensor | None = None

        self.backend = None
        if not self.force_local and self.battle_server_port > 0:
            try:
                conn = Client(
                    ("localhost", self.battle_server_port), authkey=b"battle_ltx"
                )
                conn.send({"cmd": "ping"})
                ping_resp = conn.recv()
                if isinstance(ping_resp, dict) and ping_resp.get("status") == "ok":
                    self.backend = RemoteLTXModelBackend(
                        conn=conn, ping_info=ping_resp
                    )
                    print(
                        f"⚡ [ENGINE] Подключено к серверу весов (localhost:{self.battle_server_port}). "
                        f"Режим: {self.width}x{self.height}, steps={self.steps}, cfg={self.cfg_scale:.1f}, "
                        f"stoch={self.stochastic:.2f}, wm_rad={self.wm_radius:.2f}, "
                        f"ctx_lat={self.context_lat_frames}, step_lat={self.step_lat_frames}"
                    )
            except Exception:
                self.backend = None

        if self.backend is None:
            print("ℹ️ [ENGINE] Запуск локального LTX-Video в VRAM...")
            self.backend = LTXModelBackend()

        self.device = self.backend.device
        self.dtype = self.backend.dtype
        self.referee = CoordinateFreeHistoryReferee(
            self.backend.transformer, self.device, self.dtype
        )
        self.stepper = LatentFlowWindowStepper(
            backend=self.backend,
            referee=self.referee,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            guidance_rescale=0.7,
        )

        self._warmup_manifest_semantics()

    @torch.no_grad()
    def _warmup_manifest_semantics(self):
        t0 = time.perf_counter()
        prompts_to_warm = set()
        prompts_to_warm.add(self.manifest.negative_desc)
        prompts_to_warm.add(self.manifest.static_hold_desc)

        arch_sentences: dict[str, str] = {}
        for aid, arch in self.manifest.archetypes.items():
            sentence = f"{arch.essence_desc} {arch.action_desc}".strip()
            if sentence and not sentence.endswith("."):
                sentence += "."
            arch_sentences[aid] = sentence
            probe_p = f"{self.manifest.arena_context_desc}. {sentence}".strip()
            prompts_to_warm.add(probe_p)

        for st in self.manifest.stages:
            prompts_to_warm.add(
                f"{self.manifest.arena_context_desc}. {st.quest.ongoing_hypothesis}"
            )
            prompts_to_warm.add(
                f"{self.manifest.arena_context_desc}. {st.quest.completed_hypothesis}"
            )
            if st.quest.failed_hypothesis:
                prompts_to_warm.add(
                    f"{self.manifest.arena_context_desc}. {st.quest.failed_hypothesis}"
                )

            cat_i = st.quest.catalyst_slot_idx
            sub_i = st.quest.active_slot_idx
            if 0 <= cat_i < len(st.entities) and 0 <= sub_i < len(st.entities):
                cat_pool = st.entities[cat_i].allowed_pool or list(
                    arch_sentences.keys()
                )
                sub_pool = st.entities[sub_i].allowed_pool or list(
                    arch_sentences.keys()
                )
                for c_id in cat_pool:
                    for s_id in sub_pool:
                        s_cat = arch_sentences.get(c_id, "")
                        s_sub = arch_sentences.get(s_id, "")
                        non_empty = [x for x in (s_cat, s_sub) if x]
                        scene_p = f"{self.manifest.arena_context_desc}." + (
                            " " + " ".join(non_empty) if non_empty else ""
                        )
                        prompts_to_warm.add(scene_p)

        print(
            f"🔥 [T5 WARMUP] Предвычисление {len(prompts_to_warm)} полных сцен и зондов T5...",
            flush=True,
        )
        for p in prompts_to_warm:
            self.backend.encode_semantics(p, max_len=128)
        dt = time.perf_counter() - t0
        print(
            f"✅ [T5 WARMUP] Готово за {dt:.1f}с! T5 остаётся резидентным в VRAM для мгновенной смены концептов!",
            flush=True,
        )

    def set_entity_source(self, slot_id: str, source_mode: str):
        self.source_mode_overrides[slot_id] = source_mode
        if slot_id in self.entity_controllers:
            self.entity_controllers[slot_id].source_mode = source_mode

    def set_all_entities_source(self, source_mode: str):
        self.source_mode_overrides["*"] = source_mode
        for ctrl in self.entity_controllers.values():
            ctrl.source_mode = source_mode

    def _get_effective_binding(self, slot: StageEntitySlot, idx: int) -> str:
        slot_key = slot.slot_id or f"e{idx}"
        return (
            self.source_mode_overrides.get(slot_key)
            or self.source_mode_overrides.get("*")
            or slot.control_binding
        )

    def _get_controller(
        self, slot: StageEntitySlot, idx: int
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
        ctrl.attempt_num = self.block_idx + 1
        return ctrl

    def _build_pool_candidates_payload(
        self, stage
    ) -> tuple[list[list[dict]], list[str]]:
        pool_payload: list[list[dict]] = []
        curr_states: list[str] = []
        for idx_s, slot in enumerate(stage.entities):
            s_key = slot.slot_id or f"e{idx_s}"
            pool_ids = (
                slot.allowed_pool
                if slot.allowed_pool
                else list(self.manifest.archetypes.keys())
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
                            "scene_sentence": sentence,
                        }
                    )
            pool_payload.append(slot_candidates)
        return pool_payload, curr_states

    def _build_candidate_branches_payload(self) -> list[dict]:
        branches = []
        for s_idx, st in enumerate(self.manifest.stages):
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
        return branches

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
        stage = self.manifest.stages[self.curr_idx]
        n_ents = max(1, len(stage.entities))

        if self.maneuver_phase == 0 or not self.active_entity_parties:
            self.active_entity_parties = []
            for idx_s, slot in enumerate(stage.entities):
                ctrl = self._get_controller(slot, idx_s)
                party = ctrl.select_party(stage.global_stage, slot)
                self.active_entity_parties.append(party)
            self.entity_coherences = [100.0] * n_ents
            self.entity_transitions = [
                0.0 if i == stage.quest.active_slot_idx else 100.0
                for i in range(n_ents)
            ]
            self.quest_progress_pct = 0.0

        reset_stage_stream = self.need_hard_genesis
        self.need_hard_genesis = False

        _, walk_in_prompt, _, _, _ = compose_entity_semantics(
            self.active_entity_parties, self.manifest, stage=stage
        )

        entity_pool_candidates, entity_current_states = (
            self._build_pool_candidates_payload(stage)
        )
        candidate_branch_specs = self._build_candidate_branches_payload()

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
                + int(self.block_idx * 53 * self.stochastic)
            ) % 2147483647
        spin_seed = self.base_seed + stage.global_stage

        h_lat = self.height // self.backend.spatial_ratio
        w_lat = self.width // self.backend.spatial_ratio

        if reset_stage_stream or self.local_context_latents_5d is None:
            anchor_latent = None
            is_cont = False
            total_lat_frames = max(3, self.context_lat_frames + self.step_lat_frames)
        else:
            anchor_latent = self.local_context_latents_5d[
                :, :, -self.context_lat_frames :
            ].clone()
            is_cont = True
            total_lat_frames = anchor_latent.shape[2] + self.step_lat_frames

        steered_probe_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [None] * n_ents  # type: ignore
        entity_choice_distributions: list[dict[str, float]] = [
            {} for _ in range(n_ents)
        ]
        dynamic_state_labels: list[str] = ["" for _ in range(n_ents)]
        chosen_sentences: list[str] = []

        catalyst_field_128x4096: torch.Tensor | None = None
        eval_order = [
            i for i in range(n_ents) if i != stage.quest.active_slot_idx
        ] + (
            [stage.quest.active_slot_idx]
            if 0 <= stage.quest.active_slot_idx < n_ents
            else []
        )

        for idx_e in eval_order:
            slot = stage.entities[idx_e]
            ctrl = self._get_controller(slot, idx_e)
            pool_items = entity_pool_candidates[idx_e]
            cur_st_id = entity_current_states[idx_e]
            pool_ids = [item["state_id"] for item in pool_items]
            pool_prompts = [item["probe_prompt"] for item in pool_items]
            pool_sents = {
                item["state_id"]: item["scene_sentence"] for item in pool_items
            }

            enc_pairs = [
                self.backend.encode_semantics(p, max_len=128) for p in pool_prompts
            ]
            bank_K = torch.cat([ep[0] for ep in enc_pairs], dim=0)
            masks_K = [ep[1] for ep in enc_pairs]

            ext_cat = (
                catalyst_field_128x4096
                if idx_e == stage.quest.active_slot_idx
                else None
            )

            start_id, target_id, steered_emb, probs_dict = (
                ctrl.select_concept_full_rank_524288d(
                    pool_ids=pool_ids,
                    candidate_embeds_Kx128x4096=bank_K,
                    device=self.device,
                    current_state_id=cur_st_id,
                    catalyst_field_128x4096=ext_cat,
                )
            )
            t_idx = pool_ids.index(target_id) if target_id in pool_ids else 0
            steered_probe_pairs[idx_e] = (
                steered_emb.to(self.dtype),
                masks_K[t_idx],
            )
            entity_choice_distributions[idx_e] = probs_dict

            if idx_e == stage.quest.catalyst_slot_idx:
                catalyst_field_128x4096 = steered_emb[0].detach().float()
                dynamic_state_labels[idx_e] = f"e{idx_e}:{target_id}[CAT]"
                chosen_sentences.insert(0, pool_sents.get(target_id, ""))
            elif idx_e == stage.quest.active_slot_idx:
                dynamic_state_labels[idx_e] = f"e{idx_e}:{start_id}->{target_id}"
                chosen_sentences.append(pool_sents.get(target_id, ""))
            else:
                dynamic_state_labels[idx_e] = f"e{idx_e}:{target_id}[HOLD]"

        non_empty = [s for s in chosen_sentences if s]
        if self.manifest.arena_context_desc:
            scene_prompt = f"{self.manifest.arena_context_desc}." + (
                " " + " ".join(non_empty) if non_empty else ""
            )
        else:
            scene_prompt = " ".join(non_empty) if non_empty else "abstract visual space"
        matchup_tag = " | ".join(dynamic_state_labels)

        pos_emb, pos_mask = self.backend.encode_semantics(scene_prompt, max_len=128)
        walk_in_emb, walk_in_mask = pos_emb, pos_mask
        neg_emb, neg_mask = self.backend.encode_semantics(
            self.manifest.negative_desc, max_len=128
        )
        passive_pair = self.backend.encode_semantics(
            self.manifest.static_hold_desc, max_len=128
        )
        ongoing_pair = self.backend.encode_semantics(ongoing_hyp, max_len=128)
        completed_pair = self.backend.encode_semantics(completed_hyp, max_len=128)
        failed_pair = (
            self.backend.encode_semantics(failed_hyp, max_len=128)
            if failed_hyp
            else None
        )

        drv_idx = (
            stage.quest.catalyst_slot_idx
            if 0 <= stage.quest.catalyst_slot_idx < len(stage.entities)
            else stage.quest.active_slot_idx
        )
        drv_ctrl = self._get_controller(stage.entities[drv_idx], drv_idx)
        steered_pos_emb = drv_ctrl.steer_semantic_probe_full_rank(
            pos_emb, device=self.device, sub_step_frac=0.0
        ).to(self.dtype)

        def _live_pos_emb_fn(step_i: int, total_steps: int) -> torch.Tensor:
            frac = float(step_i) / float(max(1, total_steps))
            return drv_ctrl.steer_semantic_probe_full_rank(
                pos_emb, device=self.device, sub_step_frac=frac
            ).to(self.dtype)

        sub_ctrl = self._get_controller(
            stage.entities[stage.quest.active_slot_idx], stage.quest.active_slot_idx
        )
        wm_cat_vec = drv_ctrl.latest_region_wm_19200
        wm_sub_vec = sub_ctrl.latest_region_wm_19200
        if wm_cat_vec is not None and wm_sub_vec is not None:
            combined_wm_19200 = wm_cat_vec * 0.55 + wm_sub_vec * 0.45
        else:
            combined_wm_19200 = (
                wm_cat_vec if wm_cat_vec is not None else wm_sub_vec
            )

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
            pos_emb=steered_pos_emb,
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
            wm_19200_vec=combined_wm_19200,
            live_pos_emb_fn=_live_pos_emb_fn,
        )

        if not is_cont:
            unpacked_z0 = imprint_allegiance_spin(unpacked_z0, base_seed=spin_seed)
            self.local_genesis_5d = unpacked_z0[:, :, :1].detach().clone()
            clean_video_frames = self.backend.decode_latents_chunk(
                unpacked_z0, generator=gen, decode_t=0.05, decode_noise_scale=0.0
            )
            eval_lat = unpacked_z0
        else:
            n_vae_ctx = min(1, unpacked_z0.shape[2] - self.step_lat_frames)
            decode_window_lat = n_vae_ctx + self.step_lat_frames
            decoded_chunk = self.backend.decode_latents_chunk(
                unpacked_z0[:, :, -decode_window_lat:],
                generator=gen,
                decode_t=0.05,
                decode_noise_scale=0.0,
            )
            n_new_px = self.step_lat_frames * int(self.backend.temporal_ratio)
            clean_video_frames = (
                decoded_chunk[-n_new_px:]
                if len(decoded_chunk) > n_new_px
                else decoded_chunk[1:]
            )
            eval_lat = unpacked_z0[:, :, -self.step_lat_frames :]

        # ON-MANIFOLD VAE PROJECTION (HaCohen et al., 2024):
        # Кодируем последний чистый пиксельный кадр (прошедший финальный шаг денойзинга в VAE Decoder)
        # обратно в латент для следующего окна и стабилизируем через AdaIN к генезис-кадру!
        if clean_video_frames:
            try:
                reencoded_last_5d = self.backend.encode_frame_to_latent(
                    clean_video_frames[-1]
                )
                if self.local_genesis_5d is not None:
                    reencoded_last_5d = adain_stabilize_latents(
                        reencoded_last_5d, self.local_genesis_5d, factor=0.15
                    )
                if self.context_lat_frames <= 1:
                    self.local_context_latents_5d = reencoded_last_5d
                else:
                    prev_ctx = unpacked_z0[
                        :, :, -self.context_lat_frames : -1
                    ].detach()
                    self.local_context_latents_5d = torch.cat(
                        [prev_ctx, reencoded_last_5d], dim=2
                    )
            except Exception:
                self.local_context_latents_5d = (
                    unpacked_z0[:, :, -self.context_lat_frames :].detach().clone()
                )
        else:
            self.local_context_latents_5d = (
                unpacked_z0[:, :, -self.context_lat_frames :].detach().clone()
            )

        if self.monochrome:
            clean_video_frames = [
                cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                for f in clean_video_frames
            ]

        traj, high_dim_ctx = self.referee.evaluate_generative_quest_and_entities(
            unpacked_latents_5d=eval_lat,
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
            last_velocity_5d=self.stepper.last_velocity_5d,
        )

        fin = traj[-1]
        self.entity_coherences = list(fin.entity_coherences)
        self.entity_transitions = list(fin.entity_transition_pcts)
        self.quest_progress_pct = float(fin.quest_progress_pct)

        dur_s = len(clean_video_frames) / 16.0
        quest_completed = self.quest_progress_pct >= 95.0
        act_coh = (
            self.entity_coherences[stage.quest.active_slot_idx]
            if stage.quest.active_slot_idx < len(self.entity_coherences)
            else 100.0
        )
        quest_failed = act_coh <= 5.0

        if not quest_completed and not quest_failed:
            self.maneuver_phase += 1
            decision_log = (
                f"BLOCK #{self.maneuver_phase} [{stage.quest.driver_tag} | r={high_dim_ctx.feigenbaum_r:.3f}] -> "
                f"{stage.quest.reaction_formula} | dS={self.quest_progress_pct:.1f}% | "
                f"Branches={high_dim_ctx.reaction_branch_scores}"
            )
        else:
            self.maneuver_phase = 0
            if quest_completed:
                for idx_s, slot_s in enumerate(stage.entities):
                    s_key = slot_s.slot_id or f"e{idx_s}"
                    if idx_s < len(high_dim_ctx.entity_state_labels):
                        lbl = high_dim_ctx.entity_state_labels[idx_s]
                        if "->" in lbl:
                            self.persistent_entity_states[s_key] = lbl.split(
                                "->", 1
                            )[1].strip()
                        elif ":" in lbl and "[" in lbl:
                            self.persistent_entity_states[s_key] = (
                                lbl.split(":", 1)[1].split("[", 1)[0].strip()
                            )

                if 0 <= high_dim_ctx.chosen_next_stage_idx < len(self.manifest.stages):
                    next_idx = high_dim_ctx.chosen_next_stage_idx
                else:
                    next_idx = (self.curr_idx + 1) % len(self.manifest.stages)

                next_st = self.manifest.stages[next_idx]
                decision_log = (
                    f"CHOICE COMMITTED: [{stage.quest.reaction_formula}] (100%)! "
                    f"128D Branch Winner -> Stage {next_st.global_stage} ({next_st.quest.driver_tag}) "
                    f"from {high_dim_ctx.reaction_branch_scores}!"
                )
                self.curr_idx = next_idx
            else:
                self.need_hard_genesis = True
                decision_log = (
                    f"REACTION RESET: Progress={self.quest_progress_pct:.1f}%"
                )

        t_start = self.total_rendered_frames / 16.0
        self.total_rendered_frames += len(clean_video_frames)
        t_end = self.total_rendered_frames / 16.0
        self.block_idx += 1
        gen_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        fps = len(clean_video_frames) / max(0.001, gen_ms / 1000.0)

        per_entity_choice_debug = {}
        for idx_c, c_dist in enumerate(high_dim_ctx.entity_choice_distributions):
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
                zip(high_dim_ctx.entity_state_labels, self.entity_transitions)
            ),
            "wm_524288d_choice_debug": per_entity_choice_debug,
            "reaction_branch_competition_128d": high_dim_ctx.reaction_branch_scores,
            "latent_dominance_ratio": round(dom_ratio, 3),
            "feigenbaum_chaos_metrics": {
                "feigenbaum_r": round(high_dim_ctx.feigenbaum_r, 4),
                "feigenbaum_delta_const": 4.669201609,
                "catalytic_resonance_pct": round(
                    high_dim_ctx.catalytic_resonance_pct, 1
                ),
                "chaos_regime": high_dim_ctx.chaos_Window_regime,
            },
            "quest_oracle_metrics": {
                "quest_progress_pct": round(self.quest_progress_pct, 1),
                "prob_completed_peak_pct": round(
                    high_dim_ctx.prob_completed_peak, 1
                ),
                "prob_ongoing_mean_pct": round(high_dim_ctx.prob_ongoing_mean, 1),
                "entity_coherence_end": high_dim_ctx.entity_coherence_end,
                "entity_work_done": high_dim_ctx.entity_work_done,
            },
            "transformer_physics_explanation": high_dim_ctx.transformer_physics_explanation,
            "decision": decision_log,
            "prompt": scene_prompt,
        }

        if self.block_idx % 20 == 0:
            gc.collect()

        info_tag = f"[S{stage.global_stage} {stage.quest.driver_tag} | {dur_s:.1f}s] {decision_log}"
        return clean_video_frames, traj, gen_ms, fps, info_tag, telemetry_record
