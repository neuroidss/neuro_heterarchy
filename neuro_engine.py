#!/usr/bin/env python3
"""
neuro_engine.py — Гетерархический движок N агентов рабочей памяти с полноранговым выбором ветвей
================================================================================================
- Никаких рельсов (curr_idx + 1): следующая стадия выбирается динамически на основе
  128D-резонанса всех конкурирующих каталитических ветвей (chosen_next_stage_idx).
- Каждый агент отслеживает своё текущее состояние в persistent_entity_states и выбирает
  следующее целевое состояние из своего allowed_pool через 524,288-D рабочую память.
"""

from __future__ import annotations
import os
import multiprocessing as mp
from multiprocessing.connection import Client
import time
import gc
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
mp.current_process().authkey = b"battle_ltx"

from neuro_world import GameWorldManifest, StageEntitySlot, compose_entity_semantics
from neuro_controller import SymmetricBotController
from neuro_clash_hud import draw_identity_hud


class UniversalLatentAutobattlerEngine:

    def __init__(
        self,
        manifest: GameWorldManifest,
        width: int = 640,
        height: int = 448,
        prep_seconds: float = 2.5,
        combat_seconds: float = 1.5,
        game_mode: str = "strict_ko",
        max_overtime_chunks: int = 12,
        steps: int = 20,
        cfg_scale: float = 4.0,
        seed: int = 42,
        battle_server_port: int = 6008,
        force_local: bool = False,
        monochrome: bool = False,
    ):
        self.manifest = manifest
        self.width = (width // 32) * 32
        self.height = (height // 32) * 32
        self.monochrome = bool(monochrome)

        self.game_mode = game_mode
        self.steps = max(12, int(steps))
        self.cfg_scale = float(cfg_scale)
        self.base_seed = int(seed)
        self.battle_server_port = int(battle_server_port)
        self.force_local = bool(force_local)

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
        self.local_last_clean_latent_1frame: torch.Tensor | None = None

        self.server_conn = None
        self.backend = None
        self.referee = None
        self.stepper = None

        if not self.force_local and self.battle_server_port > 0:
            try:
                conn = Client(("localhost", self.battle_server_port), authkey=b"battle_ltx")
                conn.send({"cmd": "ping"})
                ping_resp = conn.recv()
                if isinstance(ping_resp, dict) and ping_resp.get("status") == "ok":
                    self.server_conn = conn
                    print(f"⚡ [ENGINE] Подключено к neuro_battle_server (localhost:{self.battle_server_port}).")
            except Exception:
                self.server_conn = None

        if self.server_conn is None:
            print("ℹ️ [ENGINE] Запуск локального LTX-Video в VRAM...")
            from neuro_ltx_backend import LTXModelBackend
            from neuro_referee import CoordinateFreeHistoryReferee
            from neuro_solver_stepper import LatentFlowWindowStepper

            self.backend = LTXModelBackend()
            self.device = self.backend.device
            self.dtype = self.backend.dtype
            self.referee = CoordinateFreeHistoryReferee(self.backend.transformer, self.device, self.dtype)
            self.stepper = LatentFlowWindowStepper(
                backend=self.backend,
                referee=self.referee,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
                guidance_rescale=0.7,
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

    def _get_controller(self, slot: StageEntitySlot, idx: int) -> SymmetricBotController:
        slot_key = slot.slot_id or f"e{idx}"
        binding = self._get_effective_binding(slot, idx)
        if slot_key not in self.entity_controllers:
            self.entity_controllers[slot_key] = SymmetricBotController(
                slot_key, self.manifest.archetypes, source_mode=binding
            )
        ctrl = self.entity_controllers[slot_key]
        ctrl.source_mode = binding
        ctrl.attempt_num = self.block_idx + 1
        return ctrl

    def _build_pool_candidates_payload(self, stage) -> tuple[list[list[dict]], list[str]]:
        pool_payload: list[list[dict]] = []
        curr_states: list[str] = []
        for idx_s, slot in enumerate(stage.entities):
            s_key = slot.slot_id or f"e{idx_s}"
            pool_ids = slot.allowed_pool if slot.allowed_pool else list(self.manifest.archetypes.keys())
            if s_key not in self.persistent_entity_states and pool_ids:
                self.persistent_entity_states[s_key] = pool_ids[0]
            curr_states.append(self.persistent_entity_states.get(s_key, pool_ids[0] if pool_ids else ""))

            slot_candidates = []
            for aid in pool_ids:
                arch = self.manifest.archetypes.get(aid)
                if arch:
                    slot_candidates.append({
                        "state_id": aid,
                        "display_name": arch.display_name,
                        "probe_prompt": f"{self.manifest.arena_context_desc}. {arch.essence_desc} {arch.action_desc}.",
                        "scene_sentence": f"{arch.essence_desc} {arch.action_desc}.",
                    })
            pool_payload.append(slot_candidates)
        return pool_payload, curr_states

    def _build_candidate_branches_payload(self) -> list[dict]:
        branches = []
        for s_idx, st in enumerate(self.manifest.stages):
            branches.append({
                "stage_idx": s_idx,
                "global_stage": st.global_stage,
                "driver_tag": st.quest.driver_tag,
                "reaction_formula": st.quest.reaction_formula,
                "active_slot_idx": st.quest.active_slot_idx,
                "catalyst_slot_idx": st.quest.catalyst_slot_idx,
            })
        return branches

    @torch.no_grad()
    def execute_turn_step(self) -> tuple[list[np.ndarray], float, float, str, dict]:
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
            self.entity_transitions = [0.0 if i == stage.quest.active_slot_idx else 100.0 for i in range(n_ents)]
            self.quest_progress_pct = 0.0

        reset_stage_stream = self.need_hard_genesis
        self.need_hard_genesis = False

        scene_prompt, walk_in_prompt, start_probes, target_probes, matchup_tag = compose_entity_semantics(
            self.active_entity_parties, self.manifest, stage=stage
        )

        entity_bindings = [self._get_effective_binding(slot, idx_s) for idx_s, slot in enumerate(stage.entities)]
        entity_pool_candidates, entity_current_states = self._build_pool_candidates_payload(stage)
        candidate_branch_specs = self._build_candidate_branches_payload()

        ongoing_hyp = f"{self.manifest.arena_context_desc}. {stage.quest.ongoing_hypothesis}"
        completed_hyp = f"{self.manifest.arena_context_desc}. {stage.quest.completed_hypothesis}"
        failed_hyp = (
            f"{self.manifest.arena_context_desc}. {stage.quest.failed_hypothesis}"
            if stage.quest.failed_hypothesis
            else ""
        )

        step_seed = (
            self.base_seed
            + stage.global_stage * 1009
            + self.block_idx * 53
        ) % 2147483647
        spin_seed = self.base_seed + stage.global_stage

        if self.server_conn is not None:
            req = {
                "cmd": "execute_battle_window",
                "global_stage": stage.global_stage,
                "reset_stage_stream": reset_stage_stream,
                "maneuver_phase": self.maneuver_phase,
                "block_idx": self.block_idx,
                "unit_probe_cmds": target_probes,
                "entity_bindings": entity_bindings,
                "entity_pool_candidates": entity_pool_candidates,
                "entity_current_states": entity_current_states,
                "candidate_branch_specs": candidate_branch_specs,
                "arena_context_desc": self.manifest.arena_context_desc,
                "ongoing_hyp": ongoing_hyp,
                "completed_hyp": completed_hyp,
                "failed_hyp": failed_hyp,
                "completion_gain": stage.quest.completion_gain,
                "entropy_drain": stage.quest.entropy_drain,
                "active_slot_idx": stage.quest.active_slot_idx,
                "catalyst_slot_idx": stage.quest.catalyst_slot_idx,
                "driver_tag": stage.quest.driver_tag,
                "init_entity_transitions": self.entity_transitions,
                "init_entity_coherences": self.entity_coherences,
                "init_quest_progress": self.quest_progress_pct,
                "walk_in_prompt": walk_in_prompt,
                "scene_prompt": scene_prompt,
                "negative_desc": self.manifest.negative_desc,
                "static_hold_desc": self.manifest.static_hold_desc,
                "width": self.width,
                "height": self.height,
                "steps": self.steps,
                "cfg_scale": self.cfg_scale,
                "step_seed": step_seed,
                "spin_seed": spin_seed,
                "matchup_tag": matchup_tag,
            }
            self.server_conn.send(req)
            resp = self.server_conn.recv()
            full_video_frames = resp["full_video_frames"]
            dom_ratio = float(resp["dom_ratio"])
            traj = resp["traj"]
            high_dim_ctx = resp["high_dim_ctx"].from_ipc_safe()
            scene_prompt = resp.get("scene_prompt", scene_prompt)
            matchup_tag = resp.get("matchup_tag", matchup_tag)
        else:
            from neuro_solver_stepper import imprint_allegiance_spin

            h_lat = self.height // self.backend.spatial_ratio
            w_lat = self.width // self.backend.spatial_ratio
            total_lat_frames = 5

            if reset_stage_stream or self.local_last_clean_latent_1frame is None:
                anchor_latent = None
                is_cont = False
            else:
                anchor_latent = self.local_last_clean_latent_1frame.clone()
                is_cont = True

            steered_probe_pairs = []
            entity_choice_distributions = []
            dynamic_state_labels = []
            chosen_sentences = []

            for idx_e, slot in enumerate(stage.entities):
                ctrl = self._get_controller(slot, idx_e)
                pool_items = entity_pool_candidates[idx_e]
                cur_st_id = entity_current_states[idx_e]
                pool_ids = [item["state_id"] for item in pool_items]
                pool_prompts = [item["probe_prompt"] for item in pool_items]
                pool_sents = {item["state_id"]: item["scene_sentence"] for item in pool_items}

                enc_pairs = [self.backend.encode_semantics(p, max_len=128) for p in pool_prompts]
                bank_K = torch.cat([ep[0] for ep in enc_pairs], dim=0)
                masks_K = [ep[1] for ep in enc_pairs]

                start_id, target_id, steered_emb, probs_dict = ctrl.select_concept_full_rank_524288d(
                    pool_ids=pool_ids,
                    candidate_embeds_Kx128x4096=bank_K,
                    device=self.device,
                    current_state_id=cur_st_id,
                )
                t_idx = pool_ids.index(target_id) if target_id in pool_ids else 0
                steered_probe_pairs.append((steered_emb.to(self.dtype), masks_K[t_idx]))
                entity_choice_distributions.append(probs_dict)

                if idx_e == stage.quest.active_slot_idx:
                    dynamic_state_labels.append(f"e{idx_e}:{start_id}->{target_id}")
                    chosen_sentences.append(pool_sents.get(target_id, ""))
                elif idx_e == stage.quest.catalyst_slot_idx:
                    top_cat_id = max(probs_dict, key=probs_dict.get) if probs_dict else start_id
                    dynamic_state_labels.append(f"e{idx_e}:{top_cat_id}[CAT]")
                    chosen_sentences.insert(0, pool_sents.get(top_cat_id, ""))
                else:
                    top_hold_id = max(probs_dict, key=probs_dict.get) if probs_dict else start_id
                    dynamic_state_labels.append(f"e{idx_e}:{top_hold_id}[HOLD]")

            if chosen_sentences:
                scene_prompt = f"{self.manifest.arena_context_desc}. " + " ".join([s for s in chosen_sentences if s])
            matchup_tag = " | ".join(dynamic_state_labels)

            walk_in_emb, walk_in_mask = self.backend.encode_semantics(walk_in_prompt, max_len=128)
            pos_emb, pos_mask = self.backend.encode_semantics(scene_prompt, max_len=128)
            neg_emb, neg_mask = self.backend.encode_semantics(self.manifest.negative_desc, max_len=128)
            passive_pair = self.backend.encode_semantics(self.manifest.static_hold_desc, max_len=128)
            ongoing_pair = self.backend.encode_semantics(ongoing_hyp, max_len=128)
            completed_pair = self.backend.encode_semantics(completed_hyp, max_len=128)
            failed_pair = self.backend.encode_semantics(failed_hyp, max_len=128) if failed_hyp else None

            drv_idx = (
                stage.quest.catalyst_slot_idx
                if 0 <= stage.quest.catalyst_slot_idx < len(stage.entities)
                else stage.quest.active_slot_idx
            )
            drv_ctrl = self._get_controller(stage.entities[drv_idx], drv_idx)
            steered_pos_emb = drv_ctrl.steer_semantic_probe_full_rank(pos_emb, device=self.device).to(self.dtype)

            gen = torch.Generator(device=self.device).manual_seed(step_seed)
            rope_scale = (
                self.backend.temporal_ratio / 16.0,
                float(self.backend.spatial_ratio),
                float(self.backend.spatial_ratio),
            )

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
            )

            self.local_last_clean_latent_1frame = unpacked_z0[:, :, -1:].detach().clone()

            if not is_cont:
                unpacked_z0 = imprint_allegiance_spin(unpacked_z0, base_seed=spin_seed)
                self.local_genesis_5d = unpacked_z0[:, :, :2].detach().clone()
                full_video_frames = self.backend.decode_latents_chunk(unpacked_z0, generator=gen)
                eval_lat = unpacked_z0
            else:
                decoded_all = self.backend.decode_latents_chunk(unpacked_z0, generator=gen)
                full_video_frames = decoded_all[1:] if len(decoded_all) > 1 else decoded_all
                eval_lat = unpacked_z0[:, :, 1:]

            traj, high_dim_ctx = self.referee.evaluate_generative_quest_and_entities(
                unpacked_latents_5d=eval_lat,
                origin_anchor_5d=self.local_genesis_5d,
                entity_probe_embeds_list=steered_probe_pairs,
                ongoing_hyp_embeds=ongoing_pair,
                completed_hyp_embeds=completed_pair,
                failed_hyp_embeds=failed_pair,
                passive_void_embeds=passive_pair,
                rope_scale=rope_scale,
                total_video_frames=len(full_video_frames),
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
            )

        fin = traj[-1]
        self.entity_coherences = list(fin.entity_coherences)
        self.entity_transitions = list(fin.entity_transition_pcts)
        self.quest_progress_pct = float(fin.quest_progress_pct)

        annotated_frames = [
            draw_identity_hud(
                rgb,
                traj,
                idx,
                stage,
                self.active_entity_parties,
                self.manifest,
                self.maneuver_phase + 1,
                self.game_mode,
                monochrome=self.monochrome,
            )
            for idx, rgb in enumerate(full_video_frames)
        ]

        dur_s = len(annotated_frames) / 16.0
        quest_completed = (self.quest_progress_pct >= 95.0)
        act_coh = (
            self.entity_coherences[stage.quest.active_slot_idx]
            if stage.quest.active_slot_idx < len(self.entity_coherences)
            else 100.0
        )
        quest_failed = (act_coh <= 5.0)

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
                # Фиксируем новое выбранное состояние расходника в глобальной памяти мира:
                sub_idx = stage.quest.active_slot_idx
                if sub_idx < len(high_dim_ctx.entity_state_labels):
                    lbl = high_dim_ctx.entity_state_labels[sub_idx]
                    if "->" in lbl:
                        new_st = lbl.split("->", 1)[1].strip()
                        s_key = stage.entities[sub_idx].slot_id or f"e{sub_idx}"
                        self.persistent_entity_states[s_key] = new_st

                # Выбираем следующую ветку реакции НЕ по рельсам, а по победителю 128D-конкуренции ветвей:
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
                decision_log = f"REACTION RESET: Progress={self.quest_progress_pct:.1f}%"

        t_start = self.total_rendered_frames / 16.0
        self.total_rendered_frames += len(annotated_frames)
        t_end = self.total_rendered_frames / 16.0
        self.block_idx += 1
        gen_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        fps = len(annotated_frames) / max(0.001, gen_ms / 1000.0)

        # Полный дебаг всех сделанных и отвергнутых выборов каждого агента в телеметрию:
        per_entity_choice_debug = {}
        for idx_c, c_dist in enumerate(high_dim_ctx.entity_choice_distributions):
            s_id = stage.entities[idx_c].slot_id if idx_c < len(stage.entities) else f"e{idx_c}"
            if c_dist:
                sorted_c = sorted(c_dist.items(), key=lambda x: x[1], reverse=True)
                per_entity_choice_debug[s_id] = {
                    "chosen": {sorted_c[0][0]: sorted_c[0][1]},
                    "rejected_alternatives": {k: v for k, v in sorted_c[1:]},
                }

        telemetry_record = {
            "step": self.block_idx,
            "t": f"{t_start:.1f}-{t_end:.1f}",
            "duration_sec": round(dur_s, 1),
            "global_stage": stage.global_stage,
            "stage_title": stage.stage_title,
            "reaction_formula": stage.quest.reaction_formula,
            "driver_tag": stage.quest.driver_tag,
            "active_entities": matchup_tag,
            "entity_state_transitions_pct": dict(zip(high_dim_ctx.entity_state_labels, self.entity_transitions)),
            "wm_524288d_choice_debug": per_entity_choice_debug,
            "reaction_branch_competition_128d": high_dim_ctx.reaction_branch_scores,
            "latent_dominance_ratio": round(dom_ratio, 3),
            "feigenbaum_chaos_metrics": {
                "feigenbaum_r": round(high_dim_ctx.feigenbaum_r, 4),
                "feigenbaum_delta_const": 4.669201609,
                "catalytic_resonance_pct": round(high_dim_ctx.catalytic_resonance_pct, 1),
                "chaos_regime": high_dim_ctx.chaos_Window_regime,
            },
            "quest_oracle_metrics": {
                "quest_progress_pct": round(self.quest_progress_pct, 1),
                "prob_completed_peak_pct": round(high_dim_ctx.prob_completed_peak, 1),
                "prob_ongoing_mean_pct": round(high_dim_ctx.prob_ongoing_mean, 1),
                "entity_coherence_end": high_dim_ctx.entity_coherence_end,
                "entity_work_done": high_dim_ctx.entity_work_done,
            },
            "transformer_physics_explanation": high_dim_ctx.transformer_physics_explanation,
            "decision": decision_log,
            "prompt": scene_prompt,
        }

        gc.collect()
        torch.cuda.empty_cache()

        info_tag = f"[S{stage.global_stage} {stage.quest.driver_tag} | {dur_s:.1f}s] {decision_log}"
        return annotated_frames, gen_ms, fps, info_tag, telemetry_record
