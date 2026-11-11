#!/usr/bin/env python3
"""
neuro_battle_server.py — 3D-DiT сервер полнорангового 524,288-D выбора из рабочей памяти
========================================================================================
- На каждом шаге все K концептов из allowed_pool каждого агента оцениваются в полном
  пространстве [K × 128 × 4096] (524,288-D) против 96,000-D поля рабочей памяти без .mean().
- Сохраняет и возвращает полную матрицу сделанных и отвергнутых выборов (choice_distributions)
  и конкурирующих ветвей реакций (reaction_branch_scores).
"""

from __future__ import annotations
import os
import argparse
import gc
import multiprocessing as mp
from multiprocessing.connection import Listener
import time
import traceback
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
mp.current_process().authkey = b"battle_ltx"

from neuro_ltx_backend import LTXModelBackend
from neuro_referee import CoordinateFreeHistoryReferee
from neuro_solver_stepper import LatentFlowWindowStepper, imprint_allegiance_spin
from neuro_controller import SymmetricBotController


class PersistentBattleServer:

    def __init__(self):
        print("=" * 80)
        print("🚀 [BATTLE-SERVER] Loading LTX-Video into VRAM (524,288-D Full-Rank Choice Heterarchy)...")
        t0 = time.perf_counter()
        self.backend = LTXModelBackend()
        self.device = self.backend.device
        self.dtype = self.backend.dtype

        self.referee = CoordinateFreeHistoryReferee(self.backend.transformer, self.device, self.dtype)
        self.stepper = LatentFlowWindowStepper(
            backend=self.backend,
            referee=self.referee,
            steps=20,
            cfg_scale=4.0,
            guidance_rescale=0.7,
        )
        self.prompt_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.slot_controllers: dict[str, SymmetricBotController] = {}

        self.genesis_anchor_5d: torch.Tensor | None = None
        self.last_clean_latent_1frame: torch.Tensor | None = None

        dt = time.perf_counter() - t0
        print(f"✅ [BATTLE-SERVER] Ready in {dt:.1f}s! Zero-Reduction 524,288-D Choice Selection Active.")
        print("=" * 80)

    def _get_slot_controller(self, slot_id: str, binding: str, attempt_num: int) -> SymmetricBotController:
        key = f"{slot_id}:{binding}"
        if key not in self.slot_controllers:
            self.slot_controllers[key] = SymmetricBotController(bot_id=slot_id, source_mode=binding)
        ctrl = self.slot_controllers[key]
        ctrl.source_mode = binding
        ctrl.attempt_num = attempt_num
        return ctrl

    @torch.no_grad()
    def _encode_semantics_cached(self, text: str, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        key = (text, int(max_len))
        if key not in self.prompt_cache:
            if len(self.prompt_cache) > 256:
                self.prompt_cache.clear()
            self.prompt_cache[key] = self.backend.encode_semantics(text, max_len=max_len)
        return self.prompt_cache[key]

    @torch.no_grad()
    def handle_execute_battle_window(self, msg: dict) -> dict:
        t0 = time.perf_counter()

        unit_probe_cmds = msg.get("unit_probe_cmds", [])
        entity_bindings = msg.get("entity_bindings", [])
        entity_pools_candidates = msg.get("entity_pool_candidates", [])
        entity_current_states = msg.get("entity_current_states", [])
        candidate_branch_specs = msg.get("candidate_branch_specs", [])
        arena_context_desc = msg.get("arena_context_desc", "")

        ongoing_hyp = msg.get("ongoing_hyp", "")
        completed_hyp = msg.get("completed_hyp", "")
        failed_hyp = msg.get("failed_hyp", "")
        completion_gain = float(msg.get("completion_gain", 3.5))
        entropy_drain = float(msg.get("entropy_drain", 0.2))

        active_slot_idx = int(msg.get("active_slot_idx", 0))
        catalyst_slot_idx = int(msg.get("catalyst_slot_idx", -1))
        driver_tag = str(msg.get("driver_tag", "SELF_WM"))
        init_entity_transitions = msg.get("init_entity_transitions", [])

        reset_stage_stream = bool(msg.get("reset_stage_stream", False))
        init_entity_coherences = msg.get("init_entity_coherences", [100.0])
        init_quest_progress = float(msg.get("init_quest_progress", 0.0))
        maneuver_phase = int(msg.get("maneuver_phase", 0))
        block_idx = int(msg.get("block_idx", maneuver_phase))

        walk_in_prompt = msg.get("walk_in_prompt", "")
        scene_prompt = msg.get("scene_prompt", "")
        negative_desc = msg.get("negative_desc", "")
        static_hold_desc = msg.get("static_hold_desc", "")

        width = int(msg["width"])
        height = int(msg["height"])
        steps = int(msg["steps"])
        cfg_scale = float(msg["cfg_scale"])
        step_seed = int(msg["step_seed"])
        spin_seed = int(msg["spin_seed"])

        self.stepper.steps = max(12, steps)
        self.stepper.cfg_scale = cfg_scale

        h_lat = height // self.backend.spatial_ratio
        w_lat = width // self.backend.spatial_ratio
        total_lat_frames = 5

        if reset_stage_stream or self.last_clean_latent_1frame is None:
            anchor_latent = None
            is_continuation = False
        else:
            anchor_latent = self.last_clean_latent_1frame.clone()
            is_continuation = True

        # -------------------------------------------------------------------------
        # 1. ПОЛНОРАНГОВЫЙ 524,288-D ВЫБОР СОСТОЯНИЙ И ДЕЙСТВИЙ ДЛЯ КАЖДОГО АГЕНТА
        # -------------------------------------------------------------------------
        steered_unit_probe_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        entity_choice_distributions: list[dict[str, float]] = []
        dynamic_state_labels: list[str] = []
        chosen_sentences_for_scene: list[str] = []

        n_slots = max(len(unit_probe_cmds), len(entity_pools_candidates))
        for idx_u in range(n_slots):
            bind_str = entity_bindings[idx_u] if idx_u < len(entity_bindings) else "eeg:FCz"
            ctrl = self._get_slot_controller(f"e{idx_u}", bind_str, block_idx + 1)

            pool_items = entity_pools_candidates[idx_u] if idx_u < len(entity_pools_candidates) else []
            cur_st_id = entity_current_states[idx_u] if idx_u < len(entity_current_states) else None

            if pool_items:
                pool_ids = [item["state_id"] for item in pool_items]
                pool_prompts = [item["probe_prompt"] for item in pool_items]
                pool_sentences = {item["state_id"]: item["scene_sentence"] for item in pool_items}

                encoded_pairs = [self._encode_semantics_cached(p, max_len=128) for p in pool_prompts]
                bank_Kx128x4096 = torch.cat([ep[0] for ep in encoded_pairs], dim=0)
                masks_list = [ep[1] for ep in encoded_pairs]

                start_id, target_id, steered_emb, probs_dict = ctrl.select_concept_full_rank_524288d(
                    pool_ids=pool_ids,
                    candidate_embeds_Kx128x4096=bank_Kx128x4096,
                    device=self.device,
                    current_state_id=cur_st_id,
                )
                t_idx = pool_ids.index(target_id) if target_id in pool_ids else 0
                steered_unit_probe_pairs.append((steered_emb.to(self.dtype), masks_list[t_idx]))
                entity_choice_distributions.append(probs_dict)

                if idx_u == active_slot_idx:
                    dynamic_state_labels.append(f"e{idx_u}:{start_id}->{target_id}")
                    chosen_sentences_for_scene.append(pool_sentences.get(target_id, ""))
                elif idx_u == catalyst_slot_idx:
                    top_cat_id = max(probs_dict, key=probs_dict.get) if probs_dict else start_id
                    dynamic_state_labels.append(f"e{idx_u}:{top_cat_id}[CAT]")
                    chosen_sentences_for_scene.insert(0, pool_sentences.get(top_cat_id, ""))
                else:
                    top_hold_id = max(probs_dict, key=probs_dict.get) if probs_dict else start_id
                    dynamic_state_labels.append(f"e{idx_u}:{top_hold_id}[HOLD]")
            else:
                raw_cmd = unit_probe_cmds[idx_u] if idx_u < len(unit_probe_cmds) else static_hold_desc
                u_emb, u_mask = self._encode_semantics_cached(raw_cmd, max_len=128)
                steered_u = ctrl.steer_semantic_probe_full_rank(u_emb, device=self.device).to(self.dtype)
                steered_unit_probe_pairs.append((steered_u, u_mask))
                entity_choice_distributions.append({f"e{idx_u}_default": 100.0})
                dynamic_state_labels.append(f"e{idx_u}[HOLD]")

        if chosen_sentences_for_scene and arena_context_desc:
            scene_prompt = f"{arena_context_desc}. " + " ".join([s for s in chosen_sentences_for_scene if s])

        matchup_tag = " | ".join(dynamic_state_labels)

        walk_in_emb, walk_in_mask = self._encode_semantics_cached(walk_in_prompt, max_len=128)
        pos_emb, pos_mask = self._encode_semantics_cached(scene_prompt, max_len=128)
        neg_emb, neg_mask = self._encode_semantics_cached(negative_desc, max_len=128)
        passive_pair = self._encode_semantics_cached(static_hold_desc, max_len=128)
        ongoing_pair = self._encode_semantics_cached(ongoing_hyp, max_len=128)
        completed_pair = self._encode_semantics_cached(completed_hyp, max_len=128)
        failed_pair = self._encode_semantics_cached(failed_hyp, max_len=128) if failed_hyp else None

        driver_idx = catalyst_slot_idx if (0 <= catalyst_slot_idx < n_slots) else active_slot_idx
        if 0 <= driver_idx < n_slots:
            drv_bind = entity_bindings[driver_idx] if driver_idx < len(entity_bindings) else "eeg:FCz"
            drv_ctrl = self._get_slot_controller(f"e{driver_idx}", drv_bind, block_idx + 1)
            steered_pos_emb = drv_ctrl.steer_semantic_probe_full_rank(pos_emb, device=self.device).to(self.dtype)
        else:
            steered_pos_emb = pos_emb

        gen = torch.Generator(device=self.device).manual_seed(step_seed)

        unpacked_z0, dom_ratio = self.stepper.run_window(
            anchor_latent_0=anchor_latent,
            genesis_ref_latent_5d=self.genesis_anchor_5d,
            walk_in_emb=walk_in_emb,
            walk_in_mask=walk_in_mask,
            pos_emb=steered_pos_emb,
            pos_mask=pos_mask,
            neg_emb=neg_emb,
            neg_mask=neg_mask,
            entity_probes=steered_unit_probe_pairs,
            passive_emb=passive_pair[0],
            passive_mask=passive_pair[1],
            total_lat_frames=total_lat_frames,
            h_lat=h_lat,
            w_lat=w_lat,
            gen=gen,
            is_continuation_block=is_continuation,
        )

        self.last_clean_latent_1frame = unpacked_z0[:, :, -1:].detach().clone()

        if not is_continuation:
            unpacked_z0 = imprint_allegiance_spin(unpacked_z0, base_seed=spin_seed)
            self.genesis_anchor_5d = unpacked_z0[:, :, :2].detach().clone()
            full_video_frames = self.backend.decode_latents_chunk(unpacked_z0, generator=gen)
            eval_latents_5d = unpacked_z0
        else:
            decoded_all = self.backend.decode_latents_chunk(unpacked_z0, generator=gen)
            full_video_frames = decoded_all[1:] if len(decoded_all) > 1 else decoded_all
            eval_latents_5d = unpacked_z0[:, :, 1:]

        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        traj, high_dim_ctx = self.referee.evaluate_generative_quest_and_entities(
            unpacked_latents_5d=eval_latents_5d,
            origin_anchor_5d=self.genesis_anchor_5d,
            entity_probe_embeds_list=steered_unit_probe_pairs,
            ongoing_hyp_embeds=ongoing_pair,
            completed_hyp_embeds=completed_pair,
            failed_hyp_embeds=failed_pair,
            passive_void_embeds=passive_pair,
            rope_scale=rope_scale,
            total_video_frames=len(full_video_frames),
            matchup_tag=matchup_tag,
            dom_ratio=dom_ratio,
            gen=gen,
            init_entity_coherences=init_entity_coherences,
            init_quest_progress=init_quest_progress,
            completion_gain=completion_gain,
            entropy_drain=entropy_drain,
            maneuver_phase=maneuver_phase,
            active_slot_idx=active_slot_idx,
            catalyst_slot_idx=catalyst_slot_idx,
            driver_tag=driver_tag,
            entity_state_labels=dynamic_state_labels,
            init_entity_transitions=init_entity_transitions,
            entity_choice_distributions=entity_choice_distributions,
            candidate_branch_specs=candidate_branch_specs,
        )

        high_dim_ctx_ipc = high_dim_ctx.to_ipc_safe()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        sub_choices = entity_choice_distributions[active_slot_idx] if active_slot_idx < len(entity_choice_distributions) else {}
        cat_choices = entity_choice_distributions[catalyst_slot_idx] if 0 <= catalyst_slot_idx < len(entity_choice_distributions) else {}
        print(
            f"🔮 [BLOCK #{maneuver_phase+1} | {driver_tag}] {matchup_tag}\n"
            f"   ├─ 🎯 CatChoices(524k-D): {cat_choices}\n"
            f"   ├─ 🧬 SubChoices(524k-D): {sub_choices}\n"
            f"   └─ ⚖️ Branches(128D): {high_dim_ctx.reaction_branch_scores} | dS={traj[-1].quest_progress_pct:.1f}% ({dt_ms:.0f}ms)",
            flush=True,
        )

        return {
            "status": "ok",
            "full_video_frames": full_video_frames,
            "dom_ratio": float(dom_ratio),
            "traj": traj,
            "high_dim_ctx": high_dim_ctx_ipc,
            "scene_prompt": scene_prompt,
            "matchup_tag": matchup_tag,
        }


def main():
    parser = argparse.ArgumentParser(description="Persistent LTX-Video Battle Server")
    parser.add_argument("--port", type=int, default=6008, help="IPC port for Battle Server")
    args = parser.parse_args()

    server = PersistentBattleServer()
    listener = Listener(("localhost", args.port), authkey=b"battle_ltx")
    print(f"🟢 [BATTLE-SERVER] Listening on localhost:{args.port}...")

    req_count = 0
    while True:
        try:
            conn = listener.accept()
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break

                if not isinstance(msg, dict):
                    continue

                cmd = msg.get("cmd")
                if cmd == "ping":
                    conn.send({
                        "status": "ok",
                        "spatial_ratio": server.backend.spatial_ratio,
                        "temporal_ratio": server.backend.temporal_ratio,
                    })
                elif cmd == "execute_battle_window":
                    resp = server.handle_execute_battle_window(msg)
                    conn.send(resp)
                    req_count += 1
                    if req_count % 4 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                elif cmd == "clear_cache":
                    server.prompt_cache.clear()
                    server.genesis_anchor_5d = None
                    server.last_clean_latent_1frame = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    conn.send({"status": "ok"})
                else:
                    conn.send({"status": "error", "error": f"Unknown cmd: {cmd}"})

        except KeyboardInterrupt:
            print("\n🛑 [BATTLE-SERVER] Stopped.")
            break
        except Exception as e:
            print(f"⚠️ [BATTLE-SERVER ERROR]: {e}")
            traceback.print_exc()
            time.sleep(0.2)


if __name__ == "__main__":
    main()
