#!/usr/bin/env python3
"""
neuro_referee.py — Многоветвевой Rectified-Flow 3D-DiT World-Model Оракул (100% Аконцептуальный)
================================================================================================
- Оценивает условное правдоподобие реального видео-перехода P(z_new | z_prev, H_k)
  для [H_stay, H_branch_1, ..., H_branch_M] на полных [1, 128, 4096] T5-эмбеддингах.
- Усредняет по n_noise_probes=2 общим шумам eps (Common Random Numbers) и использует
  корректное эффективное число степеней свободы (без искусственного раздутия sqrt(192)).
- Чанкинг батча (max_batch=4) защищает от OOM при любом числе конкурирующих веток.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn.functional as F

FEIGENBAUM_DELTA: float = 4.669201609102990
FEIGENBAUM_ALPHA: float = 2.502907875095892
R_EDGE_OF_CHAOS: float = 3.569945671870944


def _pack_latents_p1(latents_5d: torch.Tensor) -> torch.Tensor:
    return latents_5d.permute(0, 2, 3, 4, 1).flatten(1, 3).contiguous()


def _unpack_latents_p1(packed_3d: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    b = packed_3d.size(0)
    return packed_3d.reshape(b, f, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()


@dataclass
class HighDimBattleContextTensor:
    entity_trajectories_128d: torch.Tensor | np.ndarray
    entity_action_streams_128d: torch.Tensor | np.ndarray
    interaction_matrix_NxN: torch.Tensor | np.ndarray
    entity_coherence_end: list[float] = field(default_factory=list)
    entity_work_done: list[float] = field(default_factory=list)
    entity_transition_pcts: list[float] = field(default_factory=list)
    entity_state_labels: list[str] = field(default_factory=list)
    entity_choice_distributions: list[dict[str, float]] = field(default_factory=list)
    reaction_branch_scores: dict[str, float] = field(default_factory=dict)
    chosen_next_stage_idx: int = -1
    winning_branch_stage_idx: int = -1
    winning_branch_t_stat: float = 0.0
    oracle_advantage: float = 0.0
    quest_progress_end: float = 0.0
    prob_completed_peak: float = 50.0
    prob_ongoing_mean: float = 50.0
    latent_dominance_ratio: float = 1.0
    feigenbaum_r: float = R_EDGE_OF_CHAOS
    catalytic_resonance_pct: float = 50.0
    chaos_Window_regime: str = "EDGE_OF_CHAOS"
    transformer_physics_explanation: str = ""
    prediction_error_mse: float = 0.0
    rf_t_stat: float = 0.0

    @staticmethod
    def _to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    def to_ipc_safe(self) -> HighDimBattleContextTensor:
        return HighDimBattleContextTensor(
            entity_trajectories_128d=self._to_np(self.entity_trajectories_128d),
            entity_action_streams_128d=self._to_np(self.entity_action_streams_128d),
            interaction_matrix_NxN=self._to_np(self.interaction_matrix_NxN),
            entity_coherence_end=list(self.entity_coherence_end),
            entity_work_done=list(self.entity_work_done),
            entity_transition_pcts=list(self.entity_transition_pcts),
            entity_state_labels=list(self.entity_state_labels),
            entity_choice_distributions=list(self.entity_choice_distributions),
            reaction_branch_scores=dict(self.reaction_branch_scores),
            chosen_next_stage_idx=int(self.chosen_next_stage_idx),
            winning_branch_stage_idx=int(self.winning_branch_stage_idx),
            winning_branch_t_stat=float(self.winning_branch_t_stat),
            oracle_advantage=float(self.oracle_advantage),
            quest_progress_end=float(self.quest_progress_end),
            prob_completed_peak=float(self.prob_completed_peak),
            prob_ongoing_mean=float(self.prob_ongoing_mean),
            latent_dominance_ratio=float(self.latent_dominance_ratio),
            feigenbaum_r=float(self.feigenbaum_r),
            catalytic_resonance_pct=float(self.catalytic_resonance_pct),
            chaos_Window_regime=str(self.chaos_Window_regime),
            transformer_physics_explanation=str(self.transformer_physics_explanation),
            prediction_error_mse=float(self.prediction_error_mse),
            rf_t_stat=float(self.rf_t_stat),
        )

    def from_ipc_safe(self) -> HighDimBattleContextTensor:
        return HighDimBattleContextTensor(
            entity_trajectories_128d=self._to_tensor(self.entity_trajectories_128d),
            entity_action_streams_128d=self._to_tensor(self.entity_action_streams_128d),
            interaction_matrix_NxN=self._to_tensor(self.interaction_matrix_NxN),
            entity_coherence_end=list(self.entity_coherence_end),
            entity_work_done=list(self.entity_work_done),
            entity_transition_pcts=list(self.entity_transition_pcts),
            entity_state_labels=list(self.entity_state_labels),
            entity_choice_distributions=list(self.entity_choice_distributions),
            reaction_branch_scores=dict(self.reaction_branch_scores),
            chosen_next_stage_idx=int(self.chosen_next_stage_idx),
            winning_branch_stage_idx=int(self.winning_branch_stage_idx),
            winning_branch_t_stat=float(self.winning_branch_t_stat),
            oracle_advantage=float(self.oracle_advantage),
            quest_progress_end=float(self.quest_progress_end),
            prob_completed_peak=float(self.prob_completed_peak),
            prob_ongoing_mean=float(self.prob_ongoing_mean),
            latent_dominance_ratio=float(self.latent_dominance_ratio),
            feigenbaum_r=float(self.feigenbaum_r),
            catalytic_resonance_pct=float(self.catalytic_resonance_pct),
            chaos_Window_regime=str(self.chaos_Window_regime),
            transformer_physics_explanation=str(self.transformer_physics_explanation),
            prediction_error_mse=float(self.prediction_error_mse),
            rf_t_stat=float(self.rf_t_stat),
        )


@dataclass
class LatentIdentityFrameTelemetry:
    frame_idx: int
    time_sec: float
    phase_label: str
    primary_coherence_pct: float
    quest_progress_pct: float
    quest_remaining_pct: float
    prob_completed_pct: float
    prob_ongoing_pct: float
    semantic_flux_pct: float
    matchup_tag: str
    feigenbaum_r: float = R_EDGE_OF_CHAOS
    catalytic_resonance_pct: float = 50.0
    chaos_regime: str = "EDGE_OF_CHAOS"
    entity_coherences: list[float] = field(default_factory=list)
    entity_works: list[float] = field(default_factory=list)
    entity_transition_pcts: list[float] = field(default_factory=list)
    entity_state_labels: list[str] = field(default_factory=list)
    entity_choice_distributions: list[dict[str, float]] = field(default_factory=list)
    reaction_branch_scores: dict[str, float] = field(default_factory=dict)


class CoordinateFreeHistoryReferee:

    def __init__(
        self,
        transformer: any,
        device: torch.device,
        dtype: torch.dtype,
        probe_sigma: float = 0.55,
        n_noise_probes: int = 2,
        max_batch: int = 4,
    ):
        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self.probe_sigma = float(np.clip(probe_sigma, 0.25, 0.80))
        self.n_noise_probes = max(1, int(n_noise_probes))
        self.max_batch = max(1, int(max_batch))

    @torch.no_grad()
    def _score_hypotheses_rectified_flow_3d_dit(
        self,
        z0_full_5d: torch.Tensor,
        n_cond: int,
        hyp_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        rope_scale: tuple[float, float, float],
        gen: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, t_lat, h_lat, w_lat = z0_full_5d.shape
        m_hyps = len(hyp_pairs)
        n_cond_safe = max(0, min(int(n_cond), t_lat - 1))

        z0_f = z0_full_5d.to(device=self.device, dtype=torch.float32)
        sigma = self.probe_sigma
        t_val = sigma * 1000.0

        cond_mask_5d = torch.zeros(
            (1, 1, t_lat, h_lat, w_lat), device=self.device, dtype=torch.float32
        )
        if n_cond_safe > 0:
            cond_mask_5d[:, :, :n_cond_safe] = 1.0
        cond_packed = _pack_latents_p1(cond_mask_5d).squeeze(-1)  # [1, S]

        enc_embeds_all = torch.cat(
            [p[0].to(device=self.device, dtype=self.dtype) for p in hyp_pairs], dim=0
        )
        enc_masks_all = torch.cat(
            [p[1].to(device=self.device, dtype=torch.bool) for p in hyp_pairs], dim=0
        )

        # Маска реального пространственно-временного движения:
        if n_cond_safe > 0:
            ref_anchor = z0_f[:, :, n_cond_safe - 1 : n_cond_safe]
            motion = (z0_f[:, :, n_cond_safe:] - ref_anchor).pow(2).mean(dim=1).sqrt()
        elif t_lat >= 2:
            diff_m = (z0_f[:, :, 1:] - z0_f[:, :, :-1]).pow(2).mean(dim=1).sqrt()
            motion = torch.cat([diff_m[:, :1], diff_m], dim=1)
        else:
            motion = torch.ones((1, t_lat - n_cond_safe, h_lat, w_lat), device=self.device)

        w_raw = 0.25 + 0.75 * (motion / (motion.mean() + 1e-5))
        w_tok = w_raw / (w_raw.mean() + 1e-6)  # [1, T_gen, H, W]

        sq_err_accum = torch.zeros(
            (m_hyps, t_lat - n_cond_safe, h_lat, w_lat),
            device=self.device,
            dtype=torch.float32,
        )

        # Усредняем по n_noise_probes общим шумам eps (резко снижает дисперсию при 0 лишней VRAM):
        for _ in range(self.n_noise_probes):
            eps_5d = torch.randn(
                z0_f.shape, generator=gen, device=self.device, dtype=torch.float32
            )
            z_sigma_5d = (1.0 - sigma) * z0_f + sigma * eps_5d
            z_input_5d = cond_mask_5d * z0_f + (1.0 - cond_mask_5d) * z_sigma_5d
            z_packed_1 = _pack_latents_p1(z_input_5d).to(self.dtype)
            v_target_gen = (eps_5d - z0_f)[:, :, n_cond_safe:]

            # Безопасный чанкинг по max_batch=4:
            for start_i in range(0, m_hyps, self.max_batch):
                end_i = min(m_hyps, start_i + self.max_batch)
                b_cur = end_i - start_i
                z_b = z_packed_1.expand(b_cur, -1, -1)
                t_b = ((1.0 - cond_packed) * t_val).expand(b_cur, -1).contiguous()
                v_pred_packed = self.transformer(
                    hidden_states=z_b,
                    encoder_hidden_states=enc_embeds_all[start_i:end_i],
                    timestep=t_b,
                    encoder_attention_mask=enc_masks_all[start_i:end_i],
                    num_frames=t_lat,
                    height=h_lat,
                    width=w_lat,
                    rope_interpolation_scale=rope_scale,
                    return_dict=False,
                )[0].float()
                v_pred_5d = _unpack_latents_p1(v_pred_packed, t_lat, h_lat, w_lat)
                v_pred_gen = v_pred_5d[:, :, n_cond_safe:]
                sq_err_accum[start_i:end_i] += (v_pred_gen - v_target_gen).pow(2).mean(dim=1)

        sq_err_tok = sq_err_accum / float(self.n_noise_probes)
        mse_per_hyp = (sq_err_tok * w_tok).mean(dim=(1, 2, 3))
        return mse_per_hyp, sq_err_tok, w_tok

    @staticmethod
    def _paired_t_stat(
        err_a: torch.Tensor, err_b: torch.Tensor, w_tok: torch.Tensor
    ) -> float:
        """
        Парная t-статистика с учётом пространственно-временной корреляции токенов
        (n_eff = n_tok / 16): исключает ложные 95%/2% на неподвижной сцене.
        > 0 означает, что гипотеза B объясняет видео-переход лучше, чем A.
        """
        diff = (err_a - err_b) * w_tok[0]
        n_eff = max(1.0, float(diff.numel()) / 16.0)
        se = float(diff.std(unbiased=False).item()) / math.sqrt(n_eff) + 1e-6
        return float(diff.mean().item()) / se

    @torch.no_grad()
    def evaluate_generative_quest_and_entities(
        self,
        unpacked_latents_5d: torch.Tensor,
        origin_anchor_5d: torch.Tensor | None,
        entity_probe_embeds_list: list[tuple[torch.Tensor, torch.Tensor]],
        ongoing_hyp_embeds: tuple[torch.Tensor, torch.Tensor],
        completed_hyp_embeds: tuple[torch.Tensor, torch.Tensor],
        failed_hyp_embeds: tuple[torch.Tensor, torch.Tensor] | None,
        passive_void_embeds: tuple[torch.Tensor, torch.Tensor],
        rope_scale: tuple[float, float, float],
        total_video_frames: int,
        matchup_tag: str,
        dom_ratio: float,
        gen: torch.Generator,
        init_entity_coherences: list[float],
        init_quest_progress: float = 0.0,
        completion_gain: float = 3.2,
        entropy_drain: float = 0.2,
        maneuver_phase: int = 0,
        active_slot_idx: int = 0,
        catalyst_slot_idx: int = -1,
        driver_tag: str = "SELF_WM",
        entity_state_labels: list[str] | None = None,
        init_entity_transitions: list[float] | None = None,
        entity_choice_distributions: list[dict[str, float]] | None = None,
        candidate_branch_specs: list[dict] | None = None,
        candidate_branch_hyp_pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        current_stage_idx: int = 0,
        last_velocity_5d: torch.Tensor | None = None,
        n_cond_frames: int | None = None,
        **unused_kwargs,
    ) -> tuple[list[LatentIdentityFrameTelemetry], HighDimBattleContextTensor]:
        _, c_lat, t_lat, h_lat, w_lat = unpacked_latents_5d.shape
        n_ent = max(1, len(entity_probe_embeds_list))

        if n_cond_frames is not None:
            z0_full = unpacked_latents_5d
            n_cond = int(n_cond_frames)
        elif origin_anchor_5d is not None:
            anc_tail = origin_anchor_5d[:, :, -1:].to(
                device=unpacked_latents_5d.device, dtype=unpacked_latents_5d.dtype
            )
            if float((unpacked_latents_5d[:, :, :1] - anc_tail).abs().max().item()) < 1e-4:
                z0_full = unpacked_latents_5d
                n_cond = 1 if t_lat >= 2 else 0
            else:
                z0_full = torch.cat([anc_tail, unpacked_latents_5d], dim=2)
                n_cond = 1
        else:
            z0_full = unpacked_latents_5d
            n_cond = 1 if t_lat >= 2 else 0

        # Матрица резонанса сущностей в полном 524,288-D T5-пространстве:
        pas_524k = passive_void_embeds[0][0].float()
        ent_deltas_flat = []
        interaction_mat = torch.zeros((n_ent, n_ent), device=self.device)
        for u_i in range(n_ent):
            e_524k = entity_probe_embeds_list[u_i][0][0].float() - pas_524k
            ent_deltas_flat.append(F.normalize(e_524k.reshape(1, -1), p=2, dim=-1))
        for i in range(n_ent):
            for j in range(i + 1, n_ent):
                cos_524k = float(torch.mm(ent_deltas_flat[i], ent_deltas_flat[j].T).item())
                interaction_mat[i, j] = float(np.clip((cos_524k + 0.35) * 80.0, 10.0, 100.0))
                interaction_mat[j, i] = interaction_mat[i, j]

        cat_resonance = 0.65
        if 0 <= catalyst_slot_idx < n_ent and 0 <= active_slot_idx < n_ent and catalyst_slot_idx != active_slot_idx:
            cat_resonance = float(interaction_mat[catalyst_slot_idx, active_slot_idx].item()) / 100.0

        # МНОГОВЕТВЕВОЙ 3D-DiT RECTIFIED-FLOW ОРАКУЛ:
        # Оцениваем [H_stay(passive/ongoing), H_branch_0, H_branch_1, ..., H_branch_{M-1}] (плюс optional failed)
        has_multi_branches = (
            candidate_branch_specs is not None
            and candidate_branch_hyp_pairs is not None
            and len(candidate_branch_specs) == len(candidate_branch_hyp_pairs)
            and len(candidate_branch_specs) > 1
        )

        reaction_branch_scores: dict[str, float] = {}
        winning_branch_stage_idx = current_stage_idx
        winning_branch_t_stat = 0.0
        chosen_next_stage_idx = -1

        rf_mse_ong = 0.0
        rf_mse_cmp = 0.0
        eff_t_stat = 0.0

        if self.transformer is not None:
            if has_multi_branches:
                # 0: H_stay (нейтральная / текущая ongoing), 1..M: гипотезы всех исходящих веток
                hyp_list = [passive_void_embeds] + list(candidate_branch_hyp_pairs)
                mse_hyps, sq_err_tok, w_tok = self._score_hypotheses_rectified_flow_3d_dit(
                    z0_full_5d=z0_full,
                    n_cond=n_cond,
                    hyp_pairs=hyp_list,
                    rope_scale=rope_scale,
                    gen=gen,
                )
                err_stay = sq_err_tok[0]
                rf_mse_ong = float(mse_hyps[0].item())

                n_br = len(candidate_branch_specs)
                branch_t_vals = []
                for b_i in range(n_br):
                    t_b = self._paired_t_stat(err_stay, sq_err_tok[1 + b_i], w_tok)
                    branch_t_vals.append(t_b)

                t_tensor = torch.tensor(branch_t_vals, device=self.device, dtype=torch.float32)
                branch_probs = torch.softmax(t_tensor.clamp(-5.0, 5.0) / 1.0, dim=0)

                for b_i, b_spec in enumerate(candidate_branch_specs):
                    b_tag = str(b_spec.get("driver_tag", f"B{b_i}"))
                    reaction_branch_scores[b_tag] = round(float(branch_probs[b_i].item()) * 100.0, 1)

                best_b_idx = int(torch.argmax(t_tensor).item())
                winning_branch_stage_idx = int(candidate_branch_specs[best_b_idx].get("stage_idx", best_b_idx))
                winning_branch_t_stat = float(branch_t_vals[best_b_idx])
                chosen_next_stage_idx = winning_branch_stage_idx

                # Сравниваем активную ветку (или победившую ветку) с H_stay И со средней альтернативой:
                cur_b_pos = next(
                    (i for i, s in enumerate(candidate_branch_specs) if int(s.get("stage_idx", i)) == current_stage_idx),
                    0,
                )
                rf_mse_cmp = float(mse_hyps[1 + cur_b_pos].item())
                t_vs_stay = float(branch_t_vals[cur_b_pos])

                other_errs = [sq_err_tok[1 + j] for j in range(n_br) if j != cur_b_pos]
                mean_other_err = torch.stack(other_errs, dim=0).mean(dim=0)
                t_vs_others = self._paired_t_stat(mean_other_err, sq_err_tok[1 + cur_b_pos], w_tok)
                eff_t_stat = 0.50 * t_vs_stay + 0.50 * t_vs_others
            else:
                hyp_list = [ongoing_hyp_embeds, completed_hyp_embeds]
                has_failed = failed_hyp_embeds is not None
                if has_failed:
                    hyp_list.append(failed_hyp_embeds)
                mse_hyps, sq_err_tok, w_tok = self._score_hypotheses_rectified_flow_3d_dit(
                    z0_full_5d=z0_full,
                    n_cond=n_cond,
                    hyp_pairs=hyp_list,
                    rope_scale=rope_scale,
                    gen=gen,
                )
                rf_mse_ong = float(mse_hyps[0].item())
                rf_mse_cmp = float(mse_hyps[1].item())
                t_ong = self._paired_t_stat(sq_err_tok[0], sq_err_tok[1], w_tok)
                if has_failed:
                    t_fail = self._paired_t_stat(sq_err_tok[2], sq_err_tok[1], w_tok)
                    eff_t_stat = 0.50 * t_ong + 0.50 * t_fail
                else:
                    eff_t_stat = t_ong
                reaction_branch_scores[driver_tag] = 100.0
                winning_branch_t_stat = eff_t_stat

        clamped_t = float(np.clip(eff_t_stat, -5.0, 5.0))
        probs_hyp = torch.softmax(torch.tensor([0.0, clamped_t / 1.2], device=self.device), dim=0)
        p_ong_pct = float(probs_hyp[0].item()) * 100.0
        p_cmp_pct = float(probs_hyp[1].item()) * 100.0
        oracle_advantage = (p_cmp_pct - p_ong_pct) / 100.0  # [-1.0, +1.0]

        x_curr = float(np.clip(init_quest_progress / 100.0, 0.0, 1.0))
        gain_scale = 0.14 * float(completion_gain)
        if oracle_advantage > 0.06:
            net_step = oracle_advantage * gain_scale
        elif oracle_advantage < -0.06:
            net_step = oracle_advantage * (gain_scale * 0.75)
        else:
            net_step = 0.0

        running_quest_prog = float(np.clip((x_curr + net_step) * 100.0, 0.0, 100.0))

        r_feigenbaum = float(
            np.clip(
                R_EDGE_OF_CHAOS + oracle_advantage * (1.2 / FEIGENBAUM_DELTA),
                2.90,
                3.92,
            )
        )
        if r_feigenbaum < 3.0:
            chaos_regime = "ORDER_ATTRACTOR"
        elif r_feigenbaum < R_EDGE_OF_CHAOS:
            chaos_regime = "PERIOD_DOUBLING"
        elif r_feigenbaum < 3.8284:
            chaos_regime = "EDGE_OF_CHAOS(4.669)"
        else:
            chaos_regime = "P3_ORDER_WINDOW"

        transitions = [100.0] * n_ent
        if 0 <= active_slot_idx < n_ent:
            transitions[active_slot_idx] = round(running_quest_prog, 1)

        labels = (
            list(entity_state_labels)
            if entity_state_labels and len(entity_state_labels) == n_ent
            else [f"e{i}" for i in range(n_ent)]
        )
        choice_dists = (
            list(entity_choice_distributions)
            if entity_choice_distributions and len(entity_choice_distributions) == n_ent
            else [{} for _ in range(n_ent)]
        )

        cur_coh = (
            init_entity_coherences[active_slot_idx]
            if (init_entity_coherences and active_slot_idx < len(init_entity_coherences))
            else 100.0
        )
        sub_coh = float(
            np.clip(cur_coh - entropy_drain + 2.5 * oracle_advantage, 25.0, 100.0)
        )
        coherences_end = [100.0] * n_ent
        if 0 <= active_slot_idx < n_ent:
            coherences_end[active_slot_idx] = round(sub_coh, 1)

        work_done_end = [round(t_val_i, 1) for t_val_i in transitions]
        cat_res_pct = round(cat_resonance * 100.0, 1)

        world_traj_t = z0_full[0, :, -t_lat:].mean(dim=(2, 3)).T.float()
        entity_traj_128d = world_traj_t.unsqueeze(1).expand(t_lat, n_ent, c_lat).contiguous()
        entity_act_128d = (
            last_velocity_5d[0, :, -t_lat:].mean(dim=(2, 3)).T.float().unsqueeze(1).expand(t_lat, n_ent, c_lat).contiguous()
            if last_velocity_5d is not None and last_velocity_5d.shape[2] >= t_lat
            else entity_traj_128d
        )

        phys_explanation = (
            f"3D-DiT MULTI-BRANCH RF-ORACLE [{driver_tag} | sigma={self.probe_sigma:.2f}]: "
            f"MSE stay={rf_mse_ong:.4f} vs cur={rf_mse_cmp:.4f} | t={eff_t_stat:+.2f} | "
            f"win_stg={winning_branch_stage_idx+1}(t={winning_branch_t_stat:+.2f}) | "
            f"P(H+)={p_cmp_pct:.1f}% | dS={running_quest_prog:.1f}%"
        )

        high_dim_ctx = HighDimBattleContextTensor(
            entity_trajectories_128d=entity_traj_128d,
            entity_action_streams_128d=entity_act_128d,
            interaction_matrix_NxN=interaction_mat,
            entity_coherence_end=coherences_end,
            entity_work_done=work_done_end,
            entity_transition_pcts=transitions,
            entity_state_labels=labels,
            entity_choice_distributions=choice_dists,
            reaction_branch_scores=reaction_branch_scores,
            chosen_next_stage_idx=chosen_next_stage_idx,
            winning_branch_stage_idx=winning_branch_stage_idx,
            winning_branch_t_stat=winning_branch_t_stat,
            oracle_advantage=oracle_advantage,
            quest_progress_end=running_quest_prog,
            prob_completed_peak=p_cmp_pct,
            prob_ongoing_mean=p_ong_pct,
            latent_dominance_ratio=dom_ratio,
            feigenbaum_r=r_feigenbaum,
            catalytic_resonance_pct=cat_res_pct,
            chaos_Window_regime=chaos_regime,
            transformer_physics_explanation=phys_explanation,
            prediction_error_mse=rf_mse_cmp,
            rf_t_stat=eff_t_stat,
        )

        out_telemetry: list[LatentIdentityFrameTelemetry] = []
        for f_i in range(total_video_frames):
            t_sec = f_i / 16.0
            out_telemetry.append(
                LatentIdentityFrameTelemetry(
                    frame_idx=f_i,
                    time_sec=round(t_sec, 2),
                    phase_label=f"{driver_tag} | t={eff_t_stat:+.2f} | P(H+)={p_cmp_pct:.0f}%",
                    primary_coherence_pct=sub_coh,
                    quest_progress_pct=round(running_quest_prog, 1),
                    quest_remaining_pct=round(max(0.0, 100.0 - running_quest_prog), 1),
                    prob_completed_pct=round(p_cmp_pct, 1),
                    prob_ongoing_pct=round(p_ong_pct, 1),
                    semantic_flux_pct=round(abs(p_cmp_pct - p_ong_pct), 1),
                    matchup_tag=matchup_tag,
                    feigenbaum_r=r_feigenbaum,
                    catalytic_resonance_pct=cat_res_pct,
                    chaos_regime=chaos_regime,
                    entity_coherences=coherences_end,
                    entity_works=work_done_end,
                    entity_transition_pcts=transitions,
                    entity_state_labels=labels,
                    entity_choice_distributions=choice_dists,
                    reaction_branch_scores=reaction_branch_scores,
                )
            )

        return out_telemetry, high_dim_ctx
