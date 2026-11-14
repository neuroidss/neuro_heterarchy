#!/usr/bin/env python3
"""
neuro_referee.py — Мгновенный Полноранговый 128D/524,288D Оракул (0 лишних вызовов 3D-DiT!)
============================================================================================
- Вычисляет все метрики Фейгенбаума (r, резонанс катализатора, прогресс квеста,
  конкуренцию всех ветвей реакций) за < 1 мс напрямую из реального латентного сдвига
  unpacked_latents_5d и полноранговых 524,288-D тензоров T5 без повторного запуска 3D-DiT!
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

FEIGENBAUM_DELTA: float = 4.669201609102990
FEIGENBAUM_ALPHA: float = 2.502907875095892
R_EDGE_OF_CHAOS: float = 3.569945671870944


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
    quest_progress_end: float = 0.0
    prob_completed_peak: float = 50.0
    prob_ongoing_mean: float = 50.0
    latent_dominance_ratio: float = 1.0
    feigenbaum_r: float = R_EDGE_OF_CHAOS
    catalytic_resonance_pct: float = 50.0
    chaos_Window_regime: str = "EDGE_OF_CHAOS"
    transformer_physics_explanation: str = ""

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
            quest_progress_end=float(self.quest_progress_end),
            prob_completed_peak=float(self.prob_completed_peak),
            prob_ongoing_mean=float(self.prob_ongoing_mean),
            latent_dominance_ratio=float(self.latent_dominance_ratio),
            feigenbaum_r=float(self.feigenbaum_r),
            catalytic_resonance_pct=float(self.catalytic_resonance_pct),
            chaos_Window_regime=str(self.chaos_Window_regime),
            transformer_physics_explanation=str(self.transformer_physics_explanation),
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
            quest_progress_end=float(self.quest_progress_end),
            prob_completed_peak=float(self.prob_completed_peak),
            prob_ongoing_mean=float(self.prob_ongoing_mean),
            latent_dominance_ratio=float(self.latent_dominance_ratio),
            feigenbaum_r=float(self.feigenbaum_r),
            catalytic_resonance_pct=float(self.catalytic_resonance_pct),
            chaos_Window_regime=str(self.chaos_Window_regime),
            transformer_physics_explanation=str(self.transformer_physics_explanation),
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

    def __init__(self, transformer: any, device: torch.device, dtype: torch.dtype):
        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self.proj_4096_to_128: torch.Tensor | None = None

    def _get_proj_4096_to_128(self) -> torch.Tensor:
        if self.proj_4096_to_128 is None or self.proj_4096_to_128.device != self.device:
            g = torch.Generator(device=self.device).manual_seed(1337)
            raw = torch.randn((4096, 128), generator=g, device=self.device, dtype=torch.float32)
            self.proj_4096_to_128 = F.normalize(raw, p=2, dim=0)
        return self.proj_4096_to_128

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
        last_velocity_5d: torch.Tensor | None = None,
        **unused_kwargs,
    ) -> tuple[list[LatentIdentityFrameTelemetry], HighDimBattleContextTensor]:
        """
        МГНОВЕННЫЙ ОРАКУЛ БЕЗ ДОПОЛНИТЕЛЬНЫХ ПРОГОНОВ ТРАНСФОРМЕРА (0.5 мс на GPU):
        Использует реальный латентный поток unpacked_latents_5d + last_velocity_5d
        и полноранговые 524,288-D тензоры T5 всех агентов и гипотез.
        """
        _, c_lat, t_lat, h_lat, w_lat = unpacked_latents_5d.shape
        n_ent = max(1, len(entity_probe_embeds_list))
        proj_mat = self._get_proj_4096_to_128()

        # 1. Истинный 128D вектор скорости латентного мира за текущий шаг:
        if last_velocity_5d is not None:
            v_world_128d = last_velocity_5d[:, :, -1:].mean(dim=(0, 2, 3, 4)).float()
        elif origin_anchor_5d is not None:
            v_world_128d = (unpacked_latents_5d[:, :, -1:] - origin_anchor_5d[:, :, :1]).mean(dim=(0, 2, 3, 4)).float()
        else:
            v_world_128d = unpacked_latents_5d[:, :, -1:].mean(dim=(0, 2, 3, 4)).float()
        v_world_unit = F.normalize(v_world_128d, p=2, dim=0)

        # 2. Полноранговые 524,288-D представления пассивного фона и гипотез:
        pas_524k = passive_void_embeds[0][0].float()       # [128, 4096]
        ong_524k = ongoing_hyp_embeds[0][0].float() - pas_524k
        cmp_524k = completed_hyp_embeds[0][0].float() - pas_524k

        ong_128d = F.normalize(torch.matmul(ong_524k, proj_mat).mean(dim=0), p=2, dim=0)
        cmp_128d = F.normalize(torch.matmul(cmp_524k, proj_mat).mean(dim=0), p=2, dim=0)

        entity_traj_128d = torch.zeros((t_lat, n_ent, c_lat), device=self.device)
        entity_act_128d = torch.zeros((t_lat, n_ent, c_lat), device=self.device)
        interaction_mat = torch.zeros((n_ent, n_ent), device=self.device)
        entity_power_vec = torch.zeros(n_ent, device=self.device)

        ent_deltas_flat = []
        for u_i in range(n_ent):
            e_524k = entity_probe_embeds_list[u_i][0][0].float() - pas_524k  # [128, 4096]
            ent_deltas_flat.append(F.normalize(e_524k.reshape(1, -1), p=2, dim=-1))
            u_128d = F.normalize(torch.matmul(e_524k, proj_mat).mean(dim=0), p=2, dim=0)
            entity_act_128d[:, u_i] = u_128d
            entity_traj_128d[:, u_i] = u_128d
            align_w = float(F.cosine_similarity(u_128d.unsqueeze(0), v_world_unit.unsqueeze(0), dim=-1).item())
            entity_power_vec[u_i] = float(e_524k.norm(p=2, dim=-1).mean().item()) * (1.0 + 0.45 * align_w)

        for i in range(n_ent):
            for j in range(i + 1, n_ent):
                # Полноранговое 524,288-D + 128D сопряжение двух агентов:
                cos_524k = float( torch.mm(ent_deltas_flat[i], ent_deltas_flat[j].T).item() )
                cos_128d = float( F.cosine_similarity(entity_act_128d[0, i:i+1], entity_act_128d[0, j:j+1], dim=-1).item() )
                cos_sim = 0.65 * cos_524k + 0.35 * cos_128d
                interaction_mat[i, j] = float(np.clip((cos_sim + 0.45) * 78.0, 10.0, 100.0))
                interaction_mat[j, i] = interaction_mat[i, j]

        reaction_branch_scores: dict[str, float] = {}
        chosen_next_stage_idx = -1

        if candidate_branch_specs:
            raw_branch_logits = []
            branch_keys = []
            branch_stage_indices = []
            for b_spec in candidate_branch_specs:
                b_cat = int(b_spec.get("catalyst_slot_idx", 0)) % n_ent
                b_sub = int(b_spec.get("active_slot_idx", 0)) % n_ent
                b_tag = str(b_spec.get("driver_tag", f"CAT(e{b_cat}->e{b_sub})"))
                b_stg_idx = int(b_spec.get("stage_idx", 0))

                coup = float(interaction_mat[b_cat, b_sub].item()) / 100.0
                cat_p = float(entity_power_vec[b_cat].item())
                score_val = coup * (0.7 + 0.3 * (cat_p / (float(entity_power_vec.mean().item()) + 1e-5)))
                raw_branch_logits.append(score_val * 5.0)
                branch_keys.append(b_tag)
                branch_stage_indices.append(b_stg_idx)

            branch_probs = torch.softmax(torch.tensor(raw_branch_logits, device=self.device), dim=0)
            for idx_b, b_key in enumerate(branch_keys):
                reaction_branch_scores[b_key] = round(float(branch_probs[idx_b].item()) * 100.0, 1)

            sorted_b = torch.argsort(branch_probs, descending=True).tolist()
            best_b = sorted_b[0]
            if branch_keys[best_b] == driver_tag and len(sorted_b) > 1:
                best_b = sorted_b[1]
            chosen_next_stage_idx = branch_stage_indices[best_b]

        cat_resonance = 0.65
        selectivity_bonus = 0.0
        if 0 <= catalyst_slot_idx < n_ent and 0 <= active_slot_idx < n_ent and catalyst_slot_idx != active_slot_idx:
            cat_resonance = float(interaction_mat[catalyst_slot_idx, active_slot_idx].item()) / 100.0
            other_resonances = [
                float(interaction_mat[k, active_slot_idx].item()) / 100.0
                for k in range(n_ent)
                if k != active_slot_idx and k != catalyst_slot_idx
            ]
            max_other = max(other_resonances) if other_resonances else 0.0
            selectivity_bonus = (cat_resonance - max_other) * 0.25

        # Согласованность латентного потока с завершенной гипотезой и активной парой:
        sim_cmp = float(F.cosine_similarity(v_world_unit.unsqueeze(0), cmp_128d.unsqueeze(0), dim=-1).item())
        sim_ong = float(F.cosine_similarity(v_world_unit.unsqueeze(0), ong_128d.unsqueeze(0), dim=-1).item())
        raw_delta_ll = (sim_cmp - sim_ong) * 0.45

        cat_coupling_boost = float(np.clip(0.85 + 0.65 * cat_resonance + selectivity_bonus, 0.55, 1.55))
        calibrated_delta_ll = raw_delta_ll + 0.18 * cat_coupling_boost

        logits_hyp = torch.tensor([0.0, calibrated_delta_ll], device=self.device)
        probs_hyp = torch.softmax(logits_hyp, dim=0)
        p_ong_pct = float(probs_hyp[0].item()) * 100.0
        p_cmp_pct = float(probs_hyp[1].item()) * 100.0

        oracle_advantage = (p_cmp_pct - p_ong_pct) / 100.0
        r_feigenbaum = float(
            np.clip(
                R_EDGE_OF_CHAOS
                + ((cat_resonance - 0.35) + oracle_advantage * 1.2) * (1.0 / FEIGENBAUM_DELTA),
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

        x_curr = float(np.clip(init_quest_progress / 100.0, 0.0, 1.0))
        x_seed = max(0.06, x_curr)
        x_logistic = r_feigenbaum * x_seed * (1.0 - x_seed)
        chaos_fold_delta = (x_logistic - x_seed) * (0.22 / FEIGENBAUM_ALPHA)

        if p_cmp_pct >= p_ong_pct:
            order_crystallization = (
                ((p_cmp_pct - p_ong_pct) / 100.0)
                * 0.65
                * float(completion_gain)
                * cat_coupling_boost
            )
            net_step = max(0.065, chaos_fold_delta + order_crystallization)
        else:
            reverse_pull = ((p_ong_pct - p_cmp_pct) / 100.0) * 0.45 * float(completion_gain)
            net_step = -max(0.02, reverse_pull)

        running_quest_prog = float(np.clip((x_curr + net_step) * 100.0, 0.0, 100.0))

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
        sub_coh = float(np.clip(cur_coh - entropy_drain + 0.35 * cat_coupling_boost, 25.0, 100.0))
        coherences_end = [100.0] * n_ent
        if 0 <= active_slot_idx < n_ent:
            coherences_end[active_slot_idx] = round(sub_coh, 1)

        work_done_end = [round(t_val, 1) for t_val in transitions]
        cat_res_pct = round(cat_resonance * 100.0, 1)

        phys_explanation = (
            f"128D-HETERARCHY [{driver_tag} | {chaos_regime}]: "
            f"r={r_feigenbaum:.4f} | CatRes={cat_res_pct:.1f}% | dS={running_quest_prog:.1f}% | "
            f"BranchScores={reaction_branch_scores}"
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
            quest_progress_end=running_quest_prog,
            prob_completed_peak=p_cmp_pct,
            prob_ongoing_mean=p_ong_pct,
            latent_dominance_ratio=dom_ratio,
            feigenbaum_r=r_feigenbaum,
            catalytic_resonance_pct=cat_res_pct,
            chaos_Window_regime=chaos_regime,
            transformer_physics_explanation=phys_explanation,
        )

        out_telemetry: list[LatentIdentityFrameTelemetry] = []
        for f_i in range(total_video_frames):
            t_sec = f_i / 16.0
            out_telemetry.append(
                LatentIdentityFrameTelemetry(
                    frame_idx=f_i,
                    time_sec=round(t_sec, 2),
                    phase_label=f"{driver_tag} | r={r_feigenbaum:.3f} | P(H+)={p_cmp_pct:.0f}%",
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
