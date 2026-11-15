#!/usr/bin/env python3
"""
neuro_controller.py — Полноранговый 96,000-D -> 524,288-D контроллер рабочей памяти
===================================================================================
- На 100% сохранена чистая формула геодезического поворота T5 без межкадрового размытия (.lerp_),
  дающая идеальные горы, скалы и кристаллы.
- Добавлен параметр sub_step_frac для мгновенного опроса живой шины ЭЭГ на каждом шаге денойзинга 3D-DiT!
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuro_world import ArchetypeSpec, StageEntitySlot
from neuro_vram_bus import HardwareVRAMBus

CORTICAL_REGION_REGISTRY = {
    "FCZ": (1, "FCz", 0.020, 0.22),
    "F3":  (2, "F3",  0.180, 0.55),
    "F4":  (3, "F4",  0.180, 0.55),
    "AFZ": (0, "AFz", 0.650, 0.68),
    "FPZ": (4, "Fpz", 2.500, 0.78),
}


class SymmetricBotController:

    def __init__(
        self,
        bot_id: str,
        archetypes: dict[str, ArchetypeSpec] | None = None,
        source_mode: str = "eeg:FCz",
        stochastic: float = 0.0,
        wm_radius: float = 0.35,
    ):
        self.bot_id = bot_id
        self.archetypes = archetypes or {}
        self.source_mode = str(source_mode)
        self.stochastic = float(stochastic)
        self.wm_radius = float(wm_radius)

        self.vram_bus = HardwareVRAMBus()
        self.attempt_num: int = 1
        self.last_stage_id: int = -1

        self.phase_clock_5x32x120: torch.Tensor | None = None
        self.proj_120_to_4096: torch.Tensor | None = None
        self.proj_600_to_4096: torch.Tensor | None = None
        self.baseline_wm_128x4096: torch.Tensor | None = None
        self.latest_region_wm_19200: torch.Tensor | None = None

    def _get_region_specs(self) -> tuple[int, str, float, float]:
        mode_clean = self.source_mode.upper()
        for k, val in CORTICAL_REGION_REGISTRY.items():
            if k in mode_clean:
                return val
        return CORTICAL_REGION_REGISTRY["FCZ"]

    @torch.no_grad()
    def sample_working_memory_5x5x32x120(
        self,
        device: torch.device,
        sub_step_frac: float = 0.0,
    ) -> tuple[torch.Tensor, float, str]:
        reg_idx, reg_name, _, inertia = self._get_region_specs()

        if "eeg" in self.source_mode.lower() or self.source_mode.lower() == "auto":
            self.vram_bus.try_attach_to_existing_vram()
            if self.vram_bus.attached_pid > 0:
                wm_live = self.vram_bus.wm_field_5x5x32x120.to(device=device, dtype=torch.float32)
                if float(wm_live.abs().max().item()) > 1e-4:
                    self.latest_region_wm_19200 = wm_live[reg_idx].reshape(-1).clone()
                    return wm_live.clone(), inertia, f"EEG_VRAM:{reg_name}"

        s_grid = torch.linspace(0.0, 2.0 * math.pi, 32, device=device).view(1, 1, 32, 1)
        p_grid = torch.linspace(-math.pi, math.pi, 120, device=device).view(1, 1, 1, 120)
        r_grid = torch.arange(5, device=device, dtype=torch.float32).view(5, 1, 1, 1)
        b_grid = torch.arange(5, device=device, dtype=torch.float32).view(1, 5, 1, 1)

        if self.phase_clock_5x32x120 is None or self.phase_clock_5x32x120.device != device:
            gen_p = torch.Generator(device=device).manual_seed(1009 + reg_idx * 313)
            self.phase_clock_5x32x120 = torch.randn((1, 5, 32, 120), generator=gen_p, device=device) * 0.45
        elif sub_step_frac <= 0.001:
            for b_i in range(5):
                self.phase_clock_5x32x120[:, b_i] = torch.roll(
                    self.phase_clock_5x32x120[:, b_i], shifts=(b_i + 1), dims=-1
                )

        eff_t = float(self.attempt_num) + float(sub_step_frac)
        phase_t = eff_t * (0.45 + 0.35 * max(0.15, self.wm_radius))
        wm_synth = torch.sin(
            s_grid * (1.0 + 0.22 * b_grid)
            + p_grid * (1.1 + 0.18 * r_grid)
            + self.phase_clock_5x32x120
            + phase_t * (0.28 + 0.09 * r_grid)
        ) + 0.45 * torch.cos(
            s_grid * (0.7 - 0.15 * r_grid)
            - p_grid * (0.9 + 0.25 * b_grid)
            + phase_t * (0.35 + 0.12 * b_grid)
        )
        weights_5 = torch.full((5, 1, 1, 1), 0.30, device=device, dtype=torch.float32)
        weights_5[reg_idx] = 3.0
        out_wm = (wm_synth * weights_5).contiguous()
        self.latest_region_wm_19200 = out_wm[reg_idx].reshape(-1).clone()
        return out_wm, inertia, f"SYNTH_WM:{reg_name}"

    @torch.no_grad()
    def unfold_region_wm_to_524288d(
        self,
        device: torch.device,
        sub_step_frac: float = 0.0,
    ) -> tuple[torch.Tensor, float, str]:
        wm_5x5x32x120, inertia, src_tag = self.sample_working_memory_5x5x32x120(
            device, sub_step_frac=sub_step_frac
        )
        reg_idx, _, _, _ = self._get_region_specs()

        if self.proj_120_to_4096 is None or self.proj_120_to_4096.device != device:
            g_proj = torch.Generator(device=device).manual_seed(999 + reg_idx * 137)
            raw_p = torch.randn((120, 4096), generator=g_proj, device=device, dtype=torch.float32)
            self.proj_120_to_4096 = F.normalize(raw_p, p=2, dim=0)

        if self.proj_600_to_4096 is None or self.proj_600_to_4096.device != device:
            g_proj_all = torch.Generator(device=device).manual_seed(7777 + reg_idx * 251)
            raw_p600 = torch.randn((600, 4096), generator=g_proj_all, device=device, dtype=torch.float32)
            self.proj_600_to_4096 = F.normalize(raw_p600, p=2, dim=0)

        reg_160x120 = wm_5x5x32x120[reg_idx].reshape(160, 120)
        reg_160x4096 = torch.matmul(reg_160x120, self.proj_120_to_4096)

        all_96k_160x600 = wm_5x5x32x120.permute(1, 2, 0, 3).reshape(160, 600)
        all_96k_160x4096 = torch.matmul(all_96k_160x600, self.proj_600_to_4096)

        full_160x4096 = reg_160x4096 * 0.80 + all_96k_160x4096 * 0.20

        wm_128x4096 = (
            F.interpolate(
                full_160x4096.T.unsqueeze(0), size=128, mode="linear", align_corners=False
            )
            .squeeze(0)
            .T.contiguous()
        )

        if self.baseline_wm_128x4096 is None or self.baseline_wm_128x4096.shape != wm_128x4096.shape:
            self.baseline_wm_128x4096 = wm_128x4096.clone()
        else:
            self.baseline_wm_128x4096.lerp_(wm_128x4096, 0.06)

        dyn_128x4096 = wm_128x4096 - self.baseline_wm_128x4096 * 0.70
        std_val = dyn_128x4096.std(dim=-1, keepdim=True).clamp(min=1e-5)
        whitened_128x4096 = (dyn_128x4096 - dyn_128x4096.mean(dim=-1, keepdim=True)) / std_val
        return whitened_128x4096, inertia, src_tag

    @torch.no_grad()
    def select_concept_full_rank_524288d(
        self,
        pool_ids: list[str],
        candidate_embeds_Kx128x4096: torch.Tensor,
        device: torch.device,
        current_state_id: str | None = None,
        catalyst_field_128x4096: torch.Tensor | None = None,
    ) -> tuple[str, str, torch.Tensor, dict[str, float]]:
        K = len(pool_ids)
        if K == 0:
            return "", "", candidate_embeds_Kx128x4096[:1], {}

        wm_128x4096, inertia, _ = self.unfold_region_wm_to_524288d(device, sub_step_frac=0.0)
        bank_float = candidate_embeds_Kx128x4096.to(device=device, dtype=torch.float32)

        if K == 1:
            only_id = pool_ids[0]
            eff_wm = (
                wm_128x4096 * 0.65 + catalyst_field_128x4096.to(device=device, dtype=torch.float32) * 0.35
                if catalyst_field_128x4096 is not None
                else wm_128x4096
            )
            steered = self.steer_semantic_probe_full_rank(bank_float[0:1], device=device, wm_override_128x4096=eff_wm)
            return only_id, only_id, steered, {only_id: 100.0}

        bank_flat = bank_float.reshape(K, -1)
        bank_delta = bank_flat - bank_flat.mean(dim=0, keepdim=True)
        bank_unit = F.normalize(bank_delta, p=2, dim=-1)

        wm_flat = wm_128x4096.reshape(1, -1)
        wm_unit = F.normalize(wm_flat - wm_flat.mean(dim=-1, keepdim=True), p=2, dim=-1)
        own_sims = torch.mm(wm_unit, bank_unit.T).squeeze(0)
        own_best_id = pool_ids[int(torch.argmax(own_sims).item())]

        if catalyst_field_128x4096 is not None:
            cat_f = catalyst_field_128x4096.to(device=device, dtype=torch.float32)
            cat_flat = cat_f.reshape(1, -1)
            cat_unit = F.normalize(cat_flat - cat_flat.mean(dim=-1, keepdim=True), p=2, dim=-1)
            combined_unit = F.normalize(wm_unit * 0.45 + cat_unit * 0.55, p=2, dim=-1)
            raw_sims = torch.mm(combined_unit, bank_unit.T).squeeze(0)
            cat_whitened = (cat_f - cat_f.mean(dim=-1, keepdim=True)) / cat_f.std(dim=-1, keepdim=True).clamp(min=1e-5)
            eff_wm_128x4096 = wm_128x4096 * 0.50 + cat_whitened * 0.50
        else:
            raw_sims = own_sims
            eff_wm_128x4096 = wm_128x4096

        sims_std = raw_sims.std(unbiased=False).clamp(min=1e-5)
        logits_K = (raw_sims - raw_sims.mean()) / sims_std

        probs_K = torch.softmax(logits_K / 0.45, dim=0)
        choice_Order = torch.argsort(probs_K, descending=True).tolist()

        choice_probs_dict = {
            pool_ids[idx]: round(float(probs_K[idx].item()) * 100.0, 1)
            for idx in range(K)
        }

        start_id = current_state_id if (current_state_id and current_state_id in pool_ids) else own_best_id
        top_idx = choice_Order[0]
        top_id = pool_ids[top_idx]

        steered_target = self.steer_semantic_probe_full_rank(
            bank_float[top_idx : top_idx + 1],
            device=device,
            wm_override_128x4096=eff_wm_128x4096,
            all_candidates_Kx128x4096=bank_float,
            candidate_weights_K=probs_K,
        )
        return start_id, top_id, steered_target, choice_probs_dict

    @torch.no_grad()
    def steer_semantic_probe_full_rank(
        self,
        base_probe_1x128x4096: torch.Tensor,
        device: torch.device,
        wm_override_128x4096: torch.Tensor | None = None,
        all_candidates_Kx128x4096: torch.Tensor | None = None,
        candidate_weights_K: torch.Tensor | None = None,
        sub_step_frac: float = 0.0,
    ) -> torch.Tensor:
        if self.wm_radius <= 0.001 or self.stochastic >= 0.999:
            return base_probe_1x128x4096

        if wm_override_128x4096 is not None:
            wm_128x4096 = wm_override_128x4096
            _, _, _, inertia = self._get_region_specs()
        else:
            wm_128x4096, inertia, _ = self.unfold_region_wm_to_524288d(
                device, sub_step_frac=sub_step_frac
            )

        base_tokens = base_probe_1x128x4096[0].float()
        ref_norms = torch.norm(base_tokens, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        if all_candidates_Kx128x4096 is not None and candidate_weights_K is not None and all_candidates_Kx128x4096.shape[0] > 1:
            weighted_alt = torch.sum(
                all_candidates_Kx128x4096.float() * candidate_weights_K.view(-1, 1, 1), dim=0
            )
            alt_delta = weighted_alt - base_tokens
            wm_zero = wm_128x4096 * 0.70 + F.normalize(alt_delta, p=2, dim=-1) * 0.30
        else:
            wm_zero = wm_128x4096 - wm_128x4096.mean(dim=0, keepdim=True)

        dot_proj = torch.sum(wm_zero * base_tokens, dim=-1, keepdim=True) / (ref_norms ** 2 + 1e-6)
        tangent_raw = wm_zero - dot_proj * base_tokens
        tangent_unit = (tangent_raw / (torch.norm(tangent_raw, p=2, dim=-1, keepdim=True) + 1e-6)) * ref_norms

        theta_rad = (1.0 - inertia * 0.5) * (math.pi * 0.14) * float(self.wm_radius) * (1.0 - float(self.stochastic))
        steered_tokens = math.cos(theta_rad) * base_tokens + math.sin(theta_rad) * tangent_unit
        steered_normalized = F.normalize(steered_tokens, p=2, dim=-1) * ref_norms

        return steered_normalized.unsqueeze(0).to(base_probe_1x128x4096.dtype)

    def select_party(
        self,
        global_stage: int,
        my_cfg: StageEntitySlot,
    ) -> list[str]:
        if global_stage != self.last_stage_id:
            self.last_stage_id = global_stage
            self.attempt_num = 1
        pool = my_cfg.allowed_pool if my_cfg.allowed_pool else list(self.archetypes.keys())
        if not pool:
            return []
        return list(pool)
