#!/usr/bin/env python3
"""
neuro_controller.py — Полноранговый 96,000-D -> 524,288-D контроллер выбора без снижения размерности
====================================================================================================
- НИКАКОГО .mean(dim=0) по 5 частотным полосам: все 5 ритмов (Delta, Theta, Beta, Gamma, Ripples)
  масштаба региона разворачиваются в 5×32 = 160 слотов -> [128 × 4096] (524,288 измерений).
- Выбор состояния агента из allowed_pool и выбор следующего целевого состояния осуществляются
  напрямую через полноранговый тензорный резонанс в пространстве [K × 128 × 4096] (T5-XXL),
  сохраняя полный лог как сделанного выбора (★ CHOSEN), так и отвергнутых альтернатив (○ REJECTED).
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuro_world import ArchetypeSpec, StageEntitySlot
from neuro_vram_bus import HardwareVRAMBus

CORTICAL_REGION_REGISTRY = {
    "FCZ": (1, "FCz", 0.020, 0.22),
    "F3":  (2, "F3",  0.180, 0.72),
    "F4":  (3, "F4",  0.180, 0.72),
    "AFZ": (0, "AFz", 0.650, 0.88),
    "FPZ": (4, "Fpz", 2.500, 0.95),
}


class SymmetricBotController:

    def __init__(
        self,
        bot_id: str,
        archetypes: dict[str, ArchetypeSpec] | None = None,
        source_mode: str = "eeg:FCz",
    ):
        self.bot_id = bot_id
        self.archetypes = archetypes or {}
        self.source_mode = str(source_mode)

        self.vram_bus = HardwareVRAMBus()
        self.attempt_num: int = 1
        self.last_stage_id: int = -1

        self.phase_clock_5x32x120: torch.Tensor | None = None
        self.proj_120_to_4096: torch.Tensor | None = None
        self.baseline_wm_128x4096: torch.Tensor | None = None

    def _get_region_specs(self) -> tuple[int, str, float, float]:
        mode_clean = self.source_mode.upper()
        for k, val in CORTICAL_REGION_REGISTRY.items():
            if k in mode_clean:
                return val
        return CORTICAL_REGION_REGISTRY["FCZ"]

    @torch.no_grad()
    def sample_working_memory_5x5x32x120(self, device: torch.device) -> tuple[torch.Tensor, float, str]:
        reg_idx, reg_name, _, inertia = self._get_region_specs()

        if "eeg" in self.source_mode.lower() or self.source_mode.lower() == "auto":
            self.vram_bus.try_attach_to_existing_vram()
            if self.vram_bus.attached_pid > 0:
                wm_live = self.vram_bus.wm_field_5x5x32x120.to(device=device, dtype=torch.float32)
                if float(wm_live.abs().max().item()) > 1e-4:
                    return wm_live.clone(), inertia, f"EEG_VRAM:{reg_name}"

        s_grid = torch.linspace(0.0, 2.0 * math.pi, 32, device=device).view(1, 1, 32, 1)
        p_grid = torch.linspace(-math.pi, math.pi, 120, device=device).view(1, 1, 1, 120)
        r_grid = torch.arange(5, device=device, dtype=torch.float32).view(5, 1, 1, 1)
        b_grid = torch.arange(5, device=device, dtype=torch.float32).view(1, 5, 1, 1)

        if self.phase_clock_5x32x120 is None or self.phase_clock_5x32x120.device != device:
            gen_p = torch.Generator(device=device).manual_seed(1009 + reg_idx * 313 + self.attempt_num * 73)
            self.phase_clock_5x32x120 = torch.randn((1, 5, 32, 120), generator=gen_p, device=device) * 0.45
        else:
            # Каждая из 5 полос вращается со своей фазовой скоростью (без вырождения!):
            for b_i in range(5):
                self.phase_clock_5x32x120[:, b_i] = torch.roll(
                    self.phase_clock_5x32x120[:, b_i], shifts=(b_i + 1), dims=-1
                )

        wm_synth = torch.sin(
            s_grid * (1.0 + 0.22 * b_grid)
            + p_grid * (1.1 + 0.18 * r_grid)
            + self.phase_clock_5x32x120
        ) + 0.45 * torch.cos(
            s_grid * (0.7 - 0.15 * r_grid)
            - p_grid * (0.9 + 0.25 * b_grid)
            + self.attempt_num * (0.35 + 0.12 * b_grid)
        )
        weights_5 = torch.full((5, 1, 1, 1), 0.30, device=device, dtype=torch.float32)
        weights_5[reg_idx] = 3.0
        return (wm_synth * weights_5).contiguous(), inertia, f"SYNTH_WM:{reg_name}"

    @torch.no_grad()
    def unfold_region_wm_to_524288d(self, device: torch.device) -> tuple[torch.Tensor, float, str]:
        """
        Разворачивает все 19,200 мод текущего региона [5 полос × 32 слота × 120 диполей]
        плюс перекрёстный контекст остальных 4 регионов в полный тензор [128 × 4096] = 524,288-D
        БЕЗ .mean(dim=0) по частотным полосам!
        """
        wm_5x5x32x120, inertia, src_tag = self.sample_working_memory_5x5x32x120(device)
        reg_idx, _, _, _ = self._get_region_specs()

        if self.proj_120_to_4096 is None or self.proj_120_to_4096.device != device:
            g_proj = torch.Generator(device=device).manual_seed(999 + reg_idx * 137)
            raw_p = torch.randn((120, 4096), generator=g_proj, device=device, dtype=torch.float32)
            self.proj_120_to_4096 = F.normalize(raw_p, p=2, dim=0)

        # Берём все 5 частотных полос региона (5 * 32 = 160 векторов по 120 диполей) без усреднения полос:
        reg_160x120 = wm_5x5x32x120[reg_idx].reshape(160, 120)
        # Добавляем 20% контекста всего 96,000-D поля (если девайс один — все 5 полос и зон участвуют):
        global_160x120 = wm_5x5x32x120.mean(dim=0).reshape(160, 120)
        full_160x120 = reg_160x120 * 0.80 + global_160x120 * 0.20

        full_160x4096 = torch.matmul(full_160x120, self.proj_120_to_4096)
        wm_128x4096 = (
            F.interpolate(
                full_160x4096.T.unsqueeze(0), size=128, mode="linear", align_corners=False
            )
            .squeeze(0)
            .T.contiguous()
        )

        # Вычитаем медленный пьедестал, чтобы выделять живую инновацию мысли во всех 524,288 измерениях:
        if self.baseline_wm_128x4096 is None or self.baseline_wm_128x4096.shape != wm_128x4096.shape:
            self.baseline_wm_128x4096 = wm_128x4096.clone()
        else:
            self.baseline_wm_128x4096.lerp_(wm_128x4096, 0.08)

        dyn_128x4096 = wm_128x4096 - self.baseline_wm_128x4096 * 0.75
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
    ) -> tuple[str, str, torch.Tensor, dict[str, float]]:
        """
        ПОЛНОРАНГОВЫЙ ВЫБОР КОНЦЕПТА И ЦЕЛЕВОГО ПЕРЕХОДА В 524,288-D ПРОСТРАНСТВЕ T5:
        Сравнивает 524,288-мерный тензор рабочей памяти агента [128 × 4096] со всеми K
        кандидатами из allowed_pool [K × 128 × 4096] без снижения размерности!
        Возвращает:
          - start_state_id (текущее состояние)
          - target_state_id (выбранная рабочей памятью ветка перехода, отличная от start_state_id)
          - steered_target_emb_1x128x4096 (повёрнутый в 524,288-D целевой эмбеддинг)
          - choice_probabilities_dict (вероятности ВСЕХ альтернатив для дебага выборов!)
        """
        K = len(pool_ids)
        if K == 0:
            return "", "", candidate_embeds_Kx128x4096[:1], {}

        wm_128x4096, inertia, _ = self.unfold_region_wm_to_524288d(device)
        bank_float = candidate_embeds_Kx128x4096.to(device=device, dtype=torch.float32)

        if K == 1:
            only_id = pool_ids[0]
            steered = self.steer_semantic_probe_full_rank(bank_float[0:1], device=device, wm_override_128x4096=wm_128x4096)
            return only_id, only_id, steered, {only_id: 100.0}

        # Центрируем банк кандидатов в 524,288-мерном пространстве [K, 524288]:
        bank_flat = bank_float.reshape(K, -1)
        bank_delta = bank_flat - bank_flat.mean(dim=0, keepdim=True)
        bank_unit = F.normalize(bank_delta, p=2, dim=-1)

        wm_flat = wm_128x4096.reshape(1, -1)
        wm_unit = F.normalize(wm_flat - wm_flat.mean(dim=-1, keepdim=True), p=2, dim=-1)

        # Полноранговое 524,288-мерное косинусное сходство Фробениуса:
        raw_sims = torch.mm(wm_unit, bank_unit.T).squeeze(0)
        sims_std = raw_sims.std(unbiased=False).clamp(min=1e-5)
        logits_K = (raw_sims - raw_sims.mean()) / sims_std

        probs_K = torch.softmax(logits_K / 0.45, dim=0)
        choice_Order = torch.argsort(probs_K, descending=True).tolist()

        # Формируем полный словарь вероятностей всех выборов для прозрачного дебага:
        choice_probs_dict = {
            pool_ids[idx]: round(float(probs_K[idx].item()) * 100.0, 1)
            for idx in range(K)
        }

        # Определяем стартовое состояние и выбранное рабочей памятью целевое состояние:
        if current_state_id and current_state_id in pool_ids:
            start_id = current_state_id
        else:
            start_id = pool_ids[choice_Order[0]]

        # Целевое состояние — лучший выбор рабочей памяти, отличный от текущего состояния:
        target_idx = choice_Order[0]
        if pool_ids[target_idx] == start_id and K > 1:
            target_idx = choice_Order[1]
        target_id = pool_ids[target_idx]

        # Вращаем выбранный эмбеддинг в касательном подпространстве всех K альтернатив:
        steered_target = self.steer_semantic_probe_full_rank(
            bank_float[target_idx : target_idx + 1],
            device=device,
            wm_override_128x4096=wm_128x4096,
            all_candidates_Kx128x4096=bank_float,
            candidate_weights_K=probs_K,
        )
        return start_id, target_id, steered_target, choice_probs_dict

    @torch.no_grad()
    def steer_semantic_probe_full_rank(
        self,
        base_probe_1x128x4096: torch.Tensor,
        device: torch.device,
        wm_override_128x4096: torch.Tensor | None = None,
        all_candidates_Kx128x4096: torch.Tensor | None = None,
        candidate_weights_K: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        524,288-D геодезическое вращение эмбеддинга на сфере S^4095 каждого из 128 токенов
        с учётом натяжения всех альтернативных концептов пула агента.
        """
        if wm_override_128x4096 is not None:
            wm_128x4096 = wm_override_128x4096
            _, _, _, inertia = self._get_region_specs()
        else:
            wm_128x4096, inertia, _ = self.unfold_region_wm_to_524288d(device)

        base_tokens = base_probe_1x128x4096[0].float()
        ref_norms = torch.norm(base_tokens, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        # Если у агента есть альтернативные концепты в пуле, подмешиваем их касательное натяжение:
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

        theta_rad = (1.0 - inertia) * (math.pi * 0.09)
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
