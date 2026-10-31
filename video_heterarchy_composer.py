#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: UNIFIED ZERO-REDUCTION COMPOSER FOR 3D-DiT (video_heterarchy_composer.py)
===================================================================================
- ОДНА УНИВЕРСАЛЬНАЯ ОПЕРАЦИЯ В ПОЛНОМ ПРОСТРАНСТВЕ [128 × 4096] = 524,288 ИЗМЕРЕНИЙ
  НЕЗАВИСИМО ОТ ЧИСЛА КОНЦЕПТОВ (0, 1, 2, ..., ∞).
- УНИВЕРСАЛЬНЫЙ ПАРСЕР ЛОРА:
  Если в лор-файле не указаны заголовки регионов ([FPZ], [AFZ], [F3], [F4], [FCZ]),
  все строки автоматически назначаются сразу НА ВСЕ 5 РЕГИОНОВ!
===================================================================================
"""

from __future__ import annotations
import os
import zlib
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE
from neuro_amodal_engine import REGION_ROLES
from video_render_logic import NeuroVideoRenderer

REGION_TOKEN_SLICES = {
    "Fpz": (0, 25),
    "AFz": (25, 50),
    "F3":  (50, 75),
    "F4":  (75, 100),
    "FCz": (100, 128),
}
CORE_REGIONS = ("FPZ", "AFZ", "F3", "F4", "FCZ")


def normalize_lore_path(p: str | None) -> str:
    if not p or not str(p).strip():
        return ""
    s = str(p).strip()
    if s.startswith("/app/"):
        s = s[len("/app/"):]
    return s


class RegionalVideoHeterarchyComposer:
    def __init__(self, renderer: NeuroVideoRenderer, lore_path: str = ""):
        self.renderer = renderer
        self.device = DEVICE
        self.lore_path = normalize_lore_path(lore_path)

        self.current_lore_hash = None
        self.sections: dict[str, list[str]] = {
            "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
        }

        self.region_full_banks: dict[str, torch.Tensor] = {}
        self.region_labels: dict[str, list[str]] = {}
        self.holistic_scene_cache: dict[tuple[int, ...], torch.Tensor] = {}

        self.empty_embed_128x4096: torch.Tensor | None = None
        self.neg_embed_1x128x4096: torch.Tensor | None = None
        self.smoothed_scene_128x4096: torch.Tensor | None = None

        self.running_wm_baseline_128x4096 = torch.zeros((128, 4096), dtype=torch.float32, device=self.device)
        self.baseline_initialized = False

        self.smoothed_logits: dict[str, torch.Tensor] = {}
        self.active_concept_idx = {r: 0 for r in REGION_ROLES}
        self.fpz_meta_charge = 0.0
        self.afz_stage_progress = 0.0

        self.cascade_rate = 1.0
        self.psi_threshold = 0.55
        self.wm_modulation_gain = 0.24

        initial_sec = self._read_lore_file_sections()
        self.sync_lore_sections(initial_sec)

    def _read_lore_file_sections(self) -> dict[str, list[str]]:
        secs: dict[str, list[str]] = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
        candidates = [self.lore_path, f"/app/{self.lore_path}"] if self.lore_path else []
        unassigned_global = []

        for c_path in candidates:
            if c_path and os.path.isfile(c_path):
                cur = None
                with open(c_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        if s.startswith("[") and s.endswith("]"):
                            cur = s[1:-1].strip().upper()
                            if cur not in secs:
                                secs[cur] = []
                        else:
                            if cur is None:
                                unassigned_global.append(s)
                            else:
                                secs[cur].append(s)
                break

        # Если регионы в файле не выбраны — ставим этот лор сразу на все 5 регионов!
        if unassigned_global:
            for r_k in CORE_REGIONS:
                if not secs[r_k]:
                    secs[r_k] = list(unassigned_global)

        non_empty = next((secs[r] for r in CORE_REGIONS if secs[r]), [])
        if non_empty:
            for r_k in CORE_REGIONS:
                if not secs[r_k]:
                    secs[r_k] = list(non_empty)

        return secs

    @torch.no_grad()
    def sync_lore_sections(self, lore_sections: dict[str, list[str]] | None):
        if lore_sections is None:
            lore_sections = self._read_lore_file_sections()
        else:
            # Если извне передали словарь, где часть регионов не заполнена — раздаем непустые концепты на все регионы
            non_empty = next((lore_sections.get(r, []) for r in CORE_REGIONS if lore_sections.get(r)), [])
            if non_empty:
                lore_sections = dict(lore_sections)
                for r_k in CORE_REGIONS:
                    if not lore_sections.get(r_k):
                        lore_sections[r_k] = list(non_empty)

        serialized = repr(sorted((k, v) for k, v in lore_sections.items()))
        new_hash = zlib.crc32(serialized.encode("utf-8"))
        if self.current_lore_hash == new_hash and self.neg_embed_1x128x4096 is not None:
            return

        self.holistic_scene_cache.clear()
        if self.empty_embed_128x4096 is None:
            self.empty_embed_128x4096 = self.renderer.encode_holistic_128_t5("")

        for role in REGION_ROLES:
            sec_k = role.upper()
            lines = [s.strip() for s in lore_sections.get(sec_k, []) if s.strip()]
            self.sections[sec_k] = lines

            prompts_to_encode = lines if lines else [""]
            self.region_labels[role] = lines if lines else ["MEAN_VACUUM"]
            bank = torch.stack([self.renderer.encode_holistic_128_t5(p) for p in prompts_to_encode], dim=0)
            self.region_full_banks[role] = bank
            self.smoothed_logits[role] = torch.zeros(bank.shape[0], dtype=torch.float32, device=self.device)

        neg_lines = [s.strip() for s in lore_sections.get("NEGATIVE", []) if s.strip()]
        neg_text = " ".join(neg_lines)
        self.neg_embed_1x128x4096 = self.renderer.encode_holistic_128_t5(neg_text).unsqueeze(0)
        self.current_lore_hash = new_hash

        base_scene = self._get_holistic_scene_embedding((0, 0, 0, 0, 0))
        if self.smoothed_scene_128x4096 is None:
            self.smoothed_scene_128x4096 = base_scene.clone()

        print(
            "📜 [UNIFIED 524,288-D COMPOSER] Точек по регионам: " +
            ", ".join(f"{r}:{self.region_full_banks[r].shape[0]}" for r in REGION_ROLES)
        )

    @torch.no_grad()
    def _get_holistic_scene_embedding(self, idx_tuple: tuple[int, int, int, int, int]) -> torch.Tensor:
        if idx_tuple in self.holistic_scene_cache:
            return self.holistic_scene_cache[idx_tuple]

        ordered_roles = ["Fpz", "AFz", "F3", "F4", "FCz"]
        prompt_parts = []
        for r_i, role in enumerate(ordered_roles):
            lines = self.sections.get(role.upper(), [])
            if lines:
                c_idx = idx_tuple[r_i] % len(lines)
                if not prompt_parts or lines[c_idx] != prompt_parts[-1]:
                    prompt_parts.append(lines[c_idx])

        full_scene_text = ". ".join(prompt_parts).strip()
        emb_128x4096 = self.renderer.encode_holistic_128_t5(full_scene_text)
        if len(self.holistic_scene_cache) > 256:
            self.holistic_scene_cache.clear()
        self.holistic_scene_cache[idx_tuple] = emb_128x4096
        return emb_128x4096

    @torch.no_grad()
    def compose_step(
        self,
        in_wm_128x4096: torch.Tensor,
        active_region_mask: dict[str, bool],
        region_gains: dict[str, float],
        persistence: float,
        surprise: float,
        kuramoto: float,
        video_params: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, str, float]:
        if video_params and isinstance(video_params, dict):
            if "geodesic_lock" in video_params:
                self.wm_modulation_gain = float(min(0.50, max(0.05, float(video_params["geodesic_lock"]))))
            if "cascade_rate" in video_params:
                self.cascade_rate = float(min(4.0, max(0.05, float(video_params["cascade_rate"]))))
            if "psi_threshold" in video_params:
                self.psi_threshold = float(min(0.95, max(0.15, float(video_params["psi_threshold"]))))

        wm_128x4096 = in_wm_128x4096[0] if in_wm_128x4096.ndim == 3 else in_wm_128x4096
        wm_energy = float(torch.mean(torch.abs(wm_128x4096)).item())

        if not self.baseline_initialized and wm_energy > 1e-5:
            self.running_wm_baseline_128x4096.copy_(wm_128x4096)
            self.baseline_initialized = True
        elif wm_energy > 1e-5:
            self.running_wm_baseline_128x4096.lerp_(wm_128x4096, 0.03)

        delta_wm_128x4096 = wm_128x4096 - self.running_wm_baseline_128x4096 * 0.90

        ordered_roles = ["Fpz", "AFz", "F3", "F4", "FCz"]
        primary_indices = []
        continuous_region_superposition = torch.zeros((128, 4096), dtype=torch.float32, device=self.device)
        total_gain_sum = 0.0
        debug_parts = []

        for role in ordered_roles:
            is_act = bool(active_region_mask.get(role, False))
            gain = float(region_gains.get(role, 0.92 if is_act else 0.06))
            bank_Kx128x4096 = self.region_full_banks[role]
            labels = self.region_labels[role]
            K = bank_Kx128x4096.shape[0]
            s_i, e_i = REGION_TOKEN_SLICES[role]

            slice_bank = bank_Kx128x4096[:, s_i:e_i, :].reshape(K, -1)
            delta_bank = slice_bank - slice_bank.mean(dim=0, keepdim=True)
            delta_bank_unit = delta_bank / (torch.norm(delta_bank, p=2, dim=-1, keepdim=True) + 1e-6)

            reg_wm_flat = delta_wm_128x4096[s_i:e_i, :].reshape(1, -1)
            reg_wm_centered = reg_wm_flat - reg_wm_flat.mean(dim=-1, keepdim=True)
            reg_wm_norm = torch.norm(reg_wm_centered, p=2, dim=-1, keepdim=True)
            reg_wm_unit = reg_wm_centered / (reg_wm_norm + 1e-6)

            inst_sims = torch.mm(reg_wm_unit, delta_bank_unit.T).squeeze(0)
            if self.smoothed_logits[role].shape[0] != K:
                self.smoothed_logits[role] = torch.zeros(K, dtype=torch.float32, device=self.device)

            alpha = min(0.88, 0.25 + 0.60 * gain) if (is_act or role == "FCz") else min(0.35, 0.05 + 0.25 * gain * self.cascade_rate)
            self.smoothed_logits[role].lerp_(inst_sims, alpha)

            l_std = torch.std(self.smoothed_logits[role], unbiased=False)
            l_scaled = (self.smoothed_logits[role] - self.smoothed_logits[role].mean()) / (l_std + 1e-6)
            soft_w = torch.softmax(l_scaled / 0.45, dim=0)

            best_i = int(torch.argmax(soft_w).item())
            self.active_concept_idx[role] = best_i
            primary_indices.append(best_i)

            reg_point_128x4096 = torch.sum(bank_Kx128x4096 * soft_w.view(K, 1, 1), dim=0)
            continuous_region_superposition.add_(reg_point_128x4096 * gain)
            total_gain_sum += gain

            if is_act or role == "FCz":
                conf_pct = int(round(float(soft_w[best_i].item()) * 100.0))
                words_c = labels[best_i].split()
                tail_s = " ".join(words_c[-3:]) if len(words_c) >= 3 else labels[best_i][:18]
                tok_dbg = self.renderer.decode_nearest_tokens(delta_wm_128x4096[s_i:e_i].mean(dim=0), top_k=1)
                debug_parts.append(f"{'★' if is_act else '•'}{role}[#{best_i+1}/{K}({conf_pct}%):{tail_s}|{tok_dbg}]")

        holistic_base_128x4096 = self._get_holistic_scene_embedding(tuple(primary_indices))
        mean_superpos = continuous_region_superposition / max(1e-5, total_gain_sum)
        norm_h = torch.norm(holistic_base_128x4096, p=2, dim=-1, keepdim=True).clamp(min=1e-4)
        mean_superpos = F.normalize(mean_superpos + 1e-6, p=2, dim=-1) * norm_h
        target_scene_128x4096 = holistic_base_128x4096 * 0.65 + mean_superpos * 0.35
        clean_norms = torch.norm(target_scene_128x4096, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        if self.smoothed_scene_128x4096 is None:
            self.smoothed_scene_128x4096 = target_scene_128x4096.clone()
        else:
            self.smoothed_scene_128x4096.lerp_(target_scene_128x4096, 0.72)

        wm_gate = float(min(1.0, wm_energy * 20.0))
        wm_full_524k = F.normalize(delta_wm_128x4096 + 1e-6, p=2, dim=-1) * clean_norms
        mod_w = float(min(0.38, max(0.08, self.wm_modulation_gain))) * wm_gate

        final_128x4096 = F.normalize(
            self.smoothed_scene_128x4096 * (1.0 - mod_w) + wm_full_524k * mod_w, p=2, dim=-1
        ) * clean_norms

        self.afz_stage_progress = float(min(1.0, self.afz_stage_progress * 0.92 + 0.08 * persistence * wm_gate))
        self.fpz_meta_charge = float(min(1.0, self.fpz_meta_charge * 0.85 + (0.15 * persistence if surprise > 0.35 else 0.0) * wm_gate))

        fcz_gain = float(region_gains.get("FCz", 0.92))
        advance_rate = float(min(0.30, max(0.0, (0.12 + 0.18 * persistence) * fcz_gain * wm_gate)))

        return final_128x4096.unsqueeze(0), self.neg_embed_1x128x4096, " ".join(debug_parts), advance_rate
