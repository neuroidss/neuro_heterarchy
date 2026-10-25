#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: REGIONAL CONCEPT COMPOSER FOR 3D-DiT (video_heterarchy_composer.py)
===================================================================================
- Полностью отделен от рендеринга (video_render_logic.py).
- Отвечает ТОЛЬКО за то, как рабочая память [1, 128, 4096] выбирает и модулирует
  концепты в каждом из 5 регионов коры [Fpz, AFz, F3, F4, FCz].
- Никакого разрезания 128 токенов T5 на куски!
  При смене концептов в любом регионе (например, когда рабочая память на FCz
  переключает #1 -> #2 -> #3 -> #4) строится ЦЕЛОСТНЫЙ [1, 128, 4096] T5-тензор
  всей сцены с этим выбранным концептом (плюс кэшируется в VRAM за 0 мс!),
  поэтому 3D-DiT видит 100% четкую сцену и применяет действие именно к описанному
  в лоре объекту!
===================================================================================
"""

from __future__ import annotations
import os
import zlib
import math
import numpy as np
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


def normalize_lore_path(p: str | None) -> str:
    if not p or not str(p).strip():
        return ""
    s = str(p).strip()
    if s.startswith("/app/"):
        s = s[len("/app/"):]
    return s


class RegionalVideoHeterarchyComposer:
    def __init__(self, renderer: NeuroVideoRenderer, lore_path: str = "lore/maze_lore.txt"):
        self.renderer = renderer
        self.device = DEVICE
        self.lore_path = normalize_lore_path(lore_path)

        self.current_lore_hash = None
        self.sections: dict[str, list[str]] = {
            "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
        }

        # Для каждого региона r с N_r концептами храним:
        # 1) Индивидуальные эмбеддинги концептов [N_r, 128, 4096] и их центрированные полюса отличий
        # 2) Кэш ЦЕЛОСТНЫХ сцен T5 (Holistic Scene Cache) для комбинаций концептов, чтобы
        #    T5 связывал объект и его действие в единое предложение без потери качества!
        self.region_individual_banks: dict[str, torch.Tensor] = {}
        self.region_poles_svd: dict[str, torch.Tensor] = {}
        self.region_svd_axes: dict[str, torch.Tensor] = {}
        self.holistic_scene_cache: dict[tuple[int, ...], torch.Tensor] = {}

        self.neg_embed_1x128x4096: torch.Tensor | None = None
        self.smoothed_scene_128x4096: torch.Tensor | None = None

        # Состояние фазового каскада рабочей памяти по 5 регионам
        self.baseline_vecs = {r: torch.zeros(4096, device=self.device) for r in REGION_ROLES}
        self.cascade_vecs  = {r: torch.zeros(4096, device=self.device) for r in REGION_ROLES}
        self.active_concept_idx = {r: 0 for r in REGION_ROLES}
        self.fpz_meta_charge = 0.0
        self.afz_stage_progress = 0.0

        self.cascade_rate = 1.0
        self.psi_threshold = 0.55
        self.wm_modulation_gain = 0.22  # Безопасная глубина прямой модуляции (как 0.25 в первом видео!)

        initial_sec = self._read_lore_file_sections()
        self.sync_lore_sections(initial_sec)

    def _read_lore_file_sections(self) -> dict[str, list[str]]:
        secs: dict[str, list[str]] = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
        candidates = [self.lore_path, f"/app/{self.lore_path}"] if self.lore_path else []
        for c_path in candidates:
            if c_path and os.path.isfile(c_path):
                cur = "FPZ"
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
                            secs[cur].append(s)
                break
        return secs

    @torch.no_grad()
    def sync_lore_sections(self, lore_sections: dict[str, list[str]] | None):
        if lore_sections is None:
            lore_sections = self._read_lore_file_sections()

        serialized = repr(sorted((k, v) for k, v in lore_sections.items()))
        new_hash = zlib.crc32(serialized.encode("utf-8"))
        if self.current_lore_hash == new_hash and self.neg_embed_1x128x4096 is not None:
            return

        self.holistic_scene_cache.clear()
        for role in REGION_ROLES:
            sec_k = role.upper()
            lines = [s.strip() for s in lore_sections.get(sec_k, []) if s.strip()]
            self.sections[sec_k] = lines

            if lines:
                enc_list = [self.renderer.encode_holistic_128_t5(line) for line in lines]
                bank = torch.stack(enc_list, dim=0)  # [N_r, 128, 4096]
                self.region_individual_banks[role] = bank

                if len(lines) > 1:
                    # Вычисляем чистые векторы отличий между N концептами региона (Zero-Mean Poles),
                    # чтобы рабочая память переключала все концепты (#1..#N) без залипания на #1!
                    c_means = bank.mean(dim=1)  # [N_r, 4096]
                    diff_poles = F.normalize(c_means - c_means.mean(dim=0, keepdim=True) + 1e-6, p=2, dim=-1)
                    _, _, Vh = torch.linalg.svd(diff_poles, full_matrices=False)
                    svd_axes = F.normalize(Vh[:min(16, Vh.shape[0])].contiguous(), p=2, dim=-1)
                    poles_proj = torch.matmul(diff_poles, svd_axes.T)
                    poles_proj = F.normalize(poles_proj - poles_proj.mean(dim=0, keepdim=True) + 1e-6, p=2, dim=-1)
                    self.region_svd_axes[role] = svd_axes
                    self.region_poles_svd[role] = poles_proj
                else:
                    flat = bank[0] - bank[0].mean(dim=0, keepdim=True)
                    _, _, Vh = torch.linalg.svd(F.normalize(flat + 1e-6, p=2, dim=-1), full_matrices=False)
                    self.region_svd_axes[role] = F.normalize(Vh[:16].contiguous(), p=2, dim=-1)
                    self.region_poles_svd[role] = torch.ones((1, 16), device=self.device)
            else:
                empty_128 = self.renderer.encode_holistic_128_t5("")
                self.region_individual_banks[role] = empty_128.unsqueeze(0)
                self.region_svd_axes[role] = torch.eye(16, 4096, device=self.device)
                self.region_poles_svd[role] = torch.ones((1, 16), device=self.device)

        neg_lines = [s.strip() for s in lore_sections.get("NEGATIVE", []) if s.strip()]
        neg_text = " ".join(neg_lines)
        self.neg_embed_1x128x4096 = self.renderer.encode_holistic_128_t5(neg_text).unsqueeze(0)
        self.current_lore_hash = new_hash

        # Предкодируем базовую целостную сцену (0, 0, 0, 0, 0)
        base_scene = self._get_holistic_scene_embedding((0, 0, 0, 0, 0))
        if self.smoothed_scene_128x4096 is None:
            self.smoothed_scene_128x4096 = base_scene.clone()

        print(
            "📜 [COMPOSER LORE SYNC] Концептов по регионам: " +
            ", ".join(f"{r}:{len(self.sections[r.upper()])}" for r in REGION_ROLES)
        )

    @torch.no_grad()
    def _get_holistic_scene_embedding(self, idx_tuple: tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        Собирает выбранные рабочей памятью концепты 5 регионов [Fpz, AFz, F3, F4, FCz]
        в ЕДИНЫЙ ЦЕЛОСТНЫЙ ПРОМПТ и кодирует его через T5 (кэшируя в VRAM)!
        Именно поэтому 3D-DiT рисует 100% качественный мир (как в первом видео) И применяет
        действие FCz непосредственно к объекту сцены!
        """
        if idx_tuple in self.holistic_scene_cache:
            return self.holistic_scene_cache[idx_tuple]

        ordered_roles = ["Fpz", "AFz", "F3", "F4", "FCz"]
        prompt_parts = []
        for r_i, role in enumerate(ordered_roles):
            lines = self.sections.get(role.upper(), [])
            if lines:
                c_idx = idx_tuple[r_i] % len(lines)
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
        """
        Возвращает:
          1) final_embeds [1, 128, 4096]
          2) neg_embeds   [1, 128, 4096]
          3) debug_tag    (строка с активными концептами регионов)
          4) anchor_advance_rate (скорость продвижения базового кадра при активном действии)
        """
        if video_params and isinstance(video_params, dict):
            if "geodesic_lock" in video_params:
                self.wm_modulation_gain = float(np.clip(float(video_params["geodesic_lock"]), 0.05, 0.40))
            if "cascade_rate" in video_params:
                self.cascade_rate = float(np.clip(float(video_params["cascade_rate"]), 0.05, 4.0))
            if "psi_threshold" in video_params:
                self.psi_threshold = float(np.clip(float(video_params["psi_threshold"]), 0.15, 0.95))

        if in_wm_128x4096.ndim == 2:
            in_wm_128x4096 = in_wm_128x4096.unsqueeze(0)

        # 1. Вычисляем фазовые дельта-векторы рабочей памяти в каждом из 5 регионов
        delta_vecs = {}
        for r_name, (s_i, e_i) in REGION_TOKEN_SLICES.items():
            slice_r = in_wm_128x4096[0, s_i:e_i]
            half = slice_r.shape[0] // 2
            wave_chord = slice_r[half:].mean(dim=0) - slice_r[:half].mean(dim=0)
            v = F.normalize(slice_r.mean(dim=0) * 0.45 + wave_chord * 0.55, p=2, dim=-1)

            if torch.norm(self.baseline_vecs[r_name]) < 1e-5:
                self.baseline_vecs[r_name].copy_(v)
            else:
                self.baseline_vecs[r_name].lerp_(v, 0.03)

            delta_vecs[r_name] = F.normalize(v - self.baseline_vecs[r_name] * 0.94 + 1e-6, p=2, dim=-1)

        active_roles = [r for r in REGION_ROLES if active_region_mask.get(r, False)]
        lead_delta = delta_vecs[active_roles[0]] if active_roles else delta_vecs["FCz"]

        # 2. В каждом регионе выбираем активный концепт (и плавно интерполируем со вторым полюсом)
        ordered_roles = ["Fpz", "AFz", "F3", "F4", "FCz"]
        primary_indices = []
        secondary_indices = []
        secondary_weights = []
        debug_parts = []

        for role in ordered_roles:
            is_act = bool(active_region_mask.get(role, False))
            gain = float(region_gains.get(role, 0.92 if is_act else 0.06))
            lines = self.sections.get(role.upper(), [])
            n_c = len(lines)

            if is_act or role == "FCz":
                alpha = min(0.88, 0.30 + 0.58 * gain)
                src = delta_vecs[role]
            else:
                alpha = min(0.40, (0.04 + surprise * 0.12 + gain * 0.25) * self.cascade_rate)
                src = F.normalize(delta_vecs[role] * 0.50 + lead_delta * 0.50, p=2, dim=-1)

            self.cascade_vecs[role] = F.normalize(
                self.cascade_vecs[role] * (1.0 - alpha) + src * alpha, p=2, dim=-1
            )

            if n_c <= 1:
                self.active_concept_idx[role] = 0
                primary_indices.append(0)
                secondary_indices.append(0)
                secondary_weights.append(0.0)
            else:
                # Если регион НЕ активен и gain очень мал (< 0.12) — держим его концепт стабильным!
                if not is_act and gain < 0.12 and role != "FCz":
                    cur_i = self.active_concept_idx[role] % n_c
                    primary_indices.append(cur_i)
                    secondary_indices.append(cur_i)
                    secondary_weights.append(0.0)
                else:
                    svd_axes = self.region_svd_axes[role]
                    poles_svd = self.region_poles_svd[role]
                    q_proj = F.normalize(torch.matmul(self.cascade_vecs[role].unsqueeze(0), svd_axes.T), p=2, dim=-1)
                    sims = torch.mm(q_proj, poles_svd.T).squeeze(0)  # [-1 .. +1] по всем N концептам!

                    top_vals, top_i = torch.topk(sims, k=min(2, n_c))
                    idx_1 = int(top_i[0].item())
                    idx_2 = int(top_i[1].item()) if n_c >= 2 else idx_1
                    w_pair = torch.softmax(top_vals / 0.15, dim=0)

                    self.active_concept_idx[role] = idx_1
                    primary_indices.append(idx_1)
                    secondary_indices.append(idx_2)
                    secondary_weights.append(float(w_pair[1].item()) * min(0.35, gain * 0.35))

            if is_act or role == "FCz":
                c_i = self.active_concept_idx[role]
                if lines:
                    words_c = lines[c_i % n_c].split()
                    tail_s = " ".join(words_c[-3:]) if len(words_c) >= 3 else lines[c_i % n_c][:18]
                    lbl = f"#{c_i+1}:{tail_s}"
                else:
                    lbl = "OPEN"
                tok_dbg = self.renderer.decode_nearest_tokens(self.cascade_vecs[role], top_k=2)
                debug_parts.append(f"{'★' if is_act else '•'}{role}[{lbl}|{tok_dbg}]")

        # 3. Получаем ЦЕЛОСТНЫЙ [128, 4096] T5-эмбеддинг выбранной комбинации концептов!
        primary_scene_128 = self._get_holistic_scene_embedding(tuple(primary_indices))

        # Если во ведущем активном регионе есть второй близкий полюс — мягко подмешиваем его целостную сцену
        alt_tuple = list(primary_indices)
        fcz_pos = ordered_roles.index("FCz")
        if secondary_indices[fcz_pos] != primary_indices[fcz_pos] and secondary_weights[fcz_pos] > 0.05:
            alt_tuple[fcz_pos] = secondary_indices[fcz_pos]
            alt_scene_128 = self._get_holistic_scene_embedding(tuple(alt_tuple))
            w_alt = secondary_weights[fcz_pos]
            target_scene_128 = primary_scene_128 * (1.0 - w_alt) + alt_scene_128 * w_alt
        else:
            target_scene_128 = primary_scene_128

        clean_norms = torch.norm(target_scene_128, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        # Плавный переход между целостными сценами при переключении концепта
        if self.smoothed_scene_128x4096 is None:
            self.smoothed_scene_128x4096 = target_scene_128.clone()
        else:
            self.smoothed_scene_128x4096.lerp_(target_scene_128, 0.65)

        # 4. Точная формула модуляции рабочей памятью из первого видео (78% целостная сцена + 22% рабочая память):
        #    гарантирует, что сцена НИКОГДА не выгорит в белый шум, а рабочая память будет живо управлять!
        wm_normed = F.normalize(in_wm_128x4096[0], p=2, dim=-1) * clean_norms
        mod_w = float(np.clip(self.wm_modulation_gain, 0.08, 0.30))
        final_128x4096 = F.normalize(
            self.smoothed_scene_128x4096 * (1.0 - mod_w) + wm_normed * mod_w, p=2, dim=-1
        ) * clean_norms

        self.afz_stage_progress = float(min(1.0, self.afz_stage_progress * 0.92 + 0.08 * persistence))
        self.fpz_meta_charge = float(min(1.0, self.fpz_meta_charge * 0.85 + (0.15 * persistence if surprise > 0.35 else 0.0)))

        fcz_gain = float(region_gains.get("FCz", 0.92))
        advance_rate = float(np.clip((0.12 + 0.18 * persistence) * fcz_gain, 0.05, 0.30))

        return final_128x4096.unsqueeze(0), self.neg_embed_1x128x4096, " ".join(debug_parts), advance_rate
