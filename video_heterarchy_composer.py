#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ACONCEPTUAL 5-REGION HILBERT COMPOSER FOR 3D-DiT (LIVE VRAM GPU BUS)
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
from neuro_vram_bus import universal_full_rank_manifold_step, HardwareVRAMBus

CORE_REGIONS = ("FPZ", "AFZ", "F3", "F4", "FCZ")


def normalize_lore_path(p: str | None) -> str:
    if not p or not str(p).strip():
        return ""
    s = str(p).strip()
    if s.startswith("/app/"):
        s = s[len("/app/"):]
    return s


class RegionalVideoHeterarchyComposer:
    def __init__(self, renderer, lore_path: str = ""):
        self.renderer = renderer
        self.device = DEVICE
        self.lore_path = normalize_lore_path(lore_path)

        self.current_lore_hash = None
        self.sections: dict[str, list[str]] = {
            "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
        }

        self.neg_embed_1x128x4096: torch.Tensor | None = None
        self.neg_mask_1x128: torch.Tensor | None = None

        self.running_wm_baseline_128x4096 = torch.zeros((128, 4096), dtype=torch.float32, device=self.device)
        self.baseline_initialized = False

        self.active_concept_idx = {r: 0 for r in REGION_ROLES}
        self.fpz_meta_charge = 0.0
        self.afz_stage_progress = 0.0

        self.cascade_rate = 1.0
        self.psi_threshold = 0.55
        self.wm_radius = 0.25

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
        if lore_sections is None or not any(lore_sections.get(r) for r in CORE_REGIONS):
            file_secs = self._read_lore_file_sections()
            if any(file_secs.get(r) for r in CORE_REGIONS):
                lore_sections = file_secs
            elif lore_sections is None:
                lore_sections = file_secs

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

        for role in REGION_ROLES:
            sec_k = role.upper()
            lines = [s.strip() for s in lore_sections.get(sec_k, []) if s.strip()]
            self.sections[sec_k] = lines

        neg_lines = [s.strip() for s in lore_sections.get("NEGATIVE", []) if s.strip()]
        neg_prompt = ", ".join(neg_lines) if neg_lines else ""
        neg_emb, neg_mask = self.renderer.encode_prompt_with_mask(neg_prompt)

        self.neg_embed_1x128x4096 = neg_emb.unsqueeze(0)
        self.neg_mask_1x128 = neg_mask
        self.current_lore_hash = new_hash

        print(
            "📜 [HETERARCHY COMPOSER] Загружены концепты 5 регионов: " +
            ", ".join(f"{r}:{len(self.sections[r.upper()])}" for r in REGION_ROLES)
        )

    @torch.no_grad()
    def compose_step(
        self,
        in_wm_128x4096: torch.Tensor,
        active_region_mask: dict[str, bool],
        region_gains: dict[str, float],
        persistence: float,
        surprise: float,
        kuramoto: float,
        stochastic: float = 0.0,
        unassigned_stochastic: float = 1.0,
        wm_radius: float = 0.25,
        video_params: dict | None = None,
        vram_bus: HardwareVRAMBus | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, str, float]:
        if video_params and isinstance(video_params, dict):
            if "geodesic_lock" in video_params:
                wm_radius = float(min(0.80, max(0.02, float(video_params["geodesic_lock"]))))
            if "cascade_rate" in video_params:
                self.cascade_rate = float(min(4.0, max(0.05, float(video_params["cascade_rate"]))))
            if "psi_threshold" in video_params:
                self.psi_threshold = float(min(0.95, max(0.15, float(video_params["psi_threshold"]))))

        stoch_act = float(np.clip(stochastic, 0.0, 1.0))
        stoch_unass = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        eff_radius = float(np.clip(wm_radius, 0.0, 1.0))
        self.wm_radius = eff_radius

        wm_128x4096 = in_wm_128x4096[0] if in_wm_128x4096.ndim == 3 else in_wm_128x4096
        wm_energy = float(torch.mean(torch.abs(wm_128x4096)).item())

        if wm_energy <= 1e-5:
            delta_wm_128x4096 = torch.zeros_like(wm_128x4096)
        elif not self.baseline_initialized:
            self.running_wm_baseline_128x4096.copy_(wm_128x4096)
            self.baseline_initialized = True
            delta_wm_128x4096 = wm_128x4096 - self.running_wm_baseline_128x4096 * 0.85
        else:
            self.running_wm_baseline_128x4096.lerp_(wm_128x4096, 0.04)
            delta_wm_128x4096 = wm_128x4096 - self.running_wm_baseline_128x4096 * 0.85

        # Чтение фазовых углов прямо из VRAM-памяти GPU ядра
        meta = vram_bus.meta_64 if vram_bus is not None else None

        dl_phase = float(meta[3].item()) if (meta is not None and meta.numel() > 3) else 0.0
        n_sec = max(2, int(meta[1].item())) if (meta is not None and meta.numel() > 1) else 8
        cur_sec = int(meta[2].item()) % n_sec if (meta is not None and meta.numel() > 2) else 0

        norm_coord = float(vram_bus.slot_norm_coord_16[cur_sec].item()) if (vram_bus is not None and vram_bus.slot_norm_coord_16.numel() > cur_sec) else 0.0
        x_local = float(meta[9].item()) if (meta is not None and meta.numel() > 9) else 0.0
        y_global = float(meta[10].item()) if (meta is not None and meta.numel() > 10) else 0.0
        lx = float(meta[12].item()) if (meta is not None and meta.numel() > 12) else 0.0
        ly = float(meta[13].item()) if (meta is not None and meta.numel() > 13) else 0.0

        # Аконцептуальный фазовый адрес для каждого региона
        region_phase_coords = {
            "Fpz": (dl_phase + math.pi) / (2.0 * math.pi),
            "AFz": norm_coord,
            "F3":  (float(np.clip(x_local, -1.0, 1.0)) + 1.0) * 0.5,
            "F4":  (float(np.clip(y_global, -1.0, 1.0)) + 1.0) * 0.5,
            "FCz": (math.atan2(ly, lx) + math.pi) / (2.0 * math.pi) if (abs(lx) > 1e-4 or abs(ly) > 1e-4) else norm_coord,
        }

        ordered_roles = ["Fpz", "AFz", "F3", "F4", "FCz"]
        selected_phrases = []
        debug_parts = []

        for role in ordered_roles:
            is_act = bool(active_region_mask.get(role, False))
            lines = self.sections.get(role.upper(), [])
            K = len(lines)
            if K == 0:
                continue

            phi = float(np.clip(region_phase_coords.get(role, 0.0), 0.0, 0.9999))
            best_i = int(math.floor(phi * K)) % K
            self.active_concept_idx[role] = best_i

            phrase = lines[best_i]
            selected_phrases.append(phrase)
            debug_parts.append(f"{'★' if is_act else '•'}{role}[#{best_i+1}/{K}: {phrase[:20]}..]")

        full_prompt_text = ", ".join(selected_phrases) if selected_phrases else "flowing visual dynamics"

        # Полноценное кодирование текста через T5 с валидной маской внимания
        anchor_128x4096, combined_pos_mask = self.renderer.encode_prompt_with_mask(full_prompt_text)

        has_any_active = any(active_region_mask.get(r, False) for r in ordered_roles)
        effective_stoch = stoch_act if has_any_active else stoch_unass

        # Геодезический поворот в касательном подпространстве на GPU
        if effective_stoch >= 0.999 or eff_radius <= 0.001 or wm_energy <= 1e-5:
            final_embed_128x4096 = anchor_128x4096
        else:
            final_embed_128x4096, _, _, _ = universal_full_rank_manifold_step(
                anchor_bank_KxLxD=anchor_128x4096.unsqueeze(0),
                wm_tensor_LxD=delta_wm_128x4096,
                gain_gamma=eff_radius,
                stochastic=effective_stoch,
                temp=0.35,
                preserve_bos_token=False,
            )

        self.renderer.last_pos_mask = combined_pos_mask
        self.renderer.last_neg_mask = self.neg_mask_1x128

        wm_gate = float(min(1.0, wm_energy * 20.0)) if has_any_active else 0.0
        self.afz_stage_progress = float(min(1.0, self.afz_stage_progress * 0.92 + 0.08 * persistence * wm_gate))
        self.fpz_meta_charge = float(min(1.0, self.fpz_meta_charge * 0.85 + (0.15 * persistence if surprise > 0.35 else 0.0) * wm_gate))

        fcz_gain = float(region_gains.get("FCz", 0.92))
        advance_rate = float(min(0.30, max(0.0, (0.12 + 0.18 * persistence) * fcz_gain * wm_gate)))

        return final_embed_128x4096.unsqueeze(0), self.neg_embed_1x128x4096, " ".join(debug_parts), advance_rate
