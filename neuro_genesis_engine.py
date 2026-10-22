#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: PURE SEMANTIC INTER-REGION MONTY SDR HETERARCHY ENGINE
===================================================================================
- РАДИКАЛЬНЫЙ ЗАПРЕТ НА СДВИГ ПИКСЕЛЕЙ И НАВЯЗАННЫЕ ТЕКСТЫ:
  Все трансформации происходят исключительно в латентном пространстве смыслов [77 × 768].
- ЧИСТЫЕ СЕМАНТИЧЕСКИЕ ДЕЙСТВИЯ (Intrinsic SVD Affordances):
  Даже для 1 концепта базис действий V_aff [16 × 768] извлекается из SVD его собственных
  75 контекстных токенов CLIP и векторов к соседним концептам активного лора.
- НАСТОЯЩАЯ ГЕТЕРАРХИЯ SDR МОНТИ МЕЖДУ 5 РЕГИОНАМИ (FCz <-> F3/F4 <-> AFz <-> Fpz):
  * Сверху вниз (L6a -> L1): матрица W_topdown_sdr предсказывает ожидаемый SDR внизу.
  * Снизу вверх (L3 -> L4): непредсказанная разница upward_sdr_anomaly пробивает верх!
- 20-СЕКУНДНАЯ СХОДИМОСТЬ НА FCz (Maze-Identical Semantic Integrator):
  Leaky-интегратор + Persistence (1 + 4 * persistence) двигают состояние напрямую
  вдоль 16 SVD-осей смысла текущего концепта без потери размерности (32 × 120/276 -> 77 × 768).
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import os
import math
import time
import json
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y, DX_120, DY_120

MAX_SLOTS_CAPACITY = 16

REGION_TAU_SEC = {
    "FCz": 0.04,  # Мгновенная гамма-динамика действий (60 FPS)
    "F3":  0.35,  # Тета-динамика структурных форм
    "F4":  0.35,  # Тета-динамика контекста и пластичности
    "AFz": 1.20,  # Дельта-динамика композиционных правил (A ⊃ B)
    "Fpz": 5.00   # Сверхмедленная динамика смены наборов правил (Колесо Эпох)
}


def build_cortical_dipole_geometry(coords_x: np.ndarray, coords_y: np.ndarray):
    n_ch = len(coords_x)
    i_idx, j_idx = np.triu_indices(n_ch, k=1)
    n_pairs = len(i_idx)

    dip_x = (coords_x[i_idx] + coords_x[j_idx]) / 2.0
    dip_y = (coords_y[i_idx] + coords_y[i_idx]) / 2.0
    dip_y = (coords_y[i_idx] + coords_y[j_idx]) / 2.0
    dx = (coords_x[j_idx] - coords_x[i_idx]).astype(np.float32)
    dy = (coords_y[j_idx] - coords_y[i_idx]).astype(np.float32)
    dist_mm = np.hypot(dx, dy)

    dip_x_t = torch.tensor(dip_x, dtype=torch.float32, device=DEVICE).view(1, n_pairs)
    dip_y_t = torch.tensor(dip_y, dtype=torch.float32, device=DEVICE).view(1, n_pairs)

    col_x = torch.linspace(-12.0, 12.0, 32, device=DEVICE, dtype=torch.float32)
    col_y = torch.linspace(-12.0, 12.0, 24, device=DEVICE, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(col_y, col_x, indexing='ij')
    cols_x_t = grid_x.reshape(768, 1)
    cols_y_t = grid_y.reshape(768, 1)

    d_sq = (cols_x_t - dip_x_t)**2 + (cols_y_t - dip_y_t)**2
    proj_768xP = torch.exp(-d_sq / 28.0)
    angle_dip = torch.atan2(torch.tensor(dy, device=DEVICE), torch.tensor(dx, device=DEVICE)).view(1, n_pairs)
    angle_col = torch.atan2(cols_y_t, cols_x_t)
    angular_mod = 0.5 * (1.0 + torch.cos(2.0 * (angle_col - angle_dip)))
    proj_768xP = F.normalize(proj_768xP * (0.4 + 0.6 * angular_mod), p=2, dim=1)

    return {
        "n_pairs": n_pairs,
        "dip_x_t": dip_x_t, "dip_y_t": dip_y_t,
        "dx_gpu": torch.tensor(dx, dtype=torch.float32, device=DEVICE),
        "dy_gpu": torch.tensor(dy, dtype=torch.float32, device=DEVICE),
        "proj_768xP": proj_768xP,
        "micro_mask": torch.tensor(dist_mm < 9.5, dtype=torch.float32, device=DEVICE).view(1, n_pairs),
        "meso_mask":  torch.tensor((dist_mm >= 9.5) & (dist_mm < 16.5), dtype=torch.float32, device=DEVICE).view(1, n_pairs),
        "macro_mask": torch.tensor(dist_mm >= 16.5, dtype=torch.float32, device=DEVICE).view(1, n_pairs),
        "left_mask":  torch.tensor(dip_x <= 0.0, dtype=torch.float32, device=DEVICE).view(1, n_pairs),
        "right_mask": torch.tensor(dip_x > 0.0, dtype=torch.float32, device=DEVICE).view(1, n_pairs)
    }


GEOM_16 = build_cortical_dipole_geometry(COORDS_X, COORDS_Y)
PHYSICAL_PROJECTION_768x120 = GEOM_16["proj_768xP"]
DX_GPU_120 = GEOM_16["dx_gpu"]
DY_GPU_120 = GEOM_16["dy_gpu"]


# -----------------------------------------------------------------------------
# СЕМАНТИЧЕСКИЙ ИНТЕГРАТОР ДЕЙСТВИЙ (ФИЗИКА ЛАБИРИНТА ВНУТРИ 768-D СМЫСЛОВ)
# -----------------------------------------------------------------------------
class SemanticActionIntegrator:
    """
    Точный аналог Avatar3D из neuro_monty_bci_maze.py, но работающий НЕ с пикселями,
    а в 16×768-мерном касательном подпространстве смысловых аффордансов V_aff!
    """
    def __init__(self, sensitivity: float = 0.06, intent_gain: float = 1.6):
        self.sensitivity = sensitivity
        self.intent_gain = intent_gain
        self.persistence = 0.0
        self.last_chord_768 = None
        self.semantic_state_16x768 = torch.zeros((16, 768), dtype=torch.float32, device=DEVICE)

    @torch.no_grad()
    def step(self, raw_action_wave_16x768: torch.Tensor, v_aff_16x768: torch.Tensor, temp_ry: float = 0.0) -> tuple[torch.Tensor, float, float]:
        coords_16x16 = torch.matmul(raw_action_wave_16x768, v_aff_16x768.T)  # [16, 16]
        projected_16x768 = torch.matmul(coords_16x16, v_aff_16x768)          # [16, 768]

        chord_768 = projected_16x768[-1] - projected_16x768[0]
        mag = float(torch.norm(chord_768).item())

        if self.last_chord_768 is not None and mag > 0.02:
            cos_th = float(torch.sum(chord_768 * self.last_chord_768) / (mag * float(torch.norm(self.last_chord_768).item()) + 1e-6))
            if cos_th > 0.75:
                self.persistence = min(1.0, self.persistence + 0.05)
            else:
                self.persistence *= 0.94
        else:
            self.persistence *= 0.94

        self.last_chord_768 = chord_768.clone()

        smooth = 0.96 - (self.sensitivity * 0.1)
        gain = self.sensitivity * self.intent_gain
        active_boost = (1.0 + self.persistence * 4.0) * (1.0 + max(-0.4, min(0.8, temp_ry * 0.4)))

        step_delta = projected_16x768 * (gain * active_boost * (1.0 - smooth))
        self.semantic_state_16x768.mul_(smooth).add_(step_delta)

        state_norm = torch.norm(self.semantic_state_16x768, dim=-1, keepdim=True)
        max_norm = 6.0
        self.semantic_state_16x768 = torch.where(
            state_norm > max_norm,
            (self.semantic_state_16x768 / (state_norm + 1e-6)) * max_norm,
            self.semantic_state_16x768
        )
        return self.semantic_state_16x768, self.persistence, active_boost


# -----------------------------------------------------------------------------
# КОЛОНКА МОНТИ С МЕЖРЕГИОНАЛЬНОЙ ГЕТЕРАРХИЕЙ SDR (L3->L4 ВВЕРХ, L6a->L1 ВНИЗ)
# -----------------------------------------------------------------------------
class PredictiveColumnMonty(nn.Module):
    def __init__(self, role: str, geom: dict, num_columns: int = 4096, k_active: int = 80, num_slots: int = 32):
        super().__init__()
        self.role = role
        self.geom = geom
        self.n_pairs = geom["n_pairs"]
        self.num_columns = num_columns
        self.k_active = k_active
        self.num_slots = num_slots
        self.tau_sec = REGION_TAU_SEC.get(role, 0.5)

        grid_dim = int(math.isqrt(num_columns))
        cy = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(grid_dim, 1, 1)
        cx = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(1, grid_dim, 1)
        d_sq = (cx - geom["dip_x_t"].view(1, 1, self.n_pairs))**2 + (cy - geom["dip_y_t"].view(1, 1, self.n_pairs))**2

        perm_base = torch.exp(-d_sq / 36.0).view(-1, self.n_pairs)[:num_columns]
        self.register_buffer("conn_pos", (perm_base >= 0.25).float())
        self.register_buffer("conn_neg", (torch.roll(perm_base, shifts=num_columns // 2, dims=0) >= 0.25).float())

        self.register_buffer("W_out", torch.zeros((num_slots * self.n_pairs, num_columns), device=DEVICE))
        self.register_buffer("W_topdown_sdr", torch.eye(num_columns, device=DEVICE) * 0.4)
        self.register_buffer("l23_pooled_sdr", torch.zeros(num_columns, device=DEVICE))
        self.register_buffer("upward_sdr_anomaly", torch.zeros(num_columns, device=DEVICE))

        gen = torch.Generator(device=DEVICE).manual_seed(777)
        sdr_proj = torch.randn((num_columns, 768), generator=gen, device=DEVICE) / math.sqrt(k_active)
        self.register_buffer("sdr_to_768", sdr_proj)

        self.baseline_loss = 0.020
        self.action_integrator = SemanticActionIntegrator()

    @torch.no_grad()
    def forward_dynamics(self, raw_iplv_32xP: torch.Tensor, dt: float,
                         top_down_parent_sdr: Optional[torch.Tensor] = None,
                         bottom_up_child_anomaly: Optional[torch.Tensor] = None) -> dict:
        proj_T = self.geom["proj_768xP"].T

        x_pos = torch.relu(raw_iplv_32xP)
        x_neg = torch.relu(-raw_iplv_32xP)
        overlap = (torch.matmul(x_pos, self.conn_pos.T) + torch.matmul(x_neg, self.conn_neg.T)) / 20.0

        pred_from_parent = None
        if top_down_parent_sdr is not None:
            pred_from_parent = torch.mv(self.W_topdown_sdr, top_down_parent_sdr)
            overlap = overlap + pred_from_parent.unsqueeze(0) * 0.25

        if bottom_up_child_anomaly is not None:
            overlap = overlap + bottom_up_child_anomaly.unsqueeze(0) * 0.45

        _, active_idx = torch.topk(overlap, self.k_active, dim=-1)
        sdr_seq = torch.zeros((self.num_slots, self.num_columns), device=DEVICE)
        sdr_seq.scatter_(1, active_idx, 1.0)
        sdr_inst = torch.mean(sdr_seq, dim=0)

        alpha_tau = float(np.clip(dt / max(0.016, self.tau_sec), 0.01, 0.85))
        self.l23_pooled_sdr.mul_(1.0 - alpha_tau).add_(sdr_inst, alpha=alpha_tau)

        if pred_from_parent is not None and top_down_parent_sdr is not None:
            sdr_err = sdr_inst - pred_from_parent
            self.upward_sdr_anomaly.copy_(torch.relu(sdr_err))
            p_energy = torch.sum(top_down_parent_sdr ** 2) + 1.0
            self.W_topdown_sdr.mul_(0.9998).add_(torch.outer(sdr_err, top_down_parent_sdr), alpha=float(0.03 / p_energy))
        else:
            self.upward_sdr_anomaly.copy_(sdr_inst * 0.15)

        sdr_anom_norm = float(torch.norm(self.upward_sdr_anomaly).item() / math.sqrt(self.k_active))

        real_flat = raw_iplv_32xP.reshape(-1)
        pred_flat = torch.mv(self.W_out, self.l23_pooled_sdr)
        error_flat = real_flat - pred_flat
        loss = float(torch.mean(error_flat ** 2).item())

        surprise = max(0.0, (loss - self.baseline_loss) / (self.baseline_loss + 1e-5)) + sdr_anom_norm * 0.3
        self.baseline_loss = self.baseline_loss * 0.985 + loss * 0.015

        state_energy = torch.sum(self.l23_pooled_sdr ** 2) + 1e-4
        eta = min(0.25, 0.08 * (1.0 + surprise * 0.5)) / state_energy
        self.W_out.mul_(0.9995).add_(torch.outer(error_flat, self.l23_pooled_sdr), alpha=float(eta))

        z_32x768 = torch.matmul(raw_iplv_32xP, proj_T)
        z_on_16x768  = F.adaptive_avg_pool1d(torch.matmul(x_pos, proj_T).T.unsqueeze(0), 16).squeeze(0).T
        z_off_16x768 = F.adaptive_avg_pool1d(torch.matmul(x_neg, proj_T).T.unsqueeze(0), 16).squeeze(0).T

        chord_768 = z_32x768[-1] - z_32x768[0]
        chord_unit = F.normalize(chord_768.unsqueeze(0), p=2, dim=-1)
        rel_z = z_32x768 - z_32x768[0:1]
        sagitta_32x768 = rel_z - torch.sum(rel_z * chord_unit, dim=-1, keepdim=True) * chord_unit
        sagitta_16x768 = F.adaptive_avg_pool1d(sagitta_32x768.T.unsqueeze(0), 16).squeeze(0).T
        temp_asym_16x768 = z_32x768[16:32] - z_32x768[0:16]

        sdr_7x4096 = F.adaptive_avg_pool1d(sdr_seq.T.unsqueeze(0), 7).squeeze(0).T
        sdr_tokens_7x768 = torch.matmul(sdr_7x4096, self.sdr_to_768)

        return {
            "z_32x768": z_32x768,
            "z_on_16x768": z_on_16x768,
            "z_off_16x768": z_off_16x768,
            "sagitta_16x768": sagitta_16x768,
            "temp_asym_16x768": temp_asym_16x768,
            "sdr_tokens_7x768": sdr_tokens_7x768,
            "l23_sdr": self.l23_pooled_sdr,
            "upward_sdr_anomaly": self.upward_sdr_anomaly,
            "pred_iplv": pred_flat.view(self.num_slots, self.n_pairs),
            "error_iplv": error_flat.view(self.num_slots, self.n_pairs),
            "anomaly_loss": loss,
            "surprise": float(np.clip(surprise, 0.0, 2.5))
        }


class UniversalSemanticSpace:
    def __init__(self, lore_file=None, concepts_cli=None, target_k=None, model_id="openai/clip-vit-large-patch14"):
        print("⏳ [СЕМАНТИКА] Инициализация CLIP Text Model на GPU...")
        self.lock = threading.RLock()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE).eval()
        self.gpu_embed_cache = {}
        self.affordance_cache = {}

        self.empty_embed = self._compute_77_embedding("")
        self.last_active_embed = self.empty_embed.clone()

        custom_words = []
        if lore_file and os.path.exists(lore_file):
            if lore_file.endswith('.json'):
                with open(lore_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    custom_words = [str(x).strip() for x in (d if isinstance(d, list) else d.keys()) if str(x).strip()]
            else:
                with open(lore_file, 'r', encoding='utf-8') as f:
                    custom_words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        elif concepts_cli and concepts_cli.strip().lower() not in ["none", ""]:
            custom_words = [w.strip() for w in concepts_cli.split(",") if w.strip()]

        self.words = custom_words
        self._rebuild_vocab_matrix_locked()

    def _compute_77_embedding(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            return self.text_model(tokens).last_hidden_state[0]

    def _rebuild_vocab_matrix_locked(self):
        with self.lock:
            n_words = len(self.words)
            if n_words == 0:
                raw = self.last_active_embed.mean(dim=0, keepdim=True)
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
                self.affordance_cache.clear()
                return

            lore_embeds = [self.get_77_embedding_gpu(w).mean(dim=0) for w in self.words]
            raw = torch.stack(lore_embeds, dim=0)

            if n_words == 1:
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
            else:
                mu = torch.mean(raw, dim=0, keepdim=True)
                std = torch.std(raw - mu, dim=0, keepdim=True, unbiased=False) + 1e-6
                self.norm_vocab = F.normalize((raw - mu) / std, p=2, dim=-1)

            self.affordance_cache.clear()

    def update_concepts_on_the_fly(self, new_words: list[str]):
        cleaned = [w.strip() for w in new_words if w.strip()]
        with self.lock:
            if not cleaned and self.words:
                self.last_active_embed = self.get_77_embedding_gpu(self.words[0]).clone()
            self.words = list(cleaned)
            self._rebuild_vocab_matrix_locked()
            print(f"🔄 [СЕМАНТИКА GPU] Пул концептов обновлен ({len(self.words)}): {self.words}")

    def get_77_embedding_gpu(self, word: str) -> torch.Tensor:
        with self.lock:
            if not word or not word.strip():
                return self.last_active_embed
            w_clean = word.strip().lower()
            if w_clean in self.gpu_embed_cache:
                return self.gpu_embed_cache[w_clean]
            emb = self._compute_77_embedding(w_clean)
            self.gpu_embed_cache[w_clean] = emb
            return emb

    def get_concept_affordances_svd(self, concept_idx: int, k_neighbors: int = 32, max_bases: int = 16) -> torch.Tensor:
        """
        Извлекает чистый семантический базис действий V_aff [16 × 768] из собственных
        75 токенов текущего концепта и векторов к его соседям по активному лору.
        """
        with self.lock:
            n_items = len(self.words)
            if n_items == 0:
                base_77 = self.last_active_embed
                key = -1
            else:
                safe_c_idx = concept_idx % n_items
                key = safe_c_idx
                base_77 = self.get_77_embedding_gpu(self.words[safe_c_idx])

            if key in self.affordance_cache:
                return self.affordance_cache[key]

            tokens_centered = base_77[1:76] - base_77[1:76].mean(dim=0, keepdim=True)

            if n_items > 1:
                target = self.norm_vocab[key:key+1]
                sims = torch.mm(target, self.norm_vocab.T).squeeze(0)
                _, topk = torch.topk(sims, min(k_neighbors, n_items))
                neighbors_centered = (self.norm_vocab[topk] - target) * 3.0
                combined_manifold = torch.cat([neighbors_centered, F.normalize(tokens_centered, p=2, dim=-1)], dim=0)
            else:
                combined_manifold = F.normalize(tokens_centered, p=2, dim=-1)

            _, _, V = torch.linalg.svd(combined_manifold, full_matrices=False)
            tangent = F.normalize(V[:max_bases].clone(), p=2, dim=-1)
            self.affordance_cache[key] = tangent
            return tangent

    def decode_slots(self, z_slots_768: torch.Tensor):
        with self.lock:
            w_len = len(self.words)
            if w_len == 0:
                sims = torch.zeros((z_slots_768.shape[0], 1), device=DEVICE)
                soft_w = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, soft_w, ["СВОБОДНЫЙ_ДРЕЙФ"] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]

            if w_len == 1:
                sims = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                soft_w = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, soft_w, [self.words[0].upper()] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]

            z_norm = F.normalize(z_slots_768, p=2, dim=-1)
            sims = torch.mm(z_norm, self.norm_vocab.T)
            top = torch.argmax(sims, dim=-1)
            names = [self.words[int(i) % w_len].upper() for i in top]
            indices = [int(i) % w_len for i in top]
            return sims, torch.softmax(sims / 0.12, dim=-1), names, indices


class DeltaCycleSealEngine:
    def __init__(self, max_slots=16, seal_cycles=4, decay_cycles=2, seal_thresh=0.68):
        self.max_slots = max_slots
        self.seal_cycles = seal_cycles
        self.decay_cycles = decay_cycles
        self.seal_thresh = seal_thresh
        self.charge = np.zeros(max_slots, dtype=np.float32)
        self.mask = [False] * max_slots

    def step(self, n_sec, stabs, c_flow_np):
        for s in range(n_sec):
            rip = np.sum(np.abs(c_flow_np[s, :n_sec])) + np.sum(np.abs(c_flow_np[:n_sec, s]))
            if stabs[s] >= self.seal_thresh:
                charge_delta = 1.0 + float(np.clip(rip * 2.0, 0.0, 1.5))
                self.charge[s] = min(float(self.seal_cycles), self.charge[s] + charge_delta)
            else:
                loss = float(self.seal_cycles) / float(max(1, self.decay_cycles))
                self.charge[s] = max(0.0, self.charge[s] - loss)

            if self.mask[s] and self.charge[s] < (self.seal_cycles * 0.20):
                self.mask[s] = False

        ready = [s for s in range(n_sec) if not self.mask[s] and self.charge[s] >= self.seal_cycles]
        if len(ready) >= 2:
            for s in ready:
                self.mask[s] = True


class SelfSufficientHeterarchyRouter:
    ALL_ROLES = ["AFz", "F3", "F4", "FCz", "Fpz"]

    def __init__(self, users_setup=None):
        self.overrides = {}
        self.static_map = {}
        if users_setup and users_setup.strip().lower() not in ["none", "auto"]:
            for u in users_setup.split(";"):
                if ":" in u:
                    for pair in u.split(":")[1].split(","):
                        if "=" in pair:
                            r, d = pair.split("=")
                            if d.strip().isdigit():
                                self.static_map[int(d.strip())] = r.strip()

    def set_role(self, dev_idx: int, role: Optional[str] = None):
        self.overrides[dev_idx] = role

    def resolve(self, live_nodes: list) -> dict:
        resolved = {r: [] for r in self.ALL_ROLES}
        for dev_idx, node in enumerate(live_nodes):
            if dev_idx in self.overrides and self.overrides[dev_idx] is None:
                continue
            role = (
                self.overrides.get(dev_idx)
                or self.static_map.get(dev_idx)
                or getattr(node, "assigned_role", None)
                or ("AFz" if dev_idx == 0 else self.ALL_ROLES[dev_idx % len(self.ALL_ROLES)])
            )
            if role in resolved:
                resolved[role].append(node)
        return resolved


class UniversalColumnProcessor:
    def __init__(self, role: str, semantic: UniversalSemanticSpace,
                 seal_cycles: int = 4, decay_cycles: int = 2, seal_thresh=0.68, refresh_cycles: int = 2):
        self.role = role
        self.semantic = semantic
        self.monty = PredictiveColumnMonty(role, GEOM_16).to(DEVICE)
        self.seals = DeltaCycleSealEngine(MAX_SLOTS_CAPACITY, seal_cycles, decay_cycles, seal_thresh)
        self.stabs = np.zeros(MAX_SLOTS_CAPACITY, dtype=np.float32)

        self.buf_len = max(2, refresh_cycles)
        self.delta_cycle_buffer = torch.zeros((self.buf_len, MAX_SLOTS_CAPACITY, 768), dtype=torch.float32, device=DEVICE)
        self.buf_ptr = 0
        self.virtual_iplv = torch.zeros((32, 120), dtype=torch.float32, device=DEVICE)
        self.virtual_ripple = torch.zeros((32, 120), dtype=torch.float32, device=DEVICE)

    def synthesize_virtual_input(self, master_iplv: torch.Tensor, master_rip: torch.Tensor, master_error: torch.Tensor, t_now: float):
        if master_iplv is None:
            slots_t = torch.linspace(-math.pi, math.pi, 32, device=DEVICE).view(32, 1)
            spatial_phase = (GEOM_16["dip_x_t"] * math.cos(t_now * 0.4) + GEOM_16["dip_y_t"] * math.sin(t_now * 0.3)) * 0.15
            self.virtual_iplv = torch.sin(slots_t + spatial_phase + t_now * 1.5) * 0.45
            self.virtual_ripple = torch.cos(slots_t * 2.0 - spatial_phase + t_now * 2.2) * 0.35
            return self.virtual_iplv, self.virtual_ripple

        if self.role == "Fpz":
            target_iplv = (master_iplv * GEOM_16["macro_mask"] * 0.7) + (master_error * 0.3)
            alpha = 0.08
        elif self.role == "F3":
            target_iplv = torch.relu(master_iplv) * GEOM_16["left_mask"] * 1.4
            alpha = 0.25
        elif self.role == "F4":
            target_iplv = -torch.relu(-master_iplv) * GEOM_16["right_mask"] * 1.4
            alpha = 0.25
        elif self.role == "FCz":
            fast_diff = torch.roll(master_iplv, shifts=-1, dims=0) - master_iplv
            target_iplv = (fast_diff * 0.6 + master_iplv * 0.4) * GEOM_16["micro_mask"] * 1.5
            alpha = 0.50
        else:
            target_iplv = master_iplv
            alpha = 0.30

        self.virtual_iplv.mul_(1.0 - alpha).add_(target_iplv, alpha=alpha)
        self.virtual_ripple.mul_(1.0 - alpha).add_(master_rip, alpha=alpha)
        return self.virtual_iplv, self.virtual_ripple

    def process(self, raw_iplv: torch.Tensor, raw_rip: torch.Tensor, kinematics, is_physical: bool,
                n_sec: int, sec_idx: int, dl_phase: float, is_tick: bool, dt: float,
                top_down_sdr: Optional[torch.Tensor] = None,
                bottom_up_anom: Optional[torch.Tensor] = None,
                locked_concept: str = ""):
        monty_out = self.monty.forward_dynamics(
            raw_iplv, dt, top_down_parent_sdr=top_down_sdr, bottom_up_child_anomaly=bottom_up_anom
        )

        dynamic_wm = F.adaptive_avg_pool1d(raw_iplv.T.unsqueeze(0), n_sec).squeeze(0).T
        z_slots = torch.matmul(dynamic_wm, PHYSICAL_PROJECTION_768x120.T)

        with self.semantic.lock:
            words_copy = list(self.semantic.words)
            w_count = len(words_copy)
            _, soft_w, names, indices = self.semantic.decode_slots(z_slots)

            kuramoto_r = float(torch.clamp(torch.norm(dynamic_wm.mean(dim=0)) / 10.0, 0.0, 1.0).item())
            x_local = float(torch.clamp(torch.sum(dynamic_wm[sec_idx % n_sec] * DX_GPU_120) / 35.0, -1.0, 1.0).item())
            y_global = float(torch.clamp(torch.sum(dynamic_wm[sec_idx % n_sec] * DY_GPU_120) / 35.0, -1.0, 1.0).item())

            z_past, z_future = raw_iplv[0], raw_iplv[-1]
            npast, nfut = torch.norm(z_past), torch.norm(z_future)
            temporal_asymmetry = float(((nfut - npast) / (nfut + npast + 1e-6)).item())

            dyn_rip = F.adaptive_avg_pool1d(raw_rip.T.unsqueeze(0), n_sec).squeeze(0).T
            diff_rip = dyn_rip.unsqueeze(1) - dyn_rip.unsqueeze(0)
            c_flow_768 = torch.matmul(diff_rip, PHYSICAL_PROJECTION_768x120.T)
            c_flow_np = torch.norm(c_flow_768, dim=-1).cpu().numpy()

            net_leads = (c_flow_768.sum(dim=1) - c_flow_768.sum(dim=0)).sum(dim=-1).cpu().numpy()
            order = np.argsort(net_leads)[::-1]
            root_s = int(order[0])

            tree_16x768 = torch.zeros((16, 768), dtype=torch.float32, device=DEVICE)
            tree_77 = torch.zeros((77, 768), dtype=torch.float32, device=DEVICE)

            if locked_concept and locked_concept.strip():
                tree_77.copy_(self.semantic.get_77_embedding_gpu(locked_concept))
                tree_16x768.copy_(tree_77[5:21])
                dag_parts = [locked_concept.upper()]
                names = [locked_concept.upper()] * n_sec
            elif w_count == 0:
                tree_77.copy_(self.semantic.empty_embed)
                tree_16x768.copy_(monty_out["z_on_16x768"])
                dag_parts = ["СВОБОДНЫЙ_ДРЕЙФ"]
            elif w_count == 1:
                tree_77.copy_(self.semantic.get_77_embedding_gpu(words_copy[0]))
                tree_16x768.copy_(tree_77[5:21])
                dag_parts = [words_copy[0].upper()]
            else:
                k_choices = min(16, soft_w.shape[-1], w_count)
                top_w, top_i = torch.topk(soft_w[root_s, :w_count], k_choices)
                top_w = top_w / (top_w.sum() + 1e-6)

                for i in range(top_w.shape[0]):
                    w_idx = int(top_i[i]) % w_count
                    tree_77.add_(self.semantic.get_77_embedding_gpu(words_copy[w_idx]), alpha=top_w[i])

                accum = tree_77.clone()
                dag_parts = [names[root_s]]
                sec_w = (2.0 * math.pi) / float(n_sec)
                root_leads = c_flow_np[root_s, :n_sec]

                for s in order[1:]:
                    child_w = names[s]
                    t_child = self.semantic.get_77_embedding_gpu(child_w).clone()
                    t_child.add_((z_slots[s:s+1] / 12.0).expand(77, 768), alpha=0.25)
                    center = -math.pi * 0.5 + (s + 0.5) * sec_w
                    focus = 0.5 * (1.0 + math.cos(dl_phase - center))
                    is_seal = self.seals.mask[s]
                    w_slot = (0.25 + 0.55 * self.stabs[s]) * (0.4 + 0.6 * focus) * (1.3 if is_seal else 1.0)

                    if float(root_leads[s]) >= 0.03:
                        dot = torch.sum(t_child * accum, dim=-1, keepdim=True) / (torch.sum(accum**2, dim=-1, keepdim=True) + 1e-7)
                        ortho = t_child - dot * accum
                        no, nc = torch.norm(ortho, dim=-1, keepdim=True), torch.norm(t_child, dim=-1, keepdim=True)
                        ortho_n = (ortho / (no + 1e-6)) * nc * (no > (0.05 * nc)).float()
                        tree_77.add_(ortho_n, alpha=w_slot * 0.30)
                        accum.add_(ortho_n, alpha=0.15)
                        dag_parts.append(f"⊃ {child_w}")
                    else:
                        half_d = 768 // 2
                        nb = torch.norm(tree_77[:, :half_d], dim=-1, keepdim=True) + 1e-7
                        np_n = torch.norm(t_child[:, half_d:], dim=-1, keepdim=True) + 1e-7
                        tree_77[:, half_d:].mul_(0.8).add_((t_child[:, half_d:] / np_n * nb), alpha=0.22)
                        dag_parts.append(f"∥ {child_w}")

                tree_16x768.copy_(tree_77[5:21])

            if is_tick:
                self.delta_cycle_buffer[self.buf_ptr, :n_sec].copy_(z_slots)
                oldest_ptr = (self.buf_ptr + 1) % self.buf_len
                drift_cos = torch.sum(
                    F.normalize(z_slots, p=2, dim=-1) *
                    F.normalize(self.delta_cycle_buffer[oldest_ptr, :n_sec], p=2, dim=-1),
                    dim=-1
                ).cpu().numpy()
                self.buf_ptr = oldest_ptr

                for s in range(n_sec):
                    raw_stab = float(np.clip((drift_cos[s] * 0.5 + 0.5) * (0.2 + 0.8 * kuramoto_r), 0.0, 1.0))
                    self.stabs[s] = self.stabs[s] * 0.65 + raw_stab * 0.35

                self.seals.step(n_sec, self.stabs, c_flow_np)

            return {
                "is_physical": is_physical,
                "names": names, "indices": indices, "c_flow_np": c_flow_np,
                "tree_77": tree_77, "tree_16x768": tree_16x768,
                "dag_parts": dag_parts, "root_s": root_s,
                "stabs": self.stabs, "sealed_mask": list(self.seals.mask),
                "z_slots": z_slots, "kuramoto_r": kuramoto_r,
                "x_local": x_local, "y_global": y_global,
                "temporal_asymmetry": temporal_asymmetry,
                "kinematics": kinematics,
                "monty": monty_out
            }


class HeterarchyDeltaProcessor:
    def __init__(self, semantic, router, strength_low=0.50, burst_strength=0.85, burst_duration=1.2,
                 seal_cycles=4, decay_cycles=2, seal_thresh=0.68, refresh_cycles=2):
        self.lock = threading.RLock()
        self.semantic = semantic
        self.router = router
        self.strength_low, self.burst_strength, self.burst_duration = strength_low, burst_strength, burst_duration
        self.engines = {
            r: UniversalColumnProcessor(r, semantic, seal_cycles=seal_cycles, decay_cycles=decay_cycles,
                                        seal_thresh=seal_thresh, refresh_cycles=refresh_cycles)
            for r in self.router.ALL_ROLES
        }
        self.region_configs = {
            r: {"mode": "auto", "locked_concept": "", "tau_sec": REGION_TAU_SEC[r]}
            for r in self.router.ALL_ROLES
        }
        self.ladder_stage = 4
        self.measured_ratio = 4.0
        self.delta_armed = False
        self.last_root = self.semantic.words[0].upper() if self.semantic.words else "СВОБОДНЫЙ_ДРЕЙФ"
        self.burst_timer = 0.0
        self.epoch_wheel_phase = 0.0

    def configure_region_api(self, region: str, cfg: dict):
        with self.lock:
            if region in self.region_configs:
                self.region_configs[region].update(cfg)
                if "tau_sec" in cfg:
                    self.engines[region].monty.tau_sec = float(cfg["tau_sec"])
                print(f"🎛️ [REGION API] {region} -> {self.region_configs[region]}")

    def set_ladder_stage(self, stage: int, anchor_concept: str = ""):
        with self.lock:
            self.ladder_stage = int(np.clip(stage, 1, 4))
            lock_word = anchor_concept.strip() or (self.semantic.words[0] if self.semantic.words else "")
            if self.ladder_stage == 1:
                for r in ["Fpz", "AFz", "F3", "F4"]:
                    self.region_configs[r]["mode"] = "locked"
                    self.region_configs[r]["locked_concept"] = lock_word
                self.region_configs["FCz"]["mode"] = "auto"
                self.region_configs["FCz"]["locked_concept"] = ""
            elif self.ladder_stage == 2:
                for r in ["Fpz", "AFz"]:
                    self.region_configs[r]["mode"] = "locked"
                    self.region_configs[r]["locked_concept"] = lock_word
                for r in ["F3", "F4", "FCz"]:
                    self.region_configs[r]["mode"] = "auto"
                    self.region_configs[r]["locked_concept"] = ""
            elif self.ladder_stage == 3:
                self.region_configs["Fpz"]["mode"] = "locked"
                self.region_configs["Fpz"]["locked_concept"] = lock_word
                for r in ["AFz", "F3", "F4", "FCz"]:
                    self.region_configs[r]["mode"] = "auto"
                    self.region_configs[r]["locked_concept"] = ""
            else:
                for r in self.router.ALL_ROLES:
                    self.region_configs[r]["mode"] = "auto"
                    self.region_configs[r]["locked_concept"] = ""
            print(f"🪜 [API ЛЕСТНИЦА РЕГИОНОВ]: Ступень {self.ladder_stage}/4 (Якорь: '{lock_word}')")

    def step(self, frame, dt: float) -> tuple[torch.Tensor, float, dict]:
        with self.lock, self.semantic.lock:
            words_copy = list(self.semantic.words)
            w_count = len(words_copy)
            t_now = time.time()

            topo_lists = self.router.resolve(frame.nodes)
            th_hz, dl_hz = max(3.5, frame.theta_freq), max(0.5, frame.delta_freq)
            self.measured_ratio = self.measured_ratio * 0.95 + (th_hz / dl_hz) * 0.05
            k_th = int(np.clip(round(self.measured_ratio), 2, 8))
            n_sec = k_th * 2
            sec_w = (2.0 * math.pi) / float(n_sec)
            dl_phase = frame.delta_phase
            cur_sec = int(np.clip(int(((dl_phase + math.pi) % (2.0 * math.pi)) / sec_w), 0, n_sec - 1))

            is_tick = False
            if dl_phase < -1.5:
                self.delta_armed = True
            elif dl_phase > 0.5 and self.delta_armed:
                is_tick = True
                self.delta_armed = False

            real_iplv, real_rip, real_kin = {}, {}, {}
            all_ip, all_rp = [], []

            for r in self.router.ALL_ROLES:
                nodes_r = topo_lists.get(r, [])
                v_ip, v_rp = [], []
                for nd in nodes_r:
                    t_ip = torch.from_numpy(nd.iplv_32).to(DEVICE, dtype=torch.float32)
                    if torch.norm(t_ip) > 1e-4:
                        v_ip.append(t_ip)
                        v_rp.append(torch.from_numpy(nd.iplv_human_ripple).to(DEVICE, dtype=torch.float32))
                if v_ip:
                    m_ip = torch.stack(v_ip, dim=0).mean(dim=0)
                    m_rp = torch.stack(v_rp, dim=0).mean(dim=0)
                    real_iplv[r], real_rip[r], real_kin[r] = m_ip, m_rp, nodes_r[0].kinematics
                    all_ip.append(m_ip)
                    all_rp.append(m_rp)

            master_iplv = torch.stack(all_ip, dim=0).mean(dim=0) if all_ip else None
            master_rip  = torch.stack(all_rp, dim=0).mean(dim=0) if all_rp else None
            from neuro_heterarchy_core import Kinematics4D
            default_kin = next(iter(real_kin.values()), Kinematics4D())

            sdr_fcz = self.engines["FCz"].monty.l23_pooled_sdr
            sdr_f3  = self.engines["F3"].monty.l23_pooled_sdr
            sdr_f4  = self.engines["F4"].monty.l23_pooled_sdr
            sdr_afz = self.engines["AFz"].monty.l23_pooled_sdr
            sdr_fpz = self.engines["Fpz"].monty.l23_pooled_sdr

            anom_fcz = self.engines["FCz"].monty.upward_sdr_anomaly
            anom_f34 = 0.5 * (self.engines["F3"].monty.upward_sdr_anomaly + self.engines["F4"].monty.upward_sdr_anomaly)
            anom_afz = self.engines["AFz"].monty.upward_sdr_anomaly

            parent_sdr_map = {
                "FCz": 0.5 * (sdr_f3 + sdr_f4),
                "F3":  sdr_afz,
                "F4":  sdr_afz,
                "AFz": sdr_fpz,
                "Fpz": None
            }
            child_anom_map = {
                "FCz": None,
                "F3":  anom_fcz,
                "F4":  anom_fcz,
                "AFz": anom_f34,
                "Fpz": anom_afz
            }

            res = {}
            master_err = torch.zeros((32, 120), device=DEVICE)

            for r in self.router.ALL_ROLES:
                l_word = self.region_configs[r]["locked_concept"] if self.region_configs[r]["mode"] == "locked" else ""
                if r in real_iplv:
                    res[r] = self.engines[r].process(
                        real_iplv[r], real_rip[r], real_kin[r], True,
                        n_sec, cur_sec, dl_phase, is_tick, dt,
                        top_down_sdr=parent_sdr_map[r], bottom_up_anom=child_anom_map[r],
                        locked_concept=l_word
                    )
                    master_err = master_err + res[r]["monty"]["error_iplv"]

            for r in self.router.ALL_ROLES:
                if r not in res:
                    l_word = self.region_configs[r]["locked_concept"] if self.region_configs[r]["mode"] == "locked" else ""
                    v_ip, v_rp = self.engines[r].synthesize_virtual_input(master_iplv, master_rip, master_err, t_now)
                    res[r] = self.engines[r].process(
                        v_ip, v_rp, default_kin, False,
                        n_sec, cur_sec, dl_phase, is_tick, dt,
                        top_down_sdr=parent_sdr_map[r], bottom_up_anom=child_anom_map[r],
                        locked_concept=l_word
                    )

            hierarchy = ["AFz", "FCz", "F3", "F4", "Fpz"]
            apex = next((r for r in hierarchy if r in real_iplv), "AFz")
            primary = res[apex]

            target_77 = res["AFz"]["tree_77"].clone()
            root_w = res["AFz"]["dag_parts"][0]
            raw_root_idx = res["AFz"]["indices"][res["AFz"]["root_s"]]
            root_c_idx = (raw_root_idx % w_count) if w_count > 0 else 0
            clean_emb = self.semantic.last_active_embed if w_count == 0 else self.semantic.get_77_embedding_gpu(root_w)
            clean_norms = torch.norm(clean_emb, p=2, dim=-1, keepdim=True).clamp(min=1e-5)

            v_aff_16x768 = self.semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)
            raw_fcz_wave = res["FCz"]["monty"]["sagitta_16x768"] + res["FCz"]["monty"]["z_on_16x768"] * 0.5
            sem_action_16x768, fcz_persistence, _ = self.engines["FCz"].monty.action_integrator.step(
                raw_fcz_wave, v_aff_16x768, temp_ry=float(res["FCz"]["kinematics"].ry)
            )
            target_77[53:69].add_(sem_action_16x768, alpha=0.40)

            vocab_rows = self.semantic.norm_vocab.shape[0]
            if fcz_persistence > 0.65 and w_count > 1 and vocab_rows >= w_count and self.region_configs["FCz"]["mode"] != "locked":
                act_dir = F.normalize(sem_action_16x768.mean(dim=0, keepdim=True), p=2, dim=-1)
                cur_vec = self.semantic.norm_vocab[root_c_idx:root_c_idx + 1]
                shifted = F.normalize(cur_vec + act_dir * 0.50, p=2, dim=-1)
                best_nb = int(torch.argmax(torch.mm(shifted, self.semantic.norm_vocab[:w_count].T).squeeze(0)).item()) % w_count
                if best_nb != root_c_idx:
                    nb_emb = self.semantic.get_77_embedding_gpu(words_copy[best_nb])
                    target_77.mul_(0.65).add_(nb_emb, alpha=0.35)

            if self.region_configs["F3"]["mode"] != "locked":
                target_77[21:37].add_(res["F3"]["monty"]["z_on_16x768"], alpha=0.30)
            if self.region_configs["F4"]["mode"] != "locked":
                f4_ctx = res["F4"]["monty"]["z_off_16x768"] * 0.6 + res["F4"]["monty"]["temp_asym_16x768"] * 0.4
                target_77[37:53].add_(f4_ctx, alpha=0.30)

            max_surprise = max(res[r]["monty"]["surprise"] for r in self.router.ALL_ROLES)
            fpz_tension = float(np.clip(max_surprise * 0.55 + abs(res["Fpz"]["temporal_asymmetry"]) * 0.45, 0.0, 1.0))

            if self.region_configs["Fpz"]["mode"] != "locked" and w_count > 1:
                self.epoch_wheel_phase = self.epoch_wheel_phase % float(w_count)
                if max_surprise > 0.55:
                    self.epoch_wheel_phase = (self.epoch_wheel_phase + dt * (0.35 + max_surprise * 0.45)) % float(w_count)
                ep_a = int(math.floor(self.epoch_wheel_phase)) % w_count
                ep_b = (ep_a + 1) % w_count
                ep_alpha = 0.5 * (1.0 - math.cos((self.epoch_wheel_phase - math.floor(self.epoch_wheel_phase)) * math.pi))
                emb_a = self.semantic.get_77_embedding_gpu(words_copy[ep_a])
                emb_b = self.semantic.get_77_embedding_gpu(words_copy[ep_b])
                target_77[1:5].mul_(0.60).add_(emb_a[1:5] * (1.0 - ep_alpha) + emb_b[1:5] * ep_alpha, alpha=0.40)

            target_77[69:76].add_(primary["monty"]["sdr_tokens_7x768"], alpha=0.25)

            active_drive = float(np.clip(fcz_persistence * 0.70 + min(1.0, max_surprise) * 0.30, 0.0, 1.0))
            dyn_str = float(np.clip(
                self.strength_low + (self.burst_strength - self.strength_low) * active_drive,
                self.strength_low, 0.95
            ))

            target_77 = F.normalize(target_77, p=2, dim=-1) * clean_norms

            if root_w != self.last_root:
                self.last_root = root_w
                self.burst_timer = t_now
                print(f"🌲 [{apex} | Stage:{self.ladder_stage} | Pers:{fcz_persistence:.2f} | Loss:{primary['monty']['anomaly_loss']:.4f}]: {' '.join(primary['dag_parts'][:6])}")

            if (t_now - self.burst_timer) < self.burst_duration:
                dyn_str = max(dyn_str, self.burst_strength)

            regions_telemetry = {
                r: {
                    "is_physical": res[r]["is_physical"],
                    "mode": self.region_configs[r]["mode"],
                    "locked_concept": self.region_configs[r]["locked_concept"],
                    "concept": res[r]["dag_parts"][0],
                    "loss": round(res[r]["monty"]["anomaly_loss"], 5),
                    "surprise": round(res[r]["monty"]["surprise"], 3),
                    "tau_sec": self.engines[r].monty.tau_sec
                }
                for r in self.router.ALL_ROLES
            }

            telemetry = {
                "primary_role": apex, "primary_active": True,
                "ladder_stage": self.ladder_stage,
                "monty_loss": round(primary["monty"]["anomaly_loss"], 5),
                "persistence": round(fcz_persistence, 3),
                "surprise": round(max_surprise, 3),
                "regions_telemetry": regions_telemetry,
                "total_clock_sectors": n_sec, "current_clock_sec": cur_sec,
                "slot_names": primary["names"], "slot_indices": primary["indices"],
                "sealed_mask": primary["sealed_mask"], "slot_stabilities": primary["stabs"],
                "causal_flow_np": primary["c_flow_np"], "dag_log_parts": primary["dag_parts"],
                "dl_phase": dl_phase, "is_delta_tick": is_tick, "dyn_strength": dyn_str,
                "active_vocab_size": w_count,
                "fpz_active": True, "fpz_tension": fpz_tension, "fpz_concept": res["Fpz"]["dag_parts"][0],
                "f3_active": True, "f3_concept": res["F3"]["dag_parts"][0],
                "f3_x_local": res["F3"]["x_local"], "f3_y_global": res["F3"]["y_global"],
                "f3_kuramoto": res["F3"]["kuramoto_r"], "f3_slot_names": res["F3"]["names"],
                "f4_active": True, "f4_concept": res["F4"]["dag_parts"][0], "f4_entropy": active_drive,
                "fcz_active": True, "fcz_concept": res["FCz"]["dag_parts"][0],
                "fcz_lx": float(res["FCz"]["kinematics"].lx),
                "fcz_ly": float(res["FCz"]["kinematics"].ly),
                "fcz_rx": float(res["FCz"]["kinematics"].rx),
                "fcz_ry": float(res["FCz"]["kinematics"].ry)
            }
            return target_77, dyn_str, telemetry
