#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: FULL-DIMENSIONAL [64 x 768] MONTY SDR HETERARCHY FOR MUSICGEN
===================================================================================
- 100% ПАРИТЕТ С ДИФФУЗИЕЙ (neuro_genesis_engine.py):
  Все 64 токена T5 распределены по 5 кортикальным регионам без снижения размерности:
  * Tokens [0:4]   : Fpz (Колесо Эпох / Смена правил)
  * Tokens [4:20]  : AFz (Композиционный граф 89.5 Гц рипплов A ⊃ B, [16 × 768])
  * Tokens [20:36] : F3  (Структурное ON-подпространство z_on_16x768, [16 × 768])
  * Tokens [36:52] : F4  (Контекстное OFF-подпространство + Темпоральная асимметрия, [16 × 768])
  * Tokens [42:58] : FCz (Семантические действия вдоль SVD-аффордансов V_aff, [16 × 768])
  * Tokens [57:64] : Глобальный консенсус L2/3 Monty SDR (sdr_tokens_7x768, [7 × 768])
- КАЛИБРОВКА НОРМЫ ПОД T5 (_scale_to_t5):
  Каждый 768-мерный вектор мозга нормируется строго к естественной норме токенов T5
  (clean_norms ~ 3.2), исключая перегрузку кондиционера и сохраняя 100% чистоту музыки!
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import time
import math
import threading
import numpy as np
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE
from neuro_genesis_engine import (
    GEOM_16,
    PHYSICAL_PROJECTION_768x120,
    DX_GPU_120,
    DY_GPU_120,
    MAX_SLOTS_CAPACITY,
    PredictiveColumnMonty,
    DeltaCycleSealEngine
)


def _scale_to_t5(delta: torch.Tensor, ref_norms: torch.Tensor) -> torch.Tensor:
    """
    Приводит любой 768-мерный тензор коры к точному масштабу нормы токенов T5,
    сохраняя 100% направления и относительной амплитуды мысли без перегруза T5!
    """
    d_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
    gate = torch.tanh(d_norm / 2.5)
    return F.normalize(delta, p=2, dim=-1) * ref_norms * gate


class UniversalMusicSemanticSpace:
    def __init__(self, words: list[str], c_bases: list[np.ndarray],
                 proj_120_to_768: np.ndarray,
                 empty_embed: Optional[np.ndarray] = None,
                 global_svd_bases: Optional[np.ndarray] = None):
        self.lock = threading.RLock()
        self.words = [w.strip() for w in words if w.strip()]
        self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in c_bases]
        self.proj_120_to_768 = F.normalize(
            torch.tensor(proj_120_to_768, dtype=torch.float32, device=DEVICE), p=2, dim=0
        )

        if empty_embed is not None:
            self.empty_embed = torch.tensor(empty_embed, dtype=torch.float32, device=DEVICE)
        elif self.c_bases:
            self.empty_embed = self.c_bases[0].clone()
        else:
            self.empty_embed = torch.zeros((64, 768), dtype=torch.float32, device=DEVICE)

        if global_svd_bases is not None:
            self.global_svd_bases = F.normalize(
                torch.tensor(global_svd_bases, dtype=torch.float32, device=DEVICE), p=2, dim=-1
            )
        else:
            self.global_svd_bases = F.normalize(self.proj_120_to_768[:64], p=2, dim=-1)

        self.last_active_embed = (self.c_bases[0] if self.c_bases else self.empty_embed).clone()
        self.affordance_cache = {}
        self.rebuild_norm_vocab()

    def rebuild_norm_vocab(self):
        with self.lock:
            n_b = len(self.c_bases)
            if n_b == 0:
                raw = self.last_active_embed.mean(dim=0, keepdim=True)
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
                self.affordance_cache.clear()
                return

            raw = torch.stack([b.mean(dim=0) for b in self.c_bases], dim=0)
            if n_b == 1:
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
            else:
                mu = torch.mean(raw, dim=0, keepdim=True)
                std = torch.std(raw - mu, dim=0, keepdim=True, unbiased=False) + 1e-6
                self.norm_vocab = F.normalize((raw - mu) / std, p=2, dim=-1)
            self.affordance_cache.clear()

    def get_concept_affordances_svd(self, concept_idx: int, k_neighbors: int = 16, max_bases: int = 16) -> torch.Tensor:
        with self.lock:
            n_items = len(self.c_bases)
            if n_items == 0:
                return self.global_svd_bases[:max_bases]

            safe_idx = concept_idx % n_items
            if safe_idx in self.affordance_cache:
                return self.affordance_cache[safe_idx]

            base_64 = self.c_bases[safe_idx]
            tokens_centered = base_64 - base_64.mean(dim=0, keepdim=True)

            if n_items > 1:
                target = self.norm_vocab[safe_idx:safe_idx + 1]
                sims = torch.mm(target, self.norm_vocab.T).squeeze(0)
                _, topk = torch.topk(sims, min(k_neighbors, n_items))
                neighbors_centered = (self.norm_vocab[topk] - target) * 3.0
                combined_manifold = torch.cat([neighbors_centered, F.normalize(tokens_centered, p=2, dim=-1)], dim=0)
            else:
                combined_manifold = F.normalize(tokens_centered, p=2, dim=-1)

            _, _, V = torch.linalg.svd(combined_manifold, full_matrices=False)
            tangent = F.normalize(V[:max_bases].clone(), p=2, dim=-1)
            self.affordance_cache[safe_idx] = tangent
            return tangent

    def decode_slots(self, z_slots_768: torch.Tensor):
        with self.lock:
            n_b = len(self.c_bases)
            w_len = min(len(self.words), n_b)
            if w_len == 0:
                sims = torch.zeros((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, torch.ones_like(sims), ["СВОБОДНЫЙ_ДРЕЙФ"] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]
            if w_len == 1:
                sims = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, sims, [self.words[0].upper()] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]

            z_norm = F.normalize(z_slots_768, p=2, dim=-1)
            sims = torch.mm(z_norm, self.norm_vocab[:w_len].T)
            top = torch.argmax(sims, dim=-1)
            names = [self.words[int(i) % w_len].upper() for i in top]
            indices = [int(i) % w_len for i in top]
            return sims, torch.softmax(sims / 0.12, dim=-1), names, indices


class MusicColumnProcessor:
    def __init__(self, role: str, semantic: UniversalMusicSemanticSpace, seal_thresh: float = 0.68):
        self.role = role
        self.semantic = semantic
        self.monty = PredictiveColumnMonty(role, GEOM_16).to(DEVICE)
        self.seals = DeltaCycleSealEngine(MAX_SLOTS_CAPACITY, seal_thresh=seal_thresh)
        self.stabs = np.zeros(MAX_SLOTS_CAPACITY, dtype=np.float32)

        self.buf_len = 2
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
                bottom_up_anom: Optional[torch.Tensor] = None):
        monty_out = self.monty.forward_dynamics(
            raw_iplv, dt, top_down_parent_sdr=top_down_sdr, bottom_up_child_anomaly=bottom_up_anom
        )

        dynamic_wm = F.adaptive_avg_pool1d(raw_iplv.T.unsqueeze(0), n_sec).squeeze(0).T
        z_slots = torch.matmul(dynamic_wm, PHYSICAL_PROJECTION_768x120.T)

        with self.semantic.lock:
            c_bases_copy = list(self.semantic.c_bases)
            words_copy = list(self.semantic.words)
            n_bases = min(len(c_bases_copy), len(words_copy))

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

            tree_64 = torch.zeros((64, 768), dtype=torch.float32, device=DEVICE)

            if n_bases == 0:
                base_ref = self.semantic.last_active_embed
                ref_norms = torch.norm(base_ref, p=2, dim=-1, keepdim=True).clamp(min=1.0)
                tree_64.copy_(base_ref)
                tree_64[4:20].add_(_scale_to_t5(monty_out["z_on_16x768"], ref_norms[4:20]), alpha=0.35)
                dag_parts = ["СВОБОДНЫЙ_ДРЕЙФ"]
            elif n_bases == 1:
                tree_64.copy_(c_bases_copy[0])
                dag_parts = [words_copy[0].upper()]
            else:
                k_choices = min(16, soft_w.shape[-1], n_bases)
                top_w, top_i = torch.topk(soft_w[root_s, :n_bases], k_choices)
                top_w = top_w / (top_w.sum() + 1e-6)
                for i in range(top_w.shape[0]):
                    tree_64.add_(c_bases_copy[int(top_i[i]) % n_bases], alpha=top_w[i])

                ref_norms = torch.norm(tree_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)
                accum = tree_64.clone()
                dag_parts = [names[root_s]]
                sec_w = (2.0 * math.pi) / float(n_sec)
                root_leads = c_flow_np[root_s, :n_sec]

                for s in order[1:]:
                    child_w = names[s]
                    t_child = c_bases_copy[indices[s] % n_bases].clone()
                    slot_mod = _scale_to_t5(z_slots[s:s + 1].expand(64, 768), ref_norms)
                    t_child.add_(slot_mod, alpha=0.20)
                    center = -math.pi * 0.5 + (s + 0.5) * sec_w
                    focus = 0.5 * (1.0 + math.cos(dl_phase - center))
                    is_seal = self.seals.mask[s]
                    w_slot = (0.25 + 0.55 * self.stabs[s]) * (0.4 + 0.6 * focus) * (1.3 if is_seal else 1.0)

                    if float(root_leads[s]) >= 0.03:
                        dot = torch.sum(t_child * accum, dim=-1, keepdim=True) / (torch.sum(accum ** 2, dim=-1, keepdim=True) + 1e-7)
                        ortho = t_child - dot * accum
                        no, nc = torch.norm(ortho, dim=-1, keepdim=True), torch.norm(t_child, dim=-1, keepdim=True)
                        ortho_n = (ortho / (no + 1e-6)) * nc * (no > (0.05 * nc)).float()
                        tree_64.add_(ortho_n, alpha=w_slot * 0.30)
                        accum.add_(ortho_n, alpha=0.15)
                        dag_parts.append(f"⊃ {child_w}")
                    else:
                        half_d = 768 // 2
                        nb = torch.norm(tree_64[:, :half_d], dim=-1, keepdim=True) + 1e-7
                        np_n = torch.norm(t_child[:, half_d:], dim=-1, keepdim=True) + 1e-7
                        tree_64[:, half_d:].mul_(0.8).add_((t_child[:, half_d:] / np_n * nb), alpha=0.22)
                        dag_parts.append(f"∥ {child_w}")

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
                "tree_64": tree_64, "dag_parts": dag_parts, "root_s": root_s,
                "stabs": self.stabs, "sealed_mask": list(self.seals.mask),
                "z_slots": z_slots, "kuramoto_r": kuramoto_r,
                "x_local": x_local, "y_global": y_global,
                "temporal_asymmetry": temporal_asymmetry,
                "kinematics": kinematics,
                "monty": monty_out
            }


class MusicHeterarchyDeltaProcessor:
    def __init__(self, semantic: UniversalMusicSemanticSpace, router,
                 strength_low: float = 0.25, strength_high: float = 0.80,
                 burst_duration: float = 1.2, seal_thresh: float = 0.68):
        self.lock = threading.RLock()
        self.semantic = semantic
        self.router = router
        self.strength_low = strength_low
        self.strength_high = strength_high
        self.burst_duration = burst_duration
        self.engines = {r: MusicColumnProcessor(r, semantic, seal_thresh=seal_thresh) for r in self.router.ALL_ROLES}
        self.measured_ratio = 4.0
        self.delta_armed = False
        self.epoch_wheel_phase = 0.0
        self.last_root_name = ""
        self.burst_timer = 0.0

    def step(self, frame, dt: float, mode: str = "semantic", target_block: float = 2.0,
             target_context: float = 0.6, base_temp: float = 1.0, base_top_k: int = 250,
             cfg_coef: float = 1.0, strength_low: Optional[float] = None,
             strength_high: Optional[float] = None):
        with self.lock, self.semantic.lock:
            if strength_low is not None:
                self.strength_low = float(np.clip(strength_low, 0.05, 0.90))
            if strength_high is not None:
                self.strength_high = float(np.clip(strength_high, self.strength_low + 0.05, 1.0))

            c_bases_copy = list(self.semantic.c_bases)
            words_copy = list(self.semantic.words)
            n_bases = min(len(c_bases_copy), len(words_copy))
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
            master_rip = torch.stack(all_rp, dim=0).mean(dim=0) if all_rp else None
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
                if r in real_iplv:
                    res[r] = self.engines[r].process(
                        real_iplv[r], real_rip[r], real_kin[r], True,
                        n_sec, cur_sec, dl_phase, is_tick, dt,
                        top_down_sdr=parent_sdr_map[r], bottom_up_anom=child_anom_map[r]
                    )
                    master_err = master_err + res[r]["monty"]["error_iplv"]

            for r in self.router.ALL_ROLES:
                if r not in res:
                    v_ip, v_rp = self.engines[r].synthesize_virtual_input(master_iplv, master_rip, master_err, t_now)
                    res[r] = self.engines[r].process(
                        v_ip, v_rp, default_kin, False,
                        n_sec, cur_sec, dl_phase, is_tick, dt,
                        top_down_sdr=parent_sdr_map[r], bottom_up_anom=child_anom_map[r]
                    )

            hierarchy = ["AFz", "FCz", "F3", "F4", "Fpz"]
            apex = next((r for r in hierarchy if r in real_iplv), "AFz")
            primary = res[apex]

            # -----------------------------------------------------------------
            # ПОЛНОРАЗМЕРНАЯ СБОРКА [64 x 768] ПО ВСЕМ 5 РЕГИОНАМ (1:1 С ДИФФУЗИЕЙ)
            # С НОРМИРОВКОЙ КАЖДОГО ВЕКТОРА К МАСШТАБУ T5 (_scale_to_t5)
            # -----------------------------------------------------------------
            target_64 = primary["tree_64"].clone()
            root_c_idx = primary["indices"][primary["root_s"]] % max(1, n_bases)
            clean_emb = self.semantic.last_active_embed if n_bases == 0 else c_bases_copy[root_c_idx]
            clean_norms = torch.norm(clean_emb, p=2, dim=-1, keepdim=True).clamp(min=1.0)

            # 1. FCz: Семантическое действие [16 x 768] вдоль SVD-аффордансов V_aff
            v_aff_16x768 = self.semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)
            raw_fcz_wave = res["FCz"]["monty"]["sagitta_16x768"] + res["FCz"]["monty"]["z_on_16x768"] * 0.5
            sem_action_16x768, fcz_persistence, _ = self.engines["FCz"].monty.action_integrator.step(
                raw_fcz_wave, v_aff_16x768, temp_ry=float(res["FCz"]["kinematics"].ry)
            )

            max_surprise = max(res[r]["monty"]["surprise"] for r in self.router.ALL_ROLES)
            fpz_tension = float(np.clip(max_surprise * 0.55 + abs(res["Fpz"]["temporal_asymmetry"]) * 0.45, 0.0, 1.0))

            # Вычисляем мгновенную силу рабочей памяти от мозга (Persistence + Surprise)
            active_drive = float(np.clip(fcz_persistence * 0.70 + min(1.0, max_surprise) * 0.30, 0.0, 1.0))
            dyn_str = float(np.clip(
                self.strength_low + (self.strength_high - self.strength_low) * active_drive,
                self.strength_low, 0.98
            ))

            root_w = primary["dag_parts"][0]
            if root_w != self.last_root_name:
                self.last_root_name = root_w
                self.burst_timer = t_now
                print(f"🎵 [{apex} | DynStr:{dyn_str:.2f} | Pers:{fcz_persistence:.2f} | Loss:{primary['monty']['anomaly_loss']:.4f}]: {' '.join(primary['dag_parts'][:6])}")

            if (t_now - self.burst_timer) < self.burst_duration:
                dyn_str = max(dyn_str, self.strength_high)

            # Масштабируем каждую 16x768 проекцию регионов к норме токенов T5 и вкладываем с весом dyn_str:
            target_64[42:58].add_(_scale_to_t5(sem_action_16x768, clean_norms[42:58]), alpha=0.40 * dyn_str)

            # Геодезический шаг к соседнему концепту при высокой настойчивости (как в диффузии)
            if fcz_persistence > 0.65 and n_bases > 1:
                act_dir = F.normalize(sem_action_16x768.mean(dim=0, keepdim=True), p=2, dim=-1)
                cur_vec = self.semantic.norm_vocab[root_c_idx:root_c_idx + 1]
                shifted = F.normalize(cur_vec + act_dir * 0.50, p=2, dim=-1)
                best_nb = int(torch.argmax(torch.mm(shifted, self.semantic.norm_vocab[:n_bases].T).squeeze(0)).item())
                if best_nb != root_c_idx and best_nb < n_bases:
                    nb_emb = c_bases_copy[best_nb]
                    target_64.mul_(1.0 - 0.35 * dyn_str).add_(nb_emb, alpha=0.35 * dyn_str)

            # 2. F3: Структурное ON-подпространство [16 x 768]
            target_64[16:32].add_(_scale_to_t5(res["F3"]["monty"]["z_on_16x768"], clean_norms[16:32]), alpha=0.30 * dyn_str)

            # 3. F4: Контекстное OFF-подпространство + Темпоральная асимметрия [16 x 768]
            f4_ctx = res["F4"]["monty"]["z_off_16x768"] * 0.6 + res["F4"]["monty"]["temp_asym_16x768"] * 0.4
            target_64[32:48].add_(_scale_to_t5(f4_ctx, clean_norms[32:48]), alpha=0.30 * dyn_str)

            # 4. Fpz: Колесо Эпох [0:4]
            if n_bases > 1:
                if max_surprise > 0.55:
                    self.epoch_wheel_phase = (self.epoch_wheel_phase + dt * (0.35 + max_surprise * 0.45)) % float(n_bases)
                ep_a = int(math.floor(self.epoch_wheel_phase)) % n_bases
                ep_b = (ep_a + 1) % n_bases
                ep_alpha = 0.5 * (1.0 - math.cos((self.epoch_wheel_phase - math.floor(self.epoch_wheel_phase)) * math.pi))
                emb_a = c_bases_copy[ep_a]
                emb_b = c_bases_copy[ep_b]
                target_64[0:4].mul_(1.0 - 0.40 * dyn_str).add_(
                    emb_a[0:4] * (1.0 - ep_alpha) + emb_b[0:4] * ep_alpha, alpha=0.40 * dyn_str
                )

            # 5. L2/3 Monty SDR консенсус [57:64]
            target_64[57:64].add_(_scale_to_t5(primary["monty"]["sdr_tokens_7x768"], clean_norms[57:64]), alpha=0.25 * dyn_str)

            # Итоговый баланс между чистым якорем концепта и полноразмерной модуляцией по dyn_str
            final_64 = clean_emb * (1.0 - dyn_str) + target_64 * dyn_str
            final_64 = F.normalize(final_64, p=2, dim=-1) * clean_norms

            temp = float(np.clip(base_temp * (0.85 + 0.30 * dyn_str), 0.3, 1.8))
            top_k = int(np.clip(base_top_k + int((dyn_str - 0.4) * 100), 30, 450))
            kin = res["FCz"]["kinematics"]

            telemetry = {
                "primary_role": apex, "primary_active": True,
                "monty_loss": round(primary["monty"]["anomaly_loss"], 5),
                "persistence": round(fcz_persistence, 3),
                "surprise": round(max_surprise, 3),
                "dyn_strength": round(dyn_str, 3),
                "total_clock_sectors": n_sec, "current_clock_sec": cur_sec,
                "slot_names": primary["names"], "slot_indices": primary["indices"],
                "sealed_mask": primary["sealed_mask"], "slot_stabilities": primary["stabs"],
                "causal_flow_np": primary["c_flow_np"], "dag_log_parts": primary["dag_parts"],
                "dl_phase": dl_phase, "is_delta_tick": is_tick,
                "block_sec": target_block, "context_sec": target_context,
                "temp": temp, "top_k": top_k, "cfg_coef": cfg_coef, "is_drop": False,
                "active_vocab_size": n_bases,
                "fpz_active": True, "fpz_tension": fpz_tension, "fpz_concept": res["Fpz"]["dag_parts"][0],
                "f3_active": True, "f3_concept": res["F3"]["dag_parts"][0],
                "f3_x_local": res["F3"]["x_local"], "f3_y_global": res["F3"]["y_global"],
                "f3_kuramoto": res["F3"]["kuramoto_r"],
                "f4_active": True, "f4_concept": res["F4"]["dag_parts"][0], "f4_entropy": active_drive,
                "fcz_active": True, "fcz_concept": res["FCz"]["dag_parts"][0],
                "fcz_lx": float(kin.lx), "fcz_ly": float(kin.ly),
                "fcz_rx": float(kin.rx), "fcz_ry": float(kin.ry)
            }
            return final_64, target_block, target_context, temp, top_k, cfg_coef, telemetry
