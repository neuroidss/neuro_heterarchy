#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: FULL-DIMENSIONAL [64 x 768] MONTY SDR HETERARCHY FOR MUSICGEN
===================================================================================
- ЧИСТАЯ МУЗЫКАЛЬНОСТЬ ПРИ ЛЮБОМ ЧИСЛЕ РАЗНОРОДНЫХ КОНЦЕПТОВ В ЛОРЕ:
  1) Фокусированная диада/триада (1 ведущий концепт ритма/тональности Root +
     1 ортогонально вложенный дочерний концепт тембра Child A ⊃ B) вместо
     одновременного смешивания 16 несовместимых жанров в кашу.
  2) Тангенциальная SVD-проекция всех 5 регионов Монти ([64 × 768]):
     волны F3, F4, FCz и SDR проецируются на собственное 16×768 SVD-подпространство
     V_aff активного концепта и нормируются через _scale_to_t5.
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
    d_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
    gate = torch.tanh(d_norm / 2.5)
    return F.normalize(delta, p=2, dim=-1) * ref_norms * gate


def _project_into_svd_manifold(wave_Nx768: torch.Tensor, v_aff_16x768: torch.Tensor, ref_norms: torch.Tensor) -> torch.Tensor:
    """
    Проецирует полноразмерную 768-D волну региона на ортонормированный SVD-базис V_aff
    текущего музыкального концепта (или глобального словаря T5), чтобы модуляция
    меняла тембр, ритм и гармонию, но НИКОГДА не разрушала музыкальность!
    """
    coords = torch.matmul(wave_Nx768, v_aff_16x768.T)
    projected = torch.matmul(coords, v_aff_16x768)
    return _scale_to_t5(projected, ref_norms)


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

    def get_concept_affordances_svd(self, concept_idx: int, k_neighbors: int = 8, max_bases: int = 16) -> torch.Tensor:
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
                neighbors_centered = (self.norm_vocab[topk] - target) * 2.0
                combined_manifold = torch.cat([F.normalize(tokens_centered, p=2, dim=-1), neighbors_centered], dim=0)
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
            # Острый softmax (температура 0.04 вместо 0.12), чтобы не смешивать далекие жанры в серую кашу!
            return sims, torch.softmax(sims / 0.04, dim=-1), names, indices


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
                v_global = self.semantic.global_svd_bases[:16]
                tree_64[4:20].add_(
                    _project_into_svd_manifold(monty_out["z_on_16x768"], v_global, ref_norms[4:20]),
                    alpha=0.40
                )
                dag_parts = ["СВОБОДНЫЙ_ДРЕЙФ"]
            elif n_bases == 1:
                tree_64.copy_(c_bases_copy[0])
                dag_parts = [words_copy[0].upper()]
            else:
                # ФОКУСИРОВАННАЯ МУЗЫКАЛЬНАЯ КОМПОЗИЦИЯ (Root + 1 ортогональный Child A ⊃ B):
                # Не складываем 16 жанров в серый суп, а берем чистый ведущий концепт Root
                # и встраиваем в него 1 сильнейший дочерний концепт Child!
                root_c_idx = indices[root_s] % n_bases
                tree_64.copy_(c_bases_copy[root_c_idx])
                ref_norms = torch.norm(tree_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)
                dag_parts = [names[root_s]]
                root_leads = c_flow_np[root_s, :n_sec]

                # Находим главный дочерний слот с другим концептом
                child_s = next((int(s) for s in order[1:] if (indices[int(s)] % n_bases) != root_c_idx), int(order[1]) if len(order) > 1 else root_s)
                if child_s != root_s:
                    child_c_idx = indices[child_s] % n_bases
                    child_w = names[child_s]
                    t_child = c_bases_copy[child_c_idx].clone()
                    is_seal = self.seals.mask[child_s]
                    w_slot = (0.20 + 0.45 * self.stabs[child_s]) * (1.25 if is_seal else 1.0)

                    if float(root_leads[child_s]) >= 0.03:
                        # A ⊃ B: ортогональная встройка тембра Child внутрь ритма и тональности Root
                        dot = torch.sum(t_child * tree_64, dim=-1, keepdim=True) / (torch.sum(tree_64 ** 2, dim=-1, keepdim=True) + 1e-7)
                        ortho = t_child - dot * tree_64
                        no = torch.norm(ortho, dim=-1, keepdim=True)
                        ortho_n = (ortho / (no + 1e-6)) * ref_norms
                        tree_64[4:20].add_(ortho_n[4:20], alpha=w_slot * 0.35)
                        dag_parts.append(f"⊃ {child_w}")
                    else:
                        # A ∥ B: разделение подпространств только в окне AFz [4:20]
                        half_d = 768 // 2
                        nb = torch.norm(tree_64[4:20, :half_d], dim=-1, keepdim=True) + 1e-7
                        np_n = torch.norm(t_child[4:20, half_d:], dim=-1, keepdim=True) + 1e-7
                        tree_64[4:20, half_d:].mul_(0.82).add_((t_child[4:20, half_d:] / np_n * nb), alpha=w_slot * 0.25)
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
                 strength_low: float = 0.25, strength_high: float = 0.75,
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
             target_context: float = 1.2, base_temp: float = 1.0, base_top_k: int = 250,
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
            anom_afz = self.engines["AFz"].upward_sdr_anomaly if hasattr(self.engines["AFz"], "upward_sdr_anomaly") else self.engines["AFz"].monty.upward_sdr_anomaly

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

            target_64 = primary["tree_64"].clone()
            root_c_idx = primary["indices"][primary["root_s"]] % max(1, n_bases)
            clean_emb = self.semantic.last_active_embed if n_bases == 0 else c_bases_copy[root_c_idx]
            clean_norms = torch.norm(clean_emb, p=2, dim=-1, keepdim=True).clamp(min=1.0)

            # Извлекаем 16x768 SVD-базис музыкальных аффордансов текущего ведущего концепта
            v_aff_16x768 = self.semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)
            raw_fcz_wave = res["FCz"]["monty"]["sagitta_16x768"] + res["FCz"]["monty"]["z_on_16x768"] * 0.5
            sem_action_16x768, fcz_persistence, _ = self.engines["FCz"].monty.action_integrator.step(
                raw_fcz_wave, v_aff_16x768, temp_ry=float(res["FCz"]["kinematics"].ry)
            )

            max_surprise = max(res[r]["monty"]["surprise"] for r in self.router.ALL_ROLES)
            fpz_tension = float(np.clip(max_surprise * 0.55 + abs(res["Fpz"]["temporal_asymmetry"]) * 0.45, 0.0, 1.0))

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

            # 1. FCz [42:58]: Семантические действия вдоль V_aff
            target_64[42:58].add_(_scale_to_t5(sem_action_16x768, clean_norms[42:58]), alpha=0.40 * dyn_str)

            # 2. F3 [16:32] и F4 [32:48]: проецируются в SVD-многообразие V_aff, чтобы сохранять музыкальный строй!
            f3_musical = _project_into_svd_manifold(res["F3"]["monty"]["z_on_16x768"], v_aff_16x768, clean_norms[16:32])
            target_64[16:32].add_(f3_musical, alpha=0.30 * dyn_str)

            f4_ctx = res["F4"]["monty"]["z_off_16x768"] * 0.6 + res["F4"]["monty"]["temp_asym_16x768"] * 0.4
            f4_musical = _project_into_svd_manifold(f4_ctx, v_aff_16x768, clean_norms[32:48])
            target_64[32:48].add_(f4_musical, alpha=0.30 * dyn_str)

            # 3. Fpz [0:4]: Колесо Эпох (включается только при высоком удивлении surprise > 0.65)
            if n_bases > 1 and max_surprise > 0.65:
                self.epoch_wheel_phase = (self.epoch_wheel_phase + dt * 0.25) % float(n_bases)
                ep_a = int(math.floor(self.epoch_wheel_phase)) % n_bases
                ep_b = (ep_a + 1) % n_bases
                ep_alpha = 0.5 * (1.0 - math.cos((self.epoch_wheel_phase - math.floor(self.epoch_wheel_phase)) * math.pi))
                emb_a = c_bases_copy[ep_a]
                emb_b = c_bases_copy[ep_b]
                target_64[0:4].mul_(1.0 - 0.35 * dyn_str).add_(
                    emb_a[0:4] * (1.0 - ep_alpha) + emb_b[0:4] * ep_alpha, alpha=0.35 * dyn_str
                )

            # 4. L2/3 Monty SDR консенсус [57:64]
            sdr_musical = _project_into_svd_manifold(primary["monty"]["sdr_tokens_7x768"], v_aff_16x768, clean_norms[57:64])
            target_64[57:64].add_(sdr_musical, alpha=0.25 * dyn_str)

            final_64 = clean_emb * (1.0 - dyn_str) + target_64 * dyn_str
            final_64 = F.normalize(final_64, p=2, dim=-1) * clean_norms

            temp = float(np.clip(base_temp * (0.88 + 0.25 * dyn_str), 0.3, 1.8))
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
