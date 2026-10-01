#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: PURE GPU TENSOR HETERARCHY ENGINE FOR MUSICGEN (TBT 2.0)
===================================================================================
- Никаких загрузок тяжелых нейросетей внутри!
- Полная реализация Layer A (UniversalColumnProcessor) и Layer B (Heterarchy).
- Поддержка горячего назначения ролей для любых подключенных устройств.
===================================================================================
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y, DX_120, DY_120

MAX_SLOTS_CAPACITY = 16

I_IDX, J_IDX = np.triu_indices(16, k=1)
DIPOLE_X = (COORDS_X[I_IDX] + COORDS_X[J_IDX]) / 2.0
DIPOLE_Y = (COORDS_Y[I_IDX] + COORDS_Y[J_IDX]) / 2.0
dipoles_x_t = torch.tensor(DIPOLE_X, dtype=torch.float32, device=DEVICE).view(1, 120)
dipoles_y_t = torch.tensor(DIPOLE_Y, dtype=torch.float32, device=DEVICE).view(1, 120)

col_x = torch.linspace(-12.0, 12.0, 32, device=DEVICE, dtype=torch.float32)
col_y = torch.linspace(-12.0, 12.0, 24, device=DEVICE, dtype=torch.float32)
grid_y, grid_x = torch.meshgrid(col_y, col_x, indexing='ij')
cols_x_t = grid_x.reshape(768, 1)
cols_y_t = grid_y.reshape(768, 1)

d_sq = (cols_x_t - dipoles_x_t)**2 + (cols_y_t - dipoles_y_t)**2
PHYSICAL_PROJECTION_768x120 = torch.exp(-d_sq / 32.0)
PHYSICAL_PROJECTION_768x120 = torch.nn.functional.normalize(PHYSICAL_PROJECTION_768x120, p=2, dim=1)

DX_GPU_120 = torch.tensor(DX_120, dtype=torch.float32, device=DEVICE)
DY_GPU_120 = torch.tensor(DY_120, dtype=torch.float32, device=DEVICE)

class UniversalMusicSemanticSpace:
    def __init__(self, words: list[str], c_bases: list[np.ndarray], proj_120_to_768: np.ndarray):
        self.words = words
        self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in c_bases]
        self.proj_120_to_768 = torch.tensor(proj_120_to_768, dtype=torch.float32, device=DEVICE)
        
        raw = torch.stack([b.mean(dim=0) for b in self.c_bases], dim=0)
        mu = torch.mean(raw, dim=0, keepdim=True)
        std = torch.std(raw - mu, dim=0, keepdim=True) + 1e-6
        self.norm_vocab = torch.nn.functional.normalize((raw - mu) / std, p=2, dim=-1)
        self.affordance_cache = {}

    def get_concept_affordances_svd(self, concept_idx: int, k_neighbors: int = 16, max_bases: int = 16) -> torch.Tensor:
        if concept_idx in self.affordance_cache: return self.affordance_cache[concept_idx]
        target = self.norm_vocab[concept_idx:concept_idx+1]
        sims = torch.mm(target, self.norm_vocab.T).squeeze(0)
        _, topk = torch.topk(sims, min(k_neighbors, self.norm_vocab.shape[0]))
        centered = self.norm_vocab[topk] - target
        _, _, V = torch.linalg.svd(centered, full_matrices=False)
        n_found = V.shape[0]
        if n_found < max_bases:
            needed = max_bases - n_found
            gen = torch.Generator(device=DEVICE).manual_seed(concept_idx + 101)
            pad = torch.randn((needed, target.shape[-1]), device=DEVICE, dtype=target.dtype, generator=gen)
            if n_found > 0: pad = pad - torch.mm(torch.mm(pad, V.T), V)
            pad = torch.nn.functional.normalize(pad, p=2, dim=-1)
            tangent = torch.cat([V, pad], dim=0)[:max_bases]
        else:
            tangent = V[:max_bases].clone()
        self.affordance_cache[concept_idx] = tangent
        return tangent

    def decode_slots(self, z_slots_768: torch.Tensor):
        z_norm = torch.nn.functional.normalize(z_slots_768, p=2, dim=-1)
        sims = torch.mm(z_norm, self.norm_vocab.T)
        top = torch.argmax(sims, dim=-1)
        names = [self.words[int(i)].upper() for i in top]
        return sims, torch.softmax(sims / 0.12, dim=-1), names, top.tolist()

class MusicDeltaCycleSealEngine:
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
                self.charge[s] = min(float(self.seal_cycles), self.charge[s] + 1.0 + float(np.clip(rip * 2.0, 0.0, 1.5)))
            else:
                loss = float(self.seal_cycles) / float(max(1, self.decay_cycles))
                self.charge[s] = max(0.0, self.charge[s] - loss)

            if self.mask[s] and self.charge[s] < (self.seal_cycles * 0.20):
                self.mask[s] = False

        ready = [s for s in range(n_sec) if not self.mask[s] and self.charge[s] >= self.seal_cycles]
        if len(ready) >= 2:
            for s in ready: self.mask[s] = True

# -----------------------------------------------------------------------------
# LAYER A: УНИВЕРСАЛЬНЫЙ СТОЛБИКОВЫЙ ПРОЦЕССОР ДЛЯ ЛЮБОГО ДАТЧИКА
# -----------------------------------------------------------------------------
class UniversalColumnProcessor:
    def __init__(self, role: str, semantic: UniversalMusicSemanticSpace, seal_thresh: float = 0.68):
        self.role = role
        self.semantic = semantic
        self.seals = MusicDeltaCycleSealEngine(MAX_SLOTS_CAPACITY, seal_thresh=seal_thresh)
        self.stabs = np.zeros(MAX_SLOTS_CAPACITY, dtype=np.float32)
        self.buf_len = 2
        self.delta_cycle_buffer = torch.zeros((self.buf_len, MAX_SLOTS_CAPACITY, 768), dtype=torch.float32, device=DEVICE)
        self.buf_ptr = 0

    def process(self, node, n_sec: int, sec_idx: int, dl_phase: float, is_tick: bool):
        if node is None: return None
        raw_iplv = torch.from_numpy(node.iplv_32).to(DEVICE, dtype=torch.float32)
        if torch.norm(raw_iplv) <= 1e-4: return None

        # 1. Сжатие в S слотов PAC
        dynamic_wm = torch.nn.functional.adaptive_avg_pool1d(raw_iplv.T.unsqueeze(0), n_sec).squeeze(0).T
        z_slots = torch.matmul(dynamic_wm, PHYSICAL_PROJECTION_768x120.T)
        _, soft_w, names, indices = self.semantic.decode_slots(z_slots)

        # 2. Фазовый Курамото R_sync
        kuramoto_r = float(torch.clamp(torch.norm(dynamic_wm.mean(dim=0)) / 10.0, 0.0, 1.0).item())

        # 3. 2D Манифолд рангов (Fan 2024)
        x_local = float(torch.clamp(torch.sum(dynamic_wm[sec_idx % n_sec] * DX_GPU_120) / 40.0, -1.0, 1.0).item())
        y_global = float(torch.clamp(torch.sum(dynamic_wm[sec_idx % n_sec] * DY_GPU_120) / 40.0, -1.0, 1.0).item())

        # 4. Темпоральный вектор r_y (Colgin 2009 / Bieri 2014)
        z_past, z_future = raw_iplv[0], raw_iplv[-1]
        npast, nfut = torch.norm(z_past), torch.norm(z_future)
        temporal_asymmetry = float(((nfut - npast) / (nfut + npast + 1e-6)).item())

        # 5. Каузальный граф 89.5 Гц рипплов Дики (ciPLV)
        raw_rip = torch.from_numpy(node.iplv_human_ripple).to(DEVICE, dtype=torch.float32)
        dyn_rip = torch.nn.functional.adaptive_avg_pool1d(raw_rip.T.unsqueeze(0), n_sec).squeeze(0).T
        diff_rip = dyn_rip.unsqueeze(1) - dyn_rip.unsqueeze(0)
        c_flow_768 = torch.matmul(diff_rip, PHYSICAL_PROJECTION_768x120.T)
        c_flow_np = torch.norm(c_flow_768, dim=-1).cpu().numpy()

        net_leads = (c_flow_768.sum(dim=1) - c_flow_768.sum(dim=0)).sum(dim=-1).cpu().numpy()
        order = np.argsort(net_leads)[::-1]
        root_s = int(order[0])

        # 6. Рекурсивное дерево Грама–Шмидта
        L_tok = self.semantic.c_bases[0].shape[0]
        tree_out = torch.zeros((L_tok, 768), dtype=torch.float32, device=DEVICE)
        top_w, top_i = torch.topk(soft_w[root_s], min(16, len(self.semantic.words)))
        top_w = top_w / (top_w.sum() + 1e-6)
        for i in range(top_w.shape[0]):
            tree_out.add_(self.semantic.c_bases[int(top_i[i])], alpha=top_w[i])

        n_tok = min(n_sec, 12)
        tree_out[1:n_tok+1].add_((z_slots[root_s:root_s+1] / 15.0).expand(n_tok, 768), alpha=0.25)
        accum = tree_out.clone()
        dag_parts = [names[root_s]]
        sec_w = (2.0 * math.pi) / float(n_sec)
        root_leads = c_flow_np[root_s, :n_sec]

        for s in order[1:]:
            child_w = names[s]
            t_child = self.semantic.c_bases[indices[s]].clone()
            t_child.add_((z_slots[s:s+1] / 15.0).expand(L_tok, 768), alpha=0.20)
            center = -math.pi * 0.5 + (s + 0.5) * sec_w
            focus = 0.5 * (1.0 + math.cos(dl_phase - center))
            is_seal = self.seals.mask[s]
            w_slot = (0.25 + 0.55 * self.stabs[s]) * (0.4 + 0.6 * focus) * (1.3 if is_seal else 1.0)

            if float(root_leads[s]) >= 0.03:
                dot = torch.sum(t_child * accum, dim=-1, keepdim=True) / (torch.sum(accum**2, dim=-1, keepdim=True) + 1e-7)
                ortho = t_child - dot * accum
                no, nc = torch.norm(ortho, dim=-1, keepdim=True), torch.norm(t_child, dim=-1, keepdim=True)
                ortho_n = (ortho / (no + 1e-6)) * nc * (no > (0.05 * nc)).float()
                tree_out[1:n_tok+1].add_(ortho_n[1:n_tok+1], alpha=w_slot * 0.30)
                accum.add_(ortho_n, alpha=0.15)
                dag_parts.append(f"⊃ {child_w}")
            else:
                half_d = 768 // 2
                nb = torch.norm(tree_out[1:n_tok+1, :half_d], dim=-1, keepdim=True) + 1e-7
                np_n = torch.norm(t_child[1:n_tok+1, half_d:], dim=-1, keepdim=True) + 1e-7
                tree_out[1:n_tok+1, half_d:].mul_(0.8).add_((t_child[1:n_tok+1, half_d:] / np_n * nb), alpha=0.20)
                dag_parts.append(f"∥ {child_w}")

        # 7. True Cosine Delta Seals
        if is_tick:
            self.delta_cycle_buffer[self.buf_ptr, :n_sec].copy_(z_slots)
            oldest_ptr = (self.buf_ptr + 1) % self.buf_len
            drift_cos = torch.sum(
                torch.nn.functional.normalize(z_slots, p=2, dim=-1) * 
                torch.nn.functional.normalize(self.delta_cycle_buffer[oldest_ptr, :n_sec], p=2, dim=-1),
                dim=-1
            ).cpu().numpy()
            self.buf_ptr = oldest_ptr

            for s in range(n_sec):
                raw_stab = float(np.clip((drift_cos[s] * 0.5 + 0.5) * (0.2 + 0.8 * kuramoto_r), 0.0, 1.0))
                self.stabs[s] = self.stabs[s] * 0.65 + raw_stab * 0.35

            self.seals.step(n_sec, self.stabs, c_flow_np)

        return {
            "names": names, "indices": indices, "c_flow_np": c_flow_np,
            "tree_out": tree_out, "dag_parts": dag_parts, "root_s": root_s,
            "stabs": self.stabs, "sealed_mask": list(self.seals.mask),
            "z_slots": z_slots, "kuramoto_r": kuramoto_r,
            "x_local": x_local, "y_global": y_global,
            "temporal_asymmetry": temporal_asymmetry,
            "kinematics": node.kinematics,
            "raw_iplv_32": raw_iplv
        }

# -----------------------------------------------------------------------------
# LAYER B: РЕГИОНАЛЬНЫЙ СИНТЕЗАТОР МУЗЫКИ
# -----------------------------------------------------------------------------
class MusicHeterarchyDeltaProcessor:
    def __init__(self, semantic: UniversalMusicSemanticSpace, router, seal_thresh: float = 0.68):
        self.semantic = semantic
        self.router = router
        self.engines = {r: UniversalColumnProcessor(r, semantic, seal_thresh=seal_thresh) for r in self.router.ALL_ROLES}
        self.active_c_idx = 0
        self.measured_ratio = 4.0
        self.delta_armed = False
        self.fatigue = torch.zeros(len(semantic.words), dtype=torch.float32, device=DEVICE)
        self.smoothed_turbo_drift = torch.zeros((32, 768), dtype=torch.float32, device=DEVICE)

    def step(self, frame, dt: float, mode: str = "semantic", target_block: float = 2.4, 
             target_context: float = 0.4, base_temp: float = 1.0, base_top_k: int = 250, cfg_coef: float = 1.0):
        topo = self.router.resolve(frame.nodes)
        th_hz, dl_hz = max(3.5, frame.theta_freq), max(0.5, frame.delta_freq)
        self.measured_ratio = self.measured_ratio * 0.95 + (th_hz / dl_hz) * 0.05
        k_th = int(np.clip(round(self.measured_ratio), 2, 8))
        n_sec = k_th * 2
        sec_w = (2.0 * math.pi) / float(n_sec)
        dl_phase = frame.delta_phase
        cur_sec = int(np.clip(int(((dl_phase + math.pi) % (2.0 * math.pi)) / sec_w), 0, n_sec - 1))

        is_tick = False
        if dl_phase < -1.5: self.delta_armed = True
        elif dl_phase > 0.5 and self.delta_armed: is_tick = True; self.delta_armed = False

        res = {r: self.engines[r].process(topo.get(r), n_sec, cur_sec, dl_phase, is_tick) for r in self.router.ALL_ROLES}
        hierarchy = ["Fpz", "AFz", "F3", "F4", "FCz"]
        apex = next((r for r in hierarchy if res[r] is not None), "НЕТ ДАТЧИКА")
        primary = res.get(apex)

        if primary is None:
            clean_emb = self.semantic.c_bases[0]
            return clean_emb, target_block, target_context, base_temp, base_top_k, cfg_coef, {"primary_role": "НЕТ ДАТЧИКА", "primary_active": False}

        # --- РЕЖИМ 1: TURBO (Максимальная скорость) ---
        if mode == "turbo":
            raw_iplv = primary["raw_iplv_32"]
            past_anchor = raw_iplv[0:1, :]
            vine_iplv = raw_iplv - past_anchor
            raw_latent = torch.matmul(vine_iplv, self.semantic.proj_120_to_768)
            raw_latent = raw_latent / (torch.norm(raw_latent, dim=-1, keepdim=True) + 1e-6)
            self.smoothed_turbo_drift.mul_(0.85).add_(raw_latent, alpha=0.15)

            base_emb = self.semantic.c_bases[self.active_c_idx].clone()
            L = base_emb.shape[0]
            drift_interp = F.interpolate(self.smoothed_turbo_drift.unsqueeze(0).permute(0, 2, 1), size=L, mode='linear', align_corners=True).permute(0, 2, 1)[0]
            target_out = base_emb + (drift_interp * 0.35)
            target_out = torch.nn.functional.normalize(target_out, p=2, dim=-1) * torch.norm(base_emb, p=2, dim=-1, keepdim=True)

            telemetry = {
                "primary_role": f"{apex} (TURBO)", "primary_active": True,
                "total_clock_sectors": n_sec, "current_clock_sec": cur_sec,
                "slot_names": primary["names"], "slot_indices": primary["indices"],
                "sealed_mask": primary["sealed_mask"], "slot_stabilities": primary["stabs"],
                "causal_flow_np": primary["c_flow_np"], "dag_log_parts": primary["dag_parts"],
                "dl_phase": dl_phase, "is_delta_tick": is_tick,
                "block_sec": target_block, "context_sec": target_context,
                "temp": base_temp, "top_k": base_top_k, "cfg_coef": cfg_coef, "is_drop": False,
                "active_vocab_size": len(self.semantic.words),
                "fpz_active": False, "fpz_tension": 0.0, "fpz_concept": "",
                "f3_active": res["F3"] is not None, "f3_concept": res["F3"]["dag_parts"][0] if res["F3"] else "",
                "f3_x_local": res["F3"]["x_local"] if res["F3"] else 0.0,
                "f3_y_global": res["F3"]["y_global"] if res["F3"] else 0.0,
                "f3_kuramoto": res["F3"]["kuramoto_r"] if res["F3"] else 0.0,
                "f4_active": res["F4"] is not None, "f4_concept": res["F4"]["dag_parts"][0] if res["F4"] else "",
                "f4_entropy": float(1.0 - res["F4"]["kuramoto_r"]) if res["F4"] else 0.0,
                "fcz_active": res["FCz"] is not None, "fcz_concept": res["FCz"]["dag_parts"][0] if res["FCz"] else "",
                "fcz_lx": float(res["FCz"]["kinematics"].lx) if res["FCz"] else 0.0,
                "fcz_ly": float(res["FCz"]["kinematics"].ly) if res["FCz"] else 0.0,
                "fcz_rx": float(res["FCz"]["kinematics"].rx) if res["FCz"] else 0.0,
                "fcz_ry": float(res["FCz"]["kinematics"].ry) if res["FCz"] else 0.0
            }
            return target_out, target_block, target_context, base_temp, base_top_k, cfg_coef, telemetry

        # --- РЕЖИМ 2: SEMANTIC (TBT 2.0 Полный смысл) ---
        target_out = primary["tree_out"].clone()
        root_w = primary["dag_parts"][0]
        root_idx = primary["indices"][primary["root_s"]]
        clean_emb = self.semantic.c_bases[root_idx]
        clean_norms = torch.norm(clean_emb, p=2, dim=-1, keepdim=True)

        # F3 (Синтаксис ритма и 2D манифолд SWM)
        if res["F3"] and apex != "F3":
            xl, yg = res["F3"]["x_local"], res["F3"]["y_global"]
            r_sync = res["F3"]["kuramoto_r"]
            freeze_val = float(np.clip((r_sync - 0.3) * 2.0, 0.0, 0.85))
            for s in range(min(n_sec, 8)):
                t_f3 = self.semantic.c_bases[res["F3"]["indices"][s]]
                rank_scale = float(np.clip(1.0 + (xl * (s % 3) + yg * (s // 3)) * 0.25, 0.2, 2.0))
                if s + 1 < target_out.shape[0]:
                    target_out[s + 1].add_(t_f3[s + 1], alpha=0.35 * rank_scale * (1.0 - freeze_val * 0.5))

        # F4 (Тембральная энтропия)
        if res["F4"]:
            entropy = float(1.0 - res["F4"]["kuramoto_r"])
            c_mod = torch.tanh(res["F4"]["z_slots"].mean(dim=0) / 8.0).unsqueeze(0)
            target_out.mul_(1.0 + 0.25 * c_mod * entropy)
            temp = float(np.clip(base_temp * (0.8 + entropy * 0.5), 0.3, 2.0))
            top_k = int(np.clip(base_top_k + int((entropy - 0.5) * 150), 20, 500))
        else:
            temp, top_k = base_temp, base_top_k

        # FCz (4D моторный поток темпа)
        block_sec = target_block
        if res["FCz"]:
            kin = res["FCz"]["kinematics"]
            block_sec = float(np.clip(target_block - kin.ry * 0.6, max(0.5, target_block * 0.6), target_block * 1.5))
            a120 = torch.from_numpy(topo["FCz"].iplv_human_ripple[-1]).to(DEVICE, dtype=torch.float32).view(1, 120)
            z_act = torch.matmul(a120, PHYSICAL_PROJECTION_768x120.T).squeeze(0)
            if torch.norm(z_act) > 0.01:
                t_bases = self.semantic.get_concept_affordances_svd(root_idx, max_bases=16)
                proj = torch.matmul(torch.matmul(z_act, t_bases.T), t_bases)
                target_out[1:13].add_(proj.unsqueeze(0).expand(12, 768), alpha=0.25)

        # Fpz (The Drop / Фазовый срыв)
        is_drop = False
        fpz_tension = 0.0
        if res["Fpz"]:
            self.fatigue.mul_(max(0.0, 1.0 - 0.08 * dt))
            self.fatigue[self.active_c_idx] += 0.25 * dt
            phase_drift = float(np.clip(abs(res["Fpz"]["temporal_asymmetry"]), 0.0, 1.0))
            fpz_tension = float(np.clip(phase_drift * 0.6 + (self.fatigue[self.active_c_idx].item() / 2.0) * 0.4, 0.0, 1.0))

            sims = torch.mm(target_out.mean(dim=0, keepdim=True), self.semantic.norm_vocab.T)
            eff_energy = -sims.squeeze(0) + self.fatigue * (1.5 + 4.0 * phase_drift)
            if phase_drift >= 0.45:
                eff_energy[self.active_c_idx] += 10.0
                is_drop = True
            self.active_c_idx = int(torch.argmin(eff_energy))
            if is_drop:
                hw = self.semantic.words[self.active_c_idx].upper()
                print(f"🔥 [THE DROP]: Музыкальный фазовый срыв ➔ Переход в [{hw}]")
                target_out.copy_(self.semantic.c_bases[self.active_c_idx])

        target_out = torch.nn.functional.normalize(target_out, p=2, dim=-1) * clean_norms

        telemetry = {
            "primary_role": apex, "primary_active": True,
            "total_clock_sectors": n_sec, "current_clock_sec": cur_sec,
            "slot_names": primary["names"], "slot_indices": primary["indices"],
            "sealed_mask": primary["sealed_mask"], "slot_stabilities": primary["stabs"],
            "causal_flow_np": primary["c_flow_np"], "dag_log_parts": primary["dag_parts"],
            "dl_phase": dl_phase, "is_delta_tick": is_tick,
            "block_sec": block_sec, "context_sec": target_context,
            "temp": temp, "top_k": top_k, "cfg_coef": cfg_coef, "is_drop": is_drop,
            "active_vocab_size": len(self.semantic.words),
            "fpz_active": res["Fpz"] is not None, "fpz_tension": fpz_tension,
            "fpz_concept": res["Fpz"]["dag_parts"][0] if res["Fpz"] else "",
            "f3_active": res["F3"] is not None, "f3_concept": res["F3"]["dag_parts"][0] if res["F3"] else "",
            "f3_x_local": res["F3"]["x_local"] if res["F3"] else 0.0,
            "f3_y_global": res["F3"]["y_global"] if res["F3"] else 0.0,
            "f3_kuramoto": res["F3"]["kuramoto_r"] if res["F3"] else 0.0,
            "f4_active": res["F4"] is not None, "f4_concept": res["F4"]["dag_parts"][0] if res["F4"] else "",
            "f4_entropy": float(1.0 - res["F4"]["kuramoto_r"]) if res["F4"] else 0.0,
            "fcz_active": res["FCz"] is not None, "fcz_concept": res["FCz"]["dag_parts"][0] if res["FCz"] else "",
            "fcz_lx": float(res["FCz"]["kinematics"].lx) if res["FCz"] else 0.0,
            "fcz_ly": float(res["FCz"]["kinematics"].ly) if res["FCz"] else 0.0,
            "fcz_rx": float(res["FCz"]["kinematics"].rx) if res["FCz"] else 0.0,
            "fcz_ry": float(res["FCz"]["kinematics"].ry) if res["FCz"] else 0.0
        }
        return target_out, block_sec, target_context, temp, top_k, cfg_coef, telemetry
