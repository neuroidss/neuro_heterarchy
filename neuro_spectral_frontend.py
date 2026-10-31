#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: DIRECT LSL-TO-VRAM SPECTRAL FRONTEND & CORTICAL DIPOLE GEOMETRY
===================================================================================
- Поддерживает аппаратные потоки FreeEEG_<ID> с динамическим маппингом ролей
  и пользователей через router.device_mappings (Control Plane).
- Поддерживает размещение нескольких плат на одной зоне 10-20 (например, 3 платы на FCz)
  с автоматическим фьюжном их ciPLV по весам пространственной когерентности.
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import time
import math
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylsl import StreamInlet, resolve_streams

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y, parse_stream_identity, CANONICAL_ROLES
from neuro_vram_bus import (
    REGION_ROLES,
    EXTENDED_ROLE_TO_FUNCTIONAL,
    NUM_SLOTS,
)

BUF_SIZE = 256
FFT_SLOW = 1024
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 16
TWO_PI = 2.0 * math.pi


@torch.no_grad()
def build_scalable_dipole_geometry_gpu(coords_x_np, coords_y_np, chunk_limit: int = 65536) -> dict:
    cx = torch.tensor(coords_x_np, dtype=torch.float32, device=DEVICE)
    cy = torch.tensor(coords_y_np, dtype=torch.float32, device=DEVICE)
    n_ch = cx.shape[0]

    idxs = torch.triu_indices(n_ch, n_ch, offset=1, device=DEVICE)
    i_idx, j_idx = idxs[0], idxs[1]
    n_pairs = i_idx.shape[0]

    dip_x = 0.5 * (cx[i_idx] + cx[j_idx]).unsqueeze(0)
    dip_y = 0.5 * (cy[i_idx] + cy[j_idx]).unsqueeze(0)
    dx = cx[j_idx] - cx[i_idx]
    dy = cy[j_idx] - cy[i_idx]
    dist_mm = torch.hypot(dx, dy).unsqueeze(0)

    col_x = torch.linspace(-12.0, 12.0, 32, device=DEVICE)
    col_y = torch.linspace(-12.0, 12.0, 24, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(col_y, col_x, indexing="ij")
    cols_x = grid_x.reshape(768, 1)
    cols_y = grid_y.reshape(768, 1)
    angle_col = torch.atan2(cols_y, cols_x)

    proj_768xP = torch.empty((768, n_pairs), dtype=torch.float32, device=DEVICE)
    for start in range(0, n_pairs, chunk_limit):
        end = min(start + chunk_limit, n_pairs)
        d_sq = (cols_x - dip_x[:, start:end]) ** 2 + (cols_y - dip_y[:, start:end]) ** 2
        ang_dip = torch.atan2(dy[start:end], dx[start:end]).unsqueeze(0)
        ang_mod = 0.5 * (1.0 + torch.cos(2.0 * (angle_col - ang_dip)))
        proj_768xP[:, start:end] = torch.exp(-d_sq / 28.0) * (0.4 + 0.6 * ang_mod)

    proj_768xP = F.normalize(proj_768xP, p=2, dim=1)

    norm_dx = dx / (torch.norm(dx) + 1e-6)
    norm_dy = dy / (torch.norm(dy) + 1e-6)
    proj_2d = torch.stack([norm_dx, norm_dy], dim=1)

    return {
        "n_channels": n_ch,
        "n_pairs": n_pairs,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "dip_x": dip_x,
        "dip_y": dip_y,
        "dx": dx,
        "dy": dy,
        "proj_2d": proj_2d,
        "proj_768xP": proj_768xP,
        "proj_Px768": proj_768xP.T.contiguous(),
        "micro_mask": (dist_mm < 9.5).float(),
        "meso_mask":  ((dist_mm >= 9.5) & (dist_mm < 16.5)).float(),
        "macro_mask": (dist_mm >= 16.5).float(),
        "left_mask":  (dip_x <= 0.0).float(),
        "right_mask": (dip_x > 0.0).float(),
    }


GEOM_GPU = build_scalable_dipole_geometry_gpu(COORDS_X, COORDS_Y)
build_cortical_dipole_geometry_gpu = build_scalable_dipole_geometry_gpu


class DirectGPUSpectralFrontend(nn.Module):
    def __init__(self, geom: dict, gamma_max: float = 100.0):
        super().__init__()
        self.geom = geom
        self.gamma_max = gamma_max
        self.inlets = [None] * NUM_MAX_DEVICES
        self.stream_uids = [""] * NUM_MAX_DEVICES
        self.stream_names = [""] * NUM_MAX_DEVICES
        self.stream_roles = ["FCz"] * NUM_MAX_DEVICES
        self.stream_users = ["User1"] * NUM_MAX_DEVICES
        self.last_pull_time = [0.0] * NUM_MAX_DEVICES
        self.device_fs = [250.0] * NUM_MAX_DEVICES
        self.connected_uids = set()
        self._lsl_lock = threading.Lock()

        self.register_buffer("raw_buf_gpu", torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE))
        self.register_buffer("base_fftfreq_256", torch.fft.fftfreq(BUF_SIZE, d=1.0, device=DEVICE).view(1, 1, BUF_SIZE))
        self.register_buffer("base_rfftfreq_1024", torch.fft.rfftfreq(FFT_SLOW, d=1.0, device=DEVICE))
        slot_ang = -math.pi + (TWO_PI / NUM_SLOTS) * (torch.arange(NUM_SLOTS, device=DEVICE) + 0.5)
        self.register_buffer("slot_angles", slot_ang.view(1, NUM_SLOTS, 1, 1))

        t_idx = torch.linspace(-1.0, 0.0, BUF_SIZE, device=DEVICE).view(1, 1, 1, BUF_SIZE)
        self.register_buffer("recency_fast", torch.exp(t_idx * 4.5))
        self.register_buffer("recency_med",  torch.exp(t_idx * 2.2))
        self.register_buffer("recency_slow", torch.exp(t_idx * 1.0))

        self.register_buffer("delta_centers", torch.linspace(0.5, 3.5, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("theta_centers", torch.linspace(3.8, 8.5, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("beta_centers",  torch.linspace(13.0, 30.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("gamma_centers", torch.linspace(30.0, gamma_max, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("rip_centers",   torch.linspace(70.0, 100.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))

        self._hotplug_thread = threading.Thread(target=self._background_lsl_loop, daemon=True)
        self._hotplug_thread.start()

    def _background_lsl_loop(self):
        while True:
            try:
                self._poll_lsl_hotplug()
            except Exception:
                pass
            time.sleep(0.8)

    def _poll_lsl_hotplug(self):
        now = time.time()
        streams = resolve_streams(wait_time=0.04)
        live_uids = {s.source_id() for s in streams}
        with self._lsl_lock:
            for i in range(NUM_MAX_DEVICES):
                if self.inlets[i] is not None:
                    # Отключаем слот ТОЛЬКО если стрим реально исчез из LSL-сети
                    if self.stream_uids[i] not in live_uids:
                        self.connected_uids.discard(self.stream_uids[i])
                        self.inlets[i] = None
                        self.stream_uids[i] = ""
                        self.stream_names[i] = ""

            for s in streams:
                uid = s.source_id()
                if uid in self.connected_uids or s.channel_count() != NUM_CHANNELS:
                    continue
                slot = next((i for i, inl in enumerate(self.inlets) if inl is None), None)
                if slot is not None:
                    inl = StreamInlet(s, max_buflen=1, max_chunklen=32, recover=True)
                    self.inlets[slot] = inl
                    self.connected_uids.add(uid)
                    self.stream_uids[slot] = uid
                    self.stream_names[slot] = s.name()
                    self.last_pull_time[slot] = now
                    nom_sps = inl.info().nominal_srate()
                    self.device_fs[slot] = 500.0 if nom_sps > 350.0 else 250.0
                    u, r = parse_stream_identity(s.name())
                    self.stream_users[slot] = u if u else f"User{slot + 1}"
                    self.stream_roles[slot] = r if r else ("FCz" if slot == 0 else CANONICAL_ROLES[slot % len(CANONICAL_ROLES)])
                    print(f"📡 [DIRECT GPU LSL] Подключен аппаратный стрим '{s.name()}' -> Слот {slot}")

    @torch.no_grad()
    def _extract_ciplv_band(self, fft_exp: torch.Tensor, freqs_4d: torch.Tensor, pos_mask: torch.Tensor,
                            centers: torch.Tensor, bw: float, w_phase: torch.Tensor) -> torch.Tensor:
        i_idx, j_idx = self.geom["i_idx"], self.geom["j_idx"]
        filt = torch.exp(-0.5 * ((freqs_4d - centers) / bw) ** 2) * 2.0 * pos_mask.unsqueeze(1)
        Z = torch.fft.ifft(fft_exp * filt, dim=-1)
        P = Z / (torch.abs(Z) + 1e-12)
        cg = P[:, :, i_idx, :] * torch.conj(P[:, :, j_idx, :])
        psi = torch.sum(cg * w_phase, dim=-1)
        re_psi = torch.clamp(torch.real(psi), -0.9999, 0.9999)
        im_psi = torch.imag(psi)
        ciplv = im_psi / torch.sqrt(1.0 - re_psi * re_psi + 1e-7)
        return torch.clamp(ciplv, -1.0, 1.0)

    @torch.no_grad()
    def compute_from_buffer_gpu(self, sub_buf: torch.Tensor, fs_vec: torch.Tensor,
                                slot_roles: list[str], slot_users: Optional[list[str]] = None) -> dict:
        D = sub_buf.shape[0]
        if slot_users is None:
            slot_users = ["User1"] * D
        freqs_dev = self.base_fftfreq_256 * fs_vec

        centered = sub_buf - torch.mean(sub_buf, dim=2, keepdim=True)
        notch = ((torch.abs(freqs_dev) < 48.5) | (torch.abs(freqs_dev) > 51.5)) & \
                ((torch.abs(freqs_dev) < 98.5) | (torch.abs(freqs_dev) > 101.5))
        fft_clean = torch.fft.fft(centered, dim=-1) * notch.float()

        fft_1024 = torch.fft.rfft(centered[0], n=FFT_SLOW, dim=-1)
        freqs_slow = self.base_rfftfreq_1024 * fs_vec[0, 0, 0]

        dl_mask = (freqs_slow >= 0.5) & (freqs_slow <= 3.2)
        dl_pwr = torch.sum(torch.abs(fft_1024[:, dl_mask]) ** 2, dim=0)
        dl_freq = torch.clamp(torch.sum(dl_pwr * freqs_slow[dl_mask]) / (torch.sum(dl_pwr) + 1e-6), 0.5, 3.2)

        th_mask = (freqs_slow >= 3.8) & (freqs_slow <= 8.5)
        th_pwr = torch.sum(torch.abs(fft_1024[:, th_mask]) ** 2, dim=0)
        th_freq = torch.clamp(torch.sum(th_pwr * freqs_slow[th_mask]) / (torch.sum(th_pwr) + 1e-6), 3.5, 9.0)

        pos_mask = (self.base_fftfreq_256 >= 0).float()

        f_theta = torch.exp(-0.5 * ((freqs_dev - 6.0) / 1.5) ** 2) * 2.0 * pos_mask
        Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
        P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
        phi_theta_all = torch.angle(torch.mean(P_theta, dim=1))

        f_delta = torch.exp(-0.5 * ((freqs_dev - 1.5) / 0.6) ** 2) * 2.0 * pos_mask
        Z_delta = torch.fft.ifft(fft_clean * f_delta, dim=-1)
        P_delta = Z_delta / (torch.abs(Z_delta) + 1e-12)
        phi_delta_all = torch.angle(torch.mean(P_delta, dim=1))

        dl_phase = phi_delta_all[0, -1]
        th_phase = phi_theta_all[0, -1]

        fft_exp = fft_clean.unsqueeze(1)
        freqs_4d = freqs_dev.unsqueeze(1)

        w_raw = torch.exp(3.2 * torch.cos(phi_theta_all.unsqueeze(1).unsqueeze(1) - self.slot_angles))
        w_fast = w_raw * self.recency_fast
        w_fast = w_fast / (torch.sum(w_fast, dim=-1, keepdim=True) + 1e-6)

        w_med = w_raw * self.recency_med
        w_med = w_med / (torch.sum(w_med, dim=-1, keepdim=True) + 1e-6)

        w_delta_raw = torch.exp(2.8 * torch.cos(phi_delta_all.unsqueeze(1).unsqueeze(1) - self.slot_angles)) * self.recency_slow
        w_delta = w_delta_raw / (torch.sum(w_delta_raw, dim=-1, keepdim=True) + 1e-6)

        wm_delta     = self._extract_ciplv_band(fft_exp, freqs_4d, pos_mask, self.delta_centers, 0.6, w_delta)
        wm_theta     = self._extract_ciplv_band(fft_exp, freqs_4d, pos_mask, self.theta_centers, 1.2, w_med)
        wm_beta      = self._extract_ciplv_band(fft_exp, freqs_4d, pos_mask, self.beta_centers,  3.0, w_med)
        gamma_anchor = self._extract_ciplv_band(fft_exp, freqs_4d, pos_mask, self.gamma_centers, 4.5, w_fast)
        h_rip        = self._extract_ciplv_band(fft_exp, freqs_4d, pos_mask, self.rip_centers,   4.0, w_fast)

        bands_Dx5x32x120 = torch.stack([wm_delta, wm_theta, wm_beta, gamma_anchor, h_rip], dim=1)

        traj_2d = torch.matmul(gamma_anchor, self.geom["proj_2d"]) * 8.0
        base_xy = traj_2d[:, 0, :]
        end_xy = traj_2d[:, -1, :] - base_xy
        d_len = torch.hypot(end_xy[:, 0], end_xy[:, 1]) + 1e-6
        lx = torch.where(d_len > 1.0, end_xy[:, 0] / d_len, end_xy[:, 0])
        ly = torch.where(d_len > 1.0, end_xy[:, 1] / d_len, end_xy[:, 1])

        rel_xy = traj_2d[:, 1:-1, :] - base_xy.unsqueeze(1)
        sag_sum = torch.sum(end_xy[:, 0:1] * rel_xy[:, :, 1] - end_xy[:, 1:2] * rel_xy[:, :, 0], dim=1)
        rx = torch.clamp(sag_sum / (d_len * 16.0), -1.0, 1.0)

        mid_xy = traj_2d[:, NUM_SLOTS // 2, :] - base_xy
        l_past = torch.hypot(mid_xy[:, 0], mid_xy[:, 1])
        l_fut = torch.hypot(end_xy[:, 0] - mid_xy[:, 0], end_xy[:, 1] - mid_xy[:, 1])
        ry = (l_fut - l_past) / (l_fut + l_past + 1e-6)
        kin_all = torch.stack([lx, -ly, rx, ry], dim=-1)

        grouped_ip, grouped_rp, grouped_kin, grouped_bands = {}, {}, {}, {}
        raw_roles_present = set()

        for idx_in_sub, raw_role in enumerate(slot_roles):
            raw_roles_present.add(raw_role)
            func_role = EXTENDED_ROLE_TO_FUNCTIONAL.get(raw_role, "AFz")
            if func_role in REGION_ROLES:
                grouped_ip.setdefault(func_role, []).append(gamma_anchor[idx_in_sub])
                grouped_rp.setdefault(func_role, []).append(h_rip[idx_in_sub])
                grouped_kin.setdefault(func_role, []).append(kin_all[idx_in_sub])
                grouped_bands.setdefault(func_role, []).append(bands_Dx5x32x120[idx_in_sub])

        iplv_by_role, rip_by_role, kin_by_role, bands_by_role = {}, {}, {}, {}
        for r_name in grouped_ip:
            if len(grouped_ip[r_name]) == 1:
                iplv_by_role[r_name]  = grouped_ip[r_name][0]
                rip_by_role[r_name]   = grouped_rp[r_name][0]
                kin_by_role[r_name]   = grouped_kin[r_name][0]
                bands_by_role[r_name] = grouped_bands[r_name][0]
            else:
                stack_ip = torch.stack(grouped_ip[r_name], dim=0)
                stack_rp = torch.stack(grouped_rp[r_name], dim=0)
                stack_kn = torch.stack(grouped_kin[r_name], dim=0)
                stack_bd = torch.stack(grouped_bands[r_name], dim=0)
                coherence_w = torch.norm(stack_ip.mean(dim=1), dim=-1) + 1e-4
                coherence_w = (coherence_w / coherence_w.sum()).view(-1, 1, 1)
                iplv_by_role[r_name]  = torch.sum(stack_ip * coherence_w, dim=0)
                rip_by_role[r_name]   = torch.sum(stack_rp * coherence_w, dim=0)
                kin_by_role[r_name]   = torch.sum(stack_kn * coherence_w.squeeze(-1), dim=0)
                bands_by_role[r_name] = torch.sum(stack_bd * coherence_w.unsqueeze(1), dim=0)

        unique_users = len(set(slot_users))

        return {
            "num_live": D,
            "num_unique_users": unique_users,
            "raw_roles_present": list(raw_roles_present),
            "th_freq": th_freq,
            "dl_freq": dl_freq,
            "th_phase": th_phase,
            "dl_phase": dl_phase,
            "master_iplv": gamma_anchor.mean(dim=0),
            "master_rip": h_rip.mean(dim=0),
            "master_5x32x120": bands_Dx5x32x120.mean(dim=0),
            "default_kin": kin_all[0],
            "iplv_by_role": iplv_by_role,
            "rip_by_role": rip_by_role,
            "kin_by_role": kin_by_role,
            "bands_by_role": bands_by_role,
        }

    @torch.no_grad()
    def pull_and_compute_gpu(self, router) -> dict:
        now = time.time()
        with self._lsl_lock:
            active_slots = [i for i, inl in enumerate(self.inlets) if inl is not None]

        for i in active_slots:
            try:
                inl = self.inlets[i]
                if inl is None:
                    continue
                chunk, _ = inl.pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                if chunk:
                    self.last_pull_time[i] = now
                    t_chunk = torch.tensor(chunk, dtype=torch.float32, device=DEVICE).T
                    n = t_chunk.shape[1]
                    if n >= BUF_SIZE:
                        self.raw_buf_gpu[i].copy_(t_chunk[:, -BUF_SIZE:])
                    else:
                        self.raw_buf_gpu[i] = torch.roll(self.raw_buf_gpu[i], shifts=-n, dims=1)
                        self.raw_buf_gpu[i, :, -n:].copy_(t_chunk)
            except Exception:
                with self._lsl_lock:
                    self.connected_uids.discard(self.stream_uids[i])
                    self.inlets[i] = None

        with self._lsl_lock:
            fresh_slots = [
                i for i, inl in enumerate(self.inlets)
                if inl is not None and (now - self.last_pull_time[i] <= 2.0)
            ]
            if not fresh_slots:
                fresh_slots = [i for i, inl in enumerate(self.inlets) if inl is not None]

        if fresh_slots:
            active_valid_slots = []
            roles, users = [], []
            dev_map = getattr(router, "device_mappings", {})

            for s_i in fresh_slots:
                sname = self.stream_names[s_i]
                suid = self.stream_uids[s_i]

                # 1. Приоритет №1: Таблица маршрутизации по аппаратному имени FreeEEG_<ID>
                mapped = dev_map.get(sname) or dev_map.get(suid)
                if mapped is not None:
                    r_val = mapped.get("role", "FCz")
                    u_val = mapped.get("user", self.stream_users[s_i])
                # 2. Приоритет №2: Оверрайд по слоту
                elif s_i in router.overrides:
                    r_val = router.overrides[s_i]
                    u_val = self.stream_users[s_i]
                # 3. Приоритет №3: Статическая карта CLI или дефолт
                else:
                    r_val = router.static_map.get(s_i) or self.stream_roles[s_i]
                    u_val = self.stream_users[s_i]

                if not r_val or str(r_val).upper() in ("NONE", "OFF", "UNASSIGNED"):
                    continue

                active_valid_slots.append(s_i)
                roles.append(r_val)
                users.append(u_val)

            if active_valid_slots:
                sub_buf = self.raw_buf_gpu[active_valid_slots]
                D = sub_buf.shape[0]
                fs_vec = torch.tensor([self.device_fs[i] for i in active_valid_slots], device=DEVICE).view(D, 1, 1)
                self._last_valid_roles = list(roles)
                self._last_valid_users = list(users)
                self._last_valid_slots = list(active_valid_slots)
                return self.compute_from_buffer_gpu(sub_buf, fs_vec, roles, users)

        fallback_slots = getattr(self, "_last_valid_slots", [0])
        sub_buf = self.raw_buf_gpu[fallback_slots]
        D = sub_buf.shape[0]
        fs_vec = torch.tensor([self.device_fs[i] for i in fallback_slots], device=DEVICE).view(D, 1, 1)
        override_r = router.overrides.get(0) or router.static_map.get(0)
        roles = [override_r] * D if override_r in EXTENDED_ROLE_TO_FUNCTIONAL else getattr(self, "_last_valid_roles", ["FCz"] * D)
        users = getattr(self, "_last_valid_users", ["User1"] * D)
        return self.compute_from_buffer_gpu(sub_buf, fs_vec, roles, users)
