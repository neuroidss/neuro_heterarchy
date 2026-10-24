#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: 100% VRAM-RESIDENT COGNITIVE ENGINE (NATIVE PYTORCH _share_cuda_ IPC)
===================================================================================
- 100% МАТЕМАТИКИ В VRAM:
  Сырые отсчеты LSL сразу попадают в raw_buf_gpu. БПФ, Гильберт, ciPLV, 89.5 Гц
  рипплы Дики, 5 колонок Монти, SVD и граф A ⊃ B считаются строго на GPU.
- ПРЯМОЙ ОБЩИЙ ТЕНЗОР В VRAM МЕЖДУ PYTHON 3.10 И PYTHON 3.9:
  Использует нативный C++ механизм PyTorch (UntypedStorage._share_cuda_ / _new_shared_cuda).
  Оба процесса смотрят в один и тот же физический адрес видеопамяти (0% CPU, 0% PCIe).
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylsl import StreamInlet, resolve_streams

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y, parse_stream_identity, CANONICAL_ROLES

MAX_SLOTS = 16
BUF_SIZE = 256
FFT_SLOW = 1024
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 16
NUM_SLOTS = 32
TWO_PI = 2.0 * math.pi

IPC_HANDLE_FILE = "/dev/shm/neuro_pytorch_cuda_ipc.json"

REGION_ROLES = ["AFz", "FCz", "F3", "F4", "Fpz"]
REGION_TAU_SEC = {
    "FCz": 0.04,
    "F3":  0.35,
    "F4":  0.35,
    "AFz": 1.20,
    "Fpz": 5.00,
}

TOTAL_FLOATS = 71776


class HardwareVRAMBus:
    """
    Нативный общий тензор в видеопамяти через C++ аллокатор PyTorch (_share_cuda_).
    Оба процесса (Python 3.10 и Python 3.9) читают и пишут в один и тот же адрес VRAM!
    """
    def __init__(self):
        self.gpu_flat = torch.zeros(TOTAL_FLOATS, dtype=torch.float32, device=DEVICE)
        self._bind_views()
        self.is_owner = False
        self.attached_pid = -1

    def _bind_views(self):
        self.meta_32                = self.gpu_flat[0:32]
        self.amodal_manifold_77x768 = self.gpu_flat[32:59168].view(77, 768)
        self.z_slots_16x768         = self.gpu_flat[59168:71456].view(MAX_SLOTS, 768)
        self.causal_flow_16x16      = self.gpu_flat[71456:71712].view(MAX_SLOTS, MAX_SLOTS)
        self.slot_order_16          = self.gpu_flat[71712:71728]
        self.slot_stabs_16          = self.gpu_flat[71728:71744]
        self.slot_seals_16          = self.gpu_flat[71744:71760]
        self.slot_norm_coord_16     = self.gpu_flat[71760:71776]

    def allocate_and_publish_vram_handle(self):
        if self.is_owner:
            return
        storage = self.gpu_flat.untyped_storage()
        tup = storage._share_cuda_()
        payload = {
            "pid": os.getpid(),
            "ts": time.time(),
            "device_idx": int(tup[0]),
            "handle": tup[1].hex(),
            "size_bytes": int(tup[2]),
            "offset_bytes": int(tup[3]),
            "ref_counter_handle": tup[4].hex(),
            "ref_counter_offset": int(tup[5]),
            "event_handle": tup[6].hex(),
            "event_sync_required": bool(tup[7]),
        }
        tmp_path = f"{IPC_HANDLE_FILE}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, IPC_HANDLE_FILE)
        self.is_owner = True
        print(f"⚡ [VRAM CUDA-IPC] Опубликован прямой тензор в VRAM (PID={os.getpid()}, Ptr={hex(self.gpu_flat.data_ptr())})")

    def try_attach_to_existing_vram(self) -> bool:
        if self.is_owner or not os.path.exists(IPC_HANDLE_FILE):
            return False
        try:
            with open(IPC_HANDLE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            owner_pid = int(d["pid"])
            if owner_pid == os.getpid() or not os.path.exists(f"/proc/{owner_pid}"):
                return False

            if self.attached_pid == owner_pid:
                return True

            storage_cls = getattr(torch, "UntypedStorage", getattr(torch, "_UntypedStorage", None))
            shared_storage = storage_cls._new_shared_cuda(
                int(d["device_idx"]),
                bytes.fromhex(d["handle"]),
                int(d["size_bytes"]),
                int(d["offset_bytes"]),
                bytes.fromhex(d["ref_counter_handle"]),
                int(d["ref_counter_offset"]),
                bytes.fromhex(d["event_handle"]),
                bool(d["event_sync_required"]),
            )
            self.gpu_flat = torch.empty(0, dtype=torch.float32, device=DEVICE).set_(
                shared_storage, 0, (TOTAL_FLOATS,)
            )
            self._bind_views()
            self.attached_pid = owner_pid
            print(f"🔗 [VRAM CUDA-IPC] Прямой маппинг в VRAM процесса PID={owner_pid} (Ptr={hex(self.gpu_flat.data_ptr())}, 0% CPU!)")
            return True
        except Exception:
            # Сбрасываем возможную ошибку CUDA в случае мертвого дескриптора от старого запуска
            torch.cuda.synchronize()
            self.attached_pid = -1
            return False


@torch.no_grad()
def build_scalable_dipole_geometry_gpu(coords_x_np: np.ndarray, coords_y_np: np.ndarray, chunk_limit: int = 65536) -> dict:
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


class DirectGPUSpectralFrontend(nn.Module):
    def __init__(self, geom: dict, gamma_max: float = 100.0):
        super().__init__()
        self.geom = geom
        self.gamma_max = gamma_max
        self.inlets = [None] * NUM_MAX_DEVICES
        self.stream_uids = [""] * NUM_MAX_DEVICES
        self.stream_roles = ["AFz"] * NUM_MAX_DEVICES
        self.device_fs = [250.0] * NUM_MAX_DEVICES
        self.connected_uids = set()
        self.last_resolve_t = 0.0

        self.register_buffer("raw_buf_gpu", torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE))
        self.register_buffer("base_fftfreq_256", torch.fft.fftfreq(BUF_SIZE, d=1.0, device=DEVICE).view(1, 1, BUF_SIZE))
        self.register_buffer("base_rfftfreq_1024", torch.fft.rfftfreq(FFT_SLOW, d=1.0, device=DEVICE))
        slot_ang = -math.pi + (TWO_PI / NUM_SLOTS) * (torch.arange(NUM_SLOTS, device=DEVICE) + 0.5)
        self.register_buffer("slot_angles", slot_ang.view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("gamma_centers", torch.linspace(30.0, gamma_max, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("rip_centers", torch.linspace(70.0, 100.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))

    def _poll_lsl_hotplug(self):
        now = time.time()
        if now - self.last_resolve_t < 1.0:
            return
        self.last_resolve_t = now
        try:
            streams = resolve_streams(wait_time=0.02)
            live_uids = {s.source_id() for s in streams}
            for i in range(NUM_MAX_DEVICES):
                if self.inlets[i] is not None and self.stream_uids[i] not in live_uids:
                    self.connected_uids.discard(self.stream_uids[i])
                    self.inlets[i] = None
                    self.stream_uids[i] = ""

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
                    nom_sps = inl.info().nominal_srate()
                    self.device_fs[slot] = 500.0 if nom_sps > 350.0 else 250.0
                    _, r = parse_stream_identity(s.name())
                    self.stream_roles[slot] = r if r else CANONICAL_ROLES[slot % len(CANONICAL_ROLES)]
                    print(f"📡 [DIRECT GPU LSL] Стрим '{s.name()}' -> GPU Слот {slot} (Роль: {self.stream_roles[slot]})")
        except Exception:
            pass

    @torch.no_grad()
    def pull_and_compute_gpu(self, router) -> Optional[dict]:
        self._poll_lsl_hotplug()
        active_slots = [i for i, inl in enumerate(self.inlets) if inl is not None]
        if not active_slots:
            return None

        for i in active_slots:
            try:
                chunk, _ = self.inlets[i].pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                if chunk:
                    t_chunk = torch.tensor(chunk, dtype=torch.float32, device=DEVICE).T
                    n = t_chunk.shape[1]
                    if n >= BUF_SIZE:
                        self.raw_buf_gpu[i].copy_(t_chunk[:, -BUF_SIZE:])
                    else:
                        self.raw_buf_gpu[i] = torch.roll(self.raw_buf_gpu[i], shifts=-n, dims=1)
                        self.raw_buf_gpu[i, :, -n:].copy_(t_chunk)
            except Exception:
                self.connected_uids.discard(self.stream_uids[i])
                self.inlets[i] = None

        active_slots = [i for i, inl in enumerate(self.inlets) if inl is not None]
        if not active_slots:
            return None

        sub_buf = self.raw_buf_gpu[active_slots]
        D = sub_buf.shape[0]
        fs_vec = torch.tensor([self.device_fs[i] for i in active_slots], device=DEVICE).view(D, 1, 1)
        freqs_dev = self.base_fftfreq_256 * fs_vec

        centered = sub_buf - torch.mean(sub_buf, dim=2, keepdim=True)
        notch = ((torch.abs(freqs_dev) < 48.5) | (torch.abs(freqs_dev) > 51.5)) & \
                ((torch.abs(freqs_dev) < 98.5) | (torch.abs(freqs_dev) > 101.5))
        fft_clean = torch.fft.fft(centered, dim=-1) * notch.float()

        fft_1024 = torch.fft.rfft(centered[0], n=FFT_SLOW, dim=-1)
        freqs_slow = self.base_rfftfreq_1024 * self.device_fs[active_slots[0]]

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

        fft_exp = fft_clean.unsqueeze(1)
        freqs_4d = freqs_dev.unsqueeze(1)
        w = torch.exp(3.2 * torch.cos(phi_theta_all.unsqueeze(1).unsqueeze(1) - self.slot_angles))
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

        i_idx, j_idx = self.geom["i_idx"], self.geom["j_idx"]

        g_filt = torch.exp(-0.5 * ((freqs_4d - self.gamma_centers) / 4.5) ** 2) * 2.0 * pos_mask.unsqueeze(1)
        Z_gamma = torch.fft.ifft(fft_exp * g_filt, dim=-1)
        P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)
        cg_gamma = P_gamma[:, :, i_idx, :] * torch.conj(P_gamma[:, :, j_idx, :])
        psi_gamma = torch.sum(cg_gamma * w, dim=-1)
        gamma_anchor = torch.imag(psi_gamma * torch.conj(psi_gamma[:, 0:1, :]))

        r_filt = torch.exp(-0.5 * ((freqs_4d - self.rip_centers) / 4.0) ** 2) * 2.0 * pos_mask.unsqueeze(1)
        Z_rip = torch.fft.ifft(fft_exp * r_filt, dim=-1)
        P_rip = Z_rip / (torch.abs(Z_rip) + 1e-12)
        cg_rip = P_rip[:, :, i_idx, :] * torch.conj(P_rip[:, :, j_idx, :])
        psi_rip = torch.sum(cg_rip * w, dim=-1)
        h_rip = torch.imag(psi_rip) / torch.sqrt(torch.clamp(1.0 - torch.real(psi_rip) ** 2, min=1e-5))

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

        iplv_by_role, rip_by_role, kin_by_role = {}, {}, {}
        for idx_in_sub, slot_i in enumerate(active_slots):
            role = (
                router.overrides.get(slot_i)
                or router.static_map.get(slot_i)
                or self.stream_roles[slot_i]
            )
            if role in REGION_ROLES:
                iplv_by_role[role] = gamma_anchor[idx_in_sub]
                rip_by_role[role]  = h_rip[idx_in_sub]
                kin_by_role[role]  = kin_all[idx_in_sub]

        return {
            "num_live": D,
            "th_freq": th_freq,
            "dl_freq": dl_freq,
            "dl_phase": dl_phase,
            "master_iplv": gamma_anchor.mean(dim=0),
            "master_rip": h_rip.mean(dim=0),
            "default_kin": kin_all[0],
            "iplv_by_role": iplv_by_role,
            "rip_by_role": rip_by_role,
            "kin_by_role": kin_by_role,
        }


class GPUSealEngine(nn.Module):
    def __init__(self, max_slots: int = MAX_SLOTS, seal_cycles: float = 4.0,
                 decay_cycles: float = 2.0, seal_thresh: float = 0.68):
        super().__init__()
        self.max_slots = max_slots
        self.seal_cycles = float(seal_cycles)
        self.decay_step = float(seal_cycles) / max(1.0, float(decay_cycles))
        self.seal_thresh = float(seal_thresh)

        self.register_buffer("charge", torch.zeros(max_slots, dtype=torch.float32, device=DEVICE))
        self.register_buffer("mask", torch.zeros(max_slots, dtype=torch.float32, device=DEVICE))
        self.register_buffer("stabs", torch.zeros(max_slots, dtype=torch.float32, device=DEVICE))

    @torch.no_grad()
    def reset_gpu(self):
        self.charge.zero_()
        self.mask.zero_()

    @torch.no_grad()
    def step_gpu(self, n_sec: int, c_flow_mat: torch.Tensor):
        sub_flow = torch.abs(c_flow_mat[:n_sec, :n_sec])
        rip = torch.sum(sub_flow, dim=1) + torch.sum(sub_flow, dim=0)
        charge_gain = 1.0 + torch.clamp(rip * 2.0, 0.0, 1.5)

        cur_stabs = self.stabs[:n_sec]
        cur_charge = self.charge[:n_sec]

        updated_charge = torch.where(
            cur_stabs >= self.seal_thresh,
            torch.clamp(cur_charge + charge_gain, max=self.seal_cycles),
            torch.clamp(cur_charge - self.decay_step, min=0.0)
        )
        self.charge[:n_sec].copy_(updated_charge)

        cur_mask = (self.mask[:n_sec] > 0.5) & (updated_charge >= (self.seal_cycles * 0.20))
        ready = (~cur_mask) & (updated_charge >= self.seal_cycles)
        activate_ready = torch.sum(ready.int()) >= 2
        final_mask = torch.where(activate_ready, cur_mask | ready, cur_mask)
        self.mask[:n_sec].copy_(final_mask.float())


class AmodalActionIntegratorGPU(nn.Module):
    def __init__(self, sensitivity: float = 0.06, intent_gain: float = 1.6):
        super().__init__()
        self.sensitivity = sensitivity
        self.intent_gain = intent_gain
        self.register_buffer("persistence", torch.zeros(1, dtype=torch.float32, device=DEVICE))
        self.register_buffer("last_chord_768", torch.zeros(768, dtype=torch.float32, device=DEVICE))
        self.register_buffer("state_16x768", torch.zeros((16, 768), dtype=torch.float32, device=DEVICE))

    @torch.no_grad()
    def step_gpu(self, raw_wave_16x768: torch.Tensor, temp_ry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = raw_wave_16x768 - raw_wave_16x768.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(F.normalize(centered, p=2, dim=-1), full_matrices=False)
        v_aff = Vh[:16]
        projected = torch.matmul(torch.matmul(raw_wave_16x768, v_aff.T), v_aff)

        chord = projected[-1] - projected[0]
        mag = torch.norm(chord)
        last_mag = torch.norm(self.last_chord_768)
        cos_th = torch.sum(chord * self.last_chord_768) / (mag * last_mag + 1e-6)

        grow = (mag > 0.02) & (last_mag > 1e-4) & (cos_th > 0.75)
        new_pers = torch.where(
            grow,
            torch.clamp(self.persistence + 0.05, max=1.0),
            self.persistence * 0.94
        )
        self.persistence.copy_(new_pers)
        self.last_chord_768.copy_(chord)

        smooth = 0.96 - (self.sensitivity * 0.1)
        gain = self.sensitivity * self.intent_gain
        ry_clamped = torch.clamp(temp_ry * 0.4, -0.4, 0.8)
        active_boost = (1.0 + self.persistence * 4.0) * (1.0 + ry_clamped)

        step_delta = projected * (gain * active_boost * (1.0 - smooth))
        self.state_16x768.mul_(smooth).add_(step_delta)

        s_norm = torch.norm(self.state_16x768, dim=-1, keepdim=True)
        self.state_16x768.copy_(
            torch.where(s_norm > 6.0, (self.state_16x768 / (s_norm + 1e-6)) * 6.0, self.state_16x768)
        )
        return self.state_16x768, self.persistence


class AmodalMontyColumnGPU(nn.Module):
    def __init__(self, role: str, geom: dict, num_columns: int = 4096, k_active: int = 80, num_slots: int = 32):
        super().__init__()
        self.role = role
        self.geom = geom
        self.num_columns = num_columns
        self.k_active = k_active
        self.num_slots = num_slots
        self.tau_sec = REGION_TAU_SEC.get(role, 0.5)

        role_seed = sum(ord(c) for c in role) * 137 + 777
        gen = torch.Generator(device=DEVICE).manual_seed(role_seed)
        rf_pos = F.normalize(torch.randn((768, num_columns), generator=gen, device=DEVICE), p=2, dim=0)
        rf_neg = torch.roll(rf_pos, shifts=num_columns // 2, dims=1)
        self.register_buffer("rf_pos", rf_pos)
        self.register_buffer("rf_neg", rf_neg)

        self.register_buffer("W_out", torch.zeros((num_slots * 768, num_columns), device=DEVICE))
        self.register_buffer("W_topdown_sdr", torch.eye(num_columns, device=DEVICE) * 0.4)
        self.register_buffer("l23_pooled_sdr", torch.zeros(num_columns, device=DEVICE))
        self.register_buffer("upward_sdr_anomaly", torch.zeros(num_columns, device=DEVICE))
        self.register_buffer("baseline_loss", torch.tensor(0.020, dtype=torch.float32, device=DEVICE))

        sdr_proj = torch.randn((num_columns, 768), generator=gen, device=DEVICE) / math.sqrt(k_active)
        self.register_buffer("sdr_to_768", sdr_proj)
        self.action_integrator = AmodalActionIntegratorGPU()
        self.seals = GPUSealEngine(MAX_SLOTS)
        self.register_buffer("delta_cycle_buf", torch.zeros((2, MAX_SLOTS, 768), device=DEVICE))
        self.register_buffer("virtual_iplv", torch.zeros((32, geom["n_pairs"]), device=DEVICE))
        self.register_buffer("virtual_rip", torch.zeros((32, geom["n_pairs"]), device=DEVICE))
        self.buf_ptr = 0

    @torch.no_grad()
    def synthesize_virtual_gpu(self, master_iplv: torch.Tensor, master_rip: torch.Tensor):
        if self.role == "Fpz":
            tgt = master_iplv * self.geom["macro_mask"] * 0.85
            alpha = 0.08
        elif self.role == "F3":
            tgt = torch.relu(master_iplv) * self.geom["left_mask"] * 1.4
            alpha = 0.25
        elif self.role == "F4":
            tgt = -torch.relu(-master_iplv) * self.geom["right_mask"] * 1.4
            alpha = 0.25
        elif self.role == "FCz":
            fast_diff = torch.roll(master_iplv, shifts=-1, dims=0) - master_iplv
            tgt = (fast_diff * 0.6 + master_iplv * 0.4) * self.geom["micro_mask"] * 1.5
            alpha = 0.50
        else:
            tgt = master_iplv
            alpha = 0.30

        self.virtual_iplv.mul_(1.0 - alpha).add_(tgt, alpha=alpha)
        self.virtual_rip.mul_(1.0 - alpha).add_(master_rip, alpha=alpha)
        return self.virtual_iplv, self.virtual_rip

    @torch.no_grad()
    def forward_gpu(self, raw_iplv_32xP: torch.Tensor, raw_rip_32xP: torch.Tensor,
                    n_sec: int, cur_sec: int, is_tick: bool, dt: float,
                    top_down_parent_sdr: Optional[torch.Tensor] = None,
                    bottom_up_child_anomaly: Optional[torch.Tensor] = None) -> dict:
        proj_T = self.geom["proj_Px768"]

        x_pos_768 = torch.matmul(torch.relu(raw_iplv_32xP), proj_T)
        x_neg_768 = torch.matmul(torch.relu(-raw_iplv_32xP), proj_T)
        z_32x768 = x_pos_768 - x_neg_768

        overlap = (torch.matmul(x_pos_768, self.rf_pos) + torch.matmul(x_neg_768, self.rf_neg)) / 12.0

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

        alpha_tau = min(0.85, max(0.01, dt / max(0.016, self.tau_sec)))
        self.l23_pooled_sdr.mul_(1.0 - alpha_tau).add_(sdr_inst, alpha=alpha_tau)

        if pred_from_parent is not None and top_down_parent_sdr is not None:
            sdr_err = sdr_inst - pred_from_parent
            self.upward_sdr_anomaly.copy_(torch.relu(sdr_err))
            p_energy = torch.sum(top_down_parent_sdr ** 2) + 1.0
            self.W_topdown_sdr.mul_(0.9998).add_(torch.outer(sdr_err, top_down_parent_sdr) * (0.03 / p_energy))
        else:
            self.upward_sdr_anomaly.copy_(sdr_inst * 0.15)

        sdr_anom_norm = torch.norm(self.upward_sdr_anomaly) / math.sqrt(self.k_active)

        real_flat = z_32x768.reshape(-1)
        pred_flat = torch.mv(self.W_out, self.l23_pooled_sdr)
        error_flat = real_flat - pred_flat
        loss_t = torch.mean(error_flat ** 2)

        surprise_t = torch.clamp(
            torch.relu((loss_t - self.baseline_loss) / (self.baseline_loss + 1e-5)) + sdr_anom_norm * 0.3,
            0.0, 2.5
        )
        self.baseline_loss.mul_(0.985).add_(loss_t * 0.015)

        state_energy = torch.sum(self.l23_pooled_sdr ** 2) + 1e-4
        eta = torch.clamp(0.08 * (1.0 + surprise_t * 0.5), max=0.25) / state_energy
        self.W_out.mul_(0.9995).add_(torch.outer(error_flat, self.l23_pooled_sdr) * eta)

        z_on_16x768  = F.adaptive_avg_pool1d(x_pos_768.T.unsqueeze(0), 16).squeeze(0).T
        z_off_16x768 = F.adaptive_avg_pool1d(x_neg_768.T.unsqueeze(0), 16).squeeze(0).T

        chord_768 = z_32x768[-1] - z_32x768[0]
        chord_unit = F.normalize(chord_768.unsqueeze(0), p=2, dim=-1)
        rel_z = z_32x768 - z_32x768[0:1]
        sagitta_32x768 = rel_z - torch.sum(rel_z * chord_unit, dim=-1, keepdim=True) * chord_unit
        sagitta_16x768 = F.adaptive_avg_pool1d(sagitta_32x768.T.unsqueeze(0), 16).squeeze(0).T
        temp_asym_16x768 = z_32x768[16:32] - z_32x768[0:16]

        sdr_7x4096 = F.adaptive_avg_pool1d(sdr_seq.T.unsqueeze(0), 7).squeeze(0).T
        sdr_tokens_7x768 = torch.matmul(sdr_7x4096, self.sdr_to_768)

        dynamic_wm = F.adaptive_avg_pool1d(raw_iplv_32xP.T.unsqueeze(0), n_sec).squeeze(0).T
        z_slots = torch.matmul(dynamic_wm, proj_T)

        kuramoto_r = torch.clamp(torch.norm(dynamic_wm.mean(dim=0)) / 10.0, 0.0, 1.0)
        slot_cur = dynamic_wm[cur_sec % n_sec]
        x_local = torch.clamp(torch.sum(slot_cur * self.geom["dx"]) / 35.0, -1.0, 1.0)
        y_global = torch.clamp(torch.sum(slot_cur * self.geom["dy"]) / 35.0, -1.0, 1.0)

        slot_x = torch.matmul(dynamic_wm, self.geom["dx"])
        slot_y = torch.matmul(dynamic_wm, self.geom["dy"])
        slot_norm_coord = (torch.atan2(slot_y, slot_x) + math.pi) / (2.0 * math.pi)

        npast = torch.norm(raw_iplv_32xP[0])
        nfut = torch.norm(raw_iplv_32xP[-1])
        temp_asym_scalar = (nfut - npast) / (nfut + npast + 1e-6)

        dyn_rip_768 = torch.matmul(
            F.adaptive_avg_pool1d(raw_rip_32xP.T.unsqueeze(0), n_sec).squeeze(0).T,
            proj_T
        )
        diff_rip = dyn_rip_768.unsqueeze(1) - dyn_rip_768.unsqueeze(0)
        c_flow_mat = torch.norm(diff_rip, dim=-1)

        net_leads = torch.sum(diff_rip.sum(dim=1) - diff_rip.sum(dim=0), dim=-1)
        order_gpu = torch.argsort(net_leads, descending=True)

        ordered_slots = z_slots[order_gpu]
        root_vec = ordered_slots[0:1]
        root_leads = c_flow_mat[order_gpu[0], order_gpu]

        dot = torch.sum(ordered_slots * root_vec, dim=-1, keepdim=True) / (torch.sum(root_vec ** 2) + 1e-7)
        ortho = ordered_slots - dot * root_vec
        ortho_norm = F.normalize(ortho, p=2, dim=-1) * torch.norm(ordered_slots, dim=-1, keepdim=True)

        is_supset = (root_leads >= 0.03).unsqueeze(-1).float()
        composed_slots = root_vec + is_supset * ortho_norm * 0.35 + (1.0 - is_supset) * ordered_slots * 0.25
        tree_16x768 = F.adaptive_avg_pool1d(composed_slots.T.unsqueeze(0), 16).squeeze(0).T

        if is_tick:
            self.delta_cycle_buf[self.buf_ptr, :n_sec].copy_(z_slots)
            oldest = (self.buf_ptr + 1) % 2
            drift_cos = torch.sum(
                F.normalize(z_slots, p=2, dim=-1) * F.normalize(self.delta_cycle_buf[oldest, :n_sec], p=2, dim=-1),
                dim=-1
            )
            self.buf_ptr = oldest
            raw_stab = torch.clamp((drift_cos * 0.5 + 0.5) * (0.2 + 0.8 * kuramoto_r), 0.0, 1.0)
            self.seals.stabs[:n_sec].mul_(0.65).add_(raw_stab * 0.35)
            self.seals.step_gpu(n_sec, c_flow_mat)

        return {
            "z_slots": z_slots,
            "slot_norm_coord": slot_norm_coord,
            "order_gpu": order_gpu,
            "c_flow_mat": c_flow_mat,
            "tree_16x768": tree_16x768,
            "z_on_16x768": z_on_16x768,
            "z_off_16x768": z_off_16x768,
            "sagitta_16x768": sagitta_16x768,
            "temp_asym_16x768": temp_asym_16x768,
            "sdr_tokens_7x768": sdr_tokens_7x768,
            "kuramoto_r": kuramoto_r,
            "x_local": x_local,
            "y_global": y_global,
            "temp_asym_scalar": temp_asym_scalar,
            "loss": loss_t,
            "surprise": surprise_t,
        }


class SharedAmodalCognitiveEngine:
    def __init__(self, router, gamma_max: float = 100.0, strength_low: float = 0.50, strength_high: float = 0.85):
        self.router = router
        self.gamma_max = gamma_max
        self.strength_low = strength_low
        self.strength_high = strength_high
        self.bus = HardwareVRAMBus()
        self.high_pri_stream = torch.cuda.Stream(device=DEVICE, priority=-1) if DEVICE.type == "cuda" else None

        self.is_host = False
        self.spectral_gpu: Optional[DirectGPUSpectralFrontend] = None
        self.columns: Optional[nn.ModuleDict] = None
        self.measured_ratio = 4.0
        self.delta_armed = False

    def _ensure_host_initialized(self):
        if self.spectral_gpu is None:
            print("🧠 [100% GPU AMODAL ENGINE] Старт прямого LSL->VRAM конвейера и 5 регионов Монти...")
            self.bus.allocate_and_publish_vram_handle()
            self.spectral_gpu = DirectGPUSpectralFrontend(GEOM_GPU, gamma_max=self.gamma_max).to(DEVICE)
            self.columns = nn.ModuleDict({
                r: AmodalMontyColumnGPU(r, GEOM_GPU).to(DEVICE) for r in REGION_ROLES
            })
        self.is_host = True

    @torch.no_grad()
    def reset_seals(self):
        if self.columns is not None:
            for col in self.columns.values():
                col.seals.reset_gpu()

    @torch.no_grad()
    def step(self, dt: float) -> tuple[HardwareVRAMBus, bool]:
        if not self.is_host and self.bus.try_attach_to_existing_vram():
            return self.bus, bool((self.bus.meta_32[17] > 0.5).item())

        self._ensure_host_initialized()
        ctx = torch.cuda.stream(self.high_pri_stream) if self.high_pri_stream is not None else torch.no_grad()
        with ctx:
            eeg_out = self.spectral_gpu.pull_and_compute_gpu(self.router)
            if eeg_out is None:
                self.bus.meta_32[17] = 0.0
                return self.bus, False

            th_hz = float(eeg_out["th_freq"].item())
            dl_hz = float(eeg_out["dl_freq"].item())
            dl_phase = float(eeg_out["dl_phase"].item())

            self.measured_ratio = self.measured_ratio * 0.95 + (th_hz / dl_hz) * 0.05
            k_th = int(min(8, max(2, round(self.measured_ratio))))
            n_sec = k_th * 2
            sec_w = (2.0 * math.pi) / float(n_sec)
            cur_sec = int(min(n_sec - 1, max(0, int(((dl_phase + math.pi) % (2.0 * math.pi)) / sec_w))))

            is_tick = False
            if dl_phase < -1.5:
                self.delta_armed = True
            elif dl_phase > 0.5 and self.delta_armed:
                is_tick = True
                self.delta_armed = False

            real_iplv = eeg_out["iplv_by_role"]
            real_rip  = eeg_out["rip_by_role"]
            real_kin  = eeg_out["kin_by_role"]
            master_iplv = eeg_out["master_iplv"]
            master_rip  = eeg_out["master_rip"]

            sdr_fcz = self.columns["FCz"].l23_pooled_sdr
            sdr_f3  = self.columns["F3"].l23_pooled_sdr
            sdr_f4  = self.columns["F4"].l23_pooled_sdr
            sdr_afz = self.columns["AFz"].l23_pooled_sdr
            sdr_fpz = self.columns["Fpz"].l23_pooled_sdr

            anom_fcz = self.columns["FCz"].upward_sdr_anomaly
            anom_f34 = 0.5 * (self.columns["F3"].upward_sdr_anomaly + self.columns["F4"].upward_sdr_anomaly)
            anom_afz = self.columns["AFz"].upward_sdr_anomaly

            parent_map = {"FCz": 0.5 * (sdr_f3 + sdr_f4), "F3": sdr_afz, "F4": sdr_afz, "AFz": sdr_fpz, "Fpz": None}
            child_map  = {"FCz": None, "F3": anom_fcz, "F4": anom_fcz, "AFz": anom_f34, "Fpz": anom_afz}

            res = {}
            for r in REGION_ROLES:
                if r in real_iplv:
                    ip_r, rp_r = real_iplv[r], real_rip[r]
                else:
                    ip_r, rp_r = self.columns[r].synthesize_virtual_gpu(master_iplv, master_rip)
                res[r] = self.columns[r].forward_gpu(
                    ip_r, rp_r, n_sec=n_sec, cur_sec=cur_sec, is_tick=is_tick, dt=dt,
                    top_down_parent_sdr=parent_map[r], bottom_up_child_anomaly=child_map[r]
                )

            apex_idx = next((i for i, r in enumerate(["AFz", "FCz", "F3", "F4", "Fpz"]) if r in real_iplv), 0)
            apex_role = ["AFz", "FCz", "F3", "F4", "Fpz"][apex_idx]
            primary = res[apex_role]
            fcz_kin = real_kin.get("FCz", eeg_out["default_kin"])

            raw_fcz_wave = res["FCz"]["sagitta_16x768"] + res["FCz"]["z_on_16x768"] * 0.5
            fcz_act_16x768, fcz_pers = self.columns["FCz"].action_integrator.step_gpu(raw_fcz_wave, fcz_kin[3])

            max_surp = torch.stack([res[r]["surprise"] for r in REGION_ROLES]).max()
            fpz_tension = torch.clamp(max_surp * 0.55 + torch.abs(res["Fpz"]["temp_asym_scalar"]) * 0.45, 0.0, 1.0)
            active_drive = torch.clamp(fcz_pers.squeeze(0) * 0.70 + torch.clamp(max_surp, max=1.0) * 0.30, 0.0, 1.0)
            dyn_str = torch.clamp(
                self.strength_low + (self.strength_high - self.strength_low) * active_drive,
                self.strength_low, 0.95
            )

            m = self.bus.amodal_manifold_77x768
            m.zero_()
            m[1:5].copy_(res["Fpz"]["tree_16x768"][:4])
            m[5:21].copy_(res["AFz"]["tree_16x768"])
            m[21:37].copy_(res["F3"]["z_on_16x768"])
            m[37:53].copy_(res["F4"]["z_off_16x768"] * 0.6 + res["F4"]["temp_asym_16x768"] * 0.4)
            m[53:69].copy_(fcz_act_16x768)
            m[69:76].copy_(primary["sdr_tokens_7x768"])

            self.bus.z_slots_16x768.zero_()
            self.bus.z_slots_16x768[:n_sec].copy_(primary["z_slots"])
            self.bus.causal_flow_16x16.zero_()
            self.bus.causal_flow_16x16[:n_sec, :n_sec].copy_(primary["c_flow_mat"])
            self.bus.slot_order_16[:n_sec].copy_(primary["order_gpu"].float())
            self.bus.slot_stabs_16[:n_sec].copy_(self.columns[apex_role].seals.stabs[:n_sec])
            self.bus.slot_seals_16[:n_sec].copy_(self.columns[apex_role].seals.mask[:n_sec])
            self.bus.slot_norm_coord_16[:n_sec].copy_(primary["slot_norm_coord"])

            meta = self.bus.meta_32
            meta[1]  = float(n_sec)
            meta[2]  = float(cur_sec)
            meta[3]  = eeg_out["dl_phase"]
            meta[4]  = 1.0 if is_tick else 0.0
            meta[5]  = dyn_str
            meta[6]  = fcz_pers.squeeze(0)
            meta[7]  = max_surp
            meta[8]  = primary["loss"]
            meta[9]  = res["F3"]["x_local"]
            meta[10] = res["F3"]["y_global"]
            meta[11] = res["F3"]["kuramoto_r"]
            meta[12:16].copy_(fcz_kin)
            meta[16] = fpz_tension
            meta[17] = float(eeg_out["num_live"])
            meta[18] = float(apex_idx)

            return self.bus, True

    def stop(self):
        if self.bus.is_owner:
            try:
                os.remove(IPC_HANDLE_FILE)
            except OSError:
                pass


@torch.no_grad()
def project_amodal_bus_to_diffusion_gpu(bus: HardwareVRAMBus, semantic,
                                        strength_low: float, burst_strength: float,
                                        need_ui_telemetry: bool = False) -> tuple[torch.Tensor, float, dict]:
    meta = bus.meta_32
    n_sec = max(2, int(meta[1].item()))
    fcz_pers_t = meta[6]
    max_surp_t = meta[7]

    active_drive_t = torch.clamp(fcz_pers_t * 0.70 + torch.clamp(max_surp_t, max=1.0) * 0.30, 0.0, 1.0)
    dyn_str_t = torch.clamp(strength_low + (burst_strength - strength_low) * active_drive_t, strength_low, 0.95)
    dyn_str = float(dyn_str_t.item())

    order = bus.slot_order_16[:n_sec].long()
    norm_coords = bus.slot_norm_coord_16[:n_sec]
    c_flow = bus.causal_flow_16x16[:n_sec, :n_sec]
    amodal_77 = bus.amodal_manifold_77x768

    with semantic.lock:
        words_copy = list(semantic.words)
        w_count = len(words_copy)

        if w_count == 0:
            target_77 = semantic.empty_embed.clone()
            clean_norms = torch.norm(target_77, p=2, dim=-1, keepdim=True).clamp(min=1e-5)
            v_aff = semantic.get_concept_affordances_svd(0, max_bases=16)
            mod_77 = torch.matmul(torch.matmul(amodal_77, v_aff.T), v_aff)
            target_77.add_(mod_77, alpha=0.35 * dyn_str)
            target_77 = F.normalize(target_77, p=2, dim=-1) * clean_norms
            slot_idx_gpu = torch.zeros(n_sec, dtype=torch.long, device=DEVICE)
        else:
            slot_idx_gpu = torch.clamp((norm_coords * w_count).long(), 0, w_count - 1)
            root_s = order[0] % n_sec
            root_c_idx = int(slot_idx_gpu[root_s].item()) % w_count
            base_77 = semantic.get_77_embedding_gpu(words_copy[root_c_idx]).clone()
            clean_norms = torch.norm(base_77, p=2, dim=-1, keepdim=True).clamp(min=1e-5)

            if w_count > 1 and n_sec > 1:
                child_s = order[1] % n_sec
                child_c_idx = int(slot_idx_gpu[child_s].item()) % w_count
                if child_c_idx != root_c_idx:
                    child_emb = semantic.get_77_embedding_gpu(words_copy[child_c_idx])
                    dot = torch.sum(child_emb * base_77, dim=-1, keepdim=True) / (torch.sum(base_77 ** 2, dim=-1, keepdim=True) + 1e-7)
                    ortho = F.normalize(child_emb - dot * base_77, p=2, dim=-1) * clean_norms
                    is_sup = (c_flow[root_s, child_s] >= 0.03).float()
                    base_77.add_(ortho * is_sup * 0.32 + child_emb * (1.0 - is_sup) * 0.20)

            v_aff = semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)
            mod_77 = torch.matmul(torch.matmul(amodal_77, v_aff.T), v_aff)
            target_77 = base_77 + mod_77 * (0.35 * dyn_str)
            target_77 = F.normalize(target_77, p=2, dim=-1) * clean_norms

    telemetry = {}
    if need_ui_telemetry:
        indices = [int(i) % max(1, w_count) for i in slot_idx_gpu.tolist()]
        names = [words_copy[i].upper() for i in indices] if w_count > 0 else ["СВОБОДНЫЙ_ДРЕЙФ"] * n_sec
        root_name = names[int(order[0].item()) % n_sec]
        telemetry = {
            "primary_role": "AFz", "primary_active": True,
            "ladder_stage": 4,
            "monty_loss": round(float(meta[8].item()), 5),
            "persistence": round(float(fcz_pers_t.item()), 3),
            "surprise": round(float(max_surp_t.item()), 3),
            "regions_telemetry": {},
            "total_clock_sectors": n_sec, "current_clock_sec": int(meta[2].item()) % n_sec,
            "slot_names": names, "slot_indices": indices,
            "sealed_mask": [bool(x > 0.5) for x in bus.slot_seals_16[:n_sec].tolist()],
            "slot_stabilities": bus.slot_stabs_16[:n_sec].cpu().numpy(),
            "causal_flow_np": c_flow.cpu().numpy(),
            "dag_log_parts": [root_name],
            "dl_phase": float(meta[3].item()), "is_delta_tick": bool(meta[4].item() > 0.5),
            "dyn_strength": dyn_str, "active_vocab_size": w_count,
            "fpz_active": True, "fpz_tension": float(meta[16].item()), "fpz_concept": root_name,
            "f3_active": True, "f3_concept": root_name,
            "f3_x_local": float(meta[9].item()), "f3_y_global": float(meta[10].item()),
            "f3_kuramoto": float(meta[11].item()), "f3_slot_names": names,
            "f4_active": True, "f4_concept": root_name, "f4_entropy": float(active_drive_t.item()),
            "fcz_active": True, "fcz_concept": root_name,
            "fcz_lx": float(meta[12].item()), "fcz_ly": float(meta[13].item()),
            "fcz_rx": float(meta[14].item()), "fcz_ry": float(meta[15].item()),
        }
    return target_77, dyn_str, telemetry


@torch.no_grad()
def project_amodal_bus_to_musicgen_gpu(bus: HardwareVRAMBus, semantic,
                                       target_block: float, target_context: float,
                                       base_temp: float, base_top_k: int, cfg_coef: float,
                                       strength_low: float, strength_high: float,
                                       need_ui_telemetry: bool = False):
    meta = bus.meta_32
    n_sec = max(2, int(meta[1].item()))
    fcz_pers_t = meta[6]
    max_surp_t = meta[7]

    active_drive_t = torch.clamp(fcz_pers_t * 0.70 + torch.clamp(max_surp_t, max=1.0) * 0.30, 0.0, 1.0)
    dyn_str_t = torch.clamp(strength_low + (strength_high - strength_low) * active_drive_t, strength_low, 0.98)
    dyn_str = float(dyn_str_t.item())

    order = bus.slot_order_16[:n_sec].long()
    norm_coords = bus.slot_norm_coord_16[:n_sec]
    c_flow = bus.causal_flow_16x16[:n_sec, :n_sec]
    amodal_64 = bus.amodal_manifold_77x768[1:65]

    with semantic.lock:
        c_bases_copy = list(semantic.c_bases)
        words_copy = list(semantic.words)
        n_bases = min(len(c_bases_copy), len(words_copy))

        if n_bases == 0:
            base_64 = semantic.last_active_embed.clone()
            clean_norms = torch.norm(base_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)
            v_aff = semantic.global_svd_bases[:16]
            slot_idx_gpu = torch.zeros(n_sec, dtype=torch.long, device=DEVICE)
        else:
            slot_idx_gpu = torch.clamp((norm_coords * n_bases).long(), 0, n_bases - 1)
            root_s = order[0] % n_sec
            root_c_idx = int(slot_idx_gpu[root_s].item()) % n_bases
            base_64 = c_bases_copy[root_c_idx].clone()
            clean_norms = torch.norm(base_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)

            if n_bases > 1 and n_sec > 1:
                child_s = order[1] % n_sec
                child_c_idx = int(slot_idx_gpu[child_s].item()) % n_bases
                if child_c_idx != root_c_idx:
                    t_child = c_bases_copy[child_c_idx]
                    dot = torch.sum(t_child * base_64, dim=-1, keepdim=True) / (torch.sum(base_64 ** 2, dim=-1, keepdim=True) + 1e-7)
                    ortho = F.normalize(t_child - dot * base_64, p=2, dim=-1) * clean_norms
                    is_sup = (c_flow[root_s, child_s] >= 0.03).float()
                    base_64[4:20].add_(ortho[4:20] * is_sup * 0.32 + t_child[4:20] * (1.0 - is_sup) * 0.22)

            v_aff = semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)

        coords = torch.matmul(amodal_64, v_aff.T)
        musical_mod = F.normalize(torch.matmul(coords, v_aff), p=2, dim=-1) * clean_norms

        final_64 = base_64 * (1.0 - 0.35 * dyn_str_t) + musical_mod * (0.35 * dyn_str_t)
        final_64 = F.normalize(final_64, p=2, dim=-1) * clean_norms

    temp = float(min(1.8, max(0.3, base_temp * (0.88 + 0.25 * dyn_str))))
    top_k = int(min(450, max(30, base_top_k + int((dyn_str - 0.4) * 100))))

    telemetry = {"dyn_strength": round(dyn_str, 3), "is_drop": False}
    if need_ui_telemetry:
        indices = [int(i) % max(1, n_bases) for i in slot_idx_gpu.tolist()]
        names = [words_copy[i].upper() for i in indices] if n_bases > 0 else ["СВОБОДНЫЙ_ДРЕЙФ"] * n_sec
        root_name = names[int(order[0].item()) % n_sec]
        telemetry.update({
            "primary_role": "AFz", "primary_active": True,
            "total_clock_sectors": n_sec, "current_clock_sec": int(meta[2].item()) % n_sec,
            "slot_names": names, "slot_indices": indices,
            "sealed_mask": [bool(x > 0.5) for x in bus.slot_seals_16[:n_sec].tolist()],
            "slot_stabilities": bus.slot_stabs_16[:n_sec].cpu().numpy(),
            "causal_flow_np": c_flow.cpu().numpy(),
            "dag_log_parts": [root_name],
            "dl_phase": float(meta[3].item()), "is_delta_tick": bool(meta[4].item() > 0.5),
            "active_vocab_size": n_bases,
            "fpz_active": True, "fpz_tension": float(meta[16].item()), "fpz_concept": root_name,
            "f3_active": True, "f3_concept": root_name,
            "f3_x_local": float(meta[9].item()), "f3_y_global": float(meta[10].item()), "f3_kuramoto": float(meta[11].item()),
            "f4_active": True, "f4_concept": root_name, "f4_entropy": float(active_drive_t.item()),
            "fcz_active": True, "fcz_concept": root_name,
            "fcz_lx": float(meta[12].item()), "fcz_ly": float(meta[13].item()),
            "fcz_rx": float(meta[14].item()), "fcz_ry": float(meta[15].item()),
        })
    return final_64, target_block, target_context, temp, top_k, cfg_coef, telemetry
