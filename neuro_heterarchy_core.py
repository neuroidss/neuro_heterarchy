#!/usr/bin/env python3
"""
===================================================================================
NEURO-HETERARCHY CORE v136.0 (FULL KINEMATICS + LIE SO(3) + ROBUST NODESTATE)
===================================================================================
- NodeState защищен значениями по умолчанию: исключены любые TypeError при вызове.
- 100% сохранение диффузии: Ли-алгебра SO(3), матрица поз pose_matrix (3x3), disp_xyz.
- 89.5 Гц Рипплы Дики (iplv_human_ripple) для SVD-глубины и каузального порядка.
- 4 кинематические оси gamepad_axes (хорда намерения L, сагитта rx, темпоральный сдвиг ry).
- Горячее подключение и отключение LSL-потоков без перезапуска ядра.
===================================================================================
"""

import os
import time
import math
import ctypes
import re
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass, field
import torch
from pylsl import StreamInlet, resolve_streams

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

BUF_SIZE = 256
FFT_SLOW = 1024
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 16
NUM_SLOTS = 32
NUM_FREQS = 32
NUM_PAIRS = 120
NUM_PAIRS_136 = 136
TWO_PI = 2.0 * math.pi
SCALE_28_120 = 28.0 / 120.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CANONICAL_ROLES = ["AFz", "Fpz", "F3", "F4", "FCz", "Pz", "Cz", "Oz"]

# Физические координаты сенсора FreeEEG16-alpha2 (26 мм)
COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)

REF_16_X = 5.50
REF_16_Y = 0.0
GND_16_X = -5.49
GND_16_Y = 0.0

I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)
DX_120 = (COORDS_X[J_IDX] - COORDS_X[I_IDX]).astype(np.float32)
DY_120 = (COORDS_Y[J_IDX] - COORDS_Y[I_IDX]).astype(np.float32)
TQ_MULT_120 = ((COORDS_X[I_IDX] * DY_120 - COORDS_Y[I_IDX] * DX_120) / 100.0).astype(np.float32)

DX_REF = (REF_16_X - COORDS_X).astype(np.float32)
DY_REF = (REF_16_Y - COORDS_Y).astype(np.float32)
DX_PAIR_136 = np.concatenate([DX_120, DX_REF]).astype(np.float32)
DY_PAIR_136 = np.concatenate([DY_120, DY_REF]).astype(np.float32)

DX_GPU_136 = torch.from_numpy(DX_PAIR_136 * 1.35).to(DEVICE)
DY_GPU_136 = torch.from_numpy(DY_PAIR_136 * 1.00).to(DEVICE)
I_GPU = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
J_GPU = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)
DX_GPU_120 = torch.from_numpy(DX_120).to(DEVICE).view(1, NUM_PAIRS)
DY_GPU_120 = torch.from_numpy(DY_120).to(DEVICE).view(1, NUM_PAIRS)
TQ_GPU_120 = torch.from_numpy(TQ_MULT_120).to(DEVICE).view(1, NUM_PAIRS)

norm_dx = DX_120 / (np.linalg.norm(DX_120) + 1e-6)
norm_dy = DY_120 / (np.linalg.norm(DY_120) + 1e-6)
PROJ_BATCH_GPU = torch.from_numpy(np.stack([np.stack([norm_dx, norm_dy], axis=1) for _ in range(NUM_MAX_DEVICES)], axis=0)).to(DEVICE)

@dataclass
class Kinematics4D:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0

UniversalGamepadAxes = Kinematics4D

@dataclass
class NodeState:
    device_id: int = 0
    name: str = ""
    source_id: str = ""
    sps: float = 250.0
    is_connected: bool = False
    phase_theta: float = 0.0
    phase_delta: float = 0.0
    beta_stability: float = 0.0
    beta_power: float = 0.0
    gamma_power: float = 0.0
    gating_ratio: float = 0.0
    torus_u: float = 0.0
    torus_v: float = 0.0
    disp_xyz: np.ndarray = None
    pose_matrix: np.ndarray = None
    kinematics: Kinematics4D = None
    traj_32: np.ndarray = None
    iplv_32: np.ndarray = None
    iplv_gamma: np.ndarray = None
    iplv_human_ripple: np.ndarray = None
    iplv_fast_ripple: np.ndarray = None
    vx: float = 0.0
    vy: float = 0.0
    tq: float = 0.0
    thrust: float = 0.0
    now_s0: np.ndarray = None
    future_sN: np.ndarray = None
    assigned_role: str = "AFz"
    assigned_user: str = "User1"

    def __post_init__(self):
        if self.kinematics is None: self.kinematics = Kinematics4D()
        if self.disp_xyz is None: self.disp_xyz = np.zeros(3, dtype=np.float64)
        if self.pose_matrix is None: self.pose_matrix = np.eye(3, dtype=np.float64)
        if self.traj_32 is None: self.traj_32 = np.zeros((NUM_SLOTS, 2), dtype=np.float64)
        if self.iplv_32 is None: self.iplv_32 = np.zeros((NUM_SLOTS, 120), dtype=np.float64)
        if self.iplv_gamma is None: self.iplv_gamma = np.zeros((NUM_SLOTS, 120), dtype=np.float64)
        if self.iplv_human_ripple is None: self.iplv_human_ripple = np.zeros((NUM_SLOTS, 120), dtype=np.float64)
        if self.iplv_fast_ripple is None: self.iplv_fast_ripple = np.zeros((NUM_SLOTS, 120), dtype=np.float64)
        if self.now_s0 is None: self.now_s0 = np.zeros(2, dtype=np.float64)
        if self.future_sN is None: self.future_sN = np.zeros(2, dtype=np.float64)

    @property
    def gamepad_axes(self) -> Kinematics4D:
        return self.kinematics

@dataclass
class UniversalFrame:
    nodes: list[NodeState]
    theta_freq: float
    delta_freq: float
    theta_phase: float
    delta_phase: float
    is_real: bool
    num_live: int

MultimodalFrame = UniversalFrame

def make_lie_so3_generator(omega: float, axis=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    norm = np.linalg.norm(axis) + 1e-7
    kx, ky, kz = axis / norm
    M_skew = np.array([
        [ 0.0, -kz,   ky],
        [ kz,   0.0, -kx],
        [-ky,   kx,   0.0]
    ], dtype=np.float32) * float(omega)
    return torch.from_numpy(M_skew).to(DEVICE)

def parse_stream_identity(name: str):
    user_match = re.search(r'(User\d+|[A-Za-z]+)', name, re.IGNORECASE)
    role_match = re.search(r'(AFz|Fpz|FCz|F3|F4|Pz|Cz|Oz)', name, re.IGNORECASE)

    user = user_match.group(1).capitalize() if user_match and "FreeEEG" not in user_match.group(1) else None
    role = role_match.group(1).capitalize() if role_match else None
    role_map = {"Afz": "AFz", "Fpz": "Fpz", "Fcz": "FCz", "F3": "F3", "F4": "F4"}
    role = role_map.get(role, role)
    return user, role

class GPU_Daemon_Process(mp.Process):
    def __init__(self, shared_mem, gamma_max=100.0):
        super().__init__()
        self.daemon = True
        self.shm = shared_mem
        self.gamma_max = gamma_max

    def run(self):
        inlets = [None] * NUM_MAX_DEVICES
        stream_uids = [""] * NUM_MAX_DEVICES
        stream_names = [""] * NUM_MAX_DEVICES
        connected_uids = set()
        device_fs = [250.0] * NUM_MAX_DEVICES

        raw_buffers = np.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        raw_buf_gpu = torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        base_fftfreq_256 = torch.fft.fftfreq(BUF_SIZE, d=1.0).to(DEVICE).view(1, 1, BUF_SIZE)
        base_rfftfreq_1024 = torch.fft.rfftfreq(FFT_SLOW, d=1.0).to(DEVICE).view(1, 1, 513)
        slot_angles = (-math.pi + (TWO_PI / NUM_SLOTS) * (torch.arange(NUM_SLOTS, device=DEVICE) + 0.5)).view(1, NUM_SLOTS, 1, 1)

        prev_beta_vecs = torch.zeros((NUM_MAX_DEVICES, 2), device=DEVICE)
        last_resolve_time = 0.0

        # Ссылки на Shared Memory
        sh_dev_th_phase   = np.frombuffer(self.shm['dev_th_phase'].get_obj(), dtype=np.float64)
        sh_dev_dl_phase   = np.frombuffer(self.shm['dev_dl_phase'].get_obj(), dtype=np.float64)
        sh_dev_sps        = np.frombuffer(self.shm['dev_sps'].get_obj(), dtype=np.float64)
        sh_beta_power     = np.frombuffer(self.shm['beta_power'].get_obj(), dtype=np.float64)
        sh_gamma_power    = np.frombuffer(self.shm['gamma_power'].get_obj(), dtype=np.float64)
        sh_gating_ratio   = np.frombuffer(self.shm['gating_ratio'].get_obj(), dtype=np.float64)
        sh_torus_coords   = np.frombuffer(self.shm['torus_coords'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 2)
        sh_kinematics     = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        sh_disp           = np.frombuffer(self.shm['disp'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3)
        sh_pose           = np.frombuffer(self.shm['pose'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 9)
        sh_beta_stab      = np.frombuffer(self.shm['beta_stab'].get_obj(), dtype=np.float64)
        sh_iplv_32        = np.frombuffer(self.shm['iplv_32'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        sh_iplv_gamma     = np.frombuffer(self.shm['iplv_gamma'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        sh_iplv_h_ripple  = np.frombuffer(self.shm['iplv_h_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        sh_iplv_f_ripple  = np.frombuffer(self.shm['iplv_f_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        sh_vx             = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        sh_vy             = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        sh_tq             = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        sh_gx             = np.frombuffer(self.shm['gx'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS)
        sh_gy             = np.frombuffer(self.shm['gy'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS)

        while self.shm['is_running'].value:
            now = time.time()

            # Динамический Hot-Plug сканер
            if now - last_resolve_time > 1.0:
                last_resolve_time = now
                try:
                    streams = resolve_streams(wait_time=0.05)
                    live_uids = {s.source_id() for s in streams}

                    for slot_i in range(NUM_MAX_DEVICES):
                        if inlets[slot_i] is not None and stream_uids[slot_i] not in live_uids:
                            print(f"🔌 [HOT-PLUG] Стрим '{stream_names[slot_i]}' отключен. Слот {slot_i} освобожден.")
                            connected_uids.discard(stream_uids[slot_i])
                            inlets[slot_i] = None
                            stream_uids[slot_i] = ""
                            stream_names[slot_i] = ""

                    for s in streams:
                        s_uid = s.source_id()
                        s_name = s.name()
                        if s_uid in connected_uids or s.channel_count() != NUM_CHANNELS:
                            continue

                        target_slot = next((i for i, inl in enumerate(inlets) if inl is None), None)
                        if target_slot is not None:
                            try:
                                inlets[target_slot] = StreamInlet(s, max_buflen=1, max_chunklen=32, recover=True)
                                connected_uids.add(s_uid)
                                stream_uids[target_slot] = s_uid
                                stream_names[target_slot] = s_name
                                nom_sps = inlets[target_slot].info().nominal_srate()
                                device_fs[target_slot] = 500.0 if nom_sps > 350.0 else 250.0
                                sh_dev_sps[target_slot] = device_fs[target_slot]

                                u, r = parse_stream_identity(s_name)
                                self.shm[f'slot_name_{target_slot}'].value = s_name.encode('utf-8')[:32]
                                self.shm[f'slot_parsed_role_{target_slot}'].value = (r or "").encode('utf-8')[:16]
                                self.shm[f'slot_parsed_user_{target_slot}'].value = (u or "").encode('utf-8')[:16]

                                print(f"📡 [HOT-PLUG] Подключен '{s_name}' -> Слот {target_slot} (Role: {r or 'Auto'}, User: {u or 'Auto'})")
                            except Exception:
                                pass
                except Exception:
                    pass

            num_live = sum(1 for inl in inlets if inl is not None)
            is_real = (num_live > 0)
            self.shm['is_real'].value = is_real
            self.shm['num_live'].value = num_live

            if not is_real:
                time.sleep(0.01)
                continue

            max_n_pulled = 0
            pulled_counts = [0] * NUM_MAX_DEVICES

            for i in range(NUM_MAX_DEVICES):
                if inlets[i] is not None:
                    try:
                        chunk, _ = inlets[i].pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                        if chunk:
                            arr = np.array(chunk, dtype=np.float32).T
                            n = arr.shape[1]
                            pulled_counts[i] = n
                            if n > max_n_pulled: max_n_pulled = n
                            if n >= BUF_SIZE:
                                raw_buffers[i] = arr[:NUM_CHANNELS, -BUF_SIZE:]
                            else:
                                raw_buffers[i] = np.roll(raw_buffers[i], -n, axis=1)
                                raw_buffers[i][:, -n:] = arr[:NUM_CHANNELS, :]
                    except Exception:
                        connected_uids.discard(stream_uids[i])
                        inlets[i] = None

            if max_n_pulled == 0:
                time.sleep(0.001)
                continue

            raw_buf_gpu.copy_(torch.from_numpy(raw_buffers))
            first_active = next(i for i, inl in enumerate(inlets) if inl is not None)
            fs_tensor = torch.tensor(device_fs, device=DEVICE, dtype=torch.float32).view(NUM_MAX_DEVICES, 1, 1)

            with torch.inference_mode():
                freqs_dev = base_fftfreq_256 * fs_tensor
                centered = raw_buf_gpu - torch.mean(raw_buf_gpu, dim=2, keepdim=True)

                notch = torch.ones_like(freqs_dev)
                notch[(torch.abs(freqs_dev) >= 48.5) & (torch.abs(freqs_dev) <= 51.5)] = 0.0
                notch[(torch.abs(freqs_dev) >= 98.5) & (torch.abs(freqs_dev) <= 101.5)] = 0.0
                fft_clean = torch.fft.fft(centered, dim=-1) * notch

                fft_1024 = torch.fft.rfft(centered[first_active], n=FFT_SLOW, dim=-1)
                active_fs = device_fs[first_active]
                freqs_slow = (base_rfftfreq_1024[0, 0] * active_fs)

                dl_mask = (freqs_slow >= 0.5) & (freqs_slow <= 3.2)
                dl_pwr = torch.sum(torch.abs(fft_1024[:, dl_mask])**2, dim=0)
                total_dl = torch.sum(dl_pwr)
                raw_dl_hz = (torch.sum(dl_pwr * freqs_slow[dl_mask]) / total_dl).item() if total_dl > 1e-6 else 1.5
                inst_delta_freq = float(np.clip(raw_dl_hz, 0.5, 3.2))

                th_mask = (freqs_slow >= 3.8) & (freqs_slow <= 8.5)
                th_pwr = torch.sum(torch.abs(fft_1024[:, th_mask])**2, dim=0)
                total_th = torch.sum(th_pwr)
                raw_th_hz = (torch.sum(th_pwr * freqs_slow[th_mask]) / total_th).item() if total_th > 1e-6 else 6.0
                inst_theta_freq = float(np.clip(raw_th_hz, 3.5, 9.0))

                f_theta = (torch.exp(-0.5 * ((freqs_dev - 6.0) / 1.5)**2) * 2.0)
                f_theta[:, :, base_fftfreq_256[0, 0] < 0] = 0.0
                Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                phi_theta_all = torch.angle(torch.mean(P_theta, dim=1))
                phi_theta_4d = phi_theta_all.unsqueeze(1).unsqueeze(1)

                f_delta = (torch.exp(-0.5 * ((freqs_dev - 1.5) / 0.6)**2) * 2.0)
                f_delta[:, :, base_fftfreq_256[0, 0] < 0] = 0.0
                Z_delta = torch.fft.ifft(fft_clean * f_delta, dim=-1)
                P_delta = Z_delta / (torch.abs(Z_delta) + 1e-12)
                phi_delta_all = torch.angle(torch.mean(P_delta, dim=1))

                self.shm['theta_phase'].value = float(phi_theta_all[first_active, -1].item())
                self.shm['delta_phase'].value = float(phi_delta_all[first_active, -1].item())
                self.shm['theta_freq'].value = inst_theta_freq
                self.shm['delta_freq'].value = inst_delta_freq

                # Бета-поток
                f_beta = (torch.exp(-0.5 * ((freqs_dev - 22.0) / 5.0)**2) * 2.0)
                f_beta[:, :, base_fftfreq_256[0, 0] < 0] = 0.0
                Z_beta = torch.fft.ifft(fft_clean * f_beta, dim=-1)
                P_beta = Z_beta / (torch.abs(Z_beta) + 1e-12)
                b_power = (torch.mean(torch.abs(torch.mean(P_beta, dim=1)), dim=-1)).cpu().numpy()

                cg_beta_120 = P_beta[:, I_GPU, :] * torch.conj(P_beta[:, J_GPU, :])
                iplv_beta_120 = torch.mean(torch.imag(cg_beta_120), dim=-1)
                P_ref = torch.mean(P_beta[:, [0, 1, 14, 15], :], dim=1, keepdim=True)
                iplv_beta_ref = torch.mean(torch.imag(P_beta * torch.conj(P_ref)), dim=-1)
                iplv_beta_136 = torch.cat([iplv_beta_120, iplv_beta_ref], dim=-1)

                vx_beta = torch.sum(iplv_beta_136 * DX_GPU_136, dim=-1) / float(NUM_PAIRS_136)
                vy_beta = torch.sum(iplv_beta_136 * DY_GPU_136, dim=-1) / float(NUM_PAIRS_136)
                cur_beta_vecs = torch.stack([vx_beta, vy_beta], dim=-1)
                beta_stabs = torch.clamp(torch.sum(cur_beta_vecs * prev_beta_vecs, dim=-1) / (torch.norm(cur_beta_vecs, dim=-1) * torch.norm(prev_beta_vecs, dim=-1) + 1e-6), -1.0, 1.0)
                prev_beta_vecs.copy_(cur_beta_vecs)

                vx_classic = torch.sum(iplv_beta_120 * DX_GPU_120, dim=-1) * (SCALE_28_120 * 15.0)
                vy_classic = torch.sum(iplv_beta_120 * DY_GPU_120, dim=-1) * (SCALE_28_120 * 15.0)
                tq_classic = torch.sum(iplv_beta_120 * TQ_GPU_120, dim=-1) * (SCALE_28_120 * 18.0)

                # Гамма и PAC-траектория
                fft_exp = fft_clean.unsqueeze(1)
                freqs_4d = freqs_dev.unsqueeze(1)
                p_diff = phi_theta_4d - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                gamma_centers = torch.linspace(30.0, self.gamma_max, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
                gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
                gamma_filters[:, :, :, base_fftfreq_256[0, 0] < 0] = 0.0
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1)
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)
                cg_gamma_120 = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
                psi_field_120 = torch.sum(cg_gamma_120 * w, dim=-1)
                gamma_anchor_120 = torch.imag(psi_field_120 * torch.conj(psi_field_120[:, 0:1, :]))
                gamma_ci_120 = torch.imag(psi_field_120) / torch.sqrt(torch.clamp(1.0 - torch.real(psi_field_120)**2, min=1e-5))

                g_power = torch.mean(torch.abs(torch.mean(P_gamma, dim=2)), dim=(1, 2)).cpu().numpy()
                gating_ratios = g_power / (g_power + b_power + 1e-6)

                # 89.5 Гц Рипплы Дики
                h_rip_centers = torch.linspace(70.0, 100.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
                h_rip_filt = torch.exp(-0.5 * ((freqs_4d - h_rip_centers) / 4.0)**2) * 2.0
                h_rip_filt[:, :, :, base_fftfreq_256[0, 0] < 0] = 0.0
                Z_rip = torch.fft.ifft(fft_exp * h_rip_filt, dim=-1)
                P_rip = Z_rip / (torch.abs(Z_rip) + 1e-12)
                cg_rip = P_rip[:, :, I_GPU, :] * torch.conj(P_rip[:, :, J_GPU, :])
                psi_rip = torch.sum(cg_rip * w, dim=-1)
                h_rip_120 = torch.imag(psi_rip) / torch.sqrt(torch.clamp(1.0 - torch.real(psi_rip)**2, min=1e-5))

                # 100-200 Гц Быстрые рипплы
                f_rip_centers = torch.linspace(100.0, 200.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
                f_rip_filt = torch.exp(-0.5 * ((freqs_4d - f_rip_centers) / 7.0)**2) * 2.0
                f_rip_filt[:, :, :, base_fftfreq_256[0, 0] < 0] = 0.0
                Z_frip = torch.fft.ifft(fft_exp * f_rip_filt, dim=-1)
                P_frip = Z_frip / (torch.abs(Z_frip) + 1e-12)
                cg_frip = P_frip[:, :, I_GPU, :] * torch.conj(P_frip[:, :, J_GPU, :])
                psi_frip = torch.sum(cg_frip * w, dim=-1)
                f_rip_120 = torch.imag(psi_frip) / torch.sqrt(torch.clamp(1.0 - torch.real(psi_frip)**2, min=1e-5))

                # Ли-алгебра SO(3) 3D displacement и Frenet pose frame
                M_skew = make_lie_so3_generator(omega=TWO_PI * inst_theta_freq)
                v_gx = torch.sum(gamma_anchor_120[:, 31] * DX_GPU_120, dim=-1) / 120.0
                v_gy = torch.sum(gamma_anchor_120[:, 31] * DY_GPU_120, dim=-1) / 120.0
                g_vecs = torch.stack([v_gx, v_gy, v_gx * 0.3], dim=-1)

                dt_per_dev = torch.tensor(pulled_counts, device=DEVICE, dtype=torch.float32).view(NUM_MAX_DEVICES, 1) / fs_tensor.view(NUM_MAX_DEVICES, 1)
                disp_all = torch.matmul(g_vecs, M_skew.T) * dt_per_dev * 0.05

                u1 = disp_all / (torch.norm(disp_all, dim=-1, keepdim=True) + 1e-6)
                up_ref = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).view(1, 3).expand(NUM_MAX_DEVICES, 3)
                u2 = torch.cross(u1, up_ref, dim=-1)
                u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-6)
                u3 = torch.cross(u1, u2, dim=-1)
                pose_matrices = torch.stack([u1, u2, u3], dim=1)

                # Траектория PAC и кинематические оси намерения
                traj_2d = torch.bmm(gamma_anchor_120, PROJ_BATCH_GPU) * 8.0
                base_x, base_y = traj_2d[:, 0, 0], traj_2d[:, 0, 1]
                end_x, end_y = traj_2d[:, -1, 0] - base_x, traj_2d[:, -1, 1] - base_y
                d_len = torch.hypot(end_x, end_y) + 1e-6
                lx_gpu = torch.where(d_len > 1.0, end_x / d_len, end_x)
                ly_gpu = torch.where(d_len > 1.0, end_y / d_len, end_y)

                sagitta_sum = torch.zeros(NUM_MAX_DEVICES, device=DEVICE)
                for k in range(1, NUM_SLOTS - 1):
                    px_k = traj_2d[:, k, 0] - base_x
                    py_k = traj_2d[:, k, 1] - base_y
                    sagitta_sum += (end_x * py_k - end_y * px_k)
                rx_gpu = torch.clamp(sagitta_sum / (d_len * 16.0), -1.0, 1.0)

                mid_idx = NUM_SLOTS // 2
                mid_x = traj_2d[:, mid_idx, 0] - base_x
                mid_y = traj_2d[:, mid_idx, 1] - base_y
                len_past = torch.hypot(mid_x, mid_y)
                len_future = torch.hypot(end_x - mid_x, end_y - mid_y)
                ry_gpu = (len_future - len_past) / (len_future + len_past + 1e-6)

                kinematics_gpu = torch.stack([lx_gpu, -ly_gpu, rx_gpu, ry_gpu], dim=-1)

                # Запись в Shared Memory
                traj_cpu = traj_2d.cpu().numpy()
                np.copyto(sh_vx, vx_classic.cpu().numpy())
                np.copyto(sh_vy, vy_classic.cpu().numpy())
                np.copyto(sh_tq, tq_classic.cpu().numpy())
                np.copyto(sh_gx, traj_cpu[:, :, 0])
                np.copyto(sh_gy, traj_cpu[:, :, 1])
                np.copyto(sh_disp, disp_all.cpu().numpy())
                np.copyto(sh_pose, pose_matrices.contiguous().view(NUM_MAX_DEVICES, 9).cpu().numpy())
                np.copyto(sh_iplv_32, gamma_anchor_120.cpu().numpy())
                np.copyto(sh_iplv_gamma, gamma_ci_120.cpu().numpy())
                np.copyto(sh_iplv_h_ripple, h_rip_120.cpu().numpy())
                np.copyto(sh_iplv_f_ripple, f_rip_120.cpu().numpy())
                np.copyto(sh_beta_power, b_power)
                np.copyto(sh_gamma_power, g_power)
                np.copyto(sh_gating_ratio, gating_ratios)
                np.copyto(sh_kinematics, kinematics_gpu.cpu().numpy())
                np.copyto(sh_torus_coords, np.stack([(phi_theta_all[:, -1] + math.pi).cpu().numpy(), (phi_delta_all[:, -1] + math.pi).cpu().numpy()], axis=-1))
                np.copyto(sh_beta_stab, beta_stabs.cpu().numpy())
                np.copyto(sh_dev_th_phase, phi_theta_all[:, -1].cpu().numpy())
                np.copyto(sh_dev_dl_phase, phi_delta_all[:, -1].cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self, gamma_max: float = 65.0):
        self.gamma_max = gamma_max
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.8),
            'theta_freq': mp.Value('d', 6.0),
            'delta_freq': mp.Value('d', 1.5),
            'theta_phase': mp.Value('d', 0.0),
            'delta_phase': mp.Value('d', 0.0),
            'dev_th_phase': mp.Array('d', NUM_MAX_DEVICES),
            'dev_dl_phase': mp.Array('d', NUM_MAX_DEVICES),
            'dev_sps': mp.Array('d', [250.0] * NUM_MAX_DEVICES),
            'beta_power': mp.Array('d', NUM_MAX_DEVICES),
            'gamma_power': mp.Array('d', NUM_MAX_DEVICES),
            'gating_ratio': mp.Array('d', NUM_MAX_DEVICES),
            'torus_coords': mp.Array('d', NUM_MAX_DEVICES * 2),
            'kinematics': mp.Array('d', NUM_MAX_DEVICES * 4),
            'disp': mp.Array('d', NUM_MAX_DEVICES * 3),
            'pose': mp.Array('d', NUM_MAX_DEVICES * 9),
            'beta_stab': mp.Array('d', NUM_MAX_DEVICES),
            'iplv_32': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * 120),
            'iplv_gamma': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * 120),
            'iplv_h_ripple': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * 120),
            'iplv_f_ripple': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * 120),
            'vx': mp.Array('d', NUM_MAX_DEVICES),
            'vy': mp.Array('d', NUM_MAX_DEVICES),
            'tq': mp.Array('d', NUM_MAX_DEVICES),
            'gx': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS),
            'gy': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS)
        }
        for i in range(NUM_MAX_DEVICES):
            self.shm[f'slot_name_{i}'] = mp.Array('c', 32)
            self.shm[f'slot_parsed_role_{i}'] = mp.Array('c', 16)
            self.shm[f'slot_parsed_user_{i}'] = mp.Array('c', 16)

        self._dev_th_phase   = np.frombuffer(self.shm['dev_th_phase'].get_obj(), dtype=np.float64)
        self._dev_dl_phase   = np.frombuffer(self.shm['dev_dl_phase'].get_obj(), dtype=np.float64)
        self._dev_sps        = np.frombuffer(self.shm['dev_sps'].get_obj(), dtype=np.float64)
        self._beta_power     = np.frombuffer(self.shm['beta_power'].get_obj(), dtype=np.float64)
        self._gamma_power    = np.frombuffer(self.shm['gamma_power'].get_obj(), dtype=np.float64)
        self._gating_ratio   = np.frombuffer(self.shm['gating_ratio'].get_obj(), dtype=np.float64)
        self._torus_coords   = np.frombuffer(self.shm['torus_coords'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 2)
        self._kinematics     = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        self._disp           = np.frombuffer(self.shm['disp'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3)
        self._pose           = np.frombuffer(self.shm['pose'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3, 3)
        self._beta_stab      = np.frombuffer(self.shm['beta_stab'].get_obj(), dtype=np.float64)
        self._iplv_32        = np.frombuffer(self.shm['iplv_32'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        self._iplv_gamma     = np.frombuffer(self.shm['iplv_gamma'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        self._iplv_h_ripple  = np.frombuffer(self.shm['iplv_h_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        self._iplv_f_ripple  = np.frombuffer(self.shm['iplv_f_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, 120)
        self._vx             = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        self._vy             = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        self._tq             = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        self._gx             = np.frombuffer(self.shm['gx'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS)
        self._gy             = np.frombuffer(self.shm['gy'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS)

        self.process = GPU_Daemon_Process(self.shm, gamma_max=self.gamma_max)

    def start(self): self.process.start()
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)

    def get_frame(self, user_role_overrides: dict = None) -> UniversalFrame:
        num_live = self.shm['num_live'].value
        nodes = []
        for i in range(num_live):
            raw_name = self.shm[f'slot_name_{i}'].value.decode('utf-8', errors='ignore').strip()
            parsed_role = self.shm[f'slot_parsed_role_{i}'].value.decode('utf-8', errors='ignore').strip()
            parsed_user = self.shm[f'slot_parsed_user_{i}'].value.decode('utf-8', errors='ignore').strip()

            override = (user_role_overrides or {}).get(i, {})
            final_user = override.get('user') or (parsed_user if parsed_user else "User1")
            final_role = override.get('role') or (parsed_role if parsed_role else CANONICAL_ROLES[i % len(CANONICAL_ROLES)])

            traj = np.stack([self._gx[i], self._gy[i]], axis=-1)
            vx_val, vy_val = float(self._vx[i]), float(self._vy[i])
            tc = self._torus_coords[i]
            k = self._kinematics[i]
            kin = Kinematics4D(lx=float(k[0]), ly=float(k[1]), rx=float(k[2]), ry=float(k[3]))

            nodes.append(NodeState(
                device_id=i,
                name=raw_name or f"Dev_{i}",
                source_id=f"Slot_{i}",
                sps=float(self._dev_sps[i]),
                is_connected=True,
                phase_theta=float(self._dev_th_phase[i]),
                phase_delta=float(self._dev_dl_phase[i]),
                beta_stability=float(self._beta_stab[i]),
                beta_power=float(self._beta_power[i]),
                gamma_power=float(self._gamma_power[i]),
                gating_ratio=float(self._gating_ratio[i]),
                torus_u=float(tc[0]),
                torus_v=float(tc[1]),
                disp_xyz=self._disp[i].copy(),
                pose_matrix=self._pose[i].copy(),
                kinematics=kin,
                traj_32=traj,
                iplv_32=self._iplv_32[i].copy(),
                iplv_gamma=self._iplv_gamma[i].copy(),
                iplv_human_ripple=self._iplv_h_ripple[i].copy(),
                iplv_fast_ripple=self._iplv_f_ripple[i].copy(),
                vx=vx_val,
                vy=vy_val,
                tq=float(self._tq[i]),
                thrust=math.hypot(vx_val, vy_val),
                now_s0=traj[0].copy(),
                future_sN=traj[-1].copy(),
                assigned_role=final_role,
                assigned_user=final_user
            ))

        return UniversalFrame(
            nodes=nodes,
            theta_freq=self.shm['theta_freq'].value,
            delta_freq=self.shm['delta_freq'].value,
            theta_phase=self.shm['theta_phase'].value,
            delta_phase=self.shm['delta_phase'].value,
            is_real=self.shm['is_real'].value,
            num_live=num_live
        )
