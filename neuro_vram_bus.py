#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: HARDWARE VRAM CUDA-IPC BUS & UNIVERSAL FULL-RANK MANIFOLD OPERATOR
===================================================================================
- Никаких if/else и никаких token_mask:
  Вся матрица [L × D] является единым непрерывным тензорным полем в рабочей памяти!
- Непрерывный аналитический геодезический переход по параметру stochastic ∈ [0.0, 1.0]:
  * stochastic = 1.0 -> 100% чистое опорное состояние лора (0% смещения WM).
  * stochastic = 0.0 -> 100% максимальный геодезический сдвиг рабочей памяти мозга.
- Входной тензор рабочей памяти строится из ciPLV ∈ [-1.0, +1.0] (Bruña et al., 2018,
  DOI:10.1088/1741-2552/aacfe4).
- Надежное авто-переподключение try_attach_to_existing_vram() без вечной блокировки
  attach_failed, даже если сервер стартовал раньше раннера или раннер перезапускался.
===================================================================================
"""

from __future__ import annotations
from typing import Optional, Union, Tuple
import os
import json
import time
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE

MAX_SLOTS = 16
MAX_CLOCK_SECTORS = 16
NUM_REGIONS = 5
NUM_BANDS = 5       # 0: Delta, 1: Theta, 2: Beta, 3: Gamma, 4: Ripples (89.5 Hz)
NUM_SLOTS = 32
NUM_PAIRS = 120
NUM_COLUMNS = 4096
K_ACTIVE = 80

BAND_DELTA   = 0
BAND_THETA   = 1
BAND_BETA    = 2
BAND_GAMMA   = 3
BAND_RIPPLES = 4

REGION_ROLES = ["AFz", "FCz", "F3", "F4", "Fpz"]

EXTENDED_ROLE_TO_FUNCTIONAL = {
    "Fpz": "Fpz",
    "AFz": "AFz",
    "F3":  "F3",
    "F4":  "F4",
    "FCz": "FCz",
    "Cz":  "FCz",
    "Pz":  "FCz",
    "Oz":  "F4",
}

REGION_TAU_SEC = {
    "FCz": 0.020,
    "F3":  0.18,
    "F4":  0.18,
    "AFz": 0.65,
    "Fpz": 2.50,
}

MONTY_MODE_MAP = {"off": 0.0, "missing_only": 1.0, "full": 2.0}
MONTY_CODE_TO_STR = {0: "off", 1: "missing_only", 2: "full"}

IPC_HANDLE_FILE = "/dev/shm/neuro_amodal_vram_v4_ipc.json"
TOTAL_AMODAL_FLOATS = 318576


@torch.no_grad()
def universal_full_rank_manifold_step(
    anchor_bank_KxLxD: torch.Tensor,
    wm_tensor_LxD: torch.Tensor,
    gain_gamma: Union[float, torch.Tensor] = 0.35,
    stochastic: Union[float, torch.Tensor] = 0.0,
    temp: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    НЕПРЕРЫВНЫЙ АНАЛИТИЧЕСКИЙ ОПЕРАТОР РАБОЧЕЙ ПАМЯТИ В ПРОСТРАНСТВЕ [L × D]
    (БЕЗ ЕДИНОГО ЗАФИКСИРОВАННОГО ТОКЕНА И БЕЗ ВЕТВЛЕНИЙ IF/ELSE):
    1) Вычисляет координаты состояния рабочей памяти в базисе опорных концептов
       через полное скалярное произведение Фробениуса по всем L * D измерениям.
    2) Проецирует волну рабочей памяти wm_tensor_LxD на собственное касательное
       подпространство текущего состояния многообразия (Subspace Projection,
       Chen et al., Neuron 2024; Fan et al., Nature Human Behaviour 2024).
    3) Поворачивает всю матрицу [L × D] по геодезической дуге на угол theta_rad,
       который гладко зависит от (1 - stochastic) и энергии рабочей памяти wm_energy.
    """
    K, L, D = anchor_bank_KxLxD.shape
    stoch_t = torch.clamp(
        torch.as_tensor(stochastic, device=anchor_bank_KxLxD.device, dtype=torch.float32),
        0.0, 1.0
    )
    wm_influence = 1.0 - stoch_t

    # 1. Полное скалярное произведение Фробениуса во всех L * D измерениях:
    bank_flat = anchor_bank_KxLxD.reshape(K, L * D)
    delta_bank = bank_flat - bank_flat.mean(dim=0, keepdim=True)
    delta_bank_unit = delta_bank / (torch.norm(delta_bank, p=2, dim=-1, keepdim=True) + 1e-6)

    wm_flat = wm_tensor_LxD.reshape(1, L * D)
    wm_centered = wm_flat - wm_flat.mean(dim=-1, keepdim=True)
    wm_energy = torch.norm(wm_centered, p=2, dim=-1, keepdim=True)
    wm_unit = wm_centered / (wm_energy + 1e-6)

    sims_K = torch.mm(wm_unit, delta_bank_unit.T).squeeze(0)  # [K]
    sims_std = torch.std(sims_K, unbiased=False)
    sims_scaled = (sims_K - sims_K.mean()) / (sims_std + 1e-6)
    weights_K = torch.softmax(sims_scaled / temp, dim=0)      # [K]

    # 2. Непрерывная суперпозиция концептов в R^[L x D]:
    anchor_point_LxD = torch.sum(anchor_bank_KxLxD * weights_K.view(K, 1, 1), dim=0)
    ref_norms_Lx1 = torch.norm(anchor_point_LxD, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

    # 3. Проекция волны рабочей памяти wm_tensor_LxD в собственное касательное подпространство:
    anchor_centered = anchor_point_LxD - anchor_point_LxD.mean(dim=0, keepdim=True)
    inter_concept_modes = (anchor_bank_KxLxD - anchor_point_LxD.unsqueeze(0)).reshape(K * L, D)
    manifold_span = torch.cat([anchor_centered, inter_concept_modes], dim=0)
    span_unit = F.normalize(manifold_span + 1e-6, p=2, dim=-1)  # [M, D]

    wm_coords_LxM = torch.matmul(wm_tensor_LxD, span_unit.T) / float(max(1, span_unit.shape[0]) ** 0.5)
    wm_tangent_LxD = torch.matmul(wm_coords_LxM, span_unit)     # [L, D]

    dot_proj = torch.sum(wm_tangent_LxD * anchor_point_LxD, dim=-1, keepdim=True) / (ref_norms_Lx1 ** 2 + 1e-6)
    wm_geodesic_LxD = wm_tangent_LxD - dot_proj * anchor_point_LxD
    geo_norms = torch.norm(wm_geodesic_LxD, p=2, dim=-1, keepdim=True)
    wm_unit_tangent = (wm_geodesic_LxD / (geo_norms + 1e-6)) * ref_norms_Lx1

    # 4. Гладкий аналитический угол геодезического поворота всей матрицы [L × D]:
    # При wm_energy == 0 или stochastic == 1.0 угол theta_rad строго равен 0.0!
    gate_activation = torch.tanh(wm_energy * 8.0) * wm_influence
    eff_gamma = torch.clamp(
        torch.as_tensor(gain_gamma, device=anchor_point_LxD.device, dtype=torch.float32),
        0.0, 0.85
    )
    theta_rad = eff_gamma * (3.14159265 * 0.25) * gate_activation

    out_LxD = torch.cos(theta_rad) * anchor_point_LxD + torch.sin(theta_rad) * wm_unit_tangent
    out_LxD = F.normalize(out_LxD, p=2, dim=-1) * ref_norms_Lx1

    best_idx = int(torch.argmax(weights_K).item())
    conf_pct = int(round(float(weights_K[best_idx].item()) * 100.0))
    return out_LxD, weights_K, best_idx, conf_pct


class HardwareVRAMBus:
    def __init__(self):
        self.gpu_flat = torch.zeros(TOTAL_AMODAL_FLOATS, dtype=torch.float32, device=DEVICE)
        self._bind_views()
        self.is_owner = False
        self.attached_pid = -1
        self._last_attach_attempt = 0.0

    def _bind_views(self):
        self.meta_64                = self.gpu_flat[0:64]
        self.meta_32                = self.gpu_flat[0:32]
        self.amodal_manifold_77x768 = self.gpu_flat[64:59200].view(77, 768)
        self.z_slots_16x768         = self.gpu_flat[59200:71488].view(MAX_SLOTS, 768)
        self.causal_flow_16x16      = self.gpu_flat[71488:71744].view(MAX_SLOTS, MAX_SLOTS)
        self.slot_order_16          = self.gpu_flat[71744:71760]
        self.slot_stabs_16          = self.gpu_flat[71760:71776]
        self.slot_seals_16          = self.gpu_flat[71776:71792]
        self.slot_norm_coord_16     = self.gpu_flat[71792:71808]
        self.wm_field_5x5x32x120    = self.gpu_flat[71808:167808].view(NUM_REGIONS, NUM_BANDS, NUM_SLOTS, NUM_PAIRS)
        self.monty_l23_5x4096       = self.gpu_flat[167808:188288].view(NUM_REGIONS, NUM_COLUMNS)
        self.monty_anom_5x4096      = self.gpu_flat[188288:208768].view(NUM_REGIONS, NUM_COLUMNS)
        self.ripple_dag_5x16x16     = self.gpu_flat[208768:210048].view(NUM_REGIONS, MAX_CLOCK_SECTORS, MAX_CLOCK_SECTORS)
        self.seals_and_stabs_5x3x16 = self.gpu_flat[210048:210288].view(NUM_REGIONS, 3, MAX_CLOCK_SECTORS)
        self.live_clip_77x768       = self.gpu_flat[210288:269424].view(77, 768)
        self.live_music_64x768      = self.gpu_flat[269424:318576].view(64, 768)

    def get_cuda_ipc_Payload(self) -> dict:
        storage = self.gpu_flat.untyped_storage()
        tup = storage._share_cuda_()
        return {
            "pid": os.getpid(),
            "ts": time.time(),
            "total_floats": TOTAL_AMODAL_FLOATS,
            "device_idx": int(tup[0]),
            "handle": tup[1].hex(),
            "size_bytes": int(tup[2]),
            "offset_bytes": int(tup[3]),
            "ref_counter_handle": tup[4].hex(),
            "ref_counter_offset": int(tup[5]),
            "event_handle": tup[6].hex(),
            "event_sync_required": bool(tup[7]),
        }

    def allocate_and_publish_vram_handle(self):
        if self.is_owner:
            return
        try:
            payload = self.get_cuda_ipc_Payload()
            tmp_path = f"{IPC_HANDLE_FILE}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, IPC_HANDLE_FILE)
            self.is_owner = True
            print(f"⚡ [VRAM CUDA-IPC] Опубликована полноранговая шина рабочей памяти в VRAM (PID={os.getpid()})")
        except Exception as e:
            print(f"⚠️ [VRAM CUDA-IPC Note]: {e}")
            self.is_owner = True

    def attach_from_ipc_dict(self, d: dict) -> bool:
        try:
            owner_pid = int(d["pid"])
            if self.attached_pid == owner_pid and os.path.exists(f"/proc/{owner_pid}"):
                return True
            if int(d.get("total_floats", 0)) != TOTAL_AMODAL_FLOATS:
                return False
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
                shared_storage, 0, (TOTAL_AMODAL_FLOATS,)
            )
            self._bind_views()
            self.attached_pid = owner_pid
            print(f"🔗 [VRAM CUDA-IPC] Прямой Zero-Copy доступ к рабочей памяти в VRAM PID={owner_pid}")
            return True
        except Exception:
            self.attached_pid = -1
            return False

    def try_attach_to_existing_vram(self) -> bool:
        if self.is_owner:
            return False
        if self.attached_pid > 0:
            if os.path.exists(f"/proc/{self.attached_pid}"):
                return True
            self.attached_pid = -1

        now = time.time()
        if now - self._last_attach_attempt < 0.5:
            return False
        self._last_attach_attempt = now

        if not os.path.exists(IPC_HANDLE_FILE):
            return False
        try:
            with open(IPC_HANDLE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            owner_pid = int(d["pid"])
            if owner_pid == os.getpid() or not os.path.exists(f"/proc/{owner_pid}"):
                return False
            return self.attach_from_ipc_dict(d)
        except Exception:
            return False
