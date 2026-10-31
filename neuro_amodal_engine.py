#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: MODULAR AMODAL COGNITIVE ENGINE ORCHESTRATOR (neuro_amodal_engine.py)
===================================================================================
Компактный оркестратор, объединяющий 3 независимых подмодуля:
  1) neuro_vram_bus.py          — HardwareVRAMBus и universal_full_rank_manifold_step.
  2) neuro_spectral_frontend.py — GEOM_GPU и DirectGPUSpectralFrontend.
  3) neuro_monty_columns.py     — AmodalMontyColumnGPU, GPUSealEngine, AmodalActionIntegratorGPU.
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import os
import time
import math
import torch
import torch.nn as nn

from neuro_heterarchy_core import DEVICE
from neuro_vram_bus import (
    MAX_SLOTS,
    MAX_CLOCK_SECTORS,
    NUM_REGIONS,
    NUM_BANDS,
    NUM_SLOTS,
    NUM_PAIRS,
    NUM_COLUMNS,
    K_ACTIVE,
    BAND_DELTA,
    BAND_THETA,
    BAND_BETA,
    BAND_GAMMA,
    BAND_RIPPLES,
    REGION_ROLES,
    EXTENDED_ROLE_TO_FUNCTIONAL,
    REGION_TAU_SEC,
    MONTY_MODE_MAP,
    MONTY_CODE_TO_STR,
    IPC_HANDLE_FILE,
    TOTAL_AMODAL_FLOATS,
    HardwareVRAMBus,
    universal_full_rank_manifold_step,
)
from neuro_spectral_frontend import (
    BUF_SIZE,
    FFT_SLOW,
    NUM_CHANNELS,
    NUM_MAX_DEVICES,
    TWO_PI,
    GEOM_GPU,
    build_scalable_dipole_geometry_gpu,
    build_cortical_dipole_geometry_gpu,
    DirectGPUSpectralFrontend,
)
from neuro_monty_columns import (
    GPUSealEngine,
    AmodalActionIntegratorGPU,
    AmodalMontyColumnGPU,
)

PHYSICAL_PROJECTION_768x120 = GEOM_GPU["proj_768xP"]


class SelfSufficientHeterarchyRouter:
    """Маршрутизатор Control Plane: сопоставляет физические потоки FreeEEG_<ID> с ролями 10-20."""
    ALL_ROLES = ["AFz", "F3", "F4", "FCz", "Fpz"]

    def __init__(self, users_setup=None):
        self.device_mappings: dict[str, dict] = {}
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

    def set_device_mapping(self, stream_name: str, role: str, user: str = "User1"):
        clean_name = str(stream_name).strip()
        clean_role = str(role).strip()
        clean_user = str(user).strip() or "User1"
        self.device_mappings[clean_name] = {"role": clean_role, "user": clean_user}

    def set_role(self, dev_idx: int, role: Optional[str] = None):
        self.overrides[dev_idx] = role

    def resolve(self, nodes):
        res = {r: [] for r in self.ALL_ROLES}
        for nd in nodes:
            r = self.overrides.get(nd.device_id) or self.static_map.get(nd.device_id) or nd.assigned_role
            if r in res:
                res[r].append(nd)
        return res

class SharedAmodalCognitiveEngine:
    def __init__(self, router, gamma_max: float = 100.0, strength_low: float = 0.50,
                 strength_high: float = 0.85, cross_region_bleed: float = 0.65,
                 monty_mode: str = "missing_only"):
        self.router = router
        self.gamma_max = gamma_max
        self.strength_low = strength_low
        self.strength_high = strength_high
        self.cross_region_bleed = cross_region_bleed
        self.monty_mode = monty_mode if monty_mode in MONTY_MODE_MAP else "missing_only"
        self.bus = HardwareVRAMBus()
        self.high_pri_stream = torch.cuda.Stream(device=DEVICE, priority=-1) if DEVICE.type == "cuda" else None

        self.is_host = False
        self.spectral_gpu: Optional[DirectGPUSpectralFrontend] = None
        self.columns: Optional[nn.ModuleDict] = None
        self.measured_ratio = 4.0
        self.delta_armed = False

    def set_monty_mode(self, mode: str):
        clean_m = str(mode).strip().lower()
        if clean_m in ("0", "false", "none", "off", "no_monty", "pure_wm"):
            new_mode = "off"
        elif clean_m in ("1", "missing", "missing_only", "operator"):
            new_mode = "missing_only"
        else:
            new_mode = "full"

        # Если режим не изменился — не выводим лог в консоль и выходим
        if new_mode == self.monty_mode:
            return

        self.monty_mode = new_mode
        self.bus.meta_64[27] = MONTY_MODE_MAP[self.monty_mode]
        print(f"🧠 [MONTY MODE] Установлен режим Монти: '{self.monty_mode.upper()}'", flush=True)

    def _ensure_host_initialized(self):
        if self.spectral_gpu is None:
            print(f"🧠 [100% GPU AMODAL ENGINE] Старт LSL->VRAM конвейера (MontyMode={self.monty_mode})...")
            self.bus.allocate_and_publish_vram_handle()
            self.spectral_gpu = DirectGPUSpectralFrontend(GEOM_GPU, gamma_max=self.gamma_max).to(DEVICE)
            self.columns = nn.ModuleDict({
                r: AmodalMontyColumnGPU(r, GEOM_GPU).to(DEVICE) for r in REGION_ROLES
            })
            self.bus.meta_64[27] = MONTY_MODE_MAP[self.monty_mode]
        self.is_host = True

    @torch.no_grad()
    def reset_seals(self):
        if self.columns is not None:
            for col in self.columns.values():
                col.seals.reset_gpu()

    @torch.no_grad()
    def _update_bus_from_eeg_dict(self, eeg_out: dict, dt: float) -> tuple[HardwareVRAMBus, bool]:
        th_hz_t = eeg_out["th_freq"]
        dl_hz_t = eeg_out["dl_freq"]
        dl_phase_t = eeg_out["dl_phase"]
        th_phase_t = eeg_out["th_phase"]

        ratio_now = float((th_hz_t / dl_hz_t).item())
        dl_phase = float(dl_phase_t.item())

        self.measured_ratio = self.measured_ratio * 0.95 + ratio_now * 0.05
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

        real_iplv  = eeg_out["iplv_by_role"]
        real_rip   = eeg_out["rip_by_role"]
        real_kin   = eeg_out["kin_by_role"]
        real_bands = eeg_out["bands_by_role"]
        master_iplv = eeg_out["master_iplv"]
        master_rip  = eeg_out["master_rip"]
        master_5x32x120 = eeg_out["master_5x32x120"]

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

        parent_field_32xP = {
            "FCz": 0.5 * (self.columns["F3"].predicted_32xP + self.columns["F4"].predicted_32xP),
            "F3":  self.columns["AFz"].predicted_32xP,
            "F4":  self.columns["AFz"].predicted_32xP,
            "AFz": self.columns["Fpz"].predicted_32xP,
            "Fpz": None,
        }
        child_anom_32xP = {
            "FCz": None,
            "F3":  self.columns["FCz"].anomaly_32xP,
            "F4":  self.columns["FCz"].anomaly_32xP,
            "AFz": 0.5 * (self.columns["F3"].anomaly_32xP + self.columns["F4"].anomaly_32xP),
            "Fpz": self.columns["AFz"].anomaly_32xP,
        }

        res = {}
        meta = self.bus.meta_64
        for r_idx, r in enumerate(REGION_ROLES):
            is_phys = (r in real_iplv)
            meta[21 + r_idx] = 1.0 if is_phys else 0.0

            if self.monty_mode == "off":
                run_monty_here = False
            elif self.monty_mode == "missing_only":
                run_monty_here = not is_phys
            else:
                run_monty_here = True

            if is_phys:
                ip_r, rp_r = real_iplv[r], real_rip[r]
                b_5x32x120 = real_bands[r]
            else:
                eff_bleed = max(0.85, self.cross_region_bleed) if self.monty_mode == "off" else self.cross_region_bleed
                ip_r, rp_r, b_5x32x120 = self.columns[r].synthesize_virtual_gpu(
                    master_iplv, master_rip, master_5x32x120,
                    bleed_scale=eff_bleed,
                    parent_topdown_32xP=parent_field_32xP[r],
                    child_anomaly_32xP=child_anom_32xP[r],
                    use_monty_operator=(self.monty_mode != "off")
                )

            res[r] = self.columns[r].forward_gpu(
                ip_r, rp_r, b_5x32x120, n_sec=n_sec, cur_sec=cur_sec, is_tick=is_tick, dt=dt,
                top_down_parent_sdr=parent_map[r] if run_monty_here else None,
                bottom_up_child_anomaly=child_map[r] if run_monty_here else None,
                enable_monty_sdr=run_monty_here
            )
            self.bus.wm_field_5x5x32x120[r_idx].copy_(b_5x32x120)
            self.bus.monty_l23_5x4096[r_idx].copy_(self.columns[r].l23_pooled_sdr)
            self.bus.monty_anom_5x4096[r_idx].copy_(self.columns[r].upward_sdr_anomaly)
            self.bus.ripple_dag_5x16x16[r_idx].zero_()
            self.bus.ripple_dag_5x16x16[r_idx, :n_sec, :n_sec].copy_(res[r]["c_flow_mat"])
            self.bus.seals_and_stabs_5x3x16[r_idx, 0, :n_sec].copy_(self.columns[r].seals.stabs[:n_sec])
            self.bus.seals_and_stabs_5x3x16[r_idx, 1, :n_sec].copy_(self.columns[r].seals.charge[:n_sec])
            self.bus.seals_and_stabs_5x3x16[r_idx, 2, :n_sec].copy_(self.columns[r].seals.mask[:n_sec])

        apex_idx = next((i for i, r in enumerate(REGION_ROLES) if r in real_iplv), 0)
        apex_role = REGION_ROLES[apex_idx]
        primary = res[apex_role]
        fcz_kin = real_kin.get("FCz", eeg_out["default_kin"])

        raw_fcz_wave = (
            res["FCz"]["sagitta_16x768"] * 0.55 +
            res["FCz"]["z_on_16x768"] * 0.25 +
            res["FCz"]["tree_16x768"] * 0.20
        )
        fcz_act_16x768, fcz_pers = self.columns["FCz"].action_integrator.step_gpu(raw_fcz_wave, fcz_kin[3])

        max_surp = torch.stack([res[r]["surprise"] for r in REGION_ROLES]).max()
        fpz_tension = torch.clamp(max_surp * 0.55 + torch.abs(res["Fpz"]["temp_asym_scalar"]) * 0.45, 0.0, 1.0)
        active_drive = torch.clamp(
            fcz_pers.squeeze(0) * 0.55 +
            torch.clamp(max_surp, max=1.0) * 0.25 +
            primary["gamma_gate"] * 0.20,
            0.0, 1.0
        )
        dyn_str = torch.clamp(
            self.strength_low + (self.strength_high - self.strength_low) * active_drive,
            self.strength_low, 0.95
        )

        m = self.bus.amodal_manifold_77x768
        m.zero_()
        m[1:5].copy_(res["Fpz"]["tree_16x768"][:4])
        m[5:21].copy_(res["AFz"]["tree_16x768"])
        m[21:37].copy_(res["F3"]["z_on_16x768"] * 0.65 + res["F3"]["tree_16x768"] * 0.35)
        m[37:53].copy_(res["F4"]["z_off_16x768"] * 0.45 + res["F4"]["temp_asym_16x768"] * 0.30 + res["F4"]["tree_16x768"] * 0.25)
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

        meta[0]  = th_hz_t
        meta[1]  = float(n_sec)
        meta[2]  = float(cur_sec)
        meta[3]  = dl_phase_t
        meta[4]  = 1.0 if is_tick else 0.0
        meta[5]  = dyn_str
        meta[6]  = fcz_pers.squeeze(0)
        meta[7]  = max_surp
        meta[8]  = primary["loss"]
        meta[9]  = primary["x_local"]
        meta[10] = primary["y_global"]
        meta[11] = primary["kuramoto_r"]
        meta[12:16].copy_(fcz_kin)
        meta[16] = fpz_tension
        meta[17] = float(eeg_out["num_live"])
        meta[18] = float(apex_idx)
        meta[19] = dl_hz_t
        meta[20] = th_phase_t
        meta[26] = primary["gamma_gate"]
        meta[27] = MONTY_MODE_MAP[self.monty_mode]
        meta[28] = float(eeg_out.get("num_unique_users", 1))

        return self.bus, True

    @torch.no_grad()
    def step(self, dt: float) -> tuple[HardwareVRAMBus, bool]:
        if not self.is_host:
            if self.bus.try_attach_to_existing_vram():
                return self.bus, True

        self._ensure_host_initialized()
        ctx = torch.cuda.stream(self.high_pri_stream) if self.high_pri_stream is not None else torch.no_grad()
        with ctx:
            eeg_out = self.spectral_gpu.pull_and_compute_gpu(self.router)
            return self._update_bus_from_eeg_dict(eeg_out, dt)

    @torch.no_grad()
    def step_from_raw_eeg_gpu(self, raw_eeg_16x256: torch.Tensor, dt: float = 0.20, sps: float = 250.0,
                              role: str = "AFz") -> HardwareVRAMBus:
        self._ensure_host_initialized()
        sub_buf = raw_eeg_16x256.unsqueeze(0)
        fs_vec = torch.tensor([[[sps]]], dtype=torch.float32, device=DEVICE)
        eeg_out = self.spectral_gpu.compute_from_buffer_gpu(sub_buf, fs_vec, [role])
        self._update_bus_from_eeg_dict(eeg_out, dt)
        return self.bus

    def stop(self):
        if self.bus.is_owner:
            try:
                os.remove(IPC_HANDLE_FILE)
            except OSError:
                pass
