#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ISOLATED MONTY CORTICAL COLUMNS (neuro_monty_columns.py)
===================================================================================
- Изолированный модуль колонок Монти (GPUSealEngine, AmodalActionIntegratorGPU,
  AmodalMontyColumnGPU).
- При monty_mode == "off" (--no-monty) колонки полностью пропускают бинаризацию SDR
  и матрицы W_out, передавая 100% чистую полноразмерную рабочую память.
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE
from neuro_vram_bus import (
    MAX_SLOTS,
    NUM_BANDS,
    BAND_DELTA,
    BAND_THETA,
    BAND_BETA,
    BAND_GAMMA,
    REGION_TAU_SEC,
)


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
    def __init__(self, sensitivity: float = 0.11, intent_gain: float = 2.4):
        super().__init__()
        self.sensitivity = sensitivity
        self.intent_gain = intent_gain
        self.register_buffer("persistence", torch.zeros(1, dtype=torch.float32, device=DEVICE))
        self.register_buffer("last_chord_768", torch.zeros(768, dtype=torch.float32, device=DEVICE))
        self.register_buffer("state_16x768", torch.zeros((16, 768), dtype=torch.float32, device=DEVICE))

    @torch.no_grad()
    def step_gpu(self, raw_wave_16x768: torch.Tensor, temp_ry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = raw_wave_16x768 - raw_wave_16x768.mean(dim=0, keepdim=True)
        normed = F.normalize(centered + 1e-6, p=2, dim=-1)
        gram = torch.matmul(normed, normed.T)
        projected = F.normalize(torch.matmul(gram, raw_wave_16x768) + 1e-6, p=2, dim=-1) * torch.norm(raw_wave_16x768, dim=-1, keepdim=True)

        chord = projected[-1] - projected[0]
        mag = torch.norm(chord)
        last_mag = torch.norm(self.last_chord_768)
        cos_th = torch.sum(chord * self.last_chord_768) / (mag * last_mag + 1e-6)

        grow = (mag > 0.015) & (last_mag > 1e-4) & (cos_th > 0.60)
        new_pers = torch.where(
            grow,
            torch.clamp(self.persistence + 0.09, max=1.0),
            self.persistence * 0.92
        )
        self.persistence.copy_(new_pers)
        self.last_chord_768.copy_(chord)

        smooth = 0.88 - (self.sensitivity * 0.1)
        gain = self.sensitivity * self.intent_gain
        ry_clamped = torch.clamp(temp_ry * 0.5, -0.4, 0.9)
        active_boost = (1.0 + self.persistence * 5.0) * (1.0 + ry_clamped)

        step_delta = projected * (gain * active_boost * (1.0 - smooth))
        self.state_16x768.mul_(smooth).add_(step_delta)

        s_norm = torch.norm(self.state_16x768, dim=-1, keepdim=True)
        self.state_16x768.copy_(
            torch.where(s_norm > 8.0, (self.state_16x768 / (s_norm + 1e-6)) * 8.0, self.state_16x768)
        )
        return self.state_16x768, self.persistence


class AmodalMontyColumnGPU(nn.Module):
    def __init__(self, role: str, geom: dict, num_columns: int = 4096, k_active: int = 80, num_slots: int = 32):
        super().__init__()
        self.role = role
        self.geom = geom
        self.n_pairs = geom["n_pairs"]
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

        self.register_buffer("W_out", torch.zeros((num_slots * self.n_pairs, num_columns), device=DEVICE))
        self.register_buffer("W_topdown_sdr", torch.eye(num_columns, device=DEVICE) * 0.4)
        self.register_buffer("l23_pooled_sdr", torch.zeros(num_columns, device=DEVICE))
        self.register_buffer("upward_sdr_anomaly", torch.zeros(num_columns, device=DEVICE))
        self.register_buffer("predicted_32xP", torch.zeros((num_slots, self.n_pairs), device=DEVICE))
        self.register_buffer("anomaly_32xP", torch.zeros((num_slots, self.n_pairs), device=DEVICE))
        self.register_buffer("baseline_loss", torch.tensor(0.020, dtype=torch.float32, device=DEVICE))

        sdr_proj = torch.randn((num_columns, 768), generator=gen, device=DEVICE) / math.sqrt(k_active)
        self.register_buffer("sdr_to_768", sdr_proj)
        self.action_integrator = AmodalActionIntegratorGPU()
        self.seals = GPUSealEngine(MAX_SLOTS)
        self.register_buffer("delta_cycle_buf", torch.zeros((2, MAX_SLOTS, 768), device=DEVICE))
        self.register_buffer("virtual_iplv", torch.zeros((32, self.n_pairs), device=DEVICE))
        self.register_buffer("virtual_rip", torch.zeros((32, self.n_pairs), device=DEVICE))
        self.register_buffer("virtual_5x32x120", torch.zeros((NUM_BANDS, 32, self.n_pairs), device=DEVICE))
        self.buf_ptr = 0

    @torch.no_grad()
    def synthesize_virtual_gpu(self, master_iplv: torch.Tensor, master_rip: torch.Tensor,
                               master_5x32x120: torch.Tensor, bleed_scale: float = 0.35,
                               parent_topdown_32xP: Optional[torch.Tensor] = None,
                               child_anomaly_32xP: Optional[torch.Tensor] = None,
                               use_monty_operator: bool = True):
        if self.role == "Fpz":
            mask = self.geom["macro_mask"]
            tgt = master_iplv * mask * (0.85 * bleed_scale)
            alpha = 0.08
        elif self.role == "F3":
            mask = self.geom["left_mask"]
            tgt = torch.relu(master_iplv) * mask * (1.2 * bleed_scale)
            alpha = 0.22
        elif self.role == "F4":
            mask = self.geom["right_mask"]
            tgt = -torch.relu(-master_iplv) * mask * (1.2 * bleed_scale)
            alpha = 0.22
        elif self.role == "FCz":
            mask = self.geom["micro_mask"]
            fast_diff = torch.roll(master_iplv, shifts=-1, dims=0) - master_iplv
            tgt = (fast_diff * 0.6 + master_iplv * 0.4) * mask * (1.4 * bleed_scale)
            alpha = 0.45
        else:
            mask = self.geom["meso_mask"] + self.geom["macro_mask"] * 0.5
            tgt = master_iplv * (1.0 * bleed_scale)
            alpha = 0.25

        if use_monty_operator:
            if parent_topdown_32xP is not None:
                tgt = tgt * 0.65 + (parent_topdown_32xP * mask) * 0.35
            if child_anomaly_32xP is not None:
                tgt = tgt + (child_anomaly_32xP * mask) * 0.40

        self.virtual_iplv.mul_(1.0 - alpha).add_(tgt, alpha=alpha)
        self.virtual_rip.mul_(1.0 - alpha).add_(master_rip * bleed_scale, alpha=alpha)
        self.virtual_5x32x120.mul_(1.0 - alpha).add_(master_5x32x120 * mask.unsqueeze(0) * bleed_scale, alpha=alpha)
        if use_monty_operator:
            self.virtual_5x32x120[BAND_GAMMA].copy_(self.virtual_iplv)
        return self.virtual_iplv, self.virtual_rip, self.virtual_5x32x120

    @torch.no_grad()
    def forward_gpu(self, raw_iplv_32xP: torch.Tensor, raw_rip_32xP: torch.Tensor,
                    bands_5x32x120: torch.Tensor,
                    n_sec: int, cur_sec: int, is_tick: bool, dt: float,
                    top_down_parent_sdr: Optional[torch.Tensor] = None,
                    bottom_up_child_anomaly: Optional[torch.Tensor] = None,
                    enable_monty_sdr: bool = True) -> dict:
        proj_T = self.geom["proj_Px768"]
        raw_delta_32xP = bands_5x32x120[BAND_DELTA]
        raw_theta_32xP = bands_5x32x120[BAND_THETA]
        raw_beta_32xP  = bands_5x32x120[BAND_BETA]

        if self.role == "Fpz":
            nested_drive_32xP = raw_delta_32xP * 0.60 + raw_theta_32xP * 0.25 + raw_iplv_32xP * 0.15
        elif self.role == "AFz":
            nested_drive_32xP = raw_delta_32xP * 0.40 + raw_theta_32xP * 0.35 + raw_iplv_32xP * 0.25
        elif self.role in ("F3", "F4"):
            nested_drive_32xP = raw_theta_32xP * 0.40 + raw_beta_32xP * 0.25 + raw_iplv_32xP * 0.35
        else:
            nested_drive_32xP = raw_iplv_32xP * 0.65 + raw_rip_32xP * 0.25 + raw_theta_32xP * 0.10

        x_pos_768 = torch.matmul(torch.relu(nested_drive_32xP), proj_T)
        x_neg_768 = torch.matmul(torch.relu(-nested_drive_32xP), proj_T)
        z_32x768 = x_pos_768 - x_neg_768

        if enable_monty_sdr:
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
                self.W_topdown_sdr.mul_(0.9998).add_(
                    torch.outer(sdr_err, top_down_parent_sdr) * (0.03 / p_energy)
                )
            else:
                self.upward_sdr_anomaly.copy_(sdr_inst * 0.15)

            sdr_anom_norm = torch.norm(self.upward_sdr_anomaly) / math.sqrt(self.k_active)

            real_flat = nested_drive_32xP.reshape(-1)
            pred_flat = torch.mv(self.W_out, self.l23_pooled_sdr)
            error_flat = real_flat - pred_flat
            self.predicted_32xP.copy_(pred_flat.view(self.num_slots, self.n_pairs))
            self.anomaly_32xP.copy_(error_flat.view(self.num_slots, self.n_pairs))
            loss_t = torch.mean(error_flat ** 2)

            surprise_t = torch.clamp(
                torch.relu((loss_t - self.baseline_loss) / (self.baseline_loss + 1e-5)) + sdr_anom_norm * 0.3,
                0.0, 2.5
            )
            self.baseline_loss.mul_(0.985).add_(loss_t * 0.015)

            state_energy = torch.sum(self.l23_pooled_sdr ** 2) + 1e-4
            eta_t = torch.clamp(0.08 * (1.0 + surprise_t * 0.5), max=0.25) / state_energy
            self.W_out.mul_(0.9995).add_(torch.outer(error_flat, self.l23_pooled_sdr) * eta_t)

            sdr_7x4096 = F.adaptive_avg_pool1d(sdr_seq.T.unsqueeze(0), 7).squeeze(0).T
            sdr_tokens_7x768 = torch.matmul(sdr_7x4096, self.sdr_to_768)
        else:
            self.l23_pooled_sdr.zero_()
            self.upward_sdr_anomaly.zero_()
            self.predicted_32xP.copy_(nested_drive_32xP)
            diff_wm = nested_drive_32xP - torch.roll(nested_drive_32xP, shifts=1, dims=0)
            self.anomaly_32xP.copy_(diff_wm)
            loss_t = torch.mean(diff_wm ** 2)
            surprise_t = torch.clamp((loss_t - self.baseline_loss) / (self.baseline_loss + 1e-5), 0.0, 2.5)
            self.baseline_loss.mul_(0.985).add_(loss_t * 0.015)
            sdr_tokens_7x768 = F.adaptive_avg_pool1d(z_32x768.T.unsqueeze(0), 7).squeeze(0).T

        z_on_16x768  = F.adaptive_avg_pool1d(x_pos_768.T.unsqueeze(0), 16).squeeze(0).T
        z_off_16x768 = F.adaptive_avg_pool1d(x_neg_768.T.unsqueeze(0), 16).squeeze(0).T

        chord_768 = z_32x768[-1] - z_32x768[0]
        chord_unit = F.normalize(chord_768.unsqueeze(0) + 1e-6, p=2, dim=-1)
        rel_z = z_32x768 - z_32x768[0:1]
        sagitta_32x768 = rel_z - torch.sum(rel_z * chord_unit, dim=-1, keepdim=True) * chord_unit
        sagitta_16x768 = F.adaptive_avg_pool1d(sagitta_32x768.T.unsqueeze(0), 16).squeeze(0).T
        temp_asym_16x768 = z_32x768[16:32] - z_32x768[0:16]

        dynamic_wm    = F.adaptive_avg_pool1d(nested_drive_32xP.T.unsqueeze(0), n_sec).squeeze(0).T
        dynamic_theta = F.adaptive_avg_pool1d(raw_theta_32xP.T.unsqueeze(0), n_sec).squeeze(0).T
        dynamic_delta = F.adaptive_avg_pool1d(raw_delta_32xP.T.unsqueeze(0), n_sec).squeeze(0).T
        z_slots = torch.matmul(dynamic_wm, proj_T)

        kuramoto_r = torch.clamp(torch.norm(dynamic_wm.mean(dim=0)) / 10.0, 0.0, 1.0)

        cur_s = cur_sec % n_sec
        x_local  = torch.clamp(torch.sum((dynamic_theta[cur_s] * 0.65 + dynamic_wm[cur_s] * 0.35) * self.geom["dx"]) / 30.0, -1.0, 1.0)
        y_global = torch.clamp(torch.sum((dynamic_delta[cur_s] * 0.65 + dynamic_wm[cur_s] * 0.35) * self.geom["dy"]) / 30.0, -1.0, 1.0)

        slot_x = torch.matmul(dynamic_theta * 0.5 + dynamic_wm * 0.5, self.geom["dx"])
        slot_y = torch.matmul(dynamic_delta * 0.5 + dynamic_wm * 0.5, self.geom["dy"])
        slot_norm_coord = (torch.atan2(slot_y, slot_x) + math.pi) / (2.0 * math.pi)

        npast = torch.norm(raw_iplv_32xP[0])
        nfut = torch.norm(raw_iplv_32xP[-1])
        temp_asym_scalar = (nfut - npast) / (nfut + npast + 1e-6)

        dyn_rip_768 = torch.matmul(
            F.adaptive_avg_pool1d(raw_rip_32xP.T.unsqueeze(0), n_sec).squeeze(0).T,
            proj_T
        )
        dyn_delta_768 = torch.matmul(dynamic_delta, proj_T)

        diff_rip   = dyn_rip_768.unsqueeze(1) - dyn_rip_768.unsqueeze(0)
        diff_delta = dyn_delta_768.unsqueeze(1) - dyn_delta_768.unsqueeze(0)
        diff_joint = diff_rip * 0.55 + diff_delta * 0.45

        c_flow_mat = torch.norm(diff_joint, dim=-1)
        net_leads = torch.sum(diff_joint.sum(dim=1) - diff_joint.sum(dim=0), dim=-1)
        order_gpu = torch.argsort(net_leads, descending=True)

        ordered_slots = z_slots[order_gpu]
        root_vec = ordered_slots[0:1]
        root_leads = c_flow_mat[order_gpu[0], order_gpu]

        dot = torch.sum(ordered_slots * root_vec, dim=-1, keepdim=True) / (torch.sum(root_vec ** 2) + 1e-7)
        ortho = ordered_slots - dot * root_vec
        ortho_norm = F.normalize(ortho + 1e-6, p=2, dim=-1) * torch.norm(ordered_slots, dim=-1, keepdim=True)

        is_supset = (root_leads >= 0.03).unsqueeze(-1).float()
        composed_slots = root_vec + is_supset * ortho_norm * 0.35 + (1.0 - is_supset) * ordered_slots * 0.25
        tree_16x768 = F.adaptive_avg_pool1d(composed_slots.T.unsqueeze(0), 16).squeeze(0).T

        beta_energy  = torch.mean(torch.abs(raw_beta_32xP))
        gamma_energy = torch.mean(torch.abs(raw_iplv_32xP)) + torch.mean(torch.abs(raw_rip_32xP)) * 0.5
        gamma_gate   = gamma_energy / (gamma_energy + beta_energy + 1e-5)

        oldest = (self.buf_ptr + 1) % 2
        drift_cos = torch.sum(
            F.normalize(z_slots + 1e-6, p=2, dim=-1) * F.normalize(self.delta_cycle_buf[oldest, :n_sec] + 1e-6, p=2, dim=-1),
            dim=-1
        )
        raw_stab = torch.clamp((drift_cos * 0.5 + 0.5) * (0.2 + 0.8 * kuramoto_r), 0.0, 1.0)
        stab_alpha = 0.35 if is_tick else min(0.15, dt * 4.0)
        self.seals.stabs[:n_sec].mul_(1.0 - stab_alpha).add_(raw_stab * stab_alpha)
        self.seals.step_gpu(n_sec, c_flow_mat)

        if is_tick:
            self.delta_cycle_buf[self.buf_ptr, :n_sec].copy_(z_slots)
            self.buf_ptr = oldest

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
            "gamma_gate": gamma_gate,
            "loss": loss_t,
            "surprise": surprise_t,
        }
