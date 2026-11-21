#!/usr/bin/env python3
"""
neuro_solver_stepper.py — 3D-DiT I2V/V2V Stepper (0.9.8-Distilled + Опциональный мягкий CFG + t^2 Jitter)
"""

from __future__ import annotations
import math
from typing import Any, Callable
import torch
import torch.nn.functional as F

from neuro_ltx_backend import LTXModelBackend, rescale_noise_cfg


@torch.no_grad()
def adain_stabilize_latents(
    curr_latents: torch.Tensor,
    ref_latents: torch.Tensor | None,
    factor: float = 0.08,
) -> torch.Tensor:
    if ref_latents is None or factor <= 0.0:
        return curr_latents

    eps = 1e-6
    mu_curr = curr_latents.mean(dim=(2, 3, 4), keepdim=True)
    std_curr = curr_latents.std(dim=(2, 3, 4), keepdim=True).clamp(min=eps)

    mu_ref = ref_latents.mean(dim=(2, 3, 4), keepdim=True).to(
        device=curr_latents.device, dtype=curr_latents.dtype
    )
    std_ref = (
        ref_latents.std(dim=(2, 3, 4), keepdim=True)
        .to(device=curr_latents.device, dtype=curr_latents.dtype)
        .clamp(min=eps)
    )

    mu_blend = (1.0 - float(factor)) * mu_curr + float(factor) * mu_ref
    std_blend = (1.0 - float(factor)) * std_curr + float(factor) * std_ref

    return ((curr_latents - mu_curr) / std_curr) * std_blend + mu_blend


@torch.no_grad()
def imprint_allegiance_spin(
    origin_anchor_latents: torch.Tensor,
    base_seed: int = 42,
) -> torch.Tensor:
    anchored = origin_anchor_latents.clone()
    spin_gen = torch.Generator(device=anchored.device).manual_seed(base_seed + 777)
    raw_mat = torch.randn(
        (32, 12), generator=spin_gen, device=anchored.device, dtype=torch.float32
    )
    q_orth, _ = torch.linalg.qr(raw_mat)
    omega_a = F.normalize(q_orth[:, 0], p=2, dim=0).view(1, 32, 1, 1, 1)
    anchored[:, 96:, :, :, :] += 0.02 * omega_a
    return anchored


class LatentFlowWindowStepper:
    def __init__(
        self,
        backend: LTXModelBackend,
        referee: Any = None,
        steps: int = 8,
        cfg_scale: float = 1.0,
        guidance_rescale: float = 0.7,
        image_cond_noise_scale: float = 0.15,
    ):
        self.backend = backend
        self.referee = referee
        self.steps = max(4, int(steps))
        self.cfg_scale = float(cfg_scale)
        self.guidance_rescale = float(guidance_rescale)
        self.image_cond_noise_scale = float(image_cond_noise_scale)
        self.last_velocity_5d: torch.Tensor | None = None
        self._accum_phase_128xF: torch.Tensor | None = None
        self._window_step_idx: int = 0

    @torch.no_grad()
    def _build_wm_s3_noise(
        self,
        shape: tuple[int, int, int, int, int],
        base_seed: int,
        stochastic: float,
        wm_radius: float,
        wm_19200_vec: torch.Tensor | None,
        wm_probe_emb: torch.Tensor,
        gen_rand: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        _, C, F_lat, H_lat, W_lat = shape
        self._window_step_idx += 1
        win_seed = int(base_seed) + self._window_step_idx * 997

        g1 = torch.Generator(device=device).manual_seed(win_seed + 101)
        g2 = torch.Generator(device=device).manual_seed(win_seed + 202)
        g3 = torch.Generator(device=device).manual_seed(win_seed + 303)
        g4 = torch.Generator(device=device).manual_seed(win_seed + 404)

        b1 = torch.randn(shape, generator=g1, device=device, dtype=torch.float32)
        b2 = torch.randn(shape, generator=g2, device=device, dtype=torch.float32)
        b3 = torch.randn(shape, generator=g3, device=device, dtype=torch.float32)
        b4 = torch.randn(shape, generator=g4, device=device, dtype=torch.float32)

        if wm_19200_vec is not None and wm_19200_vec.numel() > 0:
            raw_wm = wm_19200_vec.to(device=device, dtype=torch.float32).reshape(-1)
        else:
            raw_wm = wm_probe_emb.to(device=device, dtype=torch.float32).reshape(-1)

        n_needed = C * F_lat * 3
        if raw_wm.numel() < n_needed:
            reps = (n_needed // raw_wm.numel()) + 1
            raw_wm = raw_wm.repeat(reps)[:n_needed]

        wm_std = raw_wm.std(unbiased=False).clamp(min=1e-5)
        wm_whitened = (raw_wm - raw_wm.mean()) / wm_std

        wm_3xCxF = F.adaptive_avg_pool1d(
            wm_whitened.view(1, 1, -1), 3 * C * F_lat
        ).view(3, 1, C, F_lat, 1, 1)
        wm_3xCxF = torch.tanh(wm_3xCxF * 1.5)

        rad = float(max(0.02, min(1.0, wm_radius)))
        if (
            self._accum_phase_128xF is None
            or self._accum_phase_128xF.shape != (1, C, F_lat, 1, 1)
        ):
            self._accum_phase_128xF = wm_3xCxF[0] * (rad * math.pi * 0.5)
        else:
            tail_ext = self._accum_phase_128xF[:, :, -1:, :, :] + wm_3xCxF[0][:, :, -1:, :, :] * (rad * 0.65)
            shifted = torch.cat([self._accum_phase_128xF[:, :, 1:, :, :], tail_ext], dim=2)
            self._accum_phase_128xF = torch.remainder(shifted + wm_3xCxF[0] * (rad * 0.55), 2.0 * math.pi)

        scale_ang = math.pi * 0.85 * rad
        phi_a = self._accum_phase_128xF + wm_3xCxF[0] * scale_ang
        phi_b = (
            torch.roll(self._accum_phase_128xF, shifts=16, dims=1)
            - wm_3xCxF[1] * scale_ang
        )
        phi_c = torch.roll(self._accum_phase_128xF, shifts=32, dims=1) * 0.5 + wm_3xCxF[2] * (scale_ang * 0.5)

        g12 = torch.cos(phi_a) * b1 + torch.sin(phi_a) * b2
        g34 = torch.cos(phi_b) * b3 + torch.sin(phi_b) * b4
        g_wm = torch.cos(phi_c) * g12 + torch.sin(phi_c) * g34

        stoch = float(max(0.0, min(1.0, stochastic)))
        if stoch <= 0.001:
            return g_wm

        g_rand = torch.randn(
            shape, generator=gen_rand, device=device, dtype=torch.float32
        )
        mix_ang = stoch * (math.pi * 0.5)
        return math.cos(mix_ang) * g_wm + math.sin(mix_ang) * g_rand

    @torch.no_grad()
    def run_window(
        self,
        anchor_latent_0: torch.Tensor | None,
        genesis_ref_latent_5d: torch.Tensor | None,
        walk_in_emb: torch.Tensor,
        walk_in_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_emb: torch.Tensor,
        neg_mask: torch.Tensor,
        entity_probes: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        passive_emb: torch.Tensor | None = None,
        passive_mask: torch.Tensor | None = None,
        total_lat_frames: int = 3,
        h_lat: int = 8,
        w_lat: int = 12,
        gen: torch.Generator | None = None,
        is_continuation_block: bool = False,
        stochastic: float = 0.0,
        wm_radius: float = 0.35,
        base_seed: int = 42,
        wm_19200_vec: torch.Tensor | None = None,
        live_pos_emb_fn: Callable[[int, int], torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, float]:
        pipe = self.backend.pipe
        device = self.backend.device
        dtype = self.backend.dtype

        in_channels = self.backend.transformer.config.in_channels
        video_seq_len = total_lat_frames * h_lat * w_lat

        is_distilled = bool(getattr(self.backend, "is_distilled", False))
        if is_distilled:
            eff_steps = max(4, min(12, self.steps))
        else:
            eff_steps = max(14, self.steps * 2) if (not is_continuation_block) else self.steps

        timesteps = self.backend.set_scheduler_timesteps(eff_steps, video_seq_len)

        n_cond = 0
        if is_continuation_block and anchor_latent_0 is not None:
            n_cond = min(anchor_latent_0.shape[2], total_lat_frames - 1)

        shape = (1, in_channels, total_lat_frames, h_lat, w_lat)
        noise_5d = self._build_wm_s3_noise(
            shape=shape,
            base_seed=base_seed,
            stochastic=stochastic,
            wm_radius=wm_radius,
            wm_19200_vec=wm_19200_vec,
            wm_probe_emb=pos_emb,
            gen_rand=gen,
            device=device,
        )

        latents_5d = noise_5d.clone()
        cond_mask_5d = torch.zeros(
            (1, 1, total_lat_frames, h_lat, w_lat), device=device, dtype=torch.float32
        )

        clean_anchor_prefix_5d = None
        if is_continuation_block and anchor_latent_0 is not None and n_cond > 0:
            clean_anchor_prefix_5d = anchor_latent_0[:, :, -n_cond:].clone().float()
            latents_5d[:, :, :n_cond] = clean_anchor_prefix_5d
            cond_mask_5d[:, :, :n_cond] = 1.0

        cond_mask_packed = pipe._pack_latents(cond_mask_5d, 1, 1).squeeze(-1).to(device)
        cond_mask_batch2 = torch.cat([cond_mask_packed, cond_mask_packed], dim=0)

        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        # Позволяет явно включать мягкий CFG (напр. --cfg 1.5..2.0) и на distilled,
        # сохраняя быстрый batch=1 при --cfg 1.0:
        use_cfg_overall = self.cfg_scale > 1.05
        curr_pos_emb = pos_emb
        batch2_prompt_mask = (
            torch.cat([neg_mask, pos_mask], dim=0) if use_cfg_overall else pos_mask
        )

        noise_pred_final = None
        n_total_steps = len(timesteps)

        for i, t in enumerate(timesteps):
            if live_pos_emb_fn is not None and i > 0:
                fresh_pos = live_pos_emb_fn(i, n_total_steps)
                if fresh_pos is not None and fresh_pos.shape == curr_pos_emb.shape:
                    curr_pos_emb = (curr_pos_emb.float() * 0.75 + fresh_pos.float() * 0.25).to(dtype)

            if clean_anchor_prefix_5d is not None and n_cond > 0:
                if self.image_cond_noise_scale > 0.0:
                    t_norm = float(t.item()) / 1000.0
                    cond_jit = torch.randn(
                        clean_anchor_prefix_5d.shape,
                        generator=gen,
                        device=device,
                        dtype=torch.float32,
                    )
                    latents_5d[:, :, :n_cond] = (
                        clean_anchor_prefix_5d
                        + self.image_cond_noise_scale * cond_jit * (t_norm ** 2)
                    )
                else:
                    latents_5d[:, :, :n_cond] = clean_anchor_prefix_5d

            latents_packed = pipe._pack_latents(latents_5d, 1, 1).to(torch.float32)

            if use_cfg_overall:
                latent_input = torch.cat([latents_packed, latents_packed], dim=0).to(dtype)
                t_expand = t.expand(2).unsqueeze(-1).float()
                t_per_token = t_expand * (1.0 - cond_mask_batch2)
                p_emb = torch.cat([neg_emb, curr_pos_emb], dim=0)
                p_msk = batch2_prompt_mask
            else:
                latent_input = latents_packed.to(dtype)
                t_expand = t.expand(1).unsqueeze(-1).float()
                t_per_token = t_expand * (1.0 - cond_mask_packed)
                p_emb = curr_pos_emb
                p_msk = pos_mask

            noise_pred_raw = self.backend.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=p_emb,
                timestep=t_per_token,
                encoder_attention_mask=p_msk,
                num_frames=total_lat_frames,
                height=h_lat,
                width=w_lat,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0].float()

            if use_cfg_overall:
                noise_uncond, noise_text = noise_pred_raw.chunk(2)
                noise_cfg = noise_uncond + self.cfg_scale * (noise_text - noise_uncond)
                noise_pred_final = rescale_noise_cfg(
                    noise_cfg, noise_text, guidance_rescale=self.guidance_rescale
                )
            else:
                noise_pred_final = noise_pred_raw

            noise_pred_5d = pipe._unpack_latents(
                noise_pred_final, total_lat_frames, h_lat, w_lat, 1, 1
            ).float()

            if is_continuation_block and n_cond > 0 and clean_anchor_prefix_5d is not None:
                pred_gen_5d = self.backend.scheduler.step(
                    noise_pred_5d[:, :, n_cond:],
                    t,
                    latents_5d[:, :, n_cond:],
                    return_dict=False,
                )[0].float()
                latents_5d = torch.cat([clean_anchor_prefix_5d, pred_gen_5d], dim=2)
            else:
                latents_5d = self.backend.scheduler.step(
                    noise_pred_5d,
                    t,
                    latents_5d,
                    return_dict=False,
                )[0].float()

        if noise_pred_final is not None:
            self.last_velocity_5d = pipe._unpack_latents(
                noise_pred_final, total_lat_frames, h_lat, w_lat, 1, 1
            ).float()

        unpacked_z0 = latents_5d.float()
        if is_continuation_block and clean_anchor_prefix_5d is not None:
            unpacked_z0[:, :, n_cond:] = adain_stabilize_latents(
                unpacked_z0[:, :, n_cond:], clean_anchor_prefix_5d[:, :, -1:], factor=0.08
            )

        return unpacked_z0, 1.0
