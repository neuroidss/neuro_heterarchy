#!/usr/bin/env python3
"""
neuro_solver_stepper.py — Полноранговый латентный интегратор 128D с живым морфингом от WM_RADIUS
================================================================================================
- 100% сохранена рабочая схема денойзинга и стабилизации, дающая четкие горы, кристаллы и лабиринт.
- Исправлен баг TypeError: math.sin(mix_ang) при ненулевой стохастике.
- Устранено моргание яркости на стыке блоков при большом wm_radius (0.84) за счет сохранения
  единичной дисперсии (norm_scale = hypot(w, 1-w)) при подмешивании шума в опорный кадр контекста.
- Добавлено реалтайм-обновление семантического поворота (live_pos_emb_fn) прямо внутри шагов Эйлера!
"""

from __future__ import annotations
import math
from typing import Any, Callable
import torch
import torch.nn.functional as F

from neuro_ltx_backend import LTXModelBackend, rescale_noise_cfg


@torch.no_grad()
def tone_map_latents_ltx(latents: torch.Tensor, compression: float = 0.12) -> torch.Tensor:
    if compression <= 0.0:
        return latents
    scale_factor = float(compression) * 0.75
    abs_lat = torch.abs(latents)
    sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_lat - 1.0))
    scales = 1.0 - 0.8 * scale_factor * sigmoid_term
    return latents * scales


@torch.no_grad()
def stabilize_sliding_latents(
    curr_latents: torch.Tensor,
    prev_tail_latents: torch.Tensor | None,
    factor: float = 0.10,
) -> torch.Tensor:
    """
    Стабилизирует дисперсию только относительно ХВОСТА ПРЕДЫДУЩЕГО ОКНА (а не первого кадра сессии!),
    сохраняя 100% свободу смены цветов и геометрии при переходе к новым стадиям и концептам.
    """
    eps = 1e-6
    mu_curr = curr_latents.mean(dim=(2, 3, 4), keepdim=True)
    sigma_curr = curr_latents.std(dim=(2, 3, 4), keepdim=True).clamp(min=eps)

    if prev_tail_latents is not None and factor > 0.0:
        mu_ref = prev_tail_latents.mean(dim=(2, 3, 4), keepdim=True).to(curr_latents.device, curr_latents.dtype)
        sigma_ref = prev_tail_latents.std(dim=(2, 3, 4), keepdim=True).to(curr_latents.device, curr_latents.dtype).clamp(min=eps)
        target_mu = (1.0 - factor) * mu_curr + factor * (mu_ref * 0.85)
        target_sigma = torch.clamp((1.0 - factor) * sigma_curr + factor * sigma_ref, min=0.78, max=1.12)
        curr_latents = ((curr_latents - mu_curr) / sigma_curr) * target_sigma + target_mu

    return tone_map_latents_ltx(curr_latents, compression=0.12)


@torch.no_grad()
def imprint_allegiance_spin(
    origin_anchor_latents: torch.Tensor,
    base_seed: int = 42,
) -> torch.Tensor:
    anchored = origin_anchor_latents.clone()
    spin_gen = torch.Generator(device=anchored.device).manual_seed(base_seed + 777)
    raw_mat = torch.randn((32, 12), generator=spin_gen, device=anchored.device, dtype=torch.float32)
    q_orth, _ = torch.linalg.qr(raw_mat)
    omega_a = F.normalize(q_orth[:, 0], p=2, dim=0).view(1, 32, 1, 1, 1)
    anchored[:, 96:, :, :, :] += 0.06 * omega_a
    return anchored


class LatentFlowWindowStepper:
    def __init__(
        self,
        backend: LTXModelBackend,
        referee: Any = None,
        steps: int = 6,
        cfg_scale: float = 3.2,
        guidance_rescale: float = 0.7,
    ):
        self.backend = backend
        self.referee = referee
        self.steps = max(2, int(steps))
        self.cfg_scale = float(cfg_scale)
        self.guidance_rescale = float(guidance_rescale)
        self.last_velocity_5d: torch.Tensor | None = None
        self._s3_bases_cache: dict[tuple, list[torch.Tensor]] = {}
        self._wm_accum_phase_field: torch.Tensor | None = None

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
        step_new_frames: int = 2,
    ) -> torch.Tensor:
        key = (shape, base_seed, device)
        n_vox = int(shape[1] * shape[2] * shape[3] * shape[4])

        if key not in self._s3_bases_cache:
            self._s3_bases_cache.clear()
            g1 = torch.Generator(device=device).manual_seed(101 + base_seed)
            g2 = torch.Generator(device=device).manual_seed(202 + base_seed)
            g3 = torch.Generator(device=device).manual_seed(303 + base_seed)
            g4 = torch.Generator(device=device).manual_seed(404 + base_seed)
            self._s3_bases_cache[key] = [
                torch.randn(shape, generator=g1, device=device, dtype=torch.float32),
                torch.randn(shape, generator=g2, device=device, dtype=torch.float32),
                torch.randn(shape, generator=g3, device=device, dtype=torch.float32),
                torch.randn(shape, generator=g4, device=device, dtype=torch.float32),
            ]
        else:
            shift_t = max(1, int(step_new_frames))
            for idx_b in range(4):
                self._s3_bases_cache[key][idx_b] = torch.roll(
                    self._s3_bases_cache[key][idx_b], shifts=-shift_t, dims=2
                )

        b1, b2, b3, b4 = self._s3_bases_cache[key]

        if wm_19200_vec is not None and wm_19200_vec.numel() > 0:
            raw_wm = wm_19200_vec.to(device=device, dtype=torch.float32).reshape(-1)
        else:
            raw_wm = wm_probe_emb.to(device=device, dtype=torch.float32).reshape(-1)

        wm_std = raw_wm.std(unbiased=False).clamp(min=1e-5)
        wm_whitened = (raw_wm - raw_wm.mean()) / wm_std

        reps = (n_vox // wm_whitened.numel()) + 1
        wm_vox = wm_whitened.repeat(reps)[:n_vox].view(shape)

        rad = float(max(0.02, min(1.0, wm_radius)))
        if self._wm_accum_phase_field is None or self._wm_accum_phase_field.shape != shape:
            self._wm_accum_phase_field = wm_vox * (rad * 1.1)
        else:
            self._wm_accum_phase_field = torch.remainder(
                torch.roll(self._wm_accum_phase_field, shifts=-max(1, int(step_new_frames)), dims=2)
                + wm_vox * (rad * 1.25),
                2.0 * math.pi,
            )

        scale_ang = math.pi * 1.35 * rad
        phi_a = self._wm_accum_phase_field + wm_vox * scale_ang
        phi_b = torch.roll(self._wm_accum_phase_field, shifts=32, dims=1) - torch.roll(wm_vox, shifts=16, dims=1) * scale_ang
        phi_c = torch.roll(self._wm_accum_phase_field, shifts=64, dims=1) * 0.7 + torch.roll(wm_vox, shifts=48, dims=1) * (scale_ang * 0.6)

        g12 = torch.cos(phi_a) * b1 + torch.sin(phi_a) * b2
        g34 = torch.cos(phi_b) * b3 + torch.sin(phi_b) * b4
        g_wm = torch.cos(phi_c) * g12 + torch.sin(phi_b) * g34 if False else (torch.cos(phi_c) * g12 + torch.sin(phi_c) * g34)

        stoch = float(max(0.0, min(1.0, stochastic)))
        if stoch <= 0.001:
            return g_wm

        g_rand = torch.randn(shape, generator=gen_rand, device=device, dtype=torch.float32)
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

        timesteps = self.backend.set_scheduler_timesteps(self.steps, video_seq_len)

        n_cond = 0
        if is_continuation_block and anchor_latent_0 is not None:
            n_cond = min(anchor_latent_0.shape[2], total_lat_frames - 1)
        step_new_frames = max(1, total_lat_frames - n_cond)

        shape = (1, in_channels, total_lat_frames, h_lat, w_lat)
        noise = self._build_wm_s3_noise(
            shape=shape,
            base_seed=base_seed,
            stochastic=stochastic,
            wm_radius=wm_radius,
            wm_19200_vec=wm_19200_vec,
            wm_probe_emb=pos_emb,
            gen_rand=gen,
            device=device,
            step_new_frames=step_new_frames,
        )

        latents_5d = noise.clone()
        cond_mask_5d = torch.zeros((1, 1, total_lat_frames, h_lat, w_lat), device=device, dtype=torch.float32)

        if is_continuation_block and anchor_latent_0 is not None and n_cond > 0:
            anchor_prefix_5d = anchor_latent_0[:, :, -n_cond:].clone()
            rad_clamped = float(max(0.02, min(0.85, wm_radius)))
            tail_lock = float(max(0.45, min(0.88, 0.88 - 0.42 * rad_clamped)))
            if n_cond == 1:
                cond_mask_5d[:, :, 0:1] = tail_lock
            else:
                for c_i in range(n_cond - 1):
                    cond_mask_5d[:, :, c_i : c_i + 1] = min(0.92, tail_lock + 0.12)
                cond_mask_5d[:, :, n_cond - 1 : n_cond] = tail_lock

            # ИСПРАВЛЕНИЕ МОРГАНИЯ ЯРКОСТИ ПРИ БОЛЬШОМ РАДИУСЕ (0.84):
            # Делим линейную смесь на sqrt(w^2 + (1-w)^2), чтобы дисперсия опорного кадра
            # всегда оставалась равной 1.0 (при w=0.52 без этого делителя амплитуда падала до 0.70,
            # вызывая вспышку освещения на каждом блоке!):
            w_m = cond_mask_5d[:, :, :n_cond]
            norm_preserve = torch.sqrt(w_m ** 2 + (1.0 - w_m) ** 2).clamp(min=1e-5)
            latents_5d[:, :, :n_cond] = (
                anchor_prefix_5d * w_m + noise[:, :, :n_cond] * (1.0 - w_m)
            ) / norm_preserve

        latents_packed = pipe._pack_latents(latents_5d, 1, 1).to(torch.float32)
        cond_mask_packed = pipe._pack_latents(cond_mask_5d, 1, 1).squeeze(-1).to(device)
        init_latents_packed = latents_packed.clone()

        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        use_cfg_overall = self.cfg_scale > 1.05
        cfg_cutoff_step = max(2, int(math.ceil(len(timesteps) * 0.75)))

        curr_pos_emb = pos_emb
        batch2_prompt_mask = torch.cat([neg_mask, pos_mask], dim=0) if use_cfg_overall else pos_mask
        cond_mask_batch2 = torch.cat([cond_mask_packed, cond_mask_packed], dim=0) if use_cfg_overall else cond_mask_packed

        noise_pred_final = None
        n_total_steps = len(timesteps)

        for i, t in enumerate(timesteps):
            # РЕАЛТАЙМ-ОПРОС РАБОЧЕЙ ПАМЯТИ НА КАЖДОМ ШАГЕ ДЕНОЙЗИНГА:
            # На лету пересчитываем геодезический поворот ТОГО ЖЕ САМОГО базового промпта (без смешивания разных слов!)
            if live_pos_emb_fn is not None and i > 0:
                fresh_pos = live_pos_emb_fn(i, n_total_steps)
                if fresh_pos is not None and fresh_pos.shape == curr_pos_emb.shape:
                    curr_pos_emb = fresh_pos

            do_cfg_step = use_cfg_overall and (i < cfg_cutoff_step)
            if do_cfg_step:
                latent_input = torch.cat([latents_packed, latents_packed], dim=0)
                t_expand = t.expand(2).unsqueeze(-1).float()
                t_per_token = torch.min(t_expand, (1.0 - cond_mask_batch2) * 1000.0)
                p_emb = torch.cat([neg_emb, curr_pos_emb], dim=0)
                p_msk = batch2_prompt_mask
            else:
                latent_input = latents_packed
                t_expand = t.expand(1).unsqueeze(-1).float()
                t_per_token = torch.min(t_expand, (1.0 - cond_mask_packed) * 1000.0)
                p_emb = curr_pos_emb
                p_msk = pos_mask

            noise_pred_raw = self.backend.transformer(
                hidden_states=latent_input.to(dtype),
                encoder_hidden_states=p_emb,
                timestep=t_per_token,
                encoder_attention_mask=p_msk,
                num_frames=total_lat_frames,
                height=h_lat,
                width=w_lat,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0].float()

            if do_cfg_step:
                noise_uncond, noise_text = noise_pred_raw.chunk(2)
                noise_cfg = noise_uncond + self.cfg_scale * (noise_text - noise_uncond)
                noise_pred_final = rescale_noise_cfg(noise_cfg, noise_text, guidance_rescale=self.guidance_rescale)
            else:
                noise_pred_final = noise_pred_raw

            denoised_packed = self.backend.scheduler.step(
                noise_pred_final, t, latents_packed, return_dict=False
            )[0]

            if is_continuation_block and n_cond > 0:
                t_norm = float(t.item()) / 1000.0
                tokens_to_denoise = (t_norm - 1e-6 < (1.0 - cond_mask_packed)).unsqueeze(-1)
                latents_packed = torch.where(tokens_to_denoise, denoised_packed, init_latents_packed)
            else:
                latents_packed = denoised_packed

        if noise_pred_final is not None:
            self.last_velocity_5d = pipe._unpack_latents(
                noise_pred_final, total_lat_frames, h_lat, w_lat, 1, 1
            ).float()

        unpacked_z0 = pipe._unpack_latents(latents_packed, total_lat_frames, h_lat, w_lat, 1, 1).float()
        unpacked_z0 = stabilize_sliding_latents(unpacked_z0, anchor_latent_0, factor=0.10)

        return unpacked_z0, 1.0
