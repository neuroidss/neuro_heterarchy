#!/usr/bin/env python3
"""
neuro_solver_stepper.py — Спектрально-когерентный 3D-DiT Шагатель (LTX-Video 0.9.8-Distilled + Base)
====================================================================================================
- Поддержка живого многокадрового контекста context_latents = 1, 2, 3 по канону LTXConditionPipeline:
  хронологический префикс 0..n_cond-1 (закодированный каузальным 3D-VAE из последних 1/9/17 кадров)
  несёт реальный импульс движения, а scheduler.step идёт строго по свободным кадрам [n_cond:].
- Раздельная стохастика: `stochastic` для активных регионов/агентов и `unassigned_stochastic`
  для пустых/неактивных регионов при сохранении точного N(0, I) в каждом вокселе.
====================================================================================================
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
    factor: float = 0.0,
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
    return origin_anchor_latents


class LatentFlowWindowStepper:
    def __init__(
        self,
        backend: LTXModelBackend,
        referee: Any = None,
        steps: int = 8,
        cfg_scale: float = 1.0,
        guidance_rescale: float = 0.7,
        image_cond_noise_scale: float = 0.05,
    ):
        self.backend = backend
        self.referee = referee
        self.steps = max(4, int(steps))
        self.cfg_scale = float(cfg_scale)
        self.guidance_rescale = float(guidance_rescale)
        self.image_cond_noise_scale = float(image_cond_noise_scale)
        self.last_velocity_5d: torch.Tensor | None = None
        self.smooth_wm_drift_5x5x32x120: torch.Tensor | None = None
        self._window_step_idx: int = 0
        self._spatial_dct_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}

    @torch.no_grad()
    def _get_2d_dct_spatial_basis(
        self, H_lat: int, W_lat: int, device: torch.device, k_max: int = 15
    ) -> torch.Tensor:
        """
        Плавный 2D-DCT базис низких/средних пространственных частот.
        Исключает разрывы соседних 32x32 патчей (никакой сетки 12x8).
        """
        key = (int(H_lat), int(W_lat), int(k_max), device)
        if key not in self._spatial_dct_cache:
            ys = (torch.arange(H_lat, device=device, dtype=torch.float32) + 0.5) * (math.pi / float(H_lat))
            xs = (torch.arange(W_lat, device=device, dtype=torch.float32) + 0.5) * (math.pi / float(W_lat))
            modes = []
            for u in range(min(4, H_lat)):
                for v in range(min(4, W_lat)):
                    if u == 0 and v == 0:
                        continue
                    m = torch.outer(torch.cos(u * ys), torch.cos(v * xs))
                    m = (m - m.mean()) / m.std(unbiased=False).clamp(min=1e-5)
                    modes.append(m)
                    if len(modes) >= k_max:
                        break
                if len(modes) >= k_max:
                    break
            if not modes:
                modes.append(torch.zeros((H_lat, W_lat), device=device, dtype=torch.float32))
            self._spatial_dct_cache[key] = torch.stack(modes, dim=0)
        return self._spatial_dct_cache[key]

    @torch.no_grad()
    def unfold_wm_spectral_3d_s3(
        self,
        wm_5x5x32x120: torch.Tensor | None,
        active_region_flags: list[float] | None,
        shape_5d: tuple[int, int, int, int, int],
        base_seed: int,
        stochastic: float,
        wm_radius: float,
        gen_rand: torch.Generator | None,
        device: torch.device,
        unassigned_stochastic: float | None = None,
    ) -> torch.Tensor:
        stoch_act = float(max(0.0, min(1.0, stochastic)))
        stoch_unass = (
            stoch_act
            if unassigned_stochastic is None
            else float(max(0.0, min(1.0, unassigned_stochastic)))
        )

        # Полный режим "из коробки": чистый torch.randn()
        if stoch_act >= 0.999 and stoch_unass >= 0.999:
            return torch.randn(shape_5d, generator=gen_rand, device=device, dtype=torch.float32)

        _, C, F_lat, H_lat, W_lat = shape_5d
        self._window_step_idx += 1
        win_seed = int(base_seed) + self._window_step_idx * 997

        g1 = torch.Generator(device=device).manual_seed(win_seed + 101)
        g2 = torch.Generator(device=device).manual_seed(win_seed + 202)
        g3 = torch.Generator(device=device).manual_seed(win_seed + 303)
        g4 = torch.Generator(device=device).manual_seed(win_seed + 404)

        b1 = torch.randn((5, C, F_lat, H_lat, W_lat), generator=g1, device=device, dtype=torch.float32)
        b2 = torch.randn((5, C, F_lat, H_lat, W_lat), generator=g2, device=device, dtype=torch.float32)
        b3 = torch.randn((5, C, F_lat, H_lat, W_lat), generator=g3, device=device, dtype=torch.float32)
        b4 = torch.randn((5, C, F_lat, H_lat, W_lat), generator=g4, device=device, dtype=torch.float32)

        if wm_5x5x32x120 is not None and wm_5x5x32x120.numel() == 96000:
            wm = wm_5x5x32x120.to(device=device, dtype=torch.float32).view(5, 5, 32, 120)
        elif wm_5x5x32x120 is not None and wm_5x5x32x120.numel() == 19200:
            wm = wm_5x5x32x120.to(device=device, dtype=torch.float32).view(1, 5, 32, 120).expand(5, -1, -1, -1).contiguous()
        else:
            wm = torch.zeros((5, 5, 32, 120), device=device, dtype=torch.float32)

        if self.smooth_wm_drift_5x5x32x120 is None or self.smooth_wm_drift_5x5x32x120.shape != wm.shape:
            self.smooth_wm_drift_5x5x32x120 = wm.clone()
        else:
            self.smooth_wm_drift_5x5x32x120.lerp_(wm, 0.04)

        dyn_wm = wm - self.smooth_wm_drift_5x5x32x120 * 0.85
        sig_gate = math.tanh(float(torch.norm(dyn_wm).item()) * 5.0)

        dyn_5x32x600 = dyn_wm.permute(0, 2, 1, 3).reshape(5, 32, 600)
        chrono_5xFx600 = F.interpolate(
            dyn_5x32x600.transpose(1, 2), size=F_lat, mode="linear", align_corners=False
        ).transpose(1, 2).contiguous()

        c_std = chrono_5xFx600.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5)
        w_5xFx600 = (chrono_5xFx600 - chrono_5xFx600.mean(dim=-1, keepdim=True)) / c_std

        ch_3 = F.adaptive_avg_pool1d(w_5xFx600, 3 * C).view(5, F_lat, 3, C).permute(2, 0, 3, 1).unsqueeze(-1).unsqueeze(-1)

        dct_sp = self._get_2d_dct_spatial_basis(H_lat, W_lat, device, k_max=15)
        K_sp = dct_sp.shape[0]
        sp_coeffs = F.adaptive_avg_pool1d(w_5xFx600, 3 * K_sp).view(5, F_lat, 3, K_sp).permute(2, 0, 1, 3)
        sp_3 = torch.einsum("srfk,khw->srfhw", sp_coeffs, dct_sp).unsqueeze(2) / math.sqrt(float(K_sp))

        rad = float(max(0.02, min(1.0, wm_radius)))
        scale_ang = math.pi * 0.85 * rad * sig_gate
        phi_a = (ch_3[0] * 0.65 + sp_3[0] * 0.35) * scale_ang
        phi_b = (ch_3[1] * 0.65 + sp_3[1] * 0.35) * scale_ang
        phi_c = (ch_3[2] * 0.65 + sp_3[2] * 0.35) * (scale_ang * 0.5)

        g_12 = torch.cos(phi_a) * b1 + torch.sin(phi_a) * b2
        g_34 = torch.cos(phi_b) * b3 + torch.sin(phi_b) * b4
        G_wm_5 = torch.cos(phi_c) * g_12 + torch.sin(phi_c) * g_34

        reg_flags = (
            list(active_region_flags)
            if (active_region_flags and len(active_region_flags) == 5)
            else [0.0, 1.0, 0.0, 0.0, 0.0]
        )
        if not any(x > 0.5 for x in reg_flags):
            reg_flags[1] = 1.0

        active_mask_t = torch.tensor(reg_flags, device=device, dtype=torch.float32)
        raw_w = torch.where(
            active_mask_t > 0.5,
            torch.tensor(3.0, device=device),
            torch.tensor(0.35, device=device),
        )
        l2_w5 = (raw_w / torch.norm(raw_w, p=2)).view(5, 1, 1, 1, 1)

        if stoch_act <= 0.001 and stoch_unass <= 0.001:
            return torch.sum(G_wm_5 * l2_w5, dim=0, keepdim=True).contiguous()

        # Раздельная стохастика для активных и пустых регионов (сохраняет точное N(0, I)):
        reg_stoch_5 = (active_mask_t * stoch_act + (1.0 - active_mask_t) * stoch_unass).view(5, 1, 1, 1, 1)
        angle_s_5 = reg_stoch_5 * (math.pi * 0.5)
        G_rand_5 = torch.randn((5, C, F_lat, H_lat, W_lat), generator=gen_rand, device=device, dtype=torch.float32)
        reg_noise_5 = torch.cos(angle_s_5) * G_wm_5 + torch.sin(angle_s_5) * G_rand_5
        return torch.sum(reg_noise_5 * l2_w5, dim=0, keepdim=True).contiguous()

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
        rostro_caudal_carriers: list[dict[str, Any]] | None = None,
        passive_emb: torch.Tensor | None = None,
        passive_mask: torch.Tensor | None = None,
        total_lat_frames: int = 3,
        h_lat: int = 8,
        w_lat: int = 12,
        gen: torch.Generator | None = None,
        is_continuation_block: bool = False,
        stochastic: float = 0.0,
        unassigned_stochastic: float | None = None,
        wm_radius: float = 0.35,
        base_seed: int = 42,
        wm_19200_vec: torch.Tensor | None = None,
        wm_5x5x32x120: torch.Tensor | None = None,
        active_region_flags: list[float] | None = None,
        temporal_overlap_cond_strength: float = 0.35,
        live_pos_emb_fn: Callable[[int, int], torch.Tensor | None] | None = None,
        live_wm_step_fn: Callable[[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]] | None] | None = None,
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
        wm_source = wm_5x5x32x120 if wm_5x5x32x120 is not None else wm_19200_vec
        noise_5d = self.unfold_wm_spectral_3d_s3(
            wm_5x5x32x120=wm_source,
            active_region_flags=active_region_flags,
            shape_5d=shape,
            base_seed=base_seed,
            stochastic=stochastic,
            wm_radius=wm_radius,
            gen_rand=gen,
            device=device,
            unassigned_stochastic=unassigned_stochastic,
        )

        latents_5d = noise_5d.clone()
        # Хронологический каузальный префикс 0..n_cond-1 закреплён как чистый видео-контекст (t_per_token = 0),
        # передавая в 3D-DiT реальный вектор скорости движения из последних 1 / 9 / 17 кадров:
        cond_mask_5d = torch.zeros(
            (1, 1, total_lat_frames, h_lat, w_lat), device=device, dtype=torch.float32
        )
        clean_anchor_prefix_5d = None
        if is_continuation_block and anchor_latent_0 is not None and n_cond > 0:
            clean_anchor_prefix_5d = anchor_latent_0[:, :, :n_cond].clone().float()
            cond_mask_5d[:, :, :n_cond] = 1.0
            latents_5d[:, :, :n_cond] = clean_anchor_prefix_5d

        conditioning_mask = pipe._pack_latents(cond_mask_5d, 1, 1).squeeze(-1).to(device)
        conditioning_mask_b2 = (
            torch.cat([conditioning_mask, conditioning_mask], dim=0)
            if self.cfg_scale > 1.05
            else conditioning_mask
        )

        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        use_cfg_overall = self.cfg_scale > 1.05
        curr_pos_emb = pos_emb
        curr_pos_mask = pos_mask

        noise_pred_final = None
        n_total_steps = len(timesteps)
        sigmas_schedule = self.backend.scheduler.sigmas.to(device=device, dtype=torch.float32)

        for i, t in enumerate(timesteps):
            sigma_cur = float(sigmas_schedule[i].item()) if i < len(sigmas_schedule) else (float(t.item()) / 1000.0)

            if i > 0 and live_wm_step_fn is not None:
                live_pack = live_wm_step_fn(i, n_total_steps)
                if live_pack is not None:
                    fresh_emb, fresh_mask, _, _ = live_pack
                    if fresh_mask.shape == curr_pos_mask.shape and torch.equal(fresh_mask, curr_pos_mask):
                        curr_pos_emb = (curr_pos_emb.float() * 0.65 + fresh_emb.float() * 0.35).to(dtype)
                    else:
                        curr_pos_emb = fresh_emb.to(dtype)
                        curr_pos_mask = fresh_mask.to(torch.bool)
            elif i > 0 and live_pos_emb_fn is not None:
                fresh_pos = live_pos_emb_fn(i, n_total_steps)
                if fresh_pos is not None and fresh_pos.shape == curr_pos_emb.shape:
                    curr_pos_emb = (curr_pos_emb.float() * 0.65 + fresh_pos.float() * 0.35).to(dtype)

            # LTXConditionPipeline: мягкий t^2 джиттер на префиксе 0..n_cond-1 (исключает залипание движения при ctx=2,3)
            step_latents_5d = latents_5d
            if is_continuation_block and n_cond > 0 and clean_anchor_prefix_5d is not None:
                eff_cond_noise = (
                    max(self.image_cond_noise_scale, 0.08)
                    if n_cond > 1
                    else self.image_cond_noise_scale
                )
                if eff_cond_noise > 0.0 and sigma_cur > 0.50:
                    t_norm = float(t.item()) / 1000.0
                    jit = torch.randn(
                        clean_anchor_prefix_5d.shape,
                        generator=gen,
                        device=device,
                        dtype=torch.float32,
                    )
                    step_latents_5d = latents_5d.clone()
                    step_latents_5d[:, :, :n_cond] = (
                        clean_anchor_prefix_5d + eff_cond_noise * jit * (t_norm ** 2)
                    )

            latents_packed = pipe._pack_latents(step_latents_5d, 1, 1)

            if use_cfg_overall:
                latent_input = torch.cat([latents_packed, latents_packed], dim=0).to(dtype)
                t_expand = t.expand(2).unsqueeze(-1).float()
                t_per_token = t_expand * (1.0 - conditioning_mask_b2)
                p_emb = torch.cat([neg_emb, curr_pos_emb], dim=0)
                p_msk = torch.cat([neg_mask, curr_pos_mask], dim=0)
            else:
                latent_input = latents_packed.to(dtype)
                t_expand = t.expand(1).unsqueeze(-1).float()
                t_per_token = t_expand * (1.0 - conditioning_mask)
                p_emb = curr_pos_emb
                p_msk = curr_pos_mask

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

            # Шагаем СТРОГО по свободным будущим кадрам [n_cond:], сохраняя чистый хронологический префикс [:n_cond]:
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

        return latents_5d.float(), 1.0
