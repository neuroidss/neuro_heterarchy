#!/usr/bin/env python3
"""
neuro_solver_stepper.py — Эталонный гибрид 3D-DiT I2V + S^3 Рабочей Памяти (Без взрыва Эйлера!)
================================================================================================
- ИСПРАВЛЕН ВЗРЫВ ЦВЕТНЫХ ПЯТЕН (Sign Inversion Bug в FlowMatchEulerDiscreteScheduler):
  В diffusers `FlowMatchEulerDiscreteScheduler.step()` при передаче `per_token_timesteps` меняет знак
  `dt = current_sigma - next_sigma > 0` (вместо стандартного `dt = sigma_next - sigma < 0`), из-за чего
  передача `+noise_pred` прибавляла шум вместо денойзинга!
  Теперь используется чистый канонический шаг `scheduler.step(noise_pred, t, latents)` без `per_token_timesteps`
  (как в `pipeline_ltx_image2video.py`, строка 631), где `dt < 0` гарантированно очищает латент до sigma = 0!
- ДИНАМИКА ПРИ БОЛЬШОМ РАДИУСЕ БЕЗ РАЗРУШЕНИЯ КАРТИНКИ:
  1) Пространственно-когерентная бегущая волна S^3 размерности [1, 128, F, 1, 1]: единая фаза по HxW
     внутри каждого кадра (0% полос и 0% пятен!), но бегущий градиент фазы между кадрами f=0..F-1.
  2) Квадратичная раскачка контекстного кадра `add_noise_to_image_conditioning_latents` (из `pipeline_ltx_condition.py`):
     на первых шагах (t ~ 1000) кадр контекста сдвигается волной рабочей памяти пропорционально `wm_radius`,
     позволяя сцене свободно трансформироваться, а к концу шагов ((t/1000)^2 -> 0) сходится к идеальной резкости!
================================================================================================
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
    factor: float = 0.20,
) -> torch.Tensor:
    """
    Мягкая по-канальная стабилизация статистики (AdaIN из pipeline_ltx_i2v_long_multi_prompt.py)
    относительно эталонного кадра генезиса. Предотвращает накопление цветового и контрастного дрейфа
    при многочасовой генерации, не искажая локальные детали.
    """
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
        cfg_scale: float = 3.2,
        guidance_rescale: float = 0.7,
    ):
        self.backend = backend
        self.referee = referee
        self.steps = max(4, int(steps))
        self.cfg_scale = float(cfg_scale)
        self.guidance_rescale = float(guidance_rescale)
        self.last_velocity_5d: torch.Tensor | None = None
        self._s3_bases_cache: dict[tuple, list[torch.Tensor]] = {}
        self._accum_phase_128xF: torch.Tensor | None = None

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
        """
        Когерентное вращение 4 ортогональных гауссовых базисов b1..b4 ~ N(0, I) на сфере S^3:
        Фазовые углы (phi_a, phi_b, phi_c) имеют размерность [1, 128, F_lat, 1, 1] —
        постоянны вдоль (H, W) внутри каждого кадра (поэтому каждый 2D-срез сохраняет
        точную ковариацию N(0, I) без полос и пятен), но меняются по каналам и времени F_lat!
        """
        _, C, F_lat, _, _ = shape
        key = (shape, base_seed, device)

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
            for idx_b in range(4):
                self._s3_bases_cache[key][idx_b] = torch.roll(
                    self._s3_bases_cache[key][idx_b], shifts=-1, dims=2
                )

        b1, b2, b3, b4 = self._s3_bases_cache[key]

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
            self._accum_phase_128xF = torch.remainder(
                torch.roll(self._accum_phase_128xF, shifts=-1, dims=2)
                + wm_3xCxF[0] * (rad * 0.85),
                2.0 * math.pi,
            )

        scale_ang = math.pi * 0.85 * rad
        phi_a = self._accum_phase_128xF + wm_3xCxF[0] * scale_ang
        phi_b = (
            torch.roll(self._accum_phase_128xF, shifts=16, dims=1)
            - wm_3xCxF[1] * scale_ang
        )
        phi_c = torch.roll(self._accum_phase_128xF, shifts=32, dims=1) * 0.5 + wm_3xCxF[
            2
        ] * (scale_ang * 0.5)

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

        timesteps = self.backend.set_scheduler_timesteps(self.steps, video_seq_len)

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

        cond_mask_packed = (
            pipe._pack_latents(cond_mask_5d, 1, 1).squeeze(-1).to(device)
        )
        cond_mask_batch2 = torch.cat([cond_mask_packed, cond_mask_packed], dim=0)

        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        use_cfg_overall = self.cfg_scale > 1.05
        curr_pos_emb = pos_emb
        batch2_prompt_mask = (
            torch.cat([neg_mask, pos_mask], dim=0) if use_cfg_overall else pos_mask
        )

        noise_pred_final = None
        n_total_steps = len(timesteps)
        rad_val = float(max(0.02, min(1.0, wm_radius)))
        # Масштаб квадратичной фазовой раскачки первого кадра (по методу Lightricks image_cond_noise_scale):
        image_cond_noise_scale = 0.04 + 0.18 * rad_val

        for i, t in enumerate(timesteps):
            if live_pos_emb_fn is not None and i > 0:
                fresh_pos = live_pos_emb_fn(i, n_total_steps)
                if fresh_pos is not None and fresh_pos.shape == curr_pos_emb.shape:
                    curr_pos_emb = fresh_pos

            # Квадратичное по (t/1000)^2 возмущение опорного кадра (из pipeline_ltx_condition.py, строка 493):
            # На ранних шагах (t ~ 1000) позволяет кадру контекста двигаться вместе с рабочей памятью,
            # а на финальных шагах (t -> 0) затухает в 0, сохраняя идеальную четкость!
            if clean_anchor_prefix_5d is not None and n_cond > 0:
                t_frac = float(t.item()) / 1000.0
                latents_5d[:, :, :n_cond] = (
                    clean_anchor_prefix_5d
                    + (image_cond_noise_scale * (t_frac**2)) * noise_5d[:, :, :n_cond]
                )

            latents_packed = pipe._pack_latents(latents_5d, 1, 1).to(torch.float32)

            if use_cfg_overall:
                latent_input = torch.cat([latents_packed, latents_packed], dim=0).to(
                    dtype
                )
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
                noise_cfg = noise_uncond + self.cfg_scale * (
                    noise_text - noise_uncond
                )
                noise_pred_final = rescale_noise_cfg(
                    noise_cfg, noise_text, guidance_rescale=self.guidance_rescale
                )
            else:
                noise_pred_final = noise_pred_raw

            # КАНОНИЧЕСКИЙ ШАГ ЭЙЛЕРА ИЗ pipeline_ltx_image2video.py (строки 613-633):
            # Никакого per_token_timesteps в scheduler.step()! Это гарантирует dt = sigma_next - sigma < 0
            # и 100% правильное вычитание шума до самого sigma = 0.0!
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
        if is_continuation_block and genesis_ref_latent_5d is not None:
            unpacked_z0[:, :, n_cond:] = adain_stabilize_latents(
                unpacked_z0[:, :, n_cond:], genesis_ref_latent_5d, factor=0.18
            )

        return unpacked_z0, 1.0
