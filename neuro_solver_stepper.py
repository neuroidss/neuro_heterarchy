#!/usr/bin/env python3
"""
neuro_solver_stepper.py — Чистый латентный интегратор 128D (Без потокенного смешивания разных предложений)
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F

from neuro_ltx_backend import LTXModelBackend, rescale_noise_cfg


@torch.no_grad()
def tone_map_latents_ltx(latents: torch.Tensor, compression: float = 0.25) -> torch.Tensor:
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
    ref_latents: torch.Tensor | None,
    factor: float = 0.25,
) -> torch.Tensor:
    eps = 1e-6
    mu_curr = curr_latents.mean(dim=(2, 3, 4), keepdim=True)
    sigma_curr = curr_latents.std(dim=(2, 3, 4), keepdim=True).clamp(min=eps)

    if ref_latents is not None and factor > 0.0:
        mu_ref = ref_latents.mean(dim=(2, 3, 4), keepdim=True).to(curr_latents.device, curr_latents.dtype)
        sigma_ref = ref_latents.std(dim=(2, 3, 4), keepdim=True).to(curr_latents.device, curr_latents.dtype).clamp(min=eps)
        target_mu = (1.0 - factor) * mu_curr + factor * (mu_ref * 0.85)
        target_sigma = torch.clamp((1.0 - factor) * sigma_curr + factor * sigma_ref, min=0.65, max=1.05)
        curr_latents = ((curr_latents - mu_curr) / sigma_curr) * target_sigma + target_mu

    return tone_map_latents_ltx(curr_latents, compression=0.25)


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
        steps: int = 20,
        cfg_scale: float = 4.0,
        guidance_rescale: float = 0.7,
    ):
        self.backend = backend
        self.referee = referee
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.guidance_rescale = guidance_rescale

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
        total_lat_frames: int = 5,
        h_lat: int = 14,
        w_lat: int = 20,
        gen: torch.Generator | None = None,
        is_continuation_block: bool = False,
    ) -> tuple[torch.Tensor, float]:
        pipe = self.backend.pipe
        device = self.backend.device
        dtype = self.backend.dtype

        in_channels = self.backend.transformer.config.in_channels
        video_seq_len = total_lat_frames * h_lat * w_lat
        timesteps = self.backend.set_scheduler_timesteps(self.steps, video_seq_len)

        shape = (1, in_channels, total_lat_frames, h_lat, w_lat)
        noise = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)

        latents_5d = noise.clone()
        cond_mask_5d = torch.zeros((1, 1, total_lat_frames, h_lat, w_lat), device=device, dtype=torch.float32)

        if is_continuation_block and anchor_latent_0 is not None:
            latents_5d[:, :, :1] = anchor_latent_0[:, :, -1:].clone()
            cond_mask_5d[:, :, :1] = 1.0

        latents_packed = pipe._pack_latents(latents_5d, 1, 1).to(torch.float32)
        cond_mask_packed = pipe._pack_latents(cond_mask_5d, 1, 1).squeeze(-1).to(device)

        rope_scale = (
            self.backend.temporal_ratio / 16.0,
            float(self.backend.spatial_ratio),
            float(self.backend.spatial_ratio),
        )

        # Используем строго 2 потока (neg_emb и чистый pos_emb без наложения двух разных предложений!)
        batch_prompt_embeds = torch.cat([neg_emb, pos_emb], dim=0)
        batch_prompt_mask = torch.cat([neg_mask, pos_mask], dim=0)
        cond_mask_batch2 = torch.cat([cond_mask_packed, cond_mask_packed], dim=0)

        dominance_ratio = 1.0
        probe_step_idx = max(2, int(len(timesteps) * 0.30))

        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents_packed, latents_packed], dim=0)
            t_expand = t.expand(latent_input.shape[0]).unsqueeze(-1).float()
            t_per_token = t_expand * (1.0 - cond_mask_batch2)

            noise_pred_2 = self.backend.transformer(
                hidden_states=latent_input.to(dtype),
                encoder_hidden_states=batch_prompt_embeds,
                timestep=t_per_token,
                encoder_attention_mask=batch_prompt_mask,
                num_frames=total_lat_frames,
                height=h_lat,
                width=w_lat,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0].float()

            noise_uncond, noise_text = noise_pred_2.chunk(2)
            t_single = t_per_token.chunk(2)[0]

            if i == probe_step_idx and self.referee is not None and entity_probes and len(entity_probes) >= 2:
                dominance_ratio = self.referee.compute_mid_clash_dominance(
                    noisy_latents_packed=latents_packed,
                    t_per_token=t_single,
                    v_uncond=noise_uncond,
                    v_clash_text=noise_text,
                    entity_probes=entity_probes,
                    passive_void_embeds=passive_emb if passive_emb is not None else walk_in_emb,
                    passive_void_mask=passive_mask if passive_mask is not None else walk_in_mask,
                    t_lat=total_lat_frames,
                    h_lat=h_lat,
                    w_lat=w_lat,
                    rope_scale=rope_scale,
                )

            noise_cfg = noise_uncond + self.cfg_scale * (noise_text - noise_uncond)
            noise_pred_final = rescale_noise_cfg(noise_cfg, noise_text, guidance_rescale=self.guidance_rescale)

            noise_pred_5d = pipe._unpack_latents(noise_pred_final, total_lat_frames, h_lat, w_lat, 1, 1)
            curr_latents_5d = pipe._unpack_latents(latents_packed, total_lat_frames, h_lat, w_lat, 1, 1)

            if is_continuation_block:
                gen_noise = noise_pred_5d[:, :, 1:]
                gen_lat = curr_latents_5d[:, :, 1:]
                step_lat = self.backend.scheduler.step(gen_noise, t, gen_lat, return_dict=False)[0]
                curr_latents_5d = torch.cat([curr_latents_5d[:, :, :1], step_lat], dim=2)
            else:
                curr_latents_5d = self.backend.scheduler.step(noise_pred_5d, t, curr_latents_5d, return_dict=False)[0]

            latents_packed = pipe._pack_latents(curr_latents_5d, 1, 1)

        unpacked_z0 = pipe._unpack_latents(latents_packed, total_lat_frames, h_lat, w_lat, 1, 1).float()
        ref_5d = genesis_ref_latent_5d if genesis_ref_latent_5d is not None else unpacked_z0
        unpacked_z0 = stabilize_sliding_latents(unpacked_z0, ref_5d, factor=0.25)

        return unpacked_z0, float(dominance_ratio)
