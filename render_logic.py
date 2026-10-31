#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ANALYTIC S^3 ROTATION DIFFUSION RENDERER (TAESD ACCELERATED)
===================================================================================
- Никакого хардкода концептов: движок 100% амодален.
- Полностью сохранен apply_color_surgery_and_blend_gpu для стабилизации цвета TAESD.
- Поканальная интерполяция шума латента: активные регионы управляются мозгом,
  неактивные регионы сохраняют живую стохастическую динамику диффузии.
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import gc
import math
import types
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
    LCMScheduler,
    AutoencoderTiny
)
from neuro_vram_bus import HardwareVRAMBus

GW, GH = 512, 384


class NeuroRender:
    def __init__(self, mode="lcm", compile_unet=False, remote_conn=None, use_taesd=True,
                 stochastic: float = 0.0, unassigned_stochastic: float = 1.0, seed: int = 0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower().replace("_", "-")
        self.remote_conn = remote_conn
        self.use_taesd = use_taesd
        self.stochastic_level = float(np.clip(stochastic, 0.0, 1.0))
        self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        self.base_seed = int(seed)
        self.vae_full = None
        self.vae_tiny = None
        self.vram_bus = HardwareVRAMBus()

        self.active_wm_5x5x32x120: Optional[torch.Tensor] = None
        self.smooth_wm_drift: Optional[torch.Tensor] = None
        self.active_seed_val: int = self.base_seed
        self._gaussian_basis_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self.proj_768_to_1024 = None
        self.proj_768_to_1280 = None
        self.proj_768_to_pooled = None

        if self.remote_conn is not None:
            return

        if self.mode in ["sdxl-turbo", "sdxl"]:
            self.mode = "sdxl-turbo"
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)
            self._build_sdxl_bridge()

        elif self.mode in ["turbo", "sd-turbo"]:
            self.mode = "sd-turbo"
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)
            self._build_sdxl_or_turbo_bridge()

        else:
            self.mode = "lcm"
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=self.dtype
            ).to(self.device)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.is_sdxl = hasattr(self.pipe, "text_encoder_2")
        self.vae_full = self.pipe.vae
        try:
            taesd_id = "madebyollin/taesdxl" if self.is_sdxl else "madebyollin/taesd"
            self.vae_tiny = AutoencoderTiny.from_pretrained(taesd_id, torch_dtype=self.dtype).to(self.device)
        except Exception:
            self.vae_tiny = None

        self.set_taesd(self.use_taesd)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        self._install_direct_wm_prepare_latents()

        if compile_unet and self.device == 'cuda':
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass

        self._warmup()

    @torch.no_grad()
    def _get_gaussian_bases_4xHxW(self, shape: tuple, device: torch.device):
        key = (shape, device, self.active_seed_val)
        if key not in self._gaussian_basis_cache:
            g1 = torch.Generator(device=device).manual_seed(101 + self.active_seed_val)
            g2 = torch.Generator(device=device).manual_seed(202 + self.active_seed_val)
            g3 = torch.Generator(device=device).manual_seed(303 + self.active_seed_val)
            g4 = torch.Generator(device=device).manual_seed(404 + self.active_seed_val)
            b1 = torch.randn(shape, generator=g1, device=device, dtype=torch.float32)
            b2 = torch.randn(shape, generator=g2, device=device, dtype=torch.float32)
            b3 = torch.randn(shape, generator=g3, device=device, dtype=torch.float32)
            b4 = torch.randn(shape, generator=g4, device=device, dtype=torch.float32)
            self._gaussian_basis_cache[key] = (b1, b2, b3, b4)
        return self._gaussian_basis_cache[key]

    @torch.no_grad()
    def unfold_wm_to_latent_tensor_4xHxW(self, wm_5x5x32x120: Optional[torch.Tensor],
                                         target_shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        b1, b2, b3, b4 = self._get_gaussian_bases_4xHxW(target_shape, device)
        stoch_act = float(np.clip(self.stochastic_level, 0.0, 1.0))
        stoch_unass = float(np.clip(self.unassigned_stochastic, 0.0, 1.0))
        B, C, H, W = target_shape
        num_voxels = H * W

        wm = (
            wm_5x5x32x120.to(device=device, dtype=torch.float32)
            if wm_5x5x32x120 is not None
            else torch.zeros((5, 5, 32, 120), device=device, dtype=torch.float32)
        )

        if self.smooth_wm_drift is None or self.smooth_wm_drift.shape != wm.shape:
            self.smooth_wm_drift = wm.clone()
        else:
            self.smooth_wm_drift.lerp_(wm, 0.12)

        # 4 канала латента соответствуют регионам: Ch0=Fpz/AFz, Ch1=F3, Ch2=F4, Ch3=FCz
        p0 = (self.smooth_wm_drift[4].mean(dim=0) * 0.5 + self.smooth_wm_drift[0].mean(dim=0) * 0.5).reshape(-1)
        p1 = self.smooth_wm_drift[2].mean(dim=0).reshape(-1)
        p2 = self.smooth_wm_drift[3].mean(dim=0).reshape(-1)
        p3 = self.smooth_wm_drift[1].mean(dim=0).reshape(-1)
        all_4_ch = torch.stack([p0, p1, p2, p3], dim=0)

        if num_voxels <= 3840:
            spatial_wm = all_4_ch[:, :num_voxels].view(1, 4, H, W)
        else:
            reps = (num_voxels // 3840) + 1
            spatial_wm = all_4_ch.repeat(1, reps)[:, :num_voxels].view(1, 4, H, W)

        phi_a = spatial_wm * (math.pi * 0.5)
        phi_b = -spatial_wm * (math.pi * 0.35)

        g_a = torch.cos(phi_a) * b1 + torch.sin(phi_a) * b2
        g_b = torch.cos(phi_b) * b3 + torch.sin(phi_b) * b4
        G_wm = math.sqrt(0.5) * (g_a + g_b)

        # Свежий гауссовский шум для стохастической динамики диффузии
        G_rand = torch.randn(target_shape, device=device, dtype=torch.float32)

        if self.vram_bus.attached_pid > 0:
            meta = self.vram_bus.meta_64
            monty_code = int(round(float(meta[27].item())))
            unass_m = 0.0 if monty_code == 0 else 1.0
            m_afz = 1.0 if float(meta[21 + 0].item()) > 0.5 else unass_m
            m_fcz = 1.0 if float(meta[21 + 1].item()) > 0.5 else unass_m
            m_f3  = 1.0 if float(meta[21 + 2].item()) > 0.5 else unass_m
            m_f4  = 1.0 if float(meta[21 + 3].item()) > 0.5 else unass_m
            m_fpz = 1.0 if float(meta[21 + 4].item()) > 0.5 else unass_m
            ch_mask = torch.tensor(
                [max(m_fpz, m_afz), m_f3, m_f4, m_fcz], device=device, dtype=torch.float32
            ).view(1, 4, 1, 1)
            eff_stoch_4 = ch_mask * stoch_act + (1.0 - ch_mask) * stoch_unass
        else:
            eff_stoch_4 = torch.full((1, 4, 1, 1), stoch_act, device=device, dtype=torch.float32)

        angle_s = eff_stoch_4 * (math.pi * 0.5)
        noise_out = torch.cos(angle_s) * G_wm + torch.sin(angle_s) * G_rand

        if B > 1:
            noise_out = noise_out.expand(B, -1, -1, -1)
        return noise_out.to(device=device, dtype=dtype)

    def _install_direct_wm_prepare_latents(self):
        render_self = self

        def hooked_prepare_latents(pipe_self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, *args, **kwargs):
            image = image.to(device=device, dtype=dtype)
            batch_size_eff = batch_size * num_images_per_prompt

            if image.shape[1] == 4:
                init_latents = image
            else:
                vae_out = pipe_self.vae.encode(image)
                if hasattr(vae_out, "latent_dist"):
                    init_latents = vae_out.latent_dist.mode()
                elif hasattr(vae_out, "latents"):
                    init_latents = vae_out.latents
                else:
                    init_latents = vae_out[0]
                scaling_factor = getattr(pipe_self.vae.config, "scaling_factor", 0.18215)
                init_latents = scaling_factor * init_latents

            if batch_size_eff > init_latents.shape[0] and batch_size_eff % init_latents.shape[0] == 0:
                init_latents = torch.cat([init_latents] * (batch_size_eff // init_latents.shape[0]), dim=0)
            else:
                init_latents = torch.cat([init_latents], dim=0)

            noise = render_self.unfold_wm_to_latent_tensor_4xHxW(
                render_self.active_wm_5x5x32x120, init_latents.shape, dtype, device
            )
            latents = pipe_self.scheduler.add_noise(init_latents, noise, timestep)
            return latents

        self.pipe.prepare_latents = types.MethodType(hooked_prepare_latents, self.pipe)

    @torch.no_grad()
    def _build_sdxl_bridge(self):
        try:
            w_l = self.pipe.text_encoder.get_input_embeddings().weight.detach().float()
            w_g = self.pipe.text_encoder_2.get_input_embeddings().weight.detach().float()
            n = min(w_l.shape[0], w_g.shape[0])
            sol = torch.linalg.lstsq(w_l[:n], w_g[:n]).solution
            self.proj_768_to_1280 = sol.to(device=self.device, dtype=self.dtype)
            if hasattr(self.pipe.text_encoder_2, "text_projection"):
                t_proj = self.pipe.text_encoder_2.text_projection.weight.detach().float()
                self.proj_768_to_pooled = (sol @ t_proj.T).to(device=self.device, dtype=self.dtype)
            else:
                self.proj_768_to_pooled = self.proj_768_to_1280
        except Exception:
            pass

    @torch.no_grad()
    def _build_sdxl_or_turbo_bridge(self):
        try:
            clip_l = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32).to(self.device)
            w_l = clip_l.get_input_embeddings().weight.detach().float()
            w_h = self.pipe.text_encoder.get_input_embeddings().weight.detach().float()
            n = min(w_l.shape[0], w_h.shape[0])
            sol = torch.linalg.lstsq(w_l[:n], w_h[:n]).solution
            self.proj_768_to_1024 = sol.to(device=self.device, dtype=self.dtype)
            del clip_l
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def set_taesd(self, enable_taesd: bool):
        self.use_taesd = bool(enable_taesd)
        if self.use_taesd and self.vae_tiny is not None:
            self.pipe.vae = self.vae_tiny
        elif self.vae_full is not None:
            self.pipe.vae = self.vae_full

    @torch.no_grad()
    def ensure_non_black_init_image(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img, dtype=np.float32)
        if np.max(arr) < 4.0:
            h, w, _ = arr.shape
            ys = np.linspace(-math.pi, math.pi, h, dtype=np.float32)[:, None]
            xs = np.linspace(-math.pi, math.pi, w, dtype=np.float32)[None, :]
            base_r = 127.5 + 45.0 * np.sin(xs * 2.0 + ys)
            base_g = 127.5 + 45.0 * np.cos(xs - ys * 2.0)
            base_b = 127.5 + 45.0 * np.sin(xs * 1.5 - ys * 1.5)
            arr = np.stack([base_r, base_g, base_b], axis=-1)
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return pil_img

    @torch.no_grad()
    def adapt_prompt_embeds(self, t_emb: torch.Tensor, pooled_emb: Optional[torch.Tensor] = None):
        if t_emb.ndim == 2:
            t_emb = t_emb.unsqueeze(0)
        t_emb = t_emb.to(device=self.device, dtype=self.dtype)
        t_emb = torch.nan_to_num(t_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        dim = t_emb.shape[-1]

        if self.is_sdxl:
            if dim == 768:
                if self.proj_768_to_1280 is not None:
                    emb_1280 = torch.matmul(t_emb, self.proj_768_to_1280)
                    norm_ref = torch.norm(t_emb, p=2, dim=-1, keepdim=True) * (1280.0 / 768.0) ** 0.5
                    emb_1280 = F.normalize(emb_1280, p=2, dim=-1) * norm_ref
                else:
                    emb_1280 = F.pad(t_emb, (0, 1280 - 768))
                if pooled_emb is None:
                    if self.proj_768_to_pooled is not None:
                        pooled_emb = torch.matmul(t_emb[:, -1, :], self.proj_768_to_pooled)
                    else:
                        pooled_emb = emb_1280.mean(dim=1)
                t_emb = torch.cat([t_emb, emb_1280], dim=-1)
            elif pooled_emb is None:
                pooled_emb = torch.zeros((t_emb.shape[0], 1280), dtype=self.dtype, device=self.device)
            return t_emb, pooled_emb

        elif self.mode == "sd-turbo":
            if dim == 768:
                if self.proj_768_to_1024 is not None:
                    emb_1024 = torch.matmul(t_emb, self.proj_768_to_1024)
                    norm_ref = torch.norm(t_emb, p=2, dim=-1, keepdim=True) * (1024.0 / 768.0) ** 0.5
                    t_emb = F.normalize(emb_1024, p=2, dim=-1) * norm_ref
                else:
                    t_emb = F.pad(t_emb, (0, 1024 - 768))
            return t_emb, None

        return t_emb, None

    def _warmup(self):
        dummy_image = Image.fromarray(np.full((GH, GW, 3), 128, dtype=np.uint8))
        dummy_768 = torch.randn((1, 77, 768), device=self.device, dtype=self.dtype)
        with torch.no_grad():
            self.generate(image=dummy_image, prompt_embeds=dummy_768, strength=0.7, num_inference_steps=2)

    def encode_prompts_universal(self, prompt_list):
        p_embeds_list, pooled_list = [], []
        with torch.no_grad():
            for p in prompt_list:
                if self.is_sdxl:
                    p_emb, _, pool, _ = self.pipe.encode_prompt(
                        prompt=p, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    p_embeds_list.append(p_emb[0].cpu().numpy())
                    pooled_list.append(pool[0].cpu().numpy())
                else:
                    p_emb, _ = self.pipe.encode_prompt(
                        prompt=p, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    p_embeds_list.append(p_emb[0].cpu().numpy())
        return {
            'c_bases': p_embeds_list,
            'pooled_bases': pooled_list if self.is_sdxl else None,
            'is_sdxl': self.is_sdxl,
            'mode': self.mode
        }

    def generate(self, **kwargs):
        req_stoch = kwargs.pop('stochastic', None)
        req_unass_stoch = kwargs.pop('unassigned_stochastic', None)
        req_seed = kwargs.pop('seed', None)
        wm_tensor = kwargs.pop('wm_5x5x32x120', None)

        self.vram_bus.try_attach_to_existing_vram()
        has_vram_bus = (self.vram_bus.attached_pid > 0)

        if req_stoch is not None:
            self.stochastic_level = float(np.clip(float(req_stoch), 0.0, 1.0))
        if req_unass_stoch is not None:
            self.unassigned_stochastic = float(np.clip(float(req_unass_stoch), 0.0, 1.0))

        active_seed = int(req_seed) if req_seed is not None else self.base_seed

        if 'use_taesd' in kwargs:
            req_t = bool(kwargs.pop('use_taesd'))
            if req_t != self.use_taesd:
                self.set_taesd(req_t)

        if ('prompt_embeds' not in kwargs or kwargs['prompt_embeds'] is None) and has_vram_bus:
            kwargs['prompt_embeds'] = self.vram_bus.live_clip_77x768.unsqueeze(0)

        if wm_tensor is None and has_vram_bus:
            wm_tensor = self.vram_bus.wm_field_5x5x32x120

        steps = int(kwargs.get("num_inference_steps", 2))
        strength = float(np.clip(kwargs.get("strength", 0.65), 0.25, 0.98))
        if int(steps * strength) < 1:
            steps = max(2, int(np.ceil(1.0 / max(0.25, strength))))
        kwargs["num_inference_steps"] = steps
        kwargs["strength"] = strength

        self.active_wm_5x5x32x120 = wm_tensor
        self.active_seed_val = active_seed

        if 'image' in kwargs and kwargs['image'] is not None:
            kwargs['image'] = self.ensure_non_black_init_image(kwargs['image'])

        if 'prompt_embeds' in kwargs and kwargs['prompt_embeds'] is not None:
            t_emb, pool_emb = self.adapt_prompt_embeds(
                kwargs['prompt_embeds'],
                kwargs.get('pooled_prompt_embeds')
            )
            kwargs['prompt_embeds'] = t_emb
            if self.is_sdxl:
                kwargs['pooled_prompt_embeds'] = pool_emb
            else:
                kwargs.pop('pooled_prompt_embeds', None)

        if self.mode in ["turbo", "sd-turbo", "sdxl-turbo", "sdxl"]:
            kwargs["guidance_scale"] = 0.0
        else:
            kwargs.setdefault("guidance_scale", 1.35)

        kwargs.pop("generator", None)

        if has_vram_bus and steps > 1:
            def _live_wm_step_callback(pipe_obj, step_index, timestep, callback_kwargs):
                live_77 = self.vram_bus.live_clip_77x768
                if torch.norm(live_77) > 1.0:
                    new_pe, _ = self.adapt_prompt_embeds(live_77.unsqueeze(0))
                    if "prompt_embeds" in callback_kwargs and callback_kwargs["prompt_embeds"] is not None:
                        old_pe = callback_kwargs["prompt_embeds"]
                        if old_pe.shape == new_pe.shape:
                            callback_kwargs["prompt_embeds"] = old_pe * 0.45 + new_pe * 0.55
                return callback_kwargs

            kwargs["callback_on_step_end"] = _live_wm_step_callback
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents", "prompt_embeds"]

        return self.pipe(**kwargs).images[0]
