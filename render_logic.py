#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ANALYTIC S^3 ROTATION DIFFUSION RENDERER (FULL-FRAME REGIONAL MIXING)
===================================================================================
- Исправлена заморозка активного региона: каждый из 5 регионов (Fpz, AFz, F3, F4, FCz)
  вносит вклад во все 4 канала латента [1, 4, H, W] (включая каналы яркости и формы!).
- Ползунок стохастики активного региона (stochastic) напрямую управляет стохастичностью
  всех активных регионов (даже если подключен всего 1 девайс на FCz и все пустые
  регионы заморожены в unassigned_stochastic = 0.0).
- При stochastic = 0.0 изменение фазовой картины ЭЭГ детерминированно поворачивает
  латентное поле на угол, пропорциональный ползунку wm_radius ∈ [0.0, 1.0].
===================================================================================
"""

from __future__ import annotations
from typing import Optional, List
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
                 stochastic: float = 0.0, unassigned_stochastic: float = 1.0,
                 wm_radius: float = 0.25, seed: int = 0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower().replace("_", "-")
        self.remote_conn = remote_conn
        self.use_taesd = use_taesd
        self.stochastic_level = float(np.clip(stochastic, 0.0, 1.0))
        self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        self.wm_radius = float(np.clip(wm_radius, 0.0, 1.0))
        self.active_region_flags: List[float] = [0.0, 1.0, 0.0, 0.0, 0.0]  # [AFz, FCz, F3, F4, Fpz]
        self.base_seed = int(seed)
        self.vae_full = None
        self.vae_tiny = None
        self.vram_bus = HardwareVRAMBus()

        self.active_wm_5x5x32x120: Optional[torch.Tensor] = None
        self.smooth_wm_drift: Optional[torch.Tensor] = None
        self.accum_wm_phase: float = 0.0
        self.active_seed_val: int = self.base_seed
        self._gaussian_basis_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

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
    def _get_regional_gaussian_bases(self, shape: tuple, device: torch.device):
        """
        Создает 4 взаимно ортогональных гауссовых базиса N(0, I) размера [5, 4, H, W]
        на сфере S^3 для всех 5 регионов коры.
        """
        key = (shape, device, self.active_seed_val)
        if key not in self._gaussian_basis_cache:
            B, C, H, W = shape
            g1 = torch.Generator(device=device).manual_seed(101 + self.active_seed_val)
            g2 = torch.Generator(device=device).manual_seed(202 + self.active_seed_val)
            g3 = torch.Generator(device=device).manual_seed(303 + self.active_seed_val)
            g4 = torch.Generator(device=device).manual_seed(404 + self.active_seed_val)
            b1 = torch.randn((5, C, H, W), generator=g1, device=device, dtype=torch.float32)
            b2 = torch.randn((5, C, H, W), generator=g2, device=device, dtype=torch.float32)
            b3 = torch.randn((5, C, H, W), generator=g3, device=device, dtype=torch.float32)
            b4 = torch.randn((5, C, H, W), generator=g4, device=device, dtype=torch.float32)

            # Детерминированные индексы перестановки для прямого 1-в-1 отображения
            # всех 19,200 измерений региона (5 * 32 * 120) в C * H * W вокселей латента без .mean()!
            gp = torch.Generator(device=device).manual_seed(707 + self.active_seed_val)
            n_vox = C * H * W
            perm_a = torch.randint(0, 19200, (5, n_vox), generator=gp, device=device)
            perm_b = torch.randint(0, 19200, (5, n_vox), generator=gp, device=device)
            perm_c = torch.randint(0, 19200, (5, n_vox), generator=gp, device=device)
            self._gaussian_basis_cache[key] = (b1, b2, b3, b4, perm_a, perm_b, perm_c)
        return self._gaussian_basis_cache[key]

    @torch.no_grad()
    def unfold_wm_to_latent_tensor_4xHxW(self, wm_5x5x32x120: Optional[torch.Tensor],
                                         target_shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        b1, b2, b3, b4, perm_a, perm_b, perm_c = self._get_regional_gaussian_bases(target_shape, device)
        stoch_act = float(np.clip(self.stochastic_level, 0.0, 1.0))
        stoch_unass = float(np.clip(self.unassigned_stochastic, 0.0, 1.0))
        rad = float(np.clip(self.wm_radius, 0.0, 1.0))
        B, C, H, W = target_shape

        wm = (
            wm_5x5x32x120.to(device=device, dtype=torch.float32)
            if wm_5x5x32x120 is not None
            else torch.zeros((5, 5, 32, 120), device=device, dtype=torch.float32)
        )

        # Отделяем медленный анатомический фон (smooth_wm_drift) от живой мысли:
        # Если девайс выключен (wm == const), то diff_wm == 0 и картинка стоит.
        # Как только на девайсе меняются любые из 96,000 мод — они усиливаются на 100%!
        if self.smooth_wm_drift is None or self.smooth_wm_drift.shape != wm.shape:
            self.smooth_wm_drift = wm.clone()
        else:
            self.smooth_wm_drift.lerp_(wm, 0.04)

        reg_flags = list(self.active_region_flags)
        if self.vram_bus.attached_pid > 0:
            meta = self.vram_bus.meta_64
            monty_code = int(round(float(meta[27].item())))
            unass_m = 0.0 if monty_code == 0 else 1.0
            bus_flags = [
                1.0 if float(meta[21 + i].item()) > 0.5 else unass_m
                for i in range(5)
            ]
            if any(x > 0.5 for x in bus_flags):
                reg_flags = bus_flags

        if not any(x > 0.5 for x in reg_flags):
            reg_flags[1] = 1.0

        active_mask_t = torch.tensor(reg_flags, device=device, dtype=torch.float32)
        raw_weights = torch.where(active_mask_t > 0.5, torch.tensor(3.0, device=device), torch.tensor(0.35, device=device))
        l2_weights_5 = (raw_weights / torch.norm(raw_weights, p=2)).view(5, 1, 1, 1)

        reg_stoch_5 = (active_mask_t * stoch_act + (1.0 - active_mask_t) * stoch_unass).view(5, 1, 1, 1)

        # БЕРЕМ ВСЕ 96,000 ИЗМЕРЕНИЙ [5 регионов, 19200 мод] БЕЗ ЕДИНОГО .mean()!
        # Вычитаем 85% медленного статического пьедестала, чтобы динамика мыслей
        # использовала весь фазовый круг [-pi, +pi]:
        dynamic_wm_5x19200 = (wm - self.smooth_wm_drift * 0.85).reshape(5, 19200)
        wm_energy = float(torch.norm(dynamic_wm_5x19200).item())
        sig_gate = math.tanh(wm_energy * 5.0)

        # Z-score whitening по всем 19,200 измерениям каждого региона:
        wm_mean = dynamic_wm_5x19200.mean(dim=-1, keepdim=True)
        wm_std = dynamic_wm_5x19200.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5)
        whitened_5x19200 = (dynamic_wm_5x19200 - wm_mean) / wm_std

        # Прямая выборка всех 19,200 измерений в 3 независимые фазовые решетки [5, C, H, W]:
        f1 = torch.gather(whitened_5x19200, 1, perm_a).view(5, C, H, W)
        f2 = torch.gather(whitened_5x19200, 1, perm_b).view(5, C, H, W)
        f3 = torch.gather(whitened_5x19200, 1, perm_c).view(5, C, H, W)

        # Полное вращение на сфере S^3 по трем углам Эйлера-Хопфа (phi_a, phi_b, phi_c),
        # масштабируемое ползунком wm_radius (rad ∈ [0.0, 1.0]):
        scale_ang = math.pi * 1.25 * rad * sig_gate
        phi_a = f1 * scale_ang
        phi_b = f2 * scale_ang
        phi_c = f3 * (scale_ang * 0.5)

        g_12 = torch.cos(phi_a) * b1 + torch.sin(phi_a) * b2
        g_34 = torch.cos(phi_b) * b3 + torch.sin(phi_b) * b4
        G_wm_5 = torch.cos(phi_c) * g_12 + torch.sin(phi_c) * g_34  # Точное N(0, I) во всех точках!

        G_rand_5 = torch.randn((5, C, H, W), device=device, dtype=torch.float32)

        angle_s_5 = reg_stoch_5 * (math.pi * 0.5)
        reg_noise_5 = torch.cos(angle_s_5) * G_wm_5 + torch.sin(angle_s_5) * G_rand_5

        noise_out = torch.sum(reg_noise_5 * l2_weights_5, dim=0, keepdim=True)

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
        req_wm_radius = kwargs.pop('wm_radius', None)
        req_reg_flags = kwargs.pop('active_region_flags', None)
        req_seed = kwargs.pop('seed', None)
        wm_tensor = kwargs.pop('wm_5x5x32x120', None)

        self.vram_bus.try_attach_to_existing_vram()
        has_vram_bus = (self.vram_bus.attached_pid > 0)

        if req_stoch is not None:
            self.stochastic_level = float(np.clip(float(req_stoch), 0.0, 1.0))
        if req_unass_stoch is not None:
            self.unassigned_stochastic = float(np.clip(float(req_unass_stoch), 0.0, 1.0))
        if req_wm_radius is not None:
            self.wm_radius = float(np.clip(float(req_wm_radius), 0.0, 1.0))
        if req_reg_flags is not None and len(req_reg_flags) == 5:
            self.active_region_flags = [float(x) for x in req_reg_flags]

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
