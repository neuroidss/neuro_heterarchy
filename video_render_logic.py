#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: 100% GPU RESIDENT 3D-DiT WORLD ENGINE (CLEAN CAUSAL I2V ROLLOUT)
===================================================================================
- ИСПРАВЛЕНО ЗАКИДЫВАНИЕ ПРОШЛЫХ 8 КАДРОВ В СЛОТ z[0]:
  Вместо записи 8-кадрового латента `unpacked_5d[:, :, -1:]` в 1-кадровый слот `z[0]`
  (и `.repeat(1, 1, f_lat, 1, 1)`), последний чистый кадр `all_frames[-1]` кодируется в
  1-кадровый каузальный латент `z[0]`, а будущие слоты `z[1:]` заполняются чистым S^3-шумом.
- Исправлен NameError `adain_blend_consequence_latents_gpu` -> `stabilize_sliding_window_latents_gpu`.
===================================================================================
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import os
import io
import gc
import math
import time
import types
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE
from neuro_vram_bus import HardwareVRAMBus

MODEL_REPO_MAP = {
    "ltx-video": "Lightricks/LTX-Video",
    "wan-1.3b":  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan2.2-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}

QUANT_T5_DIR = Path("/workspace/hf_cache/t5_xxl_4bit")


def resolve_quantized_t5_dir() -> Path:
    candidates = [
        QUANT_T5_DIR,
        Path(os.path.expanduser("~/.cache/huggingface/t5_xxl_4bit")),
        Path("/root/.cache/huggingface/t5_xxl_4bit"),
    ]
    for d in candidates:
        if d.exists() and (d / "config.json").exists() and (
            (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
        ):
            return d

    target = QUANT_T5_DIR if Path("/workspace/hf_cache").exists() else Path(os.path.expanduser("~/.cache/huggingface/t5_xxl_4bit"))
    target.mkdir(parents=True, exist_ok=True)
    return target


@torch.no_grad()
def tone_map_latents_ltx(latents: torch.Tensor, compression: float = 0.35) -> torch.Tensor:
    if compression <= 0.0:
        return latents
    scale_factor = float(compression) * 0.75
    abs_lat = torch.abs(latents)
    sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_lat - 1.0))
    scales = 1.0 - 0.8 * scale_factor * sigmoid_term
    return latents * scales


@torch.no_grad()
def stabilize_sliding_window_latents_gpu(
    curr_5d: torch.Tensor,
    prev_tail_5d: torch.Tensor | None,
    factor: float = 0.25,
) -> torch.Tensor:
    eps = 1e-6
    mu_c = curr_5d.mean(dim=(2, 3, 4), keepdim=True)
    std_c = curr_5d.std(dim=(2, 3, 4), keepdim=True).clamp(min=eps)

    if prev_tail_5d is not None and factor > 0.0:
        mu_p = prev_tail_5d.mean(dim=(2, 3, 4), keepdim=True).to(curr_5d.device, curr_5d.dtype)
        std_p = prev_tail_5d.std(dim=(2, 3, 4), keepdim=True).to(curr_5d.device, curr_5d.dtype).clamp(min=eps)
        target_mu = (1.0 - factor) * mu_c + factor * (mu_p * 0.85)
        target_std = torch.clamp((1.0 - factor) * std_c + factor * std_p, min=0.65, max=1.05)
        curr_5d = ((curr_5d - mu_c) / std_c) * target_std + target_mu

    return tone_map_latents_ltx(curr_5d, compression=0.30)


class NeuroVideoRenderer:
    def __init__(self, model_type: str = "ltx-video", width: int = 0, height: int = 0,
                 block_sec: float = 0.35, context_sec: float = 0.35,
                 steps: int = 15, genesis_steps: int = 20,
                 cfg_scale: float = 3.0, force_low_vram: bool = False,
                 stochastic: float = 0.0, unassigned_stochastic: float = 1.0,
                 wm_radius: float = 0.25, seed: int = 42,
                 num_frames: int = 9,
                 control_mode: str = "semantic_transformer"):

        self.device = DEVICE
        self.dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.model_type = model_type.lower()
        self.control_mode = control_mode
        self.stochastic = float(np.clip(stochastic, 0.0, 1.0))
        self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        self.wm_radius = float(np.clip(wm_radius, 0.0, 1.0))
        self.base_seed = int(seed)
        self.active_seed_val = int(seed)
        self.seq_len = 128

        self.active_region_flags: List[float] = [0.0, 1.0, 0.0, 0.0, 0.0]
        self.active_wm_5x5x32x120: Optional[torch.Tensor] = None
        self.smooth_wm_drift: Optional[torch.Tensor] = None
        self._s3_basis_cache: dict[tuple, tuple] = {}
        self.vram_bus = HardwareVRAMBus()

        self.total_vram_gb = 12.0
        if torch.cuda.is_available():
            self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        self.is_low_vram = force_low_vram or (self.total_vram_gb < 22.0)

        if width <= 0 or height <= 0:
            self.width, self.height = (448, 320) if self.is_low_vram else (512, 384)
        else:
            self.width = (width // 32) * 32
            self.height = (height // 32) * 32

        self.num_frames = max(9, ((num_frames - 1) // 8) * 8 + 1)
        self.block_sec = float(block_sec)
        self.context_sec = float(context_sec)
        self.steps = max(12, int(steps))
        self.genesis_steps = max(16, int(genesis_steps))
        self.cfg_scale = float(cfg_scale)

        self.tokenizer = None
        self.text_encoder = None
        self.pipe_t2v = None
        self.pipe_i2v = None
        self.last_pos_mask = torch.ones((1, self.seq_len), dtype=torch.bool, device=self.device)
        self.last_neg_mask = torch.ones((1, self.seq_len), dtype=torch.bool, device=self.device)

        self.last_clean_latent_5d: Optional[torch.Tensor] = None
        self.latest_end_latent_5d: Optional[torch.Tensor] = None
        self.recalled_episodic_latent_5d: Optional[torch.Tensor] = None
        self.custom_init_frame_gpu: Optional[torch.Tensor] = None
        self.last_generated_frames: List[np.ndarray] = []
        self.rollout_step_idx: int = 0

        self._init_resident_t5()
        self._load_video_pipelines()

    def _init_resident_t5(self):
        from transformers import AutoTokenizer, T5EncoderModel, BitsAndBytesConfig
        repo_id = MODEL_REPO_MAP.get(self.model_type, MODEL_REPO_MAP["ltx-video"])

        print("=" * 80)
        if not self.is_low_vram:
            print(f"🚀 [VRAM: {self.total_vram_gb:.1f} GB] Загрузка T5 в bfloat16...")
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
            self.text_encoder = T5EncoderModel.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        else:
            quant_dir = resolve_quantized_t5_dir()
            is_saved = quant_dir.exists() and (quant_dir / "model.safetensors").exists()
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )

            if is_saved:
                print(f"📥 [4-BIT T5] Загрузка 4-битной T5-XXL из {quant_dir} в VRAM...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
                self.text_encoder = T5EncoderModel.from_pretrained(
                    str(quant_dir),
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
            else:
                print("⚙️ [4-BIT T5] Квантование T5-XXL в 4 бита (NF4)...")
                self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
                self.text_encoder = T5EncoderModel.from_pretrained(
                    repo_id,
                    subfolder="text_encoder",
                    quantization_config=bnb_cfg,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                quant_dir.mkdir(parents=True, exist_ok=True)
                self.text_encoder.save_pretrained(str(quant_dir))
                self.tokenizer.save_pretrained(str(quant_dir))

        self.text_encoder.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def _load_video_pipelines(self):
        repo_id = MODEL_REPO_MAP.get(self.model_type, MODEL_REPO_MAP["ltx-video"])
        print("=" * 80)
        print(f"🎬 [VIDEO-RENDERER] Инициализация LTX-Video: {repo_id} (Режим: {self.control_mode})")
        t0 = time.time()
        try:
            from diffusers import LTXPipeline, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

            load_kwargs = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
            transformer = LTXVideoTransformer3DModel.from_pretrained(repo_id, subfolder="transformer", **load_kwargs)

            if self.is_low_vram and hasattr(transformer, "enable_layerwise_casting"):
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                )
                print("⚡ [FP8 LAYERWISE CASTING] Веса 3D-DiT сжаты в float8_e4m3fn!")

            self.pipe_t2v = LTXPipeline.from_pretrained(
                repo_id, text_encoder=None, tokenizer=None, transformer=transformer, torch_dtype=self.dtype
            )
            self.pipe_i2v = LTXImageToVideoPipeline(
                scheduler=self.pipe_t2v.scheduler,
                vae=self.pipe_t2v.vae,
                text_encoder=None,
                tokenizer=None,
                transformer=self.pipe_t2v.transformer,
            )

            self.pipe_t2v.set_progress_bar_config(disable=True)
            self.pipe_i2v.set_progress_bar_config(disable=True)

            if hasattr(self.pipe_t2v.vae, "enable_tiling"):
                self.pipe_t2v.vae.enable_tiling()
            if hasattr(self.pipe_t2v.vae, "enable_slicing"):
                self.pipe_t2v.vae.enable_slicing()

            self.pipe_t2v.transformer.to(self.device)
            self.pipe_t2v.vae.to(self.device)
            self.pipe_t2v.transformer.eval()
            self.pipe_t2v.vae.eval()

            self._install_ltx_prepare_latents_hook()

            gc.collect()
            torch.cuda.empty_cache()
            alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(f"✅ [VIDEO-RENDERER] Готово за {time.time() - t0:.1f}с! VRAM: {alloc_mb:.0f} МБ")
            print("=" * 80)
        except Exception as e:
            print(f"❌ [DiT Load Error]: {e}")

    @torch.no_grad()
    def _get_3d_s3_bases(self, n_vox: int, device: torch.device):
        step_seed = self.base_seed + self.rollout_step_idx * 997
        g1 = torch.Generator(device=device).manual_seed(101 + step_seed)
        g2 = torch.Generator(device=device).manual_seed(202 + step_seed)
        g3 = torch.Generator(device=device).manual_seed(303 + step_seed)
        g4 = torch.Generator(device=device).manual_seed(404 + step_seed)
        b1 = torch.randn((5, n_vox), generator=g1, device=device, dtype=torch.float32)
        b2 = torch.randn((5, n_vox), generator=g2, device=device, dtype=torch.float32)
        b3 = torch.randn((5, n_vox), generator=g3, device=device, dtype=torch.float32)
        b4 = torch.randn((5, n_vox), generator=g4, device=device, dtype=torch.float32)

        key_p = (device, self.base_seed)
        if key_p not in self._s3_basis_cache:
            gp = torch.Generator(device=device).manual_seed(707 + self.base_seed)
            proj_19200x3 = F.normalize(
                torch.randn((5, 19200, 3), generator=gp, device=device, dtype=torch.float32),
                p=2, dim=1
            )
            self._s3_basis_cache[key_p] = proj_19200x3
        return b1, b2, b3, b4, self._s3_basis_cache[key_p]

    @torch.no_grad()
    def _modulate_tensor_with_wm_s3(self, base_rand_tensor: torch.Tensor) -> torch.Tensor:
        stoch_act = float(np.clip(self.stochastic, 0.0, 1.0))
        stoch_unass = float(np.clip(self.unassigned_stochastic, 0.0, 1.0))
        rad = float(np.clip(self.wm_radius, 0.0, 1.0))

        if (stoch_act >= 0.999 and stoch_unass >= 0.999) or rad <= 0.001 or self.active_wm_5x5x32x120 is None:
            return base_rand_tensor

        device = base_rand_tensor.device
        dtype = base_rand_tensor.dtype
        target_shape = base_rand_tensor.shape
        B = target_shape[0]
        n_vox = int(np.prod(target_shape[1:]))

        b1, b2, b3, b4, proj_19200x3 = self._get_3d_s3_bases(n_vox, device)

        wm = self.active_wm_5x5x32x120.to(device=device, dtype=torch.float32)

        if self.smooth_wm_drift is None or self.smooth_wm_drift.shape != wm.shape:
            self.smooth_wm_drift = wm.clone()
        else:
            self.smooth_wm_drift.lerp_(wm, 0.08)

        reg_flags = list(self.active_region_flags)
        if not any(x > 0.5 for x in reg_flags):
            reg_flags[1] = 1.0

        active_mask_t = torch.tensor(reg_flags, device=device, dtype=torch.float32)
        raw_weights = torch.where(active_mask_t > 0.5, torch.tensor(3.0, device=device), torch.tensor(0.55, device=device))
        l2_weights_5 = (raw_weights / torch.norm(raw_weights, p=2)).view(5, 1)

        dynamic_wm_5x19200 = (wm - self.smooth_wm_drift * 0.85).reshape(5, 19200)
        wm_energy = float(torch.norm(dynamic_wm_5x19200).item())
        sig_gate = math.tanh(wm_energy * 5.0)

        wm_mean = dynamic_wm_5x19200.mean(dim=-1, keepdim=True)
        wm_std = dynamic_wm_5x19200.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5)
        whitened_5x19200 = (dynamic_wm_5x19200 - wm_mean) / wm_std

        if self.control_mode == "latent_s3_morph":
            reps = (n_vox // 19200) + 1
            wm_expanded = whitened_5x19200.repeat(1, reps)[:, :n_vox]
            scale_ang = math.pi * 1.25 * rad * sig_gate
            phi_a = wm_expanded * scale_ang
            phi_b = torch.roll(wm_expanded, shifts=128, dims=1) * scale_ang
            phi_c = torch.roll(wm_expanded, shifts=256, dims=1) * (scale_ang * 0.5)
        else:
            coords_5x3 = torch.bmm(whitened_5x19200.unsqueeze(1), proj_19200x3).squeeze(1) / math.sqrt(19200.0)
            c_std = coords_5x3.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-4)
            coords_norm = torch.tanh((coords_5x3 / c_std) * 0.85)

            if not hasattr(self, "_wm_accum_angles_5x3"):
                self._wm_accum_angles_5x3 = torch.zeros((5, 3), device=device, dtype=torch.float32)

            self._wm_accum_angles_5x3 = torch.remainder(
                self._wm_accum_angles_5x3 + coords_norm * (rad * sig_gate * 0.65),
                2.0 * math.pi
            )

            scale_ang = math.pi * rad * sig_gate
            phi_a = (coords_norm[:, 0:1] * scale_ang + self._wm_accum_angles_5x3[:, 0:1])
            phi_b = (coords_norm[:, 1:2] * scale_ang - self._wm_accum_angles_5x3[:, 1:2] * 0.7)
            phi_c = (coords_norm[:, 2:3] * (scale_ang * 0.6) + self._wm_accum_angles_5x3[:, 2:3] * 0.4)

        g_12 = torch.cos(phi_a) * b1 + torch.sin(phi_a) * b2
        g_34 = torch.cos(phi_b) * b3 + torch.sin(phi_b) * b4
        G_wm_5 = torch.cos(phi_c) * g_12 + torch.sin(phi_c) * g_34

        G_rand_1 = base_rand_tensor[0].reshape(1, n_vox).float()

        has_any_active = any(x > 0.5 for x in reg_flags)
        reg_stoch_base_5 = active_mask_t * stoch_act + (1.0 - active_mask_t) * (stoch_act if has_any_active else stoch_unass)
        eff_wm_blend_5 = ((1.0 - reg_stoch_base_5) * (0.25 + 0.75 * rad)).clamp(0.0, 1.0).view(5, 1)

        angle_w_5 = eff_wm_blend_5 * (math.pi * 0.5)
        reg_noise_5 = torch.sin(angle_w_5) * G_wm_5 + torch.cos(angle_w_5) * G_rand_1

        noise_flat = torch.sum(reg_noise_5 * l2_weights_5, dim=0, keepdim=True)
        noise_std = torch.std(noise_flat, unbiased=False).clamp(min=1e-4)
        noise_flat = (noise_flat - torch.mean(noise_flat)) / noise_std

        noise_shaped = noise_flat.view(1, *target_shape[1:])
        if B > 1:
            noise_shaped = noise_shaped.expand(B, *target_shape[1:])
        return noise_shaped.to(device=device, dtype=dtype)

    def _install_ltx_prepare_latents_hook(self):
        render_self = self

        orig_t2v_prep = self.pipe_t2v.prepare_latents

        def hooked_t2v_prepare_latents(pipe_self, *args, **kwargs):
            latents = orig_t2v_prep(*args, **kwargs)
            return render_self._modulate_tensor_with_wm_s3(latents)

        self.pipe_t2v.prepare_latents = types.MethodType(hooked_t2v_prepare_latents, self.pipe_t2v)

        orig_i2v_prep = self.pipe_i2v.prepare_latents

        def hooked_i2v_prepare_latents(pipe_self, *args, **kwargs):
            if render_self.last_clean_latent_5d is not None or render_self.custom_init_frame_gpu is not None:
                batch_size = 1
                num_channels = pipe_self.transformer.config.in_channels
                h_lat = render_self.height // pipe_self.vae_spatial_compression_ratio
                w_lat = render_self.width // pipe_self.vae_spatial_compression_ratio
                f_lat = (render_self.num_frames - 1) // pipe_self.vae_temporal_compression_ratio + 1

                shape = (batch_size, num_channels, f_lat, h_lat, w_lat)
                mask_shape = (batch_size, 1, f_lat, h_lat, w_lat)

                if render_self.last_clean_latent_5d is not None and render_self.last_clean_latent_5d.shape[3:] == (h_lat, w_lat):
                    init_lat = render_self.last_clean_latent_5d.clone().to(device=pipe_self.device, dtype=torch.float32)
                else:
                    enc_out = pipe_self.vae.encode(render_self.custom_init_frame_gpu.to(dtype=pipe_self.vae.dtype))
                    init_lat = enc_out.latent_dist.mode() if hasattr(enc_out, "latent_dist") else enc_out.latents
                    init_lat = pipe_self._normalize_latents(init_lat.float(), pipe_self.vae.latents_mean, pipe_self.vae.latents_std)

                if render_self.recalled_episodic_latent_5d is not None:
                    init_lat = stabilize_sliding_window_latents_gpu(
                        init_lat, render_self.recalled_episodic_latent_5d, factor=0.12
                    )

                # Только z[0] содержит 1-кадровый якорь, остальные f_lat-1 кадров чистые в шуме!
                init_latents = torch.zeros(shape, device=pipe_self.device, dtype=torch.float32)
                init_latents[:, :, :1] = init_lat[:, :, :1]

                conditioning_mask = torch.zeros(mask_shape, device=pipe_self.device, dtype=torch.float32)
                conditioning_mask[:, :, 0] = 1.0

                generator = kwargs.get("generator", None)
                if generator is None and len(args) > 8:
                    generator = args[8]

                noise = torch.randn(shape, generator=generator, device=pipe_self.device, dtype=torch.float32)
                cond_packed = pipe_self._pack_latents(
                    conditioning_mask, pipe_self.transformer_spatial_patch_size, pipe_self.transformer_temporal_patch_size
                ).squeeze(-1)
                init_packed = pipe_self._pack_latents(
                    init_latents, pipe_self.transformer_spatial_patch_size, pipe_self.transformer_temporal_patch_size
                )
                noise_packed = pipe_self._pack_latents(
                    noise, pipe_self.transformer_spatial_patch_size, pipe_self.transformer_temporal_patch_size
                )

                wm_noise_packed = render_self._modulate_tensor_with_wm_s3(noise_packed)
                mask_3d = cond_packed.unsqueeze(-1)
                latents_final = (init_packed * mask_3d + wm_noise_packed * (1.0 - mask_3d)).to(pipe_self.dtype)
                return latents_final, cond_packed.to(pipe_self.dtype)

            latents, conditioning_mask = orig_i2v_prep(*args, **kwargs)
            wm_latents = render_self._modulate_tensor_with_wm_s3(latents)
            mask_3d = conditioning_mask.unsqueeze(-1).to(latents.dtype)
            latents = latents * mask_3d + wm_latents * (1.0 - mask_3d)
            return latents, conditioning_mask

        self.pipe_i2v.prepare_latents = types.MethodType(hooked_i2v_prepare_latents, self.pipe_i2v)

    @torch.no_grad()
    def encode_prompt_with_mask(self, prompt_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        text_clean = (prompt_text or "").strip()
        text_inputs = self.tokenizer(
            [text_clean],
            padding="max_length",
            max_length=self.seq_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_attention_mask = text_inputs.attention_mask.bool().to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0].squeeze(0).float()
        return prompt_embeds, prompt_attention_mask

    @torch.no_grad()
    def set_custom_init_image_bytes(self, img_bytes: bytes):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            arr_np = np.array(pil_img, dtype=np.float32) / 127.5 - 1.0
            frame_gpu = torch.from_numpy(arr_np).to(self.device).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            self.custom_init_frame_gpu = frame_gpu.clone()
            enc_out = self.pipe_i2v.vae.encode(frame_gpu.to(dtype=self.pipe_i2v.vae.dtype))
            init_lat = enc_out.latent_dist.mode() if hasattr(enc_out, "latent_dist") else enc_out.latents
            self.last_clean_latent_5d = self.pipe_i2v._normalize_latents(
                init_lat.float(), self.pipe_i2v.vae.latents_mean, self.pipe_i2v.vae.latents_std
            ).clone()
            self.latest_end_latent_5d = self.last_clean_latent_5d.clone()
            self.recalled_episodic_latent_5d = None
            self.last_generated_frames = [np.array(pil_img, dtype=np.uint8)] * 8
            self.rollout_step_idx = 0
            print(f"🖼️ [RENDERER] Стартовая сцена закодирована в чистый латент VRAM ({self.width}x{self.height})!")
        except Exception as e:
            print(f"⚠️ [Init Image Error]: {e}")

    def reset_genesis(self):
        if self.custom_init_frame_gpu is not None:
            enc_out = self.pipe_i2v.vae.encode(self.custom_init_frame_gpu.to(dtype=self.pipe_i2v.vae.dtype))
            init_lat = enc_out.latent_dist.mode() if hasattr(enc_out, "latent_dist") else enc_out.latents
            self.last_clean_latent_5d = self.pipe_i2v._normalize_latents(
                init_lat.float(), self.pipe_i2v.vae.latents_mean, self.pipe_i2v.vae.latents_std
            ).clone()
        else:
            self.last_clean_latent_5d = None
        self.latest_end_latent_5d = None
        self.recalled_episodic_latent_5d = None
        self.last_generated_frames = []
        self.rollout_step_idx = 0
        if hasattr(self, "_wm_accum_angles_5x3"):
            self._wm_accum_angles_5x3.zero_()

    @torch.no_grad()
    def _decode_latents_on_gpu(self, pipe_obj, unpacked_5d: torch.Tensor, gen_arg: torch.Generator) -> list[np.ndarray]:
        """
        Декодирует латент с last_frame_fix, затем перекодирует ровно 1 последний чистый кадр
        в `last_clean_latent_5d` (1-кадровый корень z[0]) вместо 8-кадрового `unpacked_5d[:, :, -1:]`.
        """
        unpacked_5d = stabilize_sliding_window_latents_gpu(
            unpacked_5d.float(), self.last_clean_latent_5d, factor=0.20
        )

        denorm = pipe_obj._denormalize_latents(
            unpacked_5d, pipe_obj.vae.latents_mean, pipe_obj.vae.latents_std, pipe_obj.vae.config.scaling_factor
        ).to(pipe_obj.vae.dtype)

        tsf = int(pipe_obj.vae_temporal_compression_ratio)
        denorm_padded = torch.cat([denorm, denorm[:, :, -1:].contiguous()], dim=2)

        if getattr(pipe_obj.vae.config, "timestep_conditioning", False):
            dec_t = 0.05
            t_tensor = torch.tensor([dec_t], device=self.device, dtype=denorm_padded.dtype)
        else:
            t_tensor = None

        decoded_video = pipe_obj.vae.decode(denorm_padded, t_tensor, return_dict=False)[0]
        decoded_video = decoded_video[:, :, :-tsf, :, :]

        # Кодируем строго последний пиксельный кадр [-1:] обратно в 1-кадровый z[0] на GPU (~1.5 мс),
        # чтобы следующие окна не получали сжатые 8 прошлых кадров в слоте t=0:
        last_frame_gpu = decoded_video[:, :, -1:].clamp(-1.0, 1.0).to(dtype=pipe_obj.vae.dtype)
        enc_out = pipe_obj.vae.encode(last_frame_gpu)
        raw_lat = enc_out.latent_dist.mode() if hasattr(enc_out, "latent_dist") else enc_out.latents
        self.last_clean_latent_5d = pipe_obj._normalize_latents(
            raw_lat.float(), pipe_obj.vae.latents_mean, pipe_obj.vae.latents_std, pipe_obj.vae.config.scaling_factor
        ).detach().clone()
        self.latest_end_latent_5d = self.last_clean_latent_5d

        post_vid = pipe_obj.video_processor.postprocess_video(decoded_video, output_type="pt")
        frames_uint8 = (post_vid[0].permute(0, 2, 3, 1).float().clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        return [frames_uint8[i] for i in range(frames_uint8.shape[0])]

    @torch.no_grad()
    def render_rollout(
        self,
        final_embeds_1x128x4096: torch.Tensor,
        neg_embeds_1x128x4096: torch.Tensor,
        pos_attention_mask: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        wall_keep: float = 0.0,
        anchor_advance_rate: float = 1.0,
        stochastic: float | None = None,
        unassigned_stochastic: float | None = None,
        wm_radius: float | None = None,
        wm_5x5x32x120: Optional[torch.Tensor] = None,
        active_region_flags: Optional[List[float]] = None,
        seed_val: int = 42,
    ) -> list[np.ndarray]:
        h, w = self.height, self.width
        if stochastic is not None:
            self.stochastic = float(np.clip(stochastic, 0.0, 1.0))
        if unassigned_stochastic is not None:
            self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        if wm_radius is not None:
            self.wm_radius = float(np.clip(wm_radius, 0.0, 1.0))
        if wm_5x5x32x120 is not None:
            self.active_wm_5x5x32x120 = wm_5x5x32x120

        self.vram_bus.try_attach_to_existing_vram()
        if self.vram_bus.attached_pid > 0:
            self.active_wm_5x5x32x120 = self.vram_bus.wm_field_5x5x32x120
            meta = self.vram_bus.meta_64
            monty_code = int(round(float(meta[27].item())))
            unass_m = 0.0 if monty_code == 0 else 1.0
            self.active_region_flags = [
                1.0 if float(meta[21 + i].item()) > 0.5 else unass_m
                for i in range(5)
            ]
        elif active_region_flags is not None and len(active_region_flags) == 5:
            self.active_region_flags = [float(x) for x in active_region_flags]

        has_any_active = any(x > 0.5 for x in self.active_region_flags)
        max_stoch = max(self.stochastic if has_any_active else 0.0, self.unassigned_stochastic)

        self.active_seed_val = int(seed_val)
        if max_stoch <= 0.001:
            eff_seed = seed_val
        else:
            eff_seed = seed_val + self.rollout_step_idx * 19 + int(time.time() * 1000.0 * max_stoch) % 10000
        gen_arg = torch.Generator(device=self.device).manual_seed(eff_seed)

        cond_pos = final_embeds_1x128x4096.to(device=self.device, dtype=self.dtype)
        cond_neg = neg_embeds_1x128x4096.to(device=self.device, dtype=self.dtype)
        p_mask = pos_attention_mask.to(self.device) if pos_attention_mask is not None else self.last_pos_mask
        n_mask = neg_attention_mask.to(self.device) if neg_attention_mask is not None else self.last_neg_mask

        f_lat = (self.num_frames - 1) // self.pipe_t2v.vae_temporal_compression_ratio + 1
        h_lat = h // self.pipe_t2v.vae_spatial_compression_ratio
        w_lat = w // self.pipe_t2v.vae_spatial_compression_ratio

        frames_out = []
        try:
            if self.last_clean_latent_5d is None and self.custom_init_frame_gpu is None:
                pipe_out = self.pipe_t2v(
                    prompt_embeds=cond_pos,
                    prompt_attention_mask=p_mask,
                    negative_prompt_embeds=cond_neg,
                    negative_prompt_attention_mask=n_mask,
                    width=w,
                    height=h,
                    num_frames=self.num_frames,
                    num_inference_steps=max(16, self.genesis_steps),
                    guidance_scale=self.cfg_scale,
                    guidance_rescale=0.7,
                    generator=gen_arg,
                    output_type="latent"
                )
                unpacked_5d = self.pipe_t2v._unpack_latents(
                    pipe_out.frames, f_lat, h_lat, w_lat,
                    self.pipe_t2v.transformer_spatial_patch_size,
                    self.pipe_t2v.transformer_temporal_patch_size
                )
                all_frames = self._decode_latents_on_gpu(self.pipe_t2v, unpacked_5d, gen_arg)
                frames_out = all_frames
                self.last_generated_frames = frames_out

            else:
                dummy_tensor = torch.zeros((1, 3, h, w), device=self.device, dtype=self.dtype)
                pipe_out = self.pipe_i2v(
                    image=dummy_tensor,
                    prompt_embeds=cond_pos,
                    prompt_attention_mask=p_mask,
                    negative_prompt_embeds=cond_neg,
                    negative_prompt_attention_mask=n_mask,
                    width=w,
                    height=h,
                    num_frames=self.num_frames,
                    num_inference_steps=max(12, self.steps),
                    guidance_scale=self.cfg_scale,
                    guidance_rescale=0.7,
                    generator=gen_arg,
                    output_type="latent"
                )
                unpacked_5d = self.pipe_i2v._unpack_latents(
                    pipe_out.frames, f_lat, h_lat, w_lat,
                    self.pipe_i2v.transformer_spatial_patch_size,
                    self.pipe_t2v.transformer_temporal_patch_size
                )
                all_frames = self._decode_latents_on_gpu(self.pipe_i2v, unpacked_5d, gen_arg)
                frames_out = all_frames[1:] if len(all_frames) > 1 else all_frames
                self.last_generated_frames = frames_out

            self.rollout_step_idx += 1

        except torch.cuda.OutOfMemoryError:
            print("⚠️ [VRAM GUARD] Очистка кэша CUDA...")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"⚠️ [Renderer Execution Warning]: {e}")
            traceback.print_exc()

        if not frames_out:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            frames_out = [blank] * 8

        return frames_out
