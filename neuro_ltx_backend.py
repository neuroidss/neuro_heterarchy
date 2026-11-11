#!/usr/bin/env python3
"""
neuro_ltx_backend.py — Аппаратный бэкенд LTX-Video с оптимизацией VRAM для 12GB GPU
"""

from __future__ import annotations
import os
import sys
import gc
from pathlib import Path
import numpy as np
import torch

# Защита от фрагментации VRAM на 12GB GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from neuro_heterarchy_core import DEVICE
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REPO = "Lightricks/LTX-Video"
BASE_CACHE_DIR = (
    Path("/workspace/hf_cache")
    if Path("/workspace/hf_cache").exists()
    else Path(os.path.expanduser("~/.cache/huggingface"))
)
QUANT_T5_DIR = BASE_CACHE_DIR / "neuro_quant_cache" / "t5_xxl_4bit"


def is_saved_t5_valid(d: Path) -> bool:
    if not d.exists() or not (d / "config.json").exists():
        return False
    return (
        (d / "model.safetensors").exists()
        or (d / "model.safetensors.index.json").exists()
        or (d / "pytorch_model.bin").exists()
        or (d / "pytorch_model.bin.index.json").exists()
    )


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


def rescale_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_rescale: float = 0.7,
) -> torch.Tensor:
    if guidance_rescale <= 0.0:
        return noise_cfg
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(
        dim=list(range(1, noise_cfg.ndim)), keepdim=True
    ).clamp(min=1e-6)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return (
        guidance_rescale * noise_pred_rescaled
        + (1.0 - guidance_rescale) * noise_cfg
    )


class LTXModelBackend:
    """Изолированный контейнер LTX-Video, оптимизированный под 12GB VRAM."""

    def __init__(self, device: torch.device | None = None):
        from diffusers import LTXPipeline, LTXVideoTransformer3DModel
        from transformers import AutoTokenizer, BitsAndBytesConfig, T5EncoderModel

        self.device = device or DEVICE
        self.dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        high_vram = vram_gb > 22.0

        if high_vram:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_REPO, subfolder="tokenizer"
            )
            self.text_encoder = T5EncoderModel.from_pretrained(
                MODEL_REPO,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                device_map="cuda",
                low_cpu_mem_usage=True,
            )
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                MODEL_REPO,
                subfolder="transformer",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
        else:
            if is_saved_t5_valid(QUANT_T5_DIR):
                self.tokenizer = AutoTokenizer.from_pretrained(str(QUANT_T5_DIR))
                self.text_encoder = T5EncoderModel.from_pretrained(
                    str(QUANT_T5_DIR),
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
            else:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_REPO, subfolder="tokenizer"
                )
                self.text_encoder = T5EncoderModel.from_pretrained(
                    MODEL_REPO,
                    subfolder="text_encoder",
                    quantization_config=bnb_cfg,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                QUANT_T5_DIR.mkdir(parents=True, exist_ok=True)
                self.text_encoder.save_pretrained(str(QUANT_T5_DIR))
                self.tokenizer.save_pretrained(str(QUANT_T5_DIR))

            transformer = LTXVideoTransformer3DModel.from_pretrained(
                MODEL_REPO,
                subfolder="transformer",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(transformer, "enable_layerwise_casting"):
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                )

        self.text_encoder.eval()

        self.pipe = LTXPipeline.from_pretrained(
            MODEL_REPO,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            torch_dtype=self.dtype,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.transformer.to(self.device).eval()
        self.pipe.vae.to(self.device).eval()

        # КРИТИЧЕСКИ ВАЖНО: Активируем тайлинг для 12GB GPU с заниженными порогами!
        # По умолчанию порог 512x512 отключал тайлинг для 640x448, вызывая OOM.
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling(
                tile_sample_min_height=256,
                tile_sample_min_width=256,
                tile_sample_stride_height=192,
                tile_sample_stride_width=192,
            )
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        torch.cuda.empty_cache()

    @property
    def transformer(self):
        return self.pipe.transformer

    @property
    def vae(self):
        return self.pipe.vae

    @property
    def scheduler(self):
        return self.pipe.scheduler

    @property
    def spatial_ratio(self) -> int:
        return int(self.pipe.vae_spatial_compression_ratio)

    @property
    def temporal_ratio(self) -> int:
        return int(self.pipe.vae_temporal_compression_ratio)

    @torch.no_grad()
    def encode_semantics(
        self, text: str, max_len: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(self.device)
        mask = inputs.attention_mask.bool().to(self.device)
        embeds = self.text_encoder(ids, attention_mask=mask)[0].to(self.dtype)
        return embeds, mask

    @torch.no_grad()
    def set_scheduler_timesteps(self, steps: int, video_seq_len: int):
        sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()
        mu = calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        return self.scheduler.timesteps

    @torch.no_grad()
    def decode_latents_chunk(
        self,
        latents_5d: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> list[np.ndarray]:
        # Очищаем кэш перед вызовом VAE декодера
        gc.collect()
        torch.cuda.empty_cache()

        denorm = self.pipe._denormalize_latents(
            latents_5d.to(dtype=torch.float32),
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(dtype=self.vae.dtype)

        if getattr(self.vae.config, "timestep_conditioning", False):
            decode_t = 0.05
            decode_noise_scale = 0.025
            t_tensor = torch.tensor(
                [decode_t], device=self.device, dtype=denorm.dtype
            )
            noise = torch.randn(
                denorm.shape,
                generator=generator,
                device=self.device,
                dtype=denorm.dtype,
            )
            denorm = (1.0 - decode_noise_scale) * denorm + decode_noise_scale * noise
        else:
            t_tensor = None

        decoded = self.vae.decode(denorm, t_tensor, return_dict=False)[0]
        post = self.pipe.video_processor.postprocess_video(
            decoded, output_type="pt"
        )
        frames_uint8 = (
            (post[0].permute(0, 2, 3, 1).float().clamp(0.0, 1.0) * 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        del denorm, decoded, post
        gc.collect()
        torch.cuda.empty_cache()

        return [frames_uint8[i] for i in range(frames_uint8.shape[0])]
