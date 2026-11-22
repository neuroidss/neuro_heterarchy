#!/usr/bin/env python3
"""
neuro_ltx_backend.py — Аппаратный бэкенд LTX-Video (0.9.8-2B-Distilled NF4/INT8/FP8/BF16 + Base 0.9.0)
С автоматическим взаимоисключением T5 <-> VAE в VRAM, безопасным тайлингом VAE и
извлечением эмпирического конуса Кента T5-XXL без хардкода слов.
"""

from __future__ import annotations
import os
import sys
import gc
import math
import threading
from pathlib import Path
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from neuro_heterarchy_core import DEVICE
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REPO = "Lightricks/LTX-Video"
DISTILLED_CKPT_FILENAME = "ltxv-2b-0.9.8-distilled.safetensors"
DISTILLED_SIGMAS_8 = [1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219]

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


def build_distilled_sigmas(steps: int) -> list[float]:
    s = max(2, int(steps))
    if s == len(DISTILLED_SIGMAS_8):
        return list(DISTILLED_SIGMAS_8)
    xp = np.linspace(0.0, 1.0, len(DISTILLED_SIGMAS_8))
    x_new = np.linspace(0.0, 1.0, s)
    return [float(v) for v in np.interp(x_new, xp, DISTILLED_SIGMAS_8)]


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


@torch.no_grad()
def quantize_model_inplace_bnb(
    model: torch.nn.Module,
    quant_mode: str,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> int:
    import bitsandbytes as bnb

    q_count = 0

    def _replace(parent: torch.nn.Module):
        nonlocal q_count
        for name, child in list(parent.named_children()):
            if (
                isinstance(child, torch.nn.Linear)
                and child.in_features >= 512
                and child.out_features >= 512
            ):
                has_bias = child.bias is not None
                if quant_mode == "int8":
                    q_lin = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=has_bias,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )
                    q_lin.weight = bnb.nn.Int8Params(
                        child.weight.data.to("cpu", dtype=torch.float16),
                        requires_grad=False,
                        has_fp16_weights=False,
                    )
                else:
                    q_lin = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=has_bias,
                        compute_dtype=compute_dtype,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    q_lin.weight = bnb.nn.Params4bit(
                        child.weight.data.to("cpu"),
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                if has_bias:
                    q_lin.bias = torch.nn.Parameter(
                        child.bias.data.to(device=device, dtype=compute_dtype),
                        requires_grad=False,
                    )
                q_lin.to(device)
                setattr(parent, name, q_lin)
                del child
                q_count += 1
            else:
                _replace(child)

    _replace(model)

    for mod in model.modules():
        if isinstance(mod, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            continue
        for _, p in mod.named_parameters(recurse=False):
            if p is not None and p.device != device:
                p.data = p.data.to(device=device)
        for b_name, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.device != device:
                mod.register_buffer(b_name, b.to(device=device))

    gc.collect()
    torch.cuda.empty_cache()
    return q_count


class LTXModelBackend:
    """Контейнер весов LTX-Video (0.9.8-2B-Distilled или Base 0.9.0)."""

    def __init__(
        self,
        device: torch.device | None = None,
        quant_mode: str = "fp8",
        vae_tile_min_px: int = 512,
        allow_t5_offload: bool = False,
        model_variant: str = "distilled",
    ):
        from diffusers import (
            AutoencoderKLLTXVideo,
            FlowMatchEulerDiscreteScheduler,
            LTXPipeline,
            LTXVideoTransformer3DModel,
        )
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, BitsAndBytesConfig, T5EncoderModel

        self.device = device or DEVICE
        self.quant_mode = str(quant_mode).lower().strip()
        self.allow_t5_offload = bool(allow_t5_offload)
        self.model_variant = str(model_variant).lower().strip()
        self.is_distilled = self.model_variant in ("distilled", "0.9.8", "0.9.8-distilled")
        self.dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        high_vram = (vram_gb > 22.0) and (self.quant_mode == "auto")
        if self.quant_mode == "auto":
            self.quant_mode = "bf16" if high_vram else "fp8"

        if high_vram:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder="tokenizer")
            self.text_encoder = T5EncoderModel.from_pretrained(
                MODEL_REPO,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                device_map="cuda",
                low_cpu_mem_usage=True,
            )
        else:
            if is_saved_t5_valid(QUANT_T5_DIR):
                self.tokenizer = AutoTokenizer.from_pretrained(str(QUANT_T5_DIR))
                self.text_encoder = T5EncoderModel.from_pretrained(
                    str(QUANT_T5_DIR),
                    torch_dtype=self.dtype,
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
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder="tokenizer")
                self.text_encoder = T5EncoderModel.from_pretrained(
                    MODEL_REPO,
                    subfolder="text_encoder",
                    quantization_config=bnb_cfg,
                    torch_dtype=self.dtype,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                QUANT_T5_DIR.mkdir(parents=True, exist_ok=True)
                self.text_encoder.save_pretrained(str(QUANT_T5_DIR))
                self.tokenizer.save_pretrained(str(QUANT_T5_DIR))

        self.text_encoder.eval()
        self.t5_offloaded = False
        self.vae_on_gpu = False

        transformer = None
        if self.is_distilled:
            print(f"📥 [LTX 0.9.8 DISTILLED] Загрузка {DISTILLED_CKPT_FILENAME}...")
            ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=DISTILLED_CKPT_FILENAME)
            transformer = LTXVideoTransformer3DModel.from_single_file(
                ckpt_path, torch_dtype=self.dtype
            )
            if self.quant_mode in ("int8", "nf4", "4bit"):
                try:
                    n_q = quantize_model_inplace_bnb(
                        transformer,
                        quant_mode=self.quant_mode,
                        compute_dtype=self.dtype,
                        device=self.device,
                    )
                    print(
                        f"⚡ [LTX 0.9.8 {self.quant_mode.upper()}] "
                        f"Послойно квантовано {n_q} линейных слоёв на CUDA!"
                    )
                except Exception as e:
                    print(f"⚠️ [Distilled BnB -> FP8 fallback]: {e}")
                    self.quant_mode = "fp8"
                    if hasattr(transformer, "enable_layerwise_casting"):
                        transformer.enable_layerwise_casting(
                            storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                        )
                    transformer.to(self.device)
            else:
                if self.quant_mode in ("fp8", "8bit") and hasattr(transformer, "enable_layerwise_casting"):
                    transformer.enable_layerwise_casting(
                        storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                    )
                transformer.to(self.device)

            vae = AutoencoderKLLTXVideo.from_single_file(ckpt_path, torch_dtype=self.dtype)
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=False, shift=1.0, shift_terminal=None
            )
            self.pipe = LTXPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                transformer=transformer,
            )
        else:
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                MODEL_REPO,
                subfolder="transformer",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if self.quant_mode in ("int8", "nf4", "4bit"):
                try:
                    n_q = quantize_model_inplace_bnb(
                        transformer,
                        quant_mode=self.quant_mode,
                        compute_dtype=self.dtype,
                        device=self.device,
                    )
                    print(
                        f"⚡ [LTX BASE {self.quant_mode.upper()}] "
                        f"Послойно квантовано {n_q} линейных слоёв на CUDA!"
                    )
                except Exception as e:
                    print(f"⚠️ [Base BnB -> FP8 fallback]: {e}")
                    self.quant_mode = "fp8"
                    if hasattr(transformer, "enable_layerwise_casting"):
                        transformer.enable_layerwise_casting(
                            storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                        )
                    transformer.to(self.device)
            else:
                if self.quant_mode in ("fp8", "8bit") and hasattr(transformer, "enable_layerwise_casting"):
                    transformer.enable_layerwise_casting(
                        storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                    )
                transformer.to(self.device)

            self.pipe = LTXPipeline.from_pretrained(
                MODEL_REPO,
                text_encoder=None,
                tokenizer=None,
                transformer=transformer,
                torch_dtype=self.dtype,
            )
            self.pipe.scheduler.config.shift_terminal = None

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.transformer.eval()
        self.pipe.vae.eval()

        if self.allow_t5_offload:
            self.pipe.vae.to("cpu")
            self.vae_on_gpu = False
        else:
            self.pipe.vae.to(self.device)
            self.vae_on_gpu = True

        # Не режем 384x256 / 512x384 на 256px тайлы, чтобы исключить швы в авторегрессии:
        eff_tile_px = max(512, int(vae_tile_min_px)) if vram_gb > 10.0 else max(256, int(vae_tile_min_px))
        tile_min = max(256, eff_tile_px)
        tile_stride = max(192, tile_min - 64)
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling(
                tile_sample_min_height=tile_min,
                tile_sample_min_width=tile_min,
                tile_sample_stride_height=tile_stride,
                tile_sample_stride_width=tile_stride,
            )
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        self._local_prompt_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._t5_manifold_cache: dict[int, dict[str, torch.Tensor]] = {}

        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated(0) / (1024**3)
            print(
                f"📊 [VRAM USAGE] {used_gb:.2f} GB / {vram_gb:.1f} GB "
                f"(variant={'0.9.8-distilled' if self.is_distilled else '0.9.0-base'}, "
                f"quant={self.quant_mode}, vae_tile={tile_min}px, vae_on_gpu={self.vae_on_gpu})"
            )

    @torch.no_grad()
    def _move_vae_to_cpu(self):
        if hasattr(self, "pipe") and self.pipe is not None and self.vae_on_gpu:
            self.pipe.vae.to("cpu")
            self.vae_on_gpu = False
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _move_vae_to_gpu(self):
        if hasattr(self, "pipe") and self.pipe is not None and not self.vae_on_gpu:
            self.pipe.vae.to(self.device)
            self.vae_on_gpu = True

    @torch.no_grad()
    def offload_t5_to_cpu(self, force: bool = False) -> float:
        if not self.allow_t5_offload and not force:
            return torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
        if self.text_encoder is not None and not self.t5_offloaded:
            try:
                del self.text_encoder
                self.text_encoder = None
                self.t5_offloaded = True
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._move_vae_to_gpu()
        used_gb = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
        print(f"🧹 [T5 -> VAE SWAP] T5 выгружен, VAE на CUDA -> занято {used_gb:.2f} GB.", flush=True)
        return used_gb

    @torch.no_grad()
    def prepare_for_video_generation(self):
        if self.allow_t5_offload and self.text_encoder is not None:
            self.offload_t5_to_cpu(force=False)
        elif not self.vae_on_gpu:
            self._move_vae_to_gpu()

    @torch.no_grad()
    def _ensure_t5_loaded(self):
        if self.text_encoder is None:
            from transformers import T5EncoderModel

            if self.allow_t5_offload:
                self._move_vae_to_cpu()
                used_before = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
                print(
                    f"🔄 [VAE -> T5 SWAP] VAE временно на CPU (VRAM={used_before:.2f} GB), "
                    f"загрузка T5-XXL ({self.dtype})...",
                    flush=True,
                )

            self.text_encoder = T5EncoderModel.from_pretrained(
                str(QUANT_T5_DIR),
                torch_dtype=self.dtype,
                device_map="cuda",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            ).eval()
            self.t5_offloaded = False
            gc.collect()
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
    def get_t5_output_manifold(self, max_rank: int = 128) -> dict[str, torch.Tensor]:
        """
        Извлекает истинный анизотропный конус Кента (mu, V, sigma, mean_norm) из выходов
        T5-XXL по реальным SentencePiece ID словаря модели (0 слов в коде и конфигах).
        """
        r_req = int(max_rank)
        if r_req in self._t5_manifold_cache:
            return self._t5_manifold_cache[r_req]

        cache_file = QUANT_T5_DIR / f"t5_output_manifold_{r_req}x4096.pt"
        if cache_file.exists():
            raw = torch.load(str(cache_file), map_location="cpu")
            out = {
                k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)).to(device=self.device, dtype=torch.float32)
                for k, v in raw.items()
            }
            self._t5_manifold_cache[r_req] = out
            return out

        self._ensure_t5_loaded()
        vocab_sz = int(getattr(self.tokenizer, "vocab_size", 32100))
        hi_id = min(vocab_sz - 100, 31900)
        g = torch.Generator(device="cpu").manual_seed(2026)
        rand_ids = torch.randint(100, hi_id, (96, 11), generator=g, dtype=torch.long)
        eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else 1
        eos_col = torch.full((96, 1), eos_id, dtype=torch.long)
        input_ids = torch.cat([rand_ids, eos_col], dim=1).to(self.device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)

        hidden = self.text_encoder(input_ids, attention_mask=attn_mask)[0][:, :-1, :].float()
        X = hidden.reshape(-1, hidden.shape[-1])
        mean_norm = float(torch.norm(X, p=2, dim=-1).mean().item())
        mu = X.mean(dim=0)
        X_c = X - mu.unsqueeze(0)
        _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
        R = min(r_req, int(Vh.shape[0]))
        sigma_R = S[:R] / math.sqrt(max(1, X_c.shape[0] - 1))

        cpu_payload = {
            "mu": mu.detach().cpu(),
            "V": Vh[:R].contiguous().detach().cpu(),
            "sigma": sigma_R.contiguous().detach().cpu(),
            "mean_norm": torch.tensor([mean_norm], dtype=torch.float32),
        }
        QUANT_T5_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_payload, str(cache_file))

        out = {k: v.to(device=self.device, dtype=torch.float32) for k, v in cpu_payload.items()}
        self._t5_manifold_cache[r_req] = out
        print(f"🧭 [T5-XXL MANIFOLD] Извлечён истинный выходной конус T5-XXL (R={R}, norm={mean_norm:.2f}).", flush=True)
        return out

    @torch.no_grad()
    def encode_semantics(
        self, text: str, max_len: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(text), int(max_len))
        if key not in self._local_prompt_cache:
            if len(self._local_prompt_cache) > 1024:
                self._local_prompt_cache.clear()
            self._ensure_t5_loaded()
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
            self._local_prompt_cache[key] = (embeds, mask)
        return self._local_prompt_cache[key]

    @torch.no_grad()
    def set_scheduler_timesteps(self, steps: int, video_seq_len: int):
        if self.is_distilled:
            sigmas = build_distilled_sigmas(steps)
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
            return self.scheduler.timesteps

        min_sigma = min(0.10, 1.0 / max(2, steps))
        sigmas = np.linspace(1.0, min_sigma, steps).tolist()
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
    def encode_frames_to_latent(self, frames_rgb: list[np.ndarray] | np.ndarray) -> torch.Tensor:
        self.prepare_for_video_generation()
        if isinstance(frames_rgb, list):
            arr = np.stack(frames_rgb, axis=0)
        elif frames_rgb.ndim == 3:
            arr = frames_rgb[None, ...]
        else:
            arr = frames_rgb

        tsf = int(self.temporal_ratio)
        valid_n = ((arr.shape[0] - 1) // tsf) * tsf + 1
        arr = arr[-valid_n:]

        vid_t = (
            torch.from_numpy(np.ascontiguousarray(arr, dtype=np.uint8))
            .to(device=self.device, dtype=torch.float32)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )
        vid_norm = (vid_t / 127.5 - 1.0).to(dtype=self.vae.dtype)
        enc_out = self.vae.encode(vid_norm)
        raw_lat = (
            enc_out.latent_dist.mode()
            if hasattr(enc_out, "latent_dist")
            else enc_out.latents
        )
        norm_lat = self.pipe._normalize_latents(
            raw_lat.float(),
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )
        return norm_lat.to(dtype=torch.float32)

    @torch.no_grad()
    def encode_frame_to_latent(self, frame_rgb: np.ndarray) -> torch.Tensor:
        return self.encode_frames_to_latent([frame_rgb])

    @torch.no_grad()
    def decode_latents_chunk(
        self,
        latents_5d: torch.Tensor,
        generator: torch.Generator | None = None,
        decode_t: float = 0.05,
        decode_noise_scale: float = 0.0,
        last_frame_fix: bool = True,
    ) -> list[np.ndarray]:
        self.prepare_for_video_generation()
        denorm = self.pipe._denormalize_latents(
            latents_5d.to(dtype=torch.float32),
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(dtype=self.vae.dtype)

        tsf = int(self.temporal_ratio)
        if last_frame_fix:
            denorm = torch.cat([denorm, denorm[:, :, -1:].contiguous()], dim=2)

        if getattr(self.vae.config, "timestep_conditioning", False):
            t_tensor = torch.tensor(
                [float(decode_t)], device=self.device, dtype=denorm.dtype
            )
            if decode_noise_scale > 0.0:
                noise = torch.randn(
                    denorm.shape,
                    generator=generator,
                    device=self.device,
                    dtype=denorm.dtype,
                )
                denorm = (1.0 - float(decode_noise_scale)) * denorm + float(
                    decode_noise_scale
                ) * noise
        else:
            t_tensor = None

        decoded = self.vae.decode(denorm, t_tensor, return_dict=False)[0]
        if last_frame_fix and decoded.shape[2] > tsf:
            decoded = decoded[:, :, :-tsf, :, :]

        post = self.pipe.video_processor.postprocess_video(decoded, output_type="pt")
        frames_uint8 = (
            (post[0].permute(0, 2, 3, 1).float().clamp(0.0, 1.0) * 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return [frames_uint8[i] for i in range(frames_uint8.shape[0])]


class _TransformerConfigProxy:
    def __init__(self, in_channels: int = 128):
        self.in_channels = in_channels


class _RemoteTransformerProxy:
    def __init__(
        self,
        conn,
        device: torch.device,
        dtype: torch.dtype,
        in_channels: int = 128,
        rpc_lock: threading.RLock | None = None,
    ):
        self.conn = conn
        self.device = device
        self.dtype = dtype
        self.config = _TransformerConfigProxy(in_channels=in_channels)
        self._rpc_lock = rpc_lock or threading.RLock()

    @staticmethod
    def _to_np_f16(t: torch.Tensor) -> np.ndarray:
        return t.detach().to(torch.float16).cpu().numpy()

    @torch.no_grad()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: tuple[float, float, float],
        return_dict: bool = False,
    ):
        payload = {
            "cmd": "transformer_forward",
            "hidden_states": self._to_np_f16(hidden_states),
            "encoder_hidden_states": self._to_np_f16(encoder_hidden_states),
            "timestep": timestep.detach().float().cpu().numpy(),
            "encoder_attention_mask": encoder_attention_mask.detach().cpu().numpy(),
            "num_frames": int(num_frames),
            "height": int(height),
            "width": int(width),
            "rope_interpolation_scale": tuple(float(x) for x in rope_interpolation_scale),
        }
        with self._rpc_lock:
            self.conn.send(payload)
            resp = self.conn.recv()
        if not isinstance(resp, dict) or resp.get("status") != "ok" or "sample" not in resp:
            err_detail = resp.get("error", list(resp.keys())) if isinstance(resp, dict) else resp
            raise RuntimeError(f"Remote transformer error: {err_detail}")
        sample = torch.from_numpy(resp["sample"]).to(device=self.device, dtype=self.dtype)
        return (sample,)


class _LocalLatentPackerPipe:
    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size, _, num_frames, height, width = latents.shape
        latents = latents.reshape(
            batch_size, -1, num_frames // patch_size_t, patch_size_t,
            height // patch_size, patch_size, width // patch_size, patch_size,
        )
        return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int,
        patch_size: int = 1, patch_size_t: int = 1,
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size
        )
        return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class RemoteLTXModelBackend:
    """Потокобезопасный клиентский прокси к серверу весов (video_world_server.py)."""

    def __init__(self, conn, ping_info: dict, device: torch.device | None = None):
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        self.conn = conn
        self._rpc_lock = threading.RLock()
        self.device = device or DEVICE
        self.dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self._spatial_ratio = int(ping_info.get("spatial_ratio", 32))
        self._temporal_ratio = int(ping_info.get("temporal_ratio", 8))
        self.is_distilled = bool(ping_info.get("is_distilled", False))
        in_channels = int(ping_info.get("in_channels", 128))

        self.pipe = _LocalLatentPackerPipe()
        self.transformer = _RemoteTransformerProxy(
            conn=self.conn,
            device=self.device,
            dtype=self.dtype,
            in_channels=in_channels,
            rpc_lock=self._rpc_lock,
        )
        if self.is_distilled:
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=False, shift=1.0, shift_terminal=None
            )
        else:
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                base_image_seq_len=int(ping_info.get("base_image_seq_len", 256)),
                max_image_seq_len=int(ping_info.get("max_image_seq_len", 4096)),
                base_shift=float(ping_info.get("base_shift", 0.5)),
                max_shift=float(ping_info.get("max_shift", 1.15)),
                shift_terminal=None,
            )
        self._local_prompt_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._t5_manifold_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._decode_call_idx = 0

    def _rpc(self, payload: dict) -> dict:
        with self._rpc_lock:
            self.conn.send(payload)
            return self.conn.recv()

    @property
    def spatial_ratio(self) -> int:
        return self._spatial_ratio

    @property
    def temporal_ratio(self) -> int:
        return self._temporal_ratio

    def offload_t5_to_cpu(self) -> float:
        try:
            resp = self._rpc({"cmd": "evict_t5_to_cpu"})
            return float(resp.get("vram_gb", 0.0))
        except Exception:
            return 0.0

    @torch.no_grad()
    def get_t5_output_manifold(self, max_rank: int = 128) -> dict[str, torch.Tensor]:
        r_req = int(max_rank)
        if r_req in self._t5_manifold_cache:
            return self._t5_manifold_cache[r_req]

        cache_file = QUANT_T5_DIR / f"t5_output_manifold_{r_req}x4096.pt"
        if cache_file.exists():
            raw = torch.load(str(cache_file), map_location="cpu")
            out = {
                k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)).to(device=self.device, dtype=torch.float32)
                for k, v in raw.items()
            }
            self._t5_manifold_cache[r_req] = out
            return out

        resp = self._rpc({"cmd": "get_t5_manifold", "max_rank": r_req})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Remote get_t5_manifold error: {resp.get('error')}")
        out = {
            k: torch.from_numpy(resp[k]).to(device=self.device, dtype=torch.float32)
            for k in ("mu", "V", "sigma", "mean_norm")
        }
        self._t5_manifold_cache[r_req] = out
        return out

    @torch.no_grad()
    def encode_semantics(self, text: str, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(text), int(max_len))
        if key not in self._local_prompt_cache:
            if len(self._local_prompt_cache) > 1024:
                self._local_prompt_cache.clear()
            resp = self._rpc({"cmd": "encode_semantics", "text": str(text), "max_len": int(max_len)})
            if resp.get("status") != "ok":
                raise RuntimeError(f"Remote encode_semantics error: {resp.get('error')}")
            emb = torch.from_numpy(resp["embeds"]).to(device=self.device, dtype=self.dtype)
            msk = torch.from_numpy(resp["mask"]).to(device=self.device, dtype=torch.bool)
            self._local_prompt_cache[key] = (emb, msk)
        return self._local_prompt_cache[key]

    @torch.no_grad()
    def set_scheduler_timesteps(self, steps: int, video_seq_len: int):
        if self.is_distilled:
            sigmas = build_distilled_sigmas(steps)
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
            return self.scheduler.timesteps

        min_sigma = min(0.10, 1.0 / max(2, steps))
        sigmas = np.linspace(1.0, min_sigma, steps).tolist()
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
    def encode_frames_to_latent(self, frames_rgb: list[np.ndarray] | np.ndarray) -> torch.Tensor:
        if isinstance(frames_rgb, list):
            arr = np.stack(frames_rgb, axis=0)
        elif frames_rgb.ndim == 3:
            arr = frames_rgb[None, ...]
        else:
            arr = frames_rgb
        if arr.shape[0] == 1:
            return self.encode_frame_to_latent(arr[0])
        resp = self._rpc({"cmd": "encode_frames", "frames_rgb": np.ascontiguousarray(arr, dtype=np.uint8)})
        if resp.get("status") != "ok":
            return self.encode_frame_to_latent(arr[-1])
        return torch.from_numpy(resp["latent_5d"]).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def encode_frame_to_latent(self, frame_rgb: np.ndarray) -> torch.Tensor:
        resp = self._rpc({"cmd": "encode_frame", "frame_rgb": np.ascontiguousarray(frame_rgb, dtype=np.uint8)})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Remote encode_frame error: {resp.get('error')}")
        return torch.from_numpy(resp["latent_5d"]).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def decode_latents_chunk(
        self,
        latents_5d: torch.Tensor,
        generator: torch.Generator | None = None,
        decode_t: float = 0.05,
        decode_noise_scale: float = 0.0,
        last_frame_fix: bool = True,
    ) -> list[np.ndarray]:
        self._decode_call_idx += 1
        resp = self._rpc(
            {
                "cmd": "decode_latents",
                "latents_5d": latents_5d.detach().to(torch.float16).cpu().numpy(),
                "seed": 42 + self._decode_call_idx,
                "decode_t": float(decode_t),
                "decode_noise_scale": float(decode_noise_scale),
                "last_frame_fix": bool(last_frame_fix),
            }
        )
        if resp.get("status") != "ok":
            raise RuntimeError(f"Remote decode_latents error: {resp.get('error')}")
        return resp["frames"]
