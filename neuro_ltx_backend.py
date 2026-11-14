#!/usr/bin/env python3
"""
neuro_ltx_backend.py — Аппаратный бэкенд LTX-Video с выбором квантования (--quant) и Denoising VAE
==================================================================================================
- Возвращен штатный режим Denoising VAE Decoder (decode_t = 0.05, decode_noise_scale = 0.025),
  который полностью убирает крупную пиксельную сетку и дает резкие грани кристаллов!
"""

from __future__ import annotations
import os
import sys
import gc
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
    """Контейнер весов LTX-Video с поддержкой --quant (fp8 | int8 | nf4 | bf16)."""

    def __init__(
        self,
        device: torch.device | None = None,
        quant_mode: str = "fp8",
        vae_tile_min_px: int = 384,
    ):
        from diffusers import LTXPipeline, LTXVideoTransformer3DModel
        from transformers import AutoTokenizer, BitsAndBytesConfig, T5EncoderModel

        self.device = device or DEVICE
        self.quant_mode = str(quant_mode).lower().strip()
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

        self.text_encoder.eval()
        self.t5_offloaded = False

        transformer = None
        if self.quant_mode in ("int8", "nf4", "4bit"):
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
                if self.quant_mode == "int8":
                    print("⚙️ [3D-DiT QUANT] Загрузка трансформера в аппаратном 8-bit (BitsAndBytes INT8)...")
                    dit_bnb = DiffusersBnBConfig(load_in_8bit=True)
                else:
                    print("⚙️ [3D-DiT QUANT] Загрузка трансформера в ультра-легком 4-bit NF4...")
                    dit_bnb = DiffusersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=self.dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                transformer = LTXVideoTransformer3DModel.from_pretrained(
                    MODEL_REPO,
                    subfolder="transformer",
                    quantization_config=dit_bnb,
                    torch_dtype=self.dtype,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                print(f"⚠️ [BnB Quant Fallback to FP8]: {e}")
                self.quant_mode = "fp8"
                transformer = None

        if transformer is None:
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                MODEL_REPO,
                subfolder="transformer",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if self.quant_mode in ("fp8", "8bit") and hasattr(transformer, "enable_layerwise_casting"):
                print("⚡ [3D-DiT QUANT] Включен Layerwise Casting FP8 (float8_e4m3fn) -> экономия ~2 GB VRAM!")
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype
                )
            else:
                print(f"🚀 [3D-DiT QUANT] Трансформер в нативном {self.dtype} (максимальная скорость Tensor Cores)!")

        self.pipe = LTXPipeline.from_pretrained(
            MODEL_REPO,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            torch_dtype=self.dtype,
        )
        self.pipe.set_progress_bar_config(disable=True)
        if self.quant_mode not in ("int8", "nf4", "4bit"):
            self.pipe.transformer.to(self.device)
        self.pipe.transformer.eval()
        self.pipe.vae.to(self.device).eval()

        tile_min = max(256, int(vae_tile_min_px))
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
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"📊 [VRAM USAGE] Занято после инициализации: {used_gb:.2f} GB / {vram_gb:.1f} GB (quant={self.quant_mode})")

    @torch.no_grad()
    def offload_t5_to_cpu(self) -> float:
        if self.text_encoder is not None and not self.t5_offloaded:
            try:
                del self.text_encoder
                self.text_encoder = None
                self.t5_offloaded = True
                gc.collect()
                torch.cuda.empty_cache()
                used_gb = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
                print(f"🧹 [T5 OFFLOAD] T5-XXL выгружен из VRAM (все промпты в кэше)! Занято всего {used_gb:.2f} GB VRAM!")
                return used_gb
            except Exception as e:
                print(f"⚠️ [T5 Offload Note]: {e}")
        return 0.0

    def _ensure_t5_loaded(self):
        if self.text_encoder is None:
            from transformers import T5EncoderModel
            print("📥 [T5 RELOAD] Подгрузка 4-битного T5-XXL для нового промпта...")
            self.text_encoder = T5EncoderModel.from_pretrained(
                str(QUANT_T5_DIR),
                device_map="cuda",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            ).eval()
            self.t5_offloaded = False

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
        key = (str(text), int(max_len))
        if key not in self._local_prompt_cache:
            if len(self._local_prompt_cache) > 512:
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
        decode_t: float = 0.05,
        decode_noise_scale: float = 0.025,
    ) -> list[np.ndarray]:
        denorm = self.pipe._denormalize_latents(
            latents_5d.to(dtype=torch.float32),
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(dtype=self.vae.dtype)

        if getattr(self.vae.config, "timestep_conditioning", False):
            t_tensor = torch.tensor([float(decode_t)], device=self.device, dtype=denorm.dtype)
            if decode_noise_scale > 0.0:
                noise = torch.randn(
                    denorm.shape,
                    generator=generator,
                    device=self.device,
                    dtype=denorm.dtype,
                )
                denorm = (1.0 - float(decode_noise_scale)) * denorm + float(decode_noise_scale) * noise
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
        return [frames_uint8[i] for i in range(frames_uint8.shape[0])]


class _TransformerConfigProxy:
    def __init__(self, in_channels: int = 128):
        self.in_channels = in_channels


class _RemoteTransformerProxy:
    def __init__(self, conn, device: torch.device, dtype: torch.dtype, in_channels: int = 128):
        self.conn = conn
        self.device = device
        self.dtype = dtype
        self.config = _TransformerConfigProxy(in_channels=in_channels)

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
        self.conn.send({
            "cmd": "transformer_forward",
            "hidden_states": self._to_np_f16(hidden_states),
            "encoder_hidden_states": self._to_np_f16(encoder_hidden_states),
            "timestep": timestep.detach().float().cpu().numpy(),
            "encoder_attention_mask": encoder_attention_mask.detach().cpu().numpy(),
            "num_frames": int(num_frames),
            "height": int(height),
            "width": int(width),
            "rope_interpolation_scale": tuple(float(x) for x in rope_interpolation_scale),
        })
        resp = self.conn.recv()
        if resp.get("status") != "ok":
            raise RuntimeError(f"Remote transformer error: {resp.get('error')}")
        sample = torch.from_numpy(resp["sample"]).to(device=self.device, dtype=self.dtype)
        return (sample,)


class _LocalLatentPackerPipe:
    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class RemoteLTXModelBackend:
    """Клиентский прокси к бессмертному серверу весов (neuro_battle_server.py)."""

    def __init__(self, conn, ping_info: dict, device: torch.device | None = None):
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        self.conn = conn
        self.device = device or DEVICE
        self.dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self._spatial_ratio = int(ping_info.get("spatial_ratio", 32))
        self._temporal_ratio = int(ping_info.get("temporal_ratio", 8))
        in_channels = int(ping_info.get("in_channels", 128))

        self.pipe = _LocalLatentPackerPipe()
        self.transformer = _RemoteTransformerProxy(
            conn=self.conn,
            device=self.device,
            dtype=self.dtype,
            in_channels=in_channels,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_image_seq_len=int(ping_info.get("base_image_seq_len", 256)),
            max_image_seq_len=int(ping_info.get("max_image_seq_len", 4096)),
            base_shift=float(ping_info.get("base_shift", 0.5)),
            max_shift=float(ping_info.get("max_shift", 1.15)),
            shift_terminal=0.1,
        )
        self._local_prompt_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._decode_call_idx = 0

    @property
    def spatial_ratio(self) -> int:
        return self._spatial_ratio

    @property
    def temporal_ratio(self) -> int:
        return self._temporal_ratio

    def offload_t5_to_cpu(self):
        try:
            self.conn.send({"cmd": "evict_t5_to_cpu"})
            self.conn.recv()
        except Exception:
            pass

    @torch.no_grad()
    def encode_semantics(self, text: str, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(text), int(max_len))
        if key not in self._local_prompt_cache:
            if len(self._local_prompt_cache) > 512:
                self._local_prompt_cache.clear()
            self.conn.send({
                "cmd": "encode_semantics",
                "text": str(text),
                "max_len": int(max_len),
            })
            resp = self.conn.recv()
            if resp.get("status") != "ok":
                raise RuntimeError(f"Remote encode_semantics error: {resp.get('error')}")
            emb = torch.from_numpy(resp["embeds"]).to(device=self.device, dtype=self.dtype)
            msk = torch.from_numpy(resp["mask"]).to(device=self.device, dtype=torch.bool)
            self._local_prompt_cache[key] = (emb, msk)
        return self._local_prompt_cache[key]

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
        decode_t: float = 0.05,
        decode_noise_scale: float = 0.025,
    ) -> list[np.ndarray]:
        self._decode_call_idx += 1
        self.conn.send({
            "cmd": "decode_latents",
            "latents_5d": latents_5d.detach().to(torch.float16).cpu().numpy(),
            "seed": 42 + self._decode_call_idx,
            "decode_t": float(decode_t),
            "decode_noise_scale": float(decode_noise_scale),
        })
        resp = self.conn.recv()
        if resp.get("status") != "ok":
            raise RuntimeError(f"Remote decode_latents error: {resp.get('error')}")
        return resp["frames"]
