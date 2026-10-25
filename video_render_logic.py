#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ISOLATED 3D-DiT & 4-BIT T5 VIDEO RENDER ENGINE (video_render_logic.py)
===================================================================================
- ЭТОТ МОДУЛЬ ОТВЕЧАЕТ ИСКЛЮЧИТЕЛЬНО ЗА КАЧЕСТВО РЕНДЕРИНГА 3D-DiT И T5 В VRAM.
- Строго сохраняет целостность T5-эмбеддингов [1, 128, 4096] и баланс экспозиции I2V
  из эталонной рабочей версии (никаких разрезаний последовательности T5 и никакого
  белого пересвета!).
===================================================================================
"""

from __future__ import annotations
import os
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE

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

    if Path("/workspace/hf_cache").exists():
        target = QUANT_T5_DIR
    else:
        target = Path(os.path.expanduser("~/.cache/huggingface/t5_xxl_4bit"))
    target.mkdir(parents=True, exist_ok=True)
    return target


class NeuroVideoRenderer:
    """
    Чистый изолированный рендерер 3D-DiT (LTX-Video / Wan) + 4-bit T5 в VRAM.
    """
    def __init__(self, model_type: str = "ltx-video", width: int = 0, height: int = 0,
                 num_frames: int = 9, steps: int = 4, genesis_steps: int = 14,
                 cfg_scale: float = 3.5, force_low_vram: bool = False,
                 deterministic: bool = False):

        self.device = DEVICE
        self.dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.model_type = model_type.lower()
        self.deterministic = bool(deterministic)
        self.seq_len = 128

        self.total_vram_gb = 12.0
        if torch.cuda.is_available():
            self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        self.is_low_vram = force_low_vram or (self.total_vram_gb < 22.0)

        if width <= 0 or height <= 0:
            self.width, self.height = (448, 320) if self.is_low_vram else (512, 384)
        else:
            self.width = (width // 32) * 32
            self.height = (height // 32) * 32

        self.num_frames = 9 if self.is_low_vram else max(9, ((num_frames - 1) // 8) * 8 + 1)
        self.steps = max(4, steps)
        self.genesis_steps = max(12, genesis_steps)
        self.cfg_scale = float(cfg_scale)

        self.tokenizer = None
        self.text_encoder = None
        self.vocab_embeds_norm = None
        self.pipe_t2v = None
        self.pipe_i2v = None
        self.ones_mask = torch.ones((1, self.seq_len), dtype=torch.long, device=self.device)

        # Кадры визуального состояния в float32 [3, H, W] в диапазоне [0, 1]
        self.genesis_maze_gpu: torch.Tensor | None = None
        self.last_frame_gpu: torch.Tensor | None = None
        self.custom_init_image_gpu: torch.Tensor | None = None
        self.is_genesis_done: bool = False

        self._init_resident_t5()
        self._load_video_pipelines()

    def _init_resident_t5(self):
        from transformers import AutoTokenizer, T5EncoderModel
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

            if is_saved:
                print(f"📥 [4-BIT T5] Загрузка готовой 6.4 ГБ модели из {quant_dir} в VRAM...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
                self.text_encoder = T5EncoderModel.from_pretrained(
                    str(quant_dir),
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
            else:
                print("⚙️ [4-BIT T5] Первичный запуск: квантование T5-XXL в 4 бита...")
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
                )
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

        vocab_w = self.text_encoder.get_input_embeddings().weight.detach().float()
        self.vocab_embeds_norm = F.normalize(vocab_w, p=2, dim=-1).to(self.device)
        print("✅ [T5 В VRAM] Модель готова к целостному кодированию сцен.")

    def _load_video_pipelines(self):
        repo_id = MODEL_REPO_MAP.get(self.model_type, MODEL_REPO_MAP["ltx-video"])
        print("=" * 80)
        print(f"🎬 [VIDEO-RENDERER] Загрузка 3D-DiT: {repo_id}")
        t0 = time.time()
        try:
            from diffusers import LTXPipeline, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

            load_kwargs = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
            transformer = LTXVideoTransformer3DModel.from_pretrained(repo_id, subfolder="transformer", **load_kwargs)

            if self.is_low_vram and hasattr(transformer, "enable_layerwise_casting"):
                transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype)

            self.pipe_t2v = LTXPipeline.from_pretrained(
                repo_id, text_encoder=None, tokenizer=None, transformer=transformer, torch_dtype=self.dtype
            )
            self.pipe_i2v = LTXImageToVideoPipeline(
                scheduler=self.pipe_t2v.scheduler, vae=self.pipe_t2v.vae,
                text_encoder=None, tokenizer=None, transformer=self.pipe_t2v.transformer,
            )

            self.pipe_t2v.set_progress_bar_config(disable=True)
            if self.pipe_i2v is not None:
                self.pipe_i2v.set_progress_bar_config(disable=True)

            if hasattr(self.pipe_t2v.vae, "enable_tiling"): self.pipe_t2v.vae.enable_tiling()
            if hasattr(self.pipe_t2v.vae, "enable_slicing"): self.pipe_t2v.vae.enable_slicing()

            self.pipe_t2v.transformer.to(self.device)
            self.pipe_t2v.vae.to(self.device)

            torch.cuda.empty_cache()
            alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(f"✅ [VIDEO-RENDERER] Готов за {time.time() - t0:.1f}с! Занято VRAM: {alloc_mb:.0f} МБ")
            print("=" * 80)
        except Exception as e:
            print(f"❌ [DiT Load Error]: {e}")

    @torch.no_grad()
    def encode_holistic_128_t5(self, prompt_text: str) -> torch.Tensor:
        """
        100% эталонное целостное кодирование из первой рабочей версии (с защитой от пустой строки для Triton):
        Возвращает [128, 4096] в float32.
        """
        text_clean = prompt_text.strip()
        if not text_clean:
            inp = self.tokenizer([""], return_tensors="pt", padding="max_length", max_length=16)
            out = self.text_encoder(input_ids=inp.input_ids.to(self.device), attention_mask=inp.attention_mask.to(self.device))
            hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            act_emb = hidden.squeeze(0)[:1].float()
        else:
            inp = self.tokenizer([text_clean], return_tensors="pt", padding=False, truncation=True, max_length=self.seq_len)
            input_ids = inp.input_ids.to(self.device)
            if input_ids.shape[-1] < 4:
                input_ids = input_ids.repeat(1, 4)
            out = self.text_encoder(input_ids=input_ids)
            hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            act_emb = hidden.squeeze(0).float()

        if act_emb.ndim == 1:
            act_emb = act_emb.unsqueeze(0)

        l_act = act_emb.shape[0]
        if l_act >= self.seq_len:
            res = act_emb[:self.seq_len].contiguous()
        else:
            repeats = (self.seq_len // max(1, l_act)) + 1
            res = act_emb.repeat(repeats, 1)[:self.seq_len].contiguous()

        return res.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def decode_nearest_tokens(self, vec_4096: torch.Tensor, top_k: int = 2) -> str:
        if self.vocab_embeds_norm is None or self.tokenizer is None:
            return "WM_DRIVE"
        vec_norm = F.normalize(vec_4096.float(), p=2, dim=-1)
        sims = torch.mv(self.vocab_embeds_norm[256:16384], vec_norm)
        top_ids = (torch.topk(sims, top_k).indices + 256).cpu().tolist()
        words = [self.tokenizer.decode([idx]).strip() for idx in top_ids if idx > 0]
        return "/".join(w for w in words if w) or "WM_DRIVE"

    @torch.no_grad()
    def set_custom_init_image_bytes(self, img_bytes: bytes):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            t_img = torch.from_numpy(arr).to(self.device).permute(2, 0, 1).contiguous()
            self.custom_init_image_gpu = t_img.clone()
            self.genesis_maze_gpu = t_img.clone()
            self.last_frame_gpu = t_img.clone()
            self.is_genesis_done = True
            print(f"🖼️ [RENDERER] Загружена стартовая картинка мира ({self.width}x{self.height})!")
        except Exception as e:
            print(f"⚠️ [Init Image Error]: {e}")

    def reset_genesis(self):
        if self.custom_init_image_gpu is not None:
            self.genesis_maze_gpu = self.custom_init_image_gpu.clone()
            self.last_frame_gpu = self.custom_init_image_gpu.clone()
            self.is_genesis_done = True
        else:
            self.is_genesis_done = False
            self.genesis_maze_gpu = None
            self.last_frame_gpu = None

    @torch.no_grad()
    def render_rollout(
        self,
        final_embeds_1x128x4096: torch.Tensor,
        neg_embeds_1x128x4096: torch.Tensor,
        wall_keep: float = 0.55,
        anchor_advance_rate: float = 0.18,
        use_deterministic: bool | None = None,
        seed_val: int = 42,
    ) -> list[np.ndarray]:
        """
        Выполняет 1 шаг роллаута 3D-DiT:
        - wall_keep (0.45..0.70): держит картинку от распада и выгорания в белый шум (как в первом видео!).
        - anchor_advance_rate (0.0..0.35): плавно подтягивает genesis_maze_gpu к last_frame_gpu
          ТОЛЬКО в тех местах, где изменилось состояние объекта, чтобы объект не возвращался по кругу
          на старт, а продвигался вперед по миру!
        """
        h, w = self.height, self.width
        use_det = self.deterministic if use_deterministic is None else bool(use_deterministic)
        gen_arg = torch.Generator(device=self.device).manual_seed(seed_val) if use_det else None

        cond_pos = final_embeds_1x128x4096.to(device=self.device, dtype=self.dtype)
        cond_neg = neg_embeds_1x128x4096.to(device=self.device, dtype=self.dtype)

        frames_out = []
        if self.pipe_t2v is not None:
            try:
                if not self.is_genesis_done or self.genesis_maze_gpu is None or self.pipe_i2v is None:
                    vid_pt = self.pipe_t2v(
                        prompt_embeds=cond_pos,
                        prompt_attention_mask=self.ones_mask,
                        negative_prompt_embeds=cond_neg,
                        negative_prompt_attention_mask=self.ones_mask,
                        width=w,
                        height=h,
                        num_frames=self.num_frames,
                        num_inference_steps=self.genesis_steps,
                        guidance_scale=self.cfg_scale,
                        generator=gen_arg,
                        output_type="pt"
                    ).frames[0]

                    self.genesis_maze_gpu = vid_pt[0].detach().float().clone()
                    self.last_frame_gpu = vid_pt[-1].detach().float().clone()
                    self.is_genesis_done = True
                else:
                    # 1. Плавное продвижение якоря в зоне активного действия (чтобы шар НЕ возвращался на старт,
                    #    но и стены никогда не выгорали в белый шум!)
                    if anchor_advance_rate > 0.005:
                        diff_mag = torch.mean(torch.abs(self.last_frame_gpu - self.genesis_maze_gpu), dim=0, keepdim=True)
                        local_advance = torch.clamp(diff_mag * 3.0, 0.0, 1.0) * float(np.clip(anchor_advance_rate, 0.0, 0.35))
                        self.genesis_maze_gpu = self.genesis_maze_gpu * (1.0 - local_advance) + self.last_frame_gpu * local_advance

                    # 2. Точная формула стабилизации из первого видео:
                    keep = float(np.clip(wall_keep, 0.45, 0.75))
                    anchor_gpu = self.genesis_maze_gpu * keep + self.last_frame_gpu * (1.0 - keep)
                    anchor_np = (anchor_gpu.permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
                    anchor_pil = Image.fromarray(anchor_np, mode="RGB")

                    vid_pt = self.pipe_i2v(
                        image=anchor_pil,
                        prompt_embeds=cond_pos,
                        prompt_attention_mask=self.ones_mask,
                        negative_prompt_embeds=cond_neg,
                        negative_prompt_attention_mask=self.ones_mask,
                        width=w,
                        height=h,
                        num_frames=self.num_frames,
                        num_inference_steps=self.steps,
                        guidance_scale=self.cfg_scale,
                        generator=gen_arg,
                        output_type="pt"
                    ).frames[0]

                    self.last_frame_gpu = vid_pt[-1].detach().float().clone()

                frames_uint8 = (vid_pt.permute(0, 2, 3, 1).float().clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
                frames_out = [frames_uint8[i] for i in range(frames_uint8.shape[0])]
            except Exception as e:
                import traceback
                print(f"⚠️ [Renderer Execution Warning]: {e}")
                traceback.print_exc()

        if not frames_out:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            frames_out = [blank] * self.num_frames

        return frames_out
