#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: RESIDENT 4-BIT T5 & FULL-RANK CONTINUOUS 3D-DiT VIDEO SERVER
===================================================================================
- 100% GPU VRAM ONLY: T5 (6.4 ГБ) + LTX-Video FP8 (~1.9 ГБ) = ~8.5 ГБ (свободно 3 ГБ).
- ПОЛНОРАНГОВОЕ УПРАВЛЕНИЕ: Прямое модулирование через wm_embeds [1, 128, 4096].
- СТАБИЛЬНОСТЬ СТЕН: Восстановлен якорь геометрии (wall_keep), предотвращающий 
  деградацию и размытие картинки в муть при генерации.
===================================================================================
"""

from __future__ import annotations
import os
import gc
import zlib
import math
import time
import argparse
from pathlib import Path
from multiprocessing.connection import Listener
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE
from neuro_amodal_engine import HardwareVRAMBus

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


def parse_lore_dict(lore_sections: dict[str, list[str]] | None) -> tuple[str, str]:
    if not lore_sections:
        return (
            "overhead top-down view of a simple 5x5 stone maze, clear wide corridors, glowing green exit portal, giant bright luminous golden plasma sphere",
            "extreme close-up, macro photography, 13x13, tiny grid, green wireframe, green matrix, blurry, broken walls"
        )

    pos_lines = []
    neg_lines = lore_sections.get("NEGATIVE", [])

    for k in ["FPZ", "AFZ", "F3", "F4", "FCZ"]:
        if k in lore_sections and lore_sections[k]:
            pos_lines.extend(lore_sections[k])

    pos_text = " ".join(pos_lines).strip()
    neg_text = " ".join(neg_lines).strip()
    return pos_text, neg_text


@torch.no_grad()
def encode_clean_128_t5(tokenizer, text_encoder, prompt_text: str, seq_len: int = 128) -> torch.Tensor:
    text_clean = prompt_text.strip()
    inp = tokenizer([text_clean], return_tensors="pt", padding=False, truncation=True, max_length=seq_len)
    input_ids = inp.input_ids.to(DEVICE)
    out = text_encoder(input_ids=input_ids)

    hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    act_emb = hidden.squeeze(0).float()
    if act_emb.ndim == 1:
        act_emb = act_emb.unsqueeze(0)

    l_act = act_emb.shape[0]
    if l_act >= seq_len:
        res = act_emb[:seq_len].contiguous()
    else:
        repeats = (seq_len // max(1, l_act)) + 1
        res = act_emb.repeat(repeats, 1)[:seq_len].contiguous()

    return res.to(DEVICE, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


class FullRankVideoWorldEngine:
    def __init__(self, model_type: str = "ltx-video", width: int = 0, height: int = 0,
                 num_frames: int = 9, steps: int = 4, genesis_steps: int = 14,
                 cfg_scale: float = 3.5, force_low_vram: bool = False,
                 deterministic: bool = False, lore_path: str = "lore/maze_lore.txt"):

        self.device = DEVICE
        self.dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.model_type = model_type.lower()
        self.deterministic = bool(deterministic)
        self.lore_path = lore_path

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

        if self.deterministic and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.vram_bus = HardwareVRAMBus()
        self.pipe_t2v = None
        self.pipe_i2v = None
        self.seq_len = 128

        self.tokenizer = None
        self.text_encoder = None
        self.vocab_embeds_norm = None

        self.current_lore_hash = None
        self.base_embeds: torch.Tensor | None = None
        self.neg_embed: torch.Tensor | None = None
        self.ones_mask = torch.ones((1, self.seq_len), dtype=torch.long, device=self.device)

        # Стабильная визуальная память
        self.genesis_maze_gpu: torch.Tensor | None = None
        self.last_frame_gpu: torch.Tensor | None = None
        self.is_genesis_done: bool = False

        # 1. Загрузка T5 в VRAM (6.4 ГБ)
        self._init_resident_t5()

        # 2. Начальное кодирование лора
        initial_sections = self._read_lore_file_sections()
        self._encode_lore_on_the_fly(initial_sections)

        # 3. Загрузка DiT (FP8 Layerwise ~1.9 ГБ в VRAM)
        self._load_video_pipelines(self.model_type)

    def _read_lore_file_sections(self) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
        if self.lore_path and os.path.exists(self.lore_path):
            current_sec = "FPZ"
            with open(self.lore_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("[") and s.endswith("]"):
                        current_sec = s[1:-1].strip().upper()
                        if current_sec not in sections:
                            sections[current_sec] = []
                    else:
                        sections[current_sec].append(s)
        return sections

    def _init_resident_t5(self):
        from transformers import AutoTokenizer, T5EncoderModel
        repo_id = MODEL_REPO_MAP.get(self.model_type, MODEL_REPO_MAP["ltx-video"])

        print("=" * 80)
        if not self.is_low_vram:
            print(f"🚀 [VRAM: {self.total_vram_gb:.1f} GB] Загрузка T5 в bfloat16...")
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
            self.text_encoder = T5EncoderModel.from_pretrained(
                repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
            )
        else:
            quant_dir = resolve_quantized_t5_dir()
            is_saved = quant_dir.exists() and (quant_dir / "model.safetensors").exists()

            if is_saved:
                print(f"📥 [4-BIT T5] Загрузка готовой 6.4 ГБ модели из {quant_dir} в VRAM...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
                self.text_encoder = T5EncoderModel.from_pretrained(
                    str(quant_dir), device_map="cuda", low_cpu_mem_usage=True
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
                    repo_id, subfolder="text_encoder", quantization_config=bnb_cfg, device_map="cuda", low_cpu_mem_usage=True
                )
                quant_dir.mkdir(parents=True, exist_ok=True)
                self.text_encoder.save_pretrained(str(quant_dir))
                self.tokenizer.save_pretrained(str(quant_dir))

        try:
            vocab_w = self.text_encoder.get_input_embeddings().weight.detach().float()
            self.vocab_embeds_norm = F.normalize(vocab_w, p=2, dim=-1).to(self.device)
            print("✅ [T5 В VRAM] Модель готова к генерации эмбеддингов на лету.")
        except Exception:
            self.vocab_embeds_norm = None

    def _encode_lore_on_the_fly(self, lore_sections: dict[str, list[str]] | None):
        if not lore_sections:
            lore_sections = self._read_lore_file_sections()

        pos_text, neg_text = parse_lore_dict(lore_sections)
        new_hash = zlib.crc32(f"{pos_text}_{neg_text}".encode("utf-8"))

        if self.current_lore_hash == new_hash and self.base_embeds is not None:
            return

        print(f"🧠 [T5 VRAM] Кодирование лора лабиринта 5x5: {pos_text[:85]}...")
        self.base_embeds = encode_clean_128_t5(self.tokenizer, self.text_encoder, pos_text, self.seq_len).unsqueeze(0)
        self.neg_embed = encode_clean_128_t5(self.tokenizer, self.text_encoder, neg_text, self.seq_len).unsqueeze(0)
        self.current_lore_hash = new_hash

    def _load_video_pipelines(self, model_type: str):
        repo_id = MODEL_REPO_MAP.get(self.model_type, MODEL_REPO_MAP["ltx-video"])
        print("=" * 80)
        print(f"🎬 [VIDEO-WORLD-SERVER] Загрузка 3D-DiT: {repo_id}")
        t0 = time.time()
        try:
            from diffusers import LTXPipeline, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

            load_kwargs = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
            transformer = LTXVideoTransformer3DModel.from_pretrained(repo_id, subfolder="transformer", **load_kwargs)

            # Layerwise casting в FP8: LTX-Video занимает ~1.9 ГБ в VRAM!
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
            print(f"✅ [VIDEO-WORLD-SERVER] Готов за {time.time() - t0:.1f}с! Занято VRAM: {alloc_mb:.0f} МБ (свободно ~3 ГБ)")
            print("=" * 80)
        except Exception as e:
            print(f"❌ [DiT Load Error]: {e}")

    @torch.no_grad()
    def decode_nearest_tokens_for_debug(self, embed_slice_4096: torch.Tensor, top_k: int = 3) -> list[str]:
        if self.vocab_embeds_norm is None or self.tokenizer is None:
            return ["EMBED_DRIVE"]
        vec_norm = F.normalize(embed_slice_4096.float(), p=2, dim=-1)
        sims = torch.mv(self.vocab_embeds_norm, vec_norm)
        top_ids = torch.topk(sims, top_k).indices.cpu().tolist()
        words = [self.tokenizer.decode([idx]).strip() for idx in top_ids if idx > 0]
        return words or ["MAZE_ORB"]

    @torch.no_grad()
    def step_full_rank_rollout(
        self,
        wm_embeds_128x4096: torch.Tensor,
        lore_sections: dict[str, list[str]] | None = None,
        beta_wall_lock: float = 0.50,
        reset_genesis: bool = False,
        deterministic: bool | None = None
    ) -> dict:
        t_start = time.perf_counter()
        h, w = self.height, self.width
        use_det = self.deterministic if deterministic is None else bool(deterministic)

        if reset_genesis:
            self.is_genesis_done = False
            self.genesis_maze_gpu = None
            self.last_frame_gpu = None

        self._encode_lore_on_the_fly(lore_sections)

        base_128x4096 = self.base_embeds.float()
        clean_norms = torch.norm(base_128x4096, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        in_wm = wm_embeds_128x4096.to(device=self.device, dtype=torch.float32)
        if in_wm.ndim == 2:
            in_wm = in_wm.unsqueeze(0)

        debug_words_act = self.decode_nearest_tokens_for_debug(in_wm[0, 31], top_k=2)
        debug_words_l23 = self.decode_nearest_tokens_for_debug(in_wm[0, 26], top_k=2)
        debug_tag = f"FCz: {'/'.join(debug_words_act)} | L2/3: {'/'.join(debug_words_l23)}"

        # 75% — устойчивый мир лабиринта, 25% — динамическая рабочая память коры
        wm_normed = F.normalize(in_wm, p=2, dim=-1) * clean_norms
        final_embeds = (F.normalize(base_128x4096 * 0.75 + wm_normed * 0.25, p=2, dim=-1) * clean_norms).to(dtype=self.dtype)

        if use_det:
            seed_val = int(torch.clamp(torch.abs(in_wm[0, 26, 0]) * 10000.0, 1.0, 999999.0).item())
            gen_arg = torch.Generator(device=self.device).manual_seed(seed_val)
        else:
            gen_arg = None

        frames_out = []
        if self.pipe_t2v is not None:
            try:
                if not self.is_genesis_done or self.genesis_maze_gpu is None or self.pipe_i2v is None:
                    # ФАЗА 1 (GENESIS): Создание четкого 5x5 лабиринта и большого шара
                    vid_pt = self.pipe_t2v(
                        prompt_embeds=final_embeds,
                        prompt_attention_mask=self.ones_mask,
                        negative_prompt_embeds=self.neg_embed,
                        negative_prompt_attention_mask=self.ones_mask,
                        width=w,
                        height=h,
                        num_frames=self.num_frames,
                        num_inference_steps=self.genesis_steps,
                        guidance_scale=self.cfg_scale,
                        generator=gen_arg,
                        output_type="pt"
                    ).frames[0]

                    self.genesis_maze_gpu = vid_pt[0].detach().clone()
                    self.last_frame_gpu = vid_pt[-1].detach().clone()
                    self.is_genesis_done = True
                else:
                    # ФАЗА 2: Стабилизация стен (держит четкость и не дает картинке распадаться в муть)
                    wall_keep = float(np.clip(0.60 + 0.15 * beta_wall_lock, 0.50, 0.75))
                    anchor_gpu = self.genesis_maze_gpu * wall_keep + self.last_frame_gpu * (1.0 - wall_keep)
                    anchor_np = (anchor_gpu.permute(1, 2, 0).float().clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
                    anchor_pil = Image.fromarray(anchor_np, mode="RGB")

                    vid_pt = self.pipe_i2v(
                        image=anchor_pil,
                        prompt_embeds=final_embeds,
                        prompt_attention_mask=self.ones_mask,
                        negative_prompt_embeds=self.neg_embed,
                        negative_prompt_attention_mask=self.ones_mask,
                        width=w,
                        height=h,
                        num_frames=self.num_frames,
                        num_inference_steps=self.steps,
                        guidance_scale=self.cfg_scale,
                        generator=gen_arg,
                        output_type="pt"
                    ).frames[0]

                    self.last_frame_gpu = vid_pt[-1].detach().clone()

                frames_uint8 = (vid_pt.permute(0, 2, 3, 1).float().clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
                frames_out = [frames_uint8[i] for i in range(frames_uint8.shape[0])]
            except Exception as e:
                import traceback
                print(f"⚠️ [DiT Execution Warning]: {e}")
                traceback.print_exc()

        if not frames_out:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            frames_out = [blank] * self.num_frames

        gen_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "frames": frames_out,
            "gen_ms": gen_ms,
            "fps": (len(frames_out) / max(0.001, gen_ms / 1000.0)),
            "debug_tokens": debug_tag,
            "deterministic": use_det
        }


def main():
    parser = argparse.ArgumentParser(description="Fast 3D-DiT Video Server (Native BF16 / FP8)")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model", type=str, default="ltx-video", choices=["ltx-video", "wan-1.3b", "wan2.2-5b"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--genesis-steps", type=int, default=14)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--force-low-vram", action="store_true", default=False)
    parser.add_argument("--stochastic", dest="stochastic", action="store_true", default=None)
    parser.add_argument("--deterministic", "--no-stochastic", dest="stochastic", action="store_false")
    args = parser.parse_args()

    stochastic_mode = True if args.stochastic is None else bool(args.stochastic)
    deterministic = not stochastic_mode

    engine = FullRankVideoWorldEngine(
        model_type=args.model,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        genesis_steps=args.genesis_steps,
        cfg_scale=args.cfg,
        force_low_vram=args.force_low_vram,
        deterministic=deterministic
    )

    listener = Listener(("localhost", args.port), authkey=b"video_world")
    print(f"🚀 [VIDEO-WORLD-SERVER] Готов на порту {args.port} (T5 4-Bit + FP8 DiT in VRAM)")

    while True:
        try:
            conn = listener.accept()
            while True:
                msg = conn.recv()
                cmd = msg.get("cmd")
                if cmd in ("step_full_rank", "step_regional_maze"):
                    if "vram_ipc" in msg and msg["vram_ipc"]:
                        engine.vram_bus.attach_from_ipc_dict(msg["vram_ipc"])

                    wm_np = msg.get("wm_embeds_128x4096")
                    wm_t = torch.from_numpy(wm_np).to(DEVICE) if wm_np is not None else None

                    beta_lock = float(msg.get("beta_wall_lock", 0.50))
                    reset_gen = bool(msg.get("reset_genesis", False))
                    lore_sec = msg.get("lore_sections", None)

                    req_det = msg.get("deterministic", None)
                    if "stochastic" in msg and msg["stochastic"] is not None:
                        req_det = not bool(msg["stochastic"])

                    res = engine.step_full_rank_rollout(
                        wm_embeds_128x4096=wm_t,
                        lore_sections=lore_sec,
                        beta_wall_lock=beta_lock,
                        reset_genesis=reset_gen,
                        deterministic=req_det
                    )
                    conn.send(res)
                elif cmd == "reset_genesis":
                    engine.is_genesis_done = False
                    engine.genesis_maze_gpu = None
                    engine.last_frame_gpu = None
                    conn.send({"status": "ok"})
                elif cmd == "set_stochastic":
                    stoch = bool(msg.get("stochastic", True))
                    engine.deterministic = not stoch
                    conn.send({"status": "ok", "stochastic": stoch, "deterministic": engine.deterministic})
        except EOFError:
            continue
        except Exception as e:
            print(f"⚠️ [IPC Error]: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
