#!/usr/bin/env python3
"""
neuro_battle_server.py — Бессмертный сервер весов LTX-Video (Multi-Frame Causal Encode + Last-Frame Fix)
"""

from __future__ import annotations
import os
import argparse
import gc
import multiprocessing as mp
from multiprocessing.connection import Listener
import time
import traceback
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
mp.current_process().authkey = b"battle_ltx"

from neuro_ltx_backend import LTXModelBackend


class DumbWeightsServer:

    def __init__(
        self,
        quant_mode: str = "fp8",
        vae_tile_min_px: int = 384,
        allow_t5_offload: bool = False,
    ):
        print("=" * 80)
        print(
            f"🚀 [DUMB-WEIGHTS-SERVER] Загрузка LTX-Video в VRAM "
            f"(quant={quant_mode}, allow_t5_offload={allow_t5_offload})..."
        )
        t0 = time.perf_counter()
        self.allow_t5_offload = bool(allow_t5_offload)
        self.backend = LTXModelBackend(
            quant_mode=quant_mode,
            vae_tile_min_px=vae_tile_min_px,
            allow_t5_offload=self.allow_t5_offload,
        )
        self.device = self.backend.device
        self.dtype = self.backend.dtype
        self.prompt_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        dt = time.perf_counter() - t0
        print(
            f"✅ [DUMB-WEIGHTS-SERVER] Готов за {dt:.1f}с! T5-XXL постоянно резидентен в VRAM."
        )
        print("=" * 80)

    def _to_device_tensor(self, x, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.to(device=self.device)
        else:
            t = torch.from_numpy(x).to(device=self.device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    @staticmethod
    def _to_cpu_np(t: torch.Tensor):
        if t.dtype == torch.bfloat16:
            return t.detach().to(torch.float16).cpu().numpy()
        return t.detach().cpu().numpy()

    @torch.no_grad()
    def handle_ping(self) -> dict:
        sched_cfg = dict(self.backend.scheduler.config)
        return {
            "status": "ok",
            "mode": "dumb_weights_rpc",
            "quant": self.backend.quant_mode,
            "spatial_ratio": int(self.backend.spatial_ratio),
            "temporal_ratio": int(self.backend.temporal_ratio),
            "in_channels": int(self.backend.transformer.config.in_channels),
            "patch_size": int(getattr(self.backend.transformer.config, "patch_size", 1)),
            "patch_size_t": int(
                getattr(self.backend.transformer.config, "patch_size_t", 1)
            ),
            "base_image_seq_len": int(sched_cfg.get("base_image_seq_len", 256)),
            "max_image_seq_len": int(sched_cfg.get("max_image_seq_len", 4096)),
            "base_shift": float(sched_cfg.get("base_shift", 0.5)),
            "max_shift": float(sched_cfg.get("max_shift", 1.15)),
        }

    @torch.no_grad()
    def handle_encode_semantics(self, msg: dict) -> dict:
        text = str(msg.get("text", ""))
        max_len = int(msg.get("max_len", 128))
        key = (text, max_len)
        if key not in self.prompt_cache:
            if len(self.prompt_cache) > 1024:
                self.prompt_cache.clear()
            emb, msk = self.backend.encode_semantics(text, max_len=max_len)
            self.prompt_cache[key] = (emb.detach().float().cpu(), msk.detach().cpu())
        emb_cpu, msk_cpu = self.prompt_cache[key]
        return {
            "status": "ok",
            "embeds": emb_cpu.numpy(),
            "mask": msk_cpu.numpy(),
        }

    @torch.no_grad()
    def handle_transformer_forward(self, msg: dict) -> dict:
        hidden_states = self._to_device_tensor(msg["hidden_states"], dtype=self.dtype)
        encoder_hidden_states = self._to_device_tensor(
            msg["encoder_hidden_states"], dtype=self.dtype
        )
        timestep = self._to_device_tensor(msg["timestep"], dtype=torch.float32)
        encoder_attention_mask = self._to_device_tensor(
            msg["encoder_attention_mask"], dtype=torch.bool
        )

        num_frames = int(msg["num_frames"])
        height = int(msg["height"])
        width = int(msg["width"])
        rope_scale = tuple(float(x) for x in msg["rope_interpolation_scale"])

        try:
            out = self.backend.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0]
        except torch.cuda.OutOfMemoryError:
            print("⚠️ [VRAM GUARD] OOM в трансформере! Очистка кэша CUDA...")
            if self.allow_t5_offload:
                self.backend.offload_t5_to_cpu(force=True)
            gc.collect()
            torch.cuda.empty_cache()
            out = self.backend.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0]

        return {
            "status": "ok",
            "sample": self._to_cpu_np(out),
        }

    @torch.no_grad()
    def handle_encode_frame(self, msg: dict) -> dict:
        frame_rgb = msg["frame_rgb"]
        lat_5d = self.backend.encode_frame_to_latent(frame_rgb)
        return {
            "status": "ok",
            "latent_5d": lat_5d.detach().to(torch.float16).cpu().numpy(),
        }

    @torch.no_grad()
    def handle_encode_frames(self, msg: dict) -> dict:
        frames_rgb = msg["frames_rgb"]
        lat_5d = self.backend.encode_frames_to_latent(frames_rgb)
        return {
            "status": "ok",
            "latent_5d": lat_5d.detach().to(torch.float16).cpu().numpy(),
        }

    @torch.no_grad()
    def handle_decode_latents(self, msg: dict) -> dict:
        latents_5d = self._to_device_tensor(msg["latents_5d"], dtype=torch.float32)
        seed = int(msg.get("seed", 42))
        decode_t = float(msg.get("decode_t", 0.05))
        decode_noise_scale = float(msg.get("decode_noise_scale", 0.0))
        last_frame_fix = bool(msg.get("last_frame_fix", True))
        gen = torch.Generator(device=self.device).manual_seed(seed)
        try:
            frames = self.backend.decode_latents_chunk(
                latents_5d,
                generator=gen,
                decode_t=decode_t,
                decode_noise_scale=decode_noise_scale,
                last_frame_fix=last_frame_fix,
            )
        except torch.cuda.OutOfMemoryError:
            print("⚠️ [VRAM GUARD] OOM в 3D-VAE! Включаю тайлинг 256x256...")
            if self.allow_t5_offload:
                self.backend.offload_t5_to_cpu(force=True)
            if hasattr(self.backend.vae, "enable_tiling"):
                self.backend.vae.enable_tiling(
                    tile_sample_min_height=256,
                    tile_sample_min_width=256,
                    tile_sample_stride_height=192,
                    tile_sample_stride_width=192,
                )
            gc.collect()
            torch.cuda.empty_cache()
            frames = self.backend.decode_latents_chunk(
                latents_5d,
                generator=gen,
                decode_t=decode_t,
                decode_noise_scale=decode_noise_scale,
                last_frame_fix=last_frame_fix,
            )

        return {
            "status": "ok",
            "frames": frames,
        }


def main():
    parser = argparse.ArgumentParser(description="Dumb Immortal LTX-Video Weights Server")
    parser.add_argument(
        "--port", type=int, default=6008, help="IPC port for Weights Server"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="fp8",
        choices=["fp8", "int8", "nf4", "bf16", "auto"],
        help="Режим упаковки 3D-DiT в VRAM: fp8 (по умолчанию) | int8 | nf4 | bf16",
    )
    parser.add_argument(
        "--vae-tile-min",
        type=int,
        default=384,
        help="Порог включения тайлинга 3D-VAE в пикселях",
    )
    parser.add_argument(
        "--allow-t5-offload",
        action="store_true",
        default=False,
        help="Разрешить выгрузку T5-энкодера из VRAM (по умолчанию ЗАПРЕЩЕНО)",
    )
    args = parser.parse_args()

    server = DumbWeightsServer(
        quant_mode=args.quant,
        vae_tile_min_px=args.vae_tile_min,
        allow_t5_offload=args.allow_t5_offload,
    )
    listener = Listener(("localhost", args.port), authkey=b"battle_ltx")
    print(
        f"🟢 [DUMB-WEIGHTS-SERVER] Слушаю localhost:{args.port} "
        f"(quant={args.quant}, T5_resident={not args.allow_t5_offload})..."
    )

    decode_count = 0
    while True:
        try:
            conn = listener.accept()
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError, BrokenPipeError):
                    break

                if not isinstance(msg, dict):
                    continue

                cmd = msg.get("cmd")
                try:
                    if cmd == "ping":
                        conn.send(server.handle_ping())
                    elif cmd == "encode_semantics":
                        conn.send(server.handle_encode_semantics(msg))
                    elif cmd == "evict_t5_to_cpu":
                        used_gb = server.backend.offload_t5_to_cpu(force=False)
                        conn.send({"status": "ok", "vram_gb": used_gb})
                    elif cmd == "transformer_forward":
                        conn.send(server.handle_transformer_forward(msg))
                    elif cmd == "encode_frame":
                        conn.send(server.handle_encode_frame(msg))
                    elif cmd == "encode_frames":
                        conn.send(server.handle_encode_frames(msg))
                    elif cmd == "decode_latents":
                        conn.send(server.handle_decode_latents(msg))
                        decode_count += 1
                        if decode_count % 8 == 0:
                            torch.cuda.empty_cache()
                    elif cmd == "clear_cache":
                        server.prompt_cache.clear()
                        gc.collect()
                        torch.cuda.empty_cache()
                        conn.send({"status": "ok"})
                    else:
                        conn.send({"status": "error", "error": f"Unknown cmd: {cmd}"})
                except (EOFError, ConnectionResetError, BrokenPipeError):
                    break

        except KeyboardInterrupt:
            print("\n🛑 [DUMB-WEIGHTS-SERVER] Stopped.")
            break
        except Exception as e:
            print(f"⚠️ [DUMB-WEIGHTS-SERVER ERROR]: {e}")
            traceback.print_exc()
            time.sleep(0.2)


if __name__ == "__main__":
    main()
