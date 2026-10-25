#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: MODULAR 3D-DiT VIDEO WORLD SERVER (video_world_server.py)
===================================================================================
Соединяет два независимых модуля:
  1) video_render_logic.py        — Неприкосновенный рендерер T5 4-Bit + 3D-DiT.
  2) video_heterarchy_composer.py — Порегионный маппинг рабочей памяти в концепты лора.
===================================================================================
"""

from __future__ import annotations
import time
import argparse
from multiprocessing.connection import Listener
import torch

from neuro_heterarchy_core import DEVICE
from neuro_amodal_engine import HardwareVRAMBus
from video_render_logic import NeuroVideoRenderer
from video_heterarchy_composer import RegionalVideoHeterarchyComposer


def main():
    parser = argparse.ArgumentParser(description="Modular 3D-DiT Video World Server")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model", type=str, default="ltx-video", choices=["ltx-video", "wan-1.3b", "wan2.2-5b"])
    parser.add_argument("--lore", type=str, default="lore/maze_lore.txt")
    parser.add_argument("--init-image", type=str, default="")
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

    renderer = NeuroVideoRenderer(
        model_type=args.model,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        genesis_steps=args.genesis_steps,
        cfg_scale=args.cfg,
        force_low_vram=args.force_low_vram,
        deterministic=deterministic,
    )
    if args.init_image:
        with open(args.init_image, "rb") as f:
            renderer.set_custom_init_image_bytes(f.read())

    composer = RegionalVideoHeterarchyComposer(renderer=renderer, lore_path=args.lore)
    vram_bus = HardwareVRAMBus()

    listener = Listener(("localhost", args.port), authkey=b"video_world")
    print(f"🚀 [VIDEO-WORLD-SERVER] Готов на порту {args.port} (Renderer & Composer Separated!)")

    while True:
        try:
            conn = listener.accept()
            while True:
                msg = conn.recv()
                cmd = msg.get("cmd")
                if cmd in ("step_full_rank", "step_regional_maze"):
                    t_start = time.perf_counter()
                    if "vram_ipc" in msg and msg["vram_ipc"]:
                        vram_bus.attach_from_ipc_dict(msg["vram_ipc"])

                    if "init_image_bytes" in msg and msg["init_image_bytes"]:
                        renderer.set_custom_init_image_bytes(msg["init_image_bytes"])

                    if msg.get("reset_genesis", False):
                        renderer.reset_genesis()

                    lore_sec = msg.get("lore_sections", None)
                    composer.sync_lore_sections(lore_sec)

                    wm_np = msg.get("wm_embeds_128x4096")
                    wm_t = torch.from_numpy(wm_np).to(DEVICE, dtype=torch.float32) if wm_np is not None else torch.zeros((1, 128, 4096), device=DEVICE)

                    act_mask = msg.get("active_region_mask") or {"FCz": True, "AFz": False, "F3": False, "F4": False, "Fpz": False}
                    reg_gains = msg.get("region_gains") or {"Fpz": 0.06, "AFz": 0.08, "F3": 0.08, "F4": 0.10, "FCz": 0.92}
                    beta_lock = float(msg.get("beta_wall_lock", 0.50))
                    pers = float(msg.get("persistence", 0.50))
                    surp = float(msg.get("surprise", 0.20))
                    kura = float(msg.get("kuramoto", 0.50))
                    vid_params = msg.get("video_params", None)

                    pos_emb, neg_emb, debug_tag, adv_rate = composer.compose_step(
                        in_wm_128x4096=wm_t,
                        active_region_mask=act_mask,
                        region_gains=reg_gains,
                        persistence=pers,
                        surprise=surp,
                        kuramoto=kura,
                        video_params=vid_params,
                    )

                    req_det = msg.get("deterministic", None)
                    if "stochastic" in msg and msg["stochastic"] is not None:
                        req_det = not bool(msg["stochastic"])

                    wall_keep = float(min(0.72, max(0.50, 0.55 + 0.15 * beta_lock)))
                    seed_v = int(torch.clamp(torch.abs(wm_t.reshape(-1)[0]) * 10000.0, 1.0, 999999.0).item())

                    frames = renderer.render_rollout(
                        final_embeds_1x128x4096=pos_emb,
                        neg_embeds_1x128x4096=neg_emb,
                        wall_keep=wall_keep,
                        anchor_advance_rate=adv_rate,
                        use_deterministic=req_det,
                        seed_val=seed_v,
                    )
                    gen_ms = (time.perf_counter() - t_start) * 1000.0
                    conn.send({
                        "frames": frames,
                        "gen_ms": gen_ms,
                        "fps": len(frames) / max(0.001, gen_ms / 1000.0),
                        "debug_tokens": debug_tag,
                        "deterministic": renderer.deterministic if req_det is None else bool(req_det),
                        "fpz_meta_charge": round(composer.fpz_meta_charge, 3),
                        "afz_stage_progress": round(composer.afz_stage_progress, 3),
                        "active_concept_idx": dict(composer.active_concept_idx),
                    })
                elif cmd == "set_init_image":
                    if "image_bytes" in msg and msg["image_bytes"]:
                        renderer.set_custom_init_image_bytes(msg["image_bytes"])
                    conn.send({"status": "ok"})
                elif cmd == "reset_genesis":
                    renderer.reset_genesis()
                    conn.send({"status": "ok"})
                elif cmd == "set_stochastic":
                    stoch = bool(msg.get("stochastic", True))
                    renderer.deterministic = not stoch
                    conn.send({"status": "ok", "stochastic": stoch, "deterministic": renderer.deterministic})
        except EOFError:
            continue
        except Exception as e:
            print(f"⚠️ [IPC Error]: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
