#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: 3D-DiT VIDEO WORLD SERVER (DIRECT GPU VRAM BUS ROUTING)
===================================================================================
"""

from __future__ import annotations
import time
import argparse
from multiprocessing.connection import Listener
import numpy as np
import torch

from neuro_heterarchy_core import DEVICE
from neuro_vram_bus import HardwareVRAMBus, REGION_ROLES
from video_render_logic import NeuroVideoRenderer
from video_heterarchy_composer import RegionalVideoHeterarchyComposer


def main():
    parser = argparse.ArgumentParser(description="Modular 3D-DiT Video World Server")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model", type=str, default="ltx-video", choices=["ltx-video", "wan-1.3b", "wan2.2-5b"])
    parser.add_argument("--lore", type=str, default="")
    parser.add_argument("--init-image", type=str, default="")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--genesis-steps", type=int, default=18)
    parser.add_argument("--cfg", type=float, default=3.2)
    parser.add_argument("--force-low-vram", action="store_true", default=False)
    parser.add_argument("--stochastic", type=float, default=0.0)
    parser.add_argument("--unassigned-stochastic", type=float, default=1.0)
    parser.add_argument("--wm-radius", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stochastic_level = float(np.clip(args.stochastic, 0.0, 1.0))
    unassigned_stochastic_level = float(np.clip(args.unassigned_stochastic, 0.0, 1.0))
    wm_radius_val = float(np.clip(args.wm_radius, 0.0, 1.0))

    renderer = NeuroVideoRenderer(
        model_type=args.model,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        genesis_steps=args.genesis_steps,
        cfg_scale=args.cfg,
        force_low_vram=args.force_low_vram,
        stochastic=stochastic_level,
        unassigned_stochastic=unassigned_stochastic_level,
        wm_radius=wm_radius_val,
        seed=args.seed,
    )
    if args.init_image:
        with open(args.init_image, "rb") as f:
            renderer.set_custom_init_image_bytes(f.read())

    composer = RegionalVideoHeterarchyComposer(renderer=renderer, lore_path=args.lore)
    vram_bus = HardwareVRAMBus()

    listener = Listener(("localhost", args.port), authkey=b"video_world")
    print(
        f"🚀 [VIDEO-WORLD-SERVER] Готов на порту {args.port} (100% GPU VRAM-Pipeline)"
    )

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
                    else:
                        vram_bus.try_attach_to_existing_vram()

                    has_vram_bus = (vram_bus.attached_pid > 0)

                    if "init_image_bytes" in msg and msg["init_image_bytes"]:
                        renderer.set_custom_init_image_bytes(msg["init_image_bytes"])

                    if msg.get("reset_genesis", False):
                        renderer.reset_genesis()

                    lore_sec = msg.get("lore_sections", None)
                    composer.sync_lore_sections(lore_sec)

                    wm_np = msg.get("wm_embeds_128x4096")
                    wm_t = (
                        torch.from_numpy(wm_np).to(DEVICE, dtype=torch.float32)
                        if wm_np is not None
                        else torch.zeros((1, 128, 4096), device=DEVICE, dtype=torch.float32)
                    )

                    wm_5x5x32x120_np = msg.get("wm_5x5x32x120")
                    if wm_5x5x32x120_np is not None:
                        wm_field_tensor = torch.from_numpy(wm_5x5x32x120_np).to(DEVICE, dtype=torch.float32)
                    elif has_vram_bus:
                        wm_field_tensor = vram_bus.wm_field_5x5x32x120
                    else:
                        wm_field_tensor = None

                    act_mask = msg.get("active_region_mask") or {
                        "FCz": True, "AFz": False, "F3": False, "F4": False, "Fpz": False
                    }
                    reg_gains = msg.get("region_gains") or {
                        "Fpz": 0.06, "AFz": 0.08, "F3": 0.08, "F4": 0.10, "FCz": 0.92
                    }
                    beta_lock = float(msg.get("beta_wall_lock", 0.50))
                    pers = float(msg.get("persistence", 0.50))
                    surp = float(msg.get("surprise", 0.20))
                    kura = float(msg.get("kuramoto", 0.50))
                    vid_params = msg.get("video_params", None)

                    req_stoch = float(np.clip(float(msg.get("stochastic", renderer.stochastic)), 0.0, 1.0))
                    req_unass_stoch = float(np.clip(float(msg.get("unassigned_stochastic", renderer.unassigned_stochastic)), 0.0, 1.0))
                    req_wm_radius = float(np.clip(float(msg.get("wm_radius", renderer.wm_radius)), 0.0, 1.0))
                    req_seed = int(msg.get("seed", renderer.base_seed))

                    if vid_params and isinstance(vid_params, dict):
                        if "num_frames" in vid_params:
                            nf = int(vid_params["num_frames"])
                            renderer.num_frames = max(9, ((nf - 1) // 8) * 8 + 1)
                        if "steps" in vid_params:
                            renderer.steps = max(10, int(vid_params["steps"]))
                        if "genesis_steps" in vid_params:
                            renderer.genesis_steps = max(14, int(vid_params["genesis_steps"]))
                        if "cfg_scale" in vid_params:
                            renderer.cfg_scale = float(vid_params["cfg_scale"])

                    act_flags_5 = [1.0 if act_mask.get(r, False) else 0.0 for r in REGION_ROLES]

                    # 100% GPU VRAM-шина передаётся напрямую в composer
                    pos_emb, neg_emb, debug_tag, adv_rate = composer.compose_step(
                        in_wm_128x4096=wm_t,
                        active_region_mask=act_mask,
                        region_gains=reg_gains,
                        persistence=pers,
                        surprise=surp,
                        kuramoto=kura,
                        stochastic=req_stoch,
                        unassigned_stochastic=req_unass_stoch,
                        wm_radius=req_wm_radius,
                        video_params=vid_params,
                        vram_bus=vram_bus if has_vram_bus else None
                    )

                    wall_keep = float(min(0.72, max(0.50, 0.55 + 0.15 * beta_lock)))

                    frames = renderer.render_rollout(
                        final_embeds_1x128x4096=pos_emb,
                        neg_embeds_1x128x4096=neg_emb,
                        wall_keep=wall_keep,
                        anchor_advance_rate=adv_rate,
                        stochastic=req_stoch,
                        unassigned_stochastic=req_unass_stoch,
                        wm_radius=req_wm_radius,
                        wm_5x5x32x120=wm_field_tensor,
                        active_region_flags=act_flags_5,
                        seed_val=req_seed,
                    )
                    gen_ms = (time.perf_counter() - t_start) * 1000.0
                    conn.send({
                        "frames": frames,
                        "gen_ms": gen_ms,
                        "fps": len(frames) / max(0.001, gen_ms / 1000.0),
                        "debug_tokens": debug_tag,
                        "stochastic": req_stoch,
                        "unassigned_stochastic": req_unass_stoch,
                        "wm_radius": req_wm_radius,
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
                    if "stochastic" in msg:
                        renderer.stochastic = float(np.clip(float(msg["stochastic"]), 0.0, 1.0))
                    if "unassigned_stochastic" in msg:
                        renderer.unassigned_stochastic = float(np.clip(float(msg["unassigned_stochastic"]), 0.0, 1.0))
                    if "wm_radius" in msg:
                        renderer.wm_radius = float(np.clip(float(msg["wm_radius"]), 0.0, 1.0))
                    conn.send({
                        "status": "ok",
                        "stochastic": renderer.stochastic,
                        "unassigned_stochastic": renderer.unassigned_stochastic,
                        "wm_radius": renderer.wm_radius,
                    })
        except EOFError:
            continue
        except Exception as e:
            print(f"⚠️ [IPC Error]: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
