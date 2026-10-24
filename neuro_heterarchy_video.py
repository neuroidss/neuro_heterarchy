#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: FULL-RANK CONTINUOUS EMBEDDING VIDEO RUNNER (ALWAYS PASSES LORE)
===================================================================================
"""

from __future__ import annotations
import sys
import math
import time
import argparse
import os
from pathlib import Path
from multiprocessing.connection import Client
import cv2
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuro_heterarchy_core import DEVICE
from neuro_genesis_engine import SelfSufficientHeterarchyRouter
from neuro_amodal_engine import (
    SharedAmodalCognitiveEngine,
    HardwareVRAMBus,
    BAND_THETA,
    BAND_BETA,
    BAND_GAMMA,
)
from neuro_recorder import recorder_instance


class RegionalLoreReader:
    def __init__(self, lore_file: str | None = None):
        self.sections: dict[str, list[str]] = {
            "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
        }
        if lore_file and os.path.exists(lore_file):
            current_sec = "FPZ"
            with open(lore_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("[") and s.endswith("]"):
                        current_sec = s[1:-1].strip().upper()
                        if current_sec not in self.sections:
                            self.sections[current_sec] = []
                    else:
                        self.sections[current_sec].append(s)

            print("=" * 80)
            print(f"📜 [ЗАГРУЖЕН ЛОР ИЗ {lore_file}]:")
            for sec, lines in self.sections.items():
                if lines:
                    print(f"  • [{sec}] {len(lines)} строк(и): {lines[0][:60]}...")
            print("=" * 80)


@torch.no_grad()
def compile_full_rank_wm_tensor_gpu(bus: HardwareVRAMBus) -> torch.Tensor:
    wm_field = bus.wm_field_5x5x32x120
    sdr_l23  = bus.monty_l23_5x4096
    sdr_anom = bus.monty_anom_5x4096
    c_flow   = bus.ripple_dag_5x16x16

    wm_128x4096 = torch.zeros((1, 128, 4096), dtype=torch.float32, device=DEVICE)

    token_idx = 0
    for r in range(5):
        for b in range(5):
            wave_flat = wm_field[r, b].reshape(-1)
            wm_128x4096[0, token_idx, 0:3840] = wave_flat
            wm_128x4096[0, token_idx, 3840:4096] = F.adaptive_avg_pool1d(sdr_l23[r].unsqueeze(0), 256).squeeze(0)
            token_idx += 1

    for r in range(5):
        wm_128x4096[0, 25 + r, :] = sdr_l23[r]

    for r in range(5):
        wm_128x4096[0, 30 + r, :] = sdr_anom[r]

    for s in range(32):
        slot_all_bands = wm_field[:, :, s, :].reshape(-1)
        wm_128x4096[0, 35 + s, 0:3000] = slot_all_bands
        wm_128x4096[0, 35 + s, 3000:3000 + 256] = c_flow[0].reshape(-1)

    pac_field = wm_field[:, BAND_THETA] * wm_field[:, BAND_GAMMA]
    pac_reshaped = pac_field.reshape(32, 600)
    for i in range(min(61, 32)):
        wm_128x4096[0, 67 + i, 0:600] = pac_reshaped[i]
        wm_128x4096[0, 67 + i, 600:1200] = wm_field[:, BAND_BETA, i, :].reshape(-1)

    return wm_128x4096


def main():
    parser = argparse.ArgumentParser(description="Full-Rank Continuous Embedding Video Runner")
    parser.add_argument("--lore", type=str, default="lore/maze_lore.txt")
    parser.add_argument("--users", type=str, default="User1:FCz=0")
    parser.add_argument("--gamma-max", type=float, default=100.0)
    parser.add_argument("--video-port", type=int, default=6006)
    parser.add_argument("--gateway-port", type=int, default=6002)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--stochastic", dest="stochastic", action="store_true", default=True)
    parser.add_argument("--deterministic", "--no-stochastic", dest="stochastic", action="store_false")
    args = parser.parse_args()

    init_det = None if args.stochastic is None else (not bool(args.stochastic))

    print(f"⏳ Подключение к video_world_server (порт {args.video_port})...")
    vid_conn = None
    while vid_conn is None:
        try:
            vid_conn = Client(("localhost", args.video_port), authkey=b"video_world")
        except Exception:
            time.sleep(0.5)
    print("✅ [VIDEO-RUNNER] Видео-сервер подключен!")

    lore_reader = RegionalLoreReader(lore_file=args.lore)
    router = SelfSufficientHeterarchyRouter(args.users)
    amodal_engine = SharedAmodalCognitiveEngine(router=router, gamma_max=args.gamma_max)

    hud = None
    if not args.headless:
        from neuro_hud import NeuroCanvasHUD
        hud = NeuroCanvasHUD(device=DEVICE)

    gateway_conn = None
    frame_seq_id = 0
    force_reset_genesis = True
    sent_vram_ipc = False
    last_debug_tag = ""

    try:
        while True:
            dt = 0.016
            if hud:
                dt, commands = hud.poll_events(16)
                for cmd, payload in commands:
                    if cmd == "set_role":
                        router.set_role(payload[0], payload[1])
                    elif cmd in ("flush_canvas", "reset_seals"):
                        force_reset_genesis = True
                        amodal_engine.reset_seals()

            bus, has_live = amodal_engine.step(dt)
            if not has_live:
                if hud:
                    hud.render_waiting()
                time.sleep(0.02)
                continue

            wm_tensor_128x4096 = compile_full_rank_wm_tensor_gpu(bus)

            meta = bus.meta_64
            beta_wall_lock = float(torch.clamp(torch.mean(torch.abs(bus.wm_field_5x5x32x120[2, BAND_BETA])) * 2.5, 0.0, 1.0).item())

            req_msg = {
                "cmd": "step_full_rank",
                "wm_embeds_128x4096": wm_tensor_128x4096.detach().cpu().numpy(),
                "lore_sections": lore_reader.sections,
                "beta_wall_lock": beta_wall_lock,
                "reset_genesis": force_reset_genesis,
            }
            if not sent_vram_ipc and bus.is_owner:
                req_msg["vram_ipc"] = bus.get_cuda_ipc_Payload()
                sent_vram_ipc = True

            if init_det is not None:
                req_msg["deterministic"] = init_det
            force_reset_genesis = False

            vid_conn.send(req_msg)
            resp = vid_conn.recv()

            frames = resp.get("frames", [])
            fps = float(resp.get("fps", 12.0))
            gen_ms = float(resp.get("gen_ms", 80.0))
            debug_tokens = resp.get("debug_tokens", "MAZE_ORB")

            if debug_tokens != last_debug_tag:
                last_debug_tag = debug_tokens
                print(f"🧭 [DEBUG TOKENS]: {debug_tokens} | BetaLock: {beta_wall_lock:.2f} | Pers: {float(meta[6].item()):.2f}")

            n_sec = max(2, int(meta[1].item()))
            telemetry = {
                "primary_role": "FCz", "primary_active": True,
                "ladder_stage": 1,
                "monty_loss": round(float(meta[8].item()), 5),
                "persistence": round(float(meta[6].item()), 3),
                "surprise": round(float(meta[7].item()), 3),
                "total_clock_sectors": n_sec,
                "current_clock_sec": int(meta[2].item()) % n_sec,
                "slot_names": [f"S{i+1}" for i in range(n_sec)],
                "slot_indices": list(range(n_sec)),
                "sealed_mask": [bool(x > 0.5) for x in bus.seals_and_stabs_5x3x16[1, 2, :n_sec].tolist()],
                "slot_stabilities": bus.seals_and_stabs_5x3x16[1, 0, :n_sec].cpu().numpy(),
                "causal_flow_np": bus.ripple_dag_5x16x16[1, :n_sec, :n_sec].cpu().numpy(),
                "dag_log_parts": [debug_tokens, f"BetaWall:{beta_wall_lock:.2f}"],
                "dl_phase": float(meta[3].item()),
                "is_delta_tick": bool(meta[4].item() > 0.5),
                "dyn_strength": float(meta[5].item()),
                "active_vocab_size": 1,
                "fpz_active": True, "fpz_tension": float(meta[16].item()), "fpz_concept": "MAZE_RULES",
                "f3_active": True, "f3_concept": f"WALLS_LOCK:{beta_wall_lock:.2f}",
                "f3_x_local": float(meta[9].item()), "f3_y_global": float(meta[10].item()),
                "f3_kuramoto": float(meta[11].item()),
                "f4_active": True, "f4_concept": "OBSIDIAN_8K", "f4_entropy": float(meta[5].item()),
                "fcz_active": True, "fcz_concept": debug_tokens[:16],
                "fcz_lx": float(meta[12].item()), "fcz_ly": float(meta[13].item()),
                "fcz_rx": float(meta[14].item()), "fcz_ry": float(meta[15].item()),
            }

            for rgb_frame in frames:
                frame_seq_id += 1
                bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                if gateway_conn is None:
                    try:
                        gateway_conn = Client(("localhost", args.gateway_port), authkey=b"canvas")
                    except Exception:
                        gateway_conn = None

                if gateway_conn is not None:
                    try:
                        _, enc_jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        gateway_conn.send({
                            "frame_id": frame_seq_id,
                            "ts": time.time(),
                            "gen_ms": gen_ms / max(1, len(frames)),
                            "fps": fps,
                            "actual_steps": len(frames),
                            "w": rgb_frame.shape[1],
                            "h": rgb_frame.shape[0],
                            "use_taesd": False,
                            "use_color": True,
                            "jpg": enc_jpg.tobytes()
                        })
                    except Exception:
                        gateway_conn = None

                if recorder_instance.is_recording:
                    recorder_instance.push_frame(bgr, telemetry)

                if hud:
                    hud.render(rgb_frame, telemetry, fps=fps, dt=dt)

    except KeyboardInterrupt:
        pass
    finally:
        recorder_instance.stop_session()
        amodal_engine.stop()
        if hud:
            hud.close()


if __name__ == "__main__":
    main()
