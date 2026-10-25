#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: PER-REGION FULL-RANK [1, 128, 4096] VIDEO RUNNER (PORT 6007 VIDEO IPC)
===================================================================================
- Полностью отвязан от 2D-Diffusion (порт 6004) и MusicGen (порт 6003):
  общается с web_cloud_gateway.py по выделенному видео-каналу 6007 (authkey=b'video_ctrl').
- Поддерживает:
  * Горячую смену файла мира/лора (maze_lore.txt <-> elemental_shrine_lore.txt) из браузера.
  * Порегионное редактирование концептов [FPZ], [AFZ], [F3], [F4], [FCZ].
  * Индивидуальные степени стабильности/динамики для каждого региона + авто-буст
    региона, куда подключен девайс пользователя.
  * Настройку микро-блока (num_frames=5), длины кольца контекста (context_ring_len),
    лимита геодезического замка T5 (geodesic_lock) и порога Психической Атаки (psi_threshold).
===================================================================================
"""

from __future__ import annotations
import sys
import time
import argparse
import os
import threading
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
    REGION_ROLES,
    BAND_DELTA,
    BAND_THETA,
    BAND_BETA,
    BAND_GAMMA,
    BAND_RIPPLES,
)
from neuro_recorder import recorder_instance

ROLE_TO_BUS_IDX = {r: i for i, r in enumerate(REGION_ROLES)}


class RegionalLoreReader:
    def __init__(self, lore_file: str | None = None):
        self.lock = threading.RLock()
        s = (lore_file or "").strip()
        if s.startswith("/app/"):
            s = s[len("/app/"):]
        self.lore_file = s
        self.sections: dict[str, list[str]] = {
            "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
        }
        self.load_file(self.lore_file)

    def load_file(self, lore_file: str):
        with self.lock:
            self.lore_file = lore_file
            self.sections = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
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
                print(f"📜 [ЗАГРУЖЕН ПОРЕГИОННЫЙ ЛОР ВИДЕО-МИРА ИЗ {lore_file}]:")
                for sec, lines in self.sections.items():
                    if lines:
                        print(f"  • [{sec}] {len(lines)} концепт(ов): {lines[0][:65]}...")
                print("=" * 80)

    def replace_sections(self, new_sections: dict[str, list[str]], lore_file: str | None = None):
        with self.lock:
            if lore_file:
                self.lore_file = lore_file
            for k in ("FPZ", "AFZ", "F3", "F4", "FCZ", "NEGATIVE"):
                if k in new_sections and isinstance(new_sections[k], list):
                    self.sections[k] = [str(x).strip() for x in new_sections[k] if str(x).strip()]

    def get_copy(self) -> dict[str, list[str]]:
        with self.lock:
            return {k: list(v) for k, v in self.sections.items()}


@torch.no_grad()
def compile_per_region_full_rank_wm_tensor_gpu(bus: HardwareVRAMBus) -> torch.Tensor:
    wm_field = bus.wm_field_5x5x32x120
    sdr_l23  = bus.monty_l23_5x4096
    sdr_anom = bus.monty_anom_5x4096
    c_flow   = bus.ripple_dag_5x16x16

    wm_128x4096 = torch.zeros((1, 128, 4096), dtype=torch.float32, device=DEVICE)

    region_windows = [
        ("Fpz", 0, 25),
        ("AFz", 25, 50),
        ("F3",  50, 75),
        ("F4",  75, 100),
        ("FCz", 100, 128),
    ]

    for role_name, start_i, end_i in region_windows:
        r_idx = ROLE_TO_BUS_IDX[role_name]
        span = end_i - start_i
        sdr_pooled_256 = F.adaptive_avg_pool1d(sdr_l23[r_idx].unsqueeze(0), 256).squeeze(0)
        anom_pooled_256 = F.adaptive_avg_pool1d(sdr_anom[r_idx].unsqueeze(0), 256).squeeze(0)
        dag_flat_256 = c_flow[r_idx].reshape(-1)

        for b in range(5):
            wave_3840 = wm_field[r_idx, b].reshape(-1)
            wm_128x4096[0, start_i + b, 0:3840] = wave_3840
            wm_128x4096[0, start_i + b, 3840:4096] = sdr_pooled_256

        wm_128x4096[0, start_i + 5, :] = sdr_l23[r_idx]
        wm_128x4096[0, start_i + 6, :] = sdr_anom[r_idx]

        pac_tg_3840 = (wm_field[r_idx, BAND_THETA] * wm_field[r_idx, BAND_GAMMA]).reshape(-1)
        pac_bg_3840 = (wm_field[r_idx, BAND_BETA] * wm_field[r_idx, BAND_GAMMA]).reshape(-1)
        rip_g_3840  = (wm_field[r_idx, BAND_RIPPLES] * wm_field[r_idx, BAND_GAMMA]).reshape(-1)

        wm_128x4096[0, start_i + 7, 0:3840] = pac_tg_3840
        wm_128x4096[0, start_i + 7, 3840:4096] = dag_flat_256

        wm_128x4096[0, start_i + 8, 0:3840] = pac_bg_3840
        wm_128x4096[0, start_i + 8, 3840:4096] = anom_pooled_256

        wm_128x4096[0, start_i + 9, 0:3840] = rip_g_3840
        wm_128x4096[0, start_i + 9, 3840:4096] = sdr_pooled_256

        rem = span - 10
        for k in range(rem):
            s_a = (k * 2) % 32
            s_b = (k * 2 + 1) % 32
            slot_pair_1200 = wm_field[r_idx, :, [s_a, s_b], :].reshape(-1)
            chord_600 = (wm_field[r_idx, :, -1, :] - wm_field[r_idx, :, s_a, :]).reshape(-1)
            wm_128x4096[0, start_i + 10 + k, 0:1200] = slot_pair_1200
            wm_128x4096[0, start_i + 10 + k, 1200:1800] = chord_600
            wm_128x4096[0, start_i + 10 + k, 1800:2056] = dag_flat_256
            wm_128x4096[0, start_i + 10 + k, 2056:4096] = sdr_l23[r_idx, :2040]

    return wm_128x4096


def main():
    parser = argparse.ArgumentParser(description="Per-Region Full-Rank 3D-DiT Video Runner")
    parser.add_argument("--lore", type=str, default="lore/maze_lore.txt")
    parser.add_argument("--users", type=str, default="User1:FCz=0")
    parser.add_argument("--gamma-max", type=float, default=100.0)
    parser.add_argument("--video-port", type=int, default=6006)
    parser.add_argument("--gateway-port", type=int, default=6002)
    parser.add_argument("--video-ctrl-port", type=int, default=6007)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--stochastic", dest="stochastic", action="store_true", default=True)
    parser.add_argument("--deterministic", "--no-stochastic", dest="stochastic", action="store_false")

    parser.add_argument("--active-gain", type=float, default=0.92)
    parser.add_argument("--inactive-gain", type=float, default=0.06)
    parser.add_argument("--cross-bleed", type=float, default=0.25)
    args = parser.parse_args()

    init_det = None if args.stochastic is None else (not bool(args.stochastic))

    print(f"⏳ Подключение к video_world_server (порт {args.video_port})...")
    vid_conn = None
    while vid_conn is None:
        try:
            vid_conn = Client(("localhost", args.video_port), authkey=b"video_world")
        except Exception:
            time.sleep(0.5)
    print("✅ [VIDEO-RUNNER] Видео-сервер 3D-DiT подключен!")

    lore_reader = RegionalLoreReader(lore_file=args.lore)
    router = SelfSufficientHeterarchyRouter(args.users)
    amodal_engine = SharedAmodalCognitiveEngine(
        router=router,
        gamma_max=args.gamma_max,
        strength_low=args.inactive_gain,
        strength_high=args.active_gain,
        cross_region_bleed=args.cross_bleed
    )

    hud = None
    if not args.headless:
        from neuro_hud import NeuroCanvasHUD
        hud = NeuroCanvasHUD(device=DEVICE)

    runtime_state = {
        "auto_active_boost": True,
        "active_user_gain": float(args.active_gain),
        "inactive_base_gain": float(args.inactive_gain),
        "cross_region_bleed": float(args.cross_bleed),
        "region_gains": {
            "Fpz": 0.06, "AFz": 0.08, "F3": 0.08, "F4": 0.10, "FCz": 0.92
        },
        "cascade_rate": 1.0,
        "psi_threshold": 0.55,
        "geodesic_lock": 0.18,
        "num_frames": 5,
        "context_ring_len": 16,
        "steps": 4,
        "genesis_steps": 12,
        "cfg_scale": 3.2,
        "current_active_role": "FCz",
        "force_reset_genesis": True,
        "deterministic": init_det,
        "running": True,
    }
    ctrl_lock = threading.Lock()
    video_ctrl_conn = None

    def video_control_listener_thread():
        nonlocal video_ctrl_conn
        while runtime_state["running"]:
            try:
                if video_ctrl_conn is None:
                    c = Client(("localhost", args.video_ctrl_port), authkey=b"video_ctrl")
                    with ctrl_lock:
                        video_ctrl_conn = c
                        video_ctrl_conn.send({
                            "cmd": "sync_from_video_runner",
                            "lore_file": lore_reader.lore_file,
                            "sections": lore_reader.get_copy()
                        })

                has_msg = False
                with ctrl_lock:
                    if video_ctrl_conn is not None and video_ctrl_conn.poll(0.02):
                        msg = video_ctrl_conn.recv()
                        has_msg = True

                if has_msg and isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "switch_lore":
                        l_file = msg.get("lore_file", "")
                        secs = msg.get("sections")
                        if secs and isinstance(secs, dict):
                            lore_reader.replace_sections(secs, lore_file=l_file)
                        elif l_file:
                            lore_reader.load_file(l_file)
                        runtime_state["force_reset_genesis"] = True
                        print(f"🌍 [VIDEO WORLD SWITCH] Переключен мир/лор: {lore_reader.lore_file}")
                    elif cmd == "update_lore_sections":
                        secs = msg.get("sections", {})
                        if isinstance(secs, dict):
                            lore_reader.replace_sections(secs)
                            print("📝 [VIDEO LORE] Обновлены региональные концепты из браузера!")
                    elif cmd == "update_video_config":
                        cfg = msg.get("config", {})
                        for k in (
                            "auto_active_boost", "active_user_gain", "inactive_base_gain",
                            "cross_region_bleed", "cascade_rate", "psi_threshold",
                            "geodesic_lock", "num_frames", "context_ring_len",
                            "steps", "genesis_steps", "cfg_scale"
                        ):
                            if k in cfg:
                                runtime_state[k] = cfg[k]
                        if "region_gains" in cfg and isinstance(cfg["region_gains"], dict):
                            runtime_state["region_gains"].update(cfg["region_gains"])
                        amodal_engine.cross_region_bleed = float(runtime_state["cross_region_bleed"])
                    elif cmd == "set_device_role":
                        new_role = str(msg.get("role", "FCz")).strip()
                        if new_role in REGION_ROLES:
                            router.set_role(0, new_role)
                            runtime_state["current_active_role"] = new_role
                            print(f"🎛️ [VIDEO ROLE SWITCH] Девайс юзера переключен на регион: {new_role}")
                    elif cmd == "set_ladder_stage":
                        stage = int(msg.get("stage", 1))
                        if stage == 1:
                            runtime_state["inactive_base_gain"] = 0.04
                            runtime_state["active_user_gain"] = 0.94
                        elif stage == 2:
                            runtime_state["inactive_base_gain"] = 0.10
                            runtime_state["active_user_gain"] = 0.90
                        elif stage == 3:
                            runtime_state["inactive_base_gain"] = 0.16
                            runtime_state["active_user_gain"] = 0.88
                        else:
                            runtime_state["inactive_base_gain"] = 0.22
                            runtime_state["active_user_gain"] = 0.88
                    elif cmd == "reset_genesis":
                        runtime_state["force_reset_genesis"] = True
                        amodal_engine.reset_seals()
                    elif cmd == "set_init_image":
                        runtime_state["pending_init_image_bytes"] = msg.get("image_bytes")
                    elif cmd == "start_recording":
                        recorder_instance.start_session(session_name=msg.get("session_name"))
                    elif cmd == "stop_recording":
                        recorder_instance.stop_session()
                else:
                    time.sleep(0.01)
            except Exception:
                with ctrl_lock:
                    video_ctrl_conn = None
                time.sleep(0.5)

    threading.Thread(target=video_control_listener_thread, daemon=True).start()

    gateway_conn = None
    frame_seq_id = 0
    sent_vram_ipc = False
    last_debug_tag = ""
    last_telem_push = 0.0

    try:
        while True:
            dt = 0.016
            if hud:
                dt, commands = hud.poll_events(16)
                for cmd, payload in commands:
                    if cmd == "set_role":
                        router.set_role(payload[0], payload[1])
                        if payload[1]:
                            runtime_state["current_active_role"] = payload[1]
                    elif cmd in ("flush_canvas", "reset_seals"):
                        runtime_state["force_reset_genesis"] = True
                        amodal_engine.reset_seals()

            bus, has_live = amodal_engine.step(dt)
            if not has_live:
                if hud:
                    hud.render_waiting()
                time.sleep(0.02)
                continue

            wm_tensor_128x4096 = compile_per_region_full_rank_wm_tensor_gpu(bus)
            meta = bus.meta_64

            active_region_mask = {}
            for r_idx, r_name in enumerate(REGION_ROLES):
                active_region_mask[r_name] = bool(meta[21 + r_idx].item() > 0.5)

            if not any(active_region_mask.values()):
                active_region_mask[runtime_state["current_active_role"]] = True

            primary_active_role = next(
                (r for r in ["FCz", "AFz", "F3", "F4", "Fpz"] if active_region_mask.get(r, False)),
                runtime_state["current_active_role"]
            )
            runtime_state["current_active_role"] = primary_active_role

            pers_val = float(meta[6].item())
            surp_val = float(meta[7].item())
            kuramoto_val = float(meta[11].item())

            # Вычисляем эффективный gain каждого региона:
            # если включен auto_active_boost, то регион с подключенным девайсом юзера получает
            # мощную динамику active_user_gain, а остальные — свои базовые/индивидуальные значения!
            effective_gains = {}
            for r_name in REGION_ROLES:
                manual_g = float(runtime_state["region_gains"].get(r_name, runtime_state["inactive_base_gain"]))
                if runtime_state["auto_active_boost"]:
                    if active_region_mask.get(r_name, False):
                        effective_gains[r_name] = float(max(manual_g, runtime_state["active_user_gain"]))
                    else:
                        effective_gains[r_name] = float(min(manual_g, max(0.02, runtime_state["inactive_base_gain"])))
                else:
                    effective_gains[r_name] = manual_g

            beta_wall_lock = float(
                torch.clamp(torch.mean(torch.abs(bus.wm_field_5x5x32x120[2, BAND_BETA])) * 2.5, 0.0, 1.0).item()
            )
            kin_4d = [
                float(meta[12].item()),
                float(meta[13].item()),
                float(meta[14].item()),
                float(meta[15].item()),
            ]

            req_msg = {
                "cmd": "step_full_rank",
                "wm_embeds_128x4096": wm_tensor_128x4096.detach().cpu().numpy(),
                "lore_sections": lore_reader.get_copy(),
                "beta_wall_lock": beta_wall_lock,
                "active_region_mask": active_region_mask,
                "region_gains": effective_gains,
                "kin_4d": kin_4d,
                "persistence": pers_val,
                "surprise": surp_val,
                "kuramoto": kuramoto_val,
                "video_params": {
                    "cascade_rate": float(runtime_state["cascade_rate"]),
                    "psi_threshold": float(runtime_state["psi_threshold"]),
                    "geodesic_lock": float(runtime_state["geodesic_lock"]),
                    "num_frames": int(runtime_state["num_frames"]),
                    "context_ring_len": int(runtime_state["context_ring_len"]),
                    "steps": int(runtime_state["steps"]),
                    "genesis_steps": int(runtime_state["genesis_steps"]),
                    "cfg_scale": float(runtime_state["cfg_scale"]),
                },
                "reset_genesis": runtime_state["force_reset_genesis"],
            }
            if runtime_state.get("pending_init_image_bytes"):
                req_msg["init_image_bytes"] = runtime_state.pop("pending_init_image_bytes")
            if not sent_vram_ipc and bus.is_owner:
                req_msg["vram_ipc"] = bus.get_cuda_ipc_Payload()
                sent_vram_ipc = True

            if runtime_state["deterministic"] is not None:
                req_msg["deterministic"] = runtime_state["deterministic"]
            runtime_state["force_reset_genesis"] = False

            vid_conn.send(req_msg)
            resp = vid_conn.recv()

            frames = resp.get("frames", [])
            fps = float(resp.get("fps", 12.0))
            gen_ms = float(resp.get("gen_ms", 80.0))
            debug_tokens = resp.get("debug_tokens", "VIDEO_WORLD")
            fpz_charge = float(resp.get("fpz_meta_charge", 0.0))
            afz_prog = float(resp.get("afz_stage_progress", 0.0))
            act_c_idx = resp.get("active_concept_idx", {})

            if debug_tokens != last_debug_tag:
                last_debug_tag = debug_tokens
                active_list = [r for r, v in active_region_mask.items() if v]
                print(
                    f"🎬 [3D-DiT | Active:{active_list} | Ψ-Charge:{fpz_charge:.2f} | Stage:{afz_prog:.2f}] "
                    f"{debug_tokens}"
                )

            n_sec = max(2, int(meta[1].item()))
            p_bus_idx = ROLE_TO_BUS_IDX.get(primary_active_role, 1)
            telemetry = {
                "primary_role": primary_active_role,
                "primary_active": True,
                "ladder_stage": 1,
                "monty_loss": round(float(meta[8].item()), 5),
                "persistence": round(pers_val, 3),
                "surprise": round(surp_val, 3),
                "total_clock_sectors": n_sec,
                "current_clock_sec": int(meta[2].item()) % n_sec,
                "slot_names": [f"{primary_active_role}_S{i+1}" for i in range(n_sec)],
                "slot_indices": list(range(n_sec)),
                "sealed_mask": [bool(x > 0.5) for x in bus.seals_and_stabs_5x3x16[p_bus_idx, 2, :n_sec].tolist()],
                "slot_stabilities": bus.seals_and_stabs_5x3x16[p_bus_idx, 0, :n_sec].cpu().numpy(),
                "causal_flow_np": bus.ripple_dag_5x16x16[p_bus_idx, :n_sec, :n_sec].cpu().numpy(),
                "dag_log_parts": [debug_tokens, f"Ψ:{fpz_charge:.2f}"],
                "dl_phase": float(meta[3].item()),
                "is_delta_tick": bool(meta[4].item() > 0.5),
                "dyn_strength": float(effective_gains[primary_active_role]),
                "active_vocab_size": sum(len(v) for v in lore_reader.sections.values()),
                "fpz_active": active_region_mask.get("Fpz", False),
                "fpz_tension": fpz_charge,
                "fpz_concept": f"Ψ:{fpz_charge*100:.0f}%",
                "f3_active": active_region_mask.get("F3", False),
                "f3_concept": f"G:{effective_gains['F3']:.2f}",
                "f3_x_local": float(meta[9].item()),
                "f3_y_global": float(meta[10].item()),
                "f3_kuramoto": kuramoto_val,
                "f4_active": active_region_mask.get("F4", False),
                "f4_concept": f"G:{effective_gains['F4']:.2f}",
                "f4_entropy": float(meta[5].item()),
                "fcz_active": active_region_mask.get("FCz", False),
                "fcz_concept": debug_tokens[:18],
                "fcz_lx": kin_4d[0],
                "fcz_ly": kin_4d[1],
                "fcz_rx": kin_4d[2],
                "fcz_ry": kin_4d[3],
            }

            now_t = time.time()
            if now_t - last_telem_push >= 0.20:
                last_telem_push = now_t
                with ctrl_lock:
                    if video_ctrl_conn is not None:
                        try:
                            video_ctrl_conn.send({
                                "cmd": "video_telemetry",
                                "fps": round(fps, 1),
                                "gen_ms": round(gen_ms, 1),
                                "debug_tokens": debug_tokens,
                                "fpz_meta_charge": round(fpz_charge, 3),
                                "afz_stage_progress": round(afz_prog, 3),
                                "monty_loss": telemetry["monty_loss"],
                                "persistence": telemetry["persistence"],
                                "surprise": telemetry["surprise"],
                                "active_region_mask": active_region_mask,
                                "region_gains": {k: round(v, 3) for k, v in effective_gains.items()},
                                "active_concept_idx": act_c_idx,
                            })
                        except Exception:
                            video_ctrl_conn = None

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
        runtime_state["running"] = False
        recorder_instance.stop_session()
        amodal_engine.stop()
        if hud:
            hud.close()


if __name__ == "__main__":
    main()
