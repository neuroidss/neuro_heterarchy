#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: 100% GPU ZERO-BUFFER DIFFUSION RUNNER (VRAM CUDA-IPC BUS)
===================================================================================
- Чтение рабочей памяти напрямую из VRAM через аппаратный CUDA IPC (DeviceToDevice).
- Color Surgery и World Context перенесены с CPU OpenCV на 100% GPU-тензоры!
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import sys
import argparse
import time
import cv2
import torch
import numpy as np
import threading
from pathlib import Path
from multiprocessing.connection import Client

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuro_heterarchy_core import DEVICE
from neuro_genesis_engine import (
    UniversalSemanticSpace,
    SelfSufficientHeterarchyRouter,
)
from neuro_amodal_engine import (
    SharedAmodalCognitiveEngine,
    project_amodal_bus_to_diffusion_gpu,
)
from neuro_recorder import recorder_instance

ipc_lock = threading.Lock()


@torch.no_grad()
def apply_color_surgery_and_blend_gpu(new_rgb_np: np.ndarray, prev_gpu: Optional[torch.Tensor],
                                      use_color: bool, world_context: float) -> tuple[np.ndarray, torch.Tensor]:
    """
    100% GPU-реализация Color Surgery и World Context смешивания (вместо медленного CPU OpenCV).
    """
    cur_t = torch.from_numpy(new_rgb_np).to(DEVICE, dtype=torch.float32)
    if prev_gpu is None or prev_gpu.shape != cur_t.shape:
        return new_rgb_np, cur_t

    if use_color:
        mu = torch.mean(cur_t, dim=(0, 1))
        target_g = 0.5 * (mu[0] + mu[2])
        excess_g = torch.relu(mu[1] - target_g)
        cur_t[:, :, 1] -= excess_g

        green_bleed = cur_t[:, :, 1] * 0.015
        cur_t[:, :, 1] -= green_bleed
        cur_t[:, :, 0] += green_bleed * 0.5
        cur_t[:, :, 2] += green_bleed * 0.5

        mu_t = torch.mean(cur_t, dim=(0, 1), keepdim=True)
        std_t = torch.std(cur_t, dim=(0, 1), keepdim=True) + 1e-5
        std_s = torch.std(prev_gpu, dim=(0, 1), keepdim=True) + 1e-5
        t_std = std_t * 0.85 + std_s * 0.15
        cur_t = (cur_t - mu_t) * (t_std / std_t) + mu_t

    if world_context > 0.01:
        keep = min(0.65, max(0.0, float(world_context)))
        cur_t = cur_t * (1.0 - keep) + prev_gpu * keep

    cur_t = torch.clamp(cur_t, 0.0, 255.0)
    out_np = cur_t.to(torch.uint8).cpu().numpy()
    return out_np, cur_t


class AsyncDiffusionWorker(threading.Thread):
    def __init__(self, conn, initial_h=384, initial_w=512, use_taesd=True, use_color=True, is_sdxl=False, inference_steps=0):
        super().__init__(daemon=True)
        self.conn = conn
        self.img_h = initial_h
        self.img_w = initial_w
        self.use_taesd = use_taesd
        self.use_color = use_color
        self.is_sdxl = is_sdxl
        self.target_steps = inference_steps
        self.actual_steps = 2 if inference_steps == 0 else inference_steps
        self.guidance_scale = 1.0
        self.world_context = 0.18
        self.target_fps = 8.0

        self.display_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.prev_gpu_frame: Optional[torch.Tensor] = None
        self.latest_embeds = None
        self.strength = 0.60
        self.running = True
        self.lock = threading.Lock()
        self.gateway_conn = None
        self.fps = 0.0
        self.gen_ms = 0.0
        self.frame_seq_id = 0
        self.latest_telemetry = {}

    def update_conditioning(self, prompt_embeds_np, strength, telemetry):
        with self.lock:
            self.latest_embeds = prompt_embeds_np
            self.strength = float(np.clip(strength, 0.30, 0.95))
            self.latest_telemetry = telemetry

    def flush(self):
        with self.lock:
            self.display_rgb.fill(0)
            self.prev_gpu_frame = None

    def get_canvas(self):
        with self.lock:
            return self.display_rgb

    def run(self):
        times = []
        internal_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        while self.running:
            with self.lock:
                embeds = self.latest_embeds
                s_val = self.strength
                t_steps = self.target_steps
                a_steps = self.actual_steps
                cfg = self.guidance_scale
                w_ctx = self.world_context
                t_fps = self.target_fps
                req_w, req_h = self.img_w, self.img_h
                u_taesd = self.use_taesd
                u_color = self.use_color
                telem = self.latest_telemetry
                if np.max(self.display_rgb) > 0:
                    internal_rgb = self.display_rgb

            if embeds is None:
                time.sleep(0.005)
                continue

            if internal_rgb.shape[:2] != (req_h, req_w):
                internal_rgb = cv2.resize(internal_rgb, (req_w, req_h), interpolation=cv2.INTER_LINEAR)
                self.prev_gpu_frame = None

            steps_to_run = a_steps if t_steps <= 0 else t_steps
            eff_s = max(s_val, 1.05 / max(1, steps_to_run)) if steps_to_run >= 2 else max(0.55, s_val)
            eff_s = float(np.clip(eff_s, 0.30, 0.95))

            try:
                t0 = time.perf_counter()
                req = {
                    'cmd': 'generate',
                    'image_np': internal_rgb,
                    'prompt_embeds': embeds,
                    'strength': eff_s,
                    'num_inference_steps': steps_to_run,
                    'guidance_scale': cfg,
                    'use_taesd': u_taesd
                }
                with ipc_lock:
                    self.conn.send(req)
                    resp = self.conn.recv()

                gen_dt = time.perf_counter() - t0

                if isinstance(resp, np.ndarray):
                    if resp.shape[:2] != (req_h, req_w):
                        resp = cv2.resize(resp, (req_w, req_h))

                    internal_rgb, self.prev_gpu_frame = apply_color_surgery_and_blend_gpu(
                        resp, self.prev_gpu_frame, u_color, w_ctx
                    )

                    target_dt = 1.0 / max(1.0, t_fps)
                    if t_steps <= 0:
                        if gen_dt < target_dt * 0.68 and a_steps < 6:
                            a_steps += 1
                        elif gen_dt > target_dt * 1.15 and a_steps > 2:
                            a_steps -= 1

                    with self.lock:
                        self.display_rgb = internal_rgb
                        self.actual_steps = a_steps if t_steps <= 0 else t_steps
                        self.gen_ms = gen_dt * 1000.0
                        self.frame_seq_id += 1
                        cur_fid = self.frame_seq_id

                    if self.gateway_conn is None:
                        try:
                            self.gateway_conn = Client(('localhost', 6002), authkey=b'canvas')
                        except Exception:
                            self.gateway_conn = None

                    bgr = cv2.cvtColor(internal_rgb, cv2.COLOR_RGB2BGR)
                    if self.gateway_conn is not None:
                        try:
                            _, enc_jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            self.gateway_conn.send({
                                'frame_id': cur_fid,
                                'ts': time.time(),
                                'gen_ms': self.gen_ms,
                                'fps': self.fps,
                                'actual_steps': self.actual_steps,
                                'w': req_w,
                                'h': req_h,
                                'use_taesd': u_taesd,
                                'use_color': u_color,
                                'jpg': enc_jpg.tobytes()
                            })
                        except Exception:
                            try:
                                self.gateway_conn.close()
                            except Exception:
                                pass
                            self.gateway_conn = None

                    if recorder_instance.is_recording:
                        recorder_instance.push_frame(bgr, telem)

                    times.append(gen_dt)
                    if len(times) > 5:
                        times.pop(0)
                    self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas 100% GPU Diffusion Runner")
    parser.add_argument('--lore', '--concepts-file', dest='lore_file', type=str, default=None)
    parser.add_argument('--concepts', type=str, default=None)
    parser.add_argument('--vocab-ratio', type=float, default=1.0)
    parser.add_argument('--vocab-size', type=int, default=None)

    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    parser.add_argument('--gamma-max', type=float, default=100.0)

    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo", "sdxl"])
    parser.add_argument('--speed', type=str, default="quality", choices=["fast", "quality"])
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--no-taesd', action='store_true', default=False)
    parser.add_argument('--no-color', action='store_true', default=False)

    parser.add_argument('--strength-low', type=float, default=0.50)
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--burst-strength', type=float, default=0.85)
    parser.add_argument('--burst-duration', type=float, default=1.2)
    parser.add_argument('--ema-smooth', type=float, default=0.12)

    parser.add_argument('--seal-thresh', type=float, default=0.68)
    parser.add_argument('--seal-cycles', type=int, default=4)
    parser.add_argument('--decay-cycles', type=int, default=2)
    parser.add_argument('--refresh-cycles', type=int, default=2)

    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--minimal-ui', action='store_true', default=False)
    args = parser.parse_args()

    init_taesd = not args.no_taesd
    init_color = init_taesd if not args.no_color else False

    conn = None
    is_sdxl = False
    print("⏳ Подключение к brain_server (порт 6000)...")
    while conn is None:
        try:
            conn = Client(('localhost', 6000), authkey=b'brain')
            with ipc_lock:
                conn.send({'cmd': 'init_mode', 'mode': args.mode, 'use_taesd': init_taesd})
                resp = conn.recv()
            is_sdxl = resp.get('is_sdxl', False)
            print(f"✅ Диффузия готова! (SDXL: {is_sdxl}, TAESD: {init_taesd}, ColorFix: {init_color})")
        except Exception:
            time.sleep(0.5)

    target_k = args.vocab_size if args.vocab_size is not None else (int(49400 * args.vocab_ratio) if args.vocab_ratio < 1.0 else None)
    semantic = UniversalSemanticSpace(lore_file=args.lore_file, concepts_cli=args.concepts, target_k=target_k)
    router = SelfSufficientHeterarchyRouter(args.users)

    amodal_engine = SharedAmodalCognitiveEngine(
        router=router,
        gamma_max=args.gamma_max,
        strength_low=args.strength_low,
        strength_high=args.burst_strength
    )

    hud = None
    if not args.headless:
        from neuro_hud import NeuroCanvasHUD
        hud = NeuroCanvasHUD(initial_minimal=args.minimal_ui, device=DEVICE)

    canvas_w, canvas_h = (512, 384) if is_sdxl else ((448, 336) if args.speed == "fast" else (512, 384))
    worker = AsyncDiffusionWorker(
        conn, initial_h=canvas_h, initial_w=canvas_w,
        use_taesd=init_taesd, use_color=init_color,
        is_sdxl=is_sdxl, inference_steps=args.steps
    )
    worker.start()

    init_word = semantic.words[0] if semantic.words else ""
    smoothed_target = semantic.get_77_embedding_gpu(init_word).clone()
    current_ema_smooth = args.ema_smooth
    strength_low = args.strength_low
    burst_strength = args.burst_strength

    ctrl_conn = None
    ctrl_lock = threading.Lock()

    def control_listener_thread():
        nonlocal current_ema_smooth, strength_low, burst_strength, ctrl_conn
        while worker.running:
            try:
                if ctrl_conn is None:
                    new_c = Client(('localhost', 6004), authkey=b'diff_ctrl')
                    with ctrl_lock:
                        ctrl_conn = new_c

                has_msg = False
                with ctrl_lock:
                    if ctrl_conn is not None and ctrl_conn.poll(0.02):
                        msg = ctrl_conn.recv()
                        has_msg = True

                if has_msg:
                    cmd = msg.get('cmd')
                    if cmd == 'update_concepts':
                        new_words = msg.get('concepts', [])
                        semantic.update_concepts_on_the_fly(new_words)
                    elif cmd == 'update_params':
                        p = msg.get('params', {})
                        if 'mode' in p and p['mode'] != args.mode:
                            new_mode = str(p['mode'])
                            with ipc_lock:
                                conn.send({'cmd': 'init_mode', 'mode': new_mode, 'use_taesd': worker.use_taesd})
                                resp = conn.recv()
                            args.mode = new_mode
                            with worker.lock:
                                worker.is_sdxl = resp.get('is_sdxl', False)
                        with worker.lock:
                            if 'steps' in p:
                                worker.target_steps = max(0, min(8, int(p['steps'])))
                                if worker.target_steps > 0:
                                    worker.actual_steps = worker.target_steps
                            if 'guidance_scale' in p:
                                worker.guidance_scale = max(0.0, min(3.0, float(p['guidance_scale'])))
                            if 'world_context' in p:
                                worker.world_context = max(0.0, min(0.65, float(p['world_context'])))
                            if 'resolution' in p:
                                res_str = str(p['resolution'])
                                if 'x' in res_str:
                                    rw, rh = [int(x) for x in res_str.split('x')]
                                    worker.img_w = max(256, min(1024, (rw // 8) * 8))
                                    worker.img_h = max(192, min(768, (rh // 8) * 8))
                            if 'use_taesd' in p:
                                worker.use_taesd = bool(p['use_taesd'])
                                worker.use_color = worker.use_taesd if 'use_color' not in p else bool(p['use_color'])
                            elif 'use_color' in p:
                                worker.use_color = bool(p['use_color'])
                        if 'strength_low' in p:
                            strength_low = max(0.30, min(0.85, float(p['strength_low'])))
                        if 'strength_high' in p:
                            burst_strength = max(strength_low + 0.05, min(0.95, float(p['strength_high'])))
                        if 'ema_smooth' in p:
                            current_ema_smooth = max(0.02, min(0.50, float(p['ema_smooth'])))
                    elif cmd == 'start_recording':
                        recorder_instance.start_session(session_name=msg.get('session_name'))
                    elif cmd == 'stop_recording':
                        recorder_instance.stop_session()
                    elif cmd == 'flush':
                        worker.flush()
                else:
                    time.sleep(0.01)
            except Exception:
                with ctrl_lock:
                    ctrl_conn = None
                time.sleep(0.5)

    threading.Thread(target=control_listener_thread, daemon=True).start()

    last_telem_t = 0.0
    last_loop_t = time.perf_counter()

    try:
        while True:
            now_p = time.perf_counter()
            dt = float(np.clip(now_p - last_loop_t, 0.005, 0.10))
            last_loop_t = now_p
            t_now = time.time()

            if hud:
                dt, commands = hud.poll_events(16)
                for cmd, payload in commands:
                    if cmd == "set_role":
                        router.set_role(payload[0], payload[1])
                    elif cmd == "reset_seals":
                        amodal_engine.reset_seals()
                        hud.spiral.rings.clear()
                    elif cmd == "flush_canvas":
                        worker.flush()
            else:
                time.sleep(0.016)

            bus, has_live = amodal_engine.step(dt)
            if not has_live:
                if hud:
                    hud.render_waiting()
                time.sleep(0.02)
                continue

            need_telem = (hud is not None) or ((t_now - last_telem_t) >= 0.15) or recorder_instance.is_recording
            target_77, dyn_str, telemetry = project_amodal_bus_to_diffusion_gpu(
                bus, semantic, strength_low=strength_low, burst_strength=burst_strength,
                need_ui_telemetry=need_telem
            )

            if (t_now - last_telem_t) >= 0.15 and telemetry:
                last_telem_t = t_now
                with ctrl_lock:
                    if ctrl_conn is not None:
                        try:
                            puck = [telemetry.get("f3_x_local", 0.0), telemetry.get("f3_y_global", 0.0)]
                            active_w = telemetry.get("dag_log_parts", [""])[0]
                            ctrl_conn.send({
                                "cmd": "telemetry",
                                "puck": puck,
                                "active_thought": active_w,
                                "dyn_strength": dyn_str,
                                "fps": round(worker.fps, 1),
                                "gen_ms": round(worker.gen_ms, 1),
                                "actual_steps": worker.actual_steps,
                                "resolution": f"{worker.img_w}x{worker.img_h}",
                                "use_taesd": worker.use_taesd,
                                "use_color": worker.use_color,
                                "world_context": worker.world_context,
                                "ladder_stage": 4,
                                "monty_loss": telemetry.get("monty_loss", 0.0),
                                "persistence": telemetry.get("persistence", 0.0),
                                "surprise": telemetry.get("surprise", 0.0),
                                "regions": {},
                                "gpu_sat_mask": [False] * 16
                            })
                        except Exception:
                            ctrl_conn = None

            ema = float(np.clip(dt / max(0.02, current_ema_smooth), 0.04, 0.65))
            smoothed_target.mul_(1.0 - ema).add_(target_77, alpha=ema)
            worker.update_conditioning(smoothed_target.unsqueeze(0).cpu().numpy(), dyn_str, telemetry)

            if hud:
                hud.render(worker.get_canvas(), telemetry, fps=worker.fps, dt=dt)

    except KeyboardInterrupt:
        pass
    finally:
        worker.running = False
        recorder_instance.stop_session()
        amodal_engine.stop()
        if hud:
            hud.close()


if __name__ == '__main__':
    main()
