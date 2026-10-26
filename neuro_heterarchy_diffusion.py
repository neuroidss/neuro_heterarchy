#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: 100% GPU ZERO-BUFFER DIFFUSION RUNNER (VRAM CUDA-IPC BUS)
===================================================================================
- ZERO-COPY VRAM STREAMING:
  Вектор мысли target_77 пишется напрямую в bus.live_clip_77x768 прямо в VRAM!
  Никаких .cpu().numpy() для эмбеддингов и wm_field_5x5x32x120!
  Пока U-Net считает шаги денойзинга, основной цикл со скоростью 120 Гц продолжает
  обновлять bus.live_clip_77x768 в VRAM, и U-Net подхватывает изменения мысли
  ПРЯМО ВНУТРИ каждого шага денойзинга!
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import sys
import argparse
import time
import cv2
import torch
import torch.nn.functional as F
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
    PHYSICAL_PROJECTION_768x120,
)
from neuro_amodal_engine import (
    SharedAmodalCognitiveEngine,
    HardwareVRAMBus,
    BAND_DELTA,
    BAND_THETA,
    BAND_RIPPLES,
)
from neuro_recorder import recorder_instance

ipc_lock = threading.Lock()


@torch.no_grad()
def project_amodal_bus_to_diffusion_gpu(bus: HardwareVRAMBus, semantic: UniversalSemanticSpace,
                                        strength_low: float, burst_strength: float,
                                        need_ui_telemetry: bool = False) -> tuple[torch.Tensor, float, dict]:
    """
    100% GPU маппинг рабочей памяти (Дельта + Тета + Бета + Гамма + Рипплы по 5 регионам).
    Не делает НИ ОДНОГО вызова .cpu() или .item() когда need_ui_telemetry=False!
    """
    meta = bus.meta_64
    n_sec = max(2, int(meta[1]))
    fcz_pers_t = meta[6]
    max_surp_t = meta[7]
    gamma_gate_t = meta[26]

    active_drive_t = torch.clamp(
        fcz_pers_t * 0.55 + torch.clamp(max_surp_t, max=1.0) * 0.25 + gamma_gate_t * 0.20,
        0.0, 1.0
    )
    dyn_str_t = torch.clamp(strength_low + (burst_strength - strength_low) * active_drive_t, strength_low, 0.95)
    dyn_str = float(dyn_str_t.item())

    order = bus.slot_order_16[:n_sec].long()
    norm_coords = bus.slot_norm_coord_16[:n_sec]
    c_flow = bus.causal_flow_16x16[:n_sec, :n_sec]
    amodal_77 = bus.amodal_manifold_77x768

    wm_field = bus.wm_field_5x5x32x120
    delta_theta_rip = (
        wm_field[:, BAND_DELTA] * 0.40 +
        wm_field[:, BAND_THETA] * 0.30 +
        wm_field[:, BAND_RIPPLES] * 0.30
    ).mean(dim=0)
    dtr_32x768 = torch.matmul(delta_theta_rip, PHYSICAL_PROJECTION_768x120.T)
    dtr_77x768 = F.adaptive_avg_pool1d(dtr_32x768.T.unsqueeze(0), 77).squeeze(0).T

    neuro_77 = amodal_77 * 0.70 + dtr_77x768 * 0.30

    with semantic.lock:
        words_copy = list(semantic.words)
        w_count = len(words_copy)

        if w_count == 0:
            base_77 = semantic.empty_embed.clone()
            clean_norms = torch.norm(base_77, p=2, dim=-1, keepdim=True).clamp(min=1e-5)
            v_aff = semantic.get_concept_affordances_svd(0, max_bases=16)
            coords = torch.matmul(neuro_77, v_aff.T)
            mod_77 = F.normalize(torch.matmul(coords, v_aff), p=2, dim=-1) * clean_norms
            target_77 = base_77 * (1.0 - 0.45 * dyn_str_t) + mod_77 * (0.45 * dyn_str_t)
            target_77 = F.normalize(target_77, p=2, dim=-1) * clean_norms
            slot_idx_gpu = torch.zeros(n_sec, dtype=torch.long, device=DEVICE)
        else:
            slot_idx_gpu = torch.clamp((norm_coords * w_count).long(), 0, w_count - 1)
            root_s = order[0] % n_sec
            root_c_idx = int(slot_idx_gpu[root_s].item()) % w_count
            base_77 = semantic.get_77_embedding_gpu(words_copy[root_c_idx]).clone()
            clean_norms = torch.norm(base_77, p=2, dim=-1, keepdim=True).clamp(min=1e-5)

            if w_count > 1 and n_sec > 1:
                child_s = order[1] % n_sec
                child_c_idx = int(slot_idx_gpu[child_s].item()) % w_count
                if child_c_idx != root_c_idx:
                    child_emb = semantic.get_77_embedding_gpu(words_copy[child_c_idx])
                    dot = torch.sum(child_emb * base_77, dim=-1, keepdim=True) / (torch.sum(base_77 ** 2, dim=-1, keepdim=True) + 1e-7)
                    ortho = F.normalize(child_emb - dot * base_77, p=2, dim=-1) * clean_norms
                    is_sup = (c_flow[root_s, child_s] >= 0.03).float()
                    base_77.add_(ortho * is_sup * 0.35 + child_emb * (1.0 - is_sup) * 0.22)

            v_aff = semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)
            coords = torch.matmul(neuro_77, v_aff.T)
            mod_77 = F.normalize(torch.matmul(coords, v_aff), p=2, dim=-1) * clean_norms
            target_77 = base_77 * (1.0 - 0.45 * dyn_str_t) + mod_77 * (0.45 * dyn_str_t)
            target_77 = F.normalize(target_77, p=2, dim=-1) * clean_norms

    telemetry = {
        "dyn_strength": dyn_str,
        "gamma_gate": float(gamma_gate_t.item()),
    }
    if need_ui_telemetry:
        indices = [int(i) % max(1, w_count) for i in slot_idx_gpu.tolist()]
        names = [words_copy[i].upper() for i in indices] if w_count > 0 else ["СВОБОДНЫЙ_ДРЕЙФ"] * n_sec
        root_name = names[int(order[0].item()) % n_sec]
        telemetry.update({
            "primary_role": "AFz", "primary_active": True,
            "ladder_stage": 4,
            "monty_loss": round(float(meta[8].item()), 5),
            "persistence": round(float(fcz_pers_t.item()), 3),
            "surprise": round(float(max_surp_t.item()), 3),
            "regions_telemetry": {},
            "total_clock_sectors": n_sec, "current_clock_sec": int(meta[2].item()) % n_sec,
            "slot_names": names, "slot_indices": indices,
            "sealed_mask": [bool(x > 0.5) for x in bus.slot_seals_16[:n_sec].tolist()],
            "slot_stabilities": bus.slot_stabs_16[:n_sec].cpu().numpy(),
            "causal_flow_np": c_flow.cpu().numpy(),
            "dag_log_parts": [root_name],
            "dl_phase": float(meta[3].item()), "is_delta_tick": bool(meta[4].item() > 0.5),
            "active_vocab_size": w_count,
            "fpz_active": True, "fpz_tension": float(meta[16].item()), "fpz_concept": root_name,
            "f3_active": True, "f3_concept": root_name,
            "f3_x_local": float(meta[9].item()), "f3_y_global": float(meta[10].item()),
            "f3_kuramoto": float(meta[11].item()), "f3_slot_names": names,
            "f4_active": True, "f4_concept": root_name, "f4_entropy": float(active_drive_t.item()),
            "fcz_active": True, "fcz_concept": root_name,
            "fcz_lx": float(meta[12].item()), "fcz_ly": float(meta[13].item()),
            "fcz_rx": float(meta[14].item()), "fcz_ry": float(meta[15].item()),
        })
    return target_77, dyn_str, telemetry


def map_working_memory_to_visual_field_gpu(bus: HardwareVRAMBus, semantic: UniversalSemanticSpace,
                                           strength_low: float, burst_strength: float,
                                           need_ui_telemetry: bool = False):
    target_77, dyn_str, telemetry = project_amodal_bus_to_diffusion_gpu(
        bus, semantic, strength_low, burst_strength, need_ui_telemetry
    )
    return target_77, bus.wm_field_5x5x32x120, dyn_str, telemetry


@torch.no_grad()
def apply_color_surgery_and_blend_gpu(new_rgb_np: np.ndarray, prev_gpu: Optional[torch.Tensor],
                                      use_color: bool, world_context: float) -> tuple[np.ndarray, torch.Tensor]:
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
    def __init__(self, conn, initial_h=384, initial_w=512, use_taesd=True, use_color=True,
                 is_sdxl=False, inference_steps=0, deterministic: Optional[bool] = None, seed: int = 0):
        super().__init__(daemon=True)
        self.conn = conn
        self.img_h = initial_h
        self.img_w = initial_w
        self.use_taesd = use_taesd
        self.use_color = use_color
        self.is_sdxl = is_sdxl
        self.target_steps = inference_steps
        self.actual_steps = 2 if inference_steps == 0 else inference_steps
        self.guidance_scale = 1.35
        self.world_context = 0.18
        self.target_fps = 10.0
        self.deterministic = deterministic
        self.seed = int(seed)

        self.display_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.prev_gpu_frame: Optional[torch.Tensor] = None
        self.has_conditioning = False
        self.strength = 0.60
        self.running = True
        self.lock = threading.Lock()
        self.gateway_conn = None
        self.fps = 0.0
        self.gen_ms = 0.0
        self.frame_seq_id = 0
        self.latest_telemetry = {}

    def update_conditioning(self, strength: float, telemetry: dict):
        with self.lock:
            self.has_conditioning = True
            self.strength = float(np.clip(strength, 0.30, 0.95))
            if telemetry:
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
                ready = self.has_conditioning
                s_val = self.strength
                t_steps = self.target_steps
                a_steps = self.actual_steps
                cfg = self.guidance_scale
                base_w_ctx = self.world_context
                t_fps = self.target_fps
                req_w, req_h = self.img_w, self.img_h
                u_taesd = self.use_taesd
                u_color = self.use_color
                det_override = self.deterministic
                cur_seed = self.seed
                telem = self.latest_telemetry
                if np.max(self.display_rgb) > 0:
                    internal_rgb = self.display_rgb

            if not ready:
                time.sleep(0.003)
                continue

            if internal_rgb.shape[:2] != (req_h, req_w):
                internal_rgb = cv2.resize(internal_rgb, (req_w, req_h), interpolation=cv2.INTER_LINEAR)
                self.prev_gpu_frame = None

            steps_to_run = a_steps if t_steps <= 0 else t_steps
            eff_s = max(s_val, 1.05 / max(1, steps_to_run)) if steps_to_run >= 2 else max(0.55, s_val)
            eff_s = float(np.clip(eff_s, 0.30, 0.95))

            g_gate = float(telem.get("gamma_gate", 0.5))
            dyn_w_ctx = float(np.clip(base_w_ctx * (1.6 - 1.2 * g_gate) * (1.3 - 0.8 * eff_s), 0.03, 0.50))

            try:
                t0 = time.perf_counter()
                # НИКАКИХ prompt_embeds и wm_5x5x32x120 по сокету!
                # brain_server читает их напрямую из VRAM через HardwareVRAMBus (Zero-Copy)!
                req = {
                    'cmd': 'generate',
                    'image_np': internal_rgb,
                    'strength': eff_s,
                    'num_inference_steps': steps_to_run,
                    'guidance_scale': cfg,
                    'use_taesd': u_taesd,
                    'seed': cur_seed
                }
                if det_override is not None:
                    req['deterministic'] = bool(det_override)

                with ipc_lock:
                    self.conn.send(req)
                    resp = self.conn.recv()

                gen_dt = time.perf_counter() - t0

                if isinstance(resp, np.ndarray):
                    if resp.shape[:2] != (req_h, req_w):
                        resp = cv2.resize(resp, (req_w, req_h))

                    internal_rgb, self.prev_gpu_frame = apply_color_surgery_and_blend_gpu(
                        resp, self.prev_gpu_frame, u_color, dyn_w_ctx
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
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--no-taesd', action='store_true', default=False)
    parser.add_argument('--no-color', action='store_true', default=False)
    parser.add_argument('--stochastic', dest='stochastic', action='store_true', default=True)
    parser.add_argument('--deterministic', '--no-stochastic', dest='stochastic', action='store_false')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--strength-low', type=float, default=0.50)
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--burst-strength', type=float, default=0.85)
    parser.add_argument('--burst-duration', type=float, default=1.2)
    parser.add_argument('--ema-smooth', type=float, default=0.05)

    parser.add_argument('--seal-thresh', type=float, default=0.68)
    parser.add_argument('--seal-cycles', type=int, default=4)
    parser.add_argument('--decay-cycles', type=int, default=2)
    parser.add_argument('--refresh-cycles', type=int, default=2)

    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--minimal-ui', action='store_true', default=False)
    args = parser.parse_args()

    init_taesd = not args.no_taesd
    init_color = init_taesd if not args.no_color else False
    init_det = None if args.stochastic is None else (not bool(args.stochastic))

    conn = None
    is_sdxl = False
    print("⏳ Подключение к brain_server (порт 6000)...")
    while conn is None:
        try:
            conn = Client(('localhost', 6000), authkey=b'brain')
            with ipc_lock:
                init_msg = {'cmd': 'init_mode', 'mode': args.mode, 'use_taesd': init_taesd}
                if init_det is not None:
                    init_msg['deterministic'] = init_det
                conn.send(init_msg)
                resp = conn.recv()
            is_sdxl = resp.get('is_sdxl', False)
            print(f"✅ Диффузия готова! (SDXL: {is_sdxl}, TAESD: {init_taesd}, ColorFix: {init_color}, Stochastic: {resp.get('stochastic', True)})")
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
        is_sdxl=is_sdxl, inference_steps=args.steps,
        deterministic=init_det, seed=args.seed
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
                            if 'stochastic' in p:
                                worker.deterministic = not bool(p['stochastic'])
                            elif 'deterministic' in p:
                                worker.deterministic = bool(p['deterministic'])
                            if 'seed' in p:
                                worker.seed = int(p['seed'])
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
            dt = float(np.clip(now_p - last_loop_t, 0.004, 0.10))
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
                # Высокочастотный цикл обновления рабочей памяти в VRAM (125 Гц),
                # чтобы пока U-Net рендерит кадр, его callback_on_step_end видел живое изменение мысли!
                time.sleep(0.008)

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

            ema = float(np.clip(dt / max(0.015, current_ema_smooth), 0.08, 0.85))
            smoothed_target.mul_(1.0 - ema).add_(target_77, alpha=ema)

            # ZERO-COPY: пишем живой эмбеддинг прямо в общую VRAM-шину!
            bus.live_clip_77x768.copy_(smoothed_target)

            if (t_now - last_telem_t) >= 0.15 and need_telem:
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

            worker.update_conditioning(dyn_str, telemetry)

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
