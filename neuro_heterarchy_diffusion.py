#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ZERO-REDUCTION REGIONAL DIFFUSION RUNNER (TAESD COLOR SURGERY RESTORED)
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
from diffusion_semantic_space import UniversalSemanticSpace
from neuro_amodal_engine import (
    SharedAmodalCognitiveEngine,
    SelfSufficientHeterarchyRouter,
    PHYSICAL_PROJECTION_768x120,
)
from neuro_vram_bus import (
    HardwareVRAMBus,
    MONTY_CODE_TO_STR,
    universal_full_rank_manifold_step,
)
from neuro_recorder import recorder_instance

ipc_lock = threading.Lock()
CORE_REGIONS = ["FPZ", "AFZ", "F3", "F4", "FCZ"]


@torch.no_grad()
def project_amodal_bus_to_diffusion_gpu(bus: HardwareVRAMBus, semantic: UniversalSemanticSpace,
                                        strength_low: float, burst_strength: float,
                                        stochastic: float = 0.0,
                                        unassigned_stochastic: float = 1.0,
                                        need_ui_telemetry: bool = False) -> tuple[torch.Tensor, float, dict]:
    meta = bus.meta_64
    n_sec = max(2, int(meta[1]))
    monty_mode_code = int(round(float(meta[27].item())))
    monty_mode_str = MONTY_CODE_TO_STR.get(monty_mode_code, "missing_only")

    c_flow = bus.causal_flow_16x16[:n_sec, :n_sec]
    wm_field = bus.wm_field_5x5x32x120
    proj_T = PHYSICAL_PROJECTION_768x120.T

    reg_waves_32x768 = {
        "FPZ": torch.matmul(wm_field[4].mean(dim=0), proj_T),
        "AFZ": torch.matmul(wm_field[0].mean(dim=0), proj_T),
        "F3":  torch.matmul(wm_field[2].mean(dim=0), proj_T),
        "F4":  torch.matmul(wm_field[3].mean(dim=0), proj_T),
        "FCZ": torch.matmul(wm_field[1].mean(dim=0), proj_T),
    }

    stoch_act = float(np.clip(stochastic, 0.0, 1.0))
    stoch_unass = float(np.clip(unassigned_stochastic, 0.0, 1.0))

    m_afz = 1.0 if float(meta[21 + 0].item()) > 0.5 else (0.0 if monty_mode_str == "off" else 1.0)
    m_fcz = 1.0 if float(meta[21 + 1].item()) > 0.5 else (0.0 if monty_mode_str == "off" else 1.0)
    m_f3  = 1.0 if float(meta[21 + 2].item()) > 0.5 else (0.0 if monty_mode_str == "off" else 1.0)
    m_f4  = 1.0 if float(meta[21 + 3].item()) > 0.5 else (0.0 if monty_mode_str == "off" else 1.0)
    m_fpz = 1.0 if float(meta[21 + 4].item()) > 0.5 else (0.0 if monty_mode_str == "off" else 1.0)

    reg_stoch_map = {
        "FPZ": stoch_act if m_fpz > 0.5 else stoch_unass,
        "AFZ": stoch_act if m_afz > 0.5 else stoch_unass,
        "F3":  stoch_act if m_f3  > 0.5 else stoch_unass,
        "F4":  stoch_act if m_f4  > 0.5 else stoch_unass,
        "FCZ": stoch_act if m_fcz > 0.5 else stoch_unass,
    }

    with semantic.lock:
        words_copy = list(semantic.words)
        labels = [w.upper() for w in words_copy] if words_copy else ["EMPTY_MANIFOLD"]
        global_bank = (
            torch.stack([semantic.get_77_embedding_gpu(w) for w in words_copy], dim=0)
            if words_copy else semantic.empty_embed.unsqueeze(0)
        )
        K = global_bank.shape[0]

        composed_77 = torch.zeros((77, 768), dtype=torch.float32, device=DEVICE)
        for r in CORE_REGIONS:
            r_bank = semantic.region_banks.get(r, global_bank)
            r_stoch = reg_stoch_map[r]
            r_wm_77 = F.adaptive_avg_pool1d(reg_waves_32x768[r].T.unsqueeze(0), 77).squeeze(0).T

            if r_stoch >= 0.999:
                composed_77.add_(r_bank[0] * 0.20)
            else:
                r_out_77, _, _, _ = universal_full_rank_manifold_step(
                    anchor_bank_KxLxD=r_bank,
                    wm_tensor_LxD=r_wm_77,
                    gain_gamma=0.35,
                    stochastic=r_stoch,
                    temp=0.35
                )
                composed_77.add_(r_out_77 * 0.20)

        ref_norms = torch.norm(global_bank[0], p=2, dim=-1, keepdim=True).clamp(min=1e-4)
        target_77 = F.normalize(composed_77, p=2, dim=-1) * ref_norms

        z_slots_norm = F.normalize(bus.z_slots_16x768[:n_sec] + 1e-6, p=2, dim=-1)
        slot_sims = torch.mm(z_slots_norm, semantic.norm_vocab[:K].T)
        slot_indices = [int(i) % K for i in torch.argmax(slot_sims, dim=-1).tolist()]
        slot_names = [labels[i] for i in slot_indices]

    const_str = float(strength_low)
    root_label = labels[0]
    telemetry = {
        "dyn_strength": const_str,
        "monty_mode": monty_mode_str,
        "dag_log_parts": [root_label],
    }
    if need_ui_telemetry:
        telemetry.update({
            "primary_role": "AFz", "primary_active": True,
            "ladder_stage": 4,
            "monty_loss": round(float(meta[8].item()), 5),
            "persistence": round(float(meta[6].item()), 3),
            "surprise": round(float(meta[7].item()), 3),
            "regions_telemetry": {},
            "total_clock_sectors": n_sec, "current_clock_sec": int(meta[2].item()) % n_sec,
            "slot_names": slot_names, "slot_indices": slot_indices,
            "sealed_mask": [bool(x > 0.5) for x in bus.slot_seals_16[:n_sec].tolist()],
            "slot_stabilities": bus.slot_stabs_16[:n_sec].cpu().numpy(),
            "causal_flow_np": c_flow.cpu().numpy(),
            "dl_phase": float(meta[3].item()), "is_delta_tick": bool(meta[4].item() > 0.5),
            "active_vocab_size": len(words_copy),
            "fpz_active": True, "fpz_tension": float(meta[16].item()), "fpz_concept": root_label,
            "f3_active": True, "f3_concept": root_label,
            "f3_x_local": float(meta[9].item()), "f3_y_global": float(meta[10].item()),
            "f3_kuramoto": float(meta[11].item()), "f3_slot_names": slot_names,
            "f4_active": True, "f4_concept": root_label, "f4_entropy": 0.5,
            "fcz_active": True, "fcz_concept": root_label,
            "fcz_lx": float(meta[12].item()), "fcz_ly": float(meta[13].item()),
            "fcz_rx": float(meta[14].item()), "fcz_ry": float(meta[15].item()),
        })
    return target_77, const_str, telemetry


def map_working_memory_to_visual_field_gpu(bus: HardwareVRAMBus, semantic: UniversalSemanticSpace,
                                           strength_low: float, burst_strength: float,
                                           stochastic: float = 0.0,
                                           unassigned_stochastic: float = 0.0,
                                           need_ui_telemetry: bool = False):
    target_77, dyn_str, telemetry = project_amodal_bus_to_diffusion_gpu(
        bus, semantic, strength_low, burst_strength, stochastic, unassigned_stochastic, need_ui_telemetry
    )
    return target_77, bus.wm_field_5x5x32x120, dyn_str, telemetry


@torch.no_grad()
def apply_color_surgery_and_blend_gpu(new_rgb_np: np.ndarray, prev_gpu: Optional[torch.Tensor],
                                      use_color: bool, world_context: float) -> tuple[np.ndarray, torch.Tensor]:
    """
    НЕЗАМЕНИМЫЙ БЛОК УСКОРЕНИЯ TAESD:
    Устраняет накопление паразитного зелёного оттенка декодера TAESD
    и согласует контраст/динамический диапазон между последовательными шагами.
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
        keep = min(0.55, max(0.0, float(world_context)))
        cur_t = cur_t * (1.0 - keep) + prev_gpu * keep

    cur_t = torch.clamp(cur_t, 0.0, 255.0)
    out_np = cur_t.to(torch.uint8).cpu().numpy()
    return out_np, cur_t


class AsyncDiffusionWorker(threading.Thread):
    def __init__(self, conn, initial_h=384, initial_w=512, use_taesd=True, use_color=True,
                 is_sdxl=False, inference_steps=0, stochastic: float = 0.0, unassigned_stochastic: float = 1.0, seed: int = 0):
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
        self.stochastic_level = float(np.clip(stochastic, 0.0, 1.0))
        self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        self.seed = int(seed)

        self.display_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.prev_gpu_frame: Optional[torch.Tensor] = None
        self.latest_embeds = None
        self.latest_wm_5x5x32x120 = None
        self.strength = 0.60
        self.running = True
        self.lock = threading.Lock()
        self.gateway_conn = None
        self.fps = 0.0
        self.gen_ms = 0.0
        self.frame_seq_id = 0
        self.latest_telemetry = {}

    def update_conditioning(self, prompt_embeds_np, strength: float, telemetry: dict, wm_5x5x32x120_np=None):
        with self.lock:
            self.latest_embeds = prompt_embeds_np
            if wm_5x5x32x120_np is not None:
                self.latest_wm_5x5x32x120 = wm_5x5x32x120_np
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
                embeds = self.latest_embeds
                wm_np = self.latest_wm_5x5x32x120
                s_val = self.strength
                t_steps = self.target_steps
                a_steps = self.actual_steps
                cfg = self.guidance_scale
                base_w_ctx = self.world_context
                t_fps = self.target_fps
                req_w, req_h = self.img_w, self.img_h
                u_taesd = self.use_taesd
                u_color = self.use_color
                stoch_val = float(self.stochastic_level)
                unass_stoch_val = float(self.unassigned_stochastic)
                cur_seed = self.seed
                telem = self.latest_telemetry
                if np.max(self.display_rgb) > 0:
                    internal_rgb = self.display_rgb

            if embeds is None:
                time.sleep(0.003)
                continue

            if internal_rgb.shape[:2] != (req_h, req_w):
                internal_rgb = cv2.resize(internal_rgb, (req_w, req_h), interpolation=cv2.INTER_LINEAR)
                self.prev_gpu_frame = None

            steps_to_run = a_steps if t_steps <= 0 else t_steps
            eff_s = float(np.clip(s_val, 0.30, 0.95))
            dyn_w_ctx = float(np.clip(base_w_ctx, 0.0, 0.60))

            try:
                t0 = time.perf_counter()
                req = {
                    'cmd': 'generate',
                    'image_np': internal_rgb,
                    'prompt_embeds': embeds,
                    'strength': eff_s,
                    'num_inference_steps': steps_to_run,
                    'guidance_scale': cfg,
                    'use_taesd': u_taesd,
                    'stochastic': stoch_val,
                    'unassigned_stochastic': unass_stoch_val,
                    'seed': cur_seed
                }
                if wm_np is not None:
                    req['wm_5x5x32x120'] = wm_np

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
    parser.add_argument('--monty-mode', type=str, default="missing_only", choices=["off", "missing_only", "full"])
    parser.add_argument('--no-monty', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo", "sdxl"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--no-taesd', action='store_true', default=False)
    parser.add_argument('--no-color', action='store_true', default=False)
    parser.add_argument('--stochastic', type=float, default=0.0)
    parser.add_argument('--unassigned-stochastic', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strength-low', type=float, default=0.55)
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--burst-strength', type=float, default=0.85)
    parser.add_argument('--burst-duration', type=float, default=1.2)
    parser.add_argument('--ema-smooth', type=float, default=0.08)
    parser.add_argument('--seal-thresh', type=float, default=1.0)
    parser.add_argument('--seal-cycles', type=int, default=4)
    parser.add_argument('--decay-cycles', type=int, default=2)
    parser.add_argument('--refresh-cycles', type=int, default=2)
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--minimal-ui', action='store_true', default=False)
    args = parser.parse_args()

    init_monty_mode = "off" if args.no_monty else args.monty_mode
    init_taesd = not args.no_taesd
    init_color = init_taesd if not args.no_color else False
    init_stoch = float(np.clip(args.stochastic, 0.0, 1.0))
    init_unass_stoch = float(np.clip(args.unassigned_stochastic, 0.0, 1.0))

    conn = None
    is_sdxl = False
    while conn is None:
        try:
            conn = Client(('localhost', 6000), authkey=b'brain')
            with ipc_lock:
                init_msg = {'cmd': 'init_mode', 'mode': args.mode, 'use_taesd': init_taesd, 'stochastic': init_stoch}
                conn.send(init_msg)
                resp = conn.recv()
            is_sdxl = resp.get('is_sdxl', False)
        except Exception:
            time.sleep(0.5)

    target_k = args.vocab_size if args.vocab_size is not None else (int(49400 * args.vocab_ratio) if args.vocab_ratio < 1.0 else None)
    semantic = UniversalSemanticSpace(lore_file=args.lore_file, concepts_cli=args.concepts, target_k=target_k)
    router = SelfSufficientHeterarchyRouter(args.users)

    amodal_engine = SharedAmodalCognitiveEngine(
        router=router,
        gamma_max=args.gamma_max,
        strength_low=args.strength_low,
        strength_high=args.burst_strength,
        monty_mode=init_monty_mode
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
        stochastic=init_stoch, unassigned_stochastic=init_unass_stoch, seed=args.seed
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
                        with semantic.lock:
                            if semantic.words:
                                ctrl_conn.send({
                                    'cmd': 'sync_from_runner',
                                    'concepts': list(semantic.words)
                                })

                has_msg = False
                with ctrl_lock:
                    if ctrl_conn is not None and ctrl_conn.poll(0.02):
                        msg = ctrl_conn.recv()
                        has_msg = True

                if has_msg:
                    cmd = msg.get('cmd')
                    if cmd == 'update_concepts':
                        new_words = msg.get('concepts', [])
                        if msg.get('is_initial_sync') and not new_words and semantic.words:
                            with ctrl_lock:
                                if ctrl_conn is not None:
                                    ctrl_conn.send({
                                        'cmd': 'sync_from_runner',
                                        'concepts': list(semantic.words)
                                    })
                        else:
                            semantic.update_concepts_on_the_fly(new_words)
                    elif cmd == 'set_device_mapping':
                        s_name = str(msg.get('stream_name', '')).strip()
                        r_val = str(msg.get('role', 'FCz')).strip()
                        u_val = str(msg.get('user', 'User1')).strip()
                        if s_name:
                            router.set_device_mapping(s_name, r_val, u_val)
                    elif cmd == 'set_device_role':
                        new_role = str(msg.get('role', 'FCz')).strip()
                        router.set_role(0, new_role)
                    elif cmd == 'update_params':
                        p = msg.get('params', {})
                        if 'monty_mode' in p:
                            amodal_engine.set_monty_mode(str(p['monty_mode']))
                        if 'mode' in p and p['mode'] != args.mode:
                            new_mode = str(p['mode'])
                            with ipc_lock:
                                conn.send({'cmd': 'init_mode', 'mode': new_mode, 'use_taesd': worker.use_taesd, 'stochastic': worker.stochastic_level})
                                resp = conn.recv()
                            args.mode = new_mode
                            with worker.lock:
                                worker.is_sdxl = resp.get('is_sdxl', False)
                        with worker.lock:
                            if 'stochastic' in p:
                                worker.stochastic_level = float(np.clip(float(p['stochastic']), 0.0, 1.0))
                            if 'unassigned_stochastic' in p:
                                worker.unassigned_stochastic = float(np.clip(float(p['unassigned_stochastic']), 0.0, 1.0))
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
                time.sleep(0.008)

            bus, _ = amodal_engine.step(dt)

            need_telem = (hud is not None) or ((t_now - last_telem_t) >= 0.15) or recorder_instance.is_recording
            target_77, dyn_str, telemetry = project_amodal_bus_to_diffusion_gpu(
                bus, semantic, strength_low=strength_low, burst_strength=burst_strength,
                stochastic=worker.stochastic_level,
                unassigned_stochastic=worker.unassigned_stochastic,
                need_ui_telemetry=need_telem
            )

            if worker.stochastic_level >= 0.999 and worker.unassigned_stochastic >= 0.999:
                smoothed_target.copy_(target_77)
            else:
                ema = float(np.clip(dt / max(0.015, current_ema_smooth), 0.08, 0.85))
                smoothed_target.mul_(1.0 - ema).add_(target_77, alpha=ema)
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
                                "stochastic": round(worker.stochastic_level, 2),
                                "unassigned_stochastic": round(worker.unassigned_stochastic, 2),
                                "ladder_stage": 4,
                                "monty_loss": telemetry.get("monty_loss", 0.0),
                                "persistence": telemetry.get("persistence", 0.0),
                                "surprise": telemetry.get("surprise", 0.0),
                                "monty_mode": telemetry.get("monty_mode", amodal_engine.monty_mode),
                                "regions": {},
                                "gpu_sat_mask": [False] * 16
                            })
                        except Exception:
                            ctrl_conn = None

            worker.update_conditioning(
                smoothed_target.unsqueeze(0).cpu().numpy(),
                dyn_str,
                telemetry,
                wm_5x5x32x120_np=bus.wm_field_5x5x32x120.detach().cpu().numpy()
            )

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
