#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: PURE UNCONSTRAINED MULTIDIMENSIONAL HETERARCHY BCI (TBT 2.0)
===================================================================================
- ТОЧКА ВХОДА И ДИСПЕТЧЕР КОНВЕЙЕРА.
- НАСТРОЙКА ГАММЫ: --gamma-max (по умолчанию 100.0 Гц).
- Все вычисления инкапсулированы в neuro_genesis_engine.py (100% GPU, чистые фазы).
- Весь дебаг инкапсулирован в neuro_hud.py (декуплированная телеметрия).
- В режиме --headless работает БЕЗ импорта Pygame!
===================================================================================
"""

import os
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
    if p.exists() and str(p) not in sys.path: sys.path.insert(0, str(p))

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE
from neuro_workers import apply_color_surgery
from neuro_genesis_engine import (
    MAX_SLOTS_CAPACITY,
    UniversalSemanticSpace,
    SelfSufficientHeterarchyRouter,
    HeterarchyDeltaProcessor
)

ipc_lock = threading.Lock()

class AsyncDiffusionWorker(threading.Thread):
    def __init__(self, conn, initial_h=384, initial_w=512, use_color=True, is_sdxl=False, inference_steps=2):
        super().__init__(daemon=True)
        self.conn, self.img_h, self.img_w = conn, initial_h, initial_w
        self.use_color, self.is_sdxl = use_color, is_sdxl
        self.inference_steps = inference_steps
        self.display_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.latest_embeds = None
        self.strength = 0.60
        self.running = True
        self.lock = threading.Lock()
        self.gateway_conn = None
        self.fps = 0.0

    def update_conditioning(self, prompt_embeds_np, strength):
        with self.lock:
            self.latest_embeds = prompt_embeds_np
            self.strength = float(np.clip(strength, 0.35, 0.98))

    def flush(self):
        with self.lock: self.display_rgb.fill(0)

    def get_canvas(self):
        with self.lock: return self.display_rgb

    def run(self):
        times = []
        internal_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        while self.running:
            with self.lock:
                embeds = self.latest_embeds
                s_val = self.strength
                steps = self.inference_steps
                if np.max(self.display_rgb) > 0: internal_rgb = self.display_rgb.copy()

            if embeds is None: time.sleep(0.005); continue

            try:
                t0 = time.time()
                req = {
                    'cmd': 'generate',
                    'image_np': internal_rgb,
                    'prompt_embeds': embeds,
                    'strength': s_val,
                    'num_inference_steps': steps,
                    'guidance_scale': 1.0
                }
                with ipc_lock:
                    self.conn.send(req)
                    resp = self.conn.recv()

                if isinstance(resp, np.ndarray):
                    if resp.shape[:2] != (self.img_h, self.img_w): resp = cv2.resize(resp, (self.img_w, self.img_h))
                    internal_rgb = apply_color_surgery(resp, internal_rgb.astype(np.float32)) if (self.use_color and np.max(internal_rgb) > 0) else resp
                    with self.lock: self.display_rgb = internal_rgb

                    if self.gateway_conn is None:
                        try: self.gateway_conn = Client(('localhost', 6002), authkey=b'canvas')
                        except Exception: self.gateway_conn = None
                    if self.gateway_conn is not None:
                        try:
                            bgr = cv2.cvtColor(internal_rgb, cv2.COLOR_RGB2BGR)
                            _, enc_jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            self.gateway_conn.send(enc_jpg.tobytes())
                        except Exception:
                            try: self.gateway_conn.close()
                            except Exception: pass
                            self.gateway_conn = None

                    times.append(time.time() - t0)
                    if len(times) > 5: times.pop(0)
                    self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                time.sleep(0.05)

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Decoupled Runner")
    # Параметры лора и словаря
    parser.add_argument('--lore', '--concepts-file', dest='lore_file', type=str, default=None)
    parser.add_argument('--concepts', type=str, default=None)
    parser.add_argument('--vocab-ratio', type=float, default=1.0)
    parser.add_argument('--vocab-size', type=int, default=None)
    
    # Монтаж и частота дискретизации
    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    
    # Верхняя граница гаммы (по умолчанию 100.0 Гц)
    parser.add_argument('--gamma-max', type=float, default=100.0, 
                        help="Верхняя граница гамма-диапазона в Гц (по умолчанию 100.0)")
    
    # Режимы диффузии
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo", "sdxl"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--no-taesd', action='store_true', default=False)
    parser.add_argument('--no-color', action='store_true', default=False)

    # Параметры пластичности и всплесков
    parser.add_argument('--strength-low', type=float, default=0.55)
    parser.add_argument('--strength-high', type=float, default=0.75)
    parser.add_argument('--burst-strength', type=float, default=0.75)
    parser.add_argument('--burst-duration', type=float, default=0.0)

    # Параметры фазовой печати мира (World Seals)
    parser.add_argument('--seal-thresh', type=float, default=0.68, help="Порог косинусной стабильности для печати (1.0 = выключить)")
    parser.add_argument('--seal-cycles', type=int, default=4, help="Число тактов дельты для запечатывания")
    parser.add_argument('--decay-cycles', type=int, default=2, help="Число тактов нестабильности для растворения печати")
    parser.add_argument('--refresh-cycles', type=int, default=2, help="Размер буфера сравнения дельта-тактов")

    # Режимы интерфейса
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--minimal-ui', action='store_true', default=False)
    args = parser.parse_args()

    # 1. Подключение к brain_server (порт 6000)
    conn = None
    is_sdxl = False
    print("⏳ Подключение к brain_server (порт 6000)...")
    while conn is None:
        try:
            conn = Client(('localhost', 6000), authkey=b'brain')
            with ipc_lock:
                conn.send({'cmd': 'init_mode', 'mode': args.mode, 'use_taesd': not args.no_taesd})
                resp = conn.recv()
            is_sdxl = resp.get('is_sdxl', False)
            print(f"✅ Диффузия готова! (SDXL: {is_sdxl})")
        except Exception:
            time.sleep(0.5)

    # 2. Инициализация семантического пространства и независимого движка
    target_k = args.vocab_size if args.vocab_size is not None else (int(49400 * args.vocab_ratio) if args.vocab_ratio < 1.0 else None)
    semantic = UniversalSemanticSpace(lore_file=args.lore_file, concepts_cli=args.concepts, target_k=target_k)
    router = SelfSufficientHeterarchyRouter(args.users)
    
    processor = HeterarchyDeltaProcessor(
        semantic=semantic,
        router=router,
        strength_low=args.strength_low,
        burst_strength=args.burst_strength,
        burst_duration=args.burst_duration,
        seal_cycles=args.seal_cycles,
        decay_cycles=args.decay_cycles,
        seal_thresh=args.seal_thresh,
        refresh_cycles=args.refresh_cycles
    )

    hud = None
    if not args.headless:
        from neuro_hud import NeuroCanvasHUD
        hud = NeuroCanvasHUD(initial_minimal=args.minimal_ui, device=DEVICE)

    # Запуск ядра LSL с заданной верхней частотой гаммы
    print(f"⚡ [ГАММА-ДИАПАЗОН] Установлена верхняя граница: {args.gamma_max:.1f} Гц")
    engine = HeterarchicalBrainEngine(gamma_max=args.gamma_max)
    engine.start()

    canvas_w, canvas_h = (512, 384) if is_sdxl else ((448, 336) if args.speed == "fast" else (512, 384))
    worker = AsyncDiffusionWorker(conn, initial_h=canvas_h, initial_w=canvas_w, use_color=not args.no_color, is_sdxl=is_sdxl, inference_steps=args.steps)
    worker.start()

    smoothed_target = semantic.get_77_embedding_gpu(semantic.words[0]).clone()

    try:
        while True:
            dt = 0.016
            if hud:
                dt, commands = hud.poll_events(len(engine.get_frame().nodes))
                for cmd, payload in commands:
                    if cmd == "set_role": router.set_role(payload[0], payload[1])
                    elif cmd == "reset_seals":
                        for eng in processor.engines.values(): 
                            eng.seals.charge.fill(0)
                            eng.seals.mask = [False]*MAX_SLOTS_CAPACITY
                        hud.spiral.rings.clear()
                        print("🔥 [СБРОС ПЕЧАТЕЙ] Все печати растворены.")
                    elif cmd == "flush_canvas": 
                        worker.flush()
                        print("🧹 [ХОЛСТ ОЧИЩЕН] Сброс кадра.")

            frame = engine.get_frame()
            if frame.num_live == 0:
                if hud: hud.render_waiting()
                time.sleep(0.01)
                continue

            target_77, dyn_str, telemetry = processor.step(frame, dt)

            ema = float(np.clip(dt / 0.16, 0.10, 0.40))
            smoothed_target.mul_(1.0 - ema).add_(target_77, alpha=ema)
            worker.update_conditioning(smoothed_target.unsqueeze(0).cpu().numpy(), dyn_str)

            if hud:
                hud.render(worker.get_canvas(), telemetry, fps=worker.fps, dt=dt)

    except KeyboardInterrupt:
        pass
    finally:
        diff_worker.running = False
        engine.stop()
        if hud: hud.close()

if __name__ == '__main__':
    main()
