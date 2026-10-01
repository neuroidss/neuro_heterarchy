#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: REAL-TIME MUSICGEN GENERATIVE RUNNER (SAFE DECOUPLED ORCHESTRATOR)
===================================================================================
"""

from __future__ import annotations
import os
import sys
import time
import queue
import argparse
import traceback
import threading
import numpy as np
import sounddevice as sd
from pathlib import Path
from multiprocessing.connection import Client

try:
    os.setpriority(os.PRIO_PROCESS, 0, -20)
    print("⚡ [System] Auto-Priority applied: NI -20 (Real-Time PRI 0)", flush=True)
except PermissionError:
    try: os.setpriority(os.PRIO_PROCESS, 0, -10)
    except Exception: pass

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path: sys.path.insert(0, str(p))

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE
from neuro_genesis_engine import SelfSufficientHeterarchyRouter, MAX_SLOTS_CAPACITY
from music_genesis_engine import UniversalMusicSemanticSpace, MusicHeterarchyDeltaProcessor

SAMPLE_RATE = 32000
audio_playback_queue = queue.Queue(maxsize=50)
stop_playback = threading.Event()

def audio_playback_worker(target_buf=2):
    while audio_playback_queue.qsize() < target_buf and not stop_playback.is_set():
        time.sleep(0.05)
    print("🔈 [Audio Player] Вывод звука 32 kHz активен.", flush=True)
    try:
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', latency='high') as stream:
            while not stop_playback.is_set():
                try:
                    chunk = audio_playback_queue.get(timeout=0.5)
                    stream.write(np.ascontiguousarray(chunk).tobytes())
                except queue.Empty: pass
                except Exception: time.sleep(0.02)
    except Exception as e:
        print(f"⚠️ [Audio Player Note]: {e}", flush=True)

class AsyncMusicWorker(threading.Thread):
    def __init__(self, conn, runtime_cfg, gateway_port=6003):
        super().__init__(daemon=True)
        self.conn = conn
        self.cfg = runtime_cfg
        self.running = True
        self.lock = threading.Lock()
        self.latest_embeds = None
        self.block_sec = runtime_cfg['target_block']
        self.is_drop = False
        self.gateway_conn = None
        self.gateway_port = gateway_port
        self.latest_pcm = np.zeros(0, dtype=np.float32)
        self.rtf = 0.0
        self.play_speed = 1.0

    def update_conditioning(self, embeds, block_sec, is_drop):
        with self.lock:
            self.latest_embeds = embeds
            self.block_sec = block_sec
            if is_drop: self.is_drop = True

    def run(self):
        while self.running:
            with self.lock:
                emb = self.latest_embeds
                b_sec = self.block_sec
                c_sec = self.cfg['target_context']
                t_val = self.cfg['temp']
                k_val = self.cfg['top_k']
                cfg_c = self.cfg['cfg_coef']
                safety = self.cfg['speed_safety_factor']
                drop_val = self.is_drop
                self.is_drop = False

            if emb is None:
                time.sleep(0.01)
                continue

            try:
                self.conn.send({
                    'cmd': 'generate_step',
                    't5_embeds': emb,
                    'block_sec': b_sec,
                    'context_sec': c_sec,
                    'temp': t_val,
                    'top_k': k_val,
                    'cfg_coef': cfg_c,
                    'safety_factor': safety,
                    'is_reset': drop_val
                })
                resp = self.conn.recv()
                if 'pcm' in resp:
                    pcm = resp['pcm']
                    self.rtf = resp['rtf']
                    self.play_speed = resp.get('play_speed', 1.0)
                    with self.lock: self.latest_pcm = pcm

                    if not audio_playback_queue.full():
                        audio_playback_queue.put(pcm)

                    if self.gateway_conn is None:
                        try: self.gateway_conn = Client(('localhost', self.gateway_port), authkey=b'audio')
                        except Exception: self.gateway_conn = None

                    if self.gateway_conn is not None:
                        try:
                            self.gateway_conn.send(pcm.tobytes())
                            if self.gateway_conn.poll(0.0):
                                inc = self.gateway_conn.recv()
                                if isinstance(inc, dict) and inc.get('cmd') == 'update_config':
                                    with self.lock:
                                        self.cfg.update(inc.get('config', {}))
                                        print(f"🎛️ [WEB НАСТРОЙКИ ПРИНЯТЫ]: {self.cfg}")
                        except Exception:
                            self.gateway_conn = None

            except Exception:
                time.sleep(0.05)

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Dynamic MusicGen Heterarchy")
    parser.add_argument('--lore', type=str, default="music_lore.txt")
    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--gamma-max', type=float, default=100.0)
    parser.add_argument('--seal-thresh', type=float, default=0.68)
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="semantic", choices=["semantic", "turbo"])
    parser.add_argument('--gateway-port', type=int, default=6003)

    parser.add_argument('--target-block', type=float, default=2.4)
    parser.add_argument('--target-context', type=float, default=0.4)
    parser.add_argument('--target-buffer-count', type=int, default=2)
    parser.add_argument('--speed-safety-factor', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=250)
    parser.add_argument('--cfg-coef', type=float, default=1.0)
    args = parser.parse_args()

    runtime_config = {
        'target_block': args.target_block,
        'target_context': args.target_context,
        'target_buffer_count': args.target_buffer_count,
        'speed_safety_factor': args.speed_safety_factor,
        'temp': args.temp,
        'top_k': args.top_k,
        'cfg_coef': args.cfg_coef
    }

    print("=" * 70)
    print(f"🧠 NEUROCANVAS RUNNER (MODE: {args.mode.upper()})")
    print(f"🎛️ Config: {runtime_config}")
    print("=" * 70)

    print("⏳ Подключение к musicgen_server (порт 6005)...", flush=True)
    conn = None
    while conn is None:
        try:
            conn = Client(('localhost', 6005), authkey=b'music')
            print("✅ MusicGen сервер готов к работе!", flush=True)
        except Exception:
            time.sleep(0.5)

    def load_lore_from_server(lore_path):
        words = []
        if os.path.exists(lore_path):
            with open(lore_path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if not words: words = ["psytrance rolling bass", "ambient ethereal pads", "cyberpunk techno"]
        print(f"📤 Запрос предрассчета {len(words)} тем лора у сервера...", flush=True)
        conn.send({'cmd': 'init_lore', 'prompts': words})
        resp = conn.recv()
        return words, resp['c_bases'], resp['proj_120_to_768']

    words, c_bases, proj_120 = load_lore_from_server(args.lore)
    semantic = UniversalMusicSemanticSpace(words=words, c_bases=c_bases, proj_120_to_768=proj_120)
    router = SelfSufficientHeterarchyRouter(args.users)
    processor = MusicHeterarchyDeltaProcessor(semantic=semantic, router=router, seal_thresh=args.seal_thresh)

    hud = None
    if not args.headless:
        from music_hud import NeuroMusicHUD
        hud = NeuroMusicHUD(device=DEVICE)

    threading.Thread(target=audio_playback_worker, args=(runtime_config['target_buffer_count'],), daemon=True).start()

    engine = HeterarchicalBrainEngine(gamma_max=args.gamma_max)
    engine.start()

    worker = AsyncMusicWorker(conn, runtime_config, gateway_port=args.gateway_port)
    worker.start()

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
                            eng.seals.mask = [False] * MAX_SLOTS_CAPACITY
                        print("🔥 [СБРОС ПЕЧАТЕЙ] Все печати растворены.", flush=True)
                    elif cmd == "adj_param":
                        p_name, p_delta = payload
                        if p_name in runtime_config:
                            if isinstance(runtime_config[p_name], int):
                                runtime_config[p_name] = max(1, runtime_config[p_name] + int(p_delta))
                            else:
                                runtime_config[p_name] = max(0.05, round(runtime_config[p_name] + p_delta, 3))
                            print(f"🎛️ [HUD НАСТРОЙКА] {p_name} ➔ {runtime_config[p_name]}")

            frame = engine.get_frame()
            if frame.num_live == 0:
                time.sleep(0.01)
                continue

            target_out, b_sec, c_sec, t_val, k_val, cfg_c, telem = processor.step(
                frame, dt,
                mode=args.mode,
                target_block=runtime_config['target_block'],
                target_context=runtime_config['target_context'],
                base_temp=runtime_config['temp'],
                base_top_k=runtime_config['top_k'],
                cfg_coef=runtime_config['cfg_coef']
            )

            worker.update_conditioning(target_out.cpu().numpy(), b_sec, telem.get("is_drop", False))

            if hud:
                with worker.lock: pcm_chunk = worker.latest_pcm
                hud.push_pcm(pcm_chunk)
                hud.render(telem, rtf=worker.rtf, dt=dt, runtime_cfg=runtime_config)

    except KeyboardInterrupt:
        print("\n[NeuroGen] Остановка пользователем (Ctrl+C)...", flush=True)
    except Exception as e:
        print(f"\n❌ [ОШИБКА РАННЕРА]: {e}")
        traceback.print_exc()
    finally:
        stop_playback.set()
        worker.running = False
        engine.stop()
        if hud: hud.close()

if __name__ == '__main__':
    main()
