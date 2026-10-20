#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: REAL-TIME MUSICGEN GENERATIVE RUNNER (ZERO-DROP & AUTO-MINIMIZE BUFFER)
===================================================================================
- НИ ОДИН ЧАНК НЕ ВЫБРАСЫВАЕТСЯ (Zero Drop):
  Все сгенерированные сэмплы доигрываются до конца.
- Передает точный остаток очереди (queue_sec) на сервер, чтобы сервер сам
  подруливал скоростью (Speed 0.45x..1.18x) и авто-минимизировал размер блока!
- Полноразмерное сферическое EMA-сглаживание [64 × 768] с сохранением нормы T5.
===================================================================================
"""

from __future__ import annotations
import os
import sys
import time
import json
import queue
import argparse
import threading
import numpy as np
from pathlib import Path
from multiprocessing.connection import Client
import torch
import torch.nn.functional as F

try:
    os.setpriority(os.PRIO_PROCESS, 0, -20)
    print("⚡ [System] Auto-Priority applied: NI -20 (Real-Time PRI 0)", flush=True)
except PermissionError:
    try:
        os.setpriority(os.PRIO_PROCESS, 0, -10)
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE
from neuro_genesis_engine import SelfSufficientHeterarchyRouter, MAX_SLOTS_CAPACITY
from music_genesis_engine import UniversalMusicSemanticSpace, MusicHeterarchyDeltaProcessor

SAMPLE_RATE = 32000
audio_playback_queue = queue.Queue(maxsize=100)
stop_playback = threading.Event()

queued_samples_lock = threading.Lock()
queued_samples_total = 0


def push_audio_chunk_zero_drop(pcm: np.ndarray):
    """Кладет чанк в очередь БЕЗ выбрасывания предыдущих чанков!"""
    global queued_samples_total
    if pcm is None or len(pcm) == 0:
        return
    with queued_samples_lock:
        audio_playback_queue.put(pcm)
        queued_samples_total += len(pcm)


def pop_audio_chunk(timeout: float = 0.5) -> np.ndarray | None:
    global queued_samples_total
    try:
        chunk = audio_playback_queue.get(timeout=timeout)
        with queued_samples_lock:
            queued_samples_total = max(0, queued_samples_total - len(chunk))
        return chunk
    except queue.Empty:
        return None


def get_local_queue_seconds() -> float:
    with queued_samples_lock:
        return float(queued_samples_total) / float(SAMPLE_RATE)


def audio_playback_worker(target_buf=1):
    import sounddevice as sd
    while audio_playback_queue.qsize() < target_buf and not stop_playback.is_set():
        time.sleep(0.04)
    print("🔈 [Audio Player] Вывод звука 32 kHz в терминале активен (Zero-Drop).", flush=True)
    try:
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', latency='high') as stream:
            while not stop_playback.is_set():
                chunk = pop_audio_chunk(timeout=0.5)
                if chunk is not None:
                    stream.write(np.ascontiguousarray(chunk).tobytes())
    except Exception as e:
        print(f"⚠️ [Audio Player Note]: {e}", flush=True)


class AsyncMusicWorker(threading.Thread):
    def __init__(self, conn, runtime_cfg, semantic, gateway_port=6003, enable_local_audio=True):
        super().__init__(daemon=True)
        self.conn = conn
        self.cfg = runtime_cfg
        self.semantic = semantic
        self.enable_local_audio = enable_local_audio
        self.running = True
        self.lock = threading.Lock()
        self.latest_embeds = None
        self.block_sec = runtime_cfg['target_block']
        self.context_sec = runtime_cfg['target_context']
        self.temp = runtime_cfg['temp']
        self.top_k = runtime_cfg['top_k']
        self.cfg_coef = runtime_cfg['cfg_coef']
        self.is_drop = False
        self.gateway_conn = None
        self.gateway_port = gateway_port
        self.latest_pcm = np.zeros(0, dtype=np.float32)
        self.rtf = 1.0
        self.play_speed = 1.0
        self.gateway_queue_sec = 0.8
        self.latest_dyn_strength = runtime_cfg['strength_low']

    def update_conditioning(self, embeds, block_sec, context_sec, temp, top_k, cfg_coef, is_drop, dyn_strength):
        with self.lock:
            self.latest_embeds = embeds
            self.block_sec = block_sec
            self.context_sec = context_sec
            self.temp = temp
            self.top_k = top_k
            self.cfg_coef = cfg_coef
            self.latest_dyn_strength = dyn_strength
            if is_drop:
                self.is_drop = True

    def _apply_new_concepts(self, new_concepts: list[str]):
        cleaned = [str(x).strip() for x in new_concepts if str(x).strip()]
        self.conn.send({'cmd': 'init_lore', 'prompts': cleaned})
        enc_resp = self.conn.recv()
        with self.semantic.lock:
            if self.semantic.c_bases and not cleaned:
                self.semantic.last_active_embed = self.semantic.c_bases[0].clone()
            self.semantic.c_bases = [
                torch.tensor(b, dtype=torch.float32, device=DEVICE)
                for b in enc_resp['c_bases']
            ]
            self.semantic.words = list(cleaned)
            self.semantic.rebuild_norm_vocab()
        print(f"🎵 [MUSIC LORE ОБНОВЛЕН] ({len(self.semantic.words)}): {self.semantic.words}", flush=True)

    def flush(self):
        global queued_samples_total
        with self.lock:
            self.is_drop = True
        with queued_samples_lock:
            while not audio_playback_queue.empty():
                try:
                    audio_playback_queue.get_nowait()
                except Exception:
                    break
            queued_samples_total = 0

    def run(self):
        while self.running:
            with self.lock:
                emb = self.latest_embeds
                b_sec = self.block_sec
                c_sec = self.cfg['target_context']
                t_val = self.temp
                k_val = self.top_k
                cfg_c = self.cfg_coef
                safety = self.cfg['speed_safety_factor']
                drop_val = self.is_drop
                dyn_str = self.latest_dyn_strength
                self.is_drop = False

            if emb is None:
                time.sleep(0.01)
                continue

            q_sec = get_local_queue_seconds() if self.enable_local_audio else self.gateway_queue_sec

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
                    'queue_sec': q_sec,
                    'is_reset': drop_val
                })
                resp = self.conn.recv()
                if 'pcm' in resp:
                    pcm = resp['pcm']
                    self.rtf = float(resp.get('rtf', 1.0))
                    self.play_speed = float(resp.get('play_speed', 1.0))
                    self.cfg['current_speed'] = self.play_speed
                    if 'actual_block' in resp:
                        self.cfg['actual_block'] = float(resp['actual_block'])
                    with self.lock:
                        self.latest_pcm = pcm

                    if self.enable_local_audio:
                        push_audio_chunk_zero_drop(pcm)

                    if self.gateway_conn is None:
                        try:
                            self.gateway_conn = Client(('localhost', self.gateway_port), authkey=b'audio')
                            with self.semantic.lock:
                                if self.semantic.words:
                                    self.gateway_conn.send({
                                        'cmd': 'sync_from_runner',
                                        'concepts': list(self.semantic.words),
                                        'rtf': self.rtf
                                    })
                        except Exception:
                            self.gateway_conn = None

                    if self.gateway_conn is not None:
                        try:
                            self.gateway_conn.send({
                                'cmd': 'audio_chunk',
                                'pcm': pcm.tobytes(),
                                'rtf': self.rtf,
                                'play_speed': self.play_speed,
                                'dyn_strength': dyn_str
                            })
                            while self.gateway_conn.poll(0.0):
                                inc = self.gateway_conn.recv()
                                if isinstance(inc, dict):
                                    cmd = inc.get('cmd')
                                    if cmd == 'update_config':
                                        with self.lock:
                                            self.cfg.update(inc.get('config', {}))
                                            print(f"🎛️ [MUSIC CFG]: {self.cfg}", flush=True)
                                    elif cmd == 'update_concepts':
                                        gw_concepts = inc.get('concepts', [])
                                        if inc.get('is_initial_sync') and not gw_concepts and self.semantic.words:
                                            self.gateway_conn.send({
                                                'cmd': 'sync_from_runner',
                                                'concepts': list(self.semantic.words),
                                                'rtf': self.rtf
                                            })
                                        else:
                                            self._apply_new_concepts(gw_concepts)
                                    elif cmd == 'gateway_buffer_stat':
                                        self.gateway_queue_sec = float(inc.get('buffer_sec', 0.8))
                        except Exception:
                            self.gateway_conn = None

            except Exception:
                time.sleep(0.05)


def parse_initial_music_concepts(lore_file: str | None, concepts_cli: str | None) -> list[str]:
    custom_words = []
    if concepts_cli is not None:
        if concepts_cli.strip().lower() not in ["none", ""]:
            custom_words = [w.strip() for w in concepts_cli.split(",") if w.strip()]
        return custom_words

    if lore_file is not None and lore_file.strip() == "":
        return []

    candidate_files = [lore_file] if lore_file else ["lore/music_lore.txt", "music_lore.txt"]
    for fpath in candidate_files:
        if fpath and os.path.exists(fpath):
            if fpath.endswith('.json'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    custom_words = [str(x).strip() for x in (d if isinstance(d, list) else d.keys()) if str(x).strip()]
            else:
                with open(fpath, 'r', encoding='utf-8') as f:
                    custom_words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            break
    return custom_words


def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Dynamic MusicGen Heterarchy")
    parser.add_argument('--lore', '--concepts-file', dest='lore_file', type=str, default=None)
    parser.add_argument('--concepts', type=str, default=None)
    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--gamma-max', type=float, default=100.0)
    parser.add_argument('--seal-thresh', type=float, default=0.68)
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--minimal-ui', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="semantic", choices=["semantic", "turbo"])
    parser.add_argument('--gateway-port', type=int, default=6003)

    parser.add_argument('--target-block', type=float, default=2.4)
    parser.add_argument('--target-context', type=float, default=1.2)
    parser.add_argument('--target-buffer-count', type=int, default=1)
    parser.add_argument('--speed-safety-factor', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=250)
    parser.add_argument('--cfg-coef', type=float, default=1.0)

    parser.add_argument('--strength-low', type=float, default=0.25)
    parser.add_argument('--strength-high', type=float, default=0.80)
    parser.add_argument('--ema-smooth', type=float, default=0.20)
    args = parser.parse_args()

    runtime_config = {
        'target_block': args.target_block,
        'target_context': args.target_context,
        'target_buffer_count': args.target_buffer_count,
        'speed_safety_factor': args.speed_safety_factor,
        'temp': args.temp,
        'top_k': args.top_k,
        'cfg_coef': args.cfg_coef,
        'strength_low': args.strength_low,
        'strength_high': args.strength_high,
        'ema_smooth': args.ema_smooth,
        'current_speed': 1.0
    }

    enable_local_sound = not args.headless
    enable_pygame_hud = (not args.headless) and (not args.minimal_ui)

    print("=" * 70)
    print(f"🧠 NEUROCANVAS MUSIC RUNNER (HUD: {enable_pygame_hud} | LOCAL AUDIO: {enable_local_sound})")
    print("=" * 70)

    print("⏳ Подключение к musicgen_server (порт 6005)...", flush=True)
    conn = None
    while conn is None:
        try:
            conn = Client(('localhost', 6005), authkey=b'music')
            print("✅ MusicGen сервер готов к работе!", flush=True)
        except Exception:
            time.sleep(0.5)

    words = parse_initial_music_concepts(args.lore_file, args.concepts)
    print(f"📤 Запрос предрассчета {len(words)} тем лора у сервера...", flush=True)
    conn.send({'cmd': 'init_lore', 'prompts': words})
    resp = conn.recv()

    semantic = UniversalMusicSemanticSpace(
        words=words,
        c_bases=resp['c_bases'],
        proj_120_to_768=resp['proj_120_to_768'],
        empty_embed=resp.get('empty_embed'),
        global_svd_bases=resp.get('global_svd_bases')
    )
    router = SelfSufficientHeterarchyRouter(args.users)
    processor = MusicHeterarchyDeltaProcessor(
        semantic=semantic,
        router=router,
        strength_low=args.strength_low,
        strength_high=args.strength_high,
        seal_thresh=args.seal_thresh
    )

    hud = None
    if enable_pygame_hud:
        from music_hud import NeuroMusicHUD
        hud = NeuroMusicHUD(device=DEVICE)

    if enable_local_sound:
        threading.Thread(target=audio_playback_worker, args=(runtime_config['target_buffer_count'],), daemon=True).start()
    else:
        print("🔇 [Headless Mode] Локальный вывод звука в терминале отключен.", flush=True)

    engine = HeterarchicalBrainEngine(gamma_max=args.gamma_max)
    engine.start()

    worker = AsyncMusicWorker(
        conn, runtime_config, semantic=semantic,
        gateway_port=args.gateway_port, enable_local_audio=enable_local_sound
    )
    worker.start()

    smoothed_t5 = (semantic.c_bases[0] if semantic.c_bases else semantic.empty_embed).clone()
    last_term_print = 0.0

    try:
        while True:
            dt = 0.016 if hud else 0.10
            if hud:
                dt, commands = hud.poll_events(len(engine.get_frame().nodes))
                for cmd, payload in commands:
                    if cmd == "set_role":
                        router.set_role(payload[0], payload[1])
                    elif cmd == "reset_seals":
                        for eng in processor.engines.values():
                            eng.seals.charge.fill(0)
                            eng.seals.mask = [False] * MAX_SLOTS_CAPACITY
                    elif cmd == "flush_audio":
                        worker.flush()
                    elif cmd == "adj_param":
                        p_name, delta = payload
                        if p_name == "target_block":
                            runtime_config['target_block'] = float(np.clip(runtime_config['target_block'] + delta, 0.5, 4.0))
                        elif p_name == "temp":
                            runtime_config['temp'] = float(np.clip(runtime_config['temp'] + delta, 0.3, 1.8))

            frame = engine.get_frame()
            if frame.num_live == 0:
                time.sleep(0.05)
                continue

            target_out, b_sec, c_sec, t_val, k_val, cfg_c, telem = processor.step(
                frame, dt,
                mode=args.mode,
                target_block=runtime_config['target_block'],
                target_context=runtime_config['target_context'],
                base_temp=runtime_config['temp'],
                base_top_k=runtime_config['top_k'],
                cfg_coef=runtime_config['cfg_coef'],
                strength_low=runtime_config['strength_low'],
                strength_high=runtime_config['strength_high']
            )

            # Сферическое EMA-сглаживание: сохраняет точную норму каждого из 64 токенов T5!
            cur_ema_param = float(runtime_config.get('ema_smooth', 0.20))
            ema_alpha = float(np.clip(dt / max(0.04, cur_ema_param), 0.02, 0.60))
            target_norms = torch.norm(target_out, p=2, dim=-1, keepdim=True).clamp(min=1.0)
            smoothed_t5.mul_(1.0 - ema_alpha).add_(target_out, alpha=ema_alpha)
            smoothed_t5 = F.normalize(smoothed_t5, p=2, dim=-1) * target_norms

            worker.update_conditioning(
                smoothed_t5.cpu().numpy(),
                b_sec, c_sec, t_val, k_val, cfg_c,
                telem.get("is_drop", False),
                telem.get("dyn_strength", runtime_config['strength_low'])
            )

            if hud:
                with worker.lock:
                    pcm_chunk = worker.latest_pcm
                hud.push_pcm(pcm_chunk)
                hud.render(telem, rtf=worker.rtf, dt=dt, runtime_cfg=runtime_config)
            elif args.minimal_ui:
                now = time.time()
                if now - last_term_print >= 0.5:
                    last_term_print = now
                    dag_str = " ".join(telem.get("dag_log_parts", ["DRIFT"])[:4])
                    q_s = get_local_queue_seconds()
                    sys.stdout.write(
                        f"\r🎵 [MINIMAL-UI] RTF:{worker.rtf:4.2f}x | Speed:{worker.play_speed:4.2f}x | "
                        f"Buf:{q_s:3.1f}s | DynStr:{telem.get('dyn_strength', 0.25):.2f} | {dag_str}      "
                    )
                    sys.stdout.flush()
                time.sleep(0.10)
            else:
                time.sleep(0.10)

    except KeyboardInterrupt:
        pass
    finally:
        stop_playback.set()
        worker.running = False
        engine.stop()
        if hud:
            hud.close()


if __name__ == '__main__':
    main()

