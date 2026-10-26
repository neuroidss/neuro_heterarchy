#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: REAL-TIME MUSICGEN RUNNER (POWERED BY SHARED AMODAL GPU BUS)
===================================================================================
- Функция project_amodal_bus_to_musicgen_gpu 100% идентична вашему оригинальному
  рабочему файлу (но находится здесь, чтобы neuro_amodal_engine.py оставался
  полностью амодальным).
- Поддерживает переключение стохастики (--stochastic / --deterministic).
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
except Exception:
    pass

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuro_heterarchy_core import DEVICE
from neuro_genesis_engine import SelfSufficientHeterarchyRouter
from music_genesis_engine import UniversalMusicSemanticSpace
from neuro_amodal_engine import (
    SharedAmodalCognitiveEngine,
    HardwareVRAMBus,
)

SAMPLE_RATE = 32000
audio_playback_queue = queue.Queue(maxsize=100)
stop_playback = threading.Event()

queued_samples_lock = threading.Lock()
queued_samples_total = 0


@torch.no_grad()
def project_amodal_bus_to_musicgen_gpu(bus: HardwareVRAMBus, semantic: UniversalMusicSemanticSpace,
                                       target_block: float, target_context: float,
                                       base_temp: float, base_top_k: int, cfg_coef: float,
                                       strength_low: float, strength_high: float,
                                       need_ui_telemetry: bool = False):
    """
    100% точная математика из вашего рабочего файла:
    использует slot_norm_coord_16, amodal_manifold_77x768[1:65] (где лежат все 5 регионов
    Fpz/AFz/F3/F4/FCz) и SVD-проекцию v_aff!
    """
    meta = bus.meta_32
    n_sec = max(2, int(meta[1].item()))
    fcz_pers_t = meta[6]
    max_surp_t = meta[7]

    active_drive_t = torch.clamp(fcz_pers_t * 0.70 + torch.clamp(max_surp_t, max=1.0) * 0.30, 0.0, 1.0)
    dyn_str_t = torch.clamp(strength_low + (strength_high - strength_low) * active_drive_t, strength_low, 0.98)
    dyn_str = float(dyn_str_t.item())

    order = bus.slot_order_16[:n_sec].long()
    norm_coords = bus.slot_norm_coord_16[:n_sec]
    c_flow = bus.causal_flow_16x16[:n_sec, :n_sec]
    amodal_64 = bus.amodal_manifold_77x768[1:65]

    with semantic.lock:
        c_bases_copy = list(semantic.c_bases)
        words_copy = list(semantic.words)
        n_bases = min(len(c_bases_copy), len(words_copy))

        if n_bases == 0:
            base_64 = semantic.last_active_embed.clone()
            clean_norms = torch.norm(base_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)
            v_aff = semantic.global_svd_bases[:16]
            slot_idx_gpu = torch.zeros(n_sec, dtype=torch.long, device=DEVICE)
        else:
            slot_idx_gpu = torch.clamp((norm_coords * n_bases).long(), 0, n_bases - 1)
            root_s = order[0] % n_sec
            root_c_idx = int(slot_idx_gpu[root_s].item()) % n_bases
            base_64 = c_bases_copy[root_c_idx].clone()
            clean_norms = torch.norm(base_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)

            if n_bases > 1 and n_sec > 1:
                child_s = order[1] % n_sec
                child_c_idx = int(slot_idx_gpu[child_s].item()) % n_bases
                if child_c_idx != root_c_idx:
                    t_child = c_bases_copy[child_c_idx]
                    dot = torch.sum(t_child * base_64, dim=-1, keepdim=True) / (torch.sum(base_64 ** 2, dim=-1, keepdim=True) + 1e-7)
                    ortho = F.normalize(t_child - dot * base_64, p=2, dim=-1) * clean_norms
                    is_sup = (c_flow[root_s, child_s] >= 0.03).float()
                    base_64[4:20].add_(ortho[4:20] * is_sup * 0.32 + t_child[4:20] * (1.0 - is_sup) * 0.22)

            v_aff = semantic.get_concept_affordances_svd(root_c_idx, max_bases=16)

        coords = torch.matmul(amodal_64, v_aff.T)
        musical_mod = F.normalize(torch.matmul(coords, v_aff), p=2, dim=-1) * clean_norms

        final_64 = base_64 * (1.0 - 0.35 * dyn_str_t) + musical_mod * (0.35 * dyn_str_t)
        final_64 = F.normalize(final_64, p=2, dim=-1) * clean_norms

    temp = float(min(1.8, max(0.3, base_temp * (0.88 + 0.25 * dyn_str))))
    top_k = int(min(450, max(30, base_top_k + int((dyn_str - 0.4) * 100))))

    telemetry = {"dyn_strength": round(dyn_str, 3), "is_drop": False}
    if need_ui_telemetry:
        indices = [int(i) % max(1, n_bases) for i in slot_idx_gpu.tolist()]
        names = [words_copy[i].upper() for i in indices] if n_bases > 0 else ["СВОБОДНЫЙ_ДРЕЙФ"] * n_sec
        root_name = names[int(order[0].item()) % n_sec]
        telemetry.update({
            "primary_role": "AFz", "primary_active": True,
            "total_clock_sectors": n_sec, "current_clock_sec": int(meta[2].item()) % n_sec,
            "slot_names": names, "slot_indices": indices,
            "sealed_mask": [bool(x > 0.5) for x in bus.slot_seals_16[:n_sec].tolist()],
            "slot_stabilities": bus.slot_stabs_16[:n_sec].cpu().numpy(),
            "causal_flow_np": c_flow.cpu().numpy(),
            "dag_log_parts": [root_name],
            "dl_phase": float(meta[3].item()), "is_delta_tick": bool(meta[4].item() > 0.5),
            "active_vocab_size": n_bases,
            "fpz_active": True, "fpz_tension": float(meta[16].item()), "fpz_concept": root_name,
            "f3_active": True, "f3_concept": root_name,
            "f3_x_local": float(meta[9].item()), "f3_y_global": float(meta[10].item()), "f3_kuramoto": float(meta[11].item()),
            "f4_active": True, "f4_concept": root_name, "f4_entropy": float(active_drive_t.item()),
            "fcz_active": True, "fcz_concept": root_name,
            "fcz_lx": float(meta[12].item()), "fcz_ly": float(meta[13].item()),
            "fcz_rx": float(meta[14].item()), "fcz_ry": float(meta[15].item()),
        })
    return final_64, target_block, target_context, temp, top_k, cfg_coef, telemetry


def map_working_memory_to_music_field_gpu(bus: HardwareVRAMBus, semantic: UniversalMusicSemanticSpace,
                                          target_block: float, target_context: float,
                                          base_temp: float, base_top_k: int, cfg_coef: float,
                                          strength_low: float, strength_high: float,
                                          need_ui_telemetry: bool = False):
    final_64, b_sec, c_sec, t_val, k_val, cfg_c, telem = project_amodal_bus_to_musicgen_gpu(
        bus, semantic, target_block, target_context, base_temp, base_top_k, cfg_coef,
        strength_low, strength_high, need_ui_telemetry
    )
    return final_64, bus.wm_field_5x5x32x120, b_sec, c_sec, t_val, k_val, cfg_c, telem


def push_audio_chunk_zero_drop(pcm: np.ndarray, max_chunks: int = 2):
    global queued_samples_total
    if pcm is None or len(pcm) == 0:
        return
    with queued_samples_lock:
        while audio_playback_queue.qsize() > max(1, max_chunks):
            try:
                old = audio_playback_queue.get_nowait()
                queued_samples_total = max(0, queued_samples_total - len(old))
            except Exception:
                break
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


def get_local_queue_stats() -> tuple[float, int]:
    with queued_samples_lock:
        return float(queued_samples_total) / float(SAMPLE_RATE), int(audio_playback_queue.qsize())


def audio_playback_worker(target_buf=1):
    import sounddevice as sd
    while audio_playback_queue.qsize() < target_buf and not stop_playback.is_set():
        time.sleep(0.02)
    try:
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', latency='high') as stream:
            while not stop_playback.is_set():
                chunk = pop_audio_chunk(timeout=0.5)
                if chunk is not None:
                    stream.write(np.ascontiguousarray(chunk).tobytes())
    except Exception as e:
        print(f"⚠️ [Audio Player Note]: {e}", flush=True)


class AsyncMusicWorker(threading.Thread):
    def __init__(self, conn, runtime_cfg, semantic, gateway_port=6003, enable_local_audio=True, deterministic=None):
        super().__init__(daemon=True)
        self.conn = conn
        self.cfg = runtime_cfg
        self.semantic = semantic
        self.enable_local_audio = enable_local_audio
        self.deterministic = deterministic
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
        self.actual_block = 0.16
        self.gateway_queue_sec = 0.0
        self.gateway_queue_chunks = 0
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
                b_sec = float(self.cfg.get('target_block', 0.0))
                c_sec = float(self.cfg.get('target_context', 3.6))
                t_buf_cnt = int(self.cfg.get('target_buffer_count', 2))
                t_val = self.temp
                k_val = self.top_k
                cfg_c = self.cfg_coef
                safety = self.cfg['speed_safety_factor']
                drop_val = self.is_drop
                dyn_str = self.latest_dyn_strength
                det_override = self.deterministic
                self.is_drop = False

            if emb is None:
                time.sleep(0.01)
                continue

            if self.enable_local_audio:
                q_sec, q_chunks = get_local_queue_stats()
            else:
                q_sec, q_chunks = self.gateway_queue_sec, self.gateway_queue_chunks

            try:
                req = {
                    'cmd': 'generate_step',
                    't5_embeds': emb,
                    'block_sec': b_sec,
                    'context_sec': c_sec,
                    'target_buffer_count': t_buf_cnt,
                    'temp': t_val,
                    'top_k': k_val,
                    'cfg_coef': cfg_c,
                    'safety_factor': safety,
                    'queue_sec': q_sec,
                    'queue_chunks': q_chunks,
                    'is_reset': drop_val
                }
                if det_override is not None:
                    req['deterministic'] = bool(det_override)

                self.conn.send(req)
                resp = self.conn.recv()
                if 'pcm' in resp:
                    pcm = resp['pcm']
                    self.rtf = float(resp.get('rtf', 1.0))
                    self.play_speed = float(resp.get('play_speed', 1.0))
                    self.actual_block = float(resp.get('actual_block', 0.16))
                    self.cfg['current_speed'] = self.play_speed
                    self.cfg['actual_block'] = self.actual_block
                    with self.lock:
                        self.latest_pcm = pcm

                    if self.enable_local_audio:
                        push_audio_chunk_zero_drop(pcm, max_chunks=t_buf_cnt)

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
                                'actual_block': self.actual_block,
                                'dyn_strength': dyn_str
                            })
                            while self.gateway_conn.poll(0.0):
                                inc = self.gateway_conn.recv()
                                if isinstance(inc, dict):
                                    cmd = inc.get('cmd')
                                    if cmd == 'update_config':
                                        new_cfg = inc.get('config', {})
                                        new_m = new_cfg.get('model')
                                        old_m = self.cfg.get('model', 'facebook/musicgen-small')
                                        with self.lock:
                                            self.cfg.update(new_cfg)
                                            if 'stochastic' in new_cfg:
                                                self.deterministic = not bool(new_cfg['stochastic'])
                                            elif 'deterministic' in new_cfg:
                                                self.deterministic = bool(new_cfg['deterministic'])
                                        if new_m and new_m != old_m:
                                            self.conn.send({'cmd': 'switch_model', 'model': new_m})
                                            self.conn.recv()
                                            self._apply_new_concepts(self.semantic.words)
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
                                        self.gateway_queue_sec = float(inc.get('buffer_sec', 0.0))
                                        self.gateway_queue_chunks = int(inc.get('buffer_chunks', 0))
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
    parser = argparse.ArgumentParser(description="NeuroCanvas Dynamic MusicGen Heterarchy (Shared Amodal Bus)")
    parser.add_argument('--lore', '--concepts-file', dest='lore_file', type=str, default=None)
    parser.add_argument('--concepts', type=str, default=None)
    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--gamma-max', type=float, default=100.0)
    parser.add_argument('--seal-thresh', type=float, default=1.0)
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--minimal-ui', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="semantic", choices=["semantic", "turbo"])
    parser.add_argument('--gateway-port', type=int, default=6003)

    parser.add_argument('--stochastic', dest='stochastic', action='store_true', default=True,
                        help="Включить стохастику в MusicGen (по умолчанию)")
    parser.add_argument('--deterministic', '--no-stochastic', dest='stochastic', action='store_false',
                        help="Отключить стохастику в MusicGen")

    parser.add_argument('--target-block', type=float, default=0.0)
    parser.add_argument('--target-context', type=float, default=6.0)
    parser.add_argument('--target-buffer-count', type=int, default=2)
    parser.add_argument('--speed-safety-factor', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=250)
    parser.add_argument('--cfg-coef', type=float, default=1.0)

    parser.add_argument('--strength-low', type=float, default=0.25)
    parser.add_argument('--strength-high', type=float, default=0.80)
    parser.add_argument('--ema-smooth', type=float, default=0.20)
    args = parser.parse_args()

    init_det = None if args.stochastic is None else (not bool(args.stochastic))

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
        'current_speed': 1.0,
        'actual_block': 0.16
    }

    enable_local_sound = not args.headless
    enable_pygame_hud = (not args.headless) and (not args.minimal_ui)

    conn = None
    while conn is None:
        try:
            conn = Client(('localhost', 6005), authkey=b'music')
        except Exception:
            time.sleep(0.5)

    words = parse_initial_music_concepts(args.lore_file, args.concepts)
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

    amodal_engine = SharedAmodalCognitiveEngine(
        router=router,
        gamma_max=args.gamma_max,
        strength_low=args.strength_low,
        strength_high=args.strength_high
    )

    hud = None
    if enable_pygame_hud:
        from music_hud import NeuroMusicHUD
        hud = NeuroMusicHUD(device=DEVICE)

    if enable_local_sound:
        threading.Thread(target=audio_playback_worker, args=(runtime_config['target_buffer_count'],), daemon=True).start()

    worker = AsyncMusicWorker(
        conn, runtime_config, semantic=semantic,
        gateway_port=args.gateway_port, enable_local_audio=enable_local_sound,
        deterministic=init_det
    )
    worker.start()

    smoothed_t5 = (semantic.c_bases[0] if semantic.c_bases else semantic.empty_embed).clone()
    last_term_print = 0.0

    try:
        while True:
            dt = 0.016 if hud else 0.05
            if hud:
                dt, commands = hud.poll_events(16)
                for cmd, payload in commands:
                    if cmd == "set_role":
                        router.set_role(payload[0], payload[1])
                    elif cmd == "reset_seals":
                        amodal_engine.reset_seals()
                    elif cmd == "flush_audio":
                        worker.flush()
                    elif cmd == "adj_param":
                        p_name, delta = payload
                        if p_name == "target_block":
                            runtime_config['target_block'] = float(np.clip(runtime_config['target_block'] + delta, 0.0, 4.0))
                        elif p_name == "temp":
                            runtime_config['temp'] = float(np.clip(runtime_config['temp'] + delta, 0.3, 1.8))

            bus, has_live = amodal_engine.step(dt)
            if not has_live:
                time.sleep(0.05)
                continue

            target_out, b_sec, c_sec, t_val, k_val, cfg_c, telem = project_amodal_bus_to_musicgen_gpu(
                bus, semantic,
                target_block=runtime_config['target_block'],
                target_context=runtime_config['target_context'],
                base_temp=runtime_config['temp'],
                base_top_k=runtime_config['top_k'],
                cfg_coef=runtime_config['cfg_coef'],
                strength_low=runtime_config['strength_low'],
                strength_high=runtime_config['strength_high'],
                need_ui_telemetry=(hud is not None or args.minimal_ui)
            )

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
                if now - last_term_print >= 0.4:
                    last_term_print = now
                    dag_str = " ".join(telem.get("dag_log_parts", ["DRIFT"])[:4])
                    q_s, q_c = get_local_queue_stats()
                    sys.stdout.write(
                        f"\r🎵 [MINIMAL-UI] RTF:{worker.rtf:4.2f}x | Speed:{worker.play_speed:4.2f}x | "
                        f"Blk:{worker.actual_block:4.2f}s | Buf:{q_s:3.1f}s({q_c}) | {dag_str}      "
                    )
                    sys.stdout.flush()
                time.sleep(0.05)
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        stop_playback.set()
        worker.running = False
        amodal_engine.stop()
        if hud:
            hud.close()


if __name__ == '__main__':
    main()

