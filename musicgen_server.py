#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: MODULAR MUSICGEN IPC SERVER (musicgen_server.py)
===================================================================================
"""

from __future__ import annotations
import os
import time
import argparse
from multiprocessing.connection import Listener
import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

try:
    os.setpriority(os.PRIO_PROCESS, 0, -20)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

from music_render_logic import NeuroMusicRenderer, FRAME_RATE


def main():
    parser = argparse.ArgumentParser(description="Modular NeuroCanvas MusicGen Server")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument('--model', type=str, default="facebook/musicgen-small")
    parser.add_argument('--benchmark', dest='benchmark', action='store_true', default=True)
    parser.add_argument('--no-benchmark', dest='benchmark', action='store_false')
    parser.add_argument('--quiet-step', action='store_true', default=False)
    parser.add_argument('--stochastic', type=float, default=0.5,
                        help="Уровень стохастики от 0.0 (чистая рабочая память) до 1.0 (чистый AudioCraft)")
    args = parser.parse_args()

    stochastic_level = float(np.clip(args.stochastic, 0.0, 1.0))

    renderer = NeuroMusicRenderer(
        model_name=args.model,
        stochastic=stochastic_level,
        run_benchmark=args.benchmark
    )

    listener = Listener(('localhost', args.port), authkey=b'music')
    print(f"🚀 [MUSICGEN-SERVER] Слушаю порт {args.port} (Stochastic={stochastic_level:.2f})...")

    chunk_idx = 0
    while True:
        try:
            conn = listener.accept()
            renderer.reset_stream()

            while True:
                if not conn.poll(0.05):
                    continue
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break

                cmd = msg.get('cmd')

                if cmd == 'init_lore':
                    resp = renderer.encode_lore_prompts(msg.get('prompts', []))
                    conn.send(resp)

                elif cmd == 'generate_step':
                    resp, sleep_sec = renderer.generate_step(msg)
                    conn.send(resp)

                    chunk_idx += 1
                    if not args.quiet_step:
                        prof = renderer.last_profile
                        actual_ctx_s = prof.get('ctx_tokens', 0) / float(FRAME_RATE)
                        print(
                            f"⏱️ [Чанк #{chunk_idx:03d}] RTF:{resp['rtf']:4.2f}x | Speed:{resp['play_speed']:4.2f}x | "
                            f"Block:{resp['actual_block']:.2f}s->Next:{renderer.current_block:.2f}s "
                            f"(Ctx:{actual_ctx_s:.1f}s) | Steps:{prof['steps_count']} ({prof['per_step_ms']:.1f}ms/st)",
                            flush=True
                        )

                    if sleep_sec > 0.002:
                        time.sleep(sleep_sec)

                elif cmd == 'switch_model':
                    new_m = renderer.switch_model(msg.get('model', 'facebook/musicgen-small'))
                    conn.send({'status': 'ok', 'model': new_m})

                elif cmd == 'set_stochastic':
                    stoch = float(np.clip(float(msg.get('stochastic', 0.5)), 0.0, 1.0))
                    renderer.stochastic = stoch
                    conn.send({'status': 'ok', 'stochastic': renderer.stochastic})

                elif cmd == 'flush':
                    renderer.reset_stream()
                    conn.send({'status': 'ok'})

        except Exception:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
