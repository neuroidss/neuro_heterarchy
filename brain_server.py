#!/usr/bin/env python3
import os
import sys
import time
import argparse
from multiprocessing.connection import Listener
from PIL import Image
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_logic import NeuroRender


def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Fast Brain Server")
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--stochastic', type=float, default=0.0,
                        help="Уровень стохастики от 0.0 (100% чистая рабочая память ЭЭГ) до 1.0 (50% ЭЭГ + 50% рандом)")
    args = parser.parse_args()

    stochastic_level = float(np.clip(args.stochastic, 0.0, 1.0))

    print(f"🧠 STARTING FAST NEURAL BRAIN SERVER (CUDA BACKEND | Stochastic={stochastic_level:.2f}) 🧠")
    current_mode = args.mode
    render = NeuroRender(mode=current_mode, use_taesd=True, stochastic=stochastic_level)

    listener = Listener(('localhost', 6000), authkey=b'brain')
    print(f"🧠 Server is ACTIVE in [{current_mode.upper()}] mode on localhost:6000...")

    while True:
        try:
            conn = listener.accept()
            while True:
                try:
                    msg = conn.recv()
                    cmd = msg.get('cmd')

                    if cmd == 'init_mode':
                        req_mode = msg.get('mode', current_mode)
                        req_taesd = msg.get('use_taesd', render.use_taesd)
                        if 'stochastic' in msg and msg['stochastic'] is not None:
                            render.stochastic_level = float(np.clip(float(msg['stochastic']), 0.0, 1.0))

                        if req_mode != current_mode:
                            cur_stoch = render.stochastic_level
                            del render
                            torch.cuda.empty_cache()
                            render = NeuroRender(mode=req_mode, use_taesd=req_taesd, stochastic=cur_stoch)
                            current_mode = req_mode
                        else:
                            render.set_taesd(req_taesd)
                        conn.send({
                            'status': 'ok',
                            'mode': current_mode,
                            'is_sdxl': render.is_sdxl,
                            'stochastic': render.stochastic_level
                        })
                        continue

                    elif cmd == 'set_stochastic':
                        val = float(msg.get('stochastic', 0.0))
                        render.stochastic_level = float(np.clip(val, 0.0, 1.0))
                        print(f"🎲 [BRAIN SERVER] Уровень стохастики: {render.stochastic_level:.2f}")
                        conn.send({'status': 'ok', 'stochastic': render.stochastic_level})
                        continue

                    elif cmd == 'encode_base_prompts':
                        prompts = msg.get('prompts', [])
                        conn.send(render.encode_prompts_universal(prompts))
                        continue

                    elif cmd == 'generate':
                        image_np = msg.pop('image_np', None)
                        embeds_np = msg.pop('prompt_embeds', None)
                        pooled_np = msg.pop('pooled_prompt_embeds', None)
                        wm_np = msg.pop('wm_5x5x32x120', None)

                        kwargs = msg.copy()
                        del kwargs['cmd']

                        if embeds_np is not None:
                            t_emb = torch.tensor(embeds_np, dtype=render.dtype, device=render.device)
                            if t_emb.ndim == 2:
                                t_emb = t_emb.unsqueeze(0)
                            kwargs['prompt_embeds'] = t_emb

                        if wm_np is not None:
                            kwargs['wm_5x5x32x120'] = torch.tensor(wm_np, dtype=torch.float32, device=render.device)

                        if render.is_sdxl:
                            if pooled_np is not None:
                                t_pool = torch.tensor(pooled_np, dtype=render.dtype, device=render.device)
                                if t_pool.ndim == 1:
                                    t_pool = t_pool.unsqueeze(0)
                                kwargs['pooled_prompt_embeds'] = t_pool
                            else:
                                kwargs['pooled_prompt_embeds'] = torch.zeros((1, 1280), dtype=render.dtype, device=render.device)

                        if image_np is not None:
                            kwargs['image'] = Image.fromarray(image_np).convert("RGB")

                        out_img = render.generate(**kwargs)
                        conn.send(np.array(out_img, dtype=np.uint8))

                except EOFError:
                    break
                except Exception as e:
                    print(f"[SERVER ERROR]: {e}")
                    try:
                        conn.send({'error': str(e)})
                    except Exception:
                        break
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
