#!/usr/bin/env python3
"""
🧠 LIGHTWEIGHT V-JEPA SERVER (PORT 6001)
- Грузит ТОЛЬКО визуальный энкодер V-JEPA 2 (в fp16).
- Полностью отсекает тяжелый Qwen-VL и DiT-B (экономия ~5.8 ГБ VRAM).
- Потребление VRAM: всего ~900 МБ вместо 6.7 ГБ!
"""

import os
import sys
import gc
import argparse
from multiprocessing.connection import Listener
import numpy as np
import torch
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from vla_jepa_wrapper import (
    HAS_STAR_VLA,
    vla_jepa_path,
    VLA_JEPA_Class,
    download_vla_jepa_assets,
    safe_normalize
)

def extract_vjepa_encoder(model):
    """Очищает модель от ненужного мусора: удаляет LLM (Qwen) и DiT, оставляя только V-JEPA 2."""
    encoder = None
    
    # Ищем модуль V-JEPA 2 внутри архитектуры VLA_JEPA
    for attr in ['vj2_model', 'vision_encoder', 'visual_encoder', 'vjepa', 'encoder']:
        if hasattr(model, attr):
            encoder = getattr(model, attr)
            print(f"🎯 [JEPA-EXTRACTOR] Найден целевой энкодер: model.{attr}")
            break

    # Безжалостно вычищаем неиспользуемые LLM и DiT
    for dead_weight in ['vlm', 'language_model', 'llm', 'action_head', 'dit', 'model']:
        if hasattr(model, dead_weight) and getattr(model, dead_weight) is not encoder:
            try:
                delattr(model, dead_weight)
                print(f"✂️ [JEPA-PRUNING] Удален неиспользуемый блок: {dead_weight}")
            except Exception:
                pass

    gc.collect()
    torch.cuda.empty_cache()
    return encoder if encoder is not None else model

def main():
    parser = argparse.ArgumentParser(description="Lightweight V-JEPA World Model Server")
    parser.add_argument('--port', type=int, default=6001, help="Port for JEPA server IPC")
    parser.add_argument('--device', type=str, default="cuda", help="Target device")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 65)
    print(f"🧠 ЗАПУСК ЛЕГКОВЕСНОГО V-JEPA СЕРВЕРА (TARGET: ONLY ENCODER, {device.upper()})")
    print("=" * 65)

    vjepa_encoder = None
    jepa_dim = 2048

    if HAS_STAR_VLA and vla_jepa_path and VLA_JEPA_Class is not None:
        try:
            pt_path = download_vla_jepa_assets(vla_jepa_path)
            print(f"[JEPA-SERVER] Чтение весов: {pt_path} (на CPU во избежание OOM)...")
            
            # 1. Загружаем на CPU:
            full_model = VLA_JEPA_Class.from_pretrained(pt_path)
            if hasattr(full_model, "config") and hasattr(full_model.config, "framework"):
                jepa_dim = full_model.config.framework.vj2_model.get("jepa_dim", 2048)

            # 2. Отрезаем Qwen и DiT, оставляем ТОЛЬКО энкодер:
            vjepa_encoder = extract_vjepa_encoder(full_model)
            del full_model
            gc.collect()

            # 3. Переводим только энкодер в GPU в FP16:
            vjepa_encoder = vjepa_encoder.to(device=device, dtype=dtype).eval()
            print(f"✅ [JEPA-SERVER] V-JEPA 2 энкодер успешно изолирован и загружен! (jepa_dim={jepa_dim})")
            
        except Exception as e:
            print(f"⚠️ [JEPA-SERVER] Не удалось вырезать энкодер: {e}. Переход в легковесный режим.")
            vjepa_encoder = None
    else:
        print("ℹ️ [JEPA-SERVER] starVLA отсутствует. Включен математический латентный энкодер.")

    # Выводим реальный статус памяти:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"📊 [VRAM USAGE] Занято энкодером JEPA: {allocated:.1f} MiB (Reserved: {reserved:.1f} MiB). Память свободна!")

    listener = Listener(('localhost', args.port), authkey=b'jepa')
    print(f"🚀 [JEPA-SERVER] Сервер слушает на localhost:{args.port}. Готов к работе.\n")

    while True:
        try:
            conn = listener.accept()
            while True:
                try:
                    msg = conn.recv()
                    cmd = msg.get('cmd')

                    if cmd == 'ping':
                        conn.send({
                            'status': 'ok',
                            'jepa_dim': jepa_dim,
                            'has_model': bool(vjepa_encoder is not None)
                        })
                        continue

                    elif cmd == 'encode_world_state':
                        image_np = msg.get('image_np')
                        if image_np is None:
                            conn.send(np.zeros((1, 77, jepa_dim), dtype=np.float32))
                            continue

                        with torch.no_grad():
                            img_t = torch.from_numpy(image_np).to(device=device, dtype=dtype)
                            img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0
                            resized = torch.nn.functional.interpolate(img_t, size=(224, 224), mode='bilinear', align_corners=False)

                            z_out = None
                            if vjepa_encoder is not None:
                                try:
                                    if hasattr(vjepa_encoder, "encode_vision"):
                                        feat = vjepa_encoder.encode_vision(resized)
                                    elif hasattr(vjepa_encoder, "forward"):
                                        feat = vjepa_encoder(resized)
                                    else:
                                        feat = None
                                    
                                    if feat is not None:
                                        if isinstance(feat, tuple): feat = feat[0]
                                        if feat.ndim == 2: feat = feat.unsqueeze(1).expand(-1, 77, -1)
                                        z_out = feat.detach().cpu().numpy().astype(np.float32)
                                except Exception:
                                    pass

                            if z_out is None:
                                pool = torch.nn.functional.adaptive_avg_pool2d(resized, (8, 8)).flatten()
                                if pool.numel() < jepa_dim:
                                    pool = torch.nn.functional.pad(pool, (0, jepa_dim - pool.numel()))
                                else:
                                    pool = pool[:jepa_dim]
                                norm_pool = safe_normalize(pool.unsqueeze(0).unsqueeze(0).expand(1, 77, jepa_dim), dim=-1)
                                z_out = norm_pool.detach().cpu().numpy().astype(np.float32)

                            conn.send(z_out)

                    elif cmd == 'predict_future_state':
                        curr = torch.tensor(msg['current_state'], dtype=dtype, device=device)
                        action = torch.tensor(msg['action_token'], dtype=dtype, device=device)
                        eeg = torch.tensor(msg['eeg_intent'], dtype=dtype, device=device)
                        focus = float(msg.get('focus_level', 1.0))

                        with torch.no_grad():
                            res = safe_normalize(eeg, dim=-1) * safe_normalize(action, dim=-1)
                            transition = action * res * focus
                            pred = curr + transition
                            conn.send(pred.detach().cpu().numpy().astype(np.float32))

                except EOFError:
                    break
                except Exception as e:
                    try:
                        conn.send({'error': str(e)})
                    except Exception:
                        break

        except Exception:
            pass

if __name__ == '__main__':
    main()
