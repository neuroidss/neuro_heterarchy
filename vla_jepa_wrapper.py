#!/usr/bin/env python3
"""
🌐 VLA-JEPA CLIENT WRAPPER (HOT-CONNECT & STANDALONE FALLBACK)
- Мгновенное подключение к jepa_server.py на порту 6001.
- Никаких блокировок LSL-потока, чистая изоляция от сбоев.
"""

import sys
import os
import re
import traceback
import torch
import numpy as np
from PIL import Image
from multiprocessing.connection import Client

current_file_path = os.path.dirname(os.path.abspath(__file__))
vla_jepa_path = None

for _ in range(4):
    potential_path = os.path.join(check_dir := current_file_path, "VLA-JEPA")
    if os.path.exists(potential_path):
        vla_jepa_path = potential_path
        break
    sibling_path = os.path.join(os.path.dirname(check_dir), "VLA-JEPA")
    if os.path.exists(sibling_path):
        vla_jepa_path = sibling_path
        break
    current_file_path = os.path.dirname(check_dir)

if vla_jepa_path and vla_jepa_path not in sys.path:
    sys.path.insert(0, vla_jepa_path)

HAS_STAR_VLA = False
VLA_JEPA_Class = None

try:
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA as _vla
    VLA_JEPA_Class = _vla
    HAS_STAR_VLA = True
except Exception:
    HAS_STAR_VLA = False

def safe_normalize(t, dim=-1, eps=1e-5):
    norm = torch.norm(t, p=2, dim=dim, keepdim=True)
    return t / torch.clamp(norm, min=eps)

def download_vla_jepa_assets(local_vla_path):
    from huggingface_hub import hf_hub_download
    repo_id = "ginwind/VLA-JEPA"
    run_dir = os.path.join(local_vla_path, "SimplerEnv")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    pt_path = os.path.join(ckpt_dir, "VLA-JEPA-SimplerEnv.pt")
    if not os.path.exists(pt_path):
        hf_hub_download(repo_id=repo_id, filename="SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt", local_dir=local_vla_path)
    return pt_path


class VLA_JEPA_Wrapper:
    def __init__(self, render=None, device="cuda", port=6001):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.jepa_dim = 2048
        self.remote_conn = None
        self.model = None

        # 1. Проверяем наличие запущенного jepa_server.py
        try:
            self.remote_conn = Client(('localhost', port), authkey=b'jepa')
            self.remote_conn.send({'cmd': 'ping'})
            info = self.remote_conn.recv()
            self.jepa_dim = info.get('jepa_dim', 2048)
            print(f"⚡ [VLA-JEPA] Подключен к jepa_server на порту {port}! (Мгновенный старт 0.05с, jepa_dim={self.jepa_dim})")
            return
        except Exception:
            self.remote_conn = None

        print("ℹ️ [VLA-JEPA] jepa_server не обнаружен. Работа в автономном легковесном режиме.")

    def encode_world_state(self, image_input):
        """Принимает PIL.Image или np.ndarray, возвращает латентный тензор мира [1, 77, jepa_dim]."""
        if self.remote_conn is not None:
            try:
                if isinstance(image_input, Image.Image):
                    img_np = np.array(image_input, dtype=np.uint8)
                else:
                    img_np = image_input

                self.remote_conn.send({'cmd': 'encode_world_state', 'image_np': img_np})
                arr = self.remote_conn.recv()
                return torch.tensor(arr, dtype=self.dtype, device=self.device)
            except Exception:
                self.remote_conn = None

        # Автономный режим без задержек (если сервер не поднят)
        return torch.zeros((1, 77, self.jepa_dim), dtype=self.dtype, device=self.device)

    def predict_future_state(self, current_state, action_token, eeg_intent, focus_level=1.0):
        if self.remote_conn is not None:
            try:
                self.remote_conn.send({
                    'cmd': 'predict_future_state',
                    'current_state': current_state.detach().cpu().numpy(),
                    'action_token': action_token.detach().cpu().numpy(),
                    'eeg_intent': eeg_intent.detach().cpu().numpy(),
                    'focus_level': float(focus_level)
                })
                arr = self.remote_conn.recv()
                return torch.tensor(arr, dtype=self.dtype, device=self.device)
            except Exception:
                self.remote_conn = None

        return current_state
