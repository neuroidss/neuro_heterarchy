#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: PURE UNCONSTRAINED MULTIDIMENSIONAL HETERARCHY BCI (TBT 2.0)
===================================================================================
- ПОЛНОСТЬЮ АВТОНОМНОЕ ЯДРО (БЕЗ ВСТРОЕННЫХ ВЕБ-СЕРВЕРОВ):
    * Клиент к brain_server (порт 6000, authkey=b'brain')
    * Клиент к видео-шлюзу (порт 6002, authkey=b'canvas')
- 100% НАУЧНЫЙ КОНВЕЙЕР:
    1. Динамический хронометр PAC (K_theta = f_theta / f_delta) + GPU 1D pooling.
    2. Causal DAG на 89.5 Гц рипплах Дики (ciPLV lead matrix A ⊃ B vs A || B).
    3. Рекурсивный Грама-Шмидт с Collinearity Guard (защита от нулевого шума).
    4. Хроно-туннель Дельты и World Seals (запечатывание стабильных миров).
    5. BA 10 (Fpz) Вето: антиподальный поиск argmin и вычитание подпространства.
    6. FCz 4D Кинематика камеры (манифолд-варп на лету).
    7. Синаптический Sample-and-Hold при лагах и джиттере сети.
- ПОДДЕРЖКА ВСЕХ CLI АРГУМЕНТОВ: --users, --lore, --steps, --burst, --seal и др.
===================================================================================
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import argparse
import time
import math
import json
import colorsys
import numpy as np
import cv2
import pygame
import torch
import threading
from pathlib import Path
from multiprocessing.connection import Client
from transformers import CLIPTokenizer, CLIPTextModel

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuro_heterarchy_core import (
    HeterarchicalBrainEngine, DEVICE, 
    COORDS_X, COORDS_Y, DX_120, DY_120
)
from neuro_models import DynamicPilot
from neuro_workers import apply_manifold_camera_warp, apply_color_surgery

WIDTH, HEIGHT = 1800, 960
MAX_SLOTS_CAPACITY = 16

ipc_lock = threading.Lock()

# Физический детерминированный базис: 120 диполей -> 768 колонок (float32 на GPU)
I_IDX, J_IDX = np.triu_indices(16, k=1)
DIPOLE_X = (COORDS_X[I_IDX] + COORDS_X[J_IDX]) / 2.0
DIPOLE_Y = (COORDS_Y[I_IDX] + COORDS_Y[J_IDX]) / 2.0
dipoles_x_t = torch.tensor(DIPOLE_X, dtype=torch.float32, device=DEVICE).view(1, 120)
dipoles_y_t = torch.tensor(DIPOLE_Y, dtype=torch.float32, device=DEVICE).view(1, 120)

grid_dim_x, grid_dim_y = 32, 24
col_x = torch.linspace(-12.0, 12.0, grid_dim_x, device=DEVICE, dtype=torch.float32)
col_y = torch.linspace(-12.0, 12.0, grid_dim_y, device=DEVICE, dtype=torch.float32)
grid_y, grid_x = torch.meshgrid(col_y, col_x, indexing='ij')
cols_x_t = grid_x.reshape(768, 1)
cols_y_t = grid_y.reshape(768, 1)

d_sq = (cols_x_t - dipoles_x_t)**2 + (cols_y_t - dipoles_y_t)**2
PHYSICAL_PROJECTION_768x120 = torch.exp(-d_sq / 32.0)
PHYSICAL_PROJECTION_768x120 = torch.nn.functional.normalize(PHYSICAL_PROJECTION_768x120, p=2, dim=1).to(dtype=torch.float32)

PALETTE_HOTKEYS = [
    pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r,
    pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f,
    pygame.K_z, pygame.K_x, pygame.K_c, pygame.K_v
]
SLOT_HOTKEYS = [
    pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
    pygame.K_9, pygame.K_0, pygame.K_MINUS, pygame.K_EQUALS
]

def generate_concept_colors(num_concepts: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(max(16, num_concepts)):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

class UniversalSemanticSpace:
    def __init__(self, lore_file: str | None = None, concepts_cli: str | None = None, 
                 target_k: int | None = None, model_id="openai/clip-vit-large-patch14"):
        print("⏳ [СЕМАНТИКА] Инициализация CLIP Text Model на GPU...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE).eval()
        self.gpu_embed_cache = {}

        custom_words = []
        if lore_file and os.path.exists(lore_file):
            print(f"📜 [ЗАГРУЗКА ЛОРА]: Чтение из файла '{lore_file}'...")
            if lore_file.endswith('.json'):
                with open(lore_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list): custom_words = [str(x).strip() for x in data if str(x).strip()]
                    elif isinstance(data, dict): custom_words = [str(k).strip() for k in data.keys() if str(k).strip()]
            else:
                with open(lore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        clean = line.strip()
                        if clean and not clean.startswith('#'): custom_words.append(clean)
            print(f"✅ [ЛОР ЗАГРУЖЕН]: {len(custom_words)} концептов.")
        elif concepts_cli and concepts_cli.strip().lower() not in ["none", ""]:
            custom_words = [w.strip() for w in concepts_cli.split(",") if w.strip()]
            print(f"🎨 [ПАЛИТРА CLI]: {len(custom_words)} концептов: {custom_words}")

        self.is_custom_lore = len(custom_words) > 0

        if self.is_custom_lore:
            self.words = custom_words
            lore_embeds = []
            for w in self.words:
                emb = self.get_77_embedding_gpu(w).mean(dim=0)
                lore_embeds.append(emb)
            raw_matrix = torch.stack(lore_embeds, dim=0)
            mu = torch.mean(raw_matrix, dim=0, keepdim=True)
            std = torch.std(raw_matrix - mu, dim=0, keepdim=True) + 1e-6
            self.norm_vocab = torch.nn.functional.normalize((raw_matrix - mu) / std, p=2, dim=-1)
            print(f"✅ [БАЗА ЛОРА ГОТОВА НА GPU].")
        else:
            raw_weights = self.text_model.get_input_embeddings().weight.detach()
            vocab = self.tokenizer.get_vocab()
            valid_ids, valid_words = [], []
            for word, idx in vocab.items():
                if word.endswith('</w>'):
                    clean = word[:-4]
                    if len(clean) >= 2:
                        valid_ids.append(idx)
                        valid_words.append(clean)

            self.valid_ids = torch.tensor(valid_ids, dtype=torch.long, device=DEVICE)
            self.words = valid_words
            raw_vocab = raw_weights[self.valid_ids].float()
            mu = torch.mean(raw_vocab, dim=0, keepdim=True)
            centered_vocab = raw_vocab - mu
            std = torch.std(centered_vocab, dim=0, keepdim=True) + 1e-6
            norm_vocab_full = torch.nn.functional.normalize(centered_vocab / std, p=2, dim=-1)

            n_total = len(self.words)
            k_subset = n_total if target_k is None else max(2, min(target_k, n_total))

            if k_subset < n_total:
                print(f"📐 [FPS СНИЖЕНИЕ]: Выбор {k_subset} ортогональных полюсов...")
                selected = [0]
                min_dists = 1.0 - torch.mm(norm_vocab_full, norm_vocab_full[0:1].T).squeeze(1)
                for _ in range(1, k_subset):
                    next_idx = int(torch.argmax(min_dists).item())
                    selected.append(next_idx)
                    new_dists = 1.0 - torch.mm(norm_vocab_full, norm_vocab_full[next_idx:next_idx+1].T).squeeze(1)
                    min_dists = torch.minimum(min_dists, new_dists)

                sel_t = torch.tensor(selected, dtype=torch.long, device=DEVICE)
                self.norm_vocab = norm_vocab_full[sel_t]
                self.words = [self.words[i] for i in selected]
            else:
                self.norm_vocab = norm_vocab_full
            print(f"✅ [50k СЛОВАРЬ ГОТОВ НА GPU]: {len(self.words)} понятий.")

    def get_77_embedding_gpu(self, word: str) -> torch.Tensor:
        w_clean = word.strip().lower()
        if not w_clean:
            w_clean = self.words[0].lower()
        if w_clean in self.gpu_embed_cache:
            return self.gpu_embed_cache[w_clean]
        
        tokens = self.tokenizer([w_clean], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            emb = self.text_model(tokens).last_hidden_state[0]
        self.gpu_embed_cache[w_clean] = emb
        return emb

    def decode_concepts_contrastive_gpu(self, z_slots_Sx768: torch.Tensor, is_veto: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        z_mean = torch.mean(z_slots_Sx768, dim=0, keepdim=True)
        z_contrast = z_slots_Sx768 - z_mean * 0.75
        q = torch.nn.functional.normalize(z_contrast, p=2, dim=-1)
        sims = torch.mm(q, self.norm_vocab.T)

        if is_veto:
            vals, indices = torch.min(sims, dim=-1)
        else:
            vals, indices = torch.max(sims, dim=-1)
            
        return indices, vals

class UserProfile:
    def __init__(self, user_id: str, regions_map: dict, **pilot_kwargs):
        self.user_id = user_id
        self.regions_map = regions_map
        self.pilot = DynamicPilot(**pilot_kwargs)
        self.order_drive = 0.5
        self.chaos_drive = 0.5
        self.branch_ratio = 0.0
        self.torus_u = 0.0
        self.torus_v = 0.0
        self.temporal_bias = 0.0
        self.manual_slot_concepts = [-1] * MAX_SLOTS_CAPACITY

    def get_content_nodes(self, frame_nodes: list) -> list:
        CONTENT_REGIONS = {"AFz", "F3", "F4"}
        nodes = []
        for reg_name, dev_list in self.regions_map.items():
            if reg_name in CONTENT_REGIONS:
                for idx in dev_list:
                    if 0 <= idx < len(frame_nodes):
                        nodes.append(frame_nodes[idx])
        return nodes

    def get_device_nodes(self, region_name: str, frame_nodes: list) -> list:
        nodes = []
        if region_name in self.regions_map:
            for idx in self.regions_map[region_name]:
                if 0 <= idx < len(frame_nodes):
                    nodes.append(frame_nodes[idx])
        return nodes

def parse_arbitrary_user_setup(users_str: str | None, pilot_kwargs: dict) -> list[UserProfile]:
    if not users_str or users_str.strip().lower() in ["none", "auto"]:
        return [
            UserProfile("User1", {"F3": [0], "F4": [1], "AFz": [2], "Fpz": [3], "FCz": [0]}, **pilot_kwargs),
            UserProfile("User2", {"F3": [1], "F4": [0], "AFz": [3], "Fpz": [2], "FCz": [1]}, **pilot_kwargs)
        ]

    subjects = []
    for b in users_str.split(";"):
        if not b.strip(): continue
        parts = b.strip().split(":")
        sub_id = parts[0].strip()
        reg_map = {}
        if len(parts) > 1:
            for rd in parts[1].split(";"):
                for entry in rd.split(","):
                    if "=" in entry:
                        r_name, d_str = entry.split("=")
                        dev_ids = [int(x.strip()) for x in d_str.split("/") if x.strip().isdigit()]
                        if not dev_ids and d_str.strip().isdigit(): dev_ids = [int(d_str.strip())]
                        reg_map.setdefault(r_name.strip(), []).extend(dev_ids)
                    elif entry.strip().isdigit():
                        reg_map.setdefault("AFz", []).append(int(entry.strip()))
        if not reg_map: reg_map = {"AFz": [0]}
        subjects.append(UserProfile(sub_id, reg_map, **pilot_kwargs))
    return subjects

class GPUDynamicGraph:
    def __init__(self, max_nodes=MAX_SLOTS_CAPACITY, cx=1340, cy=320, width=820, height=520, speed=4.5, repulsion=5500.0, attraction=0.45):
        self.max_nodes, self.cx, self.cy = max_nodes, cx, cy
        self.half_w, self.half_h = width * 0.46, height * 0.44
        self.speed, self.repulsion_k, self.attraction_k = speed, repulsion, attraction
        angles = torch.linspace(0, 2.0 * math.pi, max_nodes + 1, device=DEVICE)[:max_nodes]
        self.pos = torch.stack([self.cx + torch.cos(angles) * (self.half_w * 0.55), self.cy + torch.sin(angles) * (self.half_h * 0.55)], dim=-1)
        self.vel = torch.zeros((max_nodes, 2), device=DEVICE)
        self.radii_gpu = torch.full((max_nodes,), 24.0, device=DEVICE, dtype=torch.float32)
        self.center_t = torch.tensor([self.cx, self.cy], device=DEVICE)

    def update_physics_gpu(self, active_count, causal_flow_gpu, sealed_mask_t, dt=0.016):
        flow_sub = causal_flow_gpu[:active_count, :active_count]
        net_lead = torch.sum(flow_sub, dim=1) - torch.sum(flow_sub, dim=0)
        self.radii_gpu[:active_count] = torch.clamp(22.0 + net_lead * 40.0, 16.0, 42.0)

        sub_steps = 2
        sub_dt = (dt * self.speed) / sub_steps
        for _ in range(sub_steps):
            p_sub = self.pos[:active_count]
            diff = p_sub.unsqueeze(1) - p_sub.unsqueeze(0)
            dist_sq = torch.sum(diff**2, dim=-1).clamp(min=25.0)
            dist = torch.sqrt(dist_sq)

            repulsion = (diff / (dist_sq.unsqueeze(-1) * dist.unsqueeze(-1))) * self.repulsion_k
            rep_force = torch.sum(torch.nan_to_num(repulsion), dim=1)
            flow_w = torch.clamp(torch.abs(flow_sub), 0.0, 1.0).unsqueeze(-1)
            attr_force = torch.sum(diff * flow_w * self.attraction_k, dim=1)
            grav_force = (self.center_t - p_sub) * 0.04

            is_sealed = sealed_mask_t[:active_count].float().unsqueeze(-1)
            sealed_cohesion = torch.zeros_like(p_sub)
            if torch.sum(sealed_mask_t[:active_count]) >= 2:
                sealed_center = torch.mean(p_sub[sealed_mask_t[:active_count]], dim=0)
                sealed_cohesion = (sealed_center - p_sub) * is_sealed * 0.25

            total_force = rep_force + attr_force + grav_force + sealed_cohesion
            self.vel[:active_count] = (self.vel[:active_count] + total_force * sub_dt) * 0.78
            self.pos[:active_count] += self.vel[:active_count] * sub_dt
            self.pos[:active_count, 0] = torch.clamp(self.pos[:active_count, 0], self.cx - self.half_w, self.cx + self.half_w)
            self.pos[:active_count, 1] = torch.clamp(self.pos[:active_count, 1], self.cy - self.half_h, self.cy + self.half_h)

        return self.pos.cpu().numpy(), self.radii_gpu[:active_count].cpu().numpy()

class DeltaSpiralHistory:
    def __init__(self, max_rings=5):
        self.max_rings = max_rings
        self.rings = []

    def push_completed_delta(self, total_sectors, concepts_in_sectors, is_sealed):
        self.rings.insert(0, (total_sectors, list(concepts_in_sectors[:total_sectors]), is_sealed))
        if len(self.rings) > self.max_rings: self.rings.pop()

    def render(self, screen, cx, cy, R_max, R_core, dl_phase, total_sectors, current_concepts, current_sealed, concept_colors, is_active_sec):
        dr = (R_max - R_core) / float(self.max_rings + 1)
        for ring_idx in range(len(self.rings) - 1, -1, -1):
            r_out = R_max - (ring_idx + 1) * dr
            r_in = r_out - dr + 2.0
            n_sec, ring_concepts, ring_sealed = self.rings[ring_idx]
            alpha = int(240 * (1.0 - (ring_idx + 1) / float(self.max_rings + 2)))
            sec_step = (2.0 * math.pi) / float(n_sec)

            for s_i in range(n_sec):
                a_start = -math.pi * 0.5 + s_i * sec_step
                a_end = a_start + sec_step
                c_id = ring_concepts[s_i]
                c_rgb = (255, 140, 50) if ring_sealed else concept_colors[c_id % len(concept_colors)]
                arc_pts = []
                for st in range(9):
                    ang = a_start + (a_end - a_start) * (st / 8)
                    arc_pts.append((cx + math.cos(ang) * r_out, cy + math.sin(ang) * r_out))
                for st in range(8, -1, -1):
                    ang = a_start + (a_end - a_start) * (st / 8)
                    arc_pts.append((cx + math.cos(ang) * r_in, cy + math.sin(ang) * r_in))

                surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(surf, (c_rgb[0], c_rgb[1], c_rgb[2], alpha), arc_pts)
                screen.blit(surf, (0, 0))
                pygame.draw.polygon(screen, (15, 20, 30), arc_pts, 1)

        r_out_live = R_max
        r_in_live = R_max - dr + 2.0
        sec_w_live = (2.0 * math.pi) / float(total_sectors)

        for s_i in range(total_sectors):
            a_start = -math.pi * 0.5 + s_i * sec_w_live
            a_end = a_start + sec_w_live
            is_active = (s_i == is_active_sec)
            c_id = current_concepts[s_i % len(current_concepts)]
            c_rgb = (255, 140, 50) if current_sealed else concept_colors[c_id % len(concept_colors)]
            arc_pts = []
            for st in range(11):
                ang = a_start + (a_end - a_start) * (st / 10)
                arc_pts.append((cx + math.cos(ang) * r_out_live, cy + math.sin(ang) * r_out_live))
            for st in range(10, -1, -1):
                ang = a_start + (a_end - a_start) * (st / 10)
                arc_pts.append((cx + math.cos(ang) * r_in_live, cy + math.sin(ang) * r_in_live))

            surf_live = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(surf_live, (c_rgb[0], c_rgb[1], c_rgb[2], 255 if is_active else 180), arc_pts)
            screen.blit(surf_live, (0, 0))
            pygame.draw.polygon(screen, (255, 255, 255) if is_active else (40, 55, 70), arc_pts, 2 if is_active else 1)

        dl_norm = (dl_phase + math.pi) % (2.0 * math.pi)
        hand_ang = -math.pi * 0.5 + dl_norm
        hx = cx + math.cos(hand_ang) * (R_max + 6)
        hy = cy + math.sin(hand_ang) * (R_max + 6)
        pygame.draw.line(screen, (0, 255, 255), (cx, cy), (hx, hy), 3)
        pygame.draw.circle(screen, (255, 255, 255), (int(hx), int(hy)), 7)
        pygame.draw.circle(screen, (10, 14, 20), (cx, cy), int(R_core) - 2)
        pygame.draw.circle(screen, (0, 255, 200), (cx, cy), int(R_core) - 2, 2)

class DeltaCycleSealEngine:
    def __init__(self, max_slots=MAX_SLOTS_CAPACITY, seal_cycles=4, decay_cycles=2, seal_thresh=0.68):
        self.max_slots = max_slots
        self.seal_cycles, self.decay_cycles, self.seal_thresh = seal_cycles, decay_cycles, seal_thresh
        self.slot_delta_charge = np.zeros(max_slots, dtype=np.float32)
        self.sealed_mask = [False] * max_slots
        self.sealed_concept_indices = [-1] * max_slots
        self.sealed_labels = [""] * max_slots
        self.sealed_dag = np.zeros((max_slots, max_slots), dtype=np.float32)

    def step_delta_cycle(self, active_count, slot_stability, slot_dominant_idx, slot_names, causal_flow_np):
        for s in range(active_count):
            ripple_energy = np.sum(np.abs(causal_flow_np[s, :active_count])) + np.sum(np.abs(causal_flow_np[:active_count, s]))
            if slot_stability[s] >= self.seal_thresh:
                charge_delta = 1.0 + float(np.clip(ripple_energy * 2.0, 0.0, 1.5))
                self.slot_delta_charge[s] = min(float(self.seal_cycles), self.slot_delta_charge[s] + charge_delta)
            else:
                loss = float(self.seal_cycles) / float(max(1, self.decay_cycles))
                self.slot_delta_charge[s] = max(0.0, self.slot_delta_charge[s] - loss)

            if self.sealed_mask[s] and self.slot_delta_charge[s] < (self.seal_cycles * 0.20):
                self.sealed_mask[s] = False
                self.sealed_concept_indices[s] = -1
                self.sealed_labels[s] = ""

        ready_slots = [s for s in range(active_count) if not self.sealed_mask[s] and self.slot_delta_charge[s] >= self.seal_cycles]
        if len(ready_slots) >= 2:
            for s in ready_slots:
                self.sealed_mask[s] = True
                self.sealed_concept_indices[s] = slot_dominant_idx[s]
                self.sealed_labels[s] = slot_names[s]
            self.sealed_dag[:active_count, :active_count] = causal_flow_np[:active_count, :active_count].copy()
            print(f"🔒 [ПЕЧАТЬ ЗАКРЕПЛЕНА]: Слоты {[s+1 for s in ready_slots]} стали объектом мира!")

class AsyncDiffusionWorker(threading.Thread):
    def __init__(self, conn, initial_h=384, initial_w=512, use_color=True, is_sdxl=False, inference_steps=2):
        super().__init__(daemon=True)
        self.conn = conn
        self.gateway_conn = None
        self.img_h, self.img_w = initial_h, initial_w
        self.use_color, self.is_sdxl = use_color, is_sdxl
        self.inference_steps = inference_steps
        self.display_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.latest_embeds = None
        self.latest_pooled = None
        self.strength = 0.60
        self.running = True
        self.lock = threading.Lock()
        self.fps = 0.0

    def update_conditioning(self, prompt_embeds_np, pooled_np, strength):
        with self.lock:
            self.latest_embeds = prompt_embeds_np
            self.latest_pooled = pooled_np
            self.strength = float(np.clip(strength, 0.35, 0.98))

    def flush_canvas(self):
        with self.lock:
            self.display_rgb.fill(0)

    def apply_camera_warp(self, pilot: DynamicPilot, dt: float):
        with self.lock:
            if np.max(self.display_rgb) > 0:
                self.display_rgb = apply_manifold_camera_warp(self.display_rgb, pilot, dt)

    def get_canvas(self) -> np.ndarray:
        with self.lock:
            return self.display_rgb

    def run(self):
        times = []
        internal_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        while self.running:
            with self.lock:
                embeds = self.latest_embeds
                pooled = self.latest_pooled
                s_val = self.strength
                steps = self.inference_steps
                if np.max(self.display_rgb) > 0:
                    internal_rgb = self.display_rgb.copy()

            if embeds is None:
                time.sleep(0.005)
                continue

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
                if self.is_sdxl and pooled is not None:
                    req['pooled_prompt_embeds'] = pooled

                with ipc_lock:
                    self.conn.send(req)
                    resp = self.conn.recv()

                if isinstance(resp, np.ndarray):
                    if resp.shape[:2] != (self.img_h, self.img_w):
                        resp = cv2.resize(resp, (self.img_w, self.img_h))

                    if np.max(internal_rgb) == 0:
                        internal_rgb = resp
                    else:
                        treated = apply_color_surgery(resp, internal_rgb.astype(np.float32)) if self.use_color else resp
                        internal_rgb = cv2.addWeighted(internal_rgb, 0.25, treated, 0.75, 0)

                    with self.lock:
                        self.display_rgb = internal_rgb

                    # -------------------------------------------------------------
                    # ОТПРАВКА В GATEWAY ЧЕРЕЗ DIRECT MEMORY IPC (ПОРТ 6002)
                    # -------------------------------------------------------------
                    if self.gateway_conn is None:
                        try:
                            self.gateway_conn = Client(('localhost', 6002), authkey=b'canvas')
                        except Exception:
                            self.gateway_conn = None

                    if self.gateway_conn is not None:
                        try:
                            # Исправление цвета для браузера (RGB -> BGR для корректного JPEG)
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

                elif isinstance(resp, dict) and 'error' in resp:
                    time.sleep(0.1)

            except Exception:
                time.sleep(0.05)

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas: Pure Unconstrained N-Dimensional Heterarchy BCI")
    parser.add_argument('--lore', '--concepts-file', dest='lore_file', type=str, default=None,
                        help="Путь к файлу лора (.txt по строкам или .json) с полной картиной мира")
    parser.add_argument('--concepts', type=str, default=None,
                        help="Список концептов через запятую")
    parser.add_argument('--vocab-ratio', type=float, default=1.0)
    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--sim', action='store_true', default=False)
    parser.add_argument('--users', type=str, default=None,
                        help="Монтаж сенсоров (напр. 'User1:AFz=0' или 'User1:F3=0,F4=1,AFz=2,Fpz=3,FCz=0')")
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo", "sdxl"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--headless', action='store_true', default=False,
                        help="Запуск без графического окна (для удаленного GPU сервера)")
    parser.add_argument('--minimal-ui', action='store_true', default=False,
                        help="Чистый холст без дебаг-панелей")
    parser.add_argument('--burst-duration', type=float, default=0.0)
    parser.add_argument('--burst-strength', type=float, default=0.75)
    parser.add_argument('--strength-high', type=float, default=0.75)
    parser.add_argument('--strength-low', type=float, default=0.55)
    parser.add_argument('--gamma-100', action='store_true', default=False)
    parser.add_argument('--no-taesd', action='store_true', default=False)
    parser.add_argument('--no-color', action='store_true', default=False)
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    parser.add_argument('--seal-cycles', type=int, default=4)
    parser.add_argument('--decay-cycles', type=int, default=2)
    parser.add_argument('--refresh-cycles', type=int, default=2)
    parser.add_argument('--seal-thresh', type=float, default=0.68)
    parser.add_argument('--graph-speed', type=float, default=4.5)
    parser.add_argument('--graph-repulsion', type=float, default=5500.0)
    parser.add_argument('--graph-attraction', type=float, default=0.45)
    parser.add_argument('--fpz-veto-gain', type=float, default=2.5)
    parser.add_argument('--fpz-veto-thresh', type=float, default=0.35)
    args = parser.parse_args()

    pilot_kwargs = {'sensitivity': 0.05, 'max_speed': 9.0, 'fwd_scale': 0.2, 'strafe_scale': 0.2, 'turn_scale': 0.5, 'intent_gain': 1.5}
    users = parse_arbitrary_user_setup(args.users, pilot_kwargs)
    for u in users:
        print(f"👤 [{u.user_id}]: Конфигурация: {u.regions_map}")

    conn = None
    is_sdxl_mode = False
    print("⏳ Подключение к brain_server (порт 6000)...")
    while conn is None:
        try:
            conn = Client(('localhost', 6000), authkey=b'brain')
            with ipc_lock:
                conn.send({'cmd': 'init_mode', 'mode': args.mode, 'use_taesd': not args.no_taesd})
                resp = conn.recv()
            is_sdxl_mode = resp.get('is_sdxl', False)
            print(f"✅ Диффузия готова! (SDXL: {is_sdxl_mode})")
        except Exception:
            time.sleep(0.5)

    target_k = args.vocab_size if args.vocab_size is not None else (int(49400 * args.vocab_ratio) if args.vocab_ratio < 1.0 else None)
    semantic_space = UniversalSemanticSpace(
        lore_file=args.lore_file,
        concepts_cli=args.concepts,
        target_k=target_k
    )
    active_vocab_size = len(semantic_space.words)
    has_lore = semantic_space.is_custom_lore
    concept_colors = generate_concept_colors(active_vocab_size)

    # Инициализация синаптической инерции (Sample-and-Hold)
    initial_default_word = semantic_space.words[0]
    initial_prompt_tensor = semantic_space.get_77_embedding_gpu(initial_default_word).to(dtype=torch.float32)
    latent_dim = initial_prompt_tensor.shape[-1]
    
    target_77 = initial_prompt_tensor.clone()
    last_valid_target_77 = initial_prompt_tensor.clone()
    accumulated_tree_subspace = initial_prompt_tensor.clone()
    smoothed_prompt_77 = initial_prompt_tensor.clone()
    pooled_zero = torch.zeros((1280,), dtype=torch.float32, device=DEVICE) if is_sdxl_mode else None
    smoothed_pooled = pooled_zero.clone() if is_sdxl_mode else None

    causal_flow_gpu = torch.zeros((MAX_SLOTS_CAPACITY, MAX_SLOTS_CAPACITY), dtype=torch.float32, device=DEVICE)
    causal_flow_np = np.zeros((MAX_SLOTS_CAPACITY, MAX_SLOTS_CAPACITY), dtype=np.float32)

    root_slot = 0
    hierarchical_order_cpu = np.arange(MAX_SLOTS_CAPACITY, dtype=np.int64)

    CYCLE_BUF_LEN = max(2, args.refresh_cycles)
    delta_cycle_buffer = torch.zeros((CYCLE_BUF_LEN, MAX_SLOTS_CAPACITY, 768), dtype=torch.float32, device=DEVICE)
    delta_buf_ptr = 0

    slot_dominant_idx = [0] * MAX_SLOTS_CAPACITY
    slot_dominant_name = [initial_default_word.upper()] * MAX_SLOTS_CAPACITY
    slot_confidences = np.zeros(MAX_SLOTS_CAPACITY, dtype=np.float32)
    slot_stability = np.zeros(MAX_SLOTS_CAPACITY, dtype=np.float32)

    seals = DeltaCycleSealEngine(
        max_slots=MAX_SLOTS_CAPACITY, 
        seal_cycles=args.seal_cycles, 
        decay_cycles=args.decay_cycles, 
        seal_thresh=args.seal_thresh
    )

    spiral_history = DeltaSpiralHistory(max_rings=5)
    gpu_graph = GPUDynamicGraph(
        max_nodes=MAX_SLOTS_CAPACITY, 
        cx=1340, cy=320, width=820, height=520, 
        speed=args.graph_speed,
        repulsion=args.graph_repulsion,
        attraction=args.graph_attraction
    )

    engine = HeterarchicalBrainEngine(gamma_max=100.0 if args.gamma_100 else 65.0)
    engine.start()

    canvas_w, canvas_h = (512, 384) if is_sdxl_mode else ((448, 336) if args.speed == "fast" else (512, 384))
    diff_worker = AsyncDiffusionWorker(conn, initial_h=canvas_h, initial_w=canvas_w, use_color=not args.no_color, is_sdxl=is_sdxl_mode, inference_steps=args.steps)
    diff_worker.start()

    screen = None
    clock = pygame.time.Clock()
    font_b = font_s = font_large = None
    show_minimal_ui = args.minimal_ui

    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("NeuroCanvas: Pure Unconstrained N-Dimensional Heterarchy BCI")
        font_b = pygame.font.SysFont("consolas", 14, bold=True)
        font_s = pygame.font.SysFont("consolas", 11)
        font_large = pygame.font.SysFont("consolas", 18, bold=True)

    current_ui_user_idx = 0
    selected_palette_idx = 0
    measured_ratio = 4.0
    delta_cycle_armed = False
    total_delta_ticks = 0
    last_dominant_root_name = initial_default_word.upper()
    switch_burst_timer = 0.0

    try:
        while True:
            dt = clock.tick(60) / 1000.0
            mouse_pos = (0, 0)
            mouse_clicked = False

            if not args.headless:
                mouse_pos = pygame.mouse.get_pos()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: raise KeyboardInterrupt
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1: mouse_clicked = True
                        elif event.button == 4 and has_lore and active_vocab_size <= 16: selected_palette_idx = (selected_palette_idx + 1) % active_vocab_size
                        elif event.button == 5 and has_lore and active_vocab_size <= 16: selected_palette_idx = (selected_palette_idx - 1) % active_vocab_size
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_TAB: current_ui_user_idx = (current_ui_user_idx + 1) % len(users)
                        if event.key == pygame.K_h:
                            show_minimal_ui = not show_minimal_ui
                        if event.key == pygame.K_r:
                            seals.slot_delta_charge.fill(0.0)
                            seals.sealed_mask = [False] * MAX_SLOTS_CAPACITY
                            spiral_history.rings.clear()
                            print("🔥 [СБРОС ПЕЧАТЕЙ] Все печати мира растворены.")
                        if event.key == pygame.K_c:
                            diff_worker.flush_canvas()
                            print("🧹 [ХОЛСТ ОЧИЩЕН] Сброс кадра.")

            # -----------------------------------------------------------------
            # 1. ПОЛУЧЕНИЕ КАДРА И ТЕТА/ДЕЛЬТА РАСЧЕТ
            # -----------------------------------------------------------------
            frame = engine.get_frame()
            if frame.num_live == 0:
                if screen and not args.headless:
                    screen.fill((8, 12, 18))
                    screen.blit(font_large.render("ОЖИДАНИЕ LSL ПОТОКОВ...", True, (255, 140, 50)), (440, 175))
                    pygame.display.flip()
                time.sleep(0.01)
                continue

            current_ui_user_idx = current_ui_user_idx % len(users)
            current_user = users[current_ui_user_idx]

            dl_phase = frame.delta_phase
            th_phase = frame.theta_phase
            th_hz = max(3.5, frame.theta_freq)
            dl_hz = max(0.5, frame.delta_freq)
            
            measured_ratio = measured_ratio * 0.95 + (th_hz / dl_hz) * 0.05
            k_thetas = int(np.clip(round(measured_ratio), 2, 8))
            total_clock_sectors = k_thetas * 2
            sec_w = (2.0 * math.pi) / float(total_clock_sectors)

            dl_norm_angle = (dl_phase + math.pi) % (2.0 * math.pi)
            current_clock_sec = int(np.clip(int(dl_norm_angle / sec_w), 0, total_clock_sectors - 1))

            if not args.headless:
                keys = pygame.key.get_pressed()
                held_slot = -1
                for s_i, s_key in enumerate(SLOT_HOTKEYS[:total_clock_sectors]):
                    if keys[s_key]: held_slot = s_i; break

                if held_slot != -1:
                    if keys[pygame.K_x] or keys[pygame.K_BACKSPACE]:
                        current_user.manual_slot_concepts[held_slot] = -1
                        seals.slot_delta_charge[held_slot] = 0.0
                        seals.sealed_mask[held_slot] = False
                    if has_lore and active_vocab_size <= 16:
                        for c_i, c_key in enumerate(PALETTE_HOTKEYS[:active_vocab_size]):
                            if keys[c_key]:
                                current_user.manual_slot_concepts[held_slot] = c_i
                                selected_palette_idx = c_i

            is_delta_tick = False
            if dl_phase < -1.5: delta_cycle_armed = True
            elif dl_phase > 0.5 and delta_cycle_armed:
                is_delta_tick = True
                delta_cycle_armed = False
                total_delta_ticks += 1
                spiral_history.push_completed_delta(total_clock_sectors, slot_dominant_idx, any(seals.sealed_mask[:total_clock_sectors]))

            # -----------------------------------------------------------------
            # 2. РАЗДЕЛЕНИЕ УЗЛОВ: AFZ (ПОЗИТИВ) VS FPZ (ВЕТО)
            # -----------------------------------------------------------------
            content_nodes = current_user.get_content_nodes(frame.nodes)
            fpz_nodes     = current_user.get_device_nodes("Fpz", frame.nodes)
            fcz_nodes     = current_user.get_device_nodes("FCz", frame.nodes)
            f3_nodes      = current_user.get_device_nodes("F3", frame.nodes)
            f4_nodes      = current_user.get_device_nodes("F4", frame.nodes)

            # 4D Кинематика строго на FCz
            if fcz_nodes:
                ax = fcz_nodes[0].gamepad_axes
                current_user.pilot.update(dt, float(ax.lx), float(-ax.ly), float(ax.rx), float(ax.ry))
                if abs(ax.lx) > 0.04 or abs(ax.ly) > 0.04 or abs(ax.rx) > 0.03:
                    diff_worker.apply_camera_warp(current_user.pilot, dt)

            current_user.order_drive = float(np.mean([n.beta_power for n in f3_nodes])) if f3_nodes else 0.5
            current_user.chaos_drive = float(np.mean([n.beta_power for n in f4_nodes])) if f4_nodes else float(1.0 - current_user.order_drive)
            current_user.branch_ratio = float(np.mean([n.gating_ratio for n in fpz_nodes])) if fpz_nodes else 0.0

            lead_node = content_nodes[0] if content_nodes else (fpz_nodes[0] if fpz_nodes else frame.nodes[0])
            current_user.temporal_bias = float(lead_node.gamepad_axes.ry)

            is_pure_veto = (len(content_nodes) == 0 and len(fpz_nodes) > 0)
            active_calc_nodes = fpz_nodes if is_pure_veto else content_nodes

            # -----------------------------------------------------------------
            # 3. 100% GPU ВЫЧИСЛЕНИЯ С СИНАПТИЧЕСКИМ SAMPLE-AND-HOLD
            # -----------------------------------------------------------------
            valid_eeg_input = False
            if active_calc_nodes:
                node_wm_tensors = [torch.from_numpy(n.iplv_32).to(DEVICE, dtype=torch.float32) for n in active_calc_nodes]
                raw_iplv = torch.stack(node_wm_tensors, dim=0).mean(dim=0)
                
                if torch.norm(raw_iplv) > 1e-4:
                    valid_eeg_input = True
                    dynamic_wm = torch.nn.functional.adaptive_avg_pool1d(raw_iplv.T.unsqueeze(0), total_clock_sectors).squeeze(0).T
                    z_slots_768 = torch.matmul(dynamic_wm, PHYSICAL_PROJECTION_768x120.T)

                    # Фазово-контрастный выбор концептов
                    top_indices_gpu, _ = semantic_space.decode_concepts_contrastive_gpu(z_slots_768, is_veto=is_pure_veto)
                    top_indices_cpu = top_indices_gpu.cpu().numpy()
                    
                    for s in range(total_clock_sectors):
                        if current_user.manual_slot_concepts[s] != -1:
                            m_idx = current_user.manual_slot_concepts[s]
                            slot_dominant_idx[s] = m_idx
                            slot_dominant_name[s] = semantic_space.words[m_idx].upper()
                            slot_confidences[s] = 1.0
                        elif seals.sealed_mask[s]:
                            slot_dominant_name[s] = seals.sealed_labels[s]
                            slot_confidences[s] = 1.0
                        else:
                            idx = int(top_indices_cpu[s])
                            slot_dominant_idx[s] = idx
                            slot_dominant_name[s] = semantic_space.words[idx].upper()

                    # Рипплы 89.5 Гц в 1 GPU-операцию (0 циклов .item())
                    raw_rip = torch.from_numpy(active_calc_nodes[0].iplv_human_ripple).to(DEVICE, dtype=torch.float32)
                    dyn_rip = torch.nn.functional.adaptive_avg_pool1d(raw_rip.T.unsqueeze(0), total_clock_sectors).squeeze(0).T
                    
                    diff_matrix = (dyn_rip.unsqueeze(1) - dyn_rip.unsqueeze(0)).mean(dim=-1)
                    causal_flow_gpu[:total_clock_sectors, :total_clock_sectors].mul_(0.90).add_(diff_matrix, alpha=0.10)
                    causal_flow_gpu.fill_diagonal_(0.0)

                    net_leads = (causal_flow_gpu[:total_clock_sectors, :total_clock_sectors].sum(dim=1) - 
                                 causal_flow_gpu[:total_clock_sectors, :total_clock_sectors].sum(dim=0))
                    hierarchical_order = torch.argsort(net_leads, descending=True)
                    hierarchical_order_cpu = hierarchical_order.cpu().numpy()

                    root_slot = int(hierarchical_order_cpu[0])
                    root_word = slot_dominant_name[root_slot]

                    tensor_root = semantic_space.get_77_embedding_gpu(root_word)
                    target_77.copy_(tensor_root)

                    dipole_influence = (z_slots_768[root_slot:root_slot+1] / 15.0)
                    target_77.add_(dipole_influence.expand(77, 768), alpha=0.15)
                    accumulated_tree_subspace.copy_(target_77)
                    dag_log_parts = [root_word]

                    root_leads_row = causal_flow_gpu[root_slot, :total_clock_sectors].cpu().numpy()

                    # Рекурсивный Грама-Шмидт с защитой от раздувания шума
                    for s_idx in hierarchical_order_cpu[1:]:
                        child_word = slot_dominant_name[s_idx]
                        t_concept = semantic_space.get_77_embedding_gpu(child_word).clone()
                        
                        slot_dipole = (z_slots_768[s_idx:s_idx+1] / 15.0)
                        t_concept.add_(slot_dipole.expand(77, 768), alpha=0.15)

                        sec_center = -math.pi * 0.5 + (s_idx + 0.5) * sec_w
                        phase_focus = 0.5 * (1.0 + math.cos(dl_phase - sec_center))
                        is_seal = seals.sealed_mask[s_idx]
                        slot_weight = (0.2 + 0.6 * slot_stability[s_idx]) * (0.4 + 0.6 * phase_focus) * (1.4 if is_seal else 1.0)
                        
                        rel_lead = float(root_leads_row[s_idx])

                        if rel_lead >= 0.03: # ВСТРОЙКА (Root ⊃ Child)
                            dot_proj = torch.sum(t_concept * accumulated_tree_subspace, dim=-1, keepdim=True) / (
                                torch.sum(accumulated_tree_subspace**2, dim=-1, keepdim=True) + 1e-7
                            )
                            t_ortho = t_concept - dot_proj * accumulated_tree_subspace
                            ortho_norm = torch.norm(t_ortho, dim=-1, keepdim=True)
                            concept_norm = torch.norm(t_concept, dim=-1, keepdim=True)
                            
                            valid_ortho_mask = (ortho_norm > (0.05 * concept_norm)).float()
                            t_ortho_normed = (t_ortho / (ortho_norm + 1e-6)) * concept_norm * valid_ortho_mask
                            
                            target_77.add_(t_ortho_normed, alpha=slot_weight * 0.35)
                            accumulated_tree_subspace.add_(t_ortho_normed, alpha=0.20)
                            dag_log_parts.append(f"⊃ {child_word}")
                        else: # СОСЕДСТВО (Root || Peer)
                            half_d = latent_dim // 2
                            norm_base = torch.norm(target_77[:, :half_d], dim=-1, keepdim=True) + 1e-7
                            norm_peer = torch.norm(t_concept[:, half_d:], dim=-1, keepdim=True) + 1e-7
                            target_77[:, half_d:].mul_(1.0 - slot_weight * 0.25).add_(
                                (t_concept[:, half_d:] / norm_peer * norm_base), alpha=slot_weight * 0.25
                            )
                            dag_log_parts.append(f"∥ {child_word}")

                    target_77 = torch.nn.functional.normalize(target_77, p=2, dim=-1) * torch.norm(tensor_root, dim=-1, keepdim=True)
                    last_valid_target_77.copy_(target_77)
                    causal_flow_np[:total_clock_sectors, :total_clock_sectors] = causal_flow_gpu[:total_clock_sectors, :total_clock_sectors].cpu().numpy()

            # Синаптическая инерция (Sample-and-Hold)
            if not valid_eeg_input:
                target_77.copy_(last_valid_target_77)
                root_word = last_dominant_root_name
                dag_log_parts = [root_word]

            # -----------------------------------------------------------------
            # 4. FPZ ВЕТО-КОНТРОЛЬ (АНСАМБЛЬ AFZ + FPZ)
            # -----------------------------------------------------------------
            veto_root_word = "ОТКЛЮЧЕН"
            fpz_veto_fired = False

            if fpz_nodes and content_nodes:
                fpz_gating = float(fpz_nodes[0].gating_ratio)
                if fpz_gating >= args.fpz_veto_thresh:
                    fpz_veto_fired = True
                    fpz_raw_iplv = torch.from_numpy(fpz_nodes[0].iplv_32).to(DEVICE, dtype=torch.float32)
                    dyn_fpz = torch.nn.functional.adaptive_avg_pool1d(fpz_raw_iplv.T.unsqueeze(0), total_clock_sectors).squeeze(0).T
                    z_fpz = torch.matmul(dyn_fpz, PHYSICAL_PROJECTION_768x120.T)
                    
                    veto_indices_gpu, _ = semantic_space.decode_concepts_contrastive_gpu(z_fpz, is_veto=False)
                    veto_root_word = semantic_space.words[int(veto_indices_gpu[root_slot].item())].upper()
                    
                    t_veto = semantic_space.get_77_embedding_gpu(veto_root_word)
                    dot_v = torch.sum(target_77 * t_veto, dim=-1, keepdim=True) / (torch.sum(t_veto**2, dim=-1, keepdim=True) + 1e-7)
                    target_77.sub_(dot_v * t_veto, alpha=fpz_gating * args.fpz_veto_gain)
                    target_77 = torch.nn.functional.normalize(target_77, p=2, dim=-1) * torch.norm(tensor_root, dim=-1, keepdim=True)
                    
                    seals.slot_delta_charge.fill(0.0)
                    seals.sealed_mask = [False] * MAX_SLOTS_CAPACITY
            elif is_pure_veto:
                veto_root_word = root_word

            # Печати дельты
            if is_delta_tick and valid_eeg_input:
                delta_cycle_buffer[delta_buf_ptr, :total_clock_sectors].copy_(z_slots_768)
                oldest_ptr = (delta_buf_ptr + 1) % CYCLE_BUF_LEN
                drift_cos = torch.sum(
                    torch.nn.functional.normalize(z_slots_768, p=2, dim=-1) * 
                    torch.nn.functional.normalize(delta_cycle_buffer[oldest_ptr, :total_clock_sectors], p=2, dim=-1),
                    dim=-1
                ).cpu().numpy()
                delta_buf_ptr = oldest_ptr

                for s in range(total_clock_sectors):
                    raw_stab = float(np.clip((drift_cos[s] * 0.5 + 0.5) * (0.2 + 0.8 * current_user.order_drive), 0.0, 1.0))
                    slot_stability[s] = slot_stability[s] * 0.65 + raw_stab * 0.35

                seals.step_delta_cycle(total_clock_sectors, slot_stability, slot_dominant_idx, slot_dominant_name, causal_flow_np)

            # Консольный лог переключений
            if root_word != last_dominant_root_name:
                last_dominant_root_name = root_word
                if args.burst_duration > 0.0 or fpz_veto_fired:
                    switch_burst_timer = time.time()
                full_dag_str = " ".join(dag_log_parts[:6])
                if is_pure_veto:
                    print(f"🚫 [АНТИ-ГЕТЕРАРХИЯ FPZ]: {full_dag_str} (Отказ внутри лора)")
                elif fpz_veto_fired:
                    print(f"⚡ [FPZ СРЫВ ВЕТО]: Вытеснен [{veto_root_word}] ➔ {full_dag_str}")
                else:
                    print(f"🌲 [ПОЗИТИВ AFZ]: {full_dag_str} | [ВЕТО FPZ]: [{veto_root_word}]")

            is_bursting = ((time.time() - switch_burst_timer) < args.burst_duration) or fpz_veto_fired
            base_strength = args.strength_low + (args.strength_high - args.strength_low) * current_user.chaos_drive * 0.4
            dyn_strength = float(np.clip(base_strength + current_user.temporal_bias * 0.08, args.strength_low, args.strength_high))

            if is_bursting:
                dyn_strength = args.burst_strength

            ema_rate = float(np.clip(dt / 0.16, 0.10, 0.40))
            smoothed_prompt_77.mul_(1.0 - ema_rate).add_(target_77, alpha=ema_rate)

            diff_worker.update_conditioning(
                prompt_embeds_np=smoothed_prompt_77.unsqueeze(0).cpu().numpy(),
                pooled_np=smoothed_pooled.unsqueeze(0).cpu().numpy() if is_sdxl_mode else None,
                strength=dyn_strength
            )

            # -----------------------------------------------------------------
            # 5. ПОЛНОЦЕННЫЙ GUI (ПРИ НАЛИЧИИ ЭКРАНА)
            # -----------------------------------------------------------------
            if not args.headless and screen is not None:
                current_canvas = diff_worker.get_canvas()

                if show_minimal_ui:
                    screen.fill((5, 5, 8))
                    surf = pygame.image.frombuffer(current_canvas.tobytes(), (diff_worker.img_w, diff_worker.img_h), 'RGB')
                    scaled_surf = pygame.transform.scale(surf, (WIDTH - 80, HEIGHT - 80))
                    screen.blit(scaled_surf, (40, 40))
                    fps_hud = font_b.render(f"FPS: {diff_worker.fps:.1f} | [H] ВЕРНУТЬ HUD | DAG: {' '.join(dag_log_parts[:3])}", True, (0, 255, 200))
                    screen.blit(fps_hud, (50, 15))
                    pygame.display.flip()
                    continue

                sealed_mask_t = torch.tensor(seals.sealed_mask, dtype=torch.bool, device=DEVICE)
                live_graph_pos, dynamic_radii = gpu_graph.update_physics_gpu(
                    total_clock_sectors, causal_flow_gpu, sealed_mask_t, dt=dt
                )

                screen.fill((8, 12, 18))
                surf = pygame.image.frombuffer(current_canvas.tobytes(), (diff_worker.img_w, diff_worker.img_h), 'RGB')
                if (diff_worker.img_w, diff_worker.img_h) != (512, 384):
                    surf = pygame.transform.scale(surf, (512, 384))
                screen.blit(surf, (370, 45))
                pygame.draw.rect(screen, (40, 55, 75), (370, 45, 512, 384), 2, border_radius=8)

                top_bar = pygame.Rect(370, 10, 512, 28)
                pygame.draw.rect(screen, (16, 22, 32), top_bar, border_radius=4)
                dag_hud = " ".join(dag_log_parts[:4])
                hud_tag = "АНТИ-DAG" if is_pure_veto else "DAG"
                hud_col = (255, 100, 100) if is_pure_veto else (0, 255, 200)
                screen.blit(font_b.render(f"{hud_tag}: {dag_hud} [{diff_worker.fps:.1f} FPS]", True, hud_col), (380, 16))

                px, py = 20, 45
                pygame.draw.rect(screen, (14, 18, 26), (px, py, 330, 555), border_radius=8)
                border_p_col = (255, 80, 80) if is_pure_veto else (0, 255, 200)
                pygame.draw.rect(screen, border_p_col, (px, py, 330, 555), 1, border_radius=8)
                
                p_title = f"FPZ РЕЕСТР ОТКАЗА" if is_pure_veto else f"АНСАМБЛЬ [{current_user.user_id}]"
                screen.blit(font_large.render(p_title, True, border_p_col), (px + 15, py + 10))
                lore_tag = f"ЛОР: {active_vocab_size} СЛОВ" if has_lore else f"СЛОВАРЬ: {active_vocab_size}"
                screen.blit(font_s.render(lore_tag, True, (200, 220, 240)), (px + 15, py + 30))

                row_h = max(24, int(390 / total_clock_sectors))
                for s in range(total_clock_sectors):
                    sy = py + 52 + s * row_h
                    slot_box = pygame.Rect(px + 10, sy, 310, row_h - 2)

                    if mouse_clicked and slot_box.collidepoint(mouse_pos) and has_lore and active_vocab_size <= 16:
                        current_user.manual_slot_concepts[s] = selected_palette_idx

                    is_seal = seals.sealed_mask[s]
                    name = slot_dominant_name[s]
                    c_id = slot_dominant_idx[s]
                    c_rgb = (255, 80, 80) if is_pure_veto else concept_colors[c_id % len(concept_colors)]

                    bg_col = (45, 30, 15) if is_seal else ((35, 20, 20) if is_pure_veto else (20, 35, 45))
                    border_col = (255, 140, 50) if is_seal else ((255, 80, 80) if is_pure_veto else c_rgb)

                    pygame.draw.rect(screen, bg_col, slot_box, border_radius=4)
                    pygame.draw.rect(screen, border_col, slot_box, 1, border_radius=4)

                    tag = f"S{s+1}"
                    th_role = "Вето" if (is_pure_veto and s == root_slot) else ("Корень" if s == root_slot else ("Вх" if (s % 2 == 0) else "Пм"))
                    seal_icon = "🚫" if is_pure_veto else ("🔒" if is_seal else "●")
                    label_col = (255, 120, 120) if is_pure_veto else c_rgb
                    screen.blit(font_b.render(f"{seal_icon}{tag}[{th_role}] {name[:10]}", True, label_col), (px + 14, sy + 3))

                    stab_w = int(slot_stability[s] * 50)
                    pygame.draw.rect(screen, (30, 45, 40), (px + 185, sy + 6, 50, 7), border_radius=2)
                    pygame.draw.rect(screen, (80, 255, 160), (px + 185, sy + 6, stab_w, 7), border_radius=2)

                    star = " *" if (s == current_clock_sec) else ""
                    screen.blit(font_s.render(f"Сек {s+1}{star}", True, (150, 160, 170)), (px + 245, sy + 4))

                status_str = f"• Fpz Вето: ВЕТВЛЕНИЕ [{veto_root_word[:7]}]" if (is_pure_veto or fpz_nodes) else "• Fpz Вето: [ОТКЛЮЧЕН]"
                screen.blit(font_s.render(status_str, True, (255, 120, 120) if (is_pure_veto or fpz_veto_fired) else (180, 200, 220)), (px + 15, py + 485))
                screen.blit(font_s.render(f"• F3 Порядок: {current_user.order_drive:.2f} | F4 Хаос: {current_user.chaos_drive:.2f}", True, (200, 220, 240)), (px + 15, py + 505))
                screen.blit(font_s.render(f"• Сила: {dyn_strength:.2f} | 60 FPS Native", True, (100, 255, 200)), (px + 15, py + 525))

                # Правая панель: Граф гетерархии
                gx_p, gy_p, gw_p, gh_p = 910, 45, 860, 555
                pygame.draw.rect(screen, (12, 16, 24), (gx_p, gy_p, gw_p, gh_p), border_radius=8)
                border_g_col = (255, 80, 80) if is_pure_veto else (100, 180, 255)
                pygame.draw.rect(screen, border_g_col, (gx_p, gy_p, gw_p, gh_p), 1, border_radius=8)
                g_title = f"АНТИ-ГЕТЕРАРХИЯ ОТКАЗА FPZ ({total_clock_sectors} СЛОТОВ)" if is_pure_veto else f"ГЕТЕРАРХИЯ СМЫСЛОВ ({total_clock_sectors} СЛОТОВ)"
                screen.blit(font_large.render(g_title, True, border_g_col), (gx_p + 20, gy_p + 15))

                for i in range(total_clock_sectors):
                    for j in range(total_clock_sectors):
                        if i != j and causal_flow_np[i, j] > 0.015:
                            p_start = (int(live_graph_pos[i][0]), int(live_graph_pos[i][1]))
                            p_end = (int(live_graph_pos[j][0]), int(live_graph_pos[j][1]))
                            arrow_col = (255, 80, 80) if is_pure_veto else (40, 140, 200)
                            pygame.draw.line(screen, arrow_col, p_start, p_end, 2)

                for s in range(total_clock_sectors):
                    pt = (int(live_graph_pos[s][0]), int(live_graph_pos[s][1]))
                    is_seal = seals.sealed_mask[s]
                    name = slot_dominant_name[s]
                    c_id = slot_dominant_idx[s]
                    c_rgb = (255, 80, 80) if is_pure_veto else concept_colors[c_id % len(concept_colors)]
                    node_r = int(dynamic_radii[s])

                    pygame.draw.circle(screen, (c_rgb[0]//3, c_rgb[1]//3, c_rgb[2]//3), pt, node_r + 6)
                    pygame.draw.circle(screen, (255, 140, 50) if is_seal else c_rgb, pt, node_r)
                    pygame.draw.circle(screen, (255, 255, 255) if is_seal else (200, 220, 240), pt, node_r, 2)

                    txt = font_b.render(name[:8], True, (10, 20, 30))
                    screen.blit(txt, (pt[0] - txt.get_width() // 2, pt[1] - 8))
                    screen.blit(font_s.render(f"S{s+1}", True, (200, 220, 240)), (pt[0] - 10, pt[1] + node_r + 2))

                # Нижняя панель: Хроно-туннель Дельты + Палитра Лора
                by, bh = 620, 310
                pygame.draw.rect(screen, (10, 14, 20), (20, by, 1750, bh), border_radius=8)
                pygame.draw.rect(screen, (40, 60, 80), (20, by, 1750, bh), 1, border_radius=8)
                screen.blit(font_b.render(f"ХРОНО-ТУННЕЛЬ ДЕЛЬТЫ (S={total_clock_sectors} СЕКТОРОВ)", True, (0, 255, 200)), (35, by + 12))

                clock_cx, clock_cy = 440, by + 165
                spiral_history.render(
                    screen=screen, cx=clock_cx, cy=clock_cy, R_max=122, R_core=28,
                    dl_phase=dl_phase, total_sectors=total_clock_sectors,
                    current_concepts=slot_dominant_idx, current_sealed=any(seals.sealed_mask[:total_clock_sectors]),
                    concept_colors=concept_colors, is_active_sec=current_clock_sec
                )

                pal_x = 910
                if has_lore and active_vocab_size <= 16:
                    screen.blit(font_b.render("КАРТИНА МИРА (ПАЛИТРА ЛОРА):", True, (100, 180, 255)), (pal_x, by + 45))
                    for i, name in enumerate(semantic_space.words[:8]):
                        col = concept_colors[i % len(concept_colors)]
                        cx_box = pal_x + (i % 4) * 205
                        cy_box = by + 75 + (i // 4) * 95
                        pygame.draw.rect(screen, (20, 28, 40), (cx_box, cy_box, 190, 75), border_radius=6)
                        pygame.draw.rect(screen, col, (cx_box, cy_box, 190, 75), 2, border_radius=6)
                        key_tag = pygame.key.name(PALETTE_HOTKEYS[i]).upper()
                        screen.blit(font_b.render(f"[{key_tag}] {name.upper()[:12]}", True, col), (cx_box + 15, cy_box + 12))
                else:
                    title_text = f"АКТИВНЫЙ ЛОР ({active_vocab_size} КОНЦЕПТОВ):" if has_lore else f"СЛОВАРЬ ({active_vocab_size} СЛОВ):"
                    screen.blit(font_b.render(title_text, True, (100, 180, 255)), (pal_x, by + 45))
                    for i in range(min(8, total_clock_sectors)):
                        w_name = slot_dominant_name[i]
                        c_id = slot_dominant_idx[i]
                        col = (255, 80, 80) if is_pure_veto else concept_colors[c_id % len(concept_colors)]
                        cx_box = pal_x + (i % 4) * 205
                        cy_box = by + 95 + (i // 4) * 85
                        pygame.draw.rect(screen, (20, 28, 40), (cx_box, cy_box, 190, 65), border_radius=6)
                        pygame.draw.rect(screen, col, (cx_box, cy_box, 190, 65), 1, border_radius=6)
                        slot_tag = "ВЕТО" if is_pure_veto else f"S{i+1}"
                        screen.blit(font_b.render(f"{slot_tag}: {w_name[:11]}", True, col), (cx_box + 15, cy_box + 22))

                pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        diff_worker.running = False
        engine.stop()
        if not args.headless:
            pygame.quit()

if __name__ == '__main__':
    main()
