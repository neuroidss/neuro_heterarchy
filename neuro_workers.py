import time
import math
import numpy as np
import cv2
import torch
import threading
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from multiprocessing.connection import Client

from neuro_heterarchy_core import DEVICE
from neuro_models import DynamicPilot, MAX_CONCEPTS_CAPACITY

FEIGENBAUM_DELTA = 4.669201609

FRACTAL_COLORS = [
    (20, 20, 60), (30, 80, 120), (100, 100, 110), (150, 100, 50),
    (40, 100, 60), (80, 40, 80), (10, 10, 20), (20, 80, 40),
    (120, 40, 40), (40, 120, 120), (80, 80, 30), (30, 30, 90),
    (90, 40, 90), (40, 90, 40), (110, 70, 30), (50, 50, 50)
]

def apply_manifold_camera_warp(img_np: np.ndarray, pilot: DynamicPilot, dt: float = 0.016):
    h, w = img_np.shape[:2]
    zoom = 1.0 + (pilot.vy * dt * 0.18)
    dx = -pilot.vx * w * dt * 0.12
    angle = -pilot.wm_curvature * 10.0 * dt

    if abs(zoom - 1.0) < 0.0008 and abs(dx) < 0.15 and abs(angle) < 0.04: return img_np
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
    M[0, 2] += dx
    return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_color_surgery(img_np, old_f32):
    res = img_np.astype(np.float32)
    mu = np.mean(res, axis=(0, 1))
    target_g = (mu[0] + mu[2]) / 2.0
    if mu[1] > target_g: res[:, :, 1] -= (mu[1] - target_g) * 1.0
    color_p = 0.015
    green = res[:, :, 1] * color_p
    res[:, :, 1] -= green
    res[:, :, 0] += green * 0.5
    res[:, :, 2] += green * 0.5
    mu_t, std_t = cv2.meanStdDev(res)
    mu_s, std_s = cv2.meanStdDev(old_f32)
    t_std = std_t * 0.85 + std_s * 0.15
    res = (res - mu_t.reshape(1, 1, 3)) * (t_std / (std_t + 1e-5)).reshape(1, 1, 3) + mu_t.reshape(1, 1, 3)
    return np.clip(res, 0, 255).astype(np.uint8)

def calculate_emergent_treemap(x, y, w, h, active_weights, k_score, lead_sign, count):
    valid_weights = active_weights[:count]
    active_indices = [i for i, val in enumerate(valid_weights) if val > 0.03]
    if not active_indices:
        top_i = int(np.argmax(valid_weights)) if count > 0 else 0
        return {top_i: (x, y, w, h, 0)}, [top_i]

    sorted_by_weight = sorted(active_indices, key=lambda idx: valid_weights[idx], reverse=True)
    if lead_sign < 0:
        sorted_by_weight = sorted_by_weight[::-1]

    tau = float(np.clip((k_score - 1.2) / 2.0, 0.0, 1.0))
    boxes = {}

    flat_boxes = {}
    nw = w / max(1, len(sorted_by_weight))
    for idx, c_idx in enumerate(sorted_by_weight):
        flat_boxes[c_idx] = (x + idx * nw, y, nw, h)

    nest_boxes = {}
    if len(sorted_by_weight) == 1:
        nest_boxes[sorted_by_weight[0]] = (x, y, w, h)
    else:
        root_idx = sorted_by_weight[0]
        nest_boxes[root_idx] = (x, y, w, h)
        margin_x = w * 0.06
        margin_y = h * 0.08
        inner_x = x + margin_x
        inner_y = y + margin_y * 1.5
        inner_w = w - 2 * margin_x
        inner_h = h - margin_y * 2.0
        children = sorted_by_weight[1:]
        child_w = inner_w / max(1, len(children))
        for i, ch_idx in enumerate(children):
            nest_boxes[ch_idx] = (inner_x + i * child_w, inner_y, child_w, inner_h)

    for rank, c_idx in enumerate(sorted_by_weight):
        fx, fy, fw, fh = flat_boxes[c_idx]
        nx, ny, nw_b, nh = nest_boxes[c_idx]
        cur_x = fx * (1 - tau) + nx * tau
        cur_y = fy * (1 - tau) + ny * tau
        cur_w = fw * (1 - tau) + nw_b * tau
        cur_h = fh * (1 - tau) + nh * tau
        boxes[c_idx] = (int(cur_x), int(cur_y), int(max(15, cur_w)), int(max(15, cur_h)), rank)

    return boxes, sorted_by_weight

class VisualCLIPTeacher:
    def __init__(self, text_prompts):
        self.text_prompts = list(text_prompts)
        self.ready = False
        self.model, self.processor, self.text_features = None, None, None
        threading.Thread(target=self._init_bg, daemon=True).start()

    def _init_bg(self):
        try:
            model_id = "openai/clip-vit-large-patch14"
            self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
            self.processor = CLIPProcessor.from_pretrained(model_id)
            with torch.no_grad():
                inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True).to(DEVICE)
                feat = self.model.get_text_features(**inputs)
                self.text_features = feat / feat.norm(dim=-1, keepdim=True)
            self.ready = True
            print("✅ [CLIP] Фоновый учитель активен!")
        except Exception as e:
            print(f"⚠️ [CLIP ERROR]: {e}")

    def classify(self, rgb_image_np: np.ndarray, count: int) -> np.ndarray:
        if not self.ready or rgb_image_np is None or count == 0: return np.zeros(count, dtype=np.float32)
        try:
            inputs = self.processor(images=Image.fromarray(rgb_image_np), return_tensors="pt").to(DEVICE)
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            with torch.no_grad():
                img_feat = self.model.get_image_features(**inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = (img_feat @ self.text_features[:count].T) * 30.0
                return torch.softmax(logits, dim=-1).cpu().numpy()[0]
        except Exception:
            return np.zeros(count, dtype=np.float32)

class EqualPoolChaosWorker:
    def __init__(self, initial_prompts: list, initial_names: list, port: int = 6000, mode: str = "lcm", speed: str = "fast", use_taesd: bool = True, use_color: bool = True):
        self.conn = None
        self.speed, self.use_taesd, self.use_color = speed, use_taesd, use_color
        self.img_w, self.img_h = (448, 336) if speed == "fast" else (512, 384)
        self.current_rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.initialized = False
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.strength = 0.50
        self.frame_id = 0
        self.mode = mode
        
        self.prompts = list(initial_prompts)
        self.names = list(initial_names)
        self.active_count = len(self.prompts)
        self.c_bases = []
        self.depths = [0] * MAX_CONCEPTS_CAPACITY
        self.energies = [0.0] * MAX_CONCEPTS_CAPACITY
        self.parent_map = {i: None for i in range(MAX_CONCEPTS_CAPACITY)}
        
        self.chaos_enabled = False
        self.latent_active = None
        self.dir_u, self.dir_v = None, None
        self.last_split_time = time.time()
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def toggle_chaos(self):
        with self.lock:
            self.chaos_enabled = not self.chaos_enabled
            print(f"\n🌀 [ХАОС ФЕЙГЕНБАУМА]: {'ВКЛЮЧЕН' if self.chaos_enabled else 'ВЫКЛЮЧЕН'}")

    def bifurcate_concept(self, parent_idx: int, torus_u: float, torus_v: float, lead_sign: float):
        if self.active_count >= MAX_CONCEPTS_CAPACITY: return -1
        new_idx = self.active_count
        self.active_count += 1
        p_depth = self.depths[parent_idx]
        self.depths[new_idx] = p_depth + 1
        self.energies[parent_idx], self.energies[new_idx] = 0.0, 0.0
        self.parent_map[new_idx] = parent_idx
        
        drift_scale = 0.55 / (1.35 ** p_depth)
        drift = (self.dir_u * math.cos(torus_u) + self.dir_v * math.sin(torus_v)) * drift_scale
        parent_base = self.c_bases[parent_idx].clone()
        child_base = parent_base + drift
        child_base = child_base / torch.norm(child_base, dim=-1, keepdim=True) * torch.norm(parent_base, dim=-1, keepdim=True)
        self.c_bases.append(child_base)
        
        base_parent_name = self.names[parent_idx].split(".")[0]
        direction_tag = "in" if lead_sign >= 0 else "out"
        new_name = f"{base_parent_name}.{self.depths[new_idx]}{direction_tag}"
        self.names.append(new_name)
        print(f"🌿 [БИФУРКАЦИЯ ДЕРЕВА] {self.names[parent_idx]} -> Потомок: [{new_name}] (D{self.depths[new_idx]})")
        return new_idx

    def update_cycle(self, leader_idx: int, child_idx: int, beta_order: float, beta_chaos: float, 
                     torus_u: float, torus_v: float, smooth_depth: float, lead_sign: float, 
                     rx_sagitta: float, subjects: list, resolved_parent_map: dict):
        if not self.initialized or len(self.c_bases) == 0: return

        with torch.inference_mode():
            if self.chaos_enabled:
                chaos_drive = float(np.clip(1.0 - beta_chaos, 0.0, 1.0))
                order_drive = float(np.clip(beta_order, 0.0, 1.0))
                safe_l = max(0, min(leader_idx, len(self.c_bases) - 1))
                
                self.energies[safe_l] = np.clip(self.energies[safe_l] + chaos_drive * 0.07 - order_drive * 0.03, 0.0, 5.0)
                split_threshold = 0.70 / (FEIGENBAUM_DELTA ** self.depths[safe_l])

                if self.energies[safe_l] > split_threshold and (time.time() - self.last_split_time > 2.0):
                    new_child_idx = self.bifurcate_concept(safe_l, torus_u, torus_v, lead_sign)
                    if new_child_idx != -1:
                        self.last_split_time = time.time()
                        for s in subjects: s.heterarchy.inherit_synapses(safe_l, new_child_idx)

                if order_drive > 0.85 and self.active_count > 2 and (time.time() - self.last_split_time > 3.0):
                    removed_name = self.names.pop()
                    self.c_bases.pop()
                    self.active_count = len(self.c_bases)
                    self.last_split_time = time.time()
                    print(f"🕳️ [КРИЗИС ГРАНИЦ] Схлопнут узел: [{removed_name}]")

            current_len = len(self.c_bases)
            if current_len == 0: return
            l_idx = max(0, min(leader_idx, current_len - 1))
            c_idx = max(0, min(child_idx, current_len - 1))
            
            w_world = self.c_bases[l_idx]   # Форма / Контур (Лидер, Левое полушарие)
            a_agent = self.c_bases[c_idx]   # Стиль / Текстура (Ведомый, Правое полушарие)

            if l_idx != c_idx:
                target = torch.zeros_like(w_world)
                half_d = target.shape[-1] // 2
                
                norm_w = torch.norm(w_world[:, :half_d], dim=-1, keepdim=True) + 1e-7
                norm_a = torch.norm(a_agent[:, half_d:], dim=-1, keepdim=True) + 1e-7
                
                target[:, :half_d] = w_world[:, :half_d] / norm_w
                target[:, half_d:] = a_agent[:, half_d:] / norm_a
                
                target = target / torch.norm(target, dim=-1, keepdim=True) * torch.norm(w_world, dim=-1, keepdim=True)
            else:
                target = w_world

            alpha = float(np.clip(0.30 - beta_order * 0.15 + beta_chaos * 0.15 + rx_sagitta * 0.10, 0.15, 0.65))
            with self.lock:
                if self.latent_active is None: self.latent_active = target.clone()
                else: self.latent_active = self.latent_active * (1.0 - alpha) + target * alpha

    def _loop(self):
        times = []
        while self.running:
            if not self.initialized:
                try:
                    self.conn = Client(('localhost', 6000), authkey=b'brain')
                    self.conn.send({'cmd': 'init_mode', 'mode': self.mode, 'use_taesd': self.use_taesd})
                    self.conn.recv()
                    self.conn.send({'cmd': 'encode_base_prompts', 'prompts': self.prompts})
                    enc_resp = self.conn.recv()
                    self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in enc_resp['c_bases']]
                    
                    gen = torch.Generator(device=DEVICE).manual_seed(42)
                    u = torch.randn_like(self.c_bases[0], generator=gen)
                    v = torch.randn_like(self.c_bases[0], generator=gen)
                    self.dir_u = u / torch.norm(u, dim=-1, keepdim=True)
                    self.dir_v = v / torch.norm(v, dim=-1, keepdim=True)
                    
                    with self.lock:
                        if len(self.c_bases) > 0: self.latent_active = self.c_bases[0].clone()
                    self.initialized = True
                    print(f"✅ [BACKEND] Diffusion активен. Загружено базовых концептов: {len(self.c_bases)}")
                except Exception:
                    time.sleep(0.5)
                    continue

            with self.lock:
                latent = self.latent_active.clone().cpu().numpy() if self.latent_active is not None else None
                img = self.current_rgb.copy()
                s_val = self.strength

            if latent is None: time.sleep(0.04); continue

            try:
                t0 = time.time()
                req = {'cmd': 'generate', 'image_np': img, 'prompt_embeds': latent, 'strength': max(0.20, min(0.85, s_val))}
                if self.speed == "fast": req['num_inference_steps'] = 3
                self.conn.send(req)
                resp = self.conn.recv()

                with self.lock:
                    if isinstance(resp, np.ndarray):
                        if resp.shape[:2] != (self.img_h, self.img_w): resp = cv2.resize(resp, (self.img_w, self.img_h))
                        treated = apply_color_surgery(resp, img.astype(np.float32)) if self.use_color else resp
                        self.current_rgb = cv2.addWeighted(self.current_rgb, 0.25, treated, 0.75, 0)
                        self.frame_id += 1

                times.append(time.time() - t0)
                if len(times) > 5: times.pop(0)
                self.fps = 1.0 / (np.mean(times) + 1e-6)
            except Exception:
                self.initialized = False
                time.sleep(0.5)
