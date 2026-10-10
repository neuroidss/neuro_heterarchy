#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: COMPUTATIONAL GENESIS ENGINE (GPU TENSOR PIPELINE)
===================================================================================
- БЕЗОПАСНАЯ РАБОТА С 0, 1 И N КОНЦЕПТАМИ: Никаких чёрных экранов и делений на ноль.
- НУЛЕВОЙ УРОВЕНЬ: При пустом лоре стартует с [] БЕЗ выгрузки 50k токенов.
- СВОБОДНЫЙ ЛАТЕНТНЫЙ ДРЕЙФ: Генерация от последнего концепта или чистого CLIP.
- ЧИСТАЯ ФАЗОВАЯ КОГЕРЕНТНОСТЬ (НОЛЬ СКАЛЯРНОЙ МОЩНОСТИ / ZERO POWER):
    * F3:  2D-проекция 120 диполей на манифолд рангов (Fan 2024) + Курамото R_beta.
    * F4:  Фазовая энтропия и циркулярная дисперсия 120 диполей (Flesch 2022).
    * AFz: Каузальная матрица фазового опережения рипплов 89.5 Гц (Dickey et al. 2022).
    * FCz: 4D кинематика фазового потока [lx, ly, rx, ry] (Colgin 2009 / Bieri 2014).
    * Fpz: Скорость фазового дрейфа и срыв аттрактора Хопфилда по фазовому градиенту.
- ЧЕСТНАЯ ДЕЛЬТА-ПЕЧАТЬ (WORLD SEALS):
    * Буфер delta_cycle_buffer на GPU сравнивает косинусное сходство 768-D слотов.
    * При смене мысли стабильность падает ниже --seal-thresh и печать РАЗЛОЧИВАЕТСЯ.
===================================================================================
"""

from __future__ import annotations
from typing import Optional, Union
import os
import sys
import math
import time
import json
import threading
import numpy as np
import torch
from dataclasses import dataclass, field
from transformers import CLIPTokenizer, CLIPTextModel

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y, DX_120, DY_120

MAX_SLOTS_CAPACITY = 16

# -----------------------------------------------------------------------------
# ДЕТЕРМИНИРОВАННЫЙ БАЗИС: 120 ДИПОЛЕЙ -> 768 КОЛОНОК НА GPU
# -----------------------------------------------------------------------------
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

DX_GPU_120 = torch.tensor(DX_120, dtype=torch.float32, device=DEVICE)
DY_GPU_120 = torch.tensor(DY_120, dtype=torch.float32, device=DEVICE)

class UniversalSemanticSpace:
    def __init__(self, lore_file=None, concepts_cli=None, target_k=None, model_id="openai/clip-vit-large-patch14"):
        print("⏳ [СЕМАНТИКА] Инициализация CLIP Text Model на GPU...")
        self.lock = threading.RLock()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE).eval()
        self.gpu_embed_cache = {}
        self.affordance_cache = {}

        # Базовый чистый эмбеддинг пустой строки
        self.empty_embed = self._compute_77_embedding("")
        self.last_active_embed = self.empty_embed.clone()

        custom_words = []
        if lore_file and os.path.exists(lore_file):
            if lore_file.endswith('.json'):
                with open(lore_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    custom_words = [str(x).strip() for x in (d if isinstance(d, list) else d.keys()) if str(x).strip()]
            else:
                with open(lore_file, 'r', encoding='utf-8') as f:
                    custom_words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        elif concepts_cli and concepts_cli.strip().lower() not in ["none", ""]:
            custom_words = [w.strip() for w in concepts_cli.split(",") if w.strip()]

        # СТРОГО: Если пользователь не передал лор — список пуст! НЕ грузим 50 000 слов!
        self.words = custom_words
        self._rebuild_vocab_matrix_locked()

    def _compute_77_embedding(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            return self.text_model(tokens).last_hidden_state[0]

    def _rebuild_vocab_matrix_locked(self):
        with self.lock:
            n_words = len(self.words)
            if n_words == 0:
                # 0 Концептов: используем вектор свободного потока
                raw = self.last_active_embed.mean(dim=0, keepdim=True)
                self.norm_vocab = torch.nn.functional.normalize(raw, p=2, dim=-1)
                self.affordance_cache.clear()
                return

            lore_embeds = [self.get_77_embedding_gpu(w).mean(dim=0) for w in self.words]
            raw = torch.stack(lore_embeds, dim=0)

            if n_words == 1:
                # 1 Концепт: избегаем степеней свободы <= 0
                self.norm_vocab = torch.nn.functional.normalize(raw, p=2, dim=-1)
            else:
                mu = torch.mean(raw, dim=0, keepdim=True)
                std = torch.std(raw - mu, dim=0, keepdim=True, unbiased=False) + 1e-6
                self.norm_vocab = torch.nn.functional.normalize((raw - mu) / std, p=2, dim=-1)
            
            self.affordance_cache.clear()

    def update_concepts_on_the_fly(self, new_words: list[str]):
        """Потокобезопасная атомарная пересборка словаря без гонок памяти"""
        cleaned = [w.strip() for w in new_words if w.strip()]
        with self.lock:
            if not cleaned and self.words:
                # Запоминаем последний образ перед полным очищением списка
                self.last_active_embed = self.get_77_embedding_gpu(self.words[0]).clone()
            self.words = list(cleaned)
            self._rebuild_vocab_matrix_locked()
            print(f"🔄 [СЕМАНТИКА GPU] Пул концептов атомарно обновлен (всего: {len(self.words)}): {self.words}")

    def get_77_embedding_gpu(self, word: str) -> torch.Tensor:
        with self.lock:
            if not word or not word.strip():
                return self.last_active_embed
            w_clean = word.strip().lower()
            if w_clean in self.gpu_embed_cache: return self.gpu_embed_cache[w_clean]
            emb = self._compute_77_embedding(w_clean)
            self.gpu_embed_cache[w_clean] = emb
            return emb

    def get_concept_affordances_svd(self, concept_idx: int, k_neighbors: int = 32, max_bases: int = 16) -> torch.Tensor:
        with self.lock:
            n_items = self.norm_vocab.shape[0]
            if n_items <= 1:
                gen = torch.Generator(device=DEVICE).manual_seed(42)
                rand_bases = torch.randn((max_bases, self.norm_vocab.shape[-1]), device=DEVICE, dtype=torch.float32, generator=gen)
                return torch.nn.functional.normalize(rand_bases, p=2, dim=-1)

            safe_c_idx = concept_idx % n_items
            if safe_c_idx in self.affordance_cache: return self.affordance_cache[safe_c_idx]
            target = self.norm_vocab[safe_c_idx:safe_c_idx+1]
            sims = torch.mm(target, self.norm_vocab.T).squeeze(0)
            _, topk = torch.topk(sims, min(k_neighbors, n_items))
            centered = self.norm_vocab[topk] - target
            _, _, V = torch.linalg.svd(centered, full_matrices=False)
            n_found = V.shape[0]
            if n_found < max_bases:
                needed = max_bases - n_found
                gen = torch.Generator(device=DEVICE).manual_seed(safe_c_idx + 42)
                pad = torch.randn((needed, target.shape[-1]), device=DEVICE, dtype=target.dtype, generator=gen)
                if n_found > 0: pad = pad - torch.mm(torch.mm(pad, V.T), V)
                pad = torch.nn.functional.normalize(pad, p=2, dim=-1)
                tangent = torch.cat([V, pad], dim=0)[:max_bases]
            else:
                tangent = V[:max_bases].clone()
            self.affordance_cache[safe_c_idx] = tangent
            return tangent

    def decode_slots(self, z_slots_768: torch.Tensor):
        with self.lock:
            w_len = len(self.words)
            if w_len == 0:
                sims = torch.zeros((z_slots_768.shape[0], 1), device=DEVICE)
                soft_w = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, soft_w, ["СВОБОДНЫЙ_ДРЕЙФ"] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]

            if w_len == 1:
                sims = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                soft_w = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, soft_w, [self.words[0].upper()] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]

            z_norm = torch.nn.functional.normalize(z_slots_768, p=2, dim=-1)
            sims = torch.mm(z_norm, self.norm_vocab.T)
            top = torch.argmax(sims, dim=-1)
            names = [self.words[int(i) % w_len].upper() for i in top]
            indices = [int(i) % w_len for i in top]
            return sims, torch.softmax(sims / 0.12, dim=-1), names, indices

class DeltaCycleSealEngine:
    def __init__(self, max_slots=16, seal_cycles=4, decay_cycles=2, seal_thresh=0.68):
        self.max_slots = max_slots
        self.seal_cycles = seal_cycles
        self.decay_cycles = decay_cycles
        self.seal_thresh = seal_thresh
        self.charge = np.zeros(max_slots, dtype=np.float32)
        self.mask = [False] * max_slots

    def step(self, n_sec, stabs, c_flow_np):
        for s in range(n_sec):
            rip = np.sum(np.abs(c_flow_np[s, :n_sec])) + np.sum(np.abs(c_flow_np[:n_sec, s]))
            if stabs[s] >= self.seal_thresh:
                charge_delta = 1.0 + float(np.clip(rip * 2.0, 0.0, 1.5))
                self.charge[s] = min(float(self.seal_cycles), self.charge[s] + charge_delta)
            else:
                loss = float(self.seal_cycles) / float(max(1, self.decay_cycles))
                self.charge[s] = max(0.0, self.charge[s] - loss)

            if self.mask[s] and self.charge[s] < (self.seal_cycles * 0.20):
                self.mask[s] = False

        ready = [s for s in range(n_sec) if not self.mask[s] and self.charge[s] >= self.seal_cycles]
        if len(ready) >= 2:
            for s in ready: self.mask[s] = True

class SelfSufficientHeterarchyRouter:
    ALL_ROLES = ["AFz", "F3", "F4", "FCz", "Fpz"]

    def __init__(self, users_setup=None):
        self.overrides = {}
        self.static_map = {}
        if users_setup and users_setup.strip().lower() not in ["none", "auto"]:
            for u in users_setup.split(";"):
                if ":" in u:
                    for pair in u.split(":")[1].split(","):
                        if "=" in pair:
                            r, d = pair.split("=")
                            if d.strip().isdigit(): self.static_map[int(d.strip())] = r.strip()

    def set_role(self, dev_idx: int, role: Optional[str] = None):
        self.overrides[dev_idx] = role

    def resolve(self, live_nodes: list) -> dict:
        num_live = len(live_nodes)
        resolved = {r: None for r in self.ALL_ROLES}
        if num_live == 0: return resolved

        for dev_idx in range(num_live):
            role = self.overrides.get(dev_idx, self.static_map.get(dev_idx, "AFz" if dev_idx == 0 and not self.static_map else None))
            if role in self.ALL_ROLES and resolved[role] is None:
                resolved[role] = live_nodes[dev_idx]
        return resolved

class UniversalColumnProcessor:
    def __init__(self, role: str, semantic: UniversalSemanticSpace, 
                 seal_cycles: int = 4, decay_cycles: int = 2, seal_thresh: float = 0.68, refresh_cycles: int = 2):
        self.role = role
        self.semantic = semantic
        self.seals = DeltaCycleSealEngine(MAX_SLOTS_CAPACITY, seal_cycles, decay_cycles, seal_thresh)
        self.stabs = np.zeros(MAX_SLOTS_CAPACITY, dtype=np.float32)

        self.buf_len = max(2, refresh_cycles)
        self.delta_cycle_buffer = torch.zeros((self.buf_len, MAX_SLOTS_CAPACITY, 768), dtype=torch.float32, device=DEVICE)
        self.buf_ptr = 0

    def process(self, node, n_sec: int, sec_idx: int, dl_phase: float, is_tick: bool):
        if node is None: return None
        raw_iplv = torch.from_numpy(node.iplv_32).to(DEVICE, dtype=torch.float32)
        if torch.norm(raw_iplv) <= 1e-4: return None

        dynamic_wm = torch.nn.functional.adaptive_avg_pool1d(raw_iplv.T.unsqueeze(0), n_sec).squeeze(0).T
        z_slots = torch.matmul(dynamic_wm, PHYSICAL_PROJECTION_768x120.T)
        
        with self.semantic.lock:
            words_copy = list(self.semantic.words)
            w_count = len(words_copy)
            
            _, soft_w, names, indices = self.semantic.decode_slots(z_slots)

            kuramoto_r = float(torch.clamp(torch.norm(dynamic_wm.mean(dim=0)) / 10.0, 0.0, 1.0).item())
            x_local = float(torch.clamp(torch.sum(dynamic_wm[sec_idx % n_sec] * DX_GPU_120) / 40.0, -1.0, 1.0).item())
            y_global = float(torch.clamp(torch.sum(dynamic_wm[sec_idx % n_sec] * DY_GPU_120) / 40.0, -1.0, 1.0).item())

            z_past = raw_iplv[0]
            z_future = raw_iplv[-1]
            norm_past, norm_future = torch.norm(z_past), torch.norm(z_future)
            temporal_asymmetry = float(((norm_future - norm_past) / (norm_future + norm_past + 1e-6)).item())

            raw_rip = torch.from_numpy(node.iplv_human_ripple).to(DEVICE, dtype=torch.float32)
            dyn_rip = torch.nn.functional.adaptive_avg_pool1d(raw_rip.T.unsqueeze(0), n_sec).squeeze(0).T
            diff_rip = dyn_rip.unsqueeze(1) - dyn_rip.unsqueeze(0)
            c_flow_768 = torch.matmul(diff_rip, PHYSICAL_PROJECTION_768x120.T)
            c_flow_np = torch.norm(c_flow_768, dim=-1).cpu().numpy()

            net_leads = (c_flow_768.sum(dim=1) - c_flow_768.sum(dim=0)).sum(dim=-1).cpu().numpy()
            order = np.argsort(net_leads)[::-1]
            root_s = int(order[0])

            tree_77 = torch.zeros((77, 768), dtype=torch.float32, device=DEVICE)

            if w_count == 0:
                # Свободный дрейф от чистого вектора
                tree_77.copy_(self.semantic.empty_embed)
                n_tok = min(n_sec, 12)
                tree_77[1:n_tok+1].add_((z_slots[root_s:root_s+1] / 15.0).expand(n_tok, 768), alpha=0.30)
                dag_parts = ["СВОБОДНЫЙ_ДРЕЙФ"]
            elif w_count == 1:
                # 1 Концепт: устойчиво отображаем образ
                tree_77.copy_(self.semantic.get_77_embedding_gpu(words_copy[0]))
                n_tok = min(n_sec, 12)
                tree_77[1:n_tok+1].add_((z_slots[root_s:root_s+1] / 15.0).expand(n_tok, 768), alpha=0.25)
                dag_parts = [words_copy[0].upper()]
            else:
                k_choices = min(16, soft_w.shape[-1], w_count)
                top_w, top_i = torch.topk(soft_w[root_s, :w_count], k_choices)
                top_w = top_w / (top_w.sum() + 1e-6)

                for i in range(top_w.shape[0]):
                    w_idx = int(top_i[i]) % w_count
                    tree_77.add_(self.semantic.get_77_embedding_gpu(words_copy[w_idx]), alpha=top_w[i])

                n_tok = min(n_sec, 12)
                tree_77[1:n_tok+1].add_((z_slots[root_s:root_s+1] / 15.0).expand(n_tok, 768), alpha=0.25)
                accum = tree_77.clone()
                dag_parts = [names[root_s]]
                sec_w = (2.0 * math.pi) / float(n_sec)
                root_leads = c_flow_np[root_s, :n_sec]

                for s in order[1:]:
                    child_w = names[s]
                    t_child = self.semantic.get_77_embedding_gpu(child_w).clone()
                    t_child.add_((z_slots[s:s+1] / 15.0).expand(77, 768), alpha=0.20)
                    center = -math.pi * 0.5 + (s + 0.5) * sec_w
                    focus = 0.5 * (1.0 + math.cos(dl_phase - center))
                    is_seal = self.seals.mask[s]
                    w_slot = (0.25 + 0.55 * self.stabs[s]) * (0.4 + 0.6 * focus) * (1.3 if is_seal else 1.0)

                    if float(root_leads[s]) >= 0.03:
                        dot = torch.sum(t_child * accum, dim=-1, keepdim=True) / (torch.sum(accum**2, dim=-1, keepdim=True) + 1e-7)
                        ortho = t_child - dot * accum
                        no, nc = torch.norm(ortho, dim=-1, keepdim=True), torch.norm(t_child, dim=-1, keepdim=True)
                        ortho_n = (ortho / (no + 1e-6)) * nc * (no > (0.05 * nc)).float()
                        tree_77[1:n_tok+1].add_(ortho_n[1:n_tok+1], alpha=w_slot * 0.30)
                        accum.add_(ortho_n, alpha=0.15)
                        dag_parts.append(f"⊃ {child_w}")
                    else:
                        half_d = 768 // 2
                        nb = torch.norm(tree_77[1:n_tok+1, :half_d], dim=-1, keepdim=True) + 1e-7
                        np_n = torch.norm(t_child[1:n_tok+1, half_d:], dim=-1, keepdim=True) + 1e-7
                        tree_77[1:n_tok+1, half_d:].mul_(0.8).add_((t_child[1:n_tok+1, half_d:] / np_n * nb), alpha=0.20)
                        dag_parts.append(f"∥ {child_w}")

            if is_tick:
                self.delta_cycle_buffer[self.buf_ptr, :n_sec].copy_(z_slots)
                oldest_ptr = (self.buf_ptr + 1) % self.buf_len
                drift_cos = torch.sum(
                    torch.nn.functional.normalize(z_slots, p=2, dim=-1) * 
                    torch.nn.functional.normalize(self.delta_cycle_buffer[oldest_ptr, :n_sec], p=2, dim=-1),
                    dim=-1
                ).cpu().numpy()
                self.buf_ptr = oldest_ptr

                for s in range(n_sec):
                    raw_stab = float(np.clip((drift_cos[s] * 0.5 + 0.5) * (0.2 + 0.8 * kuramoto_r), 0.0, 1.0))
                    self.stabs[s] = self.stabs[s] * 0.65 + raw_stab * 0.35

                self.seals.step(n_sec, self.stabs, c_flow_np)

            return {
                "names": names, "indices": indices, "c_flow_np": c_flow_np,
                "tree_77": tree_77, "dag_parts": dag_parts, "root_s": root_s,
                "stabs": self.stabs, "sealed_mask": list(self.seals.mask),
                "z_slots": z_slots, "kuramoto_r": kuramoto_r,
                "x_local": x_local, "y_global": y_global,
                "temporal_asymmetry": temporal_asymmetry,
                "kinematics": node.kinematics
            }

class HeterarchyDeltaProcessor:
    def __init__(self, semantic, router, strength_low=0.55, burst_strength=0.75, burst_duration=0.0,
                 seal_cycles=4, decay_cycles=2, seal_thresh=0.68, refresh_cycles=2):
        self.lock = threading.RLock()
        self.semantic = semantic
        self.router = router
        self.strength_low, self.burst_strength, self.burst_duration = strength_low, burst_strength, burst_duration
        self.engines = {
            r: UniversalColumnProcessor(r, semantic, seal_cycles=seal_cycles, decay_cycles=decay_cycles, 
                                        seal_thresh=seal_thresh, refresh_cycles=refresh_cycles) 
            for r in self.router.ALL_ROLES
        }
        self.fatigue = torch.zeros(max(1, len(semantic.words)), dtype=torch.float32, device=DEVICE)
        self.active_c_idx = 0
        self.measured_ratio = 4.0
        self.delta_armed = False
        self.last_root = self.semantic.words[0].upper() if self.semantic.words else "СВОБОДНЫЙ_ДРЕЙФ"
        self.burst_timer = 0.0

    def step(self, frame, dt: float) -> tuple[torch.Tensor, float, dict]:
        with self.lock:
            w_count = len(self.semantic.words)

            topo = self.router.resolve(frame.nodes)
            th_hz, dl_hz = max(3.5, frame.theta_freq), max(0.5, frame.delta_freq)
            self.measured_ratio = self.measured_ratio * 0.95 + (th_hz / dl_hz) * 0.05
            k_th = int(np.clip(round(self.measured_ratio), 2, 8))
            n_sec = k_th * 2
            sec_w = (2.0 * math.pi) / float(n_sec)
            dl_phase = frame.delta_phase
            cur_sec = int(np.clip(int(((dl_phase + math.pi) % (2.0 * math.pi)) / sec_w), 0, n_sec - 1))

            is_tick = False
            if dl_phase < -1.5: self.delta_armed = True
            elif dl_phase > 0.5 and self.delta_armed: is_tick = True; self.delta_armed = False

            res = {r: self.engines[r].process(topo.get(r), n_sec, cur_sec, dl_phase, is_tick) for r in self.router.ALL_ROLES}
            hierarchy = ["Fpz", "AFz", "F3", "F4", "FCz"]
            apex = next((r for r in hierarchy if res[r] is not None), "НЕТ ДАТЧИКА")

            primary = res.get(apex)
            if primary is None:
                clean_emb = self.semantic.empty_embed if w_count == 0 else self.semantic.get_77_embedding_gpu(self.semantic.words[0])
                return clean_emb, self.strength_low, {"primary_role": "НЕТ ДАТЧИКА", "primary_active": False}

            target_77 = primary["tree_77"].clone()
            root_w = primary["dag_parts"][0]
            clean_emb = self.semantic.empty_embed if w_count == 0 else self.semantic.get_77_embedding_gpu(root_w)
            clean_norms = torch.norm(clean_emb, p=2, dim=-1, keepdim=True)

            if res["F3"] and apex != "F3" and w_count > 1:
                x_l, y_g = res["F3"]["x_local"], res["F3"]["y_global"]
                r_sync = res["F3"]["kuramoto_r"]
                freeze_val = float(np.clip((r_sync - 0.3) * 2.0, 0.0, 0.85))
                for s in range(min(n_sec, 8)):
                    t_f3 = self.semantic.get_77_embedding_gpu(res["F3"]["names"][s])
                    rank_scale = float(np.clip(1.0 + (x_l * (s % 3) + y_g * (s // 3)) * 0.25, 0.2, 2.0))
                    if s + 1 < 76: target_77[s + 1].add_(t_f3[s + 1], alpha=0.35 * rank_scale * (1.0 - freeze_val * 0.5))

            if res["F4"]:
                phase_entropy = float(1.0 - res["F4"]["kuramoto_r"])
                c_mod = torch.tanh(res["F4"]["z_slots"].mean(dim=0) / 8.0).unsqueeze(0)
                target_77.mul_(1.0 + 0.30 * c_mod * phase_entropy)
                dyn_str = float(np.clip(self.strength_low + (self.burst_strength - self.strength_low) * phase_entropy, self.strength_low, 0.95))
            else:
                dyn_str = self.strength_low

            if res["FCz"] and w_count > 0:
                kin = res["FCz"]["kinematics"]
                a120 = torch.from_numpy(topo["FCz"].iplv_human_ripple[-1]).to(DEVICE, dtype=torch.float32).view(1, 120)
                z_act = torch.matmul(a120, PHYSICAL_PROJECTION_768x120.T).squeeze(0)
                if torch.norm(z_act) > 0.01:
                    t_bases = self.semantic.get_concept_affordances_svd(primary["indices"][primary["root_s"]], max_bases=16)
                    proj = torch.matmul(torch.matmul(z_act, t_bases.T), t_bases)
                    speed_gain = float(np.clip(1.0 + kin.ry * 0.5, 0.4, 2.0))
                    target_77[1:13].add_(proj.unsqueeze(0).expand(12, 768), alpha=0.30 * speed_gain)

            fpz_tension = 0.0
            if res["Fpz"] and w_count > 1:
                if self.fatigue.shape[0] != w_count:
                    self.fatigue = torch.zeros(w_count, dtype=torch.float32, device=DEVICE)
                self.fatigue.mul_(max(0.0, 1.0 - 0.08 * dt))
                self.active_c_idx = self.active_c_idx % w_count
                self.fatigue[self.active_c_idx] += 0.25 * dt
                phase_drift = float(np.clip(abs(res["Fpz"]["temporal_asymmetry"]), 0.0, 1.0))
                cur_fatigue = self.fatigue[self.active_c_idx].item()
                fpz_tension = float(np.clip(phase_drift * 0.6 + (cur_fatigue / 2.0) * 0.4, 0.0, 1.0))
                
                sims = torch.mm(target_77.mean(dim=0, keepdim=True), self.semantic.norm_vocab.T)
                eff_energy = -sims.squeeze(0)[:w_count] + self.fatigue * (1.5 + 4.0 * phase_drift)
                if phase_drift >= 0.45: eff_energy[self.active_c_idx] += 10.0
                self.active_c_idx = int(torch.argmin(eff_energy)) % w_count
                if phase_drift >= 0.45:
                    hw = self.semantic.words[self.active_c_idx].upper()
                    target_77.copy_(self.semantic.get_77_embedding_gpu(hw))

            target_77 = torch.nn.functional.normalize(target_77, p=2, dim=-1) * clean_norms

            if root_w != self.last_root:
                self.last_root = root_w
                self.burst_timer = time.time()
                print(f"🌲 [{apex} [ФАКТ]]: {' '.join(primary['dag_parts'][:6])}")

            if (time.time() - self.burst_timer) < self.burst_duration: dyn_str = self.burst_strength

            telemetry = {
                "primary_role": apex, "primary_active": True,
                "total_clock_sectors": n_sec, "current_clock_sec": cur_sec,
                "slot_names": primary["names"], "slot_indices": primary["indices"],
                "sealed_mask": primary["sealed_mask"], "slot_stabilities": primary["stabs"],
                "causal_flow_np": primary["c_flow_np"], "dag_log_parts": primary["dag_parts"],
                "dl_phase": dl_phase, "is_delta_tick": is_tick, "dyn_strength": dyn_str,
                "active_vocab_size": len(self.semantic.words),
                "fpz_active": res["Fpz"] is not None, "fpz_tension": fpz_tension,
                "fpz_concept": res["Fpz"]["dag_parts"][0] if res["Fpz"] else "",
                "f3_active": res["F3"] is not None,
                "f3_concept": res["F3"]["dag_parts"][0] if res["F3"] else "",
                "f3_x_local": res["F3"]["x_local"] if res["F3"] else 0.0,
                "f3_y_global": res["F3"]["y_global"] if res["F3"] else 0.0,
                "f3_kuramoto": res["F3"]["kuramoto_r"] if res["F3"] else 0.0,
                "f3_slot_names": res["F3"]["names"] if res["F3"] else [],
                "f4_active": res["F4"] is not None,
                "f4_concept": res["F4"]["dag_parts"][0] if res["F4"] else "",
                "f4_entropy": float(1.0 - res["F4"]["kuramoto_r"]) if res["F4"] else 0.0,
                "fcz_active": res["FCz"] is not None,
                "fcz_concept": res["FCz"]["dag_parts"][0] if res["FCz"] else "",
                "fcz_lx": float(res["FCz"]["kinematics"].lx) if res["FCz"] else 0.0,
                "fcz_ly": float(res["FCz"]["kinematics"].ly) if res["FCz"] else 0.0,
                "fcz_rx": float(res["FCz"]["kinematics"].rx) if res["FCz"] else 0.0,
                "fcz_ry": float(res["FCz"]["kinematics"].ry) if res["FCz"] else 0.0
            }
            return target_77, dyn_str, telemetry

