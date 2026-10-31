#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: FULL-DIMENSIONAL [64 x 768] MONTY SDR HETERARCHY FOR MUSICGEN
===================================================================================
- Полный ранг 64 собственных ортонормированных мод SVD (max_bases=64 вместо 16),
  что полностью исключает потерю размерности при проекции рабочей памяти в T5!
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import threading
import numpy as np
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE


def _scale_to_t5(delta: torch.Tensor, ref_norms: torch.Tensor) -> torch.Tensor:
    d_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
    gate = torch.tanh(d_norm / 2.5)
    return F.normalize(delta, p=2, dim=-1) * ref_norms * gate


def _project_into_svd_manifold(wave_Nx768: torch.Tensor, v_aff_Mx768: torch.Tensor, ref_norms: torch.Tensor) -> torch.Tensor:
    coords = torch.matmul(wave_Nx768, v_aff_Mx768.T)
    projected = torch.matmul(coords, v_aff_Mx768)
    return _scale_to_t5(projected, ref_norms)


class UniversalMusicSemanticSpace:
    def __init__(self, words: list[str], c_bases: list[np.ndarray],
                 proj_120_to_768: np.ndarray,
                 empty_embed: Optional[np.ndarray] = None,
                 global_svd_bases: Optional[np.ndarray] = None):
        self.lock = threading.RLock()
        self.words = [w.strip() for w in words if w.strip()]
        self.c_bases = [torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in c_bases]
        self.proj_120_to_768 = F.normalize(
            torch.tensor(proj_120_to_768, dtype=torch.float32, device=DEVICE), p=2, dim=0
        )

        if empty_embed is not None:
            self.empty_embed = torch.tensor(empty_embed, dtype=torch.float32, device=DEVICE)
        elif self.c_bases:
            self.empty_embed = self.c_bases[0].clone()
        else:
            self.empty_embed = torch.zeros((64, 768), dtype=torch.float32, device=DEVICE)

        if global_svd_bases is not None:
            self.global_svd_bases = F.normalize(
                torch.tensor(global_svd_bases, dtype=torch.float32, device=DEVICE), p=2, dim=-1
            )
        else:
            self.global_svd_bases = F.normalize(self.proj_120_to_768[:64], p=2, dim=-1)

        self.last_active_embed = (self.c_bases[0] if self.c_bases else self.empty_embed).clone()
        self.affordance_cache = {}
        self.rebuild_norm_vocab()

    def rebuild_norm_vocab(self):
        with self.lock:
            n_b = len(self.c_bases)
            if n_b == 0:
                raw = self.last_active_embed.mean(dim=0, keepdim=True)
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
                self.affordance_cache.clear()
                return

            raw = torch.stack([b.mean(dim=0) for b in self.c_bases], dim=0)
            if n_b == 1:
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
            else:
                mu = torch.mean(raw, dim=0, keepdim=True)
                std = torch.std(raw - mu, dim=0, keepdim=True, unbiased=False) + 1e-6
                self.norm_vocab = F.normalize((raw - mu) / std, p=2, dim=-1)
            self.affordance_cache.clear()

    def get_concept_affordances_svd(self, concept_idx: int, k_neighbors: int = 16, max_bases: int = 64) -> torch.Tensor:
        with self.lock:
            n_items = len(self.c_bases)
            if n_items == 0:
                return self.global_svd_bases[:max_bases]

            safe_idx = concept_idx % n_items
            cache_key = (safe_idx, max_bases)
            if cache_key in self.affordance_cache:
                return self.affordance_cache[cache_key]

            base_64 = self.c_bases[safe_idx]
            tokens_centered = base_64 - base_64.mean(dim=0, keepdim=True)

            if n_items > 1:
                target = self.norm_vocab[safe_idx:safe_idx + 1]
                sims = torch.mm(target, self.norm_vocab.T).squeeze(0)
                _, topk = torch.topk(sims, min(k_neighbors, n_items))
                neighbors_centered = (self.norm_vocab[topk] - target) * 2.0
                combined_manifold = torch.cat([F.normalize(tokens_centered, p=2, dim=-1), neighbors_centered], dim=0)
            else:
                combined_manifold = F.normalize(tokens_centered, p=2, dim=-1)

            _, _, V = torch.linalg.svd(combined_manifold, full_matrices=False)
            tangent = F.normalize(V[:max_bases].clone(), p=2, dim=-1)
            self.affordance_cache[cache_key] = tangent
            return tangent

    def decode_slots(self, z_slots_768: torch.Tensor):
        with self.lock:
            n_b = len(self.c_bases)
            w_len = min(len(self.words), n_b)
            if w_len == 0:
                sims = torch.zeros((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, torch.ones_like(sims), ["СВОБОДНЫЙ_ДРЕЙФ"] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]
            if w_len == 1:
                sims = torch.ones((z_slots_768.shape[0], 1), device=DEVICE)
                return sims, sims, [self.words[0].upper()] * z_slots_768.shape[0], [0] * z_slots_768.shape[0]

            z_norm = F.normalize(z_slots_768, p=2, dim=-1)
            sims = torch.mm(z_norm, self.norm_vocab[:w_len].T)
            top = torch.argmax(sims, dim=-1)
            names = [self.words[int(i) % w_len].upper() for i in top]
            indices = [int(i) % w_len for i in top]
            return sims, torch.softmax(sims / 0.04, dim=-1), names, indices
