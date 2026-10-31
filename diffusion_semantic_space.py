#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: DEDICATED CLIP SEMANTIC SPACE FOR DIFFUSION (NO MONOLITHS)
===================================================================================
"""

from __future__ import annotations
import os
import json
import threading
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from neuro_heterarchy_core import DEVICE

CORE_REGIONS = ("FPZ", "AFZ", "F3", "F4", "FCZ")


class UniversalSemanticSpace:
    def __init__(self, lore_file=None, concepts_cli=None, target_k=None, model_id="openai/clip-vit-large-patch14"):
        self.lock = threading.RLock()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE).eval()
        self.gpu_embed_cache = {}
        self.affordance_cache = {}

        self.empty_embed = self._compute_77_embedding("")
        self.last_active_embed = self.empty_embed.clone()
        self.sections: dict[str, list[str]] = {r: [] for r in CORE_REGIONS}
        self.region_banks: dict[str, torch.Tensor] = {}

        custom_words = []
        if lore_file and os.path.exists(lore_file):
            if lore_file.endswith('.json'):
                with open(lore_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    custom_words = [str(x).strip() for x in (d if isinstance(d, list) else d.keys()) if str(x).strip()]
                for r in CORE_REGIONS:
                    self.sections[r] = list(custom_words)
            else:
                cur_sec = None
                unassigned = []
                with open(lore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith('#'):
                            continue
                        if s.startswith('[') and s.endswith(']'):
                            sec_name = s[1:-1].strip().upper()
                            cur_sec = sec_name if sec_name in CORE_REGIONS else "IGNORE"
                        else:
                            if cur_sec is None:
                                unassigned.append(s)
                                custom_words.append(s)
                            elif cur_sec in CORE_REGIONS:
                                self.sections[cur_sec].append(s)
                                if s not in custom_words:
                                    custom_words.append(s)
                if unassigned:
                    for r in CORE_REGIONS:
                        if not self.sections[r]:
                            self.sections[r] = list(unassigned)
                non_empty = next((self.sections[r] for r in CORE_REGIONS if self.sections[r]), [])
                if non_empty:
                    for r in CORE_REGIONS:
                        if not self.sections[r]:
                            self.sections[r] = list(non_empty)
        elif concepts_cli and concepts_cli.strip().lower() not in ["none", ""]:
            custom_words = [w.strip() for w in concepts_cli.split(",") if w.strip()]
            for r in CORE_REGIONS:
                self.sections[r] = list(custom_words)

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
                raw = self.last_active_embed.mean(dim=0, keepdim=True)
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
                for r in CORE_REGIONS:
                    self.region_banks[r] = self.last_active_embed.unsqueeze(0)
                self.affordance_cache.clear()
                return

            lore_embeds = [self.get_77_embedding_gpu(w).mean(dim=0) for w in self.words]
            raw = torch.stack(lore_embeds, dim=0)

            if n_words == 1:
                self.norm_vocab = F.normalize(raw, p=2, dim=-1)
            else:
                mu = torch.mean(raw, dim=0, keepdim=True)
                std = torch.std(raw - mu, dim=0, keepdim=True, unbiased=False) + 1e-6
                self.norm_vocab = F.normalize((raw - mu) / std, p=2, dim=-1)

            for r in CORE_REGIONS:
                r_words = self.sections.get(r) or self.words
                if r_words:
                    self.region_banks[r] = torch.stack([self.get_77_embedding_gpu(w) for w in r_words], dim=0)
                else:
                    self.region_banks[r] = self.last_active_embed.unsqueeze(0)

            self.affordance_cache.clear()

    def update_concepts_on_the_fly(self, new_words: list[str]):
        cleaned = [w.strip() for w in new_words if w.strip()]
        with self.lock:
            if not cleaned and self.words:
                self.last_active_embed = self.get_77_embedding_gpu(self.words[0]).clone()
            self.words = list(cleaned)
            for r in CORE_REGIONS:
                self.sections[r] = list(cleaned)
            self._rebuild_vocab_matrix_locked()

    def get_77_embedding_gpu(self, word: str) -> torch.Tensor:
        with self.lock:
            if not word or not word.strip():
                return self.last_active_embed
            w_clean = word.strip().lower()
            if w_clean in self.gpu_embed_cache:
                return self.gpu_embed_cache[w_clean]
            emb = self._compute_77_embedding(w_clean)
            self.gpu_embed_cache[w_clean] = emb
            return emb
