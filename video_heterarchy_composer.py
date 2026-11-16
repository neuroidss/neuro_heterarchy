#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: PURE TENSOR 5-REGION COMPOSER, SEMANTIC ENTITY ATTRACTOR & WORLD GRAPH
===================================================================================
- Zero hardcoded words, zero centers of mass, zero dimensionality reduction.
- Single-Device Universal Invariant: 1 device on ANY region (FCz, F3, F4, AFz, Fpz)
  drives the entire entity's 19,200-D working memory field across all 5 regions,
  or symbiotically combines multiple devices in the shared heterarchy.
- SemanticEntityAttractorGPU: just like drums emerging in MusicGen from 'psytrance'
  and being held by user attention, any semantic form/action discovered in the T5
  SVD affordance subspace is locked by Working Memory persistence + Kuramoto phase
  coherence and tested for compatibility against the World Keeper (Maze / Opponent).
===================================================================================
"""

from __future__ import annotations
import os
import zlib
import math
import torch
import torch.nn.functional as F

from neuro_heterarchy_core import DEVICE
from neuro_amodal_engine import REGION_ROLES
from neuro_vram_bus import HardwareVRAMBus

REGION_TOKEN_SLICES = {
    "Fpz": (0, 25),
    "AFz": (25, 50),
    "F3":  (50, 75),
    "F4":  (75, 100),
    "FCz": (100, 128),
}
CORE_REGIONS = ("FPZ", "AFZ", "F3", "F4", "FCZ")


def normalize_lore_path(p: str | None) -> str:
    if not p or not str(p).strip():
        return ""
    s = str(p).strip()
    if s.startswith("/app/"):
        s = s[len("/app/"):]
    return s


class SemanticEntityAttractorGPU:
    """
    Holds the immortal Working Memory entity state in T5 embedding space.
    When the user's phase coherence (persistence & Kuramoto synchronisation) locks onto
    a semantic nuance in the SVD tangent subspace (e.g., locomotion, elemental shield,
    or strike), this attractor sustains that semantic manifestation across frames and
    evaluates its compatibility with the World Keeper (Maze walls or Arena opponent).
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.locked_tangents: dict[str, torch.Tensor] = {}
        self.attractor_strength: dict[str, float] = {r: 0.0 for r in REGION_ROLES}
        self.manifestation_score: float = 1.0

    @torch.no_grad()
    def reset(self):
        self.locked_tangents.clear()
        for r in REGION_ROLES:
            self.attractor_strength[r] = 0.0
        self.manifestation_score = 1.0

    @torch.no_grad()
    def step_attractor(
        self,
        role: str,
        anchor_tokens: torch.Tensor,
        candidate_tangent: torch.Tensor,
        persistence: float,
        kuramoto: float,
        surprise: float,
        world_keeper_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        n_tok, d_dim = anchor_tokens.shape
        prev_t = self.locked_tangents.get(role, None)

        attention_hold = float(min(1.0, max(0.0, persistence * 0.65 + kuramoto * 0.35)))
        if prev_t is None or prev_t.shape != (n_tok, d_dim):
            fused_t = candidate_tangent.clone()
            self.attractor_strength[role] = attention_hold
        else:
            # High persistence locks the discovered semantic entity/action;
            # high surprise releases the lock to adapt to new maze corridors or attacks.
            lock_inertia = max(0.15, min(0.88, attention_hold * (1.0 - 0.45 * min(1.0, surprise))))
            fused_t = prev_t * lock_inertia + candidate_tangent * (1.0 - lock_inertia)
            self.attractor_strength[role] = lock_inertia

        # Compatibility check with the World Keeper (Fpz/AFz maze law):
        # Projects out components that directly violate immovable world constraints
        compatibility = 1.0
        if world_keeper_vec is not None and role in ("FCz", "F3", "F4"):
            wk_unit = F.normalize(world_keeper_vec.mean(dim=0, keepdim=True), p=2, dim=-1)
            tok_unit = F.normalize(fused_t, p=2, dim=-1)
            conflict_cos = torch.sum(tok_unit * wk_unit, dim=-1, keepdim=True)
            # If the entity's action opposes the World Keeper's invariant law,
            # orthogonalize the tangent so the entity glides along the corridor/law!
            violation = torch.relu(-conflict_cos - 0.15)
            fused_t = fused_t + violation * wk_unit * torch.norm(fused_t, p=2, dim=-1, keepdim=True)
            compatibility = float(torch.clamp(1.0 - violation.mean(), 0.20, 1.0).item())

        self.locked_tangents[role] = fused_t.detach().clone()
        return fused_t, compatibility


class EpisodicCausalWorldGraphGPU:
    """
    GPU-resident causal graph of world consequences (256 episodic nodes).
    Routes between Fast Gamma (real-time semantic encoding) and Slow Gamma
    (prospective retrieval of learned multi-step sequences).
    """

    def __init__(self, max_nodes: int = 256, device=DEVICE):
        self.max_nodes = max_nodes
        self.device = device
        self.num_nodes = 0
        self.write_ptr = 0
        self.current_node_idx = -1

        self.wm_keys = torch.zeros((max_nodes, 512), dtype=torch.float32, device=device)
        self.consequence_t5 = torch.zeros((max_nodes, 128, 4096), dtype=torch.float16, device=device)
        self.causal_edges = torch.zeros((max_nodes, max_nodes), dtype=torch.float32, device=device)
        self.latent_snapshots: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def reset(self):
        self.num_nodes = 0
        self.write_ptr = 0
        self.current_node_idx = -1
        self.wm_keys.zero_()
        self.consequence_t5.zero_()
        self.causal_edges.zero_()
        self.latent_snapshots.clear()

    @torch.no_grad()
    def _compress_wm_key(self, delta_wm_128x4096: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(delta_wm_128x4096.mean(dim=0, keepdim=True).unsqueeze(0), 512).squeeze()
        return F.normalize(pooled + 1e-6, p=2, dim=0)

    @torch.no_grad()
    def query_and_update(
        self,
        delta_wm_128x4096: torch.Tensor,
        current_t5_128x4096: torch.Tensor,
        valid_mask_128: torch.Tensor,
        latest_latent_5d: torch.Tensor | None,
        conflict_tension: float,
        persistence: float,
        wm_radius: float,
        stochastic: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        wm_influence = (1.0 - stochastic) * wm_radius
        if wm_influence <= 0.01:
            return current_t5_128x4096, None, self.num_nodes, -1

        q_key = self._compress_wm_key(delta_wm_128x4096)
        recalled_latent = None
        recalled_node = -1

        if self.num_nodes > 0:
            active_keys = self.wm_keys[:self.num_nodes]
            sims = torch.mv(active_keys, q_key)

            if 0 <= self.current_node_idx < self.num_nodes:
                causal_boost = self.causal_edges[self.current_node_idx, :self.num_nodes] * 0.25
                sims = sims + causal_boost

            best_sim, best_idx_t = torch.max(sims, dim=0)
            best_idx = int(best_idx_t.item())

            if float(best_sim.item()) > 0.55:
                recalled_node = best_idx
                mem_t5 = self.consequence_t5[best_idx].float()
                blend_w = min(0.18, float(best_sim.item()) * 0.15 * wm_influence)
                m_mask = valid_mask_128.unsqueeze(-1)
                blended = current_t5_128x4096 * (1.0 - blend_w) + mem_t5 * blend_w
                current_t5_128x4096 = torch.where(m_mask, blended, current_t5_128x4096)
                recalled_latent = self.latent_snapshots.get(best_idx, None)

        commit_drive = persistence * 0.60 + conflict_tension * 0.40
        is_new_consequence = (self.num_nodes == 0)
        if self.num_nodes > 0:
            max_sim = float(torch.max(torch.mv(self.wm_keys[:self.num_nodes], q_key)).item())
            is_new_consequence = (max_sim < 0.72) and (commit_drive > 0.40)

        if is_new_consequence:
            idx = self.write_ptr
            self.wm_keys[idx].copy_(q_key)
            self.consequence_t5[idx].copy_(current_t5_128x4096.to(torch.float16))
            if latest_latent_5d is not None:
                self.latent_snapshots[idx] = latest_latent_5d.detach().clone()

            if self.current_node_idx >= 0:
                self.causal_edges[self.current_node_idx, idx] = min(
                    1.0, float(self.causal_edges[self.current_node_idx, idx].item()) + 0.50
                )

            self.current_node_idx = idx
            self.num_nodes = min(self.max_nodes, self.num_nodes + 1)
            self.write_ptr = (self.write_ptr + 1) % self.max_nodes

        return current_t5_128x4096, recalled_latent, self.num_nodes, recalled_node


class RegionalVideoHeterarchyComposer:
    def __init__(self, renderer, lore_path: str = ""):
        self.renderer = renderer
        self.device = DEVICE
        self.lore_path = normalize_lore_path(lore_path)

        self.current_lore_hash = None
        self.sections: dict[str, list[str]] = {
            "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
        }

        self.region_full_banks: dict[str, torch.Tensor] = {}
        self.region_masks_banks: dict[str, torch.Tensor] = {}
        self.region_token_counts: dict[str, list[int]] = {}
        self.region_labels: dict[str, list[str]] = {}
        self.region_svd_bases: dict[str, torch.Tensor] = {}

        self.empty_pad_embed_128x4096: torch.Tensor | None = None
        self.eos_token_embed_1x4096: torch.Tensor | None = None
        self.neg_embed_1x128x4096: torch.Tensor | None = None
        self.neg_mask_1x128: torch.Tensor | None = None

        self.running_wm_baseline_128x4096 = torch.zeros((128, 4096), dtype=torch.float32, device=self.device)
        self.baseline_initialized = False

        self.entity_attractor = SemanticEntityAttractorGPU(device=self.device)
        self.world_graph = EpisodicCausalWorldGraphGPU(max_nodes=256, device=self.device)

        self.active_concept_idx = {r: 0 for r in REGION_ROLES}
        self.fpz_meta_charge = 0.0
        self.afz_stage_progress = 0.0

        self.cascade_rate = 1.0
        self.psi_threshold = 0.55
        self.wm_radius = 0.25

        initial_sec = self._read_lore_file_sections()
        self.sync_lore_sections(initial_sec)

    def _read_lore_file_sections(self) -> dict[str, list[str]]:
        secs: dict[str, list[str]] = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
        candidates = [self.lore_path, f"/app/{self.lore_path}"] if self.lore_path else []
        unassigned_global = []

        for c_path in candidates:
            if c_path and os.path.isfile(c_path):
                cur = None
                with open(c_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        if s.startswith("[") and s.endswith("]"):
                            cur = s[1:-1].strip().upper()
                            if cur not in secs:
                                secs[cur] = []
                        else:
                            if cur is None:
                                unassigned_global.append(s)
                            else:
                                secs[cur].append(s)
                break

        if unassigned_global:
            for r_k in CORE_REGIONS:
                if not secs[r_k]:
                    secs[r_k] = list(unassigned_global)

        non_empty = next((secs[r] for r in CORE_REGIONS if secs[r]), [])
        if non_empty:
            for r_k in CORE_REGIONS:
                if not secs[r_k]:
                    secs[r_k] = list(non_empty)

        return secs

    @torch.no_grad()
    def sync_lore_sections(self, lore_sections: dict[str, list[str]] | None):
        if lore_sections is None or not any(lore_sections.get(r) for r in CORE_REGIONS):
            file_secs = self._read_lore_file_sections()
            if any(file_secs.get(r) for r in CORE_REGIONS):
                lore_sections = file_secs
            elif lore_sections is None:
                lore_sections = file_secs

        non_empty = next((lore_sections.get(r, []) for r in CORE_REGIONS if lore_sections.get(r)), [])
        if non_empty:
            lore_sections = dict(lore_sections)
            for r_k in CORE_REGIONS:
                if not lore_sections.get(r_k):
                    lore_sections[r_k] = list(non_empty)

        serialized = repr(sorted((k, v) for k, v in lore_sections.items()))
        new_hash = zlib.crc32(serialized.encode("utf-8"))
        if self.current_lore_hash == new_hash and self.neg_embed_1x128x4096 is not None:
            return

        empty_emb, _ = self.renderer.encode_prompt_with_mask("")
        self.empty_pad_embed_128x4096 = empty_emb.clone()
        self.eos_token_embed_1x4096 = empty_emb[0:1, :].clone()

        for role in REGION_ROLES:
            sec_k = role.upper()
            lines = [s.strip() for s in lore_sections.get(sec_k, []) if s.strip()]
            self.sections[sec_k] = lines

            prompts_to_encode = lines if lines else [""]
            self.region_labels[role] = lines if lines else [""]
            encoded_pairs = [self.renderer.encode_prompt_with_mask(p) for p in prompts_to_encode]
            bank = torch.stack([pair[0] for pair in encoded_pairs], dim=0)
            self.region_full_banks[role] = bank
            masks_t = torch.stack([pair[1].squeeze(0) for pair in encoded_pairs], dim=0)
            self.region_masks_banks[role] = masks_t

            counts = []
            for m in masks_t:
                c = int(m.sum().item())
                counts.append(max(1, c - 1) if c > 1 else 1)
            self.region_token_counts[role] = counts

            # Build orthonormal SVD/QR affordance basis around the region's concepts
            # (Just like MusicGen's get_concept_affordances_svd, enabling emergent semantic control!)
            flat_tokens = bank[:, : max(counts), :].reshape(-1, 4096)
            centered = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
            q_mat, _ = torch.linalg.qr(centered.T, mode="reduced")
            self.region_svd_bases[role] = q_mat.T.contiguous()

        neg_lines = [s.strip() for s in lore_sections.get("NEGATIVE", []) if s.strip()]
        neg_joined = ", ".join(neg_lines) if neg_lines else ""
        neg_emb, neg_mask = self.renderer.encode_prompt_with_mask(neg_joined)

        self.neg_embed_1x128x4096 = neg_emb.unsqueeze(0)
        self.neg_mask_1x128 = neg_mask
        self.current_lore_hash = new_hash
        self.entity_attractor.reset()

    @torch.no_grad()
    def _select_and_modulate_region_concept(
        self,
        role: str,
        bank_Kx128x4096: torch.Tensor,
        token_counts: list[int],
        r_wm_span: torch.Tensor,
        eff_radius: float,
        r_stoch: float,
        persistence: float,
        kuramoto: float,
        surprise: float,
        world_keeper_vec: torch.Tensor | None = None,
        max_tokens_budget: int = 25,
    ) -> tuple[torch.Tensor, int, int, float]:
        K, L, D = bank_Kx128x4096.shape
        wm_influence = float(np.clip((1.0 - r_stoch) * eff_radius, 0.0, 1.0))

        wm_flat = r_wm_span.reshape(1, -1)
        wm_centered = wm_flat - wm_flat.mean(dim=-1, keepdim=True)
        wm_norm = torch.norm(wm_centered, p=2, dim=-1, keepdim=True)

        if K > 1 and float(wm_norm.item()) > 1e-5:
            wm_unit = wm_centered / (wm_norm + 1e-6)
            span_len = r_wm_span.shape[0]
            bank_span_flat = bank_Kx128x4096[:, :span_len, :].reshape(K, -1)
            delta_bank = bank_span_flat - bank_span_flat.mean(dim=0, keepdim=True)
            delta_unit = delta_bank / (torch.norm(delta_bank, p=2, dim=-1, keepdim=True) + 1e-6)
            sims_K = torch.mm(wm_unit, delta_unit.T).squeeze(0)
            sims_std = torch.std(sims_K, unbiased=False).clamp(min=1e-5)
            sims_scaled = (sims_K - sims_K.mean()) / sims_std
            probs_K = torch.softmax(sims_scaled / 0.15, dim=0)
            best_i = int(torch.argmax(probs_K).item())
            conf_pct = int(round(float(probs_K[best_i].item()) * 100.0))
        else:
            best_i = 0
            conf_pct = 100

        n_tok = min(max_tokens_budget, token_counts[best_i])
        anchor_tokens = bank_Kx128x4096[best_i, :n_tok, :].clone()

        if wm_influence <= 1e-4 or float(wm_norm.item()) <= 1e-5:
            return anchor_tokens, best_i, conf_pct, 1.0

        ref_norms = torch.norm(anchor_tokens, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        wm_slice = (
            r_wm_span[:n_tok, :]
            if r_wm_span.shape[0] >= n_tok
            else r_wm_span.repeat((n_tok // r_wm_span.shape[0]) + 1, 1)[:n_tok, :]
        )
        wm_zero = wm_slice - wm_slice.mean(dim=0, keepdim=True)

        # Project Working Memory wave onto the SVD/QR semantic affordance basis of this region
        q_basis = self.region_svd_bases.get(role, None)
        if q_basis is not None and q_basis.shape[0] > 0:
            wm_in_span = torch.matmul(torch.matmul(wm_zero, q_basis.T), q_basis)
            wm_combined = wm_in_span * 0.70 + wm_zero * 0.30
        else:
            wm_combined = wm_zero

        dot_proj = torch.sum(wm_combined * anchor_tokens, dim=-1, keepdim=True) / (ref_norms ** 2 + 1e-6)
        wm_tangent = wm_combined - dot_proj * anchor_tokens

        # Lock and filter the semantic tangent through the Entity Attractor & World Keeper compatibility
        locked_tangent, compatibility = self.entity_attractor.step_attractor(
            role=role,
            anchor_tokens=anchor_tokens,
            candidate_tangent=wm_tangent,
            persistence=persistence,
            kuramoto=kuramoto,
            surprise=surprise,
            world_keeper_vec=world_keeper_vec,
        )
        wm_tangent_unit = (locked_tangent / (torch.norm(locked_tangent, p=2, dim=-1, keepdim=True) + 1e-6)) * ref_norms

        sig_gate = float(torch.tanh(wm_norm * 10.0).item())
        theta_rad = wm_influence * (math.pi * 0.12) * sig_gate * compatibility

        rotated_tokens = math.cos(theta_rad) * anchor_tokens + math.sin(theta_rad) * wm_tangent_unit
        rotated_tokens = F.normalize(rotated_tokens, p=2, dim=-1) * ref_norms
        return rotated_tokens, best_i, conf_pct, compatibility

    @torch.no_grad()
    def compose_step(
        self,
        in_wm_128x4096: torch.Tensor,
        active_region_mask: dict[str, bool],
        region_gains: dict[str, float],
        persistence: float,
        surprise: float,
        kuramoto: float,
        stochastic: float = 0.0,
        unassigned_stochastic: float = 1.0,
        wm_radius: float = 0.25,
        video_params: dict | None = None,
        vram_bus: HardwareVRAMBus | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, str, float]:
        if video_params and isinstance(video_params, dict):
            if "geodesic_lock" in video_params:
                wm_radius = float(min(0.80, max(0.02, float(video_params["geodesic_lock"]))))
            if "cascade_rate" in video_params:
                self.cascade_rate = float(min(4.0, max(0.05, float(video_params["cascade_rate"]))))
            if "psi_threshold" in video_params:
                self.psi_threshold = float(min(0.95, max(0.15, float(video_params["psi_threshold"]))))

        stoch_act = float(np.clip(stochastic, 0.0, 1.0))
        stoch_unass = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        eff_radius = float(np.clip(wm_radius, 0.0, 1.0))
        self.wm_radius = eff_radius

        wm_128x4096 = in_wm_128x4096[0] if in_wm_128x4096.ndim == 3 else in_wm_128x4096
        wm_energy = float(torch.mean(torch.abs(wm_128x4096)).item())

        if wm_energy <= 1e-5:
            delta_wm_128x4096 = torch.zeros_like(wm_128x4096)
        elif not self.baseline_initialized:
            self.running_wm_baseline_128x4096.copy_(wm_128x4096)
            self.baseline_initialized = True
            delta_wm_128x4096 = wm_128x4096 - self.running_wm_baseline_128x4096 * 0.85
        else:
            self.running_wm_baseline_128x4096.lerp_(wm_128x4096, 0.04)
            delta_wm_128x4096 = wm_128x4096 - self.running_wm_baseline_128x4096 * 0.85

        ch_std = delta_wm_128x4096.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-5)
        delta_wm_whitened = (delta_wm_128x4096 - delta_wm_128x4096.mean(dim=0, keepdim=True)) / ch_std

        ordered_roles = ["Fpz", "AFz", "F3", "F4", "FCz"]
        active_roles = [r for r in ordered_roles if active_region_mask.get(r, False)]
        has_any_active = len(active_roles) > 0

        # SINGLE-DEVICE UNIVERSAL INVARIANT:
        # If the user has 1 device connected on ANY region (e.g., F3, AFz, or FCz),
        # its full-rank 19,200-D working memory slice propagates across the entire
        # entity heterarchy so 1 device can both walk through the maze and manifest actions!
        if len(active_roles) == 1:
            src_role = active_roles[0]
            s_src, e_src = REGION_TOKEN_SLICES[src_role]
            master_slice = delta_wm_whitened[s_src:e_src, :]
            for r in ordered_roles:
                if r != src_role:
                    s_dst, e_dst = REGION_TOKEN_SLICES[r]
                    dst_len = e_dst - s_dst
                    adapted = (
                        master_slice[:dst_len, :]
                        if master_slice.shape[0] >= dst_len
                        else master_slice.repeat(2, 1)[:dst_len, :]
                    )
                    delta_wm_whitened[s_dst:e_dst, :] = (
                        delta_wm_whitened[s_dst:e_dst, :] * 0.35 + adapted * 0.65
                    )

        fcz_vec = delta_wm_whitened[100:125, :].reshape(-1)
        afz_vec = delta_wm_whitened[25:50, :].reshape(-1)
        cos_sim = float(F.cosine_similarity(fcz_vec.unsqueeze(0), afz_vec.unsqueeze(0)).item())
        conflict_tension = float(np.clip((1.0 - cos_sim) * 0.5 * (0.45 + 0.55 * surprise), 0.0, 1.0))

        composed_128x4096 = self.empty_pad_embed_128x4096.clone()
        combined_pos_mask = torch.zeros((128,), dtype=torch.bool, device=self.device)
        cursor = 0
        debug_parts = []
        seen_identical_prompts = set()
        world_keeper_tokens: torch.Tensor | None = None
        compat_scores = []

        for role in ordered_roles:
            is_act = bool(active_region_mask.get(role, False))
            r_stoch = stoch_act if (is_act or has_any_active) else stoch_unass

            bank_Kx128x4096 = self.region_full_banks[role]
            token_counts = self.region_token_counts[role]
            labels = self.region_labels[role]
            K = bank_Kx128x4096.shape[0]
            s_i, e_i = REGION_TOKEN_SLICES[role]
            span = e_i - s_i

            r_wm_span = delta_wm_whitened[s_i:e_i, :]
            eff_reg_rad = eff_radius if (is_act or has_any_active) else (eff_radius * 0.50)

            budget = min(span, max(1, 127 - cursor))
            reg_tokens, best_i, conf_pct, compat = self._select_and_modulate_region_concept(
                role=role,
                bank_Kx128x4096=bank_Kx128x4096,
                token_counts=token_counts,
                r_wm_span=r_wm_span,
                eff_radius=eff_reg_rad,
                r_stoch=r_stoch,
                persistence=persistence,
                kuramoto=kuramoto,
                surprise=surprise,
                world_keeper_vec=world_keeper_tokens,
                max_tokens_budget=budget,
            )
            compat_scores.append(compat)
            if role in ("Fpz", "AFz") and world_keeper_tokens is None:
                world_keeper_tokens = reg_tokens

            self.active_concept_idx[role] = best_i
            chosen_label = labels[best_i].strip()

            if K == 1 and chosen_label in seen_identical_prompts and role != "FCz":
                continue
            if chosen_label:
                seen_identical_prompts.add(chosen_label)

            n_t = reg_tokens.shape[0]
            if cursor + n_t <= 127:
                composed_128x4096[cursor:cursor + n_t, :] = reg_tokens
                combined_pos_mask[cursor:cursor + n_t] = True
                cursor += n_t

            if is_act or role == "FCz" or K > 1:
                debug_parts.append(f"{'★' if is_act else '•'}{role}[#{best_i+1}/{K}({conf_pct}%): {chosen_label[:20]}]")

        if cursor < 128 and self.eos_token_embed_1x4096 is not None:
            composed_128x4096[cursor:cursor + 1, :] = self.eos_token_embed_1x4096
            combined_pos_mask[cursor] = True
        elif cursor == 0:
            combined_pos_mask[0] = True

        latest_lat = getattr(self.renderer, "latest_end_latent_5d", None)
        composed_128x4096, recalled_lat, n_mem_nodes, recalled_id = self.world_graph.query_and_update(
            delta_wm_128x4096=delta_wm_whitened,
            current_t5_128x4096=composed_128x4096,
            valid_mask_128=combined_pos_mask,
            latest_latent_5d=latest_lat,
            conflict_tension=conflict_tension,
            persistence=persistence,
            wm_radius=eff_radius,
            stochastic=stoch_act,
        )
        self.renderer.recalled_episodic_latent_5d = recalled_lat

        self.renderer.last_pos_mask = combined_pos_mask.unsqueeze(0)
        self.renderer.last_neg_mask = self.neg_mask_1x128

        mean_compat = sum(compat_scores) / max(1, len(compat_scores))
        self.entity_attractor.manifestation_score = mean_compat

        wm_gate = float(min(1.0, wm_energy * 20.0)) if has_any_active else 0.0
        self.afz_stage_progress = float(min(1.0, self.afz_stage_progress * 0.92 + 0.08 * persistence * mean_compat * wm_gate))
        self.fpz_meta_charge = float(min(1.0, self.fpz_meta_charge * 0.85 + (0.15 * persistence if surprise > 0.35 else 0.0) * wm_gate))

        fcz_gain = float(region_gains.get("FCz", 0.92))
        advance_rate = float(min(0.30, max(0.0, (0.12 + 0.18 * persistence * mean_compat) * fcz_gain * wm_gate)))

        return composed_128x4096.unsqueeze(0), self.neg_embed_1x128x4096, " ".join(debug_parts), advance_rate
