#!/usr/bin/env python3
"""
neuro_controller.py — 96,000-D Контроллер Рабочей Памяти для 3D-DiT (LTX-Video)
=================================================================================
- Никаких случайных 4096-D поворотов, уводящих T5-XXL за пределы конуса смыслов.
- При пустом промпте: навигация строго внутри эмпирического анизотропного эллипсоида Кента
  самой T5-XXL (mu, V, sigma, mean_norm).
- При K >= 2 концептах: RichSpace-навигация внутри активных (не-PAD) токенов пула +
  тета-гамма мультиплексирование всех активных регионов (e0..e4) в [1, 128, 4096].
- При stochastic = 1.0: 0° искажения T5-эмбеддингов (идеальное качество коробки).
=================================================================================
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuro_world import ArchetypeSpec, StageEntitySlot
from neuro_vram_bus import HardwareVRAMBus

CORTICAL_REGION_REGISTRY: dict[str, tuple[int, str, float, float]] = {
    "FCZ": (1, "FCz", 0.080, 0.25),
    "F3":  (2, "F3",  0.350, 0.55),
    "F4":  (3, "F4",  0.350, 0.55),
    "AFZ": (0, "AFz", 0.850, 0.70),
    "FPZ": (4, "Fpz", 2.500, 0.82),
    "CZ":  (1, "Cz",  0.045, 0.18),
    "PZ":  (2, "Pz",  0.022, 0.10),
    "OZ":  (3, "Oz",  0.010, 0.05),
}

ALL_GAMEPAD_CHANNELS = (
    "LS_UP", "LS_DOWN", "LS_LEFT", "LS_RIGHT",
    "RS_UP", "RS_DOWN", "RS_LEFT", "RS_RIGHT",
    "LT", "RT", "LB", "RB",
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
    "BTN_A", "BTN_B", "BTN_X", "BTN_Y",
    "L3", "R3", "BACK", "START",
)

DEFAULT_REGION_BINDINGS = {
    "FCZ": ["LS_UP", "LS_DOWN", "LS_LEFT", "LS_RIGHT", "BTN_A", "BTN_B"],
    "AFZ": ["RS_UP", "RS_DOWN", "RS_LEFT", "RS_RIGHT", "BTN_X", "BTN_Y"],
    "F3":  ["DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT", "LB", "RB"],
    "F4":  ["BTN_A", "BTN_B", "BTN_X", "BTN_Y", "LT", "RT"],
    "FPZ": ["LT", "RT", "START", "BACK"],
}

_CANONICAL_WM_BASIS_CACHE: dict[tuple[int, int, torch.device], torch.Tensor] = {}


@torch.no_grad()
def multiplex_t5_concept_slots(
    embeds_list: list[torch.Tensor],
    masks_list: list[torch.Tensor] | None = None,
    weights: list[float] | None = None,
    max_len: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Тета-гамма мультиплексирование M концептов/регионов в непересекающиеся слоты [1, 128, 4096].
    В LTX-Video attn2 не использует позиционный RoPE для текста (image_rotary_emb=None),
    поэтому упаковка активных токенов каждого региона/агента в отдельные слоты сохраняет 100%
    резкость без размытия и позволяет всем активным агентам (e0..e4) совместно строить сцену.
    """
    if not embeds_list:
        raise ValueError("multiplex_t5_concept_slots requires at least one embedding")

    ref_emb = embeds_list[0]
    dev = ref_emb.device
    orig_dtype = ref_emb.dtype
    seq_len = int(ref_emb.shape[1]) if ref_emb.ndim == 3 else int(max_len)
    dim = int(ref_emb.shape[-1])

    if masks_list is None or len(masks_list) != len(embeds_list):
        inferred_masks = []
        for e in embeds_list:
            e2 = e[0].float()
            diff = torch.norm(e2 - e2[-1:], p=2, dim=-1)
            m = diff > 1e-3
            if not bool(m.any().item()):
                m[0] = True
            inferred_masks.append(m.unsqueeze(0))
        masks_list = inferred_masks

    if len(embeds_list) == 1:
        return embeds_list[0], masks_list[0].to(device=dev, dtype=torch.bool)

    w_vals = (
        [float(w) for w in weights]
        if (weights is not None and len(weights) == len(embeds_list))
        else [1.0] * len(embeds_list)
    )
    w_max = max(max(w_vals), 1e-6)

    content_slices: list[tuple[torch.Tensor, float]] = []
    eos_token_vec: torch.Tensor | None = None
    pad_token_vec: torch.Tensor = ref_emb[0, -1:].float().clone()

    for emb_i, msk_i, w_i in zip(embeds_list, masks_list, w_vals):
        e_2d = emb_i[0].to(device=dev, dtype=torch.float32)
        m_1d = msk_i[0].to(device=dev, dtype=torch.bool)
        active_idx = torch.nonzero(m_1d, as_tuple=False).squeeze(-1)
        n_act = int(active_idx.numel())
        if n_act == 0:
            continue
        if eos_token_vec is None:
            eos_token_vec = e_2d[active_idx[-1] : active_idx[-1] + 1].clone()
        n_content = max(0, n_act - 1)
        if n_content > 0:
            c_idx = active_idx[:n_content]
            content_slices.append((e_2d[c_idx], w_i))

    if not content_slices:
        return embeds_list[0], masks_list[0].to(device=dev, dtype=torch.bool)

    m_cnt = len(content_slices)
    usable_len = max(1, seq_len - 1)
    total_raw = sum(s[0].shape[0] for s in content_slices)

    out_emb = pad_token_vec.expand(seq_len, dim).clone()
    out_msk = torch.zeros((seq_len,), device=dev, dtype=torch.bool)

    cursor = 0
    for tok_mat, w_i in content_slices:
        if cursor >= usable_len:
            break
        remaining_slots = usable_len - cursor
        if total_raw <= usable_len:
            take = min(tok_mat.shape[0], remaining_slots)
        else:
            fair_cap = max(2, usable_len // m_cnt)
            take = min(tok_mat.shape[0], fair_cap, remaining_slots)
        if take <= 0:
            continue

        # Иерархический gain-гейтинг (0.72..1.05): сохраняет T5 на многообразии
        rel_w = float(max(0.0, min(1.0, w_i / w_max)))
        gain = 0.72 + 0.30 * rel_w
        out_emb[cursor : cursor + take] = tok_mat[:take] * gain
        out_msk[cursor : cursor + take] = True
        cursor += take

    if eos_token_vec is not None and cursor < seq_len:
        out_emb[cursor : cursor + 1] = eos_token_vec
        out_msk[cursor] = True

    return out_emb.unsqueeze(0).to(orig_dtype), out_msk.unsqueeze(0)


@torch.no_grad()
def synthesize_empty_t5_from_kent_manifold(
    empty_pair: tuple[torch.Tensor, torch.Tensor],
    t5_manifold: dict[str, torch.Tensor] | None,
    wm_reg_5x32x120: torch.Tensor,
    wm_radius: float,
    stochastic: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    При пустом промпте ("") строит токены СТРОГО внутри эмпирического анизотропного
    эллипсоида Кента (mu, V, sigma, mean_norm), извлечённого из самой T5-XXL.
    При любом wm_radius вектор остаётся на реальном многообразии выходов T5-XXL.
    """
    empty_emb, empty_msk = empty_pair
    dev = empty_emb.device
    if (
        t5_manifold is None
        or stochastic >= 0.999
        or wm_radius <= 0.001
        or float(wm_reg_5x32x120.abs().max().item()) < 1e-4
    ):
        return empty_emb, empty_msk

    mu = t5_manifold["mu"].to(dev, dtype=torch.float32)
    V = t5_manifold["V"].to(dev, dtype=torch.float32)
    sigma = t5_manifold["sigma"].to(dev, dtype=torch.float32)
    mean_norm = float(t5_manifold["mean_norm"].item())

    L_free, R = 16, V.shape[0]
    wm_flat = wm_reg_5x32x120.to(dev, dtype=torch.float32).reshape(-1)
    wm_w = (wm_flat - wm_flat.mean()) / wm_flat.std(unbiased=False).clamp(min=1e-5)

    coords = F.adaptive_avg_pool1d(wm_w.view(1, 1, -1), L_free * R).view(L_free, R)
    coords = (coords - coords.mean(dim=0, keepdim=True)) / coords.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-5)

    scale = float(wm_radius) * (1.0 - float(stochastic)) * 1.15
    delta = torch.matmul(coords * (sigma.unsqueeze(0) * scale), V)
    raw_toks = mu.unsqueeze(0) + delta
    toks_on_cone = F.normalize(raw_toks, p=2, dim=-1) * mean_norm

    out_emb = empty_emb[0].float().clone()
    out_msk = torch.zeros((out_emb.shape[0],), device=dev, dtype=torch.bool)
    out_emb[:L_free] = toks_on_cone
    out_emb[L_free] = empty_emb[0, 0].float()
    out_msk[: L_free + 1] = True
    return out_emb.unsqueeze(0).to(empty_emb.dtype), out_msk.unsqueeze(0)


@torch.no_grad()
def steer_pool_t5_within_manifold(
    anchor_emb_1x128x4096: torch.Tensor,
    anchor_msk_1x128: torch.Tensor,
    bank_Kx128x4096: torch.Tensor,
    masks_Kx128: torch.Tensor,
    wm_reg_5x32x120: torch.Tensor,
    t5_manifold: dict[str, torch.Tensor] | None,
    wm_radius: float,
    stochastic: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Внутри-многообразная навигация T5-XXL (0 PAD-токенов в SVD, 0 случайных 4096-D осей):
    Сдвиг идёт строго вдоль реальных семантических осей пула V_pool (с добором
    только из анизотропного спектра T5-XXL при малом K) в безопасном конусе <= 14°.
    """
    dev = anchor_emb_1x128x4096.device
    anchor_msk = anchor_msk_1x128.to(dev, dtype=torch.bool)
    if (
        stochastic >= 0.999
        or wm_radius <= 0.001
        or float(wm_reg_5x32x120.abs().max().item()) < 1e-4
    ):
        return anchor_emb_1x128x4096, anchor_msk

    act_idx = torch.nonzero(anchor_msk[0], as_tuple=False).squeeze(-1)
    if act_idx.numel() < 3:
        return anchor_emb_1x128x4096, anchor_msk
    content_idx = act_idx[1:-1]

    active_vecs = []
    K = bank_Kx128x4096.shape[0]
    for k_i in range(K):
        m_i = masks_Kx128[k_i].to(dev, dtype=torch.bool)
        idx_i = torch.nonzero(m_i, as_tuple=False).squeeze(-1)
        if idx_i.numel() > 2:
            active_vecs.append(bank_Kx128x4096[k_i, idx_i[1:-1]].float())
    if not active_vecs:
        return anchor_emb_1x128x4096, anchor_msk

    sem_mat = torch.cat(active_vecs, dim=0)
    sem_c = sem_mat - sem_mat.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(sem_c, full_matrices=False)
    valid_r = max(1, int((S > S[0] * 1e-3).sum().item())) if S.numel() > 0 else 1
    V_pool = Vh[: min(valid_r, 48)]
    sigma_pool = S[: V_pool.shape[0]] / math.sqrt(max(1, sem_c.shape[0] - 1))

    if t5_manifold is not None and V_pool.shape[0] < 32:
        V_t5 = t5_manifold["V"].to(dev, dtype=torch.float32)[:32]
        sig_t5 = t5_manifold["sigma"].to(dev, dtype=torch.float32)[:32] * 0.35
        V_basis = torch.cat([V_pool, V_t5], dim=0)
        sig_basis = torch.cat([sigma_pool, sig_t5], dim=0)
    else:
        V_basis, sig_basis = V_pool, sigma_pool

    wm_flat = wm_reg_5x32x120.to(dev, dtype=torch.float32).reshape(-1)
    wm_w = (wm_flat - wm_flat.mean()) / wm_flat.std(unbiased=False).clamp(min=1e-5)
    L_c, R_b = content_idx.numel(), V_basis.shape[0]
    coords = F.adaptive_avg_pool1d(wm_w.view(1, 1, -1), L_c * R_b).view(L_c, R_b)
    coords = (coords - coords.mean(dim=0, keepdim=True)) / coords.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-5)

    wm_sem = torch.matmul(coords * sig_basis.unsqueeze(0), V_basis)

    tok_c = anchor_emb_1x128x4096[0, content_idx].float()
    ref_norms = torch.norm(tok_c, p=2, dim=-1, keepdim=True).clamp(min=1e-4)
    dot_p = torch.sum(wm_sem * tok_c, dim=-1, keepdim=True) / (ref_norms ** 2 + 1e-6)
    tangent = wm_sem - dot_p * tok_c
    t_norms = torch.norm(tangent, p=2, dim=-1, keepdim=True)
    tangent_unit = torch.where(
        t_norms > 1e-5, (tangent / (t_norms + 1e-6)) * ref_norms, torch.zeros_like(tok_c)
    )

    theta = float(wm_radius) * (1.0 - float(stochastic)) * (math.pi * 0.08)
    steered_c = F.normalize(math.cos(theta) * tok_c + math.sin(theta) * tangent_unit, p=2, dim=-1) * ref_norms

    out = anchor_emb_1x128x4096[0].float().clone()
    out[content_idx] = steered_c
    return out.unsqueeze(0).to(anchor_emb_1x128x4096.dtype), anchor_msk


@torch.no_grad()
def get_canonical_concept_wm_basis_Kx5x32x120(
    K: int, reg_idx: int, device: torch.device
) -> torch.Tensor:
    K_safe = max(1, int(K))
    key = (K_safe, int(reg_idx), device)
    if key not in _CANONICAL_WM_BASIS_CACHE:
        s_grid = torch.linspace(0.0, 2.0 * math.pi, 32, device=device).view(1, 32, 1)
        p_grid = torch.linspace(-math.pi, math.pi, 120, device=device).view(1, 1, 120)
        b_grid = torch.arange(5, device=device, dtype=torch.float32).view(5, 1, 1)

        modes = []
        for k_i in range(K_safe):
            freq_s = float(k_i + 1)
            freq_p = float((k_i % 5) + 1)
            phase_shift = (k_i * 2.0 * math.pi / float(K_safe)) + reg_idx * 1.13
            wave = torch.sin(
                s_grid * freq_s + p_grid * freq_p + b_grid * float(k_i + 1) * 0.7 + phase_shift
            ) + torch.cos(
                s_grid * (freq_p + 0.5) - p_grid * freq_s - b_grid * 0.5 + phase_shift * 1.7
            )
            wave = wave - wave.mean()
            modes.append(wave.reshape(-1))

        raw_Kx19200 = torch.stack(modes, dim=0)
        raw_Kx19200 = raw_Kx19200 - raw_Kx19200.mean(dim=-1, keepdim=True)
        q_19200xK, _ = torch.linalg.qr(raw_Kx19200.T, mode="reduced")
        ortho_Kx5x32x120 = (
            q_19200xK.T * math.sqrt(19200.0)
        ).view(K_safe, 5, 32, 120).contiguous()
        _CANONICAL_WM_BASIS_CACHE[key] = ortho_Kx5x32x120
    return _CANONICAL_WM_BASIS_CACHE[key]


@torch.no_grad()
def synthesize_region_gamepad_wm_5x32x120(
    reg_key: str,
    reg_idx: int,
    K_reg: int,
    gamepad_channels: dict[str, float] | None,
    concept_bindings_by_region: dict[str, list[str]] | None,
    phase_step: float,
    device: torch.device,
) -> tuple[torch.Tensor | None, float, list[str]]:
    if not gamepad_channels:
        return None, 0.0, []

    r_up = reg_key.upper()
    bindings = (concept_bindings_by_region or {}).get(
        r_up, DEFAULT_REGION_BINDINGS.get(r_up, [])
    )
    K_safe = max(1, int(K_reg))
    basis_Kx5x32x120 = get_canonical_concept_wm_basis_Kx5x32x120(
        K_safe, reg_idx, device
    )

    reg_wave = torch.zeros((5, 32, 120), device=device, dtype=torch.float32)
    active_parts: list[str] = []
    max_act = 0.0

    for k_i in range(K_safe):
        bind_str = (
            bindings[k_i].strip().upper()
            if k_i < len(bindings)
            else (
                DEFAULT_REGION_BINDINGS.get(r_up, [])[k_i]
                if k_i < len(DEFAULT_REGION_BINDINGS.get(r_up, []))
                else "NONE"
            )
        )
        if not bind_str or bind_str == "NONE":
            continue
        sub_keys = [s.strip() for s in bind_str.split("+") if s.strip()]
        if not sub_keys:
            continue
        vals = [float(gamepad_channels.get(sk, 0.0)) for sk in sub_keys]
        act_val = min(vals) if len(vals) > 1 else vals[0]
        if act_val > 0.05:
            reg_wave.add_(basis_Kx5x32x120[k_i] * float(act_val))
            max_act = max(max_act, float(act_val))
            active_parts.append(f"#{k_i+1}:{bind_str}({act_val:.2f})")

    if not active_parts:
        return None, 0.0, []

    return reg_wave.contiguous(), max_act, active_parts


class SymmetricBotController:

    def __init__(
        self,
        bot_id: str,
        archetypes: dict[str, ArchetypeSpec] | None = None,
        source_mode: str = "eeg:FCz",
        stochastic: float = 0.0,
        wm_radius: float = 0.35,
    ):
        self.bot_id = bot_id
        self.archetypes = archetypes or {}
        self.source_mode = str(source_mode)
        self.stochastic = float(stochastic)
        self.wm_radius = float(wm_radius)

        self.eeg_wm_gain: float = 1.0
        self.gamepad_wm_mix: float = 1.0
        self.pool_size_hint: int = 4
        self.gamepad_channels: dict[str, float] = {}
        self.concept_bindings_by_region: dict[str, list[str]] = {
            k: list(v) for k, v in DEFAULT_REGION_BINDINGS.items()
        }
        self.t5_manifold: dict[str, torch.Tensor] | None = None

        self.vram_bus = HardwareVRAMBus()
        self.attempt_num: int = 1
        self.last_stage_id: int = -1

        self.phase_clock_5x32x120: torch.Tensor | None = None
        self.autonomous_wm_field_5x32x120: torch.Tensor | None = None
        self.latest_full_wm_5x5x32x120: torch.Tensor | None = None
        self.latest_region_wm_19200: torch.Tensor | None = None
        self.latest_region_wm_5x32x120: torch.Tensor | None = None
        self.latest_source_tag: str = "WM"

        self.npc_logits_K: torch.Tensor | None = None
        self.last_chosen_idx: int = -1

    def is_npc_binding(self) -> bool:
        m = self.source_mode.upper()
        return m.startswith("NPC") or ":NPC" in m

    def _get_region_specs(self) -> tuple[int, str, float, float]:
        mode_clean = self.source_mode.upper()
        for k, val in CORTICAL_REGION_REGISTRY.items():
            if k in mode_clean:
                return val
        return CORTICAL_REGION_REGISTRY["FCZ"]

    @torch.no_grad()
    def observe_oracle_feedback(self, reward_scalar: float):
        if not self.is_npc_binding() or self.npc_logits_K is None:
            return
        k_len = self.npc_logits_K.numel()
        if 0 <= self.last_chosen_idx < k_len:
            r_clamped = float(max(-1.0, min(1.0, reward_scalar)))
            self.npc_logits_K[self.last_chosen_idx] += 0.65 * r_clamped
            self.npc_logits_K.mul_(0.92)
            self.npc_logits_K.clamp_(-3.5, 3.5)

    @torch.no_grad()
    def inject_dit_velocity_feedback(self, dit_velocity_5d: torch.Tensor | None):
        if dit_velocity_5d is None or self.latest_region_wm_5x32x120 is None:
            return
        if not self.is_npc_binding():
            return
        dev = self.latest_region_wm_5x32x120.device
        _, _, _, inertia = self._get_region_specs()
        v_flat = dit_velocity_5d.to(device=dev, dtype=torch.float32).reshape(1, 1, -1)
        v_19200 = F.adaptive_avg_pool1d(v_flat, 19200).view(5, 32, 120)
        v_norm = (v_19200 - v_19200.mean()) / v_19200.std(unbiased=False).clamp(min=1e-5)

        cur_w = (
            self.autonomous_wm_field_5x32x120
            if self.autonomous_wm_field_5x32x120 is not None
            else self.latest_region_wm_5x32x120
        )
        rot_angle = 0.28 * (1.0 - inertia * 0.5)
        phase_torque = torch.sin(v_norm * (math.pi * 0.65) - cur_w * 0.5)
        self.autonomous_wm_field_5x32x120 = (
            math.cos(rot_angle) * cur_w + math.sin(rot_angle) * phase_torque * 2.2
        ).contiguous()

    @torch.no_grad()
    def sample_working_memory_5x5x32x120(
        self,
        device: torch.device,
        sub_step_frac: float = 0.0,
    ) -> tuple[torch.Tensor, float, str]:
        reg_idx, reg_name, _, inertia = self._get_region_specs()
        reg_key_up = reg_name.upper()
        eff_t = float(self.attempt_num) + float(sub_step_frac)

        # 1. Автономный NPC-агент (репликаторная + нелинейная динамика на собственных модах)
        if self.is_npc_binding():
            K_safe = max(1, int(self.pool_size_hint))
            if (
                self.npc_logits_K is None
                or self.npc_logits_K.numel() != K_safe
                or self.npc_logits_K.device != device
            ):
                self.npc_logits_K = torch.zeros(K_safe, device=device, dtype=torch.float32)

            k_grid = torch.arange(K_safe, device=device, dtype=torch.float32)
            explore_wave = 0.45 * torch.sin(
                eff_t * 0.55 + k_grid * (2.0 * math.pi / float(K_safe)) + reg_idx * 0.9
            )
            probs_npc = torch.softmax((self.npc_logits_K + explore_wave) / 0.45, dim=0)

            basis_Kx5x32x120 = get_canonical_concept_wm_basis_Kx5x32x120(
                K_safe, reg_idx, device
            )
            npc_wave = torch.sum(
                basis_Kx5x32x120 * probs_npc.view(K_safe, 1, 1, 1), dim=0
            ) * 3.0

            if self.autonomous_wm_field_5x32x120 is not None and self.autonomous_wm_field_5x32x120.device == device:
                npc_wave = npc_wave * 0.70 + self.autonomous_wm_field_5x32x120 * 0.30
            else:
                self.autonomous_wm_field_5x32x120 = npc_wave.clone()

            out_wm = torch.zeros((5, 5, 32, 120), device=device, dtype=torch.float32)
            out_wm[reg_idx] = npc_wave
            top_k = int(torch.argmax(probs_npc).item())
            src_tag = f"NPC_WM:{reg_name}(#{top_k+1}:{probs_npc[top_k].item()*100:.0f}%)"
            self.latest_full_wm_5x5x32x120 = out_wm
            self.latest_region_wm_5x32x120 = out_wm[reg_idx].clone()
            self.latest_region_wm_19200 = out_wm[reg_idx].reshape(-1).clone()
            self.latest_source_tag = src_tag
            return out_wm, inertia, src_tag

        # 2. Биологическая ЭЭГ / Синтезатор от Геймпада
        base_wm_5x5x32x120 = None
        src_tag = f"SYNTH_WM:{reg_name}"

        if "eeg" in self.source_mode.lower() or self.source_mode.lower() == "auto":
            self.vram_bus.try_attach_to_existing_vram()
            if self.vram_bus.attached_pid > 0:
                wm_live = self.vram_bus.wm_field_5x5x32x120.to(
                    device=device, dtype=torch.float32
                )
                if float(wm_live[reg_idx].abs().max().item()) > 1e-4:
                    base_wm_5x5x32x120 = wm_live.clone()
                    src_tag = f"EEG_VRAM:{reg_name}"

        if base_wm_5x5x32x120 is None:
            s_grid = torch.linspace(0.0, 2.0 * math.pi, 32, device=device).view(1, 1, 32, 1)
            p_grid = torch.linspace(-math.pi, math.pi, 120, device=device).view(1, 1, 1, 120)
            r_grid = torch.arange(5, device=device, dtype=torch.float32).view(5, 1, 1, 1)
            b_grid = torch.arange(5, device=device, dtype=torch.float32).view(1, 5, 1, 1)

            if self.phase_clock_5x32x120 is None or self.phase_clock_5x32x120.device != device:
                gen_p = torch.Generator(device=device).manual_seed(1009 + reg_idx * 313)
                self.phase_clock_5x32x120 = (
                    torch.randn((1, 5, 32, 120), generator=gen_p, device=device) * 0.45
                )
            elif sub_step_frac <= 0.001 and self.eeg_wm_gain > 0.01:
                for b_i in range(5):
                    self.phase_clock_5x32x120[:, b_i] = torch.roll(
                        self.phase_clock_5x32x120[:, b_i], shifts=(b_i + 1), dims=-1
                    )

            phase_t = eff_t * (0.65 + 0.45 * max(0.15, self.wm_radius)) * float(self.eeg_wm_gain)
            wm_synth = torch.sin(
                s_grid * (1.0 + 0.25 * b_grid)
                + p_grid * (1.1 + 0.20 * r_grid)
                + self.phase_clock_5x32x120
                + phase_t * (0.42 + 0.15 * r_grid)
            ) + 0.55 * torch.cos(
                s_grid * (0.7 - 0.18 * r_grid)
                - p_grid * (0.9 + 0.25 * b_grid)
                + phase_t * (0.55 + 0.18 * b_grid)
            )
            weights_5 = torch.full((5, 1, 1, 1), 0.30, device=device, dtype=torch.float32)
            weights_5[reg_idx] = 1.0
            base_wm_5x5x32x120 = (wm_synth * weights_5).contiguous()

        eeg_gain = float(max(0.0, min(1.0, self.eeg_wm_gain)))
        out_wm = base_wm_5x5x32x120 * eeg_gain

        gp_mix = float(max(0.0, min(1.0, self.gamepad_wm_mix)))
        if gp_mix > 0.001 and self.gamepad_channels:
            gp_wave_5x32x120, max_act, act_parts = synthesize_region_gamepad_wm_5x32x120(
                reg_key=reg_key_up if reg_key_up in DEFAULT_REGION_BINDINGS else "FCZ",
                reg_idx=reg_idx,
                K_reg=self.pool_size_hint,
                gamepad_channels=self.gamepad_channels,
                concept_bindings_by_region=self.concept_bindings_by_region,
                phase_step=eff_t,
                device=device,
            )
            if gp_wave_5x32x120 is not None and max_act > 0.05:
                theta_gp = gp_mix * min(1.0, max_act) * (0.5 * math.pi)
                out_wm[reg_idx] = (
                    math.cos(theta_gp) * out_wm[reg_idx]
                    + math.sin(theta_gp) * gp_wave_5x32x120 * 3.0
                )
                src_tag = f"GP_WM:{'+'.join(act_parts[:2])}"

        self.latest_full_wm_5x5x32x120 = out_wm
        self.latest_region_wm_5x32x120 = out_wm[reg_idx].clone()
        self.latest_region_wm_19200 = out_wm[reg_idx].reshape(-1).clone()
        self.latest_source_tag = src_tag
        return out_wm, inertia, src_tag

    @torch.no_grad()
    def select_concept_full_rank_524288d(
        self,
        pool_ids: list[str],
        candidate_embeds_Kx128x4096: torch.Tensor,
        device: torch.device,
        current_state_id: str | None = None,
        catalyst_field_128x4096: torch.Tensor | None = None,
        candidate_masks_Kx128: torch.Tensor | None = None,
    ) -> tuple[str, str, torch.Tensor, torch.Tensor, dict[str, float], list[int]]:
        K = len(pool_ids)
        self.pool_size_hint = max(1, K)
        if candidate_masks_Kx128 is None:
            fallback_m = torch.ones(
                (max(1, K), candidate_embeds_Kx128x4096.shape[1]),
                device=device,
                dtype=torch.bool,
            )
        else:
            fallback_m = candidate_masks_Kx128.to(device=device, dtype=torch.bool)

        if K == 0:
            self.last_chosen_idx = -1
            return "", "", candidate_embeds_Kx128x4096[:1], fallback_m[:1], {}, []

        reg_idx, _, _, _ = self._get_region_specs()
        self.sample_working_memory_5x5x32x120(device, sub_step_frac=0.0)
        bank_float = candidate_embeds_Kx128x4096.to(device=device, dtype=torch.float32)

        reg_wm_5x32x120 = (
            self.latest_region_wm_5x32x120
            if self.latest_region_wm_5x32x120 is not None
            else torch.zeros((5, 32, 120), device=device, dtype=torch.float32)
        )

        # При stochastic == 1.0 (режим "из коробки"): чистый промпт без искажения T5!
        if self.stochastic >= 0.999:
            self.last_chosen_idx = 0
            probs_dict = {pool_ids[i]: round(100.0 / float(K), 1) for i in range(K)}
            if K == 1:
                return pool_ids[0], pool_ids[0], bank_float[0:1], fallback_m[0:1], {pool_ids[0]: 100.0}, [0]
            comp_emb, comp_msk = multiplex_t5_concept_slots(
                embeds_list=[bank_float[i : i + 1] for i in range(min(K, 4))],
                masks_list=[fallback_m[i : i + 1] for i in range(min(K, 4))],
                weights=[1.0] * min(K, 4),
                max_len=bank_float.shape[1],
            )
            return pool_ids[0], pool_ids[0], comp_emb, comp_msk, probs_dict, list(range(min(K, 4)))

        # Нулевой сигнал рабочей памяти -> равновесие
        if float(reg_wm_5x32x120.abs().max().item()) < 1e-4:
            self.last_chosen_idx = -1
            eq_pct = round(100.0 / float(K), 1)
            probs_dict = {pool_ids[i]: eq_pct for i in range(K)}
            return "", "", bank_float[0:1], fallback_m[0:1], probs_dict, []

        if K == 1:
            self.last_chosen_idx = 0
            only_id = pool_ids[0]
            steered_emb, steered_msk = steer_pool_t5_within_manifold(
                anchor_emb_1x128x4096=bank_float[0:1],
                anchor_msk_1x128=fallback_m[0:1],
                bank_Kx128x4096=bank_float,
                masks_Kx128=fallback_m,
                wm_reg_5x32x120=reg_wm_5x32x120,
                t5_manifold=self.t5_manifold,
                wm_radius=self.wm_radius,
                stochastic=self.stochastic,
            )
            return only_id, only_id, steered_emb, steered_msk, {only_id: 100.0}, [0]

        basis_Kx19200 = get_canonical_concept_wm_basis_Kx5x32x120(
            K, reg_idx, device
        ).reshape(K, 19200)
        wm_reg_19200 = reg_wm_5x32x120.reshape(1, 19200)
        wm_resonances_K = torch.mm(wm_reg_19200, basis_Kx19200.T).squeeze(0) / 19200.0

        pos_res = torch.relu(wm_resonances_K - 0.10)
        if float(pos_res.max().item()) > 0.12:
            probs_K = pos_res / pos_res.sum().clamp(min=1e-6)
        else:
            r_std = wm_resonances_K.std(unbiased=False).clamp(min=1e-5)
            logits_K = (wm_resonances_K - wm_resonances_K.mean()) / r_std
            probs_K = torch.softmax(logits_K / 0.45, dim=0)

        choice_order = torch.argsort(probs_K, descending=True).tolist()
        choice_probs_dict = {
            pool_ids[idx]: round(float(probs_K[idx].item()) * 100.0, 1)
            for idx in range(K)
        }

        # Тета-гамма мультиплексирование совместно активных концептов (порог >= 15%)
        active_indices = [
            idx for idx in choice_order if float(probs_K[idx].item()) >= 0.15
        ][:4]
        if not active_indices:
            active_indices = [choice_order[0]]

        self.last_chosen_idx = int(active_indices[0])

        if len(active_indices) == 1:
            top_idx = active_indices[0]
            composed_base = bank_float[top_idx : top_idx + 1]
            composed_mask = fallback_m[top_idx : top_idx + 1]
            composite_target_id = pool_ids[top_idx]
        else:
            embeds_act = [bank_float[i : i + 1] for i in active_indices]
            masks_act = [fallback_m[i : i + 1] for i in active_indices]
            weights_act = [float(probs_K[i].item()) for i in active_indices]
            composed_base, composed_mask = multiplex_t5_concept_slots(
                embeds_list=embeds_act,
                masks_list=masks_act,
                weights=weights_act,
                max_len=bank_float.shape[1],
            )
            composite_target_id = "+".join([pool_ids[i] for i in active_indices[:3]])

        steered_emb, steered_msk = steer_pool_t5_within_manifold(
            anchor_emb_1x128x4096=composed_base,
            anchor_msk_1x128=composed_mask,
            bank_Kx128x4096=bank_float,
            masks_Kx128=fallback_m,
            wm_reg_5x32x120=reg_wm_5x32x120,
            t5_manifold=self.t5_manifold,
            wm_radius=self.wm_radius,
            stochastic=self.stochastic,
        )

        start_id = (
            current_state_id
            if (current_state_id and current_state_id in pool_ids)
            else pool_ids[active_indices[0]]
        )
        return (
            start_id,
            composite_target_id,
            steered_emb,
            steered_msk,
            choice_probs_dict,
            active_indices,
        )

    @torch.no_grad()
    def steer_semantic_probe_full_rank(
        self,
        base_probe_1x128x4096: torch.Tensor,
        device: torch.device,
        wm_override_128x4096: torch.Tensor | None = None,
        all_candidates_Kx128x4096: torch.Tensor | None = None,
        candidate_weights_K: torch.Tensor | None = None,
        sub_step_frac: float = 0.0,
        attention_mask_1x128: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.wm_radius <= 0.001
            or self.stochastic >= 0.999
            or (self.eeg_wm_gain <= 0.001 and not self.is_npc_binding() and not self.gamepad_channels)
        ):
            return base_probe_1x128x4096

        self.sample_working_memory_5x5x32x120(device, sub_step_frac=sub_step_frac)
        if self.latest_region_wm_5x32x120 is None or float(self.latest_region_wm_5x32x120.abs().max().item()) < 1e-4:
            return base_probe_1x128x4096

        msk = (
            attention_mask_1x128.to(device=device, dtype=torch.bool)
            if attention_mask_1x128 is not None
            else torch.ones((1, base_probe_1x128x4096.shape[1]), device=device, dtype=torch.bool)
        )
        bank = (
            all_candidates_Kx128x4096.to(device=device, dtype=torch.float32)
            if all_candidates_Kx128x4096 is not None
            else base_probe_1x128x4096.to(device=device, dtype=torch.float32)
        )
        b_masks = (
            msk.expand(bank.shape[0], -1)
            if bank.shape[0] > 1
            else msk
        )
        steered, _ = steer_pool_t5_within_manifold(
            anchor_emb_1x128x4096=base_probe_1x128x4096,
            anchor_msk_1x128=msk,
            bank_Kx128x4096=bank,
            masks_Kx128=b_masks,
            wm_reg_5x32x120=self.latest_region_wm_5x32x120,
            t5_manifold=self.t5_manifold,
            wm_radius=self.wm_radius * 0.5,
            stochastic=self.stochastic,
        )
        return steered

    def select_party(
        self,
        global_stage: int,
        my_cfg: StageEntitySlot,
    ) -> list[str]:
        if global_stage != self.last_stage_id:
            self.last_stage_id = global_stage
            self.attempt_num = 1
        pool = (
            my_cfg.allowed_pool
            if my_cfg.allowed_pool
            else list(self.archetypes.keys())
        )
        return list(pool) if pool else []
