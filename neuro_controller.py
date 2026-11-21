#!/usr/bin/env python3
"""
neuro_controller.py — 96,000-D Контроллер Рабочей Памяти (Аконцептуальный: ЭЭГ × Геймпад × NPC-Агенты)
С Тета-Гамма мультиплексированием нескольких концептов в ортогональные токен-слоты [1, 128, 4096].
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuro_world import ArchetypeSpec, StageEntitySlot
from neuro_vram_bus import HardwareVRAMBus

CORTICAL_REGION_REGISTRY = {
    "FCZ": (1, "FCz", 0.020, 0.22),
    "F3":  (2, "F3",  0.180, 0.55),
    "F4":  (3, "F4",  0.180, 0.55),
    "AFZ": (0, "AFz", 0.650, 0.68),
    "FPZ": (4, "Fpz", 2.500, 0.78),
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
    Тета-гамма мультиплексирование M концептов в непересекающиеся слоты [1, 128, 4096].
    В LTX-Video attn2 кросс-внимание не использует позиционный RoPE для текста (image_rotary_emb=None),
    поэтому упаковка активных токенов каждого концепта в отдельные слоты сохраняет 100% резкость
    и позволяет 3D-DiT органично соединять несколько концептов в одной сцене без размытия в муть.
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
        return embeds_list[0], masks_list[0]

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
        # Отделяем хвостовой </s>, оставляя чистые контентные токены концепта:
        n_content = max(0, n_act - 1)
        if n_content > 0:
            c_idx = active_idx[:n_content]
            content_slices.append((e_2d[c_idx], w_i))

    if not content_slices:
        return embeds_list[0], masks_list[0]

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

        # Мягкая иерархическая модуляция Value (0.72..1.0), сохраняющая все токены на многообразии T5:
        rel_w = float(max(0.0, min(1.0, w_i / w_max)))
        gain = 0.72 + 0.28 * rel_w
        out_emb[cursor : cursor + take] = tok_mat[:take] * gain
        out_msk[cursor : cursor + take] = True
        cursor += take

    if eos_token_vec is not None and cursor < seq_len:
        out_emb[cursor : cursor + 1] = eos_token_vec
        out_msk[cursor] = True

    return out_emb.unsqueeze(0).to(orig_dtype), out_msk.unsqueeze(0)


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

        self.vram_bus = HardwareVRAMBus()
        self.attempt_num: int = 1
        self.last_stage_id: int = -1

        self.phase_clock_5x32x120: torch.Tensor | None = None
        self.proj_120_to_64: torch.Tensor | None = None
        self.proj_600_to_64: torch.Tensor | None = None
        self.baseline_wm_128x64: torch.Tensor | None = None
        self.latest_region_wm_19200: torch.Tensor | None = None
        self.latest_region_wm_5x32x120: torch.Tensor | None = None
        self.latest_source_tag: str = "WM"

        # Состояние автономного NPC-агента (чистая репликаторная политика на собственных модах WM):
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
    def sample_working_memory_5x5x32x120(
        self,
        device: torch.device,
        sub_step_frac: float = 0.0,
    ) -> tuple[torch.Tensor, float, str]:
        reg_idx, reg_name, _, inertia = self._get_region_specs()
        reg_key_up = reg_name.upper()
        eff_t = float(self.attempt_num) + float(sub_step_frac)

        # 1. Автономный NPC-агент
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

            out_wm = torch.zeros((5, 5, 32, 120), device=device, dtype=torch.float32)
            out_wm[reg_idx] = npc_wave
            top_k = int(torch.argmax(probs_npc).item())
            src_tag = f"NPC_WM:{reg_name}(#{top_k+1}:{probs_npc[top_k].item()*100:.0f}%)"
            self.latest_region_wm_5x32x120 = out_wm[reg_idx].clone()
            self.latest_region_wm_19200 = out_wm[reg_idx].reshape(-1).clone()
            self.latest_source_tag = src_tag
            return out_wm, inertia, src_tag

        # 2. Человеческий / биосигнальный агент (FreeEEG16 + Gamepad Synthesizer)
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
                reg_key=reg_key_up,
                reg_idx=reg_idx,
                K_reg=self.pool_size_hint,
                gamepad_channels=self.gamepad_channels,
                concept_bindings_by_region=self.concept_bindings_by_region,
                phase_step=eff_t,
                device=device,
            )
            if gp_wave_5x32x120 is not None and max_act > 0.05:
                w_gp = gp_mix * min(1.0, max_act)
                w_eeg = eeg_gain * max(0.0, 1.0 - 0.96 * gp_mix)
                out_wm[reg_idx] = (
                    base_wm_5x5x32x120[reg_idx] * w_eeg + gp_wave_5x32x120 * (w_gp * 3.0)
                )
                src_tag = f"GP_WM({w_gp*100:.0f}%):{'+'.join(act_parts[:2])}"

        self.latest_region_wm_5x32x120 = out_wm[reg_idx].clone()
        self.latest_region_wm_19200 = out_wm[reg_idx].reshape(-1).clone()
        self.latest_source_tag = src_tag
        return out_wm, inertia, src_tag

    @torch.no_grad()
    def extract_whitened_wm_modes_128x64(
        self,
        device: torch.device,
        sub_step_frac: float = 0.0,
    ) -> tuple[torch.Tensor, float, str]:
        wm_5x5x32x120, inertia, src_tag = self.sample_working_memory_5x5x32x120(
            device, sub_step_frac=sub_step_frac
        )
        reg_idx, _, _, _ = self._get_region_specs()

        if float(wm_5x5x32x120[reg_idx].abs().max().item()) < 1e-5:
            return torch.zeros((128, 64), device=device, dtype=torch.float32), inertia, src_tag

        if self.proj_120_to_64 is None or self.proj_120_to_64.device != device:
            g_proj = torch.Generator(device=device).manual_seed(999 + reg_idx * 137)
            q_120, _ = torch.linalg.qr(
                torch.randn((120, 64), generator=g_proj, device=device, dtype=torch.float32)
            )
            self.proj_120_to_64 = q_120.contiguous()

        if self.proj_600_to_64 is None or self.proj_600_to_64.device != device:
            g_proj_all = torch.Generator(device=device).manual_seed(7777 + reg_idx * 251)
            q_600, _ = torch.linalg.qr(
                torch.randn((600, 64), generator=g_proj_all, device=device, dtype=torch.float32)
            )
            self.proj_600_to_64 = q_600.contiguous()

        reg_160x120 = wm_5x5x32x120[reg_idx].reshape(160, 120)
        reg_160x64 = torch.matmul(reg_160x120, self.proj_120_to_64)

        all_96k_160x600 = wm_5x5x32x120.permute(1, 2, 0, 3).reshape(160, 600)
        all_96k_160x64 = torch.matmul(all_96k_160x600, self.proj_600_to_64)

        full_160x64 = reg_160x64 * 0.80 + all_96k_160x64 * 0.20

        wm_128x64 = (
            F.interpolate(
                full_160x64.T.unsqueeze(0), size=128, mode="linear", align_corners=False
            )
            .squeeze(0)
            .T.contiguous()
        )

        if self.baseline_wm_128x64 is None or self.baseline_wm_128x64.shape != wm_128x64.shape:
            self.baseline_wm_128x64 = wm_128x64.clone()
        else:
            self.baseline_wm_128x64.lerp_(wm_128x64, 0.04)

        dyn_128x64 = wm_128x64 - self.baseline_wm_128x64 * 0.50
        std_val = dyn_128x64.std(dim=-1, keepdim=True).clamp(min=1e-5)
        whitened_128x64 = (dyn_128x64 - dyn_128x64.mean(dim=-1, keepdim=True)) / std_val
        return whitened_128x64, inertia, src_tag

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
        self.extract_whitened_wm_modes_128x64(device, sub_step_frac=0.0)
        bank_float = candidate_embeds_Kx128x4096.to(device=device, dtype=torch.float32)

        wm_reg_19200 = (
            self.latest_region_wm_5x32x120.reshape(1, 19200)
            if self.latest_region_wm_5x32x120 is not None
            else torch.zeros((1, 19200), device=device, dtype=torch.float32)
        )

        # Нулевой сигнал рабочей памяти -> равновесное состояние (обрабатывается движком без рывка к строке #0)
        if float(wm_reg_19200.abs().max().item()) < 1e-4:
            self.last_chosen_idx = -1
            eq_pct = round(100.0 / float(K), 1)
            probs_dict = {pool_ids[i]: eq_pct for i in range(K)}
            return "", "", bank_float[0:1], fallback_m[0:1], probs_dict, []

        if K == 1:
            self.last_chosen_idx = 0
            only_id = pool_ids[0]
            steered = self.steer_semantic_probe_full_rank(
                bank_float[0:1], device=device, attention_mask_1x128=fallback_m[0:1]
            )
            return only_id, only_id, steered, fallback_m[0:1], {only_id: 100.0}, [0]

        basis_Kx19200 = get_canonical_concept_wm_basis_Kx5x32x120(
            K, reg_idx, device
        ).reshape(K, 19200)

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

        # Разрешаем до 6 одновременно активных концептов (гамма-субциклы внутри тета-цикла региона):
        active_indices = [
            idx for idx in choice_order if float(probs_K[idx].item()) >= 0.12
        ][:6]
        if not active_indices:
            active_indices = [choice_order[0]]

        self.last_chosen_idx = int(active_indices[0])

        if len(active_indices) == 1:
            top_idx = active_indices[0]
            composed_base = bank_float[top_idx : top_idx + 1]
            composed_mask = fallback_m[top_idx : top_idx + 1]
            composite_target_id = pool_ids[top_idx]
        else:
            # ТЕТА-ГАММА МУЛЬТИПЛЕКСИРОВАНИЕ В ОРТОГОНАЛЬНЫЕ СЛОТЫ (0 размытия в муть!):
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

        start_id = (
            current_state_id
            if (current_state_id and current_state_id in pool_ids)
            else pool_ids[active_indices[0]]
        )

        steered_target = self.steer_semantic_probe_full_rank(
            composed_base, device=device, attention_mask_1x128=composed_mask
        )
        return (
            start_id,
            composite_target_id,
            steered_target,
            composed_mask,
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
            or (self.eeg_wm_gain <= 0.001 and not self.is_npc_binding())
        ):
            return base_probe_1x128x4096

        wm_128x64, inertia, _ = self.extract_whitened_wm_modes_128x64(
            device, sub_step_frac=sub_step_frac
        )
        if float(wm_128x64.abs().max().item()) < 1e-5:
            return base_probe_1x128x4096

        base_tokens = base_probe_1x128x4096[0].float()
        ref_norms = torch.norm(base_tokens, p=2, dim=-1, keepdim=True).clamp(min=1e-4)

        if attention_mask_1x128 is not None:
            active_mask = attention_mask_1x128[0].to(device=device, dtype=torch.float32).unsqueeze(-1)
        else:
            pad_ref = base_tokens[-1:]
            tok_diff = torch.norm(base_tokens - pad_ref, p=2, dim=-1)
            active_mask = (tok_diff > 1e-3).unsqueeze(-1).float()
        active_mask[0] = 0.0

        if float(active_mask.sum().item()) < 1.0:
            return base_probe_1x128x4096

        tok_centered = base_tokens - base_tokens.mean(dim=0, keepdim=True)
        q_mat, _ = torch.linalg.qr(tok_centered[:64].T, mode="reduced")
        q_basis = q_mat.T
        M = q_basis.shape[0]

        wm_sem_128x4096 = torch.matmul(wm_128x64[:, :M], q_basis)

        dot_proj = torch.sum(wm_sem_128x4096 * base_tokens, dim=-1, keepdim=True) / (
            ref_norms**2 + 1e-6
        )
        tangent_raw = (wm_sem_128x4096 - dot_proj * base_tokens) * active_mask
        t_norms = torch.norm(tangent_raw, p=2, dim=-1, keepdim=True)
        tangent_unit = torch.where(
            t_norms > 1e-5,
            (tangent_raw / (t_norms + 1e-6)) * ref_norms,
            torch.zeros_like(base_tokens),
        )

        theta_rad = (
            (1.0 - inertia * 0.4)
            * (math.pi * 0.08)
            * float(self.wm_radius)
            * (1.0 - float(self.stochastic))
            * float(self.eeg_wm_gain if not self.is_npc_binding() else 0.35)
        )
        steered_tokens = (
            torch.cos(theta_rad * active_mask) * base_tokens
            + torch.sin(theta_rad * active_mask) * tangent_unit
        )
        steered_normalized = F.normalize(steered_tokens, p=2, dim=-1) * ref_norms

        return steered_normalized.unsqueeze(0).to(base_probe_1x128x4096.dtype)

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
        if not pool:
            return []
        return list(pool)
