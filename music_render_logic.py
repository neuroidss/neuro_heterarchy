#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: REALTIME B=1 MUSIC RENDER LOGIC (ANTI-GARGLE & FLOAT32 ENCODEC)
===================================================================================
- EnCodec (`compression_model`) всегда остаётся в float32 (как в оригинальном
  AudioCraft) — исключены насыщения/трески LSTM в fp16.
- Сохранена полная энтропия сэмплинга (1.0 * g_det_base, без сжатия температуры
  в 0.65x) + добавлен мягкий детектор зацикливания кодбуков c2/c3 (период 1..4
  кадра >= 5 шагов подряд), мгновенно прерывающий акустическое "булькание" за 20 мс.
===================================================================================
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import time
import numpy as np
import torch
import torch.nn.functional as F

from audiocraft.models import MusicGen
from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import set_efficient_attention_backend

from neuro_vram_bus import HardwareVRAMBus
from musicgen_fast_kernels import (
    DEVICE,
    apply_audiocraft_fast_patches,
    prepare_lm_fast_weights,
    encode_clean_64_t5,
    set_active_t5_embeds,
    invalidate_cross_attention_cache,
    ExactStreamingEncodecDecoder,
)

SAMPLE_RATE = 32000
FRAME_RATE = 50
MIN_BLOCK_SEC = 1.0 / float(FRAME_RATE)  # 0.02s (1 токен = 20 мс)
MAX_IN_DIST_KV_POS = 1200                # 24 сек внутри 30-сек (1500) окна обучения MusicGen


class NeuroMusicRenderer:
    def __init__(self, model_name: str = "facebook/musicgen-small",
                 stochastic: float = 0.0, unassigned_stochastic: float = 1.0,
                 wm_radius: float = 0.25, run_benchmark: bool = True):
        self.device = DEVICE
        self.model_name = model_name
        self.stochastic = float(np.clip(stochastic, 0.0, 1.0))
        self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        self.wm_radius = float(np.clip(wm_radius, 0.0, 1.0))

        apply_audiocraft_fast_patches()
        set_efficient_attention_backend('torch')

        self.mg: Optional[MusicGen] = None
        self.t5_cond = None
        self.proj_120_to_768: Optional[torch.Tensor] = None
        self.global_svd_bases_64x768: Optional[torch.Tensor] = None
        self.empty_embed_t: Optional[torch.Tensor] = None
        self.empty_embed_64x768: Optional[np.ndarray] = None

        self.stream_decoder: Optional[ExactStreamingEncodecDecoder] = None
        self.vram_bus = HardwareVRAMBus()

        self.current_diag_seq: Optional[torch.Tensor] = None
        self.pending_stream_token: Optional[torch.Tensor] = None
        self._stream_ctx = None
        self._stream_active: bool = False
        self.stream_kv_pos: int = 0
        self.smooth_wm_drift: Optional[torch.Tensor] = None
        self.absolute_token_step: int = 0
        self.cached_tokenized = None
        self.current_block = 0.16
        self.ema_rtf = 1.8

        g_proj = torch.Generator(device=self.device).manual_seed(777)
        self.wm_600_to_4x50 = F.normalize(
            torch.randn((4, 600, 50), generator=g_proj, device=self.device, dtype=torch.float32),
            p=2, dim=1
        )

        g_det = torch.Generator(device=self.device).manual_seed(424242)
        u_det = torch.empty((8192, 4, 50), device=self.device, dtype=torch.float32).uniform_(
            1e-6, 1.0 - 1e-6, generator=g_det
        )
        self.det_gumbel_table = (-torch.log(-torch.log(u_det))).contiguous()
        self._cb_idx_4 = torch.arange(4, device=self.device, dtype=torch.long)
        self._cb_delay_4 = torch.arange(4, device=self.device, dtype=torch.long)

        self.last_profile = {
            "t5_ms": 0.0,
            "lm_loop_ms": 0.0,
            "per_step_ms": 0.0,
            "encodec_ms": 0.0,
            "steps_count": 0,
            "ctx_tokens": 0
        }

        self._load_model(model_name, is_initial=True)
        if run_benchmark:
            self.run_startup_diagnostics()

    def _close_lm_streaming(self):
        if self._stream_active and self._stream_ctx is not None:
            try:
                self._stream_ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._stream_ctx = None
        self._stream_active = False
        self.stream_kv_pos = 0

    def _open_lm_streaming(self, lm: LMModel):
        self._close_lm_streaming()
        self._stream_ctx = lm.streaming()
        self._stream_ctx.__enter__()
        self._stream_active = True
        self.stream_kv_pos = 0

    def _load_model(self, model_name: str, is_initial: bool = False):
        self._close_lm_streaming()
        if self.mg is not None:
            del self.mg
            torch.cuda.empty_cache()

        self.model_name = model_name
        self.mg = MusicGen.get_pretrained(model_name, device=self.device)
        self.mg.lm.eval()
        self.mg.compression_model.eval()
        if self.device.type == 'cuda':
            self.mg.lm.to(torch.float16)
            # EnCodec (LSTM + ConvTranspose) должен оставаться в float32 против треска/насыщения:
            self.mg.compression_model.to(torch.float32)

        prepare_lm_fast_weights(self.mg.lm)
        self.t5_cond = self.mg.lm.condition_provider.conditioners['description']
        self.stream_decoder = ExactStreamingEncodecDecoder(
            self.mg.compression_model, pad_tokens=32, lookahead_tokens=8
        )
        self.cached_tokenized = None
        self.reset_stream()

        if is_initial:
            raw_emb = self.t5_cond.t5.shared.weight.detach().to(torch.float32)
            mean_tok_norm = float(torch.norm(raw_emb, p=2, dim=-1).mean().item())
            norm_emb = raw_emb / (torch.norm(raw_emb, dim=-1, keepdim=True) + 1e-7)
            _, _, V_all = torch.pca_lowrank(norm_emb, q=120)
            self.proj_120_to_768 = V_all[:, :120].T.contiguous().to(self.device)
            self.global_svd_bases_64x768 = (
                F.normalize(V_all[:, :64].T.contiguous(), p=2, dim=-1) * mean_tok_norm
            ).to(self.device)

            with torch.inference_mode(), self.t5_cond.autocast:
                self.empty_embed_t = encode_clean_64_t5(self.t5_cond, "")
                self.empty_embed_64x768 = self.empty_embed_t.cpu().numpy()

    def switch_model(self, new_model_name: str) -> str:
        clean_name = str(new_model_name or "facebook/musicgen-small").strip()
        print(f"🔄 [MUSIC-RENDERER] Смена модели на: {clean_name}...", flush=True)
        self._load_model(clean_name, is_initial=False)
        return clean_name

    def reset_stream(self):
        self._close_lm_streaming()
        self.current_diag_seq = None
        self.pending_stream_token = None
        self.absolute_token_step = 0
        if self.stream_decoder:
            self.stream_decoder.reset()
        self.ema_rtf = 1.8
        self.current_block = 0.16

    @torch.no_grad()
    def encode_lore_prompts(self, prompts: List[str]) -> dict:
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        encoded_bases = []
        with torch.inference_mode(), self.t5_cond.autocast:
            for p in cleaned:
                encoded_bases.append(encode_clean_64_t5(self.t5_cond, p).cpu().numpy())
        return {
            'status': 'ok',
            'c_bases': encoded_bases,
            'empty_embed': self.empty_embed_64x768,
            'global_svd_bases': self.global_svd_bases_64x768.cpu().numpy(),
            'proj_120_to_768': self.proj_120_to_768.cpu().numpy()
        }

    @torch.no_grad()
    def _build_full_rank_wm_bias_4xK(
        self,
        wm_5x5x32x120: torch.Tensor,
        active_codebook_mask_4: Optional[torch.Tensor],
        k_eff: int,
        dt_sec: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w_f = wm_5x5x32x120 if (wm_5x5x32x120.device == device and wm_5x5x32x120.dtype == torch.float32) else wm_5x5x32x120.to(device=device, dtype=torch.float32)
        lerp_alpha = float(min(0.35, max(0.01, dt_sec * 0.25)))
        if self.smooth_wm_drift is None or self.smooth_wm_drift.shape != w_f.shape:
            self.smooth_wm_drift = w_f.clone()
        else:
            self.smooth_wm_drift.lerp_(w_f, lerp_alpha)

        dyn_wm = w_f - self.smooth_wm_drift * 0.85
        wm_energy_t = torch.norm(dyn_wm)
        sig_gate_t = torch.tanh(wm_energy_t * 5.0)

        dyn_5x32x600 = dyn_wm.permute(0, 2, 1, 3).reshape(5, 32, 600)

        if active_codebook_mask_4 is not None:
            m4 = active_codebook_mask_4.to(device=device, dtype=torch.float32)
            reg_act_5 = torch.stack([m4[0], m4[3], m4[1], m4[2], m4[0]], dim=0)
        else:
            reg_act_5 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        has_any = torch.any(reg_act_5 > 0.5)
        reg_act_5 = torch.where(has_any, reg_act_5, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device=device))

        reg_w_5 = torch.where(reg_act_5 > 0.5, torch.tensor(3.0, device=device), torch.tensor(0.35, device=device))
        reg_w_5 = (reg_w_5 / reg_w_5.sum()).view(5, 1, 1)

        global_32x600 = torch.sum(dyn_5x32x600 * reg_w_5, dim=0)
        cb0_32x600 = global_32x600 * 0.65 + (dyn_5x32x600[0] * 0.5 + dyn_5x32x600[4] * 0.5) * 0.35
        cb1_32x600 = global_32x600 * 0.65 + dyn_5x32x600[2] * 0.35
        cb2_32x600 = global_32x600 * 0.65 + dyn_5x32x600[3] * 0.35
        cb3_32x600 = global_32x600 * 0.65 + dyn_5x32x600[1] * 0.35
        cb_4x600 = torch.stack([cb0_32x600, cb1_32x600, cb2_32x600, cb3_32x600], dim=0).mean(dim=1)

        raw_bias_4xK = torch.bmm(cb_4x600.unsqueeze(1), self.wm_600_to_4x50[:, :, :k_eff]).squeeze(1)
        b_std = raw_bias_4xK.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5)
        norm_bias_4xK = (raw_bias_4xK - raw_bias_4xK.mean(dim=-1, keepdim=True)) / b_std
        return norm_bias_4xK.contiguous(), sig_gate_t

    @torch.no_grad()
    def _detect_buzz_loop(self, recent_seq: torch.Tensor) -> bool:
        """
        Детектирует акустическое булькание/треск (точный повтор коротких циклов периода 1..4
        в верхних кодбуках c2 и c3 на протяжении >= 5 шагов подряд).
        """
        if recent_seq.shape[-1] < 12:
            return False
        tail = recent_seq[0, 2:4, -12:]  # [2, 12]
        for p in (1, 2, 3, 4):
            # Проверяем последние 5 токенов на совпадение с шагом t-p в обоих кодбуках c2 и c3
            if torch.all(tail[:, -5:] == tail[:, -5 - p : -p]):
                return True
        return False

    @torch.no_grad()
    def step_continuous_diagonal_stream(
        self,
        conditions: list,
        gen_frames: int,
        ctx_frames: int,
        temp: float = 1.0,
        top_k: int = 250,
        stochastic: float = 0.0,
        unassigned_stochastic: float = 1.0,
        wm_radius: float = 0.25,
        wm_5x5x32x120: Optional[torch.Tensor] = None,
        active_codebook_mask_4: Optional[torch.Tensor] = None,
        deterministic_seed_lock: bool = False
    ) -> torch.Tensor:
        lm: LMModel = self.mg.lm
        t_start_t5 = time.perf_counter()
        device = next(iter(lm.parameters())).device

        if conditions:
            if self.cached_tokenized is None:
                self.cached_tokenized = lm.condition_provider.tokenize(conditions)
            invalidate_cross_attention_cache(lm)
            cfg_conditions = lm.condition_provider(self.cached_tokenized)
            cross_attn_src = cfg_conditions['description'][0]
        else:
            cross_attn_src = None

        if not hasattr(lm, '_fused_head_w_T'):
            prepare_lm_fast_weights(lm)

        self.last_profile["t5_ms"] = (time.perf_counter() - t_start_t5) * 1000.0
        t_start_lm = time.perf_counter()

        B, K = 1, lm.num_codebooks
        special_id = lm.special_token_id
        k_eff = max(1, min(int(top_k), 50))
        inv_temp = 1.0 / max(0.05, float(temp))
        stoch_act = float(np.clip(stochastic, 0.0, 1.0))
        stoch_unass = float(np.clip(unassigned_stochastic, 0.0, 1.0))
        rad = float(np.clip(wm_radius, 0.0, 1.0))

        if active_codebook_mask_4 is not None:
            cb_mask = active_codebook_mask_4.to(device=device, dtype=torch.float32).view(1, 4, 1)
            has_any_cb = torch.any(cb_mask > 0.5)
            eff_mask = torch.where(has_any_cb, torch.clamp(cb_mask + 0.75, 0.0, 1.0), torch.tensor(1.0, device=device))
            cb_stoch_1x4x1 = eff_mask * stoch_act + (1.0 - eff_mask) * stoch_unass
        else:
            cb_stoch_1x4x1 = torch.full((1, 4, 1), stoch_act, device=device, dtype=torch.float32)

        min_stoch = min(stoch_act, stoch_unass)
        gen_frames = max(1, int(gen_frames))
        keep_ctx = max(16, int(ctx_frames))

        wm_bias_4xK = None
        sig_gate_t = None
        if wm_5x5x32x120 is not None and min_stoch < 0.999:
            wm_bias_4xK, sig_gate_t = self._build_full_rank_wm_bias_4xK(
                wm_5x5x32x120, active_codebook_mask_4, k_eff,
                dt_sec=float(gen_frames) / float(FRAME_RATE), device=device
            )

        is_first_boot = (self.current_diag_seq is None) or (self.pending_stream_token is None)
        lookahead_extra = self.stream_decoder.lookahead_tokens if self.stream_decoder is not None else 8

        fused_emb_w = lm._fused_emb_weight
        emb_offsets = lm._emb_offsets
        head_w_T = lm._fused_head_w_T
        head_b = lm._fused_head_b

        if is_first_boot:
            self._open_lm_streaming(lm)
            steps_to_run = gen_frames + lookahead_extra + 4
            work_seq = torch.full((B, K, steps_to_run + 1), special_id, dtype=torch.long, device=device)
            start_step = 1
            self.last_profile["ctx_tokens"] = 0
        else:
            if (not self._stream_active) or (self.stream_kv_pos + gen_frames > MAX_IN_DIST_KV_POS):
                self._open_lm_streaming(lm)
                ctx_slice = self.current_diag_seq[..., -keep_ctx:]
                prefill_seq = ctx_slice[..., :-1]
                if prefill_seq.shape[-1] > 0:
                    prefill_in = F.embedding(prefill_seq + emb_offsets, fused_emb_w).sum(dim=1)
                    _ = lm.transformer(prefill_in, cross_attention_src=cross_attn_src)
                    self.stream_kv_pos = int(prefill_seq.shape[-1])
                self.pending_stream_token = ctx_slice[..., -1:].contiguous()

            self.last_profile["ctx_tokens"] = self.stream_kv_pos
            work_seq = torch.zeros((B, K, 1 + gen_frames), dtype=torch.long, device=device)
            work_seq[..., 0:1] = self.pending_stream_token
            start_step = 1
            steps_to_run = gen_frames

        total_seq_len = work_seq.shape[-1]

        for offset in range(start_step, total_seq_len):
            curr_abs_t = self.absolute_token_step
            curr_sequence = work_seq[..., offset - 1:offset]
            input_ = F.embedding(curr_sequence + emb_offsets, fused_emb_w).sum(dim=1)

            out = lm.transformer(input_, cross_attention_src=cross_attn_src)
            self.stream_kv_pos += 1
            if lm.out_norm is not None:
                out = lm.out_norm(out)

            out_last = out[:, -1, :].unsqueeze(0).expand(4, B, -1)
            if head_b is not None:
                logits = torch.baddbmm(head_b.unsqueeze(1), out_last, head_w_T)
            else:
                logits = torch.bmm(out_last, head_w_T)

            logits = logits.permute(1, 0, 2)
            if temp > 0.0:
                top_logits, top_indices = torch.topk(logits * inv_temp, k=k_eff, dim=-1)

                t_cb = (curr_abs_t - self._cb_delay_4).clamp(min=0) % 8192
                if deterministic_seed_lock:
                    g_rand = self.det_gumbel_table[t_cb, self._cb_idx_4, :k_eff].unsqueeze(0).to(top_logits.dtype)
                else:
                    u_rand = torch.empty_like(top_logits).uniform_(1e-6, 1.0 - 1e-6)
                    g_rand = -torch.log(-torch.log(u_rand))

                if min_stoch >= 0.999 or wm_bias_4xK is None:
                    score = top_logits + g_rand
                else:
                    # Сохраняем полную энтропию Gumbel (1.0 * g_det_base), не сжимая температуру!
                    g_det_base = self.det_gumbel_table[t_cb, self._cb_idx_4, :k_eff].unsqueeze(0).to(top_logits.dtype)
                    wm_bias = wm_bias_4xK.unsqueeze(0).to(top_logits.dtype)
                    eff_mod = (rad * sig_gate_t).to(top_logits.dtype)
                    det_carrier = g_det_base + wm_bias * (0.40 * eff_mod)
                    s_mask = cb_stoch_1x4x1.to(top_logits.dtype)
                    score = top_logits + (s_mask * g_rand) + ((1.0 - s_mask) * det_carrier)

                # Анти-цикл от "ьрьрьрьр" булькания: если верхние кодбуки застряли в 1..4-шаговой петле,
                # сбиваем петлю свежим случайным сэмплом (не затрагивая тесты с deterministic_seed_lock):
                if (not deterministic_seed_lock) and (offset >= 12 or self.current_diag_seq is not None):
                    hist_check = work_seq[..., :offset] if offset >= 12 else (
                        torch.cat([self.current_diag_seq[..., -12:], work_seq[..., 1:offset]], dim=-1)
                        if self.current_diag_seq is not None else work_seq[..., :offset]
                    )
                    if self._detect_buzz_loop(hist_check):
                        u_esc = torch.empty_like(top_logits).uniform_(1e-6, 1.0 - 1e-6)
                        g_esc = -torch.log(-torch.log(u_esc))
                        last_toks = hist_check[:, :, -1:]
                        rep_mask = (top_indices == last_toks).to(top_logits.dtype)
                        score = top_logits + g_esc - rep_mask * 5.0

                sample_idx = torch.argmax(score, dim=-1, keepdim=True)
                next_token = torch.gather(top_indices, -1, sample_idx)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if is_first_boot and offset < 4:
                for k_i in range(K):
                    if offset <= k_i:
                        next_token[:, k_i, :] = special_id

            work_seq[..., offset:offset + 1] = next_token
            self.absolute_token_step += 1

        lm_ms = (time.perf_counter() - t_start_lm) * 1000.0
        self.last_profile["lm_loop_ms"] = lm_ms
        self.last_profile["steps_count"] = max(1, steps_to_run)
        self.last_profile["per_step_ms"] = lm_ms / max(1, steps_to_run)

        self.pending_stream_token = work_seq[..., -1:].contiguous()

        if is_first_boot:
            self.current_diag_seq = work_seq[..., -1500:].contiguous()
            n_extract = steps_to_run - 4
        else:
            new_toks = work_seq[..., 1:]
            self.current_diag_seq = torch.cat([self.current_diag_seq, new_toks], dim=-1)[..., -1500:].contiguous()
            n_extract = gen_frames

        seq_len_hist = self.current_diag_seq.shape[-1]
        s_start = seq_len_hist - n_extract

        c0 = self.current_diag_seq[:, 0, s_start - 3: seq_len_hist - 3]
        c1 = self.current_diag_seq[:, 1, s_start - 2: seq_len_hist - 2]
        c2 = self.current_diag_seq[:, 2, s_start - 1: seq_len_hist - 1]
        c3 = self.current_diag_seq[:, 3, s_start:     seq_len_hist]
        ready_audio_codes = torch.stack([c0, c1, c2, c3], dim=1).clamp(0, lm.card - 1)
        return ready_audio_codes

    def adapt_block_and_pacing(
        self,
        gen_time: float,
        raw_audio_duration: float,
        current_block: float,
        user_target_block: float,
        queue_sec: float,
        queue_chunks: int = 0,
        target_buffer_count: int = 2,
        safety_factor: float = 1.0
    ) -> Tuple[float, float, float]:
        gen_time = max(gen_time, 0.001)
        real_rtf = raw_audio_duration / gen_time
        self.ema_rtf = self.ema_rtf * 0.70 + real_rtf * 0.30
        eff_rtf = min(real_rtf, self.ema_rtf)

        target_buf = max(1, int(target_buffer_count))
        play_speed = 1.0

        if user_target_block <= 0.0:
            max_auto_block = 0.80
            if queue_chunks >= target_buf + 1 and real_rtf > 1.25:
                shrink = 0.85 if real_rtf > 1.60 else 0.92
                current_block = max(MIN_BLOCK_SEC, current_block * shrink)
            elif queue_chunks == 0 or real_rtf < 0.95:
                grow_step = max(MIN_BLOCK_SEC, current_block * 0.25)
                current_block = min(max_auto_block, current_block + grow_step)

            current_block = float(np.clip(current_block, MIN_BLOCK_SEC, max_auto_block))
        else:
            current_block = max(MIN_BLOCK_SEC, float(user_target_block))

        if eff_rtf < 0.90:
            margin = 0.92 if queue_chunks < target_buf else 0.96
            target_speed = (eff_rtf * margin) / max(0.5, float(safety_factor))
            play_speed = float(np.clip(target_speed, 0.40, 1.0))
        else:
            play_speed = 1.0

        return current_block, real_rtf, play_speed

    @torch.no_grad()
    def generate_step(self, msg: dict) -> Tuple[dict, float]:
        t0 = time.perf_counter()
        embeds_np = msg.get('t5_embeds')
        wm_np = msg.get('wm_5x5x32x120')
        cb_mask_np = msg.get('active_codebook_mask_4')
        user_target_block = float(msg.get('block_sec', 0.0))
        context_sec = float(msg.get('context_sec', 3.6))
        temp = float(msg.get('temp', 1.0))
        top_k = int(msg.get('top_k', 250))
        safety_factor = float(msg.get('safety_factor', 1.0))
        queue_sec = float(msg.get('queue_sec', 0.0))
        queue_chunks = int(msg.get('queue_chunks', 0))
        target_buffer_count = int(msg.get('target_buffer_count', 2))
        is_reset = bool(msg.get('is_reset', False))
        lock_speed = bool(msg.get('lock_speed', False))

        stoch = float(msg.get('stochastic', self.stochastic))
        unass_stoch = float(msg.get('unassigned_stochastic', self.unassigned_stochastic))
        rad = float(msg.get('wm_radius', self.wm_radius))

        self.vram_bus.try_attach_to_existing_vram()
        has_vram_bus = (self.vram_bus.attached_pid > 0)

        if embeds_np is not None:
            set_active_t5_embeds(torch.from_numpy(embeds_np).to(self.device))
        elif has_vram_bus:
            set_active_t5_embeds(self.vram_bus.live_music_64x768)

        if wm_np is not None:
            wm_tensor = torch.from_numpy(wm_np).to(self.device, dtype=torch.float32)
        elif has_vram_bus:
            wm_tensor = self.vram_bus.wm_field_5x5x32x120
        else:
            wm_tensor = None

        cb_mask_t = None
        if cb_mask_np is not None:
            cb_mask_t = torch.tensor(cb_mask_np, dtype=torch.float32, device=self.device)

        if is_reset:
            self.reset_stream()
            self.current_block = max(MIN_BLOCK_SEC, user_target_block) if user_target_block > 0.0 else 0.16

        if user_target_block > 0.0:
            self.current_block = max(MIN_BLOCK_SEC, user_target_block)

        gen_tokens = max(1, int(round(self.current_block * FRAME_RATE)))
        ctx_frames = max(16, int(round(context_sec * FRAME_RATE)))
        attributes = [self.mg._prepare_tokens_and_attributes([""], None)[0][0]]

        with torch.inference_mode(), self.mg.autocast:
            ready_codes = self.step_continuous_diagonal_stream(
                conditions=attributes,
                gen_frames=gen_tokens,
                ctx_frames=ctx_frames,
                temp=temp,
                top_k=top_k,
                stochastic=stoch,
                unassigned_stochastic=unass_stoch,
                wm_radius=rad,
                wm_5x5x32x120=wm_tensor,
                active_codebook_mask_4=cb_mask_t
            )

        # EnCodec декодирование строго в float32 вне fp16 autocast:
        with torch.inference_mode():
            t_enc = time.perf_counter()
            pcm_tensor = self.stream_decoder.decode_exact_tokens(ready_codes)
            pcm_clean = torch.tanh(pcm_tensor * 0.95)
            self.last_profile["encodec_ms"] = (time.perf_counter() - t_enc) * 1000.0

        raw_audio_len = gen_tokens / float(FRAME_RATE)
        gen_time = time.perf_counter() - t0

        if lock_speed:
            real_rtf = raw_audio_len / max(gen_time, 0.001)
            play_speed = 1.0
        else:
            self.current_block, real_rtf, play_speed = self.adapt_block_and_pacing(
                gen_time=gen_time,
                raw_audio_duration=raw_audio_len,
                current_block=self.current_block,
                user_target_block=user_target_block,
                queue_sec=queue_sec,
                queue_chunks=queue_chunks,
                target_buffer_count=target_buffer_count,
                safety_factor=safety_factor
            )

        if play_speed < 0.995 and not lock_speed:
            orig_samples = pcm_clean.shape[-1]
            stretched_samples = max(orig_samples, int(round(orig_samples / play_speed)))
            pcm_stretched = F.interpolate(
                pcm_clean.float(), size=stretched_samples, mode='linear', align_corners=False
            )
            pcm = pcm_stretched[0, 0].cpu().numpy()
        else:
            pcm = pcm_clean[0, 0].cpu().float().numpy()

        resp = {
            'pcm': pcm,
            'gen_time': gen_time,
            'audio_len': len(pcm) / float(SAMPLE_RATE),
            'rtf': real_rtf,
            'play_speed': play_speed,
            'actual_block': raw_audio_len
        }
        return resp, 0.0

    def run_startup_diagnostics(self):
        print("\n" + "=" * 90)
        print("🔬 [ТЕСТ ИНВАРИАНТНОСТИ И СТЫКОВКИ БЛОКОВ: 0.1s x 12 VS 1.2s x 1]")
        print("=" * 90)
        set_active_t5_embeds(self.empty_embed_t)
        self.cached_tokenized = None
        attributes = [self.mg._prepare_tokens_and_attributes([""], None)[0][0]]

        self.reset_stream()
        tokens_small_list = []
        pcm_small_list = []
        for _ in range(12):
            with torch.inference_mode(), self.mg.autocast:
                toks = self.step_continuous_diagonal_stream(
                    attributes, gen_frames=5, ctx_frames=360,
                    stochastic=1.0, deterministic_seed_lock=True
                )
            tokens_small_list.append(toks)
            with torch.inference_mode():
                pcm_small_list.append(self.stream_decoder.decode_exact_tokens(toks))
        full_tokens_small = torch.cat(tokens_small_list, dim=-1)
        full_pcm_small = torch.cat(pcm_small_list, dim=-1)[0, 0]

        self.reset_stream()
        with torch.inference_mode(), self.mg.autocast:
            full_tokens_large = self.step_continuous_diagonal_stream(
                attributes, gen_frames=60, ctx_frames=360,
                stochastic=1.0, deterministic_seed_lock=True
            )
        with torch.inference_mode():
            full_pcm_large = self.stream_decoder.decode_exact_tokens(full_tokens_large)[0, 0]

        is_exact_tokens = torch.equal(full_tokens_small, full_tokens_large)
        min_len = min(len(full_pcm_small), len(full_pcm_large))
        seam_indices = [k * 3200 for k in range(1, 12)]
        max_seam_err = max(
            float(torch.max(torch.abs(full_pcm_small[idx - 16:idx + 16] - full_pcm_large[idx - 16:idx + 16])).item())
            for idx in seam_indices if idx + 16 <= min_len
        )

        print(f"👉 1. Токены (0.1s x 12 vs 1.2s x 1):                {'✅ 100% ПОБИТОВО ИДЕНТИЧНО' if is_exact_tokens else '❌ РАЗЛИЧАЮТСЯ'}")
        print(f"👉 2. Погрешность волны PCM на 11 стыках блоков:     max_err = {max_seam_err:.5f} {'✅ БЕЗ СКЛЕЕК' if max_seam_err < 0.05 else '⚠️ ЗАМЕТЕН СТЫК'}")
        print("-" * 90)
        print(f"{'Режим (Block / Context)':<28} | {'CtxТок':<6} | {'Шагов':<6} | {'LM Loop':<9} | {'1 Шаг':<8} | {'EnCodec':<7} | {'ЧИСТЫЙ RTF'}")
        print("-" * 90)

        for b_sec, c_sec in [(0.02, 3.6), (0.10, 3.6), (0.20, 3.6), (0.50, 3.6), (1.20, 3.6)]:
            self.reset_stream()
            gen_f = max(1, int(round(b_sec * FRAME_RATE)))
            ctx_f = max(16, int(round(c_sec * FRAME_RATE)))
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), self.mg.autocast:
                codes = self.step_continuous_diagonal_stream(attributes, gen_frames=gen_f, ctx_frames=ctx_f, stochastic=self.stochastic)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t_enc0 = time.perf_counter()
            with torch.inference_mode():
                _ = self.stream_decoder.decode_exact_tokens(codes)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t_enc0) * 1000.0
            total_s = time.perf_counter() - t0
            rtf = (gen_f / float(FRAME_RATE)) / total_s
            print(
                f"Block {b_sec:4.2f}s / Ctx {c_sec:4.2f}s    | "
                f"{self.last_profile['ctx_tokens']:<6d} | "
                f"{self.last_profile['steps_count']:<6d} | "
                f"{self.last_profile['lm_loop_ms']:6.1f} ms | "
                f"{self.last_profile['per_step_ms']:5.2f} ms | "
                f"{enc_ms:4.1f} ms | "
                f"👉 {rtf:5.2f}x"
            )
        print("=" * 90 + "\n", flush=True)
        self.reset_stream()
