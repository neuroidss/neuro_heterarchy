#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: REALTIME B=1 MUSIC RENDER LOGIC (CLEAN DUAL STOCHASTIC)
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
MIN_BLOCK_SEC = 1.0 / float(FRAME_RATE)


class NeuroMusicRenderer:
    def __init__(self, model_name: str = "facebook/musicgen-small",
                 stochastic: float = 0.0, unassigned_stochastic: float = 1.0,
                 run_benchmark: bool = True):
        self.device = DEVICE
        self.model_name = model_name
        self.stochastic = float(np.clip(stochastic, 0.0, 1.0))
        self.unassigned_stochastic = float(np.clip(unassigned_stochastic, 0.0, 1.0))

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
        self.absolute_token_step: int = 0
        self.cached_tokenized = None
        self.current_block = 0.20
        self.ema_rtf = 1.8

        g_proj = torch.Generator(device=self.device).manual_seed(777)
        self.wm_to_k50 = F.normalize(
            torch.randn((120, 50), generator=g_proj, device=self.device, dtype=torch.float32),
            p=2, dim=0
        )

        g_det = torch.Generator(device=self.device).manual_seed(424242)
        u_det = torch.empty((8192, 4, 50), device=self.device, dtype=torch.float32).uniform_(1e-6, 1.0 - 1e-6, generator=g_det)
        self.det_gumbel_table = (-torch.log(-torch.log(u_det))).contiguous()

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

    def _load_model(self, model_name: str, is_initial: bool = False):
        if self.mg is not None:
            del self.mg
            torch.cuda.empty_cache()

        self.model_name = model_name
        self.mg = MusicGen.get_pretrained(model_name, device=self.device)
        self.mg.lm.eval()
        self.mg.compression_model.eval()
        if self.device.type == 'cuda':
            self.mg.lm.to(torch.float16)
            self.mg.compression_model.to(torch.float16)

        prepare_lm_fast_weights(self.mg.lm)
        self.t5_cond = self.mg.lm.condition_provider.conditioners['description']
        self.stream_decoder = ExactStreamingEncodecDecoder(self.mg.compression_model, pad_tokens=48)
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
        self.current_diag_seq = None
        self.absolute_token_step = 0
        if self.stream_decoder:
            self.stream_decoder.reset()
        self.ema_rtf = 1.8
        self.current_block = 0.20

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
    def step_continuous_diagonal_stream(
        self,
        conditions: list,
        gen_frames: int,
        ctx_frames: int,
        temp: float = 1.0,
        top_k: int = 250,
        stochastic: float = 0.0,
        unassigned_stochastic: float = 1.0,
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
        k_eff = min(int(top_k), 50)
        inv_temp = 1.0 / max(0.05, float(temp))
        stoch_act = float(np.clip(stochastic, 0.0, 1.0))
        stoch_unass = float(np.clip(unassigned_stochastic, 0.0, 1.0))

        if active_codebook_mask_4 is not None:
            cb_mask = active_codebook_mask_4.to(device=device, dtype=torch.float32).view(1, 4, 1)
            cb_stoch_1x4x1 = cb_mask * stoch_act + (1.0 - cb_mask) * stoch_unass
        else:
            cb_stoch_1x4x1 = torch.full((1, 4, 1), stoch_act, device=device, dtype=torch.float32)

        min_stoch = float(cb_stoch_1x4x1.min().item())
        wm_bias_32x4xK = None

        if wm_5x5x32x120 is not None and min_stoch < 0.999:
            w_f = wm_5x5x32x120.to(device=device, dtype=torch.float32)
            cb0 = w_f[0, 0] * 0.5 + w_f[4, 0] * 0.5
            cb1 = w_f[2, 1] * 0.5 + w_f[2, 2] * 0.5
            cb2 = w_f[3, 1] * 0.5 + w_f[3, 4] * 0.5
            cb3 = w_f[1, 3] * 0.6 + w_f[1, 4] * 0.4
            wm_4x32x120 = torch.stack([cb0, cb1, cb2, cb3], dim=0)

            raw_bias = torch.matmul(wm_4x32x120, self.wm_to_k50[:, :k_eff])
            b_std = raw_bias.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            norm_bias = (raw_bias - raw_bias.mean(dim=-1, keepdim=True)) / b_std
            wm_bias_32x4xK = norm_bias.permute(1, 0, 2).contiguous()

        gen_frames = max(1, int(gen_frames))
        is_first_boot = (self.current_diag_seq is None or self.current_diag_seq.shape[-1] < 4)

        if is_first_boot:
            steps_to_run = gen_frames + 4
            work_seq = torch.full((B, K, steps_to_run + 1), special_id, dtype=torch.long, device=device)
            start_step = 1
            self.last_profile["ctx_tokens"] = 0
        else:
            avail = self.current_diag_seq.shape[-1]
            keep_ctx = max(8, min(avail, ctx_frames))
            ctx_slice = self.current_diag_seq[..., -keep_ctx:]
            work_seq = torch.zeros((B, K, keep_ctx + gen_frames), dtype=torch.long, device=device)
            work_seq[..., :keep_ctx] = ctx_slice
            start_step = keep_ctx
            steps_to_run = gen_frames
            self.last_profile["ctx_tokens"] = keep_ctx

        total_seq_len = work_seq.shape[-1]
        fused_emb_w = lm._fused_emb_weight
        emb_offsets = lm._emb_offsets
        head_w_T = lm._fused_head_w_T
        head_b = lm._fused_head_b

        with lm.streaming():
            prev_offset = 0
            for offset in range(start_step, total_seq_len):
                curr_abs_t = self.absolute_token_step
                curr_sequence = work_seq[..., prev_offset:offset]
                input_ = F.embedding(curr_sequence + emb_offsets, fused_emb_w).sum(dim=1)

                out = lm.transformer(input_, cross_attention_src=cross_attn_src)
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

                    if deterministic_seed_lock:
                        g_rand = self.det_gumbel_table[curr_abs_t % 8192, :, :k_eff].unsqueeze(0).to(top_logits.dtype)
                    else:
                        u_rand = torch.empty_like(top_logits).uniform_(1e-6, 1.0 - 1e-6)
                        g_rand = -torch.log(-torch.log(u_rand))

                    if min_stoch >= 0.999 or wm_bias_32x4xK is None:
                        score = top_logits + g_rand
                    else:
                        wm_bias = wm_bias_32x4xK[curr_abs_t % 32].unsqueeze(0).to(top_logits.dtype)
                        s_mask = cb_stoch_1x4x1.to(top_logits.dtype)
                        score = top_logits + (s_mask * g_rand) + ((1.0 - s_mask) * wm_bias * 1.5)

                    sample_idx = torch.argmax(score, dim=-1, keepdim=True)
                    next_token = torch.gather(top_indices, -1, sample_idx)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                if is_first_boot and offset < 4:
                    for k_i in range(K):
                        if offset <= k_i:
                            next_token[:, k_i, :] = special_id

                work_seq[..., offset:offset + 1] = next_token
                prev_offset = offset
                self.absolute_token_step += 1

        lm_ms = (time.perf_counter() - t_start_lm) * 1000.0
        self.last_profile["lm_loop_ms"] = lm_ms
        self.last_profile["steps_count"] = max(1, steps_to_run)
        self.last_profile["per_step_ms"] = lm_ms / max(1, steps_to_run)

        n_extract = gen_frames if not is_first_boot else (steps_to_run - 4)
        s_start = total_seq_len - n_extract

        c0 = work_seq[:, 0, s_start - 3: total_seq_len - 3]
        c1 = work_seq[:, 1, s_start - 2: total_seq_len - 2]
        c2 = work_seq[:, 2, s_start - 1: total_seq_len - 1]
        c3 = work_seq[:, 3, s_start:     total_seq_len]
        ready_audio_codes = torch.stack([c0, c1, c2, c3], dim=1).clamp(0, lm.card - 1)

        self.current_diag_seq = work_seq[..., -450:].contiguous()
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
        self.ema_rtf = self.ema_rtf * 0.65 + real_rtf * 0.35
        eff_rtf = min(real_rtf, self.ema_rtf)

        target_buf = max(1, int(target_buffer_count))
        play_speed = 1.0

        if user_target_block <= 0.0:
            max_auto_block = 0.80
            if queue_chunks >= target_buf and real_rtf > 1.15:
                shrink = 0.85 if (queue_chunks > target_buf or real_rtf > 1.45) else 0.92
                current_block = max(MIN_BLOCK_SEC, current_block * shrink)
            elif queue_chunks == 0 or real_rtf < 1.02:
                grow_step = max(MIN_BLOCK_SEC, current_block * 0.25)
                current_block = min(max_auto_block, current_block + grow_step)

            current_block = float(np.clip(current_block, MIN_BLOCK_SEC, max_auto_block))
        else:
            current_block = max(MIN_BLOCK_SEC, float(user_target_block))

        if eff_rtf < 0.98:
            margin = 0.90 if queue_chunks < target_buf else 0.96
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

        if embeds_np is not None:
            set_active_t5_embeds(torch.from_numpy(embeds_np).to(self.device))

        self.vram_bus.try_attach_to_existing_vram()
        if wm_np is not None:
            wm_tensor = torch.from_numpy(wm_np).to(self.device, dtype=torch.float32)
        elif self.vram_bus.attached_pid > 0:
            wm_tensor = self.vram_bus.wm_field_5x5x32x120
        else:
            wm_tensor = None

        cb_mask_t = None
        if cb_mask_np is not None:
            cb_mask_t = torch.tensor(cb_mask_np, dtype=torch.float32, device=self.device)

        if is_reset:
            self.reset_stream()
            self.current_block = max(MIN_BLOCK_SEC, user_target_block) if user_target_block > 0.0 else 0.20

        if user_target_block > 0.0:
            self.current_block = max(MIN_BLOCK_SEC, user_target_block)

        gen_tokens = max(1, int(round(self.current_block * FRAME_RATE)))
        ctx_frames = max(10, int(round(context_sec * FRAME_RATE)))
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
                wm_5x5x32x120=wm_tensor,
                active_codebook_mask_4=cb_mask_t
            )

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
        print("🔬 [ТЕСТ СТЫКОВКИ БЛОКОВ ENCODEC: 0.1s x 12 VS 1.2s x 1]")
        print("=" * 90)
        set_active_t5_embeds(self.empty_embed_t)
        self.cached_tokenized = None
        attributes = [self.mg._prepare_tokens_and_attributes([""], None)[0][0]]

        self.reset_stream()
        pcm_small_list = []
        with torch.inference_mode(), self.mg.autocast:
            for _ in range(12):
                toks = self.step_continuous_diagonal_stream(
                    attributes, gen_frames=5, ctx_frames=360,
                    stochastic=1.0, deterministic_seed_lock=True
                )
                pcm_small_list.append(self.stream_decoder.decode_exact_tokens(toks))
        full_pcm_small = torch.cat(pcm_small_list, dim=-1)[0, 0]

        self.reset_stream()
        with torch.inference_mode(), self.mg.autocast:
            full_tokens_large = self.step_continuous_diagonal_stream(
                attributes, gen_frames=60, ctx_frames=360,
                stochastic=1.0, deterministic_seed_lock=True
            )
            full_pcm_large = self.stream_decoder.decode_exact_tokens(full_tokens_large)[0, 0]

        min_len = min(len(full_pcm_small), len(full_pcm_large))
        seam_indices = [k * 3200 for k in range(1, 12)]
        max_seam_err = max(
            float(torch.max(torch.abs(full_pcm_small[idx - 16:idx + 16] - full_pcm_large[idx - 16:idx + 16])).item())
            for idx in seam_indices if idx + 16 <= min_len
        )

        print(f"👉 Погрешность волны PCM на стыках блоков: max_err = {max_seam_err:.5f} {'✅ БЕЗ СКЛЕЕК' if max_seam_err < 0.08 else '⚠️ ЗАМЕТЕН СТЫК'}")
        print("-" * 90)
        print(f"{'Режим (Block / Context)':<28} | {'CtxТок':<6} | {'Шагов':<6} | {'LM Loop':<9} | {'1 Шаг':<8} | {'EnCodec':<7} | {'ЧИСТЫЙ RTF'}")
        print("-" * 90)

        for b_sec, c_sec in [(0.02, 3.6), (0.10, 3.6), (0.20, 3.6), (0.50, 3.6), (1.20, 3.6)]:
            gen_f = max(1, int(round(b_sec * FRAME_RATE)))
            ctx_f = max(10, int(round(c_sec * FRAME_RATE)))
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), self.mg.autocast:
                codes = self.step_continuous_diagonal_stream(attributes, gen_frames=gen_f, ctx_frames=ctx_f, stochastic=self.stochastic)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t_enc0 = time.perf_counter()
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
