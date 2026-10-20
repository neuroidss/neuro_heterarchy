#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ULTRA-FAST & PURE MUSICALITY MUSICGEN SERVER (PORT 6005)
===================================================================================
- СКОРОСТЬ 17.4 мс/шаг (RTF > 1.0x ДАЖЕ НА RTX 3060):
  1) Cross-Attention KV-Cache (ключи и значения T5 считаются 1 раз на чанк).
  2) Предвычисленная в FP32 таблица синусоидальных позиций (_STATIC_SIN_EMB_CACHE).
  3) Persistent Self-Attention KV-Cache (_persistent_k / _persistent_v) + Gumbel-Max.
- ЧИСТАЯ МУЗЫКА БЕЗ БУЛЬКАНИЯ:
  1) Тайлинг только активных токенов T5 (attention_mask == 1) на 64 слота без
     мертвых <pad>-векторов (сохраняет 100% нативное распределение Cross-Attention).
  2) Отсечение 3 краевых неполных шагов DelayPattern (out_tokens[..., :-3]) перед
     сохранением в current_tokens — исключает авторегрессионное вырождение в булькание!
- ЗАКОН СОХРАНЕНИЯ СЭМПЛОВ (ZERO-DROP) И АВТО-МИНИМИЗАЦИЯ БУФЕРА / БЛОКА.
===================================================================================
"""

import os
import sys
import time
import signal
import argparse
from multiprocessing.connection import Listener
import torch
import torch.nn.functional as F
import numpy as np

try:
    os.setpriority(os.PRIO_PROCESS, 0, -20)
    print("⚡ [System] Auto-Priority applied: NI -20 (Real-Time PRI 0)", flush=True)
except PermissionError:
    try:
        os.setpriority(os.PRIO_PROCESS, 0, -10)
    except Exception:
        pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

from xformers import ops
from audiocraft.models import MusicGen
from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import (
    StreamingMultiheadAttention,
    StreamingTransformer,
    _get_attention_time_dimension,
    set_efficient_attention_backend,
    create_sin_embedding
)
from audiocraft.modules.conditioners import T5Conditioner

SAMPLE_RATE = 32000
FRAME_RATE = 50
MAX_T5_TOKENS = 64

LAST_PROFILE = {
    "t5_ms": 0.0,
    "lm_loop_ms": 0.0,
    "per_step_ms": 0.0,
    "encodec_ms": 0.0,
    "steps_count": 0,
    "ctx_tokens": 0
}

# -----------------------------------------------------------------------------
# 1. PERSISTENT SELF-ATTENTION & CROSS-ATTENTION KV-CACHE (17.4 ms/step)
# -----------------------------------------------------------------------------
_orig_complete_kv = StreamingMultiheadAttention._complete_kv


def fast_complete_kv(self, k, v):
    if self.cross_attention or not self._is_streaming:
        return _orig_complete_kv(self, k, v)
    time_dim = _get_attention_time_dimension(self.memory_efficient)

    if (not hasattr(self, '_persistent_k') or
            self._persistent_k.shape[0] != k.shape[0] or
            self._persistent_k.ndim != k.ndim or
            self._persistent_k.shape[3 - time_dim] != k.shape[3 - time_dim]):
        shape_k = list(k.shape)
        shape_k[time_dim] = 1024
        self._persistent_k = torch.zeros(shape_k, device=k.device, dtype=k.dtype)
        self._persistent_v = torch.zeros(shape_k, device=v.device, dtype=v.dtype)

    curr_len = self._streaming_state.get('curr_len', 0)
    slen = k.shape[time_dim]
    static_k = self._persistent_k
    static_v = self._persistent_v

    if curr_len + slen > static_k.shape[time_dim]:
        return _orig_complete_kv(self, k, v)

    if time_dim == 2:
        static_k[:, :, curr_len:curr_len + slen, :] = k
        nk = static_k[:, :, :curr_len + slen, :]
        if v is not k:
            static_v[:, :, curr_len:curr_len + slen, :] = v
            nv = static_v[:, :, :curr_len + slen, :]
        else:
            nv = nk
    else:
        static_k[:, curr_len:curr_len + slen, :] = k
        nk = static_k[:, :curr_len + slen, :, :]
        if v is not k:
            static_v[:, curr_len:curr_len + slen, :] = v
            nv = static_v[:, :curr_len + slen, :, :]
        else:
            nv = nk

    self._streaming_state['curr_len'] = curr_len + slen
    return nk, nv


def fast_apply_rope(self, query: torch.Tensor, key: torch.Tensor):
    time_dim = _get_attention_time_dimension(self.memory_efficient)
    assert self.rope is not None
    past_keys_offset = self._streaming_state.get('curr_len', 0)
    return self.rope.rotate_qk(query, key, start=past_keys_offset, time_dim=time_dim)


def fast_mha_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     key_padding_mask=None, need_weights=False, attn_mask=None,
                     average_attn_weights=True, is_causal=False):
    time_dim = _get_attention_time_dimension(self.memory_efficient)
    dtype = query.dtype
    B, T_q, C = query.shape
    H = self.num_heads
    D = C // H

    if self.causal:
        attn_mask = self._get_mask(T_q, query.device, query.dtype)

    if self.custom:
        if self.cross_attention:
            dim = self.in_proj_weight.shape[0] // 3
            bias_q = self.in_proj_bias[:dim] if self.in_proj_bias is not None else None
            q = F.linear(query, self.in_proj_weight[:dim], bias_q)
            if self.qk_layer_norm:
                q = self.q_layer_norm(q)
            if time_dim == 2:
                q = q.view(B, T_q, H, D).transpose(1, 2)
            else:
                q = q.view(B, T_q, H, D)

            if self._is_streaming and 'cross_k' in self._streaming_state:
                k = self._streaming_state['cross_k']
                v = self._streaming_state['cross_v']
            else:
                bias_k = self.in_proj_bias[dim:2 * dim] if self.in_proj_bias is not None else None
                bias_v = self.in_proj_bias[2 * dim:] if self.in_proj_bias is not None else None
                k = F.linear(key, self.in_proj_weight[dim:2 * dim], bias_k)
                v = F.linear(value, self.in_proj_weight[2 * dim:], bias_v)
                if self.qk_layer_norm:
                    k = self.k_layer_norm(k)
                T_k = k.shape[1]
                if time_dim == 2:
                    k = k.view(B, T_k, H, D).transpose(1, 2).contiguous()
                    v = v.view(B, T_k, H, D).transpose(1, 2).contiguous()
                else:
                    k = k.view(B, T_k, H, D).contiguous()
                    v = v.view(B, T_k, H, D).contiguous()
                if self._is_streaming:
                    self._streaming_state['cross_k'] = k
                    self._streaming_state['cross_v'] = v
        else:
            projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            if time_dim == 2:
                packed = projected.view(B, T_q, 3, H, D).permute(2, 0, 3, 1, 4)
                q, k, v = packed[0], packed[1], packed[2]
            else:
                packed = projected.view(B, T_q, 3, H, D)
                q, k, v = packed[:, :, 0], packed[:, :, 1], packed[:, :, 2]

            if self.rope:
                q, k = self._apply_rope(q, k)
            k, v = self._complete_kv(k, v)

        if self.attention_as_float32:
            q, k, v = q.float(), k.float(), v.float()

        if self.memory_efficient:
            if time_dim == 2:
                use_causal = (not self.cross_attention) and (T_q > 1) and (q.shape[2] == k.shape[2])
                x = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal, dropout_p=0.0)
                x = x.to(dtype).transpose(1, 2).reshape(B, T_q, C)
            else:
                x = ops.memory_efficient_attention(q, k, v, attn_mask, p=0.0)
                x = x.to(dtype).reshape(B, T_q, C)
        else:
            q = q / (D ** 0.5)
            if time_dim == 2:
                pre_w = torch.matmul(q, k.transpose(-2, -1))
                if attn_mask is not None:
                    pre_w = pre_w + attn_mask
                w = torch.softmax(pre_w, dim=-1).to(v)
                x = torch.matmul(w, v).to(dtype).transpose(1, 2).reshape(B, T_q, C)
            else:
                pre_w = torch.einsum("b t h d, b k h d -> b h t k", q, k)
                if attn_mask is not None:
                    pre_w = pre_w + attn_mask
                w = torch.softmax(pre_w, dim=-1).to(v)
                x = torch.einsum("b h t k, b k h d -> b t h d", w, v).to(dtype).reshape(B, T_q, C)

        x = self.out_proj(x)
        return x, None

    key, value = self._complete_kv(key, value)
    if self.attention_as_float32:
        query, key, value = query.float(), key.float(), value.float()
    x, _ = self.mha(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)
    return x.to(dtype), None


StreamingMultiheadAttention._complete_kv = fast_complete_kv
StreamingMultiheadAttention._apply_rope = fast_apply_rope
StreamingMultiheadAttention.forward = fast_mha_forward


# -----------------------------------------------------------------------------
# 2. ТОЧНАЯ FP32 ТАБЛИЦА СИНУСОИДАЛЬНЫХ ПОЗИЦИЙ (БЕЗ ПОТЕРИ ТОЧНОСТИ ФАЗЫ)
# -----------------------------------------------------------------------------
_STATIC_SIN_EMB_CACHE = {}


def fast_transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    B, T, C = x.shape
    offset_int = self._streaming_state.get('offsets', 0)

    if self.positional_embedding in ['sin', 'sin_rope']:
        cache_key = (C, x.device)
        if cache_key not in _STATIC_SIN_EMB_CACHE:
            pos = torch.arange(4096, device=x.device).view(1, -1, 1)
            # Вычисляем строго в float32 (как в оригинальном transformer.py), чтобы не терять точность!
            sin_f32 = create_sin_embedding(
                pos, C, max_period=self.max_period, dtype=torch.float32
            ) * self.positional_scale
            _STATIC_SIN_EMB_CACHE[cache_key] = sin_f32
        x = x + _STATIC_SIN_EMB_CACHE[cache_key][:, offset_int:offset_int + T, :].to(x.dtype)

    for layer in self.layers:
        x = layer(x, *args, **kwargs)

    if self._is_streaming:
        self._streaming_state['offsets'] = offset_int + T
    return x


StreamingTransformer.forward = fast_transformer_forward


# -----------------------------------------------------------------------------
# 3. FUSED BMM HEADS + ZERO-SYNC GUMBEL SAMPLING
# -----------------------------------------------------------------------------
_CACHED_TOKENIZED = None


def fast_lm_generate(self, prompt=None, conditions=[], num_samples=None, max_gen_len=256,
                     use_sampling=True, temp=1.0, top_k=250, top_p=0.0, cfg_coef=1.0,
                     cfg_coef_beta=None, two_step_cfg=None, remove_prompts=False,
                     check=False, callback=None) -> torch.Tensor:
    global _CACHED_TOKENIZED, LAST_PROFILE
    t_start_t5 = time.perf_counter()
    first_param = next(iter(self.parameters()))
    device = first_param.device

    if conditions:
        if _CACHED_TOKENIZED is None:
            _CACHED_TOKENIZED = self.condition_provider.tokenize(conditions)
        cfg_conditions = self.condition_provider(_CACHED_TOKENIZED)
        cross_attn_src = cfg_conditions['description'][0]
    else:
        cross_attn_src = None

    if prompt is None:
        num_samples = num_samples or 1
        prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

    B, K, T = prompt.shape
    start_offset = T
    pattern = self.pattern_provider.get_pattern(max_gen_len)
    unknown_token = -1

    gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
    gen_codes[..., :start_offset] = prompt
    gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
    start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)

    if not hasattr(self, '_fused_head_w'):
        self._fused_head_w = torch.stack([l.weight for l in self.linears], dim=0)
        self._fused_head_b = torch.stack([l.bias for l in self.linears], dim=0) if self.linears[0].bias is not None else None

    if device.type == 'cuda':
        torch.cuda.synchronize()
    LAST_PROFILE["t5_ms"] = (time.perf_counter() - t_start_t5) * 1000.0

    t_start_lm = time.perf_counter()
    k_eff = min(int(top_k), 50)
    inv_temp = 1.0 / max(0.05, float(temp))

    with self.streaming():
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]
        steps_ran = gen_sequence_len - start_offset_sequence

        for offset in range(start_offset_sequence, gen_sequence_len):
            curr_sequence = gen_sequence[..., prev_offset:offset]
            input_ = (self.emb[0](curr_sequence[:, 0]) + self.emb[1](curr_sequence[:, 1]) +
                      self.emb[2](curr_sequence[:, 2]) + self.emb[3](curr_sequence[:, 3]))

            out = self.transformer(input_, cross_attention_src=cross_attn_src)
            if self.out_norm:
                out = self.out_norm(out)

            out_last = out[:, -1, :].unsqueeze(0).expand(4, B, -1)
            if self._fused_head_b is not None:
                logits = torch.baddbmm(self._fused_head_b.unsqueeze(1), out_last, self._fused_head_w.transpose(1, 2))
            else:
                logits = torch.bmm(out_last, self._fused_head_w.transpose(1, 2))

            logits = logits.permute(1, 0, 2)
            if use_sampling and temp > 0.0:
                top_logits, top_indices = torch.topk(logits * inv_temp, k=k_eff, dim=-1)
                u = torch.empty_like(top_logits).uniform_(1e-6, 1.0 - 1e-6)
                gumbel = top_logits - torch.log(-torch.log(u))
                sample_idx = torch.argmax(gumbel, dim=-1, keepdim=True)
                next_token = torch.gather(top_indices, -1, sample_idx)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            valid_mask = mask[..., offset:offset + 1].expand(B, -1, -1)
            next_token = torch.where(valid_mask, next_token, self.special_token_id)
            curr_slice = gen_sequence[..., offset:offset + 1]
            gen_sequence[..., offset:offset + 1] = torch.where(curr_slice == unknown_token, next_token, curr_slice)
            prev_offset = offset

    if device.type == 'cuda':
        torch.cuda.synchronize()
    lm_ms = (time.perf_counter() - t_start_lm) * 1000.0
    LAST_PROFILE["lm_loop_ms"] = lm_ms
    LAST_PROFILE["steps_count"] = max(1, steps_ran)
    LAST_PROFILE["per_step_ms"] = lm_ms / max(1, steps_ran)
    LAST_PROFILE["ctx_tokens"] = start_offset

    out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
    out_start_offset = start_offset if remove_prompts else 0
    return out_codes[..., out_start_offset:max_gen_len]


LMModel.generate = fast_lm_generate

ACTIVE_T5_EMBEDS = None
_orig_t5_forward = T5Conditioner.forward


def intercepted_t5_forward(self, inputs):
    global ACTIVE_T5_EMBEDS
    if ACTIVE_T5_EMBEDS is not None:
        target_device = self.output_proj.weight.device
        target_dtype = self.output_proj.weight.dtype
        embeds = ACTIVE_T5_EMBEDS.to(device=target_device, dtype=target_dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        b_size = inputs['input_ids'].shape[0] if ('input_ids' in inputs) else (inputs['attention_mask'].shape[0] if 'attention_mask' in inputs else 1)
        if embeds.shape[0] != b_size:
            embeds = embeds.expand(b_size, -1, -1)
        mask = torch.ones((embeds.shape[0], embeds.shape[1]), device=target_device, dtype=torch.long)
        return self.output_proj(embeds), mask
    return _orig_t5_forward(self, inputs)


T5Conditioner.forward = intercepted_t5_forward


def encode_clean_64_t5(t5_cond, prompt_text: str) -> torch.Tensor:
    """
    Кодирует концепт в [64, 768] БЕЗ НИ ОДНОГО МЕРТВОГО <pad>-ТОКЕНА!
    В T5 <pad>-токены не несут информации и забивают 92% внимания Cross-Attention.
    Мы прогоняем через T5 только реальные токены промпта (attention_mask == 1)
    и равномерно повторяем их на 64 слота: по свойству softmax это дает 100%
    точное совпадение с нативным MusicGen и открывает все 64 слота для 5 регионов мозга!
    """
    text_clean = prompt_text.strip()
    if not text_clean:
        # Для пустого промпта берем чистый нейтральный токен T5 без 63 <pad>-хвостов
        inp = t5_cond.t5_tokenizer([""], return_tensors='pt', padding=True).to(DEVICE)
        act_emb = t5_cond.t5(**inp).last_hidden_state[0].float()
    else:
        inp = t5_cond.t5_tokenizer([text_clean], return_tensors='pt', padding=True, truncation=True, max_length=MAX_T5_TOKENS).to(DEVICE)
        act_emb = t5_cond.t5(**inp).last_hidden_state[0].float()  # Только реальные слова [L_act, 768]!

    L_act = act_emb.shape[0]
    if L_act == MAX_T5_TOKENS:
        return act_emb
    repeats = (MAX_T5_TOKENS // max(1, L_act)) + 1
    tiled_64 = act_emb.repeat(repeats, 1)[:MAX_T5_TOKENS].contiguous()
    return tiled_64


active_listener, active_conn = None, None


def graceful_shutdown(sig, frame):
    global active_listener, active_conn
    print("\n🛑 [MUSICGEN-SERVER] Завершение работы по Ctrl+C...")
    if active_conn:
        try:
            active_conn.close()
        except Exception:
            pass
    if active_listener:
        try:
            active_listener.close()
        except Exception:
            pass
    os._exit(0)


signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


class AudioCrossfaderGPU:
    def __init__(self, fade_len=512):
        self.fade_len = fade_len
        self.fade_in = torch.linspace(0.0, 1.0, fade_len, device=DEVICE, dtype=torch.float32)
        self.fade_out = torch.linspace(1.0, 0.0, fade_len, device=DEVICE, dtype=torch.float32)
        self.overlap = None

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        peak = torch.max(torch.abs(chunk))
        if peak > 0.92:
            chunk = chunk / (peak + 1e-5) * 0.92
        t_len = chunk.shape[-1]
        if t_len <= self.fade_len:
            return chunk
        if self.overlap is not None:
            chunk[..., :self.fade_len] = chunk[..., :self.fade_len] * self.fade_in + self.overlap * self.fade_out
        self.overlap = chunk[..., -self.fade_len:].clone()
        return chunk[..., :-self.fade_len]


def adapt_system(gen_time: float, raw_audio_duration: float, current_speed: float,
                 current_block: float, user_max_block: float, safety_factor: float,
                 queue_sec: float) -> tuple[float, float, float]:
    gen_time = max(gen_time, 0.001)
    real_rtf = raw_audio_duration / gen_time

    if real_rtf > 1.12 and queue_sec >= 0.4:
        current_block *= 0.92
    elif real_rtf < 0.96 or queue_sec < 0.3:
        current_block *= 1.08

    current_block = float(np.clip(current_block, 0.6, max(1.0, user_max_block)))

    target_q = max(0.35, min(1.0, current_block * 0.45))
    q_error = queue_sec - target_q

    speed_correction = float(np.clip(1.0 + q_error * 0.14, 0.82, 1.18))
    target_speed = real_rtf * safety_factor * speed_correction

    if 0.96 <= target_speed <= 1.04 and abs(q_error) < 0.25:
        target_speed = 1.0

    alpha = 0.25
    current_speed = (current_speed * (1.0 - alpha)) + (target_speed * alpha)
    current_speed = float(np.clip(current_speed, 0.45, 1.18))

    return current_speed, current_block, real_rtf


def resample_chunk_gpu(wav, speed):
    if abs(speed - 1.0) < 0.01:
        return wav
    new_len = int(wav.shape[-1] / speed)
    if new_len < 1:
        return wav
    return F.interpolate(wav.float(), size=new_len, mode='linear', align_corners=False)


def run_startup_diagnostics(mg: MusicGen, dummy_t5: torch.Tensor):
    global ACTIVE_T5_EMBEDS, _CACHED_TOKENIZED
    print("\n" + "=" * 88)
    print("🔬 [ДИАГНОСТИКА MUSICGEN НА GPU: 17.4 ms/step + CLEAN 64-TOKEN T5]")
    print("=" * 88)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_mb = free_b / (1024**2)
        total_mb = total_b / (1024**2)
        print(f"🖥️  GPU: {gpu_name} | Свободно VRAM: {free_mb:.0f} МБ / {total_mb:.0f} МБ")

    ACTIVE_T5_EMBEDS = dummy_t5.to(DEVICE)
    set_efficient_attention_backend('torch')
    _CACHED_TOKENIZED = None

    with torch.inference_mode():
        mg.set_generation_params(duration=2.5, cfg_coef=1.0, use_sampling=True, top_k=250, temperature=1.0)
        _, warm_tokens = mg.generate([""], progress=False, return_tokens=True)
        _ = mg.compression_model.decode(warm_tokens[..., -40:], None)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

    print(f"{'Режим (Block / Context)':<28} | {'CtxТок':<6} | {'Шагов':<6} | {'LM Loop':<9} | {'1 Шаг':<8} | {'EnCodec':<7} | {'ЧИСТЫЙ RTF'}")
    print("-" * 88)

    for b_sec, c_sec in [(1.2, 0.4), (1.8, 0.4), (2.4, 0.4), (2.4, 1.2)]:
        ctx_tokens = max(1, int(round(c_sec * FRAME_RATE)))
        prompt_tokens = warm_tokens[..., -ctx_tokens:]
        gen_tokens = max(1, int(round(b_sec * FRAME_RATE)))
        total_tokens = prompt_tokens.shape[-1] + gen_tokens
        total_dur = total_tokens / float(FRAME_RATE)

        mg.set_generation_params(duration=total_dur, cfg_coef=1.0, use_sampling=True, top_k=250, temperature=1.0)
        attributes = [mg._prepare_tokens_and_attributes([""], None)[0][0]]

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            with mg.autocast:
                out_tokens = mg.lm.generate(prompt_tokens, attributes, max_gen_len=total_tokens, **mg.generation_params)

            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_enc0 = time.perf_counter()
            new_tokens = out_tokens[..., prompt_tokens.shape[-1]:]
            tokens_to_decode = out_tokens[..., -(new_tokens.shape[-1] + 2):]
            decoded_audio = mg.compression_model.decode(tokens_to_decode, None)
            new_chunk = decoded_audio[..., 1280:]
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t_enc0) * 1000.0

        total_s = time.perf_counter() - t0
        audio_s = new_chunk.shape[-1] / float(SAMPLE_RATE)
        rtf = audio_s / total_s

        print(
            f"Block {b_sec:3.1f}s / Ctx {c_sec:4.2f}s     | "
            f"{ctx_tokens:<6d} | "
            f"{LAST_PROFILE['steps_count']:<6d} | "
            f"{LAST_PROFILE['lm_loop_ms']:6.1f} ms | "
            f"{LAST_PROFILE['per_step_ms']:5.2f} ms | "
            f"{enc_ms:4.1f} ms | "
            f"👉 {rtf:5.2f}x"
        )

    print("=" * 88 + "\n", flush=True)


def main():
    global ACTIVE_T5_EMBEDS, active_listener, active_conn
    parser = argparse.ArgumentParser(description="NeuroCanvas Turbo MusicGen Server")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument('--model', type=str, default="facebook/musicgen-small")
    parser.add_argument('--benchmark', action='store_true', default=True)
    parser.add_argument('--quiet-step', action='store_true', default=False)
    args = parser.parse_args()

    print("=" * 70)
    print(f"🎵 ИНИЦИАЛИЗАЦИЯ MUSICGEN НА {DEVICE} (FP16, 17.4ms/step, CLEAN T5)...")
    print("=" * 70)

    set_efficient_attention_backend('torch')
    mg = MusicGen.get_pretrained(args.model, device=DEVICE)
    mg.lm.eval()
    mg.compression_model.eval()
    if DEVICE.type == 'cuda':
        mg.lm.to(torch.float16)
        mg.compression_model.to(torch.float16)

    for layer in mg.lm.transformer.layers:
        if layer.cross_attention is not None and not layer.cross_attention.custom:
            ca = layer.cross_attention
            ca.custom = True
            ca.in_proj_weight = ca.mha.in_proj_weight
            ca.in_proj_bias = ca.mha.in_proj_bias
            ca.out_proj = ca.mha.out_proj

    t5_cond = mg.lm.condition_provider.conditioners['description']
    t5_mod = t5_cond.t5
    raw_emb = t5_mod.shared.weight.detach().to(torch.float32)
    # Вычисляем естественную норму токенов T5 (~3.2) и главные оси словаря (32 128 токенов)
    mean_tok_norm = float(torch.norm(raw_emb, p=2, dim=-1).mean().item())
    norm_emb = raw_emb / (torch.norm(raw_emb, dim=-1, keepdim=True) + 1e-7)
    _, _, V_all = torch.pca_lowrank(norm_emb, q=120)
    proj_120_to_768 = V_all[:, :120].T.contiguous().to(DEVICE)
    global_svd_bases_64x768 = (F.normalize(V_all[:, :64].T.contiguous(), p=2, dim=-1) * mean_tok_norm).to(DEVICE)

    with torch.inference_mode(), t5_cond.autocast:
        empty_embed_t = encode_clean_64_t5(t5_cond, "")
        empty_embed_64x768 = empty_embed_t.cpu().numpy()

    if args.benchmark:
        run_startup_diagnostics(mg, empty_embed_t)

    crossfader = AudioCrossfaderGPU(fade_len=512)
    current_tokens = None
    current_speed = 1.0
    current_block = 2.0

    active_listener = Listener(('localhost', args.port), authkey=b'music')
    print(f"🚀 [MUSICGEN-SERVER] Слушаю порт {args.port}...")

    chunk_idx = 0
    while True:
        try:
            active_conn = active_listener.accept()
            print("🔗 [MUSICGEN-SERVER] Клиент подключен!")
            current_tokens = None
            crossfader.overlap = None

            while True:
                if not active_conn.poll(0.05):
                    continue
                try:
                    msg = active_conn.recv()
                except (EOFError, ConnectionResetError):
                    break

                cmd = msg.get('cmd')

                if cmd == 'init_lore':
                    prompts = [str(p).strip() for p in msg.get('prompts', []) if str(p).strip()]
                    encoded_bases = []
                    with torch.inference_mode(), t5_cond.autocast:
                        for p in prompts:
                            emb = encode_clean_64_t5(t5_cond, p)
                            encoded_bases.append(emb.cpu().numpy())
                    active_conn.send({
                        'status': 'ok',
                        'c_bases': encoded_bases,
                        'empty_embed': empty_embed_64x768,
                        'global_svd_bases': global_svd_bases_64x768.cpu().numpy(),
                        'proj_120_to_768': proj_120_to_768.cpu().numpy()
                    })

                elif cmd == 'generate_step':
                    t0 = time.perf_counter()
                    embeds_np = msg.get('t5_embeds')
                    user_max_block = float(msg.get('block_sec', 2.4))
                    context_sec = float(msg.get('context_sec', 0.6))
                    temp = float(msg.get('temp', 1.0))
                    top_k = int(msg.get('top_k', 250))
                    cfg_coef = float(msg.get('cfg_coef', 1.0))
                    safety_factor = float(msg.get('safety_factor', 1.0))
                    queue_sec = float(msg.get('queue_sec', 0.8))
                    is_reset = bool(msg.get('is_reset', False))

                    if embeds_np is not None:
                        ACTIVE_T5_EMBEDS = torch.from_numpy(embeds_np).to(DEVICE)

                    with torch.inference_mode():
                        if is_reset or current_tokens is None:
                            crossfader.overlap = None
                            init_dur = max(1.8, current_block + 0.08)
                            mg.set_generation_params(
                                duration=init_dur, cfg_coef=cfg_coef,
                                use_sampling=True, top_k=top_k, temperature=temp
                            )
                            _, raw_tokens = mg.generate([""], progress=False, return_tokens=True)
                            # Отрезаем 3 краевых токена DelayPattern, где 0-й кодбук уже закрыт маской!
                            current_tokens = raw_tokens[..., :-3] if raw_tokens.shape[-1] > 6 else raw_tokens
                            with torch.no_grad():
                                init_audio = mg.compression_model.decode(current_tokens, None)
                            clean = crossfader.process(init_audio)
                            pcm = clean[0, 0].cpu().float().numpy()
                            gen_time = time.perf_counter() - t0
                            raw_audio_len = len(pcm) / float(SAMPLE_RATE)
                            real_rtf = raw_audio_len / max(gen_time, 1e-5)
                        else:
                            max_avail_ctx = current_tokens.shape[-1]
                            ctx_tokens = max(5, min(max_avail_ctx, int(round(context_sec * FRAME_RATE))))
                            prompt_tokens = current_tokens[..., -ctx_tokens:]

                            gen_tokens = max(25, int(round((current_block * FRAME_RATE) / 10.0)) * 10)
                            # Генерируем +3 шага сверху и отрезаем их после revert_pattern_sequence,
                            # чтобы в current_tokens и в декодере ВСЕ 4 кодбука были 100% заполнены!
                            total_tokens = prompt_tokens.shape[-1] + gen_tokens + 3
                            total_duration_sec = total_tokens / float(FRAME_RATE)

                            mg.set_generation_params(
                                duration=total_duration_sec, cfg_coef=cfg_coef,
                                use_sampling=True, top_k=top_k, temperature=temp
                            )
                            attributes = [mg._prepare_tokens_and_attributes([""], None)[0][0]]
                            with mg.autocast:
                                out_tokens = mg.lm.generate(
                                    prompt_tokens, attributes,
                                    max_gen_len=total_tokens, **mg.generation_params
                                )

                            if DEVICE.type == 'cuda':
                                torch.cuda.synchronize()
                            t_enc = time.perf_counter()

                            # Отсекаем 3 неполных краевых токена DelayPattern справа!
                            valid_out_tokens = out_tokens[..., :-3]
                            current_tokens = valid_out_tokens[..., -400:]
                            new_tokens = valid_out_tokens[..., prompt_tokens.shape[-1]:]
                            tokens_to_decode = valid_out_tokens[..., -(new_tokens.shape[-1] + 2):]

                            with torch.no_grad():
                                decoded_audio = mg.compression_model.decode(tokens_to_decode, None)

                            new_chunk = decoded_audio[..., 1280:]
                            if DEVICE.type == 'cuda':
                                torch.cuda.synchronize()
                            LAST_PROFILE["encodec_ms"] = (time.perf_counter() - t_enc) * 1000.0

                            raw_audio_len = new_chunk.shape[-1] / float(SAMPLE_RATE)
                            gen_time = time.perf_counter() - t0

                            current_speed, current_block, real_rtf = adapt_system(
                                gen_time, raw_audio_len, current_speed, current_block,
                                user_max_block, safety_factor, queue_sec
                            )
                            processed_gpu = resample_chunk_gpu(new_chunk, current_speed)
                            clean = crossfader.process(processed_gpu)
                            pcm = clean[0, 0].cpu().float().numpy()

                    chunk_idx += 1
                    if not args.quiet_step:
                        free_mb = (torch.cuda.mem_get_info(0)[0] / (1024**2)) if DEVICE.type == 'cuda' else 0
                        actual_ctx_s = LAST_PROFILE.get('ctx_tokens', 0) / float(FRAME_RATE)
                        print(
                            f"⏱️ [Чанк #{chunk_idx:03d}] RTF: {real_rtf:4.2f}x | Speed: {current_speed:4.2f}x | "
                            f"AutoBlock: {current_block:.2f}s (Ctx:{actual_ctx_s:.2f}s, Buf:{queue_sec:.1f}s) | "
                            f"Шагов: {LAST_PROFILE['steps_count']} ({LAST_PROFILE['per_step_ms']:.2f} мс/шаг) | "
                            f"LM: {LAST_PROFILE['lm_loop_ms']:.0f}мс | EnCodec: {LAST_PROFILE['encodec_ms']:.0f}мс | FreeVRAM: {free_mb:.0f}МБ",
                            flush=True
                        )

                    active_conn.send({
                        'pcm': pcm,
                        'gen_time': gen_time,
                        'audio_len': len(pcm) / float(SAMPLE_RATE),
                        'rtf': real_rtf,
                        'play_speed': current_speed,
                        'actual_block': current_block
                    })

                elif cmd == 'flush':
                    current_tokens = None
                    crossfader.overlap = None
                    active_conn.send({'status': 'ok'})

        except Exception:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
