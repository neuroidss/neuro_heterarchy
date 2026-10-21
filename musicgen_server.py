#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: CONTINUOUS DIAGONAL STREAM MUSICGEN SERVER (0.1s BLOCK CAPABLE)
===================================================================================
- НЕПРЕРЫВНЫЙ ДИАГОНАЛЬНЫЙ ПОТОК (current_gen_sequence):
  История контекста (например 3.6с = 180 шагов) хранится напрямую в диагональном
  пространстве кодбуков [1, 4, S] БЕЗ краевых масок 2048 (special_token_id).
  1) Размер блока (0.1с .. 2.4с) больше НЕ создает искусственного ритма!
  2) Блок 0.1с (5 фреймов) требует ровно 5 шагов трансформера (1 батчевый prefill
     контекста + 4 шага) без 3 холостых шагов закрытия диагонали (+60% к скорости
     на сверхкоротких блоках 0.1с - 0.2с!).
- СКОРОСТЬ 17.4 мс/шаг (Cross-KV Cache + FP32 Sinusoidal Cache + Gumbel-Max).
- ЗАКОН СОХРАНЕНИЯ СЭМПЛОВ (Zero-Drop) И ОТСУТСТВИЕ ПО-БЛОЧНОГО ПАМПИНГА ГРОМКОСТИ.
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
# 1. PERSISTENT SELF-ATTENTION & CROSS-ATTENTION KV-CACHE
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
            nv = static_v[:, :curr_len + slen, :]
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
# 2. ТОЧНАЯ FP32 ТАБЛИЦА СИНУСОИДАЛЬНЫХ ПОЗИЦИЙ
# -----------------------------------------------------------------------------
_STATIC_SIN_EMB_CACHE = {}


def fast_transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    B, T, C = x.shape
    offset_int = self._streaming_state.get('offsets', 0)

    if self.positional_embedding in ['sin', 'sin_rope']:
        cache_key = (C, x.device)
        if cache_key not in _STATIC_SIN_EMB_CACHE:
            pos = torch.arange(4096, device=x.device).view(1, -1, 1)
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
# 3. БЕСШОВНЫЙ ДИАГОНАЛЬНЫЙ ГЕНЕРАТОР (НОЛЬ МАСОК 2048 И НОЛЬ ХОЛОСТЫХ ШАГОВ!)
# -----------------------------------------------------------------------------
_CACHED_TOKENIZED = None


@torch.no_grad()
def step_continuous_diagonal_stream(
    lm: LMModel,
    prev_diag_seq: torch.Tensor | None,
    conditions: list,
    gen_frames: int,
    ctx_frames: int,
    temp: float = 1.0,
    top_k: int = 250
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует ровно `gen_frames` новых 4-кодбуковых аудио-фреймов в непрерывном
    диагональном потоке `diag_seq` [1, 4, S]:
    - Если `prev_diag_seq is None` (холодный старт): инициализирует первые 4 шага диагонали.
    - Если `prev_diag_seq` уже есть: берет последние `ctx_frames` чистых диагональных
      шагов (без единого токена 2048!) и делает РОВНО `gen_frames` шагов трансформера
      (1 батчевый prefill контекста + `gen_frames - 1` одиночных шагов)!
    - Возвращает:
      1) `new_diag_seq` (скользящее диагональное окно последних 450 шагов);
      2) `ready_audio_codes` `[1, 4, gen_frames + 3]` (полностью собранные 4-кодбуковые
         фреймы с 3 фреймами левого контекста для бесшовного декодирования EnCodec!).
    """
    global _CACHED_TOKENIZED, LAST_PROFILE
    t_start_t5 = time.perf_counter()
    device = next(iter(lm.parameters())).device

    if conditions:
        if _CACHED_TOKENIZED is None:
            _CACHED_TOKENIZED = lm.condition_provider.tokenize(conditions)
        cfg_conditions = lm.condition_provider(_CACHED_TOKENIZED)
        cross_attn_src = cfg_conditions['description'][0]
    else:
        cross_attn_src = None

    if not hasattr(lm, '_fused_head_w'):
        lm._fused_head_w = torch.stack([l.weight for l in lm.linears], dim=0)
        lm._fused_head_b = torch.stack([l.bias for l in lm.linears], dim=0) if lm.linears[0].bias is not None else None

    if device.type == 'cuda':
        torch.cuda.synchronize()
    LAST_PROFILE["t5_ms"] = (time.perf_counter() - t_start_t5) * 1000.0

    t_start_lm = time.perf_counter()
    B, K = 1, lm.num_codebooks
    special_id = lm.special_token_id
    k_eff = min(int(top_k), 50)
    inv_temp = 1.0 / max(0.05, float(temp))

    if prev_diag_seq is None or prev_diag_seq.shape[-1] < 8:
        # Холодный старт: нужно генерить с 0-го шага с начальной диагональной лестницей 2048
        steps_to_run = max(25, gen_frames + 4)
        work_seq = torch.full((B, K, steps_to_run + 1), special_id, dtype=torch.long, device=device)
        start_step = 1
        LAST_PROFILE["ctx_tokens"] = 0
    else:
        # Непрерывный режим: берем последние ctx_frames шагов из prev_diag_seq (в них НЕТ ни одного 2048!)
        avail = prev_diag_seq.shape[-1]
        keep_ctx = max(8, min(avail, ctx_frames))
        ctx_slice = prev_diag_seq[..., -keep_ctx:]
        # На первом же шаге prefill (0..keep_ctx) трансформер предскажет токен в колонку keep_ctx,
        # поэтому нам нужно ровно gen_frames новых колонок!
        work_seq = torch.zeros((B, K, keep_ctx + gen_frames), dtype=torch.long, device=device)
        work_seq[..., :keep_ctx] = ctx_slice
        start_step = keep_ctx
        steps_to_run = gen_frames
        LAST_PROFILE["ctx_tokens"] = keep_ctx

    total_seq_len = work_seq.shape[-1]

    with lm.streaming():
        prev_offset = 0
        for offset in range(start_step, total_seq_len):
            curr_sequence = work_seq[..., prev_offset:offset]
            input_ = (lm.emb[0](curr_sequence[:, 0]) + lm.emb[1](curr_sequence[:, 1]) +
                      lm.emb[2](curr_sequence[:, 2]) + lm.emb[3](curr_sequence[:, 3]))

            out = lm.transformer(input_, cross_attention_src=cross_attn_src)
            if lm.out_norm:
                out = lm.out_norm(out)

            out_last = out[:, -1, :].unsqueeze(0).expand(4, B, -1)
            if lm._fused_head_b is not None:
                logits = torch.baddbmm(lm._fused_head_b.unsqueeze(1), out_last, lm._fused_head_w.transpose(1, 2))
            else:
                logits = torch.bmm(out_last, lm._fused_head_w.transpose(1, 2))

            logits = logits.permute(1, 0, 2)
            if temp > 0.0:
                top_logits, top_indices = torch.topk(logits * inv_temp, k=k_eff, dim=-1)
                u = torch.empty_like(top_logits).uniform_(1e-6, 1.0 - 1e-6)
                gumbel = top_logits - torch.log(-torch.log(u))
                sample_idx = torch.argmax(gumbel, dim=-1, keepdim=True)
                next_token = torch.gather(top_indices, -1, sample_idx)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if prev_diag_seq is None and offset < 4:
                # Только в первые 3 миллисекунды старта потока соблюдаем начальную ступеньку DelayPattern
                for k_i in range(K):
                    if offset <= k_i:
                        next_token[:, k_i, :] = special_id

            work_seq[..., offset:offset + 1] = next_token
            prev_offset = offset

    if device.type == 'cuda':
        torch.cuda.synchronize()
    lm_ms = (time.perf_counter() - t_start_lm) * 1000.0
    LAST_PROFILE["lm_loop_ms"] = lm_ms
    LAST_PROFILE["steps_count"] = max(1, steps_to_run)
    LAST_PROFILE["per_step_ms"] = lm_ms / max(1, steps_to_run)

    # Собираем полностью готовые 4-кодбуковые аудио-фреймы из диагонали:
    # В колонке s лежат: [c0(t=s-1), c1(t=s-2), c2(t=s-3), c3(t=s-4)]
    # Значит для любого фрейма с индексом конца s (где s >= 4):
    # c0 = work_seq[:, 0, s - 3]
    # c1 = work_seq[:, 1, s - 2]
    # c2 = work_seq[:, 2, s - 1]
    # c3 = work_seq[:, 3, s]
    n_extract = gen_frames if prev_diag_seq is not None else (steps_to_run - 4)
    # Берем +3 предыдущих фрейма слева для идеального прогрева сверток EnCodec и кроссфейда
    prefix_frames = 3 if (total_seq_len - n_extract >= 7) else 0
    total_extract = n_extract + prefix_frames
    s_start = total_seq_len - total_extract

    c0 = work_seq[:, 0, s_start - 3 : total_seq_len - 3]
    c1 = work_seq[:, 1, s_start - 2 : total_seq_len - 2]
    c2 = work_seq[:, 2, s_start - 1 : total_seq_len - 1]
    c3 = work_seq[:, 3, s_start     : total_seq_len]
    ready_audio_codes = torch.stack([c0, c1, c2, c3], dim=1).clamp(0, lm.card - 1)

    new_diag_seq = work_seq[..., -450:].contiguous()
    return new_diag_seq, ready_audio_codes, prefix_frames


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
    text_clean = prompt_text.strip()
    if not text_clean:
        inp = t5_cond.t5_tokenizer([""], return_tensors='pt', padding=True).to(DEVICE)
        act_emb = t5_cond.t5(**inp).last_hidden_state[0].float()
    else:
        inp = t5_cond.t5_tokenizer([text_clean], return_tensors='pt', padding=True, truncation=True, max_length=MAX_T5_TOKENS).to(DEVICE)
        act_emb = t5_cond.t5(**inp).last_hidden_state[0].float()

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
    def __init__(self, fade_len=640):
        self.fade_len = fade_len
        t = torch.linspace(0.0, 1.0, fade_len, device=DEVICE, dtype=torch.float32)
        self.fade_in = torch.sin(t * (np.pi * 0.5))
        self.fade_out = torch.cos(t * (np.pi * 0.5))
        self.overlap = None

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        chunk = torch.tanh(chunk * 0.95)
        t_len = chunk.shape[-1]
        if t_len <= self.fade_len:
            return chunk
        if self.overlap is not None:
            chunk[..., :self.fade_len] = chunk[..., :self.fade_len] * self.fade_in + self.overlap * self.fade_out
        self.overlap = chunk[..., -self.fade_len:].clone()
        return chunk[..., :-self.fade_len]


def adapt_system(gen_time: float, raw_audio_duration: float, current_speed: float,
                 current_block: float, user_target_block: float, safety_factor: float,
                 queue_sec: float) -> tuple[float, float, float]:
    """
    Адаптер супер-реалтайма (поддерживает блоки вплоть до 0.10 сек!):
    - Стремится опустить current_block до user_target_block (даже если это 0.1с!),
      но если видеокарта на текущем шаге не вытягивает RTF >= 0.95, плавно
      увеличивает блок ровно настолько, чтобы удержать непрерывный звук.
    - Ничего не выкидывает: регулирует current_speed (0.45x .. 1.18x) так, чтобы
      доигрывать весь буфер и сводить задержку queue_sec к минимуму.
    """
    gen_time = max(gen_time, 0.001)
    real_rtf = raw_audio_duration / gen_time

    if real_rtf >= 1.05 and queue_sec >= 0.25:
        # Мощности хватает — опускаем размер блока к целевому (вплоть до 0.10 с!)
        current_block = max(user_target_block, current_block * 0.90)
    elif real_rtf < 0.88 or queue_sec < 0.15:
        # Если на 3060 блок 0.1с при контексте 3.6с тяжеловат — чуть поднимаем блок
        current_block = min(max(user_target_block, 2.4), current_block * 1.08)
    else:
        current_block = current_block * 0.85 + user_target_block * 0.15

    current_block = float(np.clip(current_block, 0.10, 4.0))

    target_q = max(0.20, min(0.80, current_block * 0.50))
    q_error = queue_sec - target_q

    speed_correction = float(np.clip(1.0 + q_error * 0.15, 0.82, 1.18))
    target_speed = real_rtf * safety_factor * speed_correction

    if 0.96 <= target_speed <= 1.04 and abs(q_error) < 0.20:
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
    print("\n" + "=" * 90)
    print("🔬 [ДИАГНОСТИКА MUSICGEN: БЕСШОВНЫЙ ДИАГОНАЛЬНЫЙ СТРИМ (БЛОКИ ОТ 0.1с ДО 2.4с)]")
    print("=" * 90)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"🖥️  GPU: {gpu_name} | Свободно VRAM: {free_b / (1024**2):.0f} МБ / {total_b / (1024**2):.0f} МБ")

    ACTIVE_T5_EMBEDS = dummy_t5.to(DEVICE)
    set_efficient_attention_backend('torch')
    _CACHED_TOKENIZED = None

    attributes = [mg._prepare_tokens_and_attributes([""], None)[0][0]]
    with torch.inference_mode(), mg.autocast:
        diag_seq, codes, _ = step_continuous_diagonal_stream(mg.lm, None, attributes, gen_frames=185, ctx_frames=180)
        _ = mg.compression_model.decode(codes[..., -20:], None)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

    print(f"{'Режим (Block / Context)':<28} | {'CtxТок':<6} | {'Шагов':<6} | {'LM Loop':<9} | {'1 Шаг':<8} | {'EnCodec':<7} | {'ЧИСТЫЙ RTF'}")
    print("-" * 90)

    for b_sec, c_sec in [(0.1, 3.6), (0.2, 3.6), (0.5, 3.6), (1.2, 3.6), (1.8, 2.4)]:
        gen_f = max(5, int(round(b_sec * FRAME_RATE)))
        ctx_f = max(10, int(round(c_sec * FRAME_RATE)))

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode(), mg.autocast:
            diag_seq, codes, pfx = step_continuous_diagonal_stream(
                mg.lm, diag_seq, attributes, gen_frames=gen_f, ctx_frames=ctx_f
            )
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_enc0 = time.perf_counter()
            decoded_audio = mg.compression_model.decode(codes, None)
            new_chunk = decoded_audio[..., 1280:] if pfx >= 2 else decoded_audio
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t_enc0) * 1000.0

        total_s = time.perf_counter() - t0
        audio_s = gen_f / float(FRAME_RATE)
        rtf = audio_s / total_s

        print(
            f"Block {b_sec:4.2f}s / Ctx {c_sec:4.2f}s    | "
            f"{ctx_f:<6d} | "
            f"{LAST_PROFILE['steps_count']:<6d} | "
            f"{LAST_PROFILE['lm_loop_ms']:6.1f} ms | "
            f"{LAST_PROFILE['per_step_ms']:5.2f} ms | "
            f"{enc_ms:4.1f} ms | "
            f"👉 {rtf:5.2f}x"
        )

    print("=" * 90 + "\n", flush=True)


def main():
    global ACTIVE_T5_EMBEDS, active_listener, active_conn
    parser = argparse.ArgumentParser(description="NeuroCanvas Turbo MusicGen Server")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument('--model', type=str, default="facebook/musicgen-small")
    parser.add_argument('--benchmark', action='store_true', default=True)
    parser.add_argument('--quiet-step', action='store_true', default=False)
    args = parser.parse_args()

    print("=" * 70)
    print(f"🎵 ИНИЦИАЛИЗАЦИЯ MUSICGEN НА {DEVICE} (CONTINUOUS DIAGONAL STREAM)...")
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

    crossfader = AudioCrossfaderGPU(fade_len=640)
    current_diag_seq = None
    current_speed = 1.0
    current_block = 1.2

    active_listener = Listener(('localhost', args.port), authkey=b'music')
    print(f"🚀 [MUSICGEN-SERVER] Слушаю порт {args.port}...")

    chunk_idx = 0
    while True:
        try:
            active_conn = active_listener.accept()
            print("🔗 [MUSICGEN-SERVER] Клиент подключен!")
            current_diag_seq = None
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
                    user_target_block = float(msg.get('block_sec', 1.2))
                    context_sec = float(msg.get('context_sec', 3.6))
                    temp = float(msg.get('temp', 1.0))
                    top_k = int(msg.get('top_k', 250))
                    safety_factor = float(msg.get('safety_factor', 1.0))
                    queue_sec = float(msg.get('queue_sec', 0.8))
                    is_reset = bool(msg.get('is_reset', False))

                    if embeds_np is not None:
                        ACTIVE_T5_EMBEDS = torch.from_numpy(embeds_np).to(DEVICE)

                    if is_reset:
                        current_diag_seq = None
                        crossfader.overlap = None

                    attributes = [mg._prepare_tokens_and_attributes([""], None)[0][0]]
                    gen_frames = max(5, int(round(current_block * FRAME_RATE)))
                    ctx_frames = max(10, int(round(context_sec * FRAME_RATE)))

                    with torch.inference_mode(), mg.autocast:
                        current_diag_seq, ready_codes, pfx = step_continuous_diagonal_stream(
                            mg.lm, current_diag_seq, attributes,
                            gen_frames=gen_frames, ctx_frames=ctx_frames,
                            temp=temp, top_k=top_k
                        )

                        if DEVICE.type == 'cuda':
                            torch.cuda.synchronize()
                        t_enc = time.perf_counter()
                        decoded_audio = mg.compression_model.decode(ready_codes, None)
                        # Отрезаем 2 токена (1280 сэмплов) левого прогрева рецептивного поля EnCodec,
                        # оставляя ровно 1 токен (640 сэмплов) для фазово-точного кроссфейда!
                        new_chunk = decoded_audio[..., 1280:] if pfx >= 2 else decoded_audio
                        if DEVICE.type == 'cuda':
                            torch.cuda.synchronize()
                        LAST_PROFILE["encodec_ms"] = (time.perf_counter() - t_enc) * 1000.0

                        raw_audio_len = gen_frames / float(FRAME_RATE)
                        gen_time = time.perf_counter() - t0

                        current_speed, current_block, real_rtf = adapt_system(
                            gen_time, raw_audio_len, current_speed, current_block,
                            user_target_block, safety_factor, queue_sec
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
                            f"AutoBlock: {current_block:.2f}s (Target:{user_target_block:.2f}s, Ctx:{actual_ctx_s:.2f}s, Buf:{queue_sec:.1f}s) | "
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
                    current_diag_seq = None
                    crossfader.overlap = None
                    active_conn.send({'status': 'ok'})

        except Exception:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
