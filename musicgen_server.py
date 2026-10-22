#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: CONTINUOUS DIAGONAL STREAM MUSICGEN SERVER (AUTO-MIN BLOCK & PACING)
===================================================================================
- РЕЖИМ БЛОК = 0 (ПО УМОЛЧАНИЮ):
  Динамически снижает размер блока вплоть до 1 токена EnCodec (0.02с = 20 мс),
  утилизируя 100% мощности GPU, и увеличивает блок только при опустошении буфера.
- ПОДДЕРЖКА СЛАБЫХ GPU (RTX 3060 и др.):
  Если при заданном контексте (например 3.6с) и блоке (например 1.2с) RTF < 1.0,
  сервер плавно замедляет темп музыки (play_speed < 1.0, без ускорения > 1.0x),
  чтобы буфер никогда не пустел.
- ЖЕСТКИЙ КОНТРОЛЬ ЧИСЛА БЛОКОВ БУФЕРА (target_buffer_count):
  При заполнении буфера (например 2 блока) сервер автоматически уменьшает размер
  блока и синхронизирует шаг генерации с реальным временем.
===================================================================================
"""
from __future__ import annotations

import os
import time
import argparse
from multiprocessing.connection import Listener
import torch
import torch.nn.functional as F
import numpy as np

try:
    os.setpriority(os.PRIO_PROCESS, 0, -20)
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

try:
    from xformers import ops
except ImportError:
    ops = None

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
MIN_BLOCK_SEC = 1.0 / float(FRAME_RATE)  # 0.02s (1 токен EnCodec = 640 сэмплов)
MAX_T5_TOKENS = 64

LAST_PROFILE = {
    "t5_ms": 0.0,
    "lm_loop_ms": 0.0,
    "per_step_ms": 0.0,
    "encodec_ms": 0.0,
    "steps_count": 0,
    "ctx_tokens": 0
}

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
            if time_dim == 2 or ops is None:
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

_STATIC_SIN_EMB_CACHE = {}


def fast_transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    B, T, C = x.shape
    offset_int = self._streaming_state.get('offsets', 0)

    if self.positional_embedding in ['sin', 'sin_rope']:
        cache_key = (C, x.device)
        if cache_key not in _STATIC_SIN_EMB_CACHE:
            pos = torch.arange(4096, device=x.device).view(1, -1, 1)
            sin_f32 = create_sin_embedding(pos, C, max_period=self.max_period, dtype=torch.float32) * self.positional_scale
            _STATIC_SIN_EMB_CACHE[cache_key] = sin_f32
        x = x + _STATIC_SIN_EMB_CACHE[cache_key][:, offset_int:offset_int + T, :].to(x.dtype)

    for layer in self.layers:
        x = layer(x, *args, **kwargs)

    if self._is_streaming:
        self._streaming_state['offsets'] = offset_int + T
    return x


StreamingTransformer.forward = fast_transformer_forward

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
) -> tuple[torch.Tensor, torch.Tensor, int]:
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

    gen_frames = max(1, int(gen_frames))

    if prev_diag_seq is None or prev_diag_seq.shape[-1] < 8:
        steps_to_run = max(12, gen_frames + 4)
        work_seq = torch.full((B, K, steps_to_run + 1), special_id, dtype=torch.long, device=device)
        start_step = 1
        LAST_PROFILE["ctx_tokens"] = 0
    else:
        avail = prev_diag_seq.shape[-1]
        keep_ctx = max(8, min(avail, ctx_frames))
        ctx_slice = prev_diag_seq[..., -keep_ctx:]
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

    n_extract = gen_frames if prev_diag_seq is not None else (steps_to_run - 4)
    prefix_frames = 3 if (total_seq_len - n_extract >= 7) else 0
    total_extract = n_extract + prefix_frames
    s_start = total_seq_len - total_extract

    c0 = work_seq[:, 0, s_start - 3: total_seq_len - 3]
    c1 = work_seq[:, 1, s_start - 2: total_seq_len - 2]
    c2 = work_seq[:, 2, s_start - 1: total_seq_len - 1]
    c3 = work_seq[:, 3, s_start:     total_seq_len]
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
    return act_emb.repeat(repeats, 1)[:MAX_T5_TOKENS].contiguous()


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


def adapt_block_and_pacing(
    gen_time: float,
    raw_audio_duration: float,
    current_block: float,
    user_target_block: float,
    queue_sec: float,
    queue_chunks: int = 0,
    target_buffer_count: int = 2,
    safety_factor: float = 1.0
) -> tuple[float, float, float, float]:
    """
    Двухрежимный контроллер баланса GPU / Буфера / Скорости:
    1) Если user_target_block <= 0.0 (Режим Авто-Минимум по умолчанию):
       - Стремится сжать current_block до предела (1 фрейм = 0.02с), пока GPU успевает,
         чтобы вся мощность уходила в мгновенный отклик без накопления буфера.
       - Как только буфер пустеет (queue_chunks == 0 или RTF < 0.98), плавно увеличивает
         размер блока, чтобы сбалансировать накладные расходы контекста.
    2) Если user_target_block > 0.0 (Ручной режим, например 1.2с на RTX 3060 или 0.001с тест):
       - Если GPU не тянет выбранный блок + контекст (RTF < 0.98) — плавно замедляет
         музыку (play_speed < 1.0, но никогда > 1.0), чтобы буфер не обнулялся.
       - Если же GPU слишком быстрый и буфер достиг target_buffer_count (например 2 чанка),
         автоматически уменьшает размер блока и тормозит генерацию, чтобы буфер не рос.
    """
    gen_time = max(gen_time, 0.001)
    real_rtf = raw_audio_duration / gen_time
    target_buf = max(1, int(target_buffer_count))
    play_speed = 1.0

    if user_target_block <= 0.0:
        # АВТО-МИНИМУМ БЛОКА (Утилизация 100% мощности ускорителя)
        if queue_chunks >= target_buf or (queue_chunks >= 1 and real_rtf > 1.06):
            # Буфер полон или есть запас скорости -> жмем блок вниз к 0.02с
            shrink = 0.76 if (queue_chunks > target_buf or real_rtf > 1.45) else 0.88
            current_block = max(MIN_BLOCK_SEC, current_block * shrink)
        elif queue_chunks == 0 or real_rtf < 0.96:
            # Буфер опустел / ускоритель не вытягивает такой мелкий блок -> растим блок
            grow_step = max(MIN_BLOCK_SEC, current_block * 0.22)
            current_block = min(4.0, current_block + grow_step)
        elif real_rtf > 1.15:
            current_block = max(MIN_BLOCK_SEC, current_block * 0.94)

        current_block = float(np.clip(current_block, MIN_BLOCK_SEC, 4.0))

        # Если даже при выросшем блоке слабая карта не тянет тяжелый контекст — замедляем музыку
        if current_block >= 0.80 and real_rtf < 0.96 and queue_chunks == 0:
            play_speed = float(np.clip((real_rtf * 0.95) / max(0.5, safety_factor), 0.40, 1.0))

    else:
        # РУЧНОЙ БЛОК (с защитой от переполнения буфера и авто-замедлением на слабых GPU)
        desired_block = max(MIN_BLOCK_SEC, float(user_target_block))

        if queue_chunks >= target_buf and real_rtf > 1.05:
            # Буфер уже набрал заданное число блоков (например 2) -> снижаем размер блока!
            current_block = max(MIN_BLOCK_SEC, current_block * 0.80)
        else:
            current_block = float(np.clip(current_block * 0.65 + desired_block * 0.35, MIN_BLOCK_SEC, 4.0))

        # Если ускоритель тормозной (например RTX 3060 с контекстом 3.6с) -> замедляем музыку (<= 1.0x)
        if real_rtf < 0.98 and queue_chunks < target_buf:
            play_speed = float(np.clip((real_rtf * 0.95) / max(0.5, safety_factor), 0.35, 1.0))
        else:
            play_speed = 1.0

    actual_out_duration = raw_audio_duration / play_speed

    # Строгий контроль очереди: если в буфере уже есть target_buf чанков,
    # спим ровно столько, чтобы не превышать лимит числа блоков буфера!
    sleep_sec = 0.0
    if queue_chunks >= target_buf:
        excess_chunks = max(0, queue_chunks - target_buf)
        base_wait = max(0.0, actual_out_duration - gen_time)
        sleep_sec = float(np.clip(base_wait + excess_chunks * actual_out_duration * 0.6, 0.0, 3.0))

    return current_block, real_rtf, play_speed, sleep_sec


def main():
    global ACTIVE_T5_EMBEDS
    parser = argparse.ArgumentParser(description="NeuroCanvas MusicGen Server Realtime Auto-Min Block")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument('--model', type=str, default="facebook/musicgen-small")
    parser.add_argument('--quiet-step', action='store_true', default=False)
    args = parser.parse_args()

    print("=" * 70)
    print(f"🎵 ИНИЦИАЛИЗАЦИЯ MUSICGEN НА {DEVICE} (AUTO-MIN 0.02s BLOCK & SLOWDOWN)...")
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
    raw_emb = t5_cond.t5.shared.weight.detach().to(torch.float32)
    mean_tok_norm = float(torch.norm(raw_emb, p=2, dim=-1).mean().item())
    norm_emb = raw_emb / (torch.norm(raw_emb, dim=-1, keepdim=True) + 1e-7)
    _, _, V_all = torch.pca_lowrank(norm_emb, q=120)
    proj_120_to_768 = V_all[:, :120].T.contiguous().to(DEVICE)
    global_svd_bases_64x768 = (F.normalize(V_all[:, :64].T.contiguous(), p=2, dim=-1) * mean_tok_norm).to(DEVICE)

    with torch.inference_mode(), t5_cond.autocast:
        empty_embed_t = encode_clean_64_t5(t5_cond, "")
        empty_embed_64x768 = empty_embed_t.cpu().numpy()

    crossfader = AudioCrossfaderGPU(fade_len=640)
    current_diag_seq = None
    current_block = 0.16  # Стартуем с малого блока и тут же адаптируемся

    listener = Listener(('localhost', args.port), authkey=b'music')
    print(f"🚀 [MUSICGEN-SERVER] Слушаю порт {args.port}...")

    chunk_idx = 0
    while True:
        try:
            conn = listener.accept()
            current_diag_seq = None
            crossfader.overlap = None

            while True:
                if not conn.poll(0.05):
                    continue
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break

                cmd = msg.get('cmd')

                if cmd == 'init_lore':
                    prompts = [str(p).strip() for p in msg.get('prompts', []) if str(p).strip()]
                    encoded_bases = []
                    with torch.inference_mode(), t5_cond.autocast:
                        for p in prompts:
                            encoded_bases.append(encode_clean_64_t5(t5_cond, p).cpu().numpy())
                    conn.send({
                        'status': 'ok',
                        'c_bases': encoded_bases,
                        'empty_embed': empty_embed_64x768,
                        'global_svd_bases': global_svd_bases_64x768.cpu().numpy(),
                        'proj_120_to_768': proj_120_to_768.cpu().numpy()
                    })

                elif cmd == 'generate_step':
                    t0 = time.perf_counter()
                    embeds_np = msg.get('t5_embeds')
                    user_target_block = float(msg.get('block_sec', 0.0))
                    context_sec = float(msg.get('context_sec', 3.6))
                    temp = float(msg.get('temp', 1.0))
                    top_k = int(msg.get('top_k', 250))
                    safety_factor = float(msg.get('safety_factor', 1.0))
                    queue_sec = float(msg.get('queue_sec', 0.0))
                    queue_chunks = int(msg.get('queue_chunks', 0))
                    target_buffer_count = int(msg.get('target_buffer_count', 2))
                    is_reset = bool(msg.get('is_reset', False))

                    if embeds_np is not None:
                        ACTIVE_T5_EMBEDS = torch.from_numpy(embeds_np).to(DEVICE)

                    if is_reset:
                        current_diag_seq = None
                        crossfader.overlap = None

                    # Если юзер вручную задал блок > 0 и буфер не переполнен, сразу берем его
                    if user_target_block > 0.0 and queue_chunks < target_buffer_count:
                        current_block = max(MIN_BLOCK_SEC, user_target_block)

                    attributes = [mg._prepare_tokens_and_attributes([""], None)[0][0]]
                    gen_frames = max(1, int(round(current_block * FRAME_RATE)))
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
                        new_chunk = decoded_audio[..., 1280:] if pfx >= 2 else decoded_audio
                        if DEVICE.type == 'cuda':
                            torch.cuda.synchronize()
                        LAST_PROFILE["encodec_ms"] = (time.perf_counter() - t_enc) * 1000.0

                        raw_audio_len = gen_frames / float(FRAME_RATE)
                        gen_time = time.perf_counter() - t0

                        current_block, real_rtf, play_speed, sleep_sec = adapt_block_and_pacing(
                            gen_time=gen_time,
                            raw_audio_duration=raw_audio_len,
                            current_block=current_block,
                            user_target_block=user_target_block,
                            queue_sec=queue_sec,
                            queue_chunks=queue_chunks,
                            target_buffer_count=target_buffer_count,
                            safety_factor=safety_factor
                        )

                        # Если слабая карта (например RTX 3060) не вытягивает RTF 1.0 — плавно замедляем
                        if play_speed < 0.995 and new_chunk.shape[-1] > 640:
                            base_new_samples = new_chunk.shape[-1] - 640
                            stretched_total = 640 + int(round(base_new_samples / play_speed))
                            new_chunk = F.interpolate(
                                new_chunk.float(), size=stretched_total, mode='linear', align_corners=False
                            )

                        clean = crossfader.process(new_chunk.float())
                        pcm = clean[0, 0].cpu().float().numpy()

                    if sleep_sec > 0.002:
                        time.sleep(sleep_sec)

                    chunk_idx += 1
                    if not args.quiet_step:
                        actual_ctx_s = LAST_PROFILE.get('ctx_tokens', 0) / float(FRAME_RATE)
                        print(
                            f"⏱️ [Чанк #{chunk_idx:03d}] RTF:{real_rtf:4.2f}x | Speed:{play_speed:4.2f}x | "
                            f"Block:{raw_audio_len:.2f}s->Next:{current_block:.2f}s (Target:{user_target_block:.3f}s, Ctx:{actual_ctx_s:.1f}s) | "
                            f"Buf:{queue_sec:.2f}s ({queue_chunks}/{target_buffer_count} blk) | "
                            f"Steps:{LAST_PROFILE['steps_count']} ({LAST_PROFILE['per_step_ms']:.1f}ms/st)",
                            flush=True
                        )

                    conn.send({
                        'pcm': pcm,
                        'gen_time': gen_time,
                        'audio_len': len(pcm) / float(SAMPLE_RATE),
                        'rtf': real_rtf,
                        'play_speed': play_speed,
                        'actual_block': raw_audio_len
                    })
                    
                elif cmd == 'switch_model':
                    global _CACHED_TOKENIZED
                    new_model_name = str(msg.get('model', 'facebook/musicgen-small')).strip()
                    print(f"🔄 [MUSICGEN-SERVER] Горячая смена модели на: {new_model_name}...", flush=True)
                    del mg
                    torch.cuda.empty_cache()
                    mg = MusicGen.get_pretrained(new_model_name, device=DEVICE)
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
                    _CACHED_TOKENIZED = None
                    current_diag_seq = None
                    crossfader.overlap = None
                    print(f"✅ [MUSICGEN-SERVER] Модель {new_model_name} успешно загружена в VRAM!", flush=True)
                    conn.send({'status': 'ok', 'model': new_model_name})

                elif cmd == 'flush':
                    current_diag_seq = None
                    crossfader.overlap = None
                    conn.send({'status': 'ok'})

        except Exception:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
