#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ZERO-PREFILL KV-CACHE & CONTINUOUS ENCODEC STREAMING
===================================================================================
- Никаких захардкоженных концептов или жанров: чистый амодальный оператор.
- Непрерывный EnCodec декодер: ровно N * 640 сэмплов на каждый блок без клипов.
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import StreamingMultiheadAttention
from audiocraft.modules.conditioners import T5Conditioner

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_T5_TOKENS = 64
KV_BUFFER_CAPACITY = 2048

ACTIVE_T5_EMBEDS: Optional[torch.Tensor] = None
_orig_complete_kv = StreamingMultiheadAttention._complete_kv
_orig_t5_forward = T5Conditioner.forward


def set_active_t5_embeds(embeds: Optional[torch.Tensor]):
    global ACTIVE_T5_EMBEDS
    ACTIVE_T5_EMBEDS = embeds


@torch.no_grad()
def invalidate_cross_attention_cache(lm: LMModel):
    for layer in lm.transformer.layers:
        if layer.cross_attention is not None and hasattr(layer.cross_attention, '_streaming_state'):
            layer.cross_attention._streaming_state.pop('cross_k', None)
            layer.cross_attention._streaming_state.pop('cross_v', None)


def fast_complete_kv(self, k, v):
    if self.cross_attention or not self._is_streaming:
        return _orig_complete_kv(self, k, v)

    if (not hasattr(self, '_persistent_k') or
            self._persistent_k.shape[0] != k.shape[0] or
            self._persistent_k.ndim != k.ndim or
            self._persistent_k.shape[1] != k.shape[1]):
        shape_k = list(k.shape)
        shape_k[2] = KV_BUFFER_CAPACITY
        self._persistent_k = torch.zeros(shape_k, device=k.device, dtype=k.dtype)
        self._persistent_v = torch.zeros(shape_k, device=v.device, dtype=v.dtype)

    curr_len = self._streaming_state.get('curr_len', 0)
    slen = k.shape[2]
    static_k = self._persistent_k
    static_v = self._persistent_v

    if curr_len + slen > KV_BUFFER_CAPACITY:
        keep = KV_BUFFER_CAPACITY // 2
        static_k[:, :, :keep, :].copy_(static_k[:, :, curr_len - keep:curr_len, :])
        if v is not k:
            static_v[:, :, :keep, :].copy_(static_v[:, :, curr_len - keep:curr_len, :])
        curr_len = keep

    new_len = curr_len + slen
    static_k[:, :, curr_len:new_len, :] = k
    nk = static_k[:, :, :new_len, :]
    if v is not k:
        static_v[:, :, curr_len:new_len, :] = v
        nv = static_v[:, :, :new_len, :]
    else:
        nv = nk

    self._streaming_state['curr_len'] = new_len
    return nk, nv


def fast_apply_rope(self, query: torch.Tensor, key: torch.Tensor):
    past_keys_offset = self._streaming_state.get('curr_len', 0)
    return self.rope.rotate_qk(query, key, start=past_keys_offset, time_dim=2)


def fast_mha_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     key_padding_mask=None, need_weights=False, attn_mask=None,
                     average_attn_weights=True, is_causal=False):
    dtype = query.dtype
    B, T_q, C = query.shape
    H = self.num_heads
    D = C // H

    if self.cross_attention:
        q = F.linear(query, self._q_weight, self._q_bias)
        if self.qk_layer_norm:
            q = self.q_layer_norm(q)
        q = q.view(B, T_q, H, D).transpose(1, 2).contiguous()

        state = self._streaming_state
        if self._is_streaming and 'cross_k' in state:
            k = state['cross_k']
            v = state['cross_v']
        else:
            k = F.linear(key, self._k_weight, self._k_bias)
            v = F.linear(value, self._v_weight, self._v_bias)
            if self.qk_layer_norm:
                k = self.k_layer_norm(k)
            T_k = k.shape[1]
            k = k.view(B, T_k, H, D).transpose(1, 2).contiguous()
            v = v.view(B, T_k, H, D).transpose(1, 2).contiguous()
            if self._is_streaming:
                state['cross_k'] = k
                state['cross_v'] = v
        use_causal = False
    else:
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        packed = projected.view(B, T_q, 3, H, D).permute(2, 0, 3, 1, 4)
        q = packed[0].contiguous()
        k = packed[1]
        v = packed[2]

        if self.rope is not None:
            past_keys_offset = self._streaming_state.get('curr_len', 0)
            q, k = self.rope.rotate_qk(q, k, start=past_keys_offset, time_dim=2)
        k, v = self._complete_kv(k, v)
        k = k.contiguous()
        v = v.contiguous()
        use_causal = (T_q > 1) and (T_q == k.shape[2])

    if self.attention_as_float32:
        q, k, v = q.float(), k.float(), v.float()

    x = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal, dropout_p=0.0)
    x = x.to(dtype).transpose(1, 2).reshape(B, T_q, C)
    return self.out_proj(x), None


def intercepted_t5_forward(self, inputs):
    if ACTIVE_T5_EMBEDS is not None:
        target_device = self.output_proj.weight.device
        target_dtype = self.output_proj.weight.dtype
        embeds = ACTIVE_T5_EMBEDS.to(device=target_device, dtype=target_dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        b_size = (
            inputs['input_ids'].shape[0] if ('input_ids' in inputs)
            else (inputs['attention_mask'].shape[0] if 'attention_mask' in inputs else 1)
        )
        if embeds.shape[0] != b_size:
            embeds = embeds.expand(b_size, -1, -1)
        mask = torch.ones((embeds.shape[0], embeds.shape[1]), device=target_device, dtype=torch.long)
        return self.output_proj(embeds), mask
    return _orig_t5_forward(self, inputs)


def apply_audiocraft_fast_patches():
    StreamingMultiheadAttention._complete_kv = fast_complete_kv
    StreamingMultiheadAttention._apply_rope = fast_apply_rope
    StreamingMultiheadAttention.forward = fast_mha_forward
    T5Conditioner.forward = intercepted_t5_forward


def prepare_lm_fast_weights(lm: LMModel):
    for layer in lm.transformer.layers:
        if layer.cross_attention is not None:
            ca = layer.cross_attention
            if not ca.custom:
                ca.custom = True
                ca.in_proj_weight = ca.mha.in_proj_weight
                ca.in_proj_bias = ca.mha.in_proj_bias
                ca.out_proj = ca.mha.out_proj
            dim = ca.in_proj_weight.shape[0] // 3
            ca._q_weight = ca.in_proj_weight[:dim].contiguous()
            ca._k_weight = ca.in_proj_weight[dim:2 * dim].contiguous()
            ca._v_weight = ca.in_proj_weight[2 * dim:].contiguous()
            ca._q_bias = ca.in_proj_bias[:dim].contiguous() if ca.in_proj_bias is not None else None
            ca._k_bias = ca.in_proj_bias[dim:2 * dim].contiguous() if ca.in_proj_bias is not None else None
            ca._v_bias = ca.in_proj_bias[2 * dim:].contiguous() if ca.in_proj_bias is not None else None

    lm._fused_head_w_T = torch.stack([l.weight.T for l in lm.linears], dim=0).contiguous()
    lm._fused_head_b = torch.stack([l.bias for l in lm.linears], dim=0).contiguous() if lm.linears[0].bias is not None else None

    vocab_sz = lm.emb[0].weight.shape[0]
    lm._fused_emb_weight = torch.cat([lm.emb[k].weight for k in range(4)], dim=0).contiguous()
    lm._emb_offsets = (torch.arange(4, device=lm._fused_emb_weight.device) * vocab_sz).view(1, 4, 1)


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


class ExactStreamingEncodecDecoder:
    """
    НЕПРЕРЫВНЫЙ ПОТОКОВЫЙ ДЕКОДЕР ENCODEC:
    - Декодирует блоки с левым прогревом pad_tokens=32.
    - Отдаёт ровно N * 640 сэмплов на каждый блок из N токенов без задержек и повторов.
    - Плавный косинусный микро-стык на границе блоков исключает щелчки.
    """
    def __init__(self, compression_model, pad_tokens: int = 32):
        self.compression_model = compression_model
        self.pad_tokens = max(16, int(pad_tokens))
        self.committed_history: Optional[torch.Tensor] = None
        self.prev_tail: Optional[torch.Tensor] = None

    def reset(self):
        self.committed_history = None
        self.prev_tail = None

    @torch.no_grad()
    def decode_exact_tokens(self, new_tokens: torch.Tensor) -> torch.Tensor:
        B, K, N = new_tokens.shape
        if N == 0:
            return torch.zeros((B, 1, 0), device=new_tokens.device, dtype=torch.float32)

        if self.committed_history is None:
            decoded = self.compression_model.decode(new_tokens, None)
            self.committed_history = new_tokens[:, :, -self.pad_tokens:].clone()
            self.prev_tail = decoded[..., -32:].clone()
            return decoded

        left_pad = self.committed_history
        pad_len = left_pad.shape[-1]
        work = torch.cat([left_pad, new_tokens], dim=-1)

        decoded_full = self.compression_model.decode(work, None)
        start_samp = pad_len * 640
        chunk_pcm = decoded_full[..., start_samp:]

        if self.prev_tail is not None:
            fade_len = min(16, chunk_pcm.shape[-1], self.prev_tail.shape[-1])
            fade_in = 0.5 * (1.0 - torch.cos(torch.linspace(0, torch.pi, fade_len, device=chunk_pcm.device)))
            chunk_pcm[..., :fade_len] = chunk_pcm[..., :fade_len] * fade_in + self.prev_tail[..., -fade_len:] * (1.0 - fade_in)

        self.committed_history = work[:, :, -self.pad_tokens:].clone()
        self.prev_tail = chunk_pcm[..., -32:].clone()
        return chunk_pcm
