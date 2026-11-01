#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ZERO-SEAM LOOKAHEAD ENCODEC DECODER & PURE B=1 FLASH-ATTN KERNELS
===================================================================================
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import (
    StreamingMultiheadAttention,
    StreamingTransformer,
    create_sin_embedding
)
from audiocraft.modules.conditioners import T5Conditioner

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_T5_TOKENS = 64
KV_BUFFER_CAPACITY = 2048

ACTIVE_T5_EMBEDS: Optional[torch.Tensor] = None
_STATIC_SIN_EMB_CACHE = {}

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
            self._persistent_k.shape[1] != k.shape[1] or
            self._persistent_k.dtype != k.dtype):
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
        # Чистый FP16 Tensor-Core путь на GPU (без конвертации в float32):
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        packed = projected.view(B, T_q, 3, H, D).permute(2, 0, 3, 1, 4)
        q = packed[0].contiguous()
        k = packed[1]
        v = packed[2]

        if self.rope is not None:
            past_keys_offset = self._streaming_state.get('curr_len', 0)
            q, k = self.rope.rotate_qk(q, k, start=past_keys_offset, time_dim=2)
        k, v = self._complete_kv(k, v)
        use_causal = (T_q > 1) and (T_q == k.shape[2])

    if self.attention_as_float32:
        q, k, v = q.float(), k.float(), v.float()

    x = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal, dropout_p=0.0)
    x = x.to(dtype).transpose(1, 2).reshape(B, T_q, C)
    return self.out_proj(x), None


def fast_transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    B, T, C = x.shape
    offset_int = int(self._streaming_state.get('offsets', 0))

    if self.positional_embedding in ['sin', 'sin_rope']:
        cache_key = (C, x.device, x.dtype)
        if cache_key not in _STATIC_SIN_EMB_CACHE:
            pos = torch.arange(65536, device=x.device).view(1, -1, 1)
            sin_emb = (create_sin_embedding(pos, C, max_period=self.max_period, dtype=torch.float32) * self.positional_scale).to(x.dtype)
            _STATIC_SIN_EMB_CACHE[cache_key] = sin_emb
        safe_off = offset_int % (65536 - T - 1)
        x = x + _STATIC_SIN_EMB_CACHE[cache_key][:, safe_off:safe_off + T, :]

    for layer in self.layers:
        x = layer(x, *args, **kwargs)

    if self._is_streaming:
        self._streaming_state['offsets'] = offset_int + T
    return x


def intercepted_t5_forward(self, inputs):
    if ACTIVE_T5_EMBEDS is not None:
        target_device = self.output_proj.weight.device
        target_dtype = self.output_proj.weight.dtype
        embeds = ACTIVE_T5_EMBEDS
        if embeds.device != target_device or embeds.dtype != target_dtype:
            embeds = embeds.to(device=target_device, dtype=target_dtype)
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
    StreamingTransformer.forward = fast_transformer_forward
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
    def __init__(self, compression_model, pad_tokens: int = 32, lookahead_tokens: int = 8):
        self.compression_model = compression_model
        self.pad_tokens = max(16, int(pad_tokens))
        self.lookahead_tokens = max(4, int(lookahead_tokens))
        self.committed_history: Optional[torch.Tensor] = None
        self.pending_lookahead: Optional[torch.Tensor] = None

    def reset(self):
        self.committed_history = None
        self.pending_lookahead = None

    @torch.no_grad()
    def decode_exact_tokens(self, new_tokens: torch.Tensor) -> torch.Tensor:
        B, K, T_new = new_tokens.shape
        if T_new == 0:
            return torch.zeros((B, 1, 0), device=new_tokens.device, dtype=torch.float32)

        if self.pending_lookahead is None:
            work_tokens = new_tokens
        else:
            work_tokens = torch.cat([self.pending_lookahead, new_tokens], dim=-1)

        total_work = work_tokens.shape[-1]
        if total_work <= self.lookahead_tokens:
            self.pending_lookahead = work_tokens
            decoded_full = self.compression_model.decode(work_tokens, None)
            return decoded_full

        emit_count = total_work - self.lookahead_tokens
        tokens_to_emit = work_tokens[:, :, :emit_count]
        self.pending_lookahead = work_tokens[:, :, emit_count:]

        if self.committed_history is None:
            total_ctx = work_tokens
            actual_left_pad = 0
        else:
            ctx_left = self.committed_history[:, :, -self.pad_tokens:]
            actual_left_pad = ctx_left.shape[-1]
            total_ctx = torch.cat([ctx_left, work_tokens], dim=-1)

        if self.committed_history is None:
            self.committed_history = tokens_to_emit
        else:
            self.committed_history = torch.cat([self.committed_history, tokens_to_emit], dim=-1)[:, :, -512:]

        decoded_full = self.compression_model.decode(total_ctx, None)
        start_sample = actual_left_pad * 640
        end_sample = start_sample + emit_count * 640
        return decoded_full[..., start_sample:end_sample]
