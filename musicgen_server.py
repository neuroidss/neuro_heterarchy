#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: TURBO MUSICGEN SERVER (PORT 6005)
===================================================================================
- Никаких зашитых слов: токены приходят только из лора пользователя.
- Фиксированная длина базисов T5 [64, 768] (исключает падения размерностей).
- Fused BMM Heads + Static KV-Cache + Адаптивный ресемплер скорости.
- Мгновенный выход по Ctrl+C (os._exit).
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
    try: os.setpriority(os.PRIO_PROCESS, 0, -10)
    except Exception: pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception: pass

from audiocraft.models import MusicGen
from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import StreamingMultiheadAttention, _get_attention_time_dimension
from audiocraft.modules.conditioners import T5Conditioner

SAMPLE_RATE = 32000
FRAME_RATE = 50
MAX_T5_TOKENS = 64

# -----------------------------------------------------------------------------
# STATIC KV-CACHE
# -----------------------------------------------------------------------------
_orig_complete_kv = StreamingMultiheadAttention._complete_kv

def fast_complete_kv(self, k, v):
    if self.cross_attention or not self._is_streaming:
        return _orig_complete_kv(self, k, v)
    time_dim = _get_attention_time_dimension(self.memory_efficient)
    if 'static_k' not in self._streaming_state:
        shape_k = list(k.shape)
        shape_k[time_dim] = 1024
        self._streaming_state['static_k'] = torch.zeros(shape_k, device=k.device, dtype=k.dtype)
        if v is not k:
            self._streaming_state['static_v'] = torch.zeros(shape_k, device=v.device, dtype=v.dtype)
        self._streaming_state['curr_len'] = 0

    curr_len = self._streaming_state['curr_len']
    slen = k.shape[time_dim]
    static_k = self._streaming_state['static_k']
    if curr_len + slen > static_k.shape[time_dim]:
        return _orig_complete_kv(self, k, v)

    if time_dim == 2: static_k[:, :, curr_len:curr_len+slen, :] = k
    else: static_k[:, curr_len:curr_len+slen, :] = k

    if v is not k:
        static_v = self._streaming_state['static_v']
        if time_dim == 2: static_v[:, :, curr_len:curr_len+slen, :] = v
        else: static_v[:, curr_len:curr_len+slen, :] = v
        nv = static_v[:, :, :curr_len+slen, :] if time_dim == 2 else static_v[:, :curr_len+slen, :, :]
    else:
        nv = static_k[:, :, :curr_len+slen, :] if time_dim == 2 else static_k[:, :curr_len+slen, :, :]

    nk = static_k[:, :, :curr_len+slen, :] if time_dim == 2 else static_k[:, :curr_len+slen, :, :]
    self._streaming_state['curr_len'] = curr_len + slen
    return nk, nv

StreamingMultiheadAttention._complete_kv = fast_complete_kv

# -----------------------------------------------------------------------------
# FUSED BMM HEADS
# -----------------------------------------------------------------------------
def fast_lm_generate(self, prompt=None, conditions=[], num_samples=None, max_gen_len=256,
                     use_sampling=True, temp=1.0, top_k=250, top_p=0.0, cfg_coef=1.0,
                     cfg_coef_beta=None, two_step_cfg=None, remove_prompts=False,
                     check=False, callback=None) -> torch.Tensor:
    assert not self.training
    first_param = next(iter(self.parameters()))
    device = first_param.device

    if conditions:
        tokenized = self.condition_provider.tokenize(conditions)
        cfg_conditions = self.condition_provider(tokenized)
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

    with self.streaming():
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]
        for offset in range(start_offset_sequence, gen_sequence_len):
            curr_sequence = gen_sequence[..., prev_offset:offset]
            input_ = (self.emb[0](curr_sequence[:, 0]) + self.emb[1](curr_sequence[:, 1]) +
                      self.emb[2](curr_sequence[:, 2]) + self.emb[3](curr_sequence[:, 3]))

            out = self.transformer(input_, cross_attention_src=cross_attn_src)
            if self.out_norm: out = self.out_norm(out)

            out_last = out[:, -1, :].unsqueeze(0).expand(4, B, -1)
            if self._fused_head_b is not None:
                logits = torch.baddbmm(self._fused_head_b.unsqueeze(1), out_last, self._fused_head_w.transpose(1, 2))
            else:
                logits = torch.bmm(out_last, self._fused_head_w.transpose(1, 2))

            logits = logits.permute(1, 0, 2)
            if use_sampling and temp > 0.0:
                top_logits, top_indices = torch.topk(logits / temp, k=min(top_k, 50), dim=-1)
                top_probs = torch.softmax(top_logits, dim=-1)
                sample_idx = torch.multinomial(top_probs.view(-1, min(top_k, 50)), 1)
                next_token = torch.gather(top_indices.view(-1, min(top_k, 50)), -1, sample_idx).view(B, K, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
            next_token[~valid_mask] = self.special_token_id
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] == unknown_token, next_token, gen_sequence[..., offset:offset+1]
            )
            prev_offset = offset

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
        if embeds.ndim == 2: embeds = embeds.unsqueeze(0)
        b_size = inputs['input_ids'].shape[0] if ('input_ids' in inputs) else (inputs['attention_mask'].shape[0] if 'attention_mask' in inputs else 1)
        if embeds.shape[0] != b_size: embeds = embeds.expand(b_size, -1, -1)
        mask = torch.ones((embeds.shape[0], embeds.shape[1]), device=target_device, dtype=torch.long)
        return self.output_proj(embeds), mask
    return _orig_t5_forward(self, inputs)

T5Conditioner.forward = intercepted_t5_forward

active_listener, active_conn = None, None

def graceful_shutdown(sig, frame):
    global active_listener, active_conn
    print("\n🛑 [MUSICGEN-SERVER] Завершение работы по Ctrl+C...")
    if active_conn:
        try: active_conn.close()
        except Exception: pass
    if active_listener:
        try: active_listener.close()
        except Exception: pass
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
        if peak > 0.92: chunk = chunk / (peak + 1e-5) * 0.92
        t_len = chunk.shape[-1]
        if t_len <= self.fade_len: return chunk
        if self.overlap is not None:
            chunk[..., :self.fade_len] = chunk[..., :self.fade_len] * self.fade_in + self.overlap * self.fade_out
        self.overlap = chunk[..., -self.fade_len:].clone()
        return chunk[..., :-self.fade_len]

def adapt_system(gen_time, audio_duration, current_speed, current_block, target_block, safety_factor):
    gen_time = max(gen_time, 0.000001)
    real_rtf = audio_duration / gen_time

    if current_block < target_block: current_block *= 1.05
    else: current_block *= 0.95
    current_block = max(0.5, min(8.0, current_block))

    base_speed = real_rtf * safety_factor
    alpha = 0.10
    current_speed = (current_speed * (1.0 - alpha)) + (base_speed * alpha)
    current_speed = max(0.4, min(1.2, current_speed))
    return current_speed, current_block, real_rtf

def resample_chunk_gpu(wav, speed):
    if abs(speed - 1.0) < 0.01: return wav
    new_len = int(wav.shape[-1] / speed)
    if new_len < 1: return wav
    return F.interpolate(wav.float(), size=new_len, mode='linear', align_corners=False)

def main():
    global ACTIVE_T5_EMBEDS, active_listener, active_conn
    parser = argparse.ArgumentParser(description="NeuroCanvas Turbo MusicGen Server")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument('--model', type=str, default="facebook/musicgen-small")
    args = parser.parse_args()

    print("=" * 70)
    print(f"🎵 ИНИЦИАЛИЗАЦИЯ MUSICGEN НА {DEVICE} (FP16)...")
    print("=" * 70)

    mg = MusicGen.get_pretrained(args.model, device=DEVICE)
    mg.lm.eval()
    mg.compression_model.eval()
    if DEVICE.type == 'cuda':
        mg.lm.to(torch.float16)
        mg.compression_model.to(torch.float16)

    t5_mod = mg.lm.condition_provider.conditioners['description'].t5
    raw_emb = t5_mod.shared.weight.detach().to(torch.float32)
    norm_emb = raw_emb / (torch.norm(raw_emb, dim=-1, keepdim=True) + 1e-7)
    _, _, V_all = torch.pca_lowrank(norm_emb, q=120)
    proj_120_to_768 = V_all[:, :120].T.contiguous().to(DEVICE)

    crossfader = AudioCrossfaderGPU(fade_len=512)
    current_tokens = None
    current_speed = 1.0
    current_block = 2.4

    active_listener = Listener(('localhost', args.port), authkey=b'music')
    print(f"🚀 [MUSICGEN-SERVER] Слушаю порт {args.port}...")

    while True:
        try:
            active_conn = active_listener.accept()
            print("🔗 [MUSICGEN-SERVER] Клиент подключен!")
            current_tokens = None
            crossfader.overlap = None

            while True:
                if not active_conn.poll(0.05): continue
                try: msg = active_conn.recv()
                except (EOFError, ConnectionResetError): break

                cmd = msg.get('cmd')

                if cmd == 'init_lore':
                    prompts = msg.get('prompts', [])
                    encoded_bases = []
                    t5_cond = mg.lm.condition_provider.conditioners['description']
                    with torch.inference_mode(), t5_cond.autocast:
                        for p in prompts:
                            # ФИКС: Строго фиксированная длина 64 токена для ВСЕХ концептов
                            inp = t5_cond.t5_tokenizer(
                                [p], return_tensors='pt', padding="max_length", 
                                max_length=MAX_T5_TOKENS, truncation=True
                            ).to(DEVICE)
                            emb = t5_cond.t5(**inp).last_hidden_state[0]  # [64, 768]
                            encoded_bases.append(emb.cpu().numpy())
                    active_conn.send({
                        'status': 'ok',
                        'c_bases': encoded_bases,
                        'proj_120_to_768': proj_120_to_768.cpu().numpy()
                    })

                elif cmd == 'generate_step':
                    t0 = time.time()
                    embeds_np = msg.get('t5_embeds')
                    block_sec = float(msg.get('block_sec', 2.4))
                    context_sec = float(msg.get('context_sec', 0.4))
                    temp = float(msg.get('temp', 1.0))
                    top_k = int(msg.get('top_k', 250))
                    cfg_coef = float(msg.get('cfg_coef', 1.0))
                    safety_factor = float(msg.get('safety_factor', 1.0))
                    is_reset = bool(msg.get('is_reset', False))

                    if embeds_np is not None:
                        ACTIVE_T5_EMBEDS = torch.from_numpy(embeds_np).to(DEVICE)

                    with torch.inference_mode():
                        if is_reset or current_tokens is None:
                            crossfader.overlap = None
                            mg.set_generation_params(
                                duration=block_sec, cfg_coef=cfg_coef,
                                use_sampling=True, top_k=top_k, temperature=temp
                            )
                            init_audio, current_tokens = mg.generate([""], progress=False, return_tokens=True)
                            clean = crossfader.process(init_audio)
                            pcm = clean[0, 0].cpu().float().numpy()
                        else:
                            ctx_tokens = max(1, int(context_sec * FRAME_RATE))
                            prompt_tokens = current_tokens[..., -ctx_tokens:]
                            gen_tokens = max(1, int(block_sec * FRAME_RATE))
                            total_tokens = prompt_tokens.shape[-1] + gen_tokens
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

                            current_tokens = out_tokens
                            new_tokens = out_tokens[..., prompt_tokens.shape[-1]:]
                            tokens_to_decode = out_tokens[..., -(new_tokens.shape[-1] + 2):]

                            with torch.no_grad():
                                decoded_audio = mg.compression_model.decode(tokens_to_decode, None)

                            new_chunk = decoded_audio[..., 1280:]
                            if DEVICE.type == 'cuda': torch.cuda.synchronize()

                            raw_audio_len = new_chunk.shape[-1] / float(SAMPLE_RATE)
                            gen_time = time.time() - t0

                            current_speed, current_block, real_rtf = adapt_system(
                                gen_time, raw_audio_len, current_speed, current_block, block_sec, safety_factor
                            )
                            processed_gpu = resample_chunk_gpu(new_chunk, current_speed)
                            clean = crossfader.process(processed_gpu)
                            pcm = clean[0, 0].cpu().float().numpy()

                    active_conn.send({
                        'pcm': pcm,
                        'gen_time': time.time() - t0,
                        'audio_len': len(pcm) / float(SAMPLE_RATE),
                        'rtf': real_rtf if 'real_rtf' in locals() else 1.0,
                        'play_speed': current_speed
                    })

                elif cmd == 'flush':
                    current_tokens = None
                    crossfader.overlap = None
                    active_conn.send({'status': 'ok'})

        except Exception as e:
            time.sleep(0.1)

if __name__ == '__main__':
    main()
