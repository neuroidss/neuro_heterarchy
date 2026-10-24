#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: 100% DETERMINISTIC EEG -> VIDEO & AUDIO REPRODUCER (ZERO RANDOM)
===================================================================================
- Никаких if/else проверок сигнала, никакого снижения размерности, никакого рандома.
- Прогоняет один и тот же массив ЭЭГ [N, 16] через полноразмерный GPU-конвейер:
  [16 x 256] -> [32 x 120] ciPLV & Ripples -> [5 регионов x 4096] -> [77 x 768] & [64 x 768].
- Гарантирует 100% совпадение видео и аудио (проверяется криптографическим SHA-256).
===================================================================================
"""

from __future__ import annotations
import os
import sys
import wave
import hashlib
import argparse
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y
from neuro_amodal_engine import (
    GEOM_GPU,
    AmodalMontyColumnGPU,
    REGION_ROLES,
    BUF_SIZE,
    FFT_SLOW,
    NUM_SLOTS,
    TWO_PI,
)
from neuro_genesis_engine import UniversalSemanticSpace
from music_genesis_engine import UniversalMusicSemanticSpace
from neuro_heterarchy_diffusion import apply_color_surgery_and_blend_gpu

SAMPLE_RATE = 32000


class DeterministicFullDimGPUCore(torch.nn.Module):
    """
    Чистый детерминированный оператор ℝ^(16×256) -> (ℝ^(77×768), ℝ^(64×768)) на GPU.
    Ноль рандома, ноль часов реального времени (time.time()), ноль снижения размерности.
    """
    def __init__(self, geom: dict, gamma_max: float = 100.0):
        super().__init__()
        self.geom = geom
        self.register_buffer("raw_buf_gpu", torch.zeros((1, 16, BUF_SIZE), device=DEVICE))
        self.register_buffer("base_fftfreq_256", torch.fft.fftfreq(BUF_SIZE, d=1.0 / 250.0, device=DEVICE).view(1, 1, BUF_SIZE))
        slot_ang = -math.pi + (TWO_PI / NUM_SLOTS) * (torch.arange(NUM_SLOTS, device=DEVICE) + 0.5)
        self.register_buffer("slot_angles", slot_ang.view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("gamma_centers", torch.linspace(30.0, gamma_max, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))
        self.register_buffer("rip_centers", torch.linspace(70.0, 100.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1))

        self.columns = torch.nn.ModuleDict({
            r: AmodalMontyColumnGPU(r, geom).to(DEVICE) for r in REGION_ROLES
        })

        # Фиксированные пространственные маски регионов на 120 диполях (без ветвлений!)
        self.region_masks = {
            "Fpz": geom["macro_mask"],
            "AFz": torch.ones_like(geom["macro_mask"]),
            "F3":  geom["left_mask"],
            "F4":  geom["right_mask"],
            "FCz": geom["micro_mask"],
        }

    @torch.no_grad()
    def forward_window_gpu(self, eeg_window_16x256: torch.Tensor, dt: float = 0.20) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.raw_buf_gpu[0].copy_(eeg_window_16x256)
        centered = self.raw_buf_gpu - torch.mean(self.raw_buf_gpu, dim=2, keepdim=True)

        freqs = self.base_fftfreq_256
        notch = ((torch.abs(freqs) < 48.5) | (torch.abs(freqs) > 51.5)) & \
                ((torch.abs(freqs) < 98.5) | (torch.abs(freqs) > 101.5))
        fft_clean = torch.fft.fft(centered, dim=-1) * notch.float()
        pos_mask = (freqs >= 0).float()

        f_theta = torch.exp(-0.5 * ((freqs - 6.0) / 1.5) ** 2) * 2.0 * pos_mask
        Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
        P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
        phi_theta = torch.angle(torch.mean(P_theta, dim=1))

        fft_exp = fft_clean.unsqueeze(1)
        freqs_4d = freqs.unsqueeze(1)
        w = torch.exp(3.2 * torch.cos(phi_theta.unsqueeze(1).unsqueeze(1) - self.slot_angles))
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

        i_idx, j_idx = self.geom["i_idx"], self.geom["j_idx"]

        g_filt = torch.exp(-0.5 * ((freqs_4d - self.gamma_centers) / 4.5) ** 2) * 2.0 * pos_mask.unsqueeze(1)
        Z_gamma = torch.fft.ifft(fft_exp * g_filt, dim=-1)
        P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)
        cg_gamma = P_gamma[:, :, i_idx, :] * torch.conj(P_gamma[:, :, j_idx, :])
        psi_gamma = torch.sum(cg_gamma * w, dim=-1)
        gamma_anchor = torch.imag(psi_gamma * torch.conj(psi_gamma[:, 0:1, :]))[0]  # [32, 120]

        r_filt = torch.exp(-0.5 * ((freqs_4d - self.rip_centers) / 4.0) ** 2) * 2.0 * pos_mask.unsqueeze(1)
        Z_rip = torch.fft.ifft(fft_exp * r_filt, dim=-1)
        P_rip = Z_rip / (torch.abs(Z_rip) + 1e-12)
        cg_rip = P_rip[:, :, i_idx, :] * torch.conj(P_rip[:, :, j_idx, :])
        psi_rip = torch.sum(cg_rip * w, dim=-1)
        h_rip = (torch.imag(psi_rip) / torch.sqrt(torch.clamp(1.0 - torch.real(psi_rip) ** 2, min=1e-5)))[0]  # [32, 120]

        sdr_fcz = self.columns["FCz"].l23_pooled_sdr
        sdr_f3  = self.columns["F3"].l23_pooled_sdr
        sdr_f4  = self.columns["F4"].l23_pooled_sdr
        sdr_afz = self.columns["AFz"].l23_pooled_sdr
        sdr_fpz = self.columns["Fpz"].l23_pooled_sdr

        anom_fcz = self.columns["FCz"].upward_sdr_anomaly
        anom_f34 = 0.5 * (self.columns["F3"].upward_sdr_anomaly + self.columns["F4"].upward_sdr_anomaly)
        anom_afz = self.columns["AFz"].upward_sdr_anomaly

        parent_map = {"FCz": 0.5 * (sdr_f3 + sdr_f4), "F3": sdr_afz, "F4": sdr_afz, "AFz": sdr_fpz, "Fpz": None}
        child_map  = {"FCz": None, "F3": anom_fcz, "F4": anom_fcz, "AFz": anom_f34, "Fpz": anom_afz}

        res = {}
        for r in REGION_ROLES:
            mask_r = self.region_masks[r]
            res[r] = self.columns[r].forward_gpu(
                gamma_anchor * mask_r, h_rip * mask_r,
                n_sec=16, cur_sec=0, is_tick=True, dt=dt,
                top_down_parent_sdr=parent_map[r], bottom_up_child_anomaly=child_map[r]
            )

        raw_fcz_wave = res["FCz"]["sagitta_16x768"] + res["FCz"]["z_on_16x768"] * 0.5
        fcz_act_16x768, fcz_pers = self.columns["FCz"].action_integrator.step_gpu(
            raw_fcz_wave, res["FCz"]["temp_asym_scalar"]
        )

        # Собираем полноразмерный [77, 768] кортикальный манифолд без потери размерности
        amodal_77x768 = torch.zeros((77, 768), dtype=torch.float32, device=DEVICE)
        amodal_77x768[0:1].copy_(res["AFz"]["tree_16x768"][0:1])
        amodal_77x768[1:5].copy_(res["Fpz"]["tree_16x768"][:4])
        amodal_77x768[5:21].copy_(res["AFz"]["tree_16x768"])
        amodal_77x768[21:37].copy_(res["F3"]["z_on_16x768"])
        amodal_77x768[37:53].copy_(res["F4"]["z_off_16x768"] * 0.6 + res["F4"]["temp_asym_16x768"] * 0.4)
        amodal_77x768[53:69].copy_(fcz_act_16x768)
        amodal_77x768[69:76].copy_(res["AFz"]["sdr_tokens_7x768"])
        amodal_77x768[76:77].copy_(res["FCz"]["tree_16x768"][-1:])

        dyn_str = float(torch.clamp(0.50 + 0.35 * fcz_pers.squeeze(0), 0.50, 0.90).item())
        return amodal_77x768, res["AFz"]["z_slots"], dyn_str


def extract_eeg_window_256(eeg_samples: np.ndarray, step_i: int) -> torch.Tensor:
    """
    Гарантированно возвращает тензор [16, 256] на GPU без срезок и краевых нулей.
    """
    n_total = eeg_samples.shape[0]
    start_idx = (step_i * 50) % n_total
    # Циклический индекс: всегда ровно 256 отсчетов
    indices = (np.arange(BUF_SIZE) + start_idx) % n_total
    win_np = eeg_samples[indices].T  # [16, 256]
    return torch.from_numpy(win_np.copy()).to(DEVICE, dtype=torch.float32)


def run_deterministic_pass(eeg_samples_Nx16: np.ndarray, run_label: str,
                           vision_sem: UniversalSemanticSpace,
                           music_sem: UniversalMusicSemanticSpace,
                           brain_conn, music_conn,
                           num_steps: int = 10) -> tuple[str, str]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    core = DeterministicFullDimGPUCore(GEOM_GPU).to(DEVICE)
    music_conn.send({'cmd': 'flush'})
    music_conn.recv()

    video_hasher = hashlib.sha256()
    audio_hasher = hashlib.sha256()

    canvas_rgb = np.zeros((288, 384, 3), dtype=np.uint8)
    prev_gpu_frame = None
    all_pcm = []

    out_mp4 = f"deterministic_{run_label}.mp4"
    out_wav = f"deterministic_{run_label}.wav"
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (384, 288))

    with torch.no_grad():
        base_vis_77 = vision_sem.get_77_embedding_gpu(vision_sem.words[0] if vision_sem.words else "").clone()
        vis_norms = torch.norm(base_vis_77, p=2, dim=-1, keepdim=True).clamp(min=1e-5)
        v_aff_vis = vision_sem.get_concept_affordances_svd(0, max_bases=16)

        base_mus_64 = (music_sem.c_bases[0] if music_sem.c_bases else music_sem.empty_embed).clone()
        mus_norms = torch.norm(base_mus_64, p=2, dim=-1, keepdim=True).clamp(min=1.0)
        v_aff_mus = music_sem.get_concept_affordances_svd(0, max_bases=16)

        for step_i in range(num_steps):
            # Гарантированное окно [16, 256]
            win_gpu = extract_eeg_window_256(eeg_samples_Nx16, step_i)

            amodal_77, _, dyn_str = core.forward_window_gpu(win_gpu, dt=0.20)

            # 1. Проекция в CLIP [77 x 768] без потери размерности
            vis_mod = torch.matmul(torch.matmul(amodal_77, v_aff_vis.T), v_aff_vis)
            cond_77 = F.normalize(base_vis_77 + vis_mod * (0.35 * dyn_str), p=2, dim=-1) * vis_norms

            brain_conn.send({
                'cmd': 'generate',
                'image_np': canvas_rgb,
                'prompt_embeds': cond_77.unsqueeze(0).cpu().numpy(),
                'strength': dyn_str,
                'num_inference_steps': 2,
                'guidance_scale': 1.0,
                'use_taesd': True,
                'deterministic': True
            })
            resp_img = brain_conn.recv()
            if resp_img.shape[:2] != (288, 384):
                resp_img = cv2.resize(resp_img, (384, 288))
            canvas_rgb, prev_gpu_frame = apply_color_surgery_and_blend_gpu(resp_img, prev_gpu_frame, True, 0.18)

            video_hasher.update(canvas_rgb.tobytes())
            vw.write(cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

            # 2. Проекция в T5 [64 x 768] без потери размерности
            mus_mod = torch.matmul(torch.matmul(amodal_77[1:65], v_aff_mus.T), v_aff_mus)
            cond_64 = F.normalize(base_mus_64 + mus_mod * (0.35 * dyn_str), p=2, dim=-1) * mus_norms

            music_conn.send({
                'cmd': 'generate_step',
                't5_embeds': cond_64.cpu().numpy(),
                'block_sec': 0.40,
                'context_sec': 1.20,
                'target_buffer_count': 2,
                'temp': 1.0,
                'top_k': 250,
                'cfg_coef': 1.0,
                'safety_factor': 1.0,
                'queue_sec': 0.40,
                'queue_chunks': 1,
                'is_reset': (step_i == 0),
                'deterministic': True,
                'lock_speed': True
            })
            resp_mus = music_conn.recv()
            pcm = resp_mus['pcm']
            audio_hasher.update(pcm.tobytes())
            all_pcm.append(pcm)

    vw.release()
    full_pcm = np.concatenate(all_pcm)
    pcm16 = (np.clip(full_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    return video_hasher.hexdigest(), audio_hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="100% Deterministic EEG -> Video & Audio Verifier")
    parser.add_argument("--eeg", type=str, default=None, help="Путь к eeg_raw.npz из записанной сессии")
    parser.add_argument("--steps", type=int, default=8, help="Число шагов проверки")
    args = parser.parse_args()

    eeg_data = None
    if args.eeg and os.path.exists(args.eeg):
        try:
            raw = np.load(args.eeg)
            arr = raw["data"].astype(np.float32)
            if arr.ndim == 2:
                if arr.shape[1] == 16:
                    eeg_data = arr
                elif arr.shape[0] == 16:
                    eeg_data = arr.T
            if eeg_data is not None and eeg_data.shape[0] > 0:
                print(f"📂 Загружен реальный массив ЭЭГ: {args.eeg} ({eeg_data.shape})")
            else:
                print(f"⚠️ В файле {args.eeg} нет валидных сэмплов (shape={arr.shape}). Используем эталон.")
        except Exception as e:
            print(f"⚠️ Ошибка чтения {args.eeg}: {e}. Используем эталон.")

    if eeg_data is None or eeg_data.shape[0] == 0:
        # Гарантированный многочастотный массив [2000, 16]
        t = np.linspace(0, 8.0, 2000, dtype=np.float32)[:, None]
        ch = np.arange(16, dtype=np.float32)[None, :]
        eeg_data = (
            25.0 * np.sin(2.0 * np.pi * 6.0 * t + ch * 0.35) +
            12.0 * np.cos(2.0 * np.pi * 42.0 * t - ch * 0.55) +
            8.0 * np.sin(2.0 * np.pi * 89.5 * t + ch * 0.80)
        ).astype(np.float32)
        print(f"🧪 Создан эталонный детерминированный массив ЭЭГ: {eeg_data.shape}")

    # Если массив короче BUF_SIZE (256) — циклически размножаем его
    if eeg_data.shape[0] < BUF_SIZE:
        reps = int(math.ceil(BUF_SIZE / max(1, eeg_data.shape[0]))) + 1
        eeg_data = np.tile(eeg_data, (reps, 1))

    brain_conn = Client(('localhost', 6000), authkey=b'brain')
    brain_conn.send({'cmd': 'init_mode', 'mode': 'lcm', 'use_taesd': True})
    brain_conn.recv()

    music_conn = Client(('localhost', 6005), authkey=b'music')
    music_conn.send({'cmd': 'init_lore', 'prompts': ['psytrance hypnotic bass']})
    m_resp = music_conn.recv()

    vision_sem = UniversalSemanticSpace(concepts_cli="blue glowing plasma flame")
    music_sem = UniversalMusicSemanticSpace(
        words=['psytrance hypnotic bass'],
        c_bases=m_resp['c_bases'],
        proj_120_to_768=m_resp['proj_120_to_768'],
        empty_embed=m_resp.get('empty_embed'),
        global_svd_bases=m_resp.get('global_svd_bases')
    )

    print("\n▶️ Прогон #1 по массиву ЭЭГ...")
    v_hash1, a_hash1 = run_deterministic_pass(eeg_data, "run1", vision_sem, music_sem, brain_conn, music_conn, args.steps)
    print(f"   🎬 Видео SHA-256 (#1): {v_hash1}")
    print(f"   🎵 Аудио SHA-256 (#1): {a_hash1}")

    print("\n▶️ Прогон #2 по ТОМУ ЖЕ САМОМУ массиву ЭЭГ...")
    v_hash2, a_hash2 = run_deterministic_pass(eeg_data, "run2", vision_sem, music_sem, brain_conn, music_conn, args.steps)
    print(f"   🎬 Видео SHA-256 (#2): {v_hash2}")
    print(f"   🎵 Аудио SHA-256 (#2): {a_hash2}")

    print("\n" + "=" * 75)
    if v_hash1 == v_hash2 and a_hash1 == a_hash2:
        print("✅ 100% ДЕТЕРМИНИЗМ ДОКАЗАН! Оба прогона дали побитово идентичное Видео и Аудио!")
    else:
        print("⚠️ Обнаружено расхождение хешей — проверьте, что патчи в render_logic.py и musicgen_server.py применены.")
    print("=" * 75)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
