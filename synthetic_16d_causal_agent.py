#!/usr/bin/env python3
"""
🤖 SYNTHETIC ACTIVE INFERENCE SWARM v2700.0 (PURE STIGMERGIC ISOLATION)
- Каждый агент замкнут в своем Марковском одеяле и видит ТОЛЬКО холст.
- Агенты ничего не знают о других агентах, их числе или решениях оркестратора.
- Выход агента: чистый 16-канальный ЭЭГ поток в свой независимый LSL-аутлет.
"""

import time
import math
import json
import os
import traceback
import threading
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 20
NUM_CHANNELS = 16
NUM_DEVICES = 8
TWO_PI = 2.0 * math.pi
VIDEO_BUFFER_LEN = 16

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0)).astype(np.float32)

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]

class CorticalMontage:
    def __init__(self, config_dict):
        self.devices = config_dict.get('montage', {}).get('devices', [])

    def get_nodes_by_names(self, names: list) -> list[int]:
        matched_ids = []
        for dev in self.devices:
            if any(dev['name'].startswith(n) for n in names):
                matched_ids.append(dev['id'])
        return list(set(matched_ids))

    def get_nodes_by_radius(self, anchor_xyz: list, radius: float) -> list[int]:
        matched_ids = []
        anchor = np.array(anchor_xyz)
        for dev in self.devices:
            dist = np.linalg.norm(np.array(dev['coords']) - anchor)
            if dist <= radius:
                matched_ids.append(dev['id'])
        return list(set(matched_ids))
        
    def check_feature(self, active_nodes: list, feature_name: str) -> bool:
        for dev in self.devices:
            if dev['id'] in active_nodes and feature_name in dev['name']:
                return True
        return False

class BaseActiveAgent:
    def __init__(self, bot_id: int, initial_idx: int, active_nodes: list, montage: CorticalMontage, num_concepts: int = 8):
        self.bot_id = bot_id
        self.current_idx = initial_idx
        self.original_idx = initial_idx
        self.num_concepts = num_concepts
        self.active_nodes = active_nodes
        
        # Полный набор 5 префронтальных зон
        self.has_f3  = montage.check_feature(active_nodes, "F3")
        self.has_f4  = montage.check_feature(active_nodes, "F4")
        self.has_afz = montage.check_feature(active_nodes, "AFz")
        self.has_fpz = montage.check_feature(active_nodes, "Fpz")
        self.has_fcz = montage.check_feature(active_nodes, "FCz")
        
        all_other = [i for i in range(num_concepts) if i != initial_idx]
        np.random.seed(bot_id * 101)
        self.plan_b_queue = list(np.random.permutation(all_other)) if all_other else [initial_idx]
        
        self.sprt_log_evidence = 0.0
        self.u_torus = (initial_idx * TWO_PI / num_concepts)
        self.v_torus = ((initial_idx * 3) * TWO_PI / num_concepts) % TWO_PI
        self.steps_in_role = 0

    def trigger_fpz_cognitive_branch(self):
        if not self.has_fpz or not self.plan_b_queue: return False 
        self.sprt_log_evidence = 0.0
        old_idx = self.current_idx
        self.plan_b_queue.append(old_idx)
        
        next_idx = self.plan_b_queue.pop(0)
        self.current_idx = next_idx
        self.original_idx = next_idx
        self.u_torus = (self.current_idx * TWO_PI / self.num_concepts)
        self.v_torus = ((self.current_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
        self.steps_in_role = 0
        print(f"🔀 [Fpz ВЕТВЛЕНИЕ АГЕНТА {self.bot_id}]: {ALL_NAMES[old_idx % len(ALL_NAMES)]} ➔ {ALL_NAMES[next_idx % len(ALL_NAMES)]}")
        return True

    def generate_waves(self, dp_so3: np.ndarray | None, t_vec: np.ndarray, theta_norm: np.ndarray, is_affordance: bool):
        freq = 35.0 + (self.current_idx % 8) * 3.5
        offset_ch = np.linspace(-1.0, 1.0, NUM_CHANNELS, dtype=np.float32)
        spatial_phase_ch = (COORDS_X * math.cos(self.u_torus) + COORDS_Y * math.sin(self.v_torus)) * 0.18 + offset_ch

        target_phase = 0.45 if is_affordance else 0.22
        amp = 4.5 if is_affordance else 5.2

        w_theta = np.exp(-((theta_norm - target_phase)**2) / 0.025)
        low_gamma = np.sin(TWO_PI * freq * t_vec[None, :] + spatial_phase_ch[:, None]) * w_theta[None, :] * amp

        high_ripple = np.zeros_like(low_gamma)
        if dp_so3 is not None or is_affordance:
            w_ripple = np.exp(-((theta_norm - ((target_phase + 0.12) % 1.0))**2) / 0.015)
            phase_ripple = (COORDS_X * 0.3 + COORDS_Y * 0.2 + COORDS_Z * 0.5) * 0.25
            high_ripple = np.sin(TWO_PI * 89.5 * t_vec[None, :] + phase_ripple[:, None]) * w_ripple[None, :] * 4.5
            
        return low_gamma, high_ripple

class FullDynamicHardcodedBot(BaseActiveAgent):
    def step(self, canvas_probs: np.ndarray):
        """Агент воспринимает ТОЛЬКО холст. Никаких знаний о других агентах!"""
        target_p = float(canvas_probs[self.current_idx % len(canvas_probs)]) if len(canvas_probs) > 0 else 0.0
        dominant_idx = int(np.argmax(canvas_probs)) if len(canvas_probs) > 0 else 0
        dominant_p = float(canvas_probs[dominant_idx]) if len(canvas_probs) > 0 else 0.0

        # Если образ агента виден на холсте — ошибка мала
        if target_p >= 0.25:
            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.20)
            self.steps_in_role += 1
            if self.steps_in_role > 60: 
                self.trigger_fpz_cognitive_branch()
            return False, None, self.sprt_log_evidence

        self.steps_in_role = 0

        # Если на холсте доминирует ДРУГОЙ образ — растет ошибка предсказания!
        if dominant_idx != self.current_idx and dominant_p >= 0.20:
            evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4))
            self.sprt_log_evidence += 0.008 * evidence_step

            # При накоплении ошибки — Fpz когнитивное ветвление
            if self.sprt_log_evidence > 1.45:
                self.trigger_fpz_cognitive_branch()
                return False, None, self.sprt_log_evidence

            # Агент генерирует рипплы аффорданса, пытаясь встроиться в то, что видит
            return True, None, self.sprt_log_evidence

        self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
        return False, None, self.sprt_log_evidence

class FullJepaVideoAgent(BaseActiveAgent):
    def __init__(self, bot_id: int, initial_idx: int, active_nodes: list, montage: CorticalMontage, jepa_wrapper, num_concepts: int = 8):
        super().__init__(bot_id, initial_idx, active_nodes, montage, num_concepts)
        self.jepa = jepa_wrapper
        self.video_buffer = []

    def push_frame(self, frame_rgb: np.ndarray):
        self.video_buffer.append(frame_rgb)
        if len(self.video_buffer) > VIDEO_BUFFER_LEN: self.video_buffer.pop(0)

    def step(self, canvas_probs: np.ndarray):
        if len(self.video_buffer) < 2: return False, None, self.sprt_log_evidence
        target_p = float(canvas_probs[self.current_idx % len(canvas_probs)]) if len(canvas_probs) > 0 else 0.0
        dominant_idx = int(np.argmax(canvas_probs)) if len(canvas_probs) > 0 else 0
        dominant_p = float(canvas_probs[dominant_idx]) if len(canvas_probs) > 0 else 0.0

        if target_p >= 0.25:
            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.25)
            self.steps_in_role += 1
            if self.steps_in_role > 60:
                self.trigger_fpz_cognitive_branch()
            return False, None, self.sprt_log_evidence

        self.steps_in_role = 0

        if dominant_idx != self.current_idx and dominant_p >= 0.20:
            evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4))
            self.sprt_log_evidence += 0.009 * evidence_step

            if self.sprt_log_evidence > 1.45:
                self.trigger_fpz_cognitive_branch()
                return False, None, self.sprt_log_evidence

            return True, None, self.sprt_log_evidence

        self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
        return False, None, self.sprt_log_evidence

class AutonomousSwarmProcess(mp.Process):
    def __init__(self, shm, config_path: str, num_hardcoded: int, num_jepa: int, num_concepts: int = 8, sps: float = 250.0):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.config_path = config_path
        self.num_hardcoded = num_hardcoded
        self.num_jepa = num_jepa
        self.num_concepts = num_concepts
        self.sps = float(sps)

    def run(self):
        FS = self.sps
        try:
            config_dict = {
                "montage": {
                    "system": "10-20-extended",
                    "devices": [
                        {"id": 0, "name": "F3",  "coords": [-4.0, 3.0, 5.0]},
                        {"id": 1, "name": "F4",  "coords": [4.0, 3.0, 5.0]},
                        {"id": 2, "name": "AFz", "coords": [0.0, 6.0, 4.0]},
                        {"id": 3, "name": "Fpz", "coords": [0.0, 8.0, 2.0]}
                    ]
                },
                "agents": []
            }
            all_targets = ["F3", "F4", "AFz", "Fpz", "FCz"]
            montage = CorticalMontage(config_dict)

            num_outlets = min(NUM_DEVICES, max(4, self.num_hardcoded + self.num_jepa))
            outlets = [
                StreamOutlet(StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}'))
                for i in range(num_outlets)
            ]

            jepa_wrapper = None
            if self.num_jepa > 0:
                try:
                    from vla_jepa_wrapper import VLA_JEPA_Wrapper
                    jepa_wrapper = VLA_JEPA_Wrapper(port=6001)
                except Exception:
                    pass

            rng = np.random.RandomState(42)
            available_slots = list(rng.permutation(self.num_concepts))
            self.bots = []
            bot_idx = 1

            for _ in range(self.num_hardcoded):
                c_idx = available_slots[(bot_idx - 1) % len(available_slots)]
                self.bots.append(FullDynamicHardcodedBot(bot_idx, c_idx, [0, 1, 2, 3], montage, self.num_concepts))
                bot_idx += 1

            for _ in range(self.num_jepa):
                c_idx = available_slots[(bot_idx - 1) % len(available_slots)]
                self.bots.append(FullJepaVideoAgent(bot_idx, c_idx, [0, 1, 2, 3], montage, jepa_wrapper, self.num_concepts))
                bot_idx += 1

            start_time = time.time()
            regional_delays = [0.0, 0.040, 0.080, 0.120]

            def async_jepa_worker():
                while self.shm['is_running'].value:
                    try:
                        if not self.shm['is_calibrating'].value:
                            raw_buf = np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8)
                            if np.any(raw_buf > 0):
                                frame_copy = raw_buf.reshape(384, 512, 3).copy()
                                for bot in self.bots:
                                    if isinstance(bot, FullJepaVideoAgent):
                                        bot.push_frame(frame_copy)
                    except Exception:
                        pass
                    time.sleep(0.08)

            if any(isinstance(b, FullJepaVideoAgent) for b in self.bots):
                threading.Thread(target=async_jepa_worker, daemon=True).start()

            self.shm['is_swarm_ready'].value = True

            while self.shm['is_running'].value:
                dt = CHUNK_SIZE / FS
                t_now = time.time() - start_time
                t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False, dtype=np.float32)
                theta_phase_norm = ((TWO_PI * 6.0 * t_vec) % TWO_PI) / TWO_PI

                is_calib = self.shm['is_calibrating'].value
                canvas_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)
                
                status_str = "СВОРМ (4 АГЕНТА):"
                
                for b_idx, bot in enumerate(self.bots):
                    if not is_calib:
                        is_affordance, dp, sprt_val = bot.step(canvas_probs)
                    else:
                        tgt = self.shm['calib_target_idx'].value
                        bot.current_idx = tgt
                        is_affordance, dp, sprt_val = False, None, 1.0

                    lg, hg = bot.generate_waves(dp, t_vec, theta_phase_norm, is_affordance)
                    
                    # КАЖДЫЙ БОТ ВЕЩАЕТ ТОЛЬКО В СВОЙ УЗЕЛ LSL
                    if b_idx < num_outlets:
                        delay = regional_delays[b_idx % len(regional_delays)]
                        noise = np.random.normal(0, 0.035, (NUM_CHANNELS, CHUNK_SIZE))
                        raw_sig = (
                            3.0 * np.sin(TWO_PI * 6.0 * t_vec + delay)[None, :] +
                            2.0 * np.sin(TWO_PI * 1.5 * t_vec + delay)[None, :] +
                            lg + hg +
                            np.sin(TWO_PI * 22.0 * t_vec)[None, :] * 1.8 +
                            noise
                        )
                        outlets[b_idx].push_chunk(raw_sig.T.tolist())

                    p_type = "H" if isinstance(bot, FullDynamicHardcodedBot) else "J"
                    status_str += f" [Аг{bot.bot_id}({p_type}): {ALL_NAMES[bot.current_idx % len(ALL_NAMES)][:5]}|E:{sprt_val:.2f}]"

                self.shm['agent_desc'].value = status_str.encode('utf-8', errors='replace')[:250]
                time.sleep(dt)

        except Exception as err:
            print(f"❌ [SWARM CRASH]: {err}")
            traceback.print_exc()

class SyntheticAutonomousAgent:
    def __init__(self, config_path: str = "swarm_config.json", num_hardcoded: int = 4, num_jepa: int = 4, num_concepts: int = 8, sps: float = 250.0):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.num_bots = num_hardcoded + num_jepa
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'is_swarm_ready': ctx.Value('b', False),
            'sensory_sub': ctx.Value('b', False), 
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [0.0] * 256),
            'swarm_priors': ctx.Array('d', [0.0] * 256), 
            'raw_rgb_frame': ctx.Array('B', 384 * 512 * 3),
            'agent_desc': ctx.Array('c', 256),
            'concept_owners': ctx.Array('i', [-1] * 8),
            'parent_map': ctx.Array('i', [-1] * 16)
        }
        self.process = AutonomousSwarmProcess(
            self.shm, config_path, num_hardcoded=num_hardcoded, num_jepa=num_jepa, num_concepts=num_concepts, sps=sps
        )
        self.process.start()

    def is_ready(self) -> bool: return bool(self.shm['is_swarm_ready'].value)
    def is_alive(self) -> bool: return self.process.is_alive()
    def set_sensory_substitution(self, state: bool): self.shm['sensory_sub'].value = bool(state)
    def update_swarm_priors(self, priors: np.ndarray):
        for i in range(min(self.num_concepts, len(priors))): self.shm['swarm_priors'][i] = float(priors[i])
    def update_visual_state(self, visual_data):
        if isinstance(visual_data, np.ndarray):
            if visual_data.ndim == 3:
                np.copyto(np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8), visual_data.reshape(-1))
            elif visual_data.ndim == 1:
                for i in range(min(self.num_concepts, len(visual_data))): self.shm['clip_probs'][i] = float(visual_data[i])
    def get_parent_map(self):
        pmap = list(self.shm['parent_map'][:self.num_concepts])
        return {i: (p if p != -1 else None) for i, p in enumerate(pmap)}
    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)
    def get_telemetry(self): return self.shm['agent_desc'].value.decode('utf-8', errors='replace').replace('\x00', '').strip()
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
