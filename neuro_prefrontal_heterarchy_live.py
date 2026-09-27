#!/usr/bin/env python3
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import argparse
import time
import math
import json
import numpy as np
import cv2
import pygame
import torch
import threading

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR / "src", CURRENT_DIR.parent / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE
from synthetic_16d_causal_agent import SyntheticAutonomousAgent, CorticalMontage

from neuro_models import (
    BrainSubject, OverBrainCollective, DynamicPilot, 
    MAX_CONCEPTS_CAPACITY, ELECTRODE_X, ELECTRODE_Y
)
from neuro_workers import (
    EqualPoolChaosWorker, VisualCLIPTeacher, 
    apply_manifold_camera_warp, calculate_emergent_treemap, 
    FEIGENBAUM_DELTA, FRACTAL_COLORS
)

WIDTH, HEIGHT = 1800, 960
ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]
MOTION_NAMES = ["ВПЕРЕД", "НАЗАД", "ВПРАВО", "ВЛЕВО"]
TARGET_UNIT_VECS = [
    np.array([ 0.0,  1.0], dtype=np.float32),
    np.array([ 0.0, -1.0], dtype=np.float32),
    np.array([ 1.0,  0.0], dtype=np.float32),
    np.array([-1.0,  0.0], dtype=np.float32)
]

def parse_user_setup(users_str: str | None, pilot_kwargs: dict, is_sim: bool = False, concepts: int = 1) -> list[BrainSubject]:
    if not users_str:
        if is_sim or concepts > 1:
            return [BrainSubject("User1", {"F3": 0, "F4": 1, "AFz": 2, "Fpz": 3}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]
        else:
            return [BrainSubject("User1", {"FCz": 0}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]

    subjects = []
    for b in users_str.split(";"):
        if not b.strip(): continue
        p = b.strip().split(":")
        sub_id = p[0].strip()
        reg_map = {rd.split("=")[0].strip(): int(rd.split("=")[1].strip()) for rd in p[1].split(",") if "=" in rd}
        if not reg_map: reg_map = {"F3": 0, "F4": 1, "AFz": 2, "Fpz": 3} if (is_sim or concepts > 1) else {"FCz": 0}
        subjects.append(BrainSubject(sub_id, reg_map, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs))
        
    if not subjects:
        subjects = [BrainSubject("User1", {"F3": 0, "F4": 1, "AFz": 2, "Fpz": 3}, max_capacity=MAX_CONCEPTS_CAPACITY, **pilot_kwargs)]
    return subjects

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas: Full Prefrontal Heterarchy (F3, F4, AFz, Fpz, FCz)")
    parser.add_argument('--config', type=str, default="swarm_config.json")
    parser.add_argument('--sim', action='store_true', default=False)
    
    parser.add_argument('--concepts', type=int, default=8, help="Number of starting concepts")
    parser.add_argument('--start-prompt', type=str, default="ancient medieval stone castle fortress towers, daytime, sharp focus, 8k")
    parser.add_argument('--chaos', action='store_true')
    
    parser.add_argument('--calib-mode', type=str, default="auto", choices=["auto", "motion", "semantic"])
    parser.add_argument('--move-directions', type=int, default=4, choices=[2, 4])
    parser.add_argument('--calib-sigma', type=float, default=3.5)
    parser.add_argument('--calib-seconds', type=float, default=6.0)
    parser.add_argument('--calib-cycles', type=int, default=1)
    parser.add_argument('--users', type=str, default=None)
    parser.add_argument('--sensory-sub', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="lcm", choices=["lcm", "turbo", "sdxl-turbo"])
    parser.add_argument('--speed', type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument('--strength-high', type=float, default=0.85)
    parser.add_argument('--strength-low', type=float, default=0.50)
    parser.add_argument('--gamma-100', action='store_true')
    parser.add_argument('--use-kinematics', action='store_true', default=True)
    
    parser.add_argument('--switch-burst-strength', type=float, default=0.85)
    parser.add_argument('--switch-burst-duration', type=float, default=1.2)
    
    parser.add_argument('--sensitivity', type=float, default=0.05)
    parser.add_argument('--max-speed', type=float, default=9.0)
    parser.add_argument('--fwd-scale', type=float, default=0.2)
    parser.add_argument('--strafe-scale', type=float, default=0.2)
    parser.add_argument('--turn-scale', type=float, default=0.5)
    parser.add_argument('--intent-gain', type=float, default=1.5)
    
    parser.add_argument('--hardcoded-bots', type=int, default=4)
    parser.add_argument('--jepa-bots', type=int, default=4)
    parser.add_argument('--weights', type=str, default="monty_ltm_weights.pt")
    parser.add_argument('--force-recalib', action='store_true')
    parser.add_argument('--no-taesd', action='store_true')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500])
    args = parser.parse_args()

    pilot_kwargs = {
        'sensitivity': args.sensitivity, 'max_speed': args.max_speed,
        'fwd_scale': args.fwd_scale, 'strafe_scale': args.strafe_scale,
        'turn_scale': args.turn_scale, 'intent_gain': args.intent_gain
    }

    if args.users is None:
        args.users = "User1:F3=0,F4=1,AFz=2,Fpz=3" if (args.sim or args.concepts > 1) else "User1:FCz=0"

    subjects = parse_user_setup(args.users, pilot_kwargs, is_sim=args.sim, concepts=args.concepts)
    collective = OverBrainCollective(subjects)
    primary_brain = subjects[0]

    has_fcz_global = any(s.has_region("FCz") for s in subjects)
    has_afz_global = any(s.has_region("AFz") for s in subjects)
    is_motion_calib = False if (args.sim or args.concepts > 1 or has_afz_global) else has_fcz_global

    BASE_PROMPTS = [
        "giant snowy mountain peak, rocky cliffs, clear blue sky, sharp focus, 8k",
        "dense lush green tropical jungle, giant trees, vines, sunlight piercing through leaves, 8k",
        "ancient medieval stone castle fortress towers, daytime, sharp focus, 8k",
        "open stormy dark blue ocean, pure water surface, giant ocean waves, sea foam, no land, 8k",
        "deep outer space, glowing colorful nebula, bright stars, galaxy, 8k, sharp detailed",
        "spherical alien planet with atmosphere, continents and oceans in space, 8k, sharp detailed",
        "futuristic cyberpunk city street, neon lights, rain, glowing signs, sharp linework, 8k",
        "modern glass skyscraper buildings, downtown city, geometric architecture, sharp focus, 8k"
    ]
    
    if args.concepts == 0:
        initial_prompts = [args.start_prompt]
        calib_block_names = ["МИР_0"]
        active_memory_classes = 1
        is_calibrating = False 
    elif is_motion_calib:
        initial_prompts = [args.start_prompt]
        calib_block_names = MOTION_NAMES[:args.move_directions]
        active_memory_classes = args.move_directions
        is_calibrating = not args.chaos
    else:
        num_c = max(2, args.concepts)
        initial_prompts = [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(num_c)]
        calib_block_names = [ALL_NAMES[i % len(ALL_NAMES)] for i in range(num_c)]
        active_memory_classes = len(calib_block_names)
        is_calibrating = not args.chaos

    total_bots = args.hardcoded_bots + args.jepa_bots
    print(f"👥 [КОЛЛЕКТИВ TBT 2.0]: {len(subjects)} субъект(ов) | {total_bots} ботов роя | Классов: {active_memory_classes}")

    agent = None
    if args.sim:
        agent = SyntheticAutonomousAgent(
            config_path=args.config, num_hardcoded=args.hardcoded_bots, num_jepa=args.jepa_bots, 
            num_concepts=max(2, active_memory_classes), sps=args.sps
        )

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("NeuroCanvas × tbp.monty: Thousand Brains Consensus Architecture")
    clock = pygame.time.Clock()
    font_b = pygame.font.SysFont("consolas", 14, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)
    font_large = pygame.font.SysFont("consolas", 18, bold=True)

    engine = HeterarchicalBrainEngine(gamma_max=100.0 if args.gamma_100 else 65.0)
    engine.start()

    worker = EqualPoolChaosWorker(
        initial_prompts=initial_prompts, initial_names=calib_block_names,
        port=6000, mode=args.mode, speed=args.speed, use_taesd=not args.no_taesd, use_color=not args.no_color
    )
    if args.chaos: worker.chaos_enabled = True

    clip_teacher = VisualCLIPTeacher(initial_prompts)

    if not args.force_recalib and os.path.exists(args.weights) and not args.chaos and args.concepts > 0:
        loaded, cnt = primary_brain.heterarchy.load_from_file(args.weights)
        if loaded and cnt == active_memory_classes: is_calibrating = False

    calib_step_idx = 0
    calib_cycle_count = 0
    
    system_warmed_up = False
    warmup_duration = 4.0
    start_time = time.time()
    epoch_start_time = time.time()
    
    last_leader_concept_idx = -1
    concept_switch_timestamp = 0.0

    calib_data = {'y_fwd': [], 'y_bwd': [], 'x_rgt': [], 'x_lft': []} if is_motion_calib else {i: [] for i in range(MAX_CONCEPTS_CAPACITY)}
    current_d_prime, dp_axis_y, dp_axis_x = 0.0, 0.0, 0.0
    concept_d_primes = [0.0] * MAX_CONCEPTS_CAPACITY
    analog_evidence = np.zeros(MAX_CONCEPTS_CAPACITY, dtype=np.float32)

    svd_spectrum = np.zeros(4)
    lead_causal_sign = 0.0
    smooth_depth = 1.0
    cur_probs = np.zeros(MAX_CONCEPTS_CAPACITY, dtype=np.float32)
    clip_lock = threading.Lock()

    def async_clip_worker():
        nonlocal cur_probs
        last_fid = -1
        while True:
            try:
                with worker.lock:
                    img_to_eval = worker.current_rgb.copy()
                    c_fid, c_cnt = worker.frame_id, worker.active_count
                if c_fid != last_fid and np.max(img_to_eval) > 0:
                    last_fid = c_fid
                    probs = clip_teacher.classify(img_to_eval, c_cnt)
                    with clip_lock: cur_probs[:c_cnt] = probs
                    if agent: agent.update_visual_state(probs)
                time.sleep(0.20)
            except Exception: time.sleep(0.2)

    threading.Thread(target=async_clip_worker, daemon=True).start()
    FULL_DUPLEX_MODE = args.sensory_sub
    frame_counter = 0

    try:
        while True:
            dt = clock.tick(60) / 1000.0
            frame_counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_SPACE, pygame.K_c]: worker.toggle_chaos()
                    elif event.key == pygame.K_m: 
                        FULL_DUPLEX_MODE = not FULL_DUPLEX_MODE
                        if agent: agent.set_sensory_substitution(FULL_DUPLEX_MODE)
                        print(f"📡 [SENSORY-SUB]: {'ВКЛЮЧЕН (FULL-DUPLEX)' if FULL_DUPLEX_MODE else 'ВЫКЛЮЧЕН (ЧИСТАЯ СТИГМЕРГИЯ)'}")
                    elif event.key == pygame.K_r:
                        is_calibrating = True
                        system_warmed_up = False
                        start_time = time.time()
                        calib_step_idx, calib_cycle_count = 0, 0
                        analog_evidence.fill(0.0)
                        calib_data = {'y_fwd': [], 'y_bwd': [], 'x_rgt': [], 'x_lft': []} if is_motion_calib else {i: [] for i in range(MAX_CONCEPTS_CAPACITY)}
                        concept_d_primes = [0.0] * MAX_CONCEPTS_CAPACITY

            frame = engine.get_frame()
            has_live_eeg = (frame.num_live > 0)

            # Вычисление L4 и L2/3 нейродинамики
            for s in subjects: 
                s.process_neurophysiology(frame.nodes, dt, worker.active_count, is_calibrating=is_calibrating)

            sorted_subj = sorted(subjects, key=lambda s: s.confidence, reverse=True)
            leader_brain = sorted_subj[0] if sorted_subj else primary_brain

            with worker.lock: 
                active_pool_size = worker.active_count
                pool_names = list(worker.names)

            with clip_lock:   
                live_probs = cur_probs[:active_pool_size].copy()

            # ЧЕСТНЫЙ ВЫБОР БЕЗ ПОДГЛЯДЫВАНИЯ В ПАМЯТЬ АГЕНТОВ
            # Лидер и Партнер берутся только из зрительного фидбека холста и оверлапа ЭЭГ
            if not is_calibrating and np.max(live_probs) > 0.30:
                leader_idx = int(np.argmax(live_probs)) % active_pool_size
                other_scores = leader_brain.wm_scores[:active_pool_size].copy()
                other_scores[leader_idx] = -1.0
                child_idx = int(np.argmax(other_scores)) % active_pool_size
            else:
                top_eeg_order = np.argsort(leader_brain.wm_scores[:active_pool_size])[::-1]
                leader_idx = int(top_eeg_order[0]) % active_pool_size
                child_idx = int(top_eeg_order[1] if len(top_eeg_order) > 1 else top_eeg_order[0]) % active_pool_size

            # Префронтальные узлы
            f3_node  = leader_brain.get_region_node("F3", frame.nodes)
            f4_node  = leader_brain.get_region_node("F4", frame.nodes)
            afz_node = leader_brain.get_region_node("AFz", frame.nodes)
            fpz_node = leader_brain.get_region_node("Fpz", frame.nodes)
            fcz_node = leader_brain.get_region_node("FCz", frame.nodes)

            primary_lead_node = afz_node if afz_node is not None else (f3_node if f3_node is not None else frame.nodes[0])

            beta_order = float(f3_node.beta_power) if f3_node is not None else 0.5
            beta_chaos = float(f4_node.beta_power) if f4_node is not None else 0.5
            torus_u = float(afz_node.torus_u) if afz_node is not None else float(primary_lead_node.torus_u)
            torus_v = float(afz_node.torus_v) if afz_node is not None else float(primary_lead_node.torus_v)

            # 4D Кинематика FCz
            if fcz_node is not None:
                axes = fcz_node.kinematics
                force_x, force_y = float(axes.lx), float(-axes.ly)
                wm_curvature, temporal_bias = float(axes.rx), float(axes.ry)
                is_motion_active = True
            else:
                force_x, force_y, wm_curvature = 0.0, 0.0, 0.0
                temporal_bias = float(primary_lead_node.kinematics.ry)
                is_motion_active = False

            # Ручные клавиши
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:    force_y += 1.0
            if keys[pygame.K_DOWN]:  force_y -= 1.0
            if keys[pygame.K_RIGHT]: force_x += 1.0
            if keys[pygame.K_LEFT]:  force_x -= 1.0
            if keys[pygame.K_PERIOD]: wm_curvature += 0.5
            if keys[pygame.K_COMMA]:  wm_curvature -= 0.5

            leader_brain.pilot.update(dt, force_x, force_y, wm_curvature, temporal_bias)

            # Manifold Camera Warp
            if abs(force_x) > 0.04 or abs(force_y) > 0.04 or abs(wm_curvature) > 0.03:
                with worker.lock:
                    worker.current_rgb = apply_manifold_camera_warp(worker.current_rgb, leader_brain.pilot, dt)

            # 89.5 Гц SVD ранг и каузальный lead
            if frame_counter % 6 == 0:
                lead_ripple = torch.tensor(primary_lead_node.iplv_human_ripple, dtype=torch.float32, device=DEVICE)
                ripple_centered = lead_ripple - torch.mean(lead_ripple, dim=0, keepdim=True)
                if torch.sum(torch.abs(ripple_centered)) > 1e-4:
                    S_vals = torch.linalg.svdvals(ripple_centered)
                    svd_spectrum = (S_vals[:4] / (S_vals[0] + 1e-6)).cpu().numpy()
                    smooth_depth = max(1.0, min(4.0, float(np.sum(svd_spectrum > 0.22))))
                else:
                    smooth_depth, svd_spectrum = 1.0, np.zeros(4)
                lead_causal_sign = float(np.mean(primary_lead_node.iplv_human_ripple[:, 0]))

            safe_step_idx = calib_step_idx % active_memory_classes
            target_now = pool_names[safe_step_idx]

            cur_nav_mag = math.hypot(force_x, force_y)
            if is_motion_calib:
                target_vec = TARGET_UNIT_VECS[safe_step_idx]
                analog_projection = float((force_x * target_vec[0] + force_y * target_vec[1]) / (cur_nav_mag + 1e-6)) if cur_nav_mag > 0.01 else 0.0
            else:
                target_score = float(leader_brain.wm_scores[safe_step_idx])
                other_scores = [float(leader_brain.wm_scores[k]) for k in range(active_memory_classes) if k != safe_step_idx]
                max_other = max(other_scores) if other_scores else 0.0
                analog_projection = float(np.tanh((target_score - max_other) / 25.0))

            persistence_gain = leader_brain.pilot.persistence
            active_evidence_flow = analog_projection * (1.0 + 4.0 * persistence_gain)
            analog_evidence[safe_step_idx] = np.clip(analog_evidence[safe_step_idx] * 0.985 + active_evidence_flow * dt * 0.35, 0.0, 5.0)

            # Выстраивание графа гетерархии по знаку 89.5 Гц опережения фаз
            parent_map = {i: None for i in range(active_pool_size)}
            if leader_idx != child_idx:
                if lead_causal_sign >= 0.03:
                    parent_map[child_idx] = leader_idx
                    rel_sym = "➔"
                elif lead_causal_sign <= -0.03:
                    parent_map[leader_idx] = child_idx
                    rel_sym = "⊂"
                else:
                    rel_sym = "∥"
            else:
                rel_sym = "★"

            is_learning_active = False

            if not system_warmed_up:
                worker.update_cycle(leader_idx=0, child_idx=0, beta_order=0.0, beta_chaos=0.0, torus_u=torus_u, torus_v=torus_v, smooth_depth=1.0, lead_sign=0.0, rx_sagitta=0.0, subjects=subjects, resolved_parent_map=parent_map)
                if has_live_eeg and (time.time() - start_time > warmup_duration):
                    system_warmed_up = True
                    epoch_start_time = time.time()
                    print("🔥 [WARM-UP ЗАВЕРШЕН] Потоки стабилизированы. Начинаем обучение TBT.")

            elif is_calibrating:
                calib_vis_idx = safe_step_idx if not is_motion_calib else 0
                worker.update_cycle(
                    leader_idx=calib_vis_idx, child_idx=calib_vis_idx,
                    beta_order=0.0, beta_chaos=0.0,
                    torus_u=torus_u, torus_v=torus_v,
                    smooth_depth=1.0, lead_sign=0.0,
                    rx_sagitta=0.0, subjects=subjects,
                    resolved_parent_map=parent_map
                )
                worker.strength = args.strength_high
                if agent: agent.set_calibration_target(True, safe_step_idx)

                vis_conf = float(live_probs[safe_step_idx]) if safe_step_idx < len(live_probs) else 0.0
                
                if vis_conf >= 0.05 and has_live_eeg:
                    is_learning_active = True
                    plasticity_mod = (float(np.tanh(analog_projection * 2.0)) * persistence_gain) if is_motion_calib else 1.0
                    for s in subjects:
                        s.learn_contrastive(safe_step_idx, active_memory_classes, lr=0.04 * plasticity_mod, ltd_factor=0.5)

                if not is_motion_calib:
                    for s in subjects:
                        if s.last_node_sdrs:
                            margin_val = s.heterarchy.compute_contrastive_margin(s.last_node_sdrs, safe_step_idx, active_memory_classes)
                            calib_data[safe_step_idx].append(margin_val)

                    for k in range(active_memory_classes):
                        vals_k = calib_data[k][-60:] if len(calib_data[k]) >= 10 else []
                        dp_k = max(0.0, float(np.mean(vals_k) / math.sqrt(np.var(vals_k) + 1e-6))) if len(vals_k) >= 10 else 0.0
                        concept_d_primes[k] = dp_k

                    current_d_prime = float(np.mean(concept_d_primes[:active_memory_classes])) if active_memory_classes > 0 else 0.0

                elapsed = time.time() - epoch_start_time
                if elapsed >= args.calib_seconds:
                    if leader_brain.ltm_scores[safe_step_idx] >= 75.0 and concept_d_primes[safe_step_idx] >= args.calib_sigma:
                        print(f"✅ [ОБУЧЕНО] Концепт {pool_names[safe_step_idx]} зафиксирован!")
                        for s in subjects:
                            s.heterarchy.reset_calcium()
                            s.heterarchy.membrane_potential.fill_(0.0)
                            
                        calib_step_idx += 1
                        epoch_start_time = time.time()
                        
                        if calib_step_idx >= active_memory_classes:
                            calib_step_idx = 0
                            calib_cycle_count += 1
                            if calib_cycle_count >= args.calib_cycles:
                                is_calibrating = False
                                leader_brain.heterarchy.save_to_file(args.weights, pool_names)
                                if agent: agent.set_calibration_target(False)
                                print("🎯 [КАЛИБРОВКА ЗАВЕРШЕНА] Система переходит в свободный режим.")

            else:
                if agent:
                    if FULL_DUPLEX_MODE: agent.update_swarm_priors(leader_brain.wm_scores)
                    else: agent.update_swarm_priors(np.zeros_like(leader_brain.wm_scores))

                worker.update_cycle(
                    leader_idx=leader_idx, child_idx=child_idx,
                    beta_order=beta_order, beta_chaos=beta_chaos,
                    torus_u=torus_u, torus_v=torus_v,
                    smooth_depth=smooth_depth, lead_sign=lead_causal_sign,
                    rx_sagitta=wm_curvature, subjects=subjects,
                    resolved_parent_map=parent_map
                )

                if leader_idx != last_leader_concept_idx:
                    last_leader_concept_idx = leader_idx
                    concept_switch_timestamp = time.time()
                    print(f"🌊 [СМЕНА МЫСЛИ]: Форма=[{pool_names[leader_idx]}] | Стиль=[{pool_names[child_idx]}]")

                if (time.time() - concept_switch_timestamp) < args.switch_burst_duration:
                    worker.strength = float(np.clip(args.switch_burst_strength, 0.10, 0.99))
                else:
                    active_drive = float(np.clip(1.0 - beta_order, 0.0, 1.0))
                    base_strength = args.strength_low + (args.strength_high - args.strength_low) * active_drive
                    if args.use_kinematics:
                        worker.strength = float(np.clip(base_strength + temporal_bias * 0.15, 0.10, 0.99))
                    else:
                        worker.strength = float(np.clip(base_strength, 0.10, 0.99))

            # =============================================================
            # РЕНДЕРИНГ ИНТЕРФЕЙСА (ГАРАНТИРОВАННЫЙ ЗАХВАТ rgb_m)
            # =============================================================
            screen.fill((10, 14, 20))

            with worker.lock:
                rgb_m = worker.current_rgb.copy()

            img_x, img_y = 370, 40
            surf_diff = pygame.image.frombuffer(rgb_m.tobytes(), (worker.img_w, worker.img_h), 'RGB')
            if (worker.img_w, worker.img_h) != (512, 384): 
                surf_diff = pygame.transform.scale(surf_diff, (512, 384))
            screen.blit(surf_diff, (img_x, img_y))
            pygame.draw.rect(screen, (40, 50, 70), (img_x, img_y, 512, 384), 2, border_radius=8)
            
            # Плашка CLIP по всем концептам
            clip_bg = pygame.Rect(img_x, 10, 512, 24)
            pygame.draw.rect(screen, (16, 22, 32), clip_bg, border_radius=4)
            pygame.draw.rect(screen, (50, 70, 100), clip_bg, 1, border_radius=4)
            top_clip_indices = np.argsort(live_probs)[::-1][:4]
            clip_parts = [f"{pool_names[i]}:{live_probs[i]*100:.0f}%" for i in top_clip_indices]
            screen.blit(font_b.render("CLIP: " + " | ".join(clip_parts), True, (255, 200, 100)), (img_x + 10, 14))

            # =============================================================
            # ВЫВОД 4 SDR ВХОДНЫХ РЕГИОНОВ И L2/3 СДР СТЫКОВКИ (60 FPS)
            # =============================================================
            sdr_box_y = 445
            region_names = ["F3:Форма", "F4:Стиль", "AFz:Тор", "Fpz:Ветвь"]
            region_colors = [(100, 255, 120), (255, 120, 220), (255, 210, 50), (100, 200, 255)]

            if leader_brain and len(leader_brain.last_node_sdrs) >= 4:
                for n_i in range(4):
                    sdr_t = leader_brain.last_node_sdrs[n_i][-1].view(64, 64).cpu().numpy() * 255.0
                    sdr_img = cv2.resize(sdr_t, (55, 55), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    
                    rgb_sdr = np.zeros((55, 55, 3), dtype=np.uint8)
                    active_cols = sdr_img > 100
                    rgb_sdr[active_cols] = region_colors[n_i]
                    rgb_sdr[~active_cols] = (15, 20, 30)

                    bx = img_x + n_i * 62
                    surf_sdr = pygame.surfarray.make_surface(np.transpose(rgb_sdr, (1, 0, 2)))
                    screen.blit(surf_sdr, (bx, sdr_box_y))
                    pygame.draw.rect(screen, region_colors[n_i], (bx, sdr_box_y, 55, 55), 1)
                    
                    screen.blit(font_s.render(region_names[n_i], True, region_colors[n_i]), (bx, sdr_box_y - 14))

            # L2/3 СДР СТЫКОВКИ (ОБНОВЛЯЕТСЯ КАЖДЫЙ КАДР НА GPU)
            if leader_brain and leader_brain.consensus_sdr is not None:
                c_sdr = leader_brain.consensus_sdr.view(64, 64).cpu().numpy() * 255.0
                c_img = cv2.resize(c_sdr, (110, 110), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                
                c_rgb = np.zeros((110, 110, 3), dtype=np.uint8)
                c_rgb[c_img > 100] = (0, 255, 220)
                c_rgb[c_img <= 100] = (10, 25, 25)

                cx = img_x + 260
                surf_c = pygame.surfarray.make_surface(np.transpose(c_rgb, (1, 0, 2)))
                screen.blit(surf_c, (cx, sdr_box_y - 15))
                pygame.draw.rect(screen, (0, 255, 200), (cx - 2, sdr_box_y - 17, 114, 114), 2, border_radius=4)
                
                dock_label = f"L2/3: {pool_names[leader_idx]} {rel_sym} {pool_names[child_idx]}"
                screen.blit(font_b.render(dock_label, True, (0, 255, 200)), (cx - 10, sdr_box_y - 32))

            # Радар Когнитивного Тора AFz
            dbg_x, dbg_y, dbg_w, dbg_h = img_x + 380, 430, 180, 175
            pygame.draw.rect(screen, (14, 18, 26), (dbg_x, dbg_y, dbg_w, dbg_h), border_radius=6)
            rx_c, ry_c, rc = dbg_x + 90, dbg_y + 88, 55
            pygame.draw.circle(screen, (20, 28, 40), (rx_c, ry_c), rc)
            pygame.draw.circle(screen, (40, 60, 85), (rx_c, ry_c), rc, 1)

            tu_x, tv_y = math.cos(torus_u), math.sin(torus_v)
            pygame.draw.line(screen, (0, 255, 255), (rx_c, ry_c), (rx_c + int(tu_x * (rc - 8)), ry_c - int(tv_y * (rc - 8))), 3)
            pygame.draw.circle(screen, (255, 255, 255), (rx_c + int(tu_x * (rc - 8)), ry_c - int(tv_y * (rc - 8))), 4)
            screen.blit(font_s.render("КОГНИТИВНЫЙ ТОР", True, (0, 255, 200)), (dbg_x + 25, dbg_y + 8))

            # Левая верхняя панель (Калибровка / Прогрев)
            panel_x, panel_y = 20, 40
            pygame.draw.rect(screen, (16, 22, 32), (panel_x, panel_y, 330, 270), border_radius=8)
            
            if not system_warmed_up:
                pygame.draw.rect(screen, (255, 80, 80), (panel_x, panel_y, 330, 270), 2, border_radius=8)
                screen.blit(font_large.render("SYSTEM WARM-UP (WAIT)", True, (255, 80, 80)), (panel_x + 15, panel_y + 10))
                rem = max(0.0, warmup_duration - (time.time() - start_time))
                screen.blit(font_s.render(f"Стабилизация LSL потока... {rem:.1f}s", True, (200, 200, 200)), (panel_x + 15, panel_y + 35))
            elif is_calibrating:
                pygame.draw.rect(screen, (255, 120, 40), (panel_x, panel_y, 330, 270), 2, border_radius=8)
                screen.blit(font_large.render(f"TBT 2.0 КАЛИБРОВКА ({active_memory_classes} КОНЦ.)", True, (255, 180, 50)), (panel_x + 15, panel_y + 10))
                rem_time = max(0.0, args.calib_seconds - (time.time() - epoch_start_time))
                screen.blit(font_b.render(f"МИШЕНЬ: [{target_now}] (Min {rem_time:.1f}s)", True, (255, 255, 100)), (panel_x + 15, panel_y + 32))
                txt_learn = "СИНАПСЫ ОБНОВЛЯЮТСЯ..." if is_learning_active else "[ЖДЕМ CLIP] Нужно >5%"
                col_learn = (100, 255, 100) if is_learning_active else (255, 80, 80)
                screen.blit(font_b.render(txt_learn, True, col_learn), (panel_x + 15, panel_y + 50))
            else:
                pygame.draw.rect(screen, (0, 255, 200), (panel_x, panel_y, 330, 270), 1, border_radius=8)
                screen.blit(font_large.render(f"TBT 2.0 КОНСЕНСУС ({active_memory_classes} КОНЦ.)", True, (0, 255, 200)), (panel_x + 15, panel_y + 10))
                screen.blit(font_b.render("LTM ОБУЧЕН (ВСЕ ЗЕЛЁНЫЕ)", True, (0, 255, 200)), (panel_x + 15, panel_y + 32))
                screen.blit(font_s.render(f"Средняя точность: d'={current_d_prime:.2f}σ", True, (100, 255, 100)), (panel_x + 15, panel_y + 50))

            row_h = min(25, int(190 / max(1, active_memory_classes)))
            for k in range(active_memory_classes):
                cy = panel_y + 72 + k * row_h
                dp_val = concept_d_primes[k] if k < len(concept_d_primes) else 0.0
                ltm_val = leader_brain.ltm_scores[k] if k < len(leader_brain.ltm_scores) else 0.0
                name = pool_names[k]
                is_act = is_calibrating and (k == safe_step_idx)
                c_col = (100, 255, 100) if (dp_val >= args.calib_sigma and ltm_val >= 75.0) else ((255, 255, 100) if is_act else (180, 180, 190))
                tag = "▶" if is_act else " "
                screen.blit(font_s.render(f"{tag}{name[:6]:6s} {dp_val:3.1f}σ ({ltm_val:2.0f}%)", True, c_col), (panel_x + 12, cy))

                bar_w, bar_h = 120, max(6, row_h - 10)
                bx, by = panel_x + 195, cy + (row_h - bar_h) // 2 - 2
                fill_w = int(np.clip(dp_val / args.calib_sigma, 0.0, 1.0) * bar_w)
                pygame.draw.rect(screen, (30, 40, 50), (bx, by, bar_w, bar_h), border_radius=3)
                if fill_w > 0: pygame.draw.rect(screen, c_col, (bx, by, fill_w, bar_h), border_radius=3)

            # Панель телеметрии
            c_x, c_y = 20, 330
            pygame.draw.rect(screen, (16, 22, 32), (c_x, c_y, 330, 270), border_radius=8)
            pygame.draw.rect(screen, (100, 180, 255), (c_x, c_y, 330, 270), 1, border_radius=8)
            screen.blit(font_b.render("АНСАМБЛЬ АГЕНТОВ МОНТИ:", True, (100, 180, 255)), (c_x + 12, c_y + 12))

            lead_name = pool_names[leader_idx] if leader_idx < len(pool_names) else "..."
            child_name = pool_names[child_idx] if child_idx < len(pool_names) else "..."
            screen.blit(font_b.render(f"Форма (Лидер) : [{lead_name}]", True, (100, 255, 120)), (c_x + 15, c_y + 34))
            screen.blit(font_b.render(f"Стиль (Встройка): [{child_name}]", True, (255, 120, 220)), (c_x + 15, c_y + 54))
            
            screen.blit(font_s.render(f"• F3 Порядок (β)  : {beta_order:.2f} (L-dlPFC Гейт)", True, (100, 255, 120)), (c_x + 15, c_y + 76))
            screen.blit(font_s.render(f"• F4 Хаос (β)     : {beta_chaos:.2f} (R-dlPFC Драйв)", True, (255, 100, 255)), (c_x + 15, c_y + 98))
            screen.blit(font_s.render(f"• AFz Тор (u, v)  : ({torus_u:.2f}, {torus_v:.2f})", True, (255, 220, 50)), (c_x + 15, c_y + 120))
            fpz_txt = f"АКТИВЕН ({fpz_node.gating_ratio:.2f})" if fpz_node else "НЕТ"
            screen.blit(font_s.render(f"• Fpz Ветвление   : {fpz_txt}", True, (200, 220, 255)), (c_x + 15, c_y + 142))
            nav_status = f"VX={leader_brain.pilot.vx:+.1f} | VY={leader_brain.pilot.vy:+.1f}" if is_motion_active else "СТАТИЧНЫЙ"
            screen.blit(font_s.render(f"• FCz Действие    : {nav_status} (SMA/pre-SMA)", True, (0, 255, 200)), (c_x + 15, c_y + 164))
            screen.blit(font_s.render(f"• Theta={frame.theta_freq:.2f}Hz | Delta={frame.delta_freq:.2f}Hz", True, (200, 220, 240)), (c_x + 15, c_y + 190))
            screen.blit(font_s.render(f"• Потоков LSL     : {frame.num_live} (FreeEEG16)", True, (180, 180, 200)), (c_x + 15, c_y + 215))

            if agent:
                agent_str = agent.get_telemetry()
                screen.blit(font_s.render(f"AGENT TELEMETRY: {agent_str[:125]}", True, (100, 255, 150)), (20, HEIGHT - 22))

            # 120-EDGE ciPLV
            BY, BH = 620, 300
            pygame.draw.rect(screen, (12, 16, 24), (20, BY, 870, BH), border_radius=8)
            screen.blit(font_b.render("120-EDGE DIRECTED ciPLV (89.5 Гц РИППЛЫ ДИКИ)", True, (0, 255, 200)), (35, BY + 15))
            g120 = primary_lead_node.iplv_human_ripple[31] if has_live_eeg else np.zeros(120)
            bw, mid_line = 800.0 / 120.0, BY + 150
            for p in range(120):
                val = g120[p]
                bh = int(abs(val) * 110)
                col = (255, 80, 80) if val < 0 else (80, 255, 180)
                pygame.draw.rect(screen, col, (35 + p * bw, mid_line - (bh if val >= 0 else 0), max(1, int(bw - 1)), bh))

            # Causal Manifold SVD
            rx_p, ry_p = 920, 620
            pygame.draw.rect(screen, (20, 15, 30), (rx_p, ry_p, 840, 300), border_radius=8)
            dir_str = "A ⊃ B (Надстройка)" if lead_causal_sign >= 0.03 else ("B ⊃ A (Встраивание)" if lead_causal_sign <= -0.03 else "A ∥ B (Соседство)")
            screen.blit(font_s.render(f"Causal Lead (90 Hz ciPLV): {lead_causal_sign:+.3f} -> {dir_str}", True, (200, 220, 220)), (rx_p + 20, ry_p + 20))
            for i, sv in enumerate(svd_spectrum):
                bh = int(sv * 35)
                pygame.draw.rect(screen, (0, 255, 180) if sv > 0.22 else (80, 80, 80), (rx_p + 20 + i*65, ry_p + 50 + (35 - bh), 45, bh))

            # Синхронизированный Treemap Гетерархии
            map_x, map_y = 950, 40
            map_w, map_h = 800, 560
            pygame.draw.rect(screen, (10, 10, 10), (map_x, map_y, map_w, map_h))
            pygame.draw.rect(screen, (255, 255, 255), (map_x-2, map_y-2, map_w+4, map_h+4), 2)

            # Передаем веса с достаточной температурой, чтобы оба активных концепта вошли в граф
            treemap_weights = leader_brain.blended_weights[:active_pool_size].copy()

            boxes, resolved_hierarchy = calculate_emergent_treemap(
                map_x, map_y, map_w, map_h, treemap_weights, smooth_depth, lead_causal_sign, active_pool_size
            )
            treemap_title = f"РЕКУРСИВНАЯ ГЕТЕРАРХИЯ СМЫСЛОВ (Пул: {active_pool_size})"
            screen.blit(font_b.render(treemap_title, True, (255, 255, 255)), (map_x, map_y - 25))

            for c_idx in resolved_hierarchy:
                if c_idx >= active_pool_size: continue
                bx, by, bw, bh, rank = boxes[c_idx]
                c_color = FRACTAL_COLORS[c_idx % len(FRACTAL_COLORS)]
                pygame.draw.rect(screen, c_color, (bx, by, bw, bh))
                pygame.draw.rect(screen, (220, 220, 220), (bx, by, bw, bh), 2)
                c_name = pool_names[c_idx]
                
                txt_line1 = f"L{rank}: D{worker.depths[c_idx]}"
                surf1 = font_s.render(txt_line1, True, (255, 255, 255))
                surf2 = font_b.render(c_name, True, (255, 255, 255))
                
                if bw >= surf2.get_width() + 8 and bh >= 40:
                    screen.blit(surf1, (bx + 5, by + 5))
                    screen.blit(surf2, (bx + 5, by + 20))
                elif bw >= 30 and bh >= 20:
                    screen.blit(font_s.render(c_name[:5], True, (255, 255, 255)), (bx + 4, by + 4))

            mode_color = (0, 255, 100) if FULL_DUPLEX_MODE else (160, 160, 160)
            mode_str = "SENSORY SUB: [FULL-DUPLEX (ПРЯМОЙ ОБМЕН)]" if FULL_DUPLEX_MODE else "SENSORY SUB: [ЧИСТАЯ СТИГМЕРГИЯ (ЧЕРЕЗ ХОЛСТ)]"
            screen.blit(font_b.render(mode_str, True, mode_color), (1350, 15))

            pygame.display.flip()

    finally:
        worker.running = False
        if primary_brain and hasattr(primary_brain, 'heterarchy'):
            primary_brain.heterarchy.save_to_file(args.weights, pool_names)
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
