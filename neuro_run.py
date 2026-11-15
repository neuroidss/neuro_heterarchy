#!/usr/bin/env python3
"""
===================================================================================
neuro_run.py — Единый 3D-DiT Видео-Модуль для Веб-Студии (:8080) и Локального Окна
===================================================================================
- Запускает фоновый 60 Гц LSL->VRAM конвейер (SharedAmodalCognitiveEngine + SelfSufficientHeterarchyRouter),
  чтобы любая подключенная в браузере BLE-плата FreeEEG16 мгновенно (каждые 16 мс!) обновляла
  96,000-D шину рабочей памяти HardwareVRAMBus прямо во время шагов денойзинга 3D-DiT!
- Двусторонняя синхронизация с web_cloud_gateway.py (порт 6002 видео + порт 6007 управление и BLE-маппинг).
===================================================================================
"""

from __future__ import annotations
import argparse
from collections import deque
from copy import copy
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import threading
import time
import cv2
import numpy as np

from neuro_world import (
    load_game_manifest,
    StageSpec,
    GameWorldManifest,
    ArchetypeSpec,
)
from neuro_engine import UniversalLatentAutobattlerEngine
from neuro_clash_hud import draw_identity_hud
from neuro_referee import LatentIdentityFrameTelemetry
from neuro_recorder import recorder_instance

try:
    from neuro_amodal_engine import (
        SharedAmodalCognitiveEngine,
        SelfSufficientHeterarchyRouter,
        EXTENDED_ROLE_TO_FUNCTIONAL,
    )
    HAS_AMODAL_LSL_ENGINE = True
except ImportError:
    HAS_AMODAL_LSL_ENGINE = False

MODE_TO_MANIFEST = {
    "walk": [
        "configs/mode_00_arena_walk_handshake.json",
        "mode_00_arena_walk_handshake.json",
    ],
    "maze": [
        "configs/mode_03_semantic_maze_gm.json",
        "mode_03_semantic_maze_gm.json",
    ],
    "ball": [
        "configs/mode_05_wm_control_test_arena.json",
        "mode_05_wm_control_test_arena.json",
    ],
    "ritual": [
        "configs/mode_04_forest_neuro_ritual.json",
        "mode_04_forest_neuro_ritual.json",
    ],
    "idle_duel": [
        "configs/mode_01_1v1_idle_target.json",
        "mode_01_1v1_idle_target.json",
    ],
    "active_duel": [
        "configs/mode_02_1v1_active_duel_melee.json",
        "mode_02_1v1_active_duel_melee.json",
    ],
}

ROLE_TO_ENTITY_SLOT = {
    "FCz": "e0",
    "AFz": "e1",
    "F3":  "e2",
    "F4":  "e3",
    "Fpz": "e0",
}


def resolve_manifest_path(mode: str | None, manifest_arg: str) -> str:
    if mode and mode in MODE_TO_MANIFEST:
        for candidate in MODE_TO_MANIFEST[mode]:
            if Path(candidate).exists():
                return candidate
        return MODE_TO_MANIFEST[mode][0]

    if Path(manifest_arg).exists():
        return manifest_arg
    alt = Path(manifest_arg).name
    if Path(alt).exists():
        return alt
    return manifest_arg


def manifest_to_web_sections(manifest: GameWorldManifest) -> dict[str, list[str]]:
    """Преобразует GameWorldManifest в региональные секции [FPZ, AFZ, F3, F4, FCZ, NEGATIVE] для веб-редактора."""
    secs: dict[str, list[str]] = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
    if manifest.negative_desc:
        secs["NEGATIVE"] = [manifest.negative_desc]
    if manifest.arena_context_desc:
        secs["FPZ"].append(manifest.arena_context_desc)

    if manifest.stages:
        st0 = manifest.stages[0]
        for slot in st0.entities:
            reg_key = "FCZ"
            b_up = slot.control_binding.upper()
            for r_k in ("FCZ", "AFZ", "F3", "F4", "FPZ"):
                if r_k in b_up:
                    reg_key = r_k
                    break
            for aid in slot.allowed_pool:
                arch = manifest.archetypes.get(aid)
                if arch:
                    full_s = f"{arch.essence_desc} {arch.action_desc}".strip()
                    if full_s and full_s not in secs[reg_key]:
                        secs[reg_key].append(full_s)
    return secs


def apply_web_sections_to_manifest(manifest: GameWorldManifest, sections: dict[str, list[str]]):
    """
    Универсальный живой редактор лора: обновляет архетипы и пулы агентов манифеста
    напрямую из отредактированных в браузере секций [FCZ, AFZ, F3, F4, FPZ, NEGATIVE]!
    """
    neg_list = [s.strip() for s in sections.get("NEGATIVE", []) if s.strip()]
    if neg_list:
        manifest.negative_desc = ", ".join(neg_list)

    fpz_list = [s.strip() for s in sections.get("FPZ", []) if s.strip()]
    fcz_list = [s.strip() for s in sections.get("FCZ", []) if s.strip()]
    if fpz_list and fpz_list == fcz_list:
        manifest.arena_context_desc = ""
    elif fpz_list:
        manifest.arena_context_desc = fpz_list[0].rstrip(".")

    for idx_s, reg_key in [(0, "FCZ"), (1, "AFZ"), (2, "F3"), (3, "F4")]:
        lines = [s.strip() for s in sections.get(reg_key, []) if s.strip()]
        if not lines:
            continue
        new_pool_ids = []
        for l_idx, line_txt in enumerate(lines):
            aid = f"e{idx_s}_s{l_idx}"
            new_pool_ids.append(aid)
            manifest.archetypes[aid] = ArchetypeSpec(
                archetype_id=aid,
                element_tag=f"{reg_key}_{l_idx}",
                role_tag="WM_AGENT",
                display_name=f"e{idx_s}:{line_txt[:18]}",
                essence_desc=line_txt,
                action_desc="",
            )
        for st in manifest.stages:
            if idx_s < len(st.entities):
                st.entities[idx_s].allowed_pool = list(new_pool_ids)


def build_compact_telemetry(telem: dict) -> dict:
    win_map = {}
    for k, v in telem.get("wm_524288d_choice_debug", {}).items():
        ch = v.get("chosen", {})
        if ch:
            c_id, c_pct = next(iter(ch.items()))
            short_id = c_id.split("_", 1)[-1] if "_" in c_id else c_id
            win_map[k] = f"{short_id}:{int(round(c_pct))}%"

    br_map = {
        k.replace("CAT(", "").replace(")", ""): int(round(v))
        for k, v in telem.get("reaction_branch_competition_128d", {}).items()
    }
    chaos = telem.get("feigenbaum_chaos_metrics", {})
    oracle = telem.get("quest_oracle_metrics", {})
    rt = telem.get("realtime_pacing", {})

    return {
        "step": telem.get("step"),
        "t": telem.get("t"),
        "stg": telem.get("global_stage"),
        "drv": telem.get("driver_tag"),
        "dS": oracle.get("quest_progress_pct"),
        "P+": oracle.get("prob_completed_peak_pct"),
        "r": round(float(chaos.get("feigenbaum_r", 3.57)), 3),
        "cat_res": chaos.get("catalytic_resonance_pct"),
        "win": win_map,
        "branches": br_map,
        "rtf": rt.get("ema_rtf"),
        "spd": rt.get("play_speed"),
        "ms": int(round(float(rt.get("gen_ms", 0)))),
    }


def draw_minimal_rtf_banner(
    frame_rgb: np.ndarray,
    rtf: float,
    play_speed: float,
    gen_ms: float,
    raw_video_ms: float,
    q_sec: float,
    ctx_lat: int,
    telem: LatentIdentityFrameTelemetry | None,
) -> np.ndarray:
    canvas = frame_rgb.copy()
    h, w, _ = canvas.shape
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 22), (8, 12, 18), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    font = cv2.FONT_HERSHEY_SIMPLEX
    diff_ms = gen_ms - raw_video_ms
    if rtf >= 1.0:
        headroom_pct = int(round((rtf - 1.0) * 100.0))
        status_col = (80, 255, 140)
        rtf_txt = f"RTF:{rtf:.2f}x (+{headroom_pct}% ZAPAS -> OK TO ADD CTX={ctx_lat+1})"
    else:
        lag_pct = int(round((1.0 - rtf) * 100.0))
        status_col = (255, 200, 60) if rtf >= 0.55 else (255, 110, 90)
        rtf_txt = f"RTF:{rtf:.2f}x (SLOWDOWN {play_speed:.2f}x | LAG +{diff_ms:.0f}ms / -{lag_pct}%)"

    d_str = f" | dS:{telem.quest_progress_pct:.0f}% r:{telem.feigenbaum_r:.3f}" if telem else ""
    line = f"{rtf_txt} | Gen:{gen_ms:.0f}ms Buf:{q_sec:.1f}s{d_str} [H:HUD]"
    cv2.putText(canvas, line[:92], (6, 15), font, 0.25, status_col, 1, cv2.LINE_AA)
    return canvas


class SmoothRealtimeVideoPacer(threading.Thread):

    def __init__(
        self,
        gateway_port: int,
        width: int,
        height: int,
        monochrome: bool,
        show_window: bool = False,
        hud_mode: str = "off",
        manifest: GameWorldManifest | None = None,
        game_mode: str = "strict_ko",
        context_latents: int = 1,
    ):
        super().__init__(daemon=True)
        self.gateway_port = gateway_port
        self.width = width
        self.height = height
        self.monochrome = monochrome
        self.show_window = show_window
        self.hud_mode = hud_mode.lower()
        self.manifest = manifest
        self.game_mode = game_mode
        self.context_latents = context_latents

        self.frame_queue: deque[tuple[np.ndarray, LatentIdentityFrameTelemetry | None, StageSpec | None, list[list[str]], int]] = deque(maxlen=160)
        self.hud_traj_history: list[LatentIdentityFrameTelemetry] = []
        self.prev_tail_clean_rgb: np.ndarray | None = None
        self.prev_quest_pct: float = 0.0
        self.prev_cat_res_pct: float = 75.0

        self.lock = threading.Lock()
        self.running = True

        self.ema_gen_sec = 0.90
        self.ema_rtf = 1.05
        self.play_speed = 1.00
        self.frame_seq_id = 0
        self.last_gen_ms = 900.0
        self.last_raw_video_ms = 1000.0
        self.actual_steps = 6
        self.gateway_conn = None

    def cycle_hud_mode(self):
        order = ["off", "minimal", "full"]
        idx = (order.index(self.hud_mode) + 1) % len(order) if self.hud_mode in order else 0
        self.hud_mode = order[idx]
        print(f"🖥️ [HUD MODE] Режим отображения на экране: {self.hud_mode.upper()} (Клавиша H для смены)", flush=True)

    def push_generated_batch(
        self,
        clean_rgb_frames: list[np.ndarray],
        traj: list[LatentIdentityFrameTelemetry],
        stage: StageSpec,
        active_parties: list[list[str]],
        attempt_num: int,
        gen_ms: float,
        actual_steps: int,
    ):
        if not clean_rgb_frames:
            return
        gen_sec = max(0.05, gen_ms / 1000.0)
        raw_video_sec = len(clean_rgb_frames) / 16.0
        inst_rtf = raw_video_sec / gen_sec

        with self.lock:
            self.last_gen_ms = gen_ms
            self.last_raw_video_ms = raw_video_sec * 1000.0
            self.actual_steps = actual_steps
            self.ema_gen_sec = 0.65 * self.ema_gen_sec + 0.35 * gen_sec
            self.ema_rtf = 0.65 * self.ema_rtf + 0.35 * inst_rtf

            frames_work = [f.copy() for f in clean_rgb_frames]
            if self.prev_tail_clean_rgb is not None and len(frames_work) >= 2:
                frames_work[0] = cv2.addWeighted(
                    self.prev_tail_clean_rgb, 0.22, frames_work[0], 0.78, 0.0
                )
            self.prev_tail_clean_rgb = clean_rgb_frames[-1].copy()

            target_out_frames = int(np.clip(round(self.ema_gen_sec * 16.0), len(frames_work), len(frames_work) * 3))
            self.play_speed = round(len(frames_work) / max(1.0, float(target_out_frames)), 2)

            base_telem = traj[-1]
            target_q_pct = float(base_telem.quest_progress_pct)
            target_c_pct = float(base_telem.catalytic_resonance_pct)
            start_q_pct = self.prev_quest_pct if target_q_pct >= self.prev_quest_pct else 0.0
            start_c_pct = self.prev_cat_res_pct

            n_src = len(frames_work)
            for k in range(target_out_frames):
                frac = k / float(max(1, target_out_frames - 1))
                pos = frac * (n_src - 1)
                i0 = int(np.floor(pos))
                i1 = min(n_src - 1, i0 + 1)
                alpha = float(pos - i0)
                if alpha <= 0.01 or i0 == i1:
                    interp_rgb = frames_work[i0]
                else:
                    interp_rgb = cv2.addWeighted(frames_work[i0], 1.0 - alpha, frames_work[i1], alpha, 0.0)

                smooth_t = copy(base_telem)
                smooth_t.quest_progress_pct = round(start_q_pct + (target_q_pct - start_q_pct) * frac, 1)
                smooth_t.catalytic_resonance_pct = round(start_c_pct + (target_c_pct - start_c_pct) * frac, 1)
                smooth_trans = list(base_telem.entity_transition_pcts)
                if 0 <= stage.quest.active_slot_idx < len(smooth_trans):
                    smooth_trans[stage.quest.active_slot_idx] = smooth_t.quest_progress_pct
                smooth_t.entity_transition_pcts = smooth_trans

                self.frame_queue.append((interp_rgb, smooth_t, stage, list(active_parties), attempt_num))

            self.prev_quest_pct = 0.0 if target_q_pct >= 95.0 else target_q_pct
            self.prev_cat_res_pct = target_c_pct

            while len(self.frame_queue) > 48:
                self.frame_queue.popleft()

    def _render_hud_on_clean_frame(
        self,
        clean_rgb: np.ndarray,
        telem: LatentIdentityFrameTelemetry | None,
        stage: StageSpec | None,
        parties: list[list[str]],
        attempt_num: int,
        q_sec: float,
    ) -> np.ndarray:
        if self.hud_mode == "off" or telem is None or stage is None or self.manifest is None:
            return clean_rgb

        if self.hud_mode == "minimal":
            return draw_minimal_rtf_banner(
                clean_rgb,
                rtf=self.ema_rtf,
                play_speed=self.play_speed,
                gen_ms=self.last_gen_ms,
                raw_video_ms=self.last_raw_video_ms,
                q_sec=q_sec,
                ctx_lat=self.context_latents,
                telem=telem,
            )

        self.hud_traj_history.append(telem)
        if len(self.hud_traj_history) > 64:
            self.hud_traj_history.pop(0)

        with_full_hud = draw_identity_hud(
            clean_rgb,
            self.hud_traj_history,
            len(self.hud_traj_history) - 1,
            stage,
            parties,
            self.manifest,
            attempt_num,
            self.game_mode,
            monochrome=False,
        )
        return draw_minimal_rtf_banner(
            with_full_hud,
            rtf=self.ema_rtf,
            play_speed=self.play_speed,
            gen_ms=self.last_gen_ms,
            raw_video_ms=self.last_raw_video_ms,
            q_sec=q_sec,
            ctx_lat=self.context_latents,
            telem=telem,
        )

    def run(self):
        pg = None
        screen = None
        win_w, win_h = max(768, self.width * 2), max(512, self.height * 2)

        if self.show_window:
            try:
                if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                    del os.environ["SDL_VIDEODRIVER"]
                import pygame
                pygame.init()
                screen = pygame.display.set_mode((win_w, win_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
                pygame.display.set_caption("Exocortex 3D-DiT Smooth World ([H] Toggle HUD | Web Studio :8080 Ready)")
                pg = pygame
            except Exception as e:
                print(f"⚠️ [Pygame Window Warning]: {e}")
                pg = None

        last_item = None
        while self.running:
            t_tick = time.perf_counter()

            if pg is not None:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.running = False
                        os._exit(0)
                    elif event.type == pg.KEYDOWN and event.key == pg.K_h:
                        self.cycle_hud_mode()

            with self.lock:
                q_len = len(self.frame_queue)
                if q_len > 0:
                    last_item = self.frame_queue.popleft()
                gen_ms = self.last_gen_ms
                act_steps = self.actual_steps

            if last_item is None:
                time.sleep(0.015)
                continue

            clean_rgb, telem, stg, parties, att_num = last_item
            q_sec = q_len / 16.0
            display_rgb = self._render_hud_on_clean_frame(clean_rgb, telem, stg, parties, att_num, q_sec)

            if pg is not None and screen is not None:
                try:
                    surf = pg.image.frombuffer(display_rgb.tobytes(), (display_rgb.shape[1], display_rgb.shape[0]), "RGB")
                    if (display_rgb.shape[1], display_rgb.shape[0]) != (win_w, win_h):
                        surf = pg.transform.smoothscale(surf, (win_w, win_h))
                    screen.blit(surf, (0, 0))
                    pg.display.flip()
                except Exception:
                    pass

            if self.gateway_conn is None and self.gateway_port > 0:
                try:
                    self.gateway_conn = Client(("localhost", self.gateway_port), authkey=b"canvas")
                except Exception:
                    self.gateway_conn = None

            if self.gateway_conn is not None:
                try:
                    bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
                    _, enc_jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    self.frame_seq_id += 1
                    self.gateway_conn.send({
                        "frame_id": self.frame_seq_id,
                        "ts": time.time(),
                        "gen_ms": gen_ms,
                        "fps": round(16.0 * self.play_speed, 1),
                        "actual_steps": act_steps,
                        "w": self.width,
                        "h": self.height,
                        "use_taesd": False,
                        "use_color": not self.monochrome,
                        "jpg": enc_jpg.tobytes(),
                    })
                    if recorder_instance.is_recording:
                        recorder_instance.push_frame(bgr, {"dag_log_parts": [stg.quest.driver_tag if stg else "3D_DIT"]})
                except Exception:
                    self.gateway_conn = None

            target_dt = 0.055 if q_len > 16 else (0.0625 if q_len > 3 else 0.075)
            elapsed = time.perf_counter() - t_tick
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)


def start_live_lsl_to_vram_bus_thread(pacer: SmoothRealtimeVideoPacer, users_arg: str = ""):
    """
    Запускает 60 Гц фоновый конвейер SharedAmodalCognitiveEngine, который принимает LSL-потоки
    FreeEEG_<ID> от подключенных в браузере BLE-плат и непрерывно заполняет HardwareVRAMBus в GPU!
    """
    if not HAS_AMODAL_LSL_ENGINE:
        return None, None

    router = SelfSufficientHeterarchyRouter(users_arg)
    amodal_engine = SharedAmodalCognitiveEngine(
        router=router,
        gamma_max=100.0,
        strength_low=0.06,
        strength_high=0.92,
        cross_region_bleed=0.65,
        monty_mode="off",
    )

    def _lsl_loop():
        while pacer.running:
            try:
                amodal_engine.step(0.016)
            except Exception:
                pass
            time.sleep(0.016)

    threading.Thread(target=_lsl_loop, daemon=True).start()
    print("🧠 [LIVE BLE/LSL -> VRAM BUS] 60 Гц конвейер рабочей памяти запущен! Любая BLE-плата из браузера сразу управляет 3D-DiT.")
    return router, amodal_engine


def start_web_studio_control_bridge(
    runner: UniversalLatentAutobattlerEngine,
    pacer: SmoothRealtimeVideoPacer,
    video_ctrl_port: int,
    initial_manifest_path: str,
    router=None,
    amodal_engine=None,
):
    ctrl_lock = threading.Lock()
    ctrl_conn = None

    def _listener_loop():
        nonlocal ctrl_conn
        while pacer.running:
            try:
                if ctrl_conn is None and video_ctrl_port > 0:
                    c = Client(("localhost", video_ctrl_port), authkey=b"video_ctrl")
                    with ctrl_lock:
                        ctrl_conn = c
                        ctrl_conn.send({
                            "cmd": "sync_from_video_runner",
                            "lore_file": initial_manifest_path,
                            "sections": manifest_to_web_sections(runner.manifest),
                            "monty_mode": "off",
                            "stochastic": runner.stochastic,
                            "wm_radius": runner.wm_radius,
                            "hud_mode": pacer.hud_mode,
                            "context_latents": runner.context_lat_frames,
                            "step_latents": runner.step_lat_frames,
                            "steps": runner.steps,
                            "cfg_scale": runner.cfg_scale,
                        })
                    print(f"🌐 [WEB STUDIO BRIDGE] Подключено к web_cloud_gateway (порт {video_ctrl_port})! Управление из браузера :8080 активно.")

                has_msg = False
                msg = None
                with ctrl_lock:
                    if ctrl_conn is not None and ctrl_conn.poll(0.02):
                        msg = ctrl_conn.recv()
                        has_msg = True

                if has_msg and isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "update_video_config":
                        cfg = msg.get("config", {})
                        if "monty_mode" in cfg and amodal_engine is not None:
                            amodal_engine.set_monty_mode(str(cfg["monty_mode"]))
                        if "stochastic" in cfg:
                            runner.stochastic = float(np.clip(float(cfg["stochastic"]), 0.0, 1.0))
                        if "wm_radius" in cfg:
                            runner.wm_radius = float(np.clip(float(cfg["wm_radius"]), 0.02, 0.85))
                        elif "geodesic_lock" in cfg:
                            runner.wm_radius = float(np.clip(float(cfg["geodesic_lock"]), 0.02, 0.85))
                        if "hud_mode" in cfg:
                            pacer.hud_mode = str(cfg["hud_mode"]).lower()
                        if "context_latents" in cfg:
                            runner.context_lat_frames = max(1, min(3, int(cfg["context_latents"])))
                            pacer.context_latents = runner.context_lat_frames
                        if "step_latents" in cfg:
                            runner.step_lat_frames = max(1, min(3, int(cfg["step_latents"])))
                        elif "num_frames" in cfg:
                            nf = int(cfg["num_frames"])
                            runner.step_lat_frames = max(1, (nf - 1) // 8)
                        if "steps" in cfg:
                            runner.steps = max(3, min(20, int(cfg["steps"])))
                        if "cfg_scale" in cfg:
                            runner.cfg_scale = float(np.clip(float(cfg["cfg_scale"]), 1.2, 6.0))
                        print(
                            f"🎛️ [WEB CONFIG] Из браузера: wm_radius={runner.wm_radius:.2f}, stoch={runner.stochastic:.2f}, "
                            f"hud={pacer.hud_mode}, ctx={runner.context_lat_frames}, step={runner.step_lat_frames}, "
                            f"steps={runner.steps}, cfg={runner.cfg_scale:.1f}"
                        )
                    elif cmd == "update_lore_sections":
                        secs = msg.get("sections", {})
                        if isinstance(secs, dict):
                            apply_web_sections_to_manifest(runner.manifest, secs)
                            print("📝 [WEB LORE EDITOR] Пулы концептов 3D-DiT обновлены на лету из браузера!")
                    elif cmd == "switch_lore":
                        l_file = str(msg.get("lore_file", "")).strip()
                        secs = msg.get("sections")
                        if l_file.endswith(".json") and Path(l_file).exists():
                            runner.manifest = load_game_manifest(l_file)
                            pacer.manifest = runner.manifest
                            runner.curr_idx = 0
                            runner.maneuver_phase = 0
                            runner.need_hard_genesis = True
                            print(f"🌍 [WEB MANIFEST SWITCH] Загружен JSON-манифест: {l_file}")
                        elif isinstance(secs, dict):
                            apply_web_sections_to_manifest(runner.manifest, secs)
                            runner.need_hard_genesis = True
                            print(f"🌍 [WEB LORE SWITCH] Применен региональный лор: {l_file}")
                    elif cmd == "reset_genesis":
                        runner.need_hard_genesis = True
                        runner.maneuver_phase = 0
                        if amodal_engine is not None:
                            amodal_engine.reset_seals()
                        print("🔄 [WEB RESET] Сброс генезиса 3D-DiT по кнопке из браузера!")
                    elif cmd == "set_device_mapping":
                        s_name = str(msg.get("stream_name", "")).strip()
                        r_val = str(msg.get("role", "FCz")).strip()
                        u_val = str(msg.get("user", "User1")).strip()
                        if router is not None and s_name:
                            router.set_device_mapping(s_name, r_val, u_val)
                        if r_val and r_val.upper() not in ("NONE", "OFF", "UNASSIGNED"):
                            func_r = EXTENDED_ROLE_TO_FUNCTIONAL.get(r_val, "FCz") if HAS_AMODAL_LSL_ENGINE else r_val
                            target_slot = ROLE_TO_ENTITY_SLOT.get(func_r, "e0")
                            runner.set_entity_source(target_slot, f"eeg:{func_r}")
                            print(f"📡 [BLE DEVICE MAPPED] Плата '{s_name}' (Субъект: {u_val}) -> Зона: {r_val} ({target_slot}=eeg:{func_r})")
                    elif cmd == "start_recording":
                        recorder_instance.start_session(session_name=msg.get("session_name"))
                    elif cmd == "stop_recording":
                        recorder_instance.stop_session()
                else:
                    time.sleep(0.02)
            except Exception:
                with ctrl_lock:
                    ctrl_conn = None
                time.sleep(0.5)

    threading.Thread(target=_listener_loop, daemon=True).start()

    def push_telemetry_to_web(telem: dict, fps: float, gen_ms: float):
        with ctrl_lock:
            if ctrl_conn is not None:
                try:
                    q_metrics = telem.get("quest_oracle_metrics", {})
                    c_metrics = telem.get("feigenbaum_chaos_metrics", {})
                    ctrl_conn.send({
                        "cmd": "video_telemetry",
                        "fps": round(fps, 1),
                        "gen_ms": round(gen_ms, 1),
                        "rtf": round(pacer.ema_rtf, 2),
                        "play_speed": pacer.play_speed,
                        "feigenbaum_r": round(float(c_metrics.get("feigenbaum_r", 3.72)), 3),
                        "reaction_formula": telem.get("reaction_formula", ""),
                        "branch_scores": telem.get("reaction_branch_competition_128d", {}),
                        "debug_tokens": f"[{telem.get('driver_tag')}] dS:{q_metrics.get('quest_progress_pct', 0):.0f}% | {telem.get('active_entities', '')}",
                        "fpz_meta_charge": round(float(c_metrics.get("catalytic_resonance_pct", 50.0)) / 100.0, 3),
                        "afz_stage_progress": round(float(q_metrics.get("quest_progress_pct", 0.0)) / 100.0, 3),
                        "monty_loss": 0.015,
                        "persistence": round(float(c_metrics.get("catalytic_resonance_pct", 80.0)) / 100.0, 2),
                        "surprise": round(abs(float(c_metrics.get("feigenbaum_r", 3.7)) - 3.57), 2),
                        "monty_mode": "off",
                        "stochastic": runner.stochastic,
                        "unassigned_stochastic": runner.stochastic,
                        "wm_radius": runner.wm_radius,
                        "hud_mode": pacer.hud_mode,
                        "context_latents": runner.context_lat_frames,
                        "step_latents": runner.step_lat_frames,
                        "steps": runner.steps,
                        "cfg_scale": runner.cfg_scale,
                        "active_region_mask": {"Fpz": True, "AFz": True, "F3": True, "F4": True, "FCz": True},
                    })
                except Exception:
                    pass

    return push_telemetry_to_web


def main():
    parser = argparse.ArgumentParser(
        description="Unified 3D-DiT Video Module for NeuroCanvas Web Studio (:8080) & Local Pygame"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="maze",
        choices=["walk", "maze", "ball", "ritual", "idle_duel", "active_duel"],
        help="Быстрый пресет: maze | ball | ritual | walk | idle_duel | active_duel",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="configs/mode_03_semantic_maze_gm.json",
        help="Путь к внешнему JSON-манифесту мира",
    )
    parser.add_argument(
        "--entity-source",
        "--wm-source",
        dest="entity_source",
        type=str,
        default="auto",
        help="Источник рабочей памяти для сущностей ('auto' или 'e0=eeg:FCz,e1=eeg:AFz')",
    )
    parser.add_argument(
        "--stochastic",
        type=float,
        default=0.0,
        help="Уровень стохастики (0.0 = 100% управление из рабочей памяти без белого шума)",
    )
    parser.add_argument(
        "--wm-radius",
        type=float,
        default=0.35,
        help="Радиус поворота мира от рабочей памяти ЭЭГ (0.10 = плавное удержание, 0.45 = бурный морфинг)",
    )
    parser.add_argument(
        "--hud",
        type=str,
        default="off",
        choices=["off", "minimal", "full"],
        help="Режим дебага на видео: off (чистый мир для веб-стенда), minimal (строка RTF), full (полный HUD). Клавиша [H] меняет на лету!",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        default=False,
        help="Полностью скрыть дебаг-HUD с видео",
    )
    parser.add_argument(
        "--context-latents",
        type=int,
        default=1,
        help="Число латентных кадров контекста (1 = быстрый живой морфинг)",
    )
    parser.add_argument(
        "--step-latents",
        type=int,
        default=2,
        help="Число новых латентных кадров за 1 шаг (2 = 16 кадров / 1.0 сек живого движения)",
    )
    parser.add_argument("--steps", type=int, default=6, help="Число шагов 3D-DiT")
    parser.add_argument("--cfg", type=float, default=3.2, help="CFG scale")
    parser.add_argument("--width", type=int, default=384, help="Ширина видео")
    parser.add_argument("--height", type=int, default=256, help="Высота видео")
    parser.add_argument("--monochrome", action="store_true", default=False)
    parser.add_argument("--show-window", action="store_true", default=False, help="Показать локальное окно Pygame")
    parser.add_argument("--game-mode", type=str, default="strict_ko", choices=["strict_ko", "sudden_death"])
    parser.add_argument("--max-overtime-chunks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--gateway-port", type=int, default=6002, help="Порт видеопотока web_cloud_gateway.py (6002)")
    parser.add_argument("--video-ctrl-port", type=int, default=6007, help="Порт управления web_cloud_gateway.py (6007)")
    parser.add_argument("--battle-server-port", type=int, default=6008, help="Порт бессмертного сервера весов (6008)")
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--save-mp4", type=str, default="three_phase_slime_autobattler.mp4")
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    eff_hud_mode = "off" if args.no_hud else args.hud
    resolved_manifest = resolve_manifest_path(args.mode, args.manifest)
    manifest = load_game_manifest(resolved_manifest)

    save_path = "" if args.no_save else args.save_mp4
    compact_log_path = "telemetry_compact.jsonl"

    runner = UniversalLatentAutobattlerEngine(
        manifest=manifest,
        width=args.width,
        height=args.height,
        game_mode=args.game_mode,
        max_overtime_chunks=args.max_overtime_chunks,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        battle_server_port=args.battle_server_port,
        force_local=args.local,
        monochrome=args.monochrome,
        context_lat_frames=args.context_latents,
        step_lat_frames=args.step_latents,
        stochastic=args.stochastic,
        wm_radius=args.wm_radius,
    )

    if args.entity_source:
        if "=" in args.entity_source:
            for pair in args.entity_source.split(","):
                if "=" in pair:
                    slot_k, src_v = pair.split("=", 1)
                    runner.set_entity_source(slot_k.strip(), src_v.strip())
        else:
            runner.set_all_entities_source(args.entity_source.strip())

    pacer = SmoothRealtimeVideoPacer(
        gateway_port=args.gateway_port,
        width=runner.width,
        height=runner.height,
        monochrome=runner.monochrome,
        show_window=args.show_window,
        hud_mode=eff_hud_mode,
        manifest=manifest,
        game_mode=args.game_mode,
        context_latents=runner.context_lat_frames,
    )
    pacer.start()

    # Запускаем 60 Гц конвейер LSL -> HardwareVRAMBus для подключенных в браузере BLE-плат:
    router, amodal_engine = start_live_lsl_to_vram_bus_thread(pacer)

    push_web_telem = start_web_studio_control_bridge(
        runner=runner,
        pacer=pacer,
        video_ctrl_port=args.video_ctrl_port,
        initial_manifest_path=resolved_manifest,
        router=router,
        amodal_engine=amodal_engine,
    )

    video_writer = None
    if not args.no_save and save_path:
        video_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            16.0,
            (runner.width, runner.height),
        )

    start_time = time.time()
    compact_ring: deque[str] = deque(maxlen=200)

    try:
        while (time.time() - start_time) < args.hours * 3600.0:
            clean_frames, traj, gen_ms, fps, tag, telemetry = runner.execute_turn_step()
            cur_stage = runner.manifest.stages[runner.curr_idx]

            pacer.push_generated_batch(
                clean_frames,
                traj,
                cur_stage,
                runner.active_entity_parties,
                runner.maneuver_phase + 1,
                gen_ms,
                runner.steps,
            )

            raw_vid_ms = len(clean_frames) / 16.0 * 1000.0
            inst_rtf = raw_vid_ms / max(1.0, gen_ms)
            lag_ms = gen_ms - raw_vid_ms

            telemetry["realtime_pacing"] = {
                "gen_ms": round(gen_ms, 1),
                "raw_video_ms": round(raw_vid_ms, 1),
                "inst_rtf": round(inst_rtf, 2),
                "ema_rtf": round(pacer.ema_rtf, 2),
                "play_speed": pacer.play_speed,
                "lag_ms": round(lag_ms, 1),
            }

            push_web_telem(telemetry, fps=16.0 * pacer.play_speed, gen_ms=gen_ms)

            compact_entry = build_compact_telemetry(telemetry)
            compact_line = json.dumps(compact_entry, ensure_ascii=False)
            compact_ring.append(compact_line)
            with open(compact_log_path, "w", encoding="utf-8") as cf:
                cf.write("\n".join(compact_ring) + "\n")

            if video_writer is not None:
                for idx_f, rgb in enumerate(clean_frames):
                    out_rgb = pacer._render_hud_on_clean_frame(
                        rgb,
                        traj[min(idx_f, len(traj) - 1)],
                        cur_stage,
                        runner.active_entity_parties,
                        runner.maneuver_phase + 1,
                        0.0,
                    )
                    video_writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

            print(f"⚡ {compact_line}", flush=True)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        pacer.running = False
        if amodal_engine is not None:
            amodal_engine.stop()
        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    main()

