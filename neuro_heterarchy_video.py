#!/usr/bin/env python3
"""
===================================================================================
neuro_heterarchy_video.py — Единый 3D-DiT Видео-Раннер (Каталитический & Чистый WM)
===================================================================================
- Поддерживает КАК JSON-манифесты с катализаторами/NPC (configs/*.json),
  ТАК И обычные .txt лоры (lore/*.txt) — автоматически отключая Оракул при отсутствии гипотез.
- Подключается к video_world_server.py на порту 6006 (authkey=b"video_world").
- Работает с web_cloud_gateway.py (:6002 видео + :6007 управление, геймпад, set_init_image).
- Тихий LSL (без INFO/WARN спама) + атомарная очередь команд из браузера (0 гонок IPC).
===================================================================================
"""

from __future__ import annotations
import os
import tempfile

# Подавляем отладочный INFO/WARN спам liblsl ДО любых импортов pylsl:
if "LSLAPICFG" not in os.environ:
    _lsl_cfg = os.path.join(tempfile.gettempdir(), "lsl_api.cfg")
    try:
        with open(_lsl_cfg, "w", encoding="utf-8") as _f:
            _f.write("[logging]\nlevel = -2\n")
        os.environ["LSLAPICFG"] = _lsl_cfg
    except Exception:
        pass
os.environ.setdefault("LIBLSL_LOG_LEVEL", "-2")

import argparse
from collections import deque
from copy import copy
import json
from multiprocessing.connection import Client
from pathlib import Path
import threading
import time
import cv2
import numpy as np

from neuro_world import (
    load_game_manifest,
    StageSpec,
    StageEntitySlot,
    GenerativeQuestSpec,
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
    "ball": [
        "configs/mode_05_wm_control_test_arena.json",
        "mode_05_wm_control_test_arena.json",
    ],
    "maze": [
        "configs/mode_03_semantic_maze_gm.json",
        "mode_03_semantic_maze_gm.json",
    ],
    "ritual": [
        "configs/mode_04_forest_neuro_ritual.json",
        "mode_04_forest_neuro_ritual.json",
    ],
    "duel": [
        "configs/mode_02_catalytic_duel.json",
        "mode_02_catalytic_duel.json",
    ],
    "psy": [
        "lore/video_lore_psy.txt",
        "lore/video_lore_plain_psy.txt",
    ],
}

ROLE_TO_ENTITY_SLOT = {
    "FCz": "e0",
    "AFz": "e1",
    "F3":  "e2",
    "F4":  "e3",
    "Fpz": "e0",
}


def resolve_lore_or_manifest_path(
    mode: str | None, manifest_arg: str, lore_arg: str
) -> str:
    if lore_arg and lore_arg.strip():
        p = lore_arg.strip()
        if Path(p).exists():
            return p
        return p

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


def load_txt_lore_as_manifest(txt_path: str) -> GameWorldManifest:
    """Превращает обычный .txt лор (lore/*.txt) в 1-стадийный манифест без Оракула (чистый WM-поток)."""
    secs: dict[str, list[str]] = {
        "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
    }
    unassigned: list[str] = []
    if txt_path and Path(txt_path).is_file():
        cur = None
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("[") and s.endswith("]"):
                    cur = s[1:-1].strip().upper()
                    if cur not in secs:
                        secs[cur] = []
                else:
                    if cur is None:
                        unassigned.append(s)
                    else:
                        secs[cur].append(s)

    if unassigned and not secs["FCZ"]:
        secs["FCZ"] = list(unassigned)

    fpz_ctx = secs["FPZ"][0].rstrip(".") if secs["FPZ"] else ""
    fcz_first = secs["FCZ"][0] if secs["FCZ"] else fpz_ctx
    neg_str = ", ".join(secs["NEGATIVE"]) if secs["NEGATIVE"] else ""

    archetypes: dict[str, ArchetypeSpec] = {}
    entities: list[StageEntitySlot] = []
    reg_bindings = [
        (0, "FCZ", "eeg:FCz"),
        (1, "AFZ", "eeg:AFz"),
        (2, "F3",  "eeg:F3"),
        (3, "F4",  "eeg:F4"),
    ]
    for idx_s, r_key, binding in reg_bindings:
        lines = secs.get(r_key, [])
        pool_ids = []
        for l_i, txt in enumerate(lines):
            aid = f"e{idx_s}_s{l_i}"
            pool_ids.append(aid)
            archetypes[aid] = ArchetypeSpec(
                archetype_id=aid,
                element_tag=f"{r_key}_{l_i}",
                role_tag="WM_AGENT",
                display_name=f"e{idx_s}:{txt[:18]}",
                essence_desc=txt,
                action_desc="",
            )
        entities.append(
            StageEntitySlot(
                slot_id=f"e{idx_s}",
                control_binding=binding,
                party_size=1,
                allowed_pool=pool_ids,
            )
        )

    stage0 = StageSpec(
        global_stage=1,
        stage_title=Path(txt_path).stem if txt_path else "Pure_WM_Stream",
        entities=entities,
        quest=GenerativeQuestSpec(
            ongoing_hypothesis="",
            completed_hypothesis="",
            failed_hypothesis="",
            active_slot_idx=0,
            catalyst_slot_idx=0,
            driver_tag="PURE_WM",
            reaction_formula="PURE_WM_STREAM",
        ),
    )

    return GameWorldManifest(
        world_name=Path(txt_path).stem if txt_path else "Pure_WM_World",
        arena_context_desc=fpz_ctx if fpz_ctx != fcz_first else "",
        static_hold_desc="",
        negative_desc=neg_str,
        archetypes=archetypes,
        stages=[stage0],
    )


def load_manifest_or_txt_lore(path_str: str) -> GameWorldManifest:
    if path_str.endswith(".json"):
        return load_game_manifest(path_str)
    return load_txt_lore_as_manifest(path_str)


def manifest_to_web_sections(manifest: GameWorldManifest) -> dict[str, list[str]]:
    secs: dict[str, list[str]] = {
        "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
    }
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


def apply_web_sections_to_manifest(
    manifest: GameWorldManifest, sections: dict[str, list[str]]
):
    if "NEGATIVE" in sections:
        neg_list = [s.strip() for s in sections.get("NEGATIVE", []) if s.strip()]
        manifest.negative_desc = ", ".join(neg_list) if neg_list else ""

    if "FPZ" in sections:
        fpz_list = [s.strip() for s in sections.get("FPZ", []) if s.strip()]
        fcz_list = [s.strip() for s in sections.get("FCZ", []) if s.strip()]
        if not fpz_list or (fpz_list == fcz_list):
            manifest.arena_context_desc = ""
        else:
            manifest.arena_context_desc = fpz_list[0].rstrip(".")

    for idx_s, reg_key in [(0, "FCZ"), (1, "AFZ"), (2, "F3"), (3, "F4")]:
        if reg_key not in sections:
            continue
        lines = [s.strip() for s in sections.get(reg_key, []) if s.strip()]
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

    d_str = (
        f" | dS:{telem.quest_progress_pct:.0f}% r:{telem.feigenbaum_r:.3f}"
        if telem
        else ""
    )
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
        runner: UniversalLatentAutobattlerEngine | None = None,
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
        self.runner = runner

        self.frame_queue: deque[
            tuple[
                np.ndarray,
                LatentIdentityFrameTelemetry | None,
                StageSpec | None,
                list[list[str]],
                int,
            ]
        ] = deque(maxlen=160)
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
        self.actual_steps = 8
        self.gateway_conn = None

    def reset_visual_blend(self):
        with self.lock:
            self.frame_queue.clear()
            self.prev_tail_clean_rgb = None
            self.prev_quest_pct = 0.0
            self.hud_traj_history.clear()

    def cycle_hud_mode(self):
        order = ["off", "minimal", "full"]
        idx = (order.index(self.hud_mode) + 1) % len(order) if self.hud_mode in order else 0
        self.hud_mode = order[idx]
        print(
            f"🖥️ [HUD MODE] Режим отображения на экране: {self.hud_mode.upper()} (Клавиша H для смены)",
            flush=True,
        )

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

            target_out_frames = int(
                np.clip(
                    round(self.ema_gen_sec * 16.0),
                    len(frames_work),
                    len(frames_work) * 3,
                )
            )
            self.play_speed = round(
                len(frames_work) / max(1.0, float(target_out_frames)), 2
            )

            base_telem = traj[-1]
            target_q_pct = float(base_telem.quest_progress_pct)
            target_c_pct = float(base_telem.catalytic_resonance_pct)
            start_q_pct = (
                self.prev_quest_pct if target_q_pct >= self.prev_quest_pct else 0.0
            )
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
                    interp_rgb = cv2.addWeighted(
                        frames_work[i0], 1.0 - alpha, frames_work[i1], alpha, 0.0
                    )

                smooth_t = copy(base_telem)
                smooth_t.quest_progress_pct = round(
                    start_q_pct + (target_q_pct - start_q_pct) * frac, 1
                )
                smooth_t.catalytic_resonance_pct = round(
                    start_c_pct + (target_c_pct - start_c_pct) * frac, 1
                )
                smooth_trans = list(base_telem.entity_transition_pcts)
                if 0 <= stage.quest.active_slot_idx < len(smooth_trans):
                    smooth_trans[stage.quest.active_slot_idx] = (
                        smooth_t.quest_progress_pct
                    )
                smooth_t.entity_transition_pcts = smooth_trans

                self.frame_queue.append(
                    (interp_rgb, smooth_t, stage, list(active_parties), attempt_num)
                )

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
        if (
            self.hud_mode == "off"
            or telem is None
            or stage is None
            or self.manifest is None
        ):
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
        joystick = None
        local_keys_held: dict[str, float] = {}
        win_w, win_h = max(768, self.width * 2), max(512, self.height * 2)

        if self.show_window:
            try:
                if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                    del os.environ["SDL_VIDEODRIVER"]
                import pygame
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                    print(f"🎮 [LOCAL GAMEPAD] Подключен физический геймпад: {joystick.get_name()}")
                screen = pygame.display.set_mode(
                    (win_w, win_h), pygame.HWSURFACE | pygame.DOUBLEBUF
                )
                pygame.display.set_caption(
                    "Exocortex 3D-DiT World ([WASD/Arrows/Gamepad] WM Agent | [H] Toggle HUD)"
                )
                pg = pygame
            except Exception as e:
                print(f"⚠️ [Pygame Window Warning]: {e}")
                pg = None

        last_item = None
        while self.running:
            t_tick = time.perf_counter()

            if pg is not None:
                key_to_ch = {
                    pg.K_w: "LS_UP",
                    pg.K_s: "LS_DOWN",
                    pg.K_a: "LS_LEFT",
                    pg.K_d: "LS_RIGHT",
                    pg.K_UP: "RS_UP",
                    pg.K_DOWN: "RS_DOWN",
                    pg.K_LEFT: "RS_LEFT",
                    pg.K_RIGHT: "RS_RIGHT",
                    pg.K_1: "BTN_A",
                    pg.K_2: "BTN_B",
                    pg.K_3: "BTN_X",
                    pg.K_4: "BTN_Y",
                }
                changed_local = False
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.running = False
                        os._exit(0)
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_h:
                            self.cycle_hud_mode()
                        elif event.key in key_to_ch:
                            local_keys_held[key_to_ch[event.key]] = 1.0
                            changed_local = True
                    elif event.type == pg.KEYUP:
                        if event.key in key_to_ch:
                            local_keys_held.pop(key_to_ch[event.key], None)
                            changed_local = True

                if joystick is not None and self.runner is not None:
                    try:
                        dz = 0.12
                        ax0 = float(joystick.get_axis(0))
                        ax1 = float(joystick.get_axis(1))
                        ax2 = float(joystick.get_axis(2)) if joystick.get_numaxes() > 2 else 0.0
                        ax3 = float(joystick.get_axis(3)) if joystick.get_numaxes() > 3 else 0.0
                        joy_ch = dict(local_keys_held)
                        if ax1 < -dz:
                            joy_ch["LS_UP"] = min(1.0, (-ax1 - dz) / (1.0 - dz))
                        if ax1 > dz:
                            joy_ch["LS_DOWN"] = min(1.0, (ax1 - dz) / (1.0 - dz))
                        if ax0 < -dz:
                            joy_ch["LS_LEFT"] = min(1.0, (-ax0 - dz) / (1.0 - dz))
                        if ax0 > dz:
                            joy_ch["LS_RIGHT"] = min(1.0, (ax0 - dz) / (1.0 - dz))
                        if ax3 < -dz:
                            joy_ch["RS_UP"] = min(1.0, (-ax3 - dz) / (1.0 - dz))
                        if ax3 > dz:
                            joy_ch["RS_DOWN"] = min(1.0, (ax3 - dz) / (1.0 - dz))
                        if ax2 < -dz:
                            joy_ch["RS_LEFT"] = min(1.0, (-ax2 - dz) / (1.0 - dz))
                        if ax2 > dz:
                            joy_ch["RS_RIGHT"] = min(1.0, (ax2 - dz) / (1.0 - dz))
                        self.runner.set_gamepad_channels(joy_ch)
                    except Exception:
                        pass
                elif changed_local and self.runner is not None:
                    self.runner.set_gamepad_channels(local_keys_held)

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
            display_rgb = self._render_hud_on_clean_frame(
                clean_rgb, telem, stg, parties, att_num, q_sec
            )

            if pg is not None and screen is not None:
                try:
                    surf = pg.image.frombuffer(
                        display_rgb.tobytes(),
                        (display_rgb.shape[1], display_rgb.shape[0]),
                        "RGB",
                    )
                    if (display_rgb.shape[1], display_rgb.shape[0]) != (win_w, win_h):
                        surf = pg.transform.smoothscale(surf, (win_w, win_h))
                    screen.blit(surf, (0, 0))
                    pg.display.flip()
                except Exception:
                    pass

            if self.gateway_conn is None and self.gateway_port > 0:
                try:
                    self.gateway_conn = Client(
                        ("localhost", self.gateway_port), authkey=b"canvas"
                    )
                except Exception:
                    self.gateway_conn = None

            if self.gateway_conn is not None:
                try:
                    bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
                    _, enc_jpg = cv2.imencode(
                        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    self.frame_seq_id += 1
                    self.gateway_conn.send(
                        {
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
                        }
                    )
                    if recorder_instance.is_recording:
                        recorder_instance.push_frame(
                            bgr,
                            {
                                "dag_log_parts": [
                                    stg.quest.driver_tag if stg else "3D_DIT"
                                ]
                            },
                        )
                except Exception:
                    self.gateway_conn = None

            target_dt = (
                0.055 if q_len > 16 else (0.0625 if q_len > 3 else 0.075)
            )
            elapsed = time.perf_counter() - t_tick
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)


def start_live_lsl_to_vram_bus_thread(
    pacer: SmoothRealtimeVideoPacer, users_arg: str = ""
):
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
    print(
        "🧠 [LIVE BLE/LSL -> VRAM BUS] 60 Гц конвейер рабочей памяти запущен! Любая BLE-плата из браузера сразу управляет 3D-DiT."
    )
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
                    c = Client(
                        ("localhost", video_ctrl_port), authkey=b"video_ctrl"
                    )
                    with ctrl_lock:
                        ctrl_conn = c
                        ctrl_conn.send(
                            {
                                "cmd": "sync_from_video_runner",
                                "lore_file": initial_manifest_path,
                                "sections": manifest_to_web_sections(
                                    runner.manifest
                                ),
                                "concept_bindings": dict(
                                    runner.concept_bindings_by_region
                                ),
                                "eeg_wm_gain": runner.eeg_wm_gain,
                                "gamepad_wm_mix": runner.gamepad_wm_mix,
                                "monty_mode": "off",
                                "stochastic": runner.stochastic,
                                "wm_radius": runner.wm_radius,
                                "hud_mode": pacer.hud_mode,
                                "context_latents": runner.context_lat_frames,
                                "step_latents": runner.step_lat_frames,
                                "steps": runner.steps,
                                "cfg_scale": runner.cfg_scale,
                            }
                        )
                    print(
                        f"🌐 [WEB STUDIO BRIDGE] Подключено к web_cloud_gateway (порт {video_ctrl_port})! Управление из браузера :8080 активно."
                    )

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
                            runner.stochastic = float(
                                np.clip(float(cfg["stochastic"]), 0.0, 1.0)
                            )
                        if "wm_radius" in cfg:
                            runner.wm_radius = float(
                                np.clip(float(cfg["wm_radius"]), 0.02, 0.85)
                            )
                        elif "geodesic_lock" in cfg:
                            runner.wm_radius = float(
                                np.clip(float(cfg["geodesic_lock"]), 0.02, 0.85)
                            )
                        if "eeg_wm_gain" in cfg:
                            runner.eeg_wm_gain = float(
                                np.clip(float(cfg["eeg_wm_gain"]), 0.0, 1.0)
                            )
                        if "gamepad_wm_mix" in cfg:
                            runner.gamepad_wm_mix = float(
                                np.clip(float(cfg["gamepad_wm_mix"]), 0.0, 1.0)
                            )
                        if "concept_bindings" in cfg and isinstance(
                            cfg["concept_bindings"], dict
                        ):
                            runner.set_concept_bindings(cfg["concept_bindings"])
                        if "hud_mode" in cfg:
                            pacer.hud_mode = str(cfg["hud_mode"]).lower()
                        if "context_latents" in cfg:
                            runner.context_lat_frames = max(
                                1, min(3, int(cfg["context_latents"]))
                            )
                            pacer.context_latents = runner.context_lat_frames
                        if "step_latents" in cfg:
                            runner.step_lat_frames = max(
                                1, min(3, int(cfg["step_latents"]))
                            )
                        elif "num_frames" in cfg:
                            nf = int(cfg["num_frames"])
                            runner.step_lat_frames = max(1, (nf - 1) // 8)
                        if "steps" in cfg:
                            runner.steps = max(3, min(20, int(cfg["steps"])))
                        if "cfg_scale" in cfg:
                            runner.cfg_scale = float(
                                np.clip(float(cfg["cfg_scale"]), 1.0, 6.0)
                            )
                        print(
                            f"🎛️ [WEB CONFIG] Из браузера: wm_radius={runner.wm_radius:.2f}, stoch={runner.stochastic:.2f}, "
                            f"eeg_gain={runner.eeg_wm_gain:.2f}, gp_mix={runner.gamepad_wm_mix:.2f}, "
                            f"hud={pacer.hud_mode}, ctx={runner.context_lat_frames}, step={runner.step_lat_frames}, "
                            f"steps={runner.steps}, cfg={runner.cfg_scale:.1f}"
                        )
                    elif cmd == "set_gamepad_input":
                        if "eeg_wm_gain" in msg:
                            runner.eeg_wm_gain = float(
                                np.clip(float(msg["eeg_wm_gain"]), 0.0, 1.0)
                            )
                        if "gamepad_wm_mix" in msg:
                            runner.gamepad_wm_mix = float(
                                np.clip(float(msg["gamepad_wm_mix"]), 0.0, 1.0)
                            )
                        if "concept_bindings" in msg and isinstance(
                            msg["concept_bindings"], dict
                        ):
                            runner.set_concept_bindings(msg["concept_bindings"])
                        if "channels" in msg and isinstance(
                            msg["channels"], dict
                        ):
                            runner.set_gamepad_channels(msg["channels"])
                        else:
                            runner.set_gamepad_input(
                                lx=float(msg.get("lx", 0.0)),
                                ly=float(msg.get("ly", 0.0)),
                                rx=float(msg.get("rx", 0.0)),
                                ry=float(msg.get("ry", 0.0)),
                                manual_idx=int(msg.get("manual_idx", -1)),
                                target_slot=str(msg.get("target_slot", "*")),
                            )
                    elif cmd == "update_lore_sections":
                        secs = msg.get("sections", {})
                        if isinstance(secs, dict):
                            secs_copy = {
                                k: list(v) for k, v in secs.items() if isinstance(v, list)
                            }
                            runner.request_lore_sections_update(
                                lambda m, s=secs_copy: apply_web_sections_to_manifest(m, s)
                            )
                    elif cmd == "switch_lore":
                        l_file = str(msg.get("lore_file", "")).strip()
                        secs = msg.get("sections")
                        if l_file and Path(l_file).exists():
                            new_manifest = load_manifest_or_txt_lore(l_file)
                            runner.request_manifest_switch(new_manifest, label=l_file)
                        elif not l_file:
                            empty_manifest = load_txt_lore_as_manifest("")
                            runner.request_manifest_switch(empty_manifest, label="FREE_DRIFT")
                        elif isinstance(secs, dict):
                            secs_copy = {
                                k: list(v) for k, v in secs.items() if isinstance(v, list)
                            }
                            runner.request_lore_sections_update(
                                lambda m, s=secs_copy: apply_web_sections_to_manifest(m, s)
                            )
                            runner.request_reset_genesis()
                    elif cmd == "set_init_image":
                        raw_bytes = msg.get("image_bytes")
                        if raw_bytes:
                            runner.request_init_image_bytes(raw_bytes)
                    elif cmd == "reset_genesis":
                        runner.request_reset_genesis()
                        if amodal_engine is not None:
                            amodal_engine.reset_seals()
                    elif cmd == "set_device_mapping":
                        s_name = str(msg.get("stream_name", "")).strip()
                        r_val = str(msg.get("role", "FCz")).strip()
                        u_val = str(msg.get("user", "User1")).strip()
                        if router is not None and s_name:
                            router.set_device_mapping(s_name, r_val, u_val)
                        if r_val and r_val.upper() not in (
                            "NONE",
                            "OFF",
                            "UNASSIGNED",
                        ):
                            func_r = (
                                EXTENDED_ROLE_TO_FUNCTIONAL.get(r_val, "FCz")
                                if HAS_AMODAL_LSL_ENGINE
                                else r_val
                            )
                            target_slot = ROLE_TO_ENTITY_SLOT.get(func_r, "e0")
                            runner.set_entity_source(
                                target_slot, f"eeg:{func_r}"
                            )
                            print(
                                f"📡 [BLE DEVICE MAPPED] Плата '{s_name}' (Субъект: {u_val}) -> Зона: {r_val} ({target_slot}=eeg:{func_r})"
                            )
                    elif cmd == "start_recording":
                        recorder_instance.start_session(
                            session_name=msg.get("session_name")
                        )
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
                    ctrl_conn.send(
                        {
                            "cmd": "video_telemetry",
                            "fps": round(fps, 1),
                            "gen_ms": round(gen_ms, 1),
                            "rtf": round(pacer.ema_rtf, 2),
                            "play_speed": pacer.play_speed,
                            "feigenbaum_r": round(
                                float(c_metrics.get("feigenbaum_r", 3.72)), 3
                            ),
                            "reaction_formula": telem.get(
                                "reaction_formula", ""
                            ),
                            "branch_scores": telem.get(
                                "reaction_branch_competition_128d", {}
                            ),
                            "debug_tokens": f"[{telem.get('driver_tag')}] dS:{q_metrics.get('quest_progress_pct', 0):.0f}% | {telem.get('active_entities', '')}",
                            "fpz_meta_charge": round(
                                float(
                                    c_metrics.get(
                                        "catalytic_resonance_pct", 50.0
                                    )
                                )
                                / 100.0,
                                3,
                            ),
                            "afz_stage_progress": round(
                                float(q_metrics.get("quest_progress_pct", 0.0))
                                / 100.0,
                                3,
                            ),
                            "monty_loss": 0.015,
                            "persistence": round(
                                float(
                                    c_metrics.get(
                                        "catalytic_resonance_pct", 80.0
                                    )
                                )
                                / 100.0,
                                2,
                            ),
                            "surprise": round(
                                abs(
                                    float(c_metrics.get("feigenbaum_r", 3.7))
                                    - 3.57
                                ),
                                2,
                            ),
                            "monty_mode": "off",
                            "stochastic": runner.stochastic,
                            "unassigned_stochastic": runner.stochastic,
                            "wm_radius": runner.wm_radius,
                            "eeg_wm_gain": runner.eeg_wm_gain,
                            "gamepad_wm_mix": runner.gamepad_wm_mix,
                            "hud_mode": pacer.hud_mode,
                            "context_latents": runner.context_lat_frames,
                            "step_latents": runner.step_lat_frames,
                            "steps": runner.steps,
                            "cfg_scale": runner.cfg_scale,
                            "active_region_mask": {
                                "Fpz": True,
                                "AFz": True,
                                "F3": True,
                                "F4": True,
                                "FCz": True,
                            },
                            "active_concept_idx": telem.get(
                                "active_concept_idx",
                                runner.active_concept_idx_by_region,
                            ),
                            "active_concept_indices": telem.get(
                                "active_concept_indices",
                                runner.active_concept_indices_by_region,
                            ),
                            "region_choice_probs": telem.get(
                                "region_choice_probs",
                                runner.region_choice_probs,
                            ),
                            "concept_bindings": telem.get(
                                "concept_bindings",
                                runner.concept_bindings_by_region,
                            ),
                        }
                    )
                except Exception:
                    pass

    return push_telemetry_to_web


def main():
    parser = argparse.ArgumentParser(
        description="Unified 3D-DiT Video Heterarchy Runner (Web Studio :8080 & Local Pygame)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ball",
        choices=["ball", "maze", "ritual", "duel", "psy"],
        help="Быстрый пресет: ball | maze | ritual | duel | psy",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="configs/mode_05_wm_control_test_arena.json",
        help="Путь к внешнему JSON-манифесту мира",
    )
    parser.add_argument(
        "--video-lore",
        "--lore",
        dest="video_lore",
        type=str,
        default="",
        help="Путь к .json манифесту или .txt лору (перекрывает --mode)",
    )
    parser.add_argument(
        "--init-image",
        type=str,
        default="",
        help="Опциональный путь к стартовой PNG/JPG картинке (напр. карта лабиринта)",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="",
        help="Статическая раскладка плат ЭЭГ (например 'User1:FCz=0')",
    )
    parser.add_argument(
        "--entity-source",
        "--wm-source",
        dest="entity_source",
        type=str,
        default="auto",
        help="Источник рабочей памяти ('auto' = из JSON-манифеста, или 'e0=eeg:FCz,e1=npc:AFz')",
    )
    parser.add_argument("--stochastic", type=float, default=0.0)
    parser.add_argument("--wm-radius", type=float, default=0.35)
    parser.add_argument("--eeg-wm-gain", type=float, default=1.0)
    parser.add_argument("--gamepad-wm-mix", type=float, default=1.0)
    parser.add_argument(
        "--hud",
        type=str,
        default="off",
        choices=["off", "minimal", "full"],
    )
    parser.add_argument("--no-hud", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--context-latents", type=int, default=1)
    parser.add_argument("--step-latents", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--monochrome", action="store_true", default=False)
    parser.add_argument("--show-window", action="store_true", default=False)
    parser.add_argument(
        "--game-mode",
        type=str,
        default="strict_ko",
        choices=["strict_ko", "sudden_death"],
    )
    parser.add_argument("--max-overtime-chunks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--gateway-port", type=int, default=6002)
    parser.add_argument("--video-ctrl-port", type=int, default=6007)
    parser.add_argument(
        "--video-port",
        "--battle-server-port",
        dest="video_port",
        type=int,
        default=6006,
        help="Порт video_world_server.py (по умолчанию 6006)",
    )
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--save-mp4", type=str, default="neuro_video_rollout.mp4")
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    eff_hud_mode = "off" if args.no_hud else args.hud
    resolved_manifest = resolve_lore_or_manifest_path(
        args.mode, args.manifest, args.video_lore
    )
    manifest = load_manifest_or_txt_lore(resolved_manifest)

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
        video_server_port=args.video_port,
        force_local=args.local,
        monochrome=args.monochrome,
        context_lat_frames=args.context_latents,
        step_lat_frames=args.step_latents,
        stochastic=args.stochastic,
        wm_radius=args.wm_radius,
    )
    runner.eeg_wm_gain = float(np.clip(args.eeg_wm_gain, 0.0, 1.0))
    runner.gamepad_wm_mix = float(np.clip(args.gamepad_wm_mix, 0.0, 1.0))

    if args.init_image and Path(args.init_image).is_file():
        with open(args.init_image, "rb") as f_img:
            runner.set_init_image_bytes(f_img.read())

    if args.entity_source and args.entity_source.strip().lower() != "auto":
        if "=" in args.entity_source:
            for pair in args.entity_source.split(","):
                if "=" in pair:
                    slot_k, src_v = pair.split("=", 1)
                    runner.set_entity_source(slot_k.strip(), src_v.strip())
        else:
            runner.set_all_entities_source(args.entity_source.strip())

    show_win = bool(args.show_window) and not bool(args.headless)
    pacer = SmoothRealtimeVideoPacer(
        gateway_port=args.gateway_port,
        width=runner.width,
        height=runner.height,
        monochrome=runner.monochrome,
        show_window=show_win,
        hud_mode=eff_hud_mode,
        manifest=manifest,
        game_mode=args.game_mode,
        context_latents=runner.context_lat_frames,
        runner=runner,
    )
    pacer.start()

    router, amodal_engine = start_live_lsl_to_vram_bus_thread(pacer, args.users)

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
            clean_frames, traj, gen_ms, fps, tag, telemetry = (
                runner.execute_turn_step()
            )
            pacer.manifest = runner.manifest
            if runner.consume_visual_reset_flag():
                pacer.reset_visual_blend()
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

            push_web_telem(
                telemetry, fps=16.0 * pacer.play_speed, gen_ms=gen_ms
            )

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

