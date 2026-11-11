#!/usr/bin/env python3
"""
neuro_clash_hud.py — Чистый Дебаг-HUD Всех 524,288-D Выборов Рабочей Памяти и Ветвлений Реакций
===============================================================================================
- Исключительно диагностическая визуализация (0% влияния на расчёты движка).
- В карточке каждого из N агентов показывает:
  1) Текущую роль ([CAT:UNCONSUMED], [SUBSTRATE], [HOLD_WM])
  2) ВЫБРАННЫЙ концепт (★ CHOSEN с %) и ОТВЕРГНУТЫЕ альтернативы из пула (○ REJECTED с %)
- В нижней панели показывает конкуренцию ВСЕХ доступных правил реакций (RPS Branches).
"""

from __future__ import annotations
import colorsys
import cv2
import numpy as np
from neuro_world import GameWorldManifest, StageSpec
from neuro_referee import LatentIdentityFrameTelemetry


def _get_ui_color(idx: int) -> tuple[int, int, int]:
    hue = (0.55 + idx * 0.381966) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_identity_hud(
    frame_rgb: np.ndarray,
    traj: list[LatentIdentityFrameTelemetry],
    curr_idx: int,
    stage: StageSpec,
    entity_parties: list[list[str]],
    manifest: GameWorldManifest,
    attempt_num: int,
    game_mode: str,
    monochrome: bool = False,
) -> np.ndarray:
    canvas = frame_rgb.copy()
    h, w, _ = canvas.shape

    if monochrome:
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    cur = traj[curr_idx]
    n_f = len(traj)
    n_total = max(1, len(stage.entities))

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 76), (10, 12, 18), -1)
    cv2.rectangle(overlay, (0, h - 76), (w, h), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.84, canvas, 0.16, 0, canvas)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_bar(x0: int, y0: int, bw: int, bh: int, val: float, col: tuple[int, int, int], title: str):
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (35, 40, 50), -1)
        f_px = int(bw * np.clip(val / 100.0, 0.0, 1.0))
        if f_px > 0:
            cv2.rectangle(canvas, (x0, y0), (x0 + f_px, y0 + bh), col, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (150, 160, 175), 1)
        cv2.putText(canvas, f"{title}: {val:3.0f}%", (x0, y0 - 3), font, 0.28, (235, 240, 250), 1, cv2.LINE_AA)

    draw_bar(12, 14, 165, 8, cur.catalytic_resonance_pct, (55, 185, 255), "CAT RESONANCE")
    draw_bar(190, 14, 165, 8, cur.prob_completed_pct, (255, 195, 40), "ORACLE P(H+)")
    draw_bar(368, 14, 260, 8, cur.quest_progress_pct, (80, 235, 135), f"SUBSTRATE [{stage.quest.driver_tag}]")

    # Карточки N агентов рабочей памяти с дебагом ВЫБРАННОГО (★) и ОТВЕРГНУТЫХ (○) состояний:
    card_w = min(204, int((w - 24) // n_total) - 4)
    for idx, slot in enumerate(stage.entities):
        x_slot = 12 + idx * (card_w + 4)
        slot_color = _get_ui_color(idx)

        trans_pct = cur.entity_transition_pcts[idx] if idx < len(cur.entity_transition_pcts) else 0.0
        state_lbl = (
            cur.entity_state_labels[idx]
            if idx < len(cur.entity_state_labels)
            else f"{slot.slot_id}:{slot.start_state_id}->{slot.target_state_id}"
        )
        choice_dist = (
            cur.entity_choice_distributions[idx]
            if idx < len(cur.entity_choice_distributions)
            else {}
        )

        is_substrate = (idx == stage.quest.active_slot_idx)
        is_catalyst = (idx == stage.quest.catalyst_slot_idx)

        cv2.rectangle(canvas, (x_slot, 26), (x_slot + card_w, 72), (22, 26, 36), -1)
        fill_w = int(card_w * np.clip((trans_pct if is_substrate else 100.0) / 100.0, 0.0, 1.0))
        if fill_w > 0:
            bar_col = (
                (int(slot_color[0] * 0.65), int(slot_color[1] * 0.65), int(slot_color[2] * 0.65))
                if is_substrate
                else (int(slot_color[0] * 0.35), int(slot_color[1] * 0.35), int(slot_color[2] * 0.35))
            )
            cv2.rectangle(canvas, (x_slot, 26), (x_slot + fill_w, 72), bar_col, -1)

        border_col = (80, 255, 140) if is_substrate else ((255, 215, 50) if is_catalyst else (100, 110, 125))
        cv2.rectangle(canvas, (x_slot, 26), (x_slot + card_w, 72), border_col, 2 if (is_substrate or is_catalyst) else 1)

        if is_substrate:
            role_badge = f"dS:{trans_pct:2.0f}%[SUB]"
        elif is_catalyst:
            role_badge = "INV[CAT]"
        else:
            role_badge = "HOLD_WM"

        # Строка 1: Состояние и привязка региона ЭЭГ
        cv2.putText(
            canvas,
            f"{state_lbl[:22]}",
            (x_slot + 3, 37),
            font,
            0.23,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # Строка 2: Роль + регион рабочей памяти
        cv2.putText(
            canvas,
            f"{role_badge} ({slot.control_binding})",
            (x_slot + 3, 48),
            font,
            0.21,
            (180, 255, 190) if is_substrate else ((255, 235, 130) if is_catalyst else (190, 200, 215)),
            1,
            cv2.LINE_AA,
        )

        # Строки 3 и 4: ДЕБАГ ВЫБОРА (★ Выбранный vs ○ Невыбранные альтернативы из 524,288-D WM)
        if choice_dist:
            sorted_c = sorted(choice_dist.items(), key=lambda x: x[1], reverse=True)
            win_k, win_v = sorted_c[0]
            short_win = win_k.split("_", 1)[-1] if "_" in win_k else win_k
            cv2.putText(
                canvas,
                f"*WIN: {short_win}({win_v:.0f}%)",
                (x_slot + 3, 59),
                font,
                0.20,
                (120, 255, 160),
                1,
                cv2.LINE_AA,
            )
            if len(sorted_c) > 1:
                alt_str = " ".join([f"{k.split('_', 1)[-1]}:{v:.0f}%" for k, v in sorted_c[1:3]])
                cv2.putText(
                    canvas,
                    f"oALT: {alt_str[:24]}",
                    (x_slot + 3, 69),
                    font,
                    0.19,
                    (170, 180, 200),
                    1,
                    cv2.LINE_AA,
                )

    # Нижняя панель: График + Дебаг конкуренции всех ветвей реакций (RPS Branches)
    gx0, gy0 = 12, h - 70
    gw, gh = w - 24, 62
    cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (20, 24, 32), -1)
    cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (75, 85, 100), 1)

    def to_xy(idx: int, val_100: float) -> tuple[int, int]:
        px = gx0 + int((idx / max(1, n_f - 1)) * gw)
        py = gy0 + gh - int(np.clip(val_100 / 100.0, 0.0, 1.0) * (gh - 22)) - 4
        return px, py

    for idx in range(1, curr_idx + 1):
        cv2.line(canvas, to_xy(idx - 1, traj[idx - 1].catalytic_resonance_pct), to_xy(idx, traj[idx].catalytic_resonance_pct), (55, 185, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, to_xy(idx - 1, traj[idx - 1].prob_completed_pct), to_xy(idx, traj[idx].prob_completed_pct), (255, 200, 40), 2, cv2.LINE_AA)
        cv2.line(canvas, to_xy(idx - 1, traj[idx - 1].quest_progress_pct), to_xy(idx, traj[idx].quest_progress_pct), (80, 235, 135), 2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"ACTIVE: {stage.quest.reaction_formula} | r={cur.feigenbaum_r:.4f} ({cur.chaos_regime})",
        (gx0 + 6, gy0 + 11),
        font,
        0.23,
        (225, 240, 255),
        1,
        cv2.LINE_AA,
    )

    if cur.reaction_branch_scores:
        b_sorted = sorted(cur.reaction_branch_scores.items(), key=lambda x: x[1], reverse=True)
        b_parts = []
        for b_k, b_v in b_sorted:
            mark = "*" if b_k == stage.quest.driver_tag else "o"
            b_parts.append(f"{mark}{b_k}:{b_v:.0f}%")
        branches_line = "BRANCH CHOICES (128D): " + "  |  ".join(b_parts)
        cv2.putText(
            canvas,
            branches_line[:98],
            (gx0 + 6, gy0 + 22),
            font,
            0.21,
            (255, 220, 120),
            1,
            cv2.LINE_AA,
        )

    return canvas
