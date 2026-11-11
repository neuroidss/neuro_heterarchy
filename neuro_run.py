#!/usr/bin/env python3
"""
===================================================================================
neuro_run.py — Универсальный лаунчер для произвольного числа N сущностей рабочей памяти
===================================================================================
"""

from __future__ import annotations
import argparse
import gc
import json
from multiprocessing.connection import Client
from pathlib import Path
import time
import cv2

from neuro_world import load_game_manifest
from neuro_engine import UniversalLatentAutobattlerEngine

MODE_TO_MANIFEST = {
    "walk": [
        "configs/mode_00_arena_walk_handshake.json",
        "mode_00_arena_walk_handshake.json",
    ],
    "maze": [
        "configs/mode_03_semantic_maze_gm.json",
        "mode_03_semantic_maze_gm.json",
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


def main():
    parser = argparse.ArgumentParser(
        description="Universal Continuous Multi-Block Semantic World for N Entities"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["walk", "maze", "idle_duel", "active_duel"],
        help="Быстрый пресет: walk | maze | idle_duel | active_duel",
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
        help="Источник рабочей памяти для сущностей (например 'auto', 'eeg:auto' или 'WM_0=eeg:FCz,WM_1=eeg:AFz,WM_2=eeg:F3')",
    )
    parser.add_argument(
        "--context-seconds",
        "--prep-seconds",
        dest="prep_seconds",
        type=float,
        default=2.5,
        help="Длительность контекста",
    )
    parser.add_argument(
        "--combat-seconds",
        "--block-seconds",
        dest="combat_seconds",
        type=float,
        default=1.5,
        help="Длительность генерируемого блока",
    )
    parser.add_argument(
        "--monochrome",
        action="store_true",
        default=False,
        help="Монохромный режим отображения",
    )
    parser.add_argument(
        "--game-mode",
        type=str,
        default="strict_ko",
        choices=["strict_ko", "sudden_death"],
    )
    parser.add_argument("--max-overtime-chunks", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--gateway-port", type=int, default=6002)
    parser.add_argument(
        "--battle-server-port",
        type=int,
        default=6008,
        help="Порт neuro_battle_server.py (по умолчанию 6008)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Запустить LTX-Video локально в этом процессе",
    )
    parser.add_argument(
        "--save-mp4", type=str, default="three_phase_slime_autobattler.mp4"
    )
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    resolved_manifest = resolve_manifest_path(args.mode, args.manifest)
    manifest = load_game_manifest(resolved_manifest)

    save_path = "" if args.no_save else args.save_mp4
    telemetry_path = (
        str(Path(save_path).with_suffix("")) + "_telemetry.jsonl"
        if save_path
        else "three_phase_slime_autobattler_telemetry.jsonl"
    )

    runner = UniversalLatentAutobattlerEngine(
        manifest=manifest,
        width=args.width,
        height=args.height,
        prep_seconds=args.prep_seconds,
        combat_seconds=args.combat_seconds,
        game_mode=args.game_mode,
        max_overtime_chunks=args.max_overtime_chunks,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        battle_server_port=args.battle_server_port,
        force_local=args.local,
        monochrome=args.monochrome,
    )

    if args.entity_source:
        if "=" in args.entity_source:
            for pair in args.entity_source.split(","):
                if "=" in pair:
                    slot_k, src_v = pair.split("=", 1)
                    runner.set_entity_source(slot_k.strip(), src_v.strip())
        else:
            runner.set_all_entities_source(args.entity_source.strip())

    gateway_conn = None
    try:
        gateway_conn = Client(("localhost", args.gateway_port), authkey=b"canvas")
    except Exception:
        pass

    video_writer = None
    if not args.no_save and save_path:
        video_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            16.0,
            (runner.width, runner.height),
        )

    start_time = time.time()
    frame_seq_id = 0

    try:
        with open(telemetry_path, "w", encoding="utf-8") as log_f:
            while (time.time() - start_time) < args.hours * 3600.0:
                frames, gen_ms, fps, tag, telemetry = runner.execute_turn_step()

                log_f.write(json.dumps(telemetry, ensure_ascii=False) + "\n")
                log_f.flush()

                if video_writer is not None:
                    for rgb in frames:
                        video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                if gateway_conn is not None:
                    encoded_batch = [
                        cv2.imencode(
                            ".jpg",
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )[1].tobytes()
                        for rgb in frames
                    ]
                    frame_seq_id += len(encoded_batch)
                    try:
                        gateway_conn.send({
                            "frame_id": frame_seq_id,
                            "ts": time.time(),
                            "gen_ms": gen_ms,
                            "fps": fps,
                            "actual_steps": runner.steps,
                            "w": runner.width,
                            "h": runner.height,
                            "use_taesd": False,
                            "use_color": not args.monochrome,
                            "jpg_frames": encoded_batch,
                            "jpg": encoded_batch[-1],
                        })
                    except Exception:
                        gateway_conn = None

                if runner.block_idx % 10 == 0:
                    gc.collect()

                print(
                    f"⏱️ [#{runner.block_idx:03d} | {telemetry['t']}s]"
                    f" {tag} | {gen_ms:.0f}ms",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        if video_writer is not None:
            video_writer.release()
        if gateway_conn is not None:
            try:
                gateway_conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
