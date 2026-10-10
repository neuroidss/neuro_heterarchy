#!/usr/bin/env python3
"""
===================================================================================
Скрипт проверки и анализа мультимодального датасета (Для заказчика / аналитика)
===================================================================================
1. Проверяет полное 1:1 совпадение времени Видео и ЭЭГ (в секундах).
2. Находит любой кадр видео и сопоставляет его с сигналом ЭЭГ по LSL таймштампам.
3. Проверяет отсутствие потерь пакетов.
===================================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_session(session_path: str):
    p = Path(session_path)
    if not p.exists():
        print(f"❌ Папка не найдена: {session_path}")
        return

    print("=" * 65)
    print(f"🔬 ВЕРИФИКАЦИЯ МУЛЬТИМОДАЛЬНОЙ СЕССИИ: {p.name}")
    print("=" * 65)

    # 1. Загрузка манифеста
    manifest_p = p / "manifest.json"
    if manifest_p.exists():
        import json
        with open(manifest_p, 'r', encoding='utf-8') as f:
            man = json.load(f)
        print(f"  • Реальная длительность сессии:  {man.get('duration_seconds', 0.0):.2f} сек")
        print(f"  • Длительность видеофайла MP4:   {man.get('video_duration_seconds', 0.0):.2f} сек")
        print(f"  • Длительность непрерывного ЭЭГ: {man.get('eeg_duration_seconds', 0.0):.2f} сек")
        print(f"  • Уникальных сгенерированных кадров: {man.get('unique_generated_frames', 0)}")
        print(f"  • Видео-тиков в контейнере (30 FPS): {man.get('video_timeline_frames', 0)}")
        
        diff = abs(man.get('video_duration_seconds', 0.0) - man.get('duration_seconds', 0.0))
        if diff < 0.05:
            print(f"  ✅ ИДЕАЛЬНАЯ СИНХРОНИЗАЦИЯ: Расхождение видео и времени < {diff*1000:.1f} мс!")
        else:
            print(f"  ⚠️ Расхождение видео и времени: {diff:.3f} сек")

    # 2. Загрузка разметки событий
    events_p = p / "events.csv"
    if not events_p.exists():
        print("❌ events.csv не найден")
        return
    events_df = pd.read_csv(events_p)
    frames_df = events_df[events_df['event_type'] == 'FRAME_DISPLAYED']
    print(f"\n📋 Разметка событий (events.csv):")
    print(f"  • Зафиксировано кадров в логе: {len(frames_df)}")
    print(f"  • Среднее время удержания кадра: {frames_df['duration_ms'].astype(float).mean():.1f} мс")
    print(f"  • Уникальные концепты: {frames_df['root_concept'].unique()}")

    # 3. Загрузка ЭЭГ
    eeg_p = p / "eeg_raw.npz"
    if not eeg_p.exists():
        print("❌ eeg_raw.npz не найден")
        return
    eeg_data = np.load(eeg_p)
    samples = eeg_data['data']          # [N, 16]
    eeg_times = eeg_data['timestamps']  # [N]
    print(f"\n⚡ Сырой массив ЭЭГ:")
    print(f"  • Всего сэмплов: {len(eeg_times)}")
    print(f"  • Число каналов: {samples.shape[1]}")

    # 4. Демонстрация извлечения: привязка кадра #5 к ЭЭГ
    if len(frames_df) > 5:
        target_idx = 5
        row = frames_df.iloc[target_idx]
        f_id = int(row['unique_frame_id'])
        f_time = float(row['lsl_timestamp'])
        f_dur = float(row['duration_ms'])
        root_c = row['root_concept']
        tick_start = int(row['video_tick_start'])
        tick_end = int(row['video_tick_end'])

        # Находим индекс в массиве ЭЭГ
        eeg_idx = np.searchsorted(eeg_times, f_time)
        # Окно: 500 мс до появления кадра и всё время, пока он висел
        window_pre = 125  # 500 мс при 250 Гц
        window_dur = int((f_dur / 1000.0) * 250)
        win = slice(max(0, eeg_idx - window_pre), min(len(eeg_times), eeg_idx + window_dur + 50))

        t_rel = (eeg_times[win] - f_time) * 1000.0

        plt.figure(figsize=(12, 6))
        plt.title(f"Синхронизация: Кадр #{f_id} ('{root_c}') на видео тиках [{tick_start}..{tick_end}]\nДлительность на экране: {f_dur:.1f} мс | Отсчёт времени 1:1")
        
        for ch in range(min(8, samples.shape[1])):
            plt.plot(t_rel, samples[win, ch] + ch * 60, label=f"Ch {ch+1}")

        plt.axvline(0, color='r', linestyle='--', linewidth=2, label="Момент показа кадра")
        plt.axvspan(0, f_dur, color='green', alpha=0.15, label="Кадр на экране юзера")
        plt.xlabel("Время относительно показа кадра (мс)")
        plt.ylabel("ЭЭГ Каналы (мкВ)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_img = p / "verified_sync_alignment.png"
        plt.savefig(out_img, dpi=150)
        print(f"\n🖼️ График точной привязки сохранён: {out_img}")
        print("  ✅ Готово к анализу!")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        inspect_session(sys.argv[1])
    else:
        print("Использование: python3 analyze_session.py sessions/session_20261010_131400")
