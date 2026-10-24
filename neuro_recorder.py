#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS SESSION RECORDER (AUTO-RECONNECTING ZERO-LOSS MULTIMODAL CAPTURE)
===================================================================================
- НЕПРЕРЫВНЫЙ АВТО-ПОИСК LSL (resolve_streams каждые 0.5с):
  Поток ЭЭГ подхватывается на лету, даже если донгл/BLE включили ПОСЛЕ старта записи!
- БУФЕР 360 СЕКУНД: Ноль потерь пакетов при любой нагрузке GPU.
- ИСТИННЫЙ 1:1 ХРОНОМЕТРАЖ: Видео и ЭЭГ привязаны к аппаратному local_clock().
===================================================================================
"""

from __future__ import annotations
import os
import time
import json
import csv
import threading
import numpy as np
from pathlib import Path
from pylsl import local_clock, StreamInfo, StreamOutlet, resolve_streams, StreamInlet

try:
    import cv2
except ImportError:
    cv2 = None


class NeuroSessionRecorder:
    def __init__(self, output_root="sessions", video_fps=30.0):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.session_dir = None
        self.lock = threading.Lock()
        
        self.eeg_thread = None
        self.stop_eeg_event = threading.Event()
        self.eeg_timestamps = []
        self.eeg_samples = []
        
        self.video_fps = float(video_fps)
        self.video_writer = None
        self.video_path = None
        self.video_dims = None
        
        self.t_session_start = None
        self.last_frame_bgr = None
        self.last_frame_t = None
        self.last_frame_id = -1
        self.last_telemetry = None
        self.total_video_ticks = 0
        self.unique_frames_count = 0
        
        self.events_file = None
        self.events_writer = None
        
        info = StreamInfo('NeuroCanvas_Markers', 'Markers', 1, 0, 'string', 'neurocanvas_mark_01')
        self.marker_outlet = StreamOutlet(info)

    def _eeg_capture_loop(self):
        inlets = []
        connected_uids = set()
        last_resolve_t = 0.0

        while not self.stop_eeg_event.is_set():
            now = time.time()
            # Каждые 0.5с проверяем наличие новых/переподключенных LSL потоков ЭЭГ!
            if now - last_resolve_t >= 0.50:
                last_resolve_t = now
                try:
                    all_streams = resolve_streams(wait_time=0.05)
                    live_uids = {s.source_id() for s in all_streams}
                    
                    # Убираем отключенные
                    inlets = [inl for inl in inlets if inl.info().source_id() in live_uids]
                    connected_uids = {inl.info().source_id() for inl in inlets}

                    for s in all_streams:
                        uid = s.source_id()
                        stype = s.type().upper()
                        sname = s.name().upper()
                        if uid not in connected_uids and (stype == 'EEG' or 'FREEEEG' in sname or 'USER' in sname):
                            # Буфер 360 секунд, чтобы ни один сэмпл не пропал!
                            inl = StreamInlet(s, max_buflen=360, max_chunklen=64, recover=True)
                            inlets.append(inl)
                            connected_uids.add(uid)
                            print(f"⚡ [RECORDER LSL] Захвачен поток ЭЭГ: '{s.name()}' ({s.channel_count()} кан, {s.nominal_srate():.0f} Гц)")
                except Exception:
                    pass

            pulled_any = False
            for inl in list(inlets):
                try:
                    chunk, timestamps = inl.pull_chunk(timeout=0.0, max_samples=128)
                    if timestamps and len(timestamps) > 0:
                        pulled_any = True
                        with self.lock:
                            self.eeg_timestamps.extend(timestamps)
                            self.eeg_samples.extend(chunk)
                except Exception:
                    pass

            if not pulled_any:
                time.sleep(0.005)

    def start_session(self, session_name=None):
        with self.lock:
            if self.is_recording:
                return str(self.session_dir)
            
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            s_name = f"session_{ts_str}" if not session_name else f"{session_name}_{ts_str}"
            self.session_dir = self.output_root / s_name
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / "frames").mkdir(exist_ok=True)
            
            csv_path = self.session_dir / "events.csv"
            self.events_file = open(csv_path, mode="w", newline="", encoding="utf-8")
            self.events_writer = csv.writer(self.events_file)
            self.events_writer.writerow([
                "unique_frame_id", "lsl_timestamp", "wall_time", "duration_ms",
                "video_tick_start", "video_tick_end", "event_type",
                "root_concept", "child_concept", "dag_relation",
                "x_local", "y_global", "dyn_strength", "active_pool"
            ])
            self.events_file.flush()

            self.video_path = self.session_dir / "video_timeline.mp4"
            self.video_writer = None
            self.video_dims = None
            
            self.eeg_timestamps.clear()
            self.eeg_samples.clear()
            
            self.t_session_start = local_clock()
            self.last_frame_bgr = None
            self.last_frame_t = None
            self.last_frame_id = -1
            self.last_telemetry = None
            self.total_video_ticks = 0
            self.unique_frames_count = 0
            
            self.is_recording = True
            self.stop_eeg_event.clear()
            
            self.eeg_thread = threading.Thread(target=self._eeg_capture_loop, daemon=True)
            self.eeg_thread.start()

            self.marker_outlet.push_sample([f"SESSION_START|{s_name}"], timestamp=self.t_session_start)
            print(f"\n🔴 [RECORDER] Сессия запущена: {self.session_dir}")
            return str(self.session_dir)

    def _flush_previous_frame_ticks(self, current_t: float):
        if self.last_frame_bgr is None or self.video_writer is None:
            return

        elapsed_sec = current_t - self.t_session_start
        target_total_ticks = int(round(elapsed_sec * self.video_fps))
        ticks_to_write = max(1, target_total_ticks - self.total_video_ticks)
        
        tick_start = self.total_video_ticks
        for _ in range(ticks_to_write):
            self.video_writer.write(self.last_frame_bgr)
        self.total_video_ticks += ticks_to_write
        tick_end = self.total_video_ticks - 1

        duration_ms = (current_t - self.last_frame_t) * 1000.0

        telem = self.last_telemetry or {}
        dag = telem.get("dag_log_parts", ["NONE"])
        root_c = dag[0] if len(dag) > 0 else "NONE"
        child_c = dag[1] if len(dag) > 1 else ""
        dag_rel = "⊃" if "⊃" in " ".join(dag) else ("∥" if "∥" in " ".join(dag) else "●")
        xl = telem.get("f3_x_local", 0.0)
        yg = telem.get("f3_y_global", 0.0)
        dyn_str = telem.get("dyn_strength", 0.60)
        active_pool = "|".join(telem.get("slot_names", []))

        self.events_writer.writerow([
            self.last_frame_id, f"{self.last_frame_t:.6f}", f"{time.time():.4f}", f"{duration_ms:.1f}",
            tick_start, tick_end, "FRAME_DISPLAYED",
            root_c, child_c, dag_rel,
            f"{xl:.4f}", f"{yg:.4f}", f"{dyn_str:.4f}", active_pool
        ])
        self.events_file.flush()

    def push_frame(self, frame_bgr: np.ndarray, telemetry: dict):
        if not self.is_recording or frame_bgr is None or cv2 is None:
            return
        t_now = local_clock()

        with self.lock:
            h, w = frame_bgr.shape[:2]

            if self.video_writer is None:
                self.video_dims = (w, h)
                for codec in ['mp4v', 'avc1', 'MJPG']:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(str(self.video_path), fourcc, self.video_fps, (w, h))
                    if writer.isOpened():
                        self.video_writer = writer
                        break

            if self.last_frame_bgr is not None:
                self._flush_previous_frame_ticks(t_now)

            f_id = self.unique_frames_count
            self.unique_frames_count += 1
            frame_jpg_path = self.session_dir / "frames" / f"frame_{f_id:06d}.jpg"
            cv2.imwrite(str(frame_jpg_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

            self.last_frame_bgr = frame_bgr.copy()
            self.last_frame_t = t_now
            self.last_frame_id = f_id
            self.last_telemetry = telemetry.copy() if telemetry else {}

            root_c = telemetry.get("dag_log_parts", ["NONE"])[0] if telemetry else "NONE"
            self.marker_outlet.push_sample([f"FRAME:{f_id}|{root_c}"], timestamp=t_now)

    def log_custom_event(self, event_name: str, payload_str: str = ""):
        if not self.is_recording:
            return
        t_lsl = local_clock()
        with self.lock:
            self.events_writer.writerow([
                -1, f"{t_lsl:.6f}", f"{time.time():.4f}", "0.0",
                self.total_video_ticks, self.total_video_ticks, event_name,
                "", "", "", "", "", "", payload_str
            ])
            self.events_file.flush()
            self.marker_outlet.push_sample([f"EVENT:{event_name}|{payload_str}"], timestamp=t_lsl)

    def stop_session(self):
        with self.lock:
            if not self.is_recording:
                return None
            self.is_recording = False
            t_stop = local_clock()
            self.stop_eeg_event.set()

            if self.last_frame_bgr is not None:
                self._flush_previous_frame_ticks(t_stop)

            self.marker_outlet.push_sample(["SESSION_STOP"], timestamp=t_stop)

            if self.events_file:
                self.events_file.close()
                self.events_file = None
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            total_sec = float(t_stop - self.t_session_start) if self.t_session_start else 0.0

            if cv2 is not None and self.video_path.exists() and self.video_path.stat().st_size < 1000 and self.unique_frames_count > 0:
                self._rebuild_timeline_video()

            # Сохранение массива ЭЭГ
            n_samples = len(self.eeg_samples)
            if n_samples > 0:
                eeg_arr = np.array(self.eeg_samples, dtype=np.float32)
                ts_arr = np.array(self.eeg_timestamps, dtype=np.float64)
                eeg_dur = float(ts_arr[-1] - ts_arr[0]) if len(ts_arr) > 1 else 0.0
                ch_count = eeg_arr.shape[1] if eeg_arr.ndim > 1 else 16
                print(f"💾 [RECORDER] Успешно записано ЭЭГ: {n_samples} сэмплов ({ch_count} каналов, {eeg_dur:.2f} сек)")
            else:
                print(f"⚠️ [RECORDER ПРЕДУПРЕЖДЕНИЕ] За сессию не пришло ни одного сэмпла ЭЭГ! (Проверьте, что LSL-поток был активен)")
                eeg_arr = np.zeros((0, 16), dtype=np.float32)
                ts_arr = np.zeros(0, dtype=np.float64)
                eeg_dur = 0.0

            np.savez_compressed(
                self.session_dir / "eeg_raw.npz",
                data=eeg_arr,
                timestamps=ts_arr,
                channels=[f"Ch_{i+1}" for i in range(16)]
            )

            manifest = {
                "session_path": str(self.session_dir),
                "unique_generated_frames": self.unique_frames_count,
                "video_timeline_frames": self.total_video_ticks,
                "video_fps": self.video_fps,
                "duration_seconds": total_sec,
                "video_duration_seconds": self.total_video_ticks / self.video_fps,
                "eeg_samples_count": n_samples,
                "eeg_duration_seconds": eeg_dur,
                "created_at": time.ctime()
            }
            with open(self.session_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"⏹️ [RECORDER] Сессия полностью закрыта и запечатана: {self.session_dir}")
            res_dir = str(self.session_dir)
            self.session_dir = None
            return res_dir

    def _rebuild_timeline_video(self):
        if cv2 is None:
            return
        try:
            csv_path = self.session_dir / "events.csv"
            if not csv_path.exists():
                return
            first_img = cv2.imread(str(self.session_dir / "frames" / "frame_000000.jpg"))
            if first_img is None:
                return
            h, w = first_img.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            re_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.video_fps, (w, h))

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('event_type') != 'FRAME_DISPLAYED':
                        continue
                    f_id = int(row['unique_frame_id'])
                    t_start = int(row['video_tick_start'])
                    t_end = int(row['video_tick_end'])
                    repeats = max(1, t_end - t_start + 1)
                    im_file = self.session_dir / "frames" / f"frame_{f_id:06d}.jpg"
                    if im_file.exists():
                        img = cv2.imread(str(im_file))
                        for _ in range(repeats):
                            re_writer.write(img)
            re_writer.release()
        except Exception as e:
            print(f"❌ [RECORDER ERROR]: {e}")


recorder_instance = NeuroSessionRecorder()
