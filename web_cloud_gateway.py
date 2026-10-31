#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY v10.5 (DUAL STOCHASTIC & PERSISTENT LSL)
===================================================================================
- По умолчанию:
  * stochastic = 0.0 (100% управление рабочей памятью на активных регионах).
  * unassigned_stochastic = 1.0 (живая генерация на пустых регионах без девайса и Монти,
    а при 0.0 — 100% детерминизм для ученых без необходимости набора статистики).
===================================================================================
"""

from __future__ import annotations
import os
import ssl
import wave
import json
import time
import queue
import socket
import struct
import shutil
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from pylsl import StreamInfo, StreamOutlet
import numpy as np

from neuro_recorder import recorder_instance

HTTP_PORT = 8080
IPC_PORT = 6002
IPC_AUTHKEY = b'canvas'
IPC_AUDIO_PORT = 6003
IPC_AUDIO_AUTHKEY = b'audio'
IPC_DIFF_CONTROL_PORT = 6004
IPC_DIFF_AUTHKEY = b'diff_ctrl'
IPC_VIDEO_CONTROL_PORT = 6007
IPC_VIDEO_AUTHKEY = b'video_ctrl'

NUM_CHANNELS = 16
DEFAULT_SPS = 250.0
SAMPLE_RATE = 32000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

latest_jpeg_frame = None
latest_jpeg_id = 0
latest_jpeg_ts = 0.0
last_served_jpeg_id = 0
jpeg_lock = threading.Lock()
jpeg_cond = threading.Condition(jpeg_lock)

audio_listeners: list[queue.Queue] = []
audio_listeners_lock = threading.Lock()

session_wav_writer = None
session_wav_lock = threading.Lock()

active_music_conn = None
music_conn_lock = threading.Lock()

active_diff_ctrl_conn = None
diff_ctrl_lock = threading.Lock()

active_video_ctrl_conn = None
video_ctrl_lock = threading.Lock()

music_state = {
    "concepts": [],
    "preset": "balanced",
    "stochastic": 0.0,
    "unassigned_stochastic": 1.0,
    "target_block": 0.0,
    "actual_block": 0.24,
    "play_speed": 1.0,
    "target_context": 3.6,
    "target_buffer_count": 2,
    "temp": 1.0,
    "top_k": 250,
    "cfg_coef": 1.0,
    "strength_low": 0.25,
    "strength_high": 0.80,
    "ema_smooth": 0.20,
    "dyn_strength": 0.25,
    "buffer_sec": 0.0,
    "buffer_chunks": 0,
    "rtf": 1.5
}
music_state_lock = threading.Lock()

world_state = {
    "concepts": [],
    "preset": "balanced",
    "stochastic": 0.0,
    "unassigned_stochastic": 1.0,
    "steps": 0,
    "actual_steps": 2,
    "resolution": "512x384",
    "use_taesd": True,
    "use_color": True,
    "world_context": 0.18,
    "guidance_scale": 1.0,
    "strength_low": 0.50,
    "strength_high": 0.85,
    "ema_smooth": 0.12,
    "fps": 0.0,
    "gen_ms": 0.0,
    "buffer_frames": 0,
    "buffer_sec": 0.0,
    "is_recording": False,
    "active_session_path": "",
    "current_puck": [0.0, 0.0],
    "active_thought": "",
    "ladder_stage": 4,
    "monty_loss": 0.0,
    "persistence": 0.0,
    "surprise": 0.0,
    "regions": {}
}
world_state_lock = threading.Lock()

sse_clients = []
sse_lock = threading.Lock()


def normalize_lore_relpath(p: str | None) -> str:
    if not p or not str(p).strip():
        return ""
    s = str(p).strip()
    base_prefix = BASE_DIR.rstrip("/") + "/"
    if s.startswith(base_prefix):
        s = s[len(base_prefix):]
    elif s.startswith("/app/"):
        s = s[len("/app/"):]
    return s


def parse_regional_lore_file(lore_path: str) -> dict[str, list[str]]:
    sections = {"FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []}
    rel_p = normalize_lore_relpath(lore_path)
    if not rel_p:
        return sections
    full_p = os.path.join(BASE_DIR, rel_p)
    if os.path.isfile(full_p):
        cur = "FPZ"
        with open(full_p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("[") and s.endswith("]"):
                    cur = s[1:-1].strip().upper()
                    if cur not in sections:
                        sections[cur] = []
                else:
                    sections[cur].append(s)
    return sections


def discover_available_lore_files() -> list[str]:
    lore_dir = os.path.join(BASE_DIR, "lore")
    found = [""]
    if os.path.isdir(lore_dir):
        for fname in sorted(os.listdir(lore_dir)):
            if fname.endswith(".txt") or fname.endswith(".json"):
                found.append(f"lore/{fname}")
    return found


video_state = {
    "connected": False,
    "lore_file": "",
    "available_lores": discover_available_lore_files(),
    "sections": {
        "FPZ": [], "AFZ": [], "F3": [], "F4": [], "FCZ": [], "NEGATIVE": []
    },
    "region_gains": {
        "Fpz": 0.06,
        "AFz": 0.08,
        "F3":  0.08,
        "F4":  0.10,
        "FCz": 0.92
    },
    "auto_active_boost": True,
    "active_user_gain": 0.92,
    "inactive_base_gain": 0.06,
    "cross_region_bleed": 0.25,
    "cascade_rate": 1.0,
    "psi_threshold": 0.55,
    "geodesic_lock": 0.18,
    "num_frames": 5,
    "context_ring_len": 16,
    "steps": 4,
    "genesis_steps": 12,
    "cfg_scale": 3.2,
    "fps": 0.0,
    "gen_ms": 0.0,
    "debug_tokens": "",
    "fpz_meta_charge": 0.0,
    "afz_stage_progress": 0.0,
    "active_region_mask": {"Fpz": False, "AFz": False, "F3": False, "F4": False, "FCz": True},
    "active_concept_idx": {"Fpz": 0, "AFz": 0, "F3": 0, "F4": 0, "FCz": 0}
}
video_state_lock = threading.Lock()


def broadcast_sse(event_name: str, payload: dict):
    msg = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode('utf-8')
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for d in dead:
            if d in sse_clients:
                sse_clients.remove(d)


class DynamicLSLRouter:
    def __init__(self):
        self.device_outlets: dict[str, dict] = {}
        self.lock = threading.Lock()

    def push(self, dev_id: str, samples: list, sps: float = DEFAULT_SPS):
        clean_dev = "".join(c for c in dev_id if c.isalnum())[-6:].upper() or "DEV0"
        stream_name = f"FreeEEG_{clean_dev}"

        with self.lock:
            dev_entry = self.device_outlets.get(dev_id)
            if dev_entry is None:
                info = StreamInfo(
                    name=stream_name,
                    type='EEG',
                    channel_count=NUM_CHANNELS,
                    nominal_srate=float(sps),
                    channel_format='float32',
                    source_id=f"uid_freeeeg_{clean_dev}"
                )
                new_outlet = StreamOutlet(info)
                dev_entry = {
                    "stream_name": stream_name,
                    "outlet": new_outlet
                }
                self.device_outlets[dev_id] = dev_entry
                print(f"🔌 [LSL OUTLET] Открыт постоянный аппаратный поток: '{stream_name}'", flush=True)

            outlet = dev_entry["outlet"]

        for s in samples:
            if len(s) == NUM_CHANNELS:
                outlet.push_sample(s)


lsl_router = DynamicLSLRouter()


def make_wav_header(sample_rate=SAMPLE_RATE, num_channels=1, bits_per_sample=16):
    data_size = 0x7FFFFFFF - 44
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ',
        16, 1, num_channels, sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),
        num_channels * (bits_per_sample // 8),
        bits_per_sample, b'data', data_size
    )


class UnifiedCloudHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _serve_file(self, file_path, content_type):
        if not os.path.isfile(file_path):
            self.send_error(404, f"Файл не найден: {file_path}")
            return
        try:
            file_size = os.path.getsize(file_path)
            self.send_response(200)
            self.send_header('Content-Type', f'{content_type}; charset=utf-8')
            self.send_header('Content-Length', str(file_size))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            with open(file_path, 'rb') as f:
                shutil.copyfileobj(f, self.wfile, length=64 * 1024)
            self.wfile.flush()
        except Exception:
            pass

    def _send_json(self, code, obj):
        payload = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        global last_served_jpeg_id
        clean_path = self.path.split('?')[0].rstrip('/')

        if clean_path in ['', '/index.html']:
            candidate_paths = [
                os.path.join(BASE_DIR, 'static', 'index.html'),
                os.path.join(BASE_DIR, 'index.html')
            ]
            found = next((p for p in candidate_paths if os.path.isfile(p)), None)
            if found:
                self._serve_file(found, 'text/html')
            else:
                self.send_error(404, "index.html не найден")

        elif clean_path in ['/app.js', '/static/app.js']:
            candidate_js = [
                os.path.join(BASE_DIR, 'static', 'app.js'),
                os.path.join(BASE_DIR, 'app.js')
            ]
            found_js = next((p for p in candidate_js if os.path.isfile(p)), None)
            if found_js:
                self._serve_file(found_js, 'application/javascript')
            else:
                self.send_error(404, "app.js не найден")

        elif clean_path == '/frame.jpg':
            with jpeg_lock:
                frame = latest_jpeg_frame
                fid = latest_jpeg_id
                fts = latest_jpeg_ts
                unconsumed = max(0, fid - last_served_jpeg_id - 1)
                last_served_jpeg_id = fid
                age_sec = max(0.0, time.time() - fts) if fts > 0 else 0.0

            with world_state_lock:
                world_state["buffer_frames"] = unconsumed
                world_state["buffer_sec"] = round(age_sec, 2)

            if frame is None:
                self.send_response(204)
                self.end_headers()
                return

            try:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.flush()
            except Exception:
                pass

        elif clean_path in ['/stream.mjpg', '/live']:
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            except Exception:
                pass

            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()

            sent_id = -1
            while True:
                with jpeg_cond:
                    if latest_jpeg_id == sent_id:
                        jpeg_cond.wait(timeout=0.5)
                    if latest_jpeg_frame is None or latest_jpeg_id == sent_id:
                        continue
                    frame = latest_jpeg_frame
                    sent_id = latest_jpeg_id

                try:
                    header = (
                        f"--frame\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n"
                    ).encode('ascii')
                    self.wfile.write(header + frame + b"\r\n")
                    self.wfile.flush()
                except Exception:
                    break

        elif clean_path in ['/audio.wav', '/stream.wav']:
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            self.send_response(200)
            self.send_header('Content-Type', 'audio/x-wav')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            try:
                self.wfile.write(make_wav_header())
                self.wfile.flush()
            except Exception:
                return

            client_audio_q = queue.Queue(maxsize=12)
            with audio_listeners_lock:
                audio_listeners.append(client_audio_q)

            try:
                while True:
                    pcm16_bytes = client_audio_q.get(timeout=20.0)
                    self.wfile.write(pcm16_bytes)
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                with audio_listeners_lock:
                    if client_audio_q in audio_listeners:
                        audio_listeners.remove(client_audio_q)

        elif clean_path == '/api/state_stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            client_q = queue.Queue(maxsize=30)
            with sse_lock:
                sse_clients.append(client_q)

            with world_state_lock, music_state_lock, video_state_lock:
                combo = {"world": world_state, "music": music_state, "video": video_state}
                init_msg = f"event: state\ndata: {json.dumps(combo)}\n\n".encode('utf-8')
            self.wfile.write(init_msg)
            self.wfile.flush()

            try:
                while True:
                    msg = client_q.get(timeout=20.0)
                    self.wfile.write(msg)
                    self.wfile.flush()
            except Exception:
                with sse_lock:
                    if client_q in sse_clients:
                        sse_clients.remove(client_q)

        elif clean_path == '/api/world/concepts':
            with world_state_lock:
                self._send_json(200, {"concepts": world_state["concepts"]})

        elif clean_path == '/api/music/concepts':
            with music_state_lock:
                self._send_json(200, {"concepts": music_state["concepts"]})

        elif clean_path == '/api/video/state':
            with video_state_lock:
                video_state["available_lores"] = discover_available_lore_files()
                self._send_json(200, video_state)

        elif clean_path == '/api/world/telemetry':
            with world_state_lock:
                self._send_json(200, {
                    "active_thought": world_state["active_thought"],
                    "puck": world_state["current_puck"],
                    "ladder_stage": world_state["ladder_stage"],
                    "monty_loss": world_state["monty_loss"],
                    "persistence": world_state["persistence"],
                    "surprise": world_state["surprise"],
                    "regions": world_state["regions"],
                    "concepts": world_state["concepts"]
                })
        else:
            self.send_error(404)

    def do_POST(self):
        global session_wav_writer
        clean_path = self.path.split('?')[0].rstrip('/')
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length).decode('utf-8')) if length else {}

        if clean_path == '/api/eeg_push':
            dev_id = str(data.get('device_id') or "dev_0").strip()
            lsl_router.push(
                dev_id=dev_id,
                samples=data.get('samples', []),
                sps=float(data.get('sps', DEFAULT_SPS))
            )
            self._send_json(200, {"status": "ok"})

        elif clean_path in ['/api/device/register', '/api/device/configure']:
            dev_id = str(data.get('device_id', 'dev_0')).strip()
            clean_dev = "".join(c for c in dev_id if c.isalnum())[-6:].upper() or "DEV0"
            role = str(data.get('role', 'FCz')).strip()
            user = str(data.get('user', 'User1')).strip()

            mapping_msg = {
                "cmd": "set_device_mapping",
                "device_id": dev_id,
                "stream_name": f"FreeEEG_{clean_dev}",
                "role": role,
                "user": user
            }
            send_diff_ipc(mapping_msg)
            send_video_ipc(mapping_msg)
            send_music_ipc(mapping_msg)
            self._send_json(200, {"status": "ok", "mapped": mapping_msg})

        elif clean_path == '/api/device/command':
            cmd_type = data.get('cmd')
            val = data.get('val')
            dev_id = str(data.get('device_id', 'dev_0')).strip()
            clean_dev = "".join(c for c in dev_id if c.isalnum())[-6:].upper() or "DEV0"
            user = str(data.get('user', 'User1')).strip()

            if cmd_type in ('set_role', 'set_user') and val:
                role = str(val).strip() if cmd_type == 'set_role' else str(data.get('role', 'FCz')).strip()
                usr = str(val).strip() if cmd_type == 'set_user' else user
                mapping_msg = {
                    "cmd": "set_device_mapping",
                    "device_id": dev_id,
                    "stream_name": f"FreeEEG_{clean_dev}",
                    "role": role,
                    "user": usr
                }
                send_diff_ipc(mapping_msg)
                send_video_ipc(mapping_msg)
                send_music_ipc(mapping_msg)

            broadcast_sse("ble_cmd", data)
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/preset/ladder':
            stg = int(data.get('stage', 4))
            with world_state_lock:
                world_state["ladder_stage"] = stg
            send_diff_ipc({"cmd": "set_ladder_stage", "stage": stg})
            send_video_ipc({"cmd": "set_ladder_stage", "stage": stg})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "ladder_stage": stg})

        elif clean_path == '/api/video/config':
            with video_state_lock:
                if "region_gains" in data and isinstance(data["region_gains"], dict):
                    for k, v in data["region_gains"].items():
                        video_state["region_gains"][k] = float(np.clip(float(v), 0.0, 1.0))
                for k in (
                    "auto_active_boost", "active_user_gain", "inactive_base_gain",
                    "cross_region_bleed", "cascade_rate", "psi_threshold",
                    "geodesic_lock", "num_frames", "context_ring_len",
                    "steps", "genesis_steps", "cfg_scale"
                ):
                    if k in data:
                        video_state[k] = data[k]
                cfg_copy = {
                    "region_gains": dict(video_state["region_gains"]),
                    "auto_active_boost": bool(video_state["auto_active_boost"]),
                    "active_user_gain": float(video_state["active_user_gain"]),
                    "inactive_base_gain": float(video_state["inactive_base_gain"]),
                    "cross_region_bleed": float(video_state["cross_region_bleed"]),
                    "cascade_rate": float(video_state["cascade_rate"]),
                    "psi_threshold": float(video_state["psi_threshold"]),
                    "geodesic_lock": float(video_state["geodesic_lock"]),
                    "num_frames": int(video_state["num_frames"]),
                    "context_ring_len": int(video_state["context_ring_len"]),
                    "steps": int(video_state["steps"]),
                    "genesis_steps": int(video_state["genesis_steps"]),
                    "cfg_scale": float(video_state["cfg_scale"]),
                }
            send_video_ipc({"cmd": "update_video_config", "config": cfg_copy})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "video": cfg_copy})

        elif clean_path == '/api/video/switch_lore':
            lore_file = str(data.get("lore_file", "")).strip()
            new_sections = parse_regional_lore_file(lore_file)
            with video_state_lock:
                video_state["lore_file"] = lore_file
                video_state["sections"] = new_sections
                video_state["available_lores"] = discover_available_lore_files()
            send_video_ipc({
                "cmd": "switch_lore",
                "lore_file": lore_file,
                "sections": new_sections,
                "reset_genesis": True
            })
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "lore_file": lore_file, "sections": new_sections})

        elif clean_path == '/api/video/reset_genesis':
            send_video_ipc({"cmd": "reset_genesis"})
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/video/init_image':
            import base64
            b64_str = data.get("image_base64", "")
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_img_bytes = base64.b64decode(b64_str) if b64_str else b""
            if raw_img_bytes:
                send_video_ipc({"cmd": "set_init_image", "image_bytes": raw_img_bytes})
            self._send_json(200, {"status": "ok", "bytes": len(raw_img_bytes)})

        elif clean_path == '/api/video/region_concepts':
            region_key = str(data.get("region", "FCZ")).strip().upper()
            action = str(data.get("action", "add")).strip().lower()
            concept = str(data.get("concept", "")).strip()

            with video_state_lock:
                if region_key not in video_state["sections"]:
                    video_state["sections"][region_key] = []
                cur_list = video_state["sections"][region_key]

                if action == "add" and concept and concept not in cur_list:
                    cur_list.append(concept)
                elif action == "remove" and concept in cur_list:
                    cur_list.remove(concept)
                elif action == "set" and isinstance(data.get("concepts"), list):
                    video_state["sections"][region_key] = [
                        str(x).strip() for x in data["concepts"] if str(x).strip()
                    ]
                sec_copy = {k: list(v) for k, v in video_state["sections"].items()}

            send_video_ipc({"cmd": "update_lore_sections", "sections": sec_copy})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "sections": sec_copy})

        elif clean_path == '/api/music_config':
            with music_state_lock:
                music_state.update(data)
            send_music_ipc({'cmd': 'update_config', 'config': music_state})
            broadcast_full_state()
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/world/concepts/add':
            concept = data.get('concept', '').strip()
            if concept:
                with world_state_lock:
                    if concept not in world_state["concepts"]:
                        world_state["concepts"].append(concept)
                send_diff_ipc({"cmd": "update_concepts", "concepts": world_state["concepts"]})
                broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": world_state["concepts"]})

        elif clean_path == '/api/world/concepts/set':
            new_list = [str(c).strip() for c in data.get('concepts', []) if str(c).strip()]
            with world_state_lock:
                world_state["concepts"] = new_list
            send_diff_ipc({"cmd": "update_concepts", "concepts": world_state["concepts"]})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": world_state["concepts"]})

        elif clean_path == '/api/world/concepts/remove':
            concept = data.get('concept', '').strip()
            with world_state_lock:
                if concept in world_state["concepts"]:
                    world_state["concepts"].remove(concept)
            send_diff_ipc({"cmd": "update_concepts", "concepts": world_state["concepts"]})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": world_state["concepts"]})

        elif clean_path == '/api/music/concepts/add':
            concept = data.get('concept', '').strip()
            if concept:
                with music_state_lock:
                    if concept not in music_state["concepts"]:
                        music_state["concepts"].append(concept)
                send_music_ipc({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
                broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/music/concepts/set':
            new_list = [str(c).strip() for c in data.get('concepts', []) if str(c).strip()]
            with music_state_lock:
                music_state["concepts"] = new_list
            send_music_ipc({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/music/concepts/remove':
            concept = data.get('concept', '').strip()
            with music_state_lock:
                if concept in music_state["concepts"]:
                    music_state["concepts"].remove(concept)
            send_music_ipc({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/diffusion/config':
            with world_state_lock:
                world_state.update(data)
            send_diff_ipc({"cmd": "update_params", "params": data})
            broadcast_full_state()
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/recording/start':
            req_name = data.get("name")
            actual_session_dir = recorder_instance.start_session(session_name=req_name)
            with session_wav_lock:
                if session_wav_writer is not None:
                    try:
                        session_wav_writer.close()
                    except Exception:
                        pass
                wav_path = os.path.join(actual_session_dir, "music_timeline.wav")
                session_wav_writer = wave.open(wav_path, 'wb')
                session_wav_writer.setnchannels(1)
                session_wav_writer.setsampwidth(2)
                session_wav_writer.setframerate(SAMPLE_RATE)

            with world_state_lock:
                world_state["is_recording"] = True
                world_state["active_session_path"] = actual_session_dir
            send_diff_ipc({"cmd": "start_recording", "session_name": req_name})
            send_video_ipc({"cmd": "start_recording", "session_name": req_name})
            broadcast_full_state()
            self._send_json(200, {"status": "recording_started", "session_path": actual_session_dir})

        elif clean_path == '/api/recording/stop':
            with session_wav_lock:
                if session_wav_writer is not None:
                    try:
                        session_wav_writer.close()
                    except Exception:
                        pass
                    session_wav_writer = None

            saved_dir = recorder_instance.stop_session()
            with world_state_lock:
                world_state["is_recording"] = False
                saved_p = saved_dir or world_state.get("active_session_path", "sessions/latest")
            send_diff_ipc({"cmd": "stop_recording"})
            send_video_ipc({"cmd": "stop_recording"})
            broadcast_full_state()
            self._send_json(200, {"status": "recording_stopped", "saved_path": saved_p})
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def broadcast_full_state():
    with world_state_lock, music_state_lock, video_state_lock:
        st = {
            "world": world_state.copy(),
            "music": music_state.copy(),
            "video": video_state.copy()
        }
    broadcast_sse("state", st)


def send_diff_ipc(payload):
    global active_diff_ctrl_conn
    with diff_ctrl_lock:
        if active_diff_ctrl_conn is not None:
            try:
                active_diff_ctrl_conn.send(payload)
            except Exception:
                active_diff_ctrl_conn = None


def send_video_ipc(payload):
    global active_video_ctrl_conn
    with video_ctrl_lock:
        if active_video_ctrl_conn is not None:
            try:
                active_video_ctrl_conn.send(payload)
            except Exception:
                active_video_ctrl_conn = None


def send_music_ipc(payload):
    global active_music_conn
    with music_conn_lock:
        if active_music_conn is not None:
            try:
                active_music_conn.send(payload)
            except Exception:
                active_music_conn = None


class ThreadedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    allow_reuse_address = True


def ipc_frame_receiver():
    global latest_jpeg_frame, latest_jpeg_id, latest_jpeg_ts
    try:
        listener = Listener(('localhost', IPC_PORT), authkey=IPC_AUTHKEY)
    except Exception:
        return
    while True:
        try:
            conn = listener.accept()
            while True:
                data = conn.recv()
                dropped_ipc = 0
                while conn.poll(0.0):
                    data = conn.recv()
                    dropped_ipc += 1

                now = time.time()
                if isinstance(data, bytes):
                    with jpeg_cond:
                        latest_jpeg_frame = data
                        latest_jpeg_id += 1
                        latest_jpeg_ts = now
                        jpeg_cond.notify_all()
                    with world_state_lock:
                        world_state["buffer_frames"] = dropped_ipc
                        world_state["buffer_sec"] = 0.0
                elif isinstance(data, dict) and 'jpg' in data:
                    with jpeg_cond:
                        latest_jpeg_frame = data['jpg']
                        latest_jpeg_id = int(data.get('frame_id', latest_jpeg_id + 1))
                        latest_jpeg_ts = float(data.get('ts', now))
                        age_sec = max(0.0, now - latest_jpeg_ts)
                        jpeg_cond.notify_all()
                    with world_state_lock:
                        world_state["fps"] = round(float(data.get('fps', 0.0)), 1)
                        world_state["gen_ms"] = round(float(data.get('gen_ms', 0.0)), 1)
                        world_state["actual_steps"] = int(data.get('actual_steps', 2))
                        world_state["resolution"] = f"{data.get('w', 512)}x{data.get('h', 384)}"
                        world_state["use_taesd"] = bool(data.get('use_taesd', True))
                        world_state["use_color"] = bool(data.get('use_color', True))
                        world_state["buffer_frames"] = dropped_ipc
                        world_state["buffer_sec"] = round(age_sec, 2)
        except Exception:
            time.sleep(0.5)


def ipc_control_listener():
    global active_diff_ctrl_conn
    last_sse_push = 0.0
    try:
        listener = Listener(('localhost', IPC_DIFF_CONTROL_PORT), authkey=IPC_DIFF_AUTHKEY)
    except Exception:
        return
    while True:
        try:
            conn = listener.accept()
            with diff_ctrl_lock:
                active_diff_ctrl_conn = conn
                with world_state_lock:
                    if world_state["concepts"]:
                        active_diff_ctrl_conn.send({
                            "cmd": "update_concepts",
                            "concepts": list(world_state["concepts"]),
                            "is_initial_sync": True
                        })
            while True:
                msg = conn.recv()
                if isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "sync_from_runner":
                        runner_c = msg.get("concepts", [])
                        with world_state_lock:
                            if runner_c and not world_state["concepts"]:
                                world_state["concepts"] = list(runner_c)
                        broadcast_full_state()
                    elif cmd == "telemetry":
                        with world_state_lock:
                            world_state["current_puck"] = msg.get("puck", [0.0, 0.0])
                            world_state["active_thought"] = msg.get("active_thought", "")
                            world_state["ladder_stage"] = msg.get("ladder_stage", 4)
                            world_state["monty_loss"] = msg.get("monty_loss", 0.0)
                            world_state["persistence"] = msg.get("persistence", 0.0)
                            world_state["surprise"] = msg.get("surprise", 0.0)
                            world_state["regions"] = msg.get("regions", {})
                            if "stochastic" in msg: world_state["stochastic"] = msg["stochastic"]
                            if "unassigned_stochastic" in msg: world_state["unassigned_stochastic"] = msg["unassigned_stochastic"]
                            if "fps" in msg: world_state["fps"] = msg["fps"]
                            if "gen_ms" in msg: world_state["gen_ms"] = msg["gen_ms"]
                            if "actual_steps" in msg: world_state["actual_steps"] = msg["actual_steps"]
                            if "resolution" in msg: world_state["resolution"] = msg["resolution"]
                            if "use_taesd" in msg: world_state["use_taesd"] = msg["use_taesd"]
                            if "use_color" in msg: world_state["use_color"] = msg["use_color"]
                        now = time.time()
                        if now - last_sse_push > 0.20:
                            last_sse_push = now
                            broadcast_full_state()
        except Exception:
            time.sleep(0.5)
        finally:
            with diff_ctrl_lock:
                active_diff_ctrl_conn = None


def ipc_video_control_listener():
    global active_video_ctrl_conn
    last_sse_push = 0.0
    try:
        listener = Listener(('localhost', IPC_VIDEO_CONTROL_PORT), authkey=IPC_VIDEO_AUTHKEY)
    except Exception:
        return
    while True:
        try:
            conn = listener.accept()
            with video_ctrl_lock:
                active_video_ctrl_conn = conn
            with video_state_lock:
                video_state["connected"] = True
            broadcast_full_state()

            while True:
                msg = conn.recv()
                if isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "sync_from_video_runner":
                        with video_state_lock:
                            video_state["connected"] = True
                            if "lore_file" in msg:
                                video_state["lore_file"] = msg["lore_file"]
                            if "sections" in msg and isinstance(msg["sections"], dict):
                                video_state["sections"] = msg["sections"]
                            video_state["available_lores"] = discover_available_lore_files()
                        broadcast_full_state()
                    elif cmd == "video_telemetry":
                        with video_state_lock:
                            video_state["connected"] = True
                            video_state["fps"] = msg.get("fps", 0.0)
                            video_state["gen_ms"] = msg.get("gen_ms", 0.0)
                            video_state["debug_tokens"] = msg.get("debug_tokens", "")
                            video_state["fpz_meta_charge"] = msg.get("fpz_meta_charge", 0.0)
                            video_state["afz_stage_progress"] = msg.get("afz_stage_progress", 0.0)
                            if "active_region_mask" in msg:
                                video_state["active_region_mask"] = msg["active_region_mask"]
                            if "region_gains" in msg:
                                video_state["region_gains"] = msg["region_gains"]
                            if "active_concept_idx" in msg:
                                video_state["active_concept_idx"] = msg["active_concept_idx"]
                        with world_state_lock:
                            world_state["active_thought"] = msg.get("debug_tokens", "")
                            world_state["monty_loss"] = msg.get("monty_loss", 0.0)
                            world_state["persistence"] = msg.get("persistence", 0.0)
                            world_state["surprise"] = msg.get("surprise", 0.0)

                        now = time.time()
                        if now - last_sse_push > 0.20:
                            last_sse_push = now
                            broadcast_full_state()
        except Exception:
            time.sleep(0.5)
        finally:
            with video_ctrl_lock:
                active_video_ctrl_conn = None
            with video_state_lock:
                video_state["connected"] = False


def ipc_audio_receiver():
    global active_music_conn, session_wav_writer
    try:
        listener = Listener(('localhost', IPC_AUDIO_PORT), authkey=IPC_AUDIO_AUTHKEY)
    except Exception:
        return
    while True:
        try:
            conn = listener.accept()
            with music_conn_lock:
                active_music_conn = conn
                with music_state_lock:
                    if music_state["concepts"]:
                        active_music_conn.send({
                            'cmd': 'update_concepts',
                            'concepts': list(music_state["concepts"]),
                            'is_initial_sync': True
                        })
            while True:
                data = conn.recv()
                raw_bytes = None
                if isinstance(data, bytes):
                    raw_bytes = data
                elif isinstance(data, dict):
                    cmd = data.get('cmd')
                    if cmd == 'audio_chunk':
                        raw_bytes = data.get('pcm')
                        with music_state_lock:
                            music_state["rtf"] = round(float(data.get('rtf', 1.0)), 2)
                            music_state["play_speed"] = round(float(data.get('play_speed', 1.0)), 2)
                            music_state["actual_block"] = round(float(data.get('actual_block', 0.24)), 3)
                            if 'dyn_strength' in data:
                                music_state["dyn_strength"] = round(float(data['dyn_strength']), 2)
                    elif cmd == 'sync_from_runner':
                        runner_concepts = data.get('concepts', [])
                        with music_state_lock:
                            if runner_concepts and not music_state["concepts"]:
                                music_state["concepts"] = list(runner_concepts)
                            music_state["rtf"] = round(float(data.get('rtf', 1.0)), 2)
                        broadcast_full_state()

                if raw_bytes is not None:
                    flt = np.frombuffer(raw_bytes, dtype=np.float32)
                    pcm16_bytes = (np.clip(flt, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    chunk_sec = len(flt) / float(SAMPLE_RATE)

                    with session_wav_lock:
                        if session_wav_writer is not None:
                            try:
                                session_wav_writer.writeframes(pcm16_bytes)
                            except Exception:
                                pass

                    max_q_chunks = 0
                    with audio_listeners_lock:
                        for q in list(audio_listeners):
                            try:
                                q.put_nowait(pcm16_bytes)
                            except queue.Full:
                                try:
                                    q.get_nowait()
                                    q.put_nowait(pcm16_bytes)
                                except Exception:
                                    pass
                            if q.qsize() > max_q_chunks:
                                max_q_chunks = q.qsize()

                    est_buf_sec = round(max_q_chunks * chunk_sec, 2)
                    with music_state_lock:
                        music_state["buffer_chunks"] = max_q_chunks
                        music_state["buffer_sec"] = est_buf_sec

                    send_music_ipc({
                        'cmd': 'gateway_buffer_stat',
                        'buffer_sec': est_buf_sec,
                        'buffer_chunks': max_q_chunks
                    })
                    broadcast_full_state()
        except Exception:
            time.sleep(0.5)
        finally:
            with music_conn_lock:
                active_music_conn = None


def load_initial_lore(lore_path):
    if lore_path and os.path.exists(lore_path):
        with open(lore_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() for line in f
                if line.strip() and not line.startswith('#') and not (line.strip().startswith('[') and line.strip().endswith(']'))
            ]
    return []


def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Production Web & LSL Gateway")
    parser.add_argument('pos_port', nargs='?', type=int, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--cert', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--lore', type=str, default=None)
    parser.add_argument('--video-lore', type=str, default=None)
    parser.add_argument('--music-lore', type=str, default=None)
    args = parser.parse_args()

    port = args.port or args.pos_port or int(os.environ.get("HTTP_PORT", HTTP_PORT))

    raw_diff_lore  = args.lore if args.lore is not None else os.environ.get("LORE_PATH", "")
    raw_video_lore = args.video_lore if args.video_lore is not None else os.environ.get("VIDEO_LORE_PATH", "")
    raw_music_lore = args.music_lore if args.music_lore is not None else os.environ.get("MUSIC_LORE_PATH", "")

    diff_lore_rel  = normalize_lore_relpath(raw_diff_lore)
    video_lore_rel = normalize_lore_relpath(raw_video_lore)

    initial_world_words = load_initial_lore(os.path.join(BASE_DIR, diff_lore_rel)) if diff_lore_rel else []
    initial_music_words = load_initial_lore(raw_music_lore)

    with world_state_lock:
        world_state["concepts"] = initial_world_words
    with music_state_lock:
        music_state["concepts"] = initial_music_words
    with video_state_lock:
        video_state["lore_file"] = video_lore_rel
        video_state["sections"] = parse_regional_lore_file(video_lore_rel)
        video_state["available_lores"] = discover_available_lore_files()

    threading.Thread(target=ipc_frame_receiver, daemon=True).start()
    threading.Thread(target=ipc_audio_receiver, daemon=True).start()
    threading.Thread(target=ipc_control_listener, daemon=True).start()
    threading.Thread(target=ipc_video_control_listener, daemon=True).start()

    server = ThreadedHTTPServer(('0.0.0.0', port), UnifiedCloudHandler)

    scheme = "http"
    if args.cert and args.key and os.path.exists(args.cert) and os.path.exists(args.key):
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            scheme = "https"
        except Exception:
            pass

    print(f"🚀 [GATEWAY] Веб-терминал: {scheme}://0.0.0.0:{port}/ (Video IPC:6007 | Diff IPC:6004 | Music IPC:6003)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

