#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY v9.0 (SERVER-SIDE ENGINE BUFFER CLOCK)
===================================================================================
- Буфер музыки принадлежит СТРОГО серверному движку (расходуется на сервере
  с точной частотой 32 000 Гц независимо от того, включен ли звук в браузере).
- Браузер подключается к /audio.wav как к готовому радиоэфиру и ничего сам не считает.
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

NUM_CHANNELS = 16
DEFAULT_SPS = 250.0
SAMPLE_RATE = 32000
TICK_SEC = 0.040  # 40 мс серверный квант вещания (1280 сэмплов = 2560 байт int16)
TICK_SAMPLES = int(SAMPLE_RATE * TICK_SEC)
TICK_FLOAT_BYTES = TICK_SAMPLES * 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

latest_jpeg_frame = None
latest_jpeg_id = 0
latest_jpeg_ts = 0.0
last_served_jpeg_id = 0
jpeg_lock = threading.Lock()
jpeg_cond = threading.Condition(jpeg_lock)

# Внутренний буфер аудио-движка на сервере: список байтовых чанков float32
engine_audio_chunks: list[bytearray] = []
engine_audio_lock = threading.Lock()

# Подписчики живого аудио-потока (браузеры с включенным звуком)
audio_listeners: list[queue.Queue] = []
audio_listeners_lock = threading.Lock()

session_wav_writer = None
session_wav_lock = threading.Lock()

active_music_conn = None
music_conn_lock = threading.Lock()

active_diff_ctrl_conn = None
diff_ctrl_lock = threading.Lock()

music_state = {
    "concepts": [],
    "preset": "balanced",
    "target_block": 0.0,
    "actual_block": 0.16,
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
    "rtf": 1.0
}
music_state_lock = threading.Lock()

world_state = {
    "concepts": [],
    "preset": "balanced",
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


def _recalc_and_push_buffer_locked():
    """Пересчитывает реальный остаток серверного буфера движка и сообщает в MusicGen."""
    total_bytes = sum(len(c) for c in engine_audio_chunks)
    total_sec = total_bytes / float(SAMPLE_RATE * 4)
    total_chunks = len(engine_audio_chunks)

    with music_state_lock:
        music_state["buffer_sec"] = round(total_sec, 2)
        music_state["buffer_chunks"] = total_chunks

    with music_conn_lock:
        if active_music_conn is not None:
            try:
                active_music_conn.send({
                    'cmd': 'gateway_buffer_stat',
                    'buffer_sec': total_sec,
                    'buffer_chunks': total_chunks
                })
            except Exception:
                pass


def server_realtime_audio_clock():
    """
    Аппаратный серверный метроном 32 000 Гц.
    Непрерывно расходует буфер движка со скоростью 1:1 реального времени на сервере,
    даже если в браузере выключен звук. Браузер получает уже готовый поток!
    """
    next_tick = time.perf_counter()
    last_sse_broadcast = 0.0

    while True:
        now = time.perf_counter()
        sleep_dt = next_tick - now
        if sleep_dt > 0.001:
            time.sleep(sleep_dt)
        next_tick = max(time.perf_counter(), next_tick + TICK_SEC)

        pulled_bytes = bytearray()
        with engine_audio_lock:
            needed = TICK_FLOAT_BYTES
            while needed > 0 and len(engine_audio_chunks) > 0:
                head = engine_audio_chunks[0]
                if len(head) <= needed:
                    pulled_bytes.extend(head)
                    needed -= len(head)
                    engine_audio_chunks.pop(0)
                else:
                    pulled_bytes.extend(head[:needed])
                    del head[:needed]
                    needed = 0
            _recalc_and_push_buffer_locked()

        if len(pulled_bytes) > 0:
            floats = np.frombuffer(pulled_bytes, dtype=np.float32)
            pcm16_bytes = (np.clip(floats, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

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

        if time.perf_counter() - last_sse_broadcast >= 0.25:
            last_sse_broadcast = time.perf_counter()
            broadcast_full_state()


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
        self.outlets = {}
        self.lock = threading.Lock()

    def get_or_create_outlet(self, user: str, role: str, sps: float = DEFAULT_SPS):
        clean_user = "".join(c for c in user if c.isalnum() or c == "_").capitalize() or "User1"
        clean_role = "".join(c for c in role if c.isalnum()).capitalize() or "Afz"
        role_map = {"Afz": "AFz", "Fpz": "Fpz", "Fcz": "FCz", "F3": "F3", "F4": "F4", "Pz": "Pz", "Cz": "Cz"}
        clean_role = role_map.get(clean_role, clean_role)
        stream_name = f"{clean_user}_{clean_role}"

        with self.lock:
            if stream_name not in self.outlets:
                info = StreamInfo(
                    name=stream_name, type='EEG', channel_count=NUM_CHANNELS,
                    nominal_srate=float(sps), channel_format='float32',
                    source_id=f"ble_{stream_name}_{int(time.time())}"
                )
                self.outlets[stream_name] = StreamOutlet(info)
            return self.outlets[stream_name]

    def push(self, user, role, samples, sps=DEFAULT_SPS):
        outlet = self.get_or_create_outlet(user, role, sps)
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

            client_audio_q = queue.Queue(maxsize=25)
            with audio_listeners_lock:
                audio_listeners.append(client_audio_q)

            try:
                while True:
                    pcm16_bytes = client_audio_q.get(timeout=15.0)
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

            with world_state_lock, music_state_lock:
                combo = {"world": world_state, "music": music_state}
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

        elif clean_path == '/api/world/telemetry':
            with world_state_lock:
                self._send_json(200, {
                    "active_thought": world_state["active_thought"],
                    "puck": world_state["current_puck"],
                    "ladder_stage": world_state["ladder_stage"],
                    "monty_loss": world_state["monty_loss"],
                    "persistence": world_state["persistence"],
                    "surprise": world_state["surprise"],
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
            lsl_router.push(
                data.get('user', 'User1'),
                data.get('role', 'AFz'),
                data.get('samples', []),
                float(data.get('sps', DEFAULT_SPS))
            )
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/music_config':
            with music_state_lock:
                music_state.update(data)
            with music_conn_lock:
                if active_music_conn is not None:
                    try:
                        active_music_conn.send({'cmd': 'update_config', 'config': music_state})
                    except Exception:
                        pass
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
                with music_conn_lock:
                    if active_music_conn is not None:
                        try:
                            active_music_conn.send({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
                        except Exception:
                            pass
                broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/music/concepts/set':
            new_list = [str(c).strip() for c in data.get('concepts', []) if str(c).strip()]
            with music_state_lock:
                music_state["concepts"] = new_list
            with music_conn_lock:
                if active_music_conn is not None:
                    try:
                        active_music_conn.send({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
                    except Exception:
                            pass
                broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/music/concepts/remove':
            concept = data.get('concept', '').strip()
            with music_state_lock:
                if concept in music_state["concepts"]:
                    music_state["concepts"].remove(concept)
            with music_conn_lock:
                if active_music_conn is not None:
                    try:
                        active_music_conn.send({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
                    except Exception:
                        pass
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
            broadcast_full_state()
            self._send_json(200, {"status": "recording_stopped", "saved_path": saved_p})
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def broadcast_full_state():
    with world_state_lock, music_state_lock:
        st = {"world": world_state.copy(), "music": music_state.copy()}
    broadcast_sse("state", st)


def send_diff_ipc(payload):
    global active_diff_ctrl_conn
    with diff_ctrl_lock:
        if active_diff_ctrl_conn is not None:
            try:
                active_diff_ctrl_conn.send(payload)
            except Exception:
                active_diff_ctrl_conn = None


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
                    active_diff_ctrl_conn.send({"cmd": "update_concepts", "concepts": list(world_state["concepts"])})
            while True:
                msg = conn.recv()
                if isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "telemetry":
                        with world_state_lock:
                            world_state["current_puck"] = msg.get("puck", [0.0, 0.0])
                            world_state["active_thought"] = msg.get("active_thought", "")
                            world_state["ladder_stage"] = msg.get("ladder_stage", 4)
                            world_state["monty_loss"] = msg.get("monty_loss", 0.0)
                            world_state["persistence"] = msg.get("persistence", 0.0)
                            world_state["surprise"] = msg.get("surprise", 0.0)
                            world_state["regions"] = msg.get("regions", {})
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
                            music_state["actual_block"] = round(float(data.get('actual_block', 0.16)), 3)
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
                    with session_wav_lock:
                        if session_wav_writer is not None:
                            try:
                                flt = np.frombuffer(raw_bytes, dtype=np.float32)
                                pcm16 = (np.clip(flt, -1.0, 1.0) * 32767).astype(np.int16)
                                session_wav_writer.writeframes(pcm16.tobytes())
                            except Exception:
                                pass

                    with engine_audio_lock:
                        max_blocks = max(1, int(music_state.get("target_buffer_count", 2)))
                        engine_audio_chunks.append(bytearray(raw_bytes))
                        while len(engine_audio_chunks) > max_blocks + 1:
                            engine_audio_chunks.pop(0)
                        _recalc_and_push_buffer_locked()

                    broadcast_full_state()
        except Exception:
            time.sleep(0.5)
        finally:
            with music_conn_lock:
                active_music_conn = None


def load_initial_lore(lore_path):
    if lore_path and os.path.exists(lore_path):
        with open(lore_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Production Web & LSL Gateway")
    parser.add_argument('pos_port', nargs='?', type=int, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--cert', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--lore', type=str, default=None)
    parser.add_argument('--music-lore', type=str, default=None)
    args = parser.parse_args()

    port = args.port or args.pos_port or int(os.environ.get("HTTP_PORT", HTTP_PORT))

    initial_world_words = load_initial_lore(args.lore or os.environ.get("LORE_PATH"))
    initial_music_words = load_initial_lore(args.music_lore or os.environ.get("MUSIC_LORE_PATH"))

    with world_state_lock:
        world_state["concepts"] = initial_world_words
    with music_state_lock:
        music_state["concepts"] = initial_music_words

    threading.Thread(target=ipc_frame_receiver, daemon=True).start()
    threading.Thread(target=ipc_audio_receiver, daemon=True).start()
    threading.Thread(target=ipc_control_listener, daemon=True).start()
    threading.Thread(target=server_realtime_audio_clock, daemon=True).start()

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

    print(f"🚀 [GATEWAY] Веб-терминал: {scheme}://0.0.0.0:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
