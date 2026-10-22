#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY v8.6 (TRUE BUFFER BLOCKS & AUTO-MIN)
===================================================================================
"""

from __future__ import annotations
import os
import ssl
import wave
import json
import time
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

latest_jpeg_frame = None
jpeg_lock = threading.Lock()

audio_pcm_queue = []
browser_playout_horizon = 0.0
audio_pcm_lock = threading.Lock()

session_wav_writer = None
session_wav_lock = threading.Lock()

active_music_conn = None
music_conn_lock = threading.Lock()

active_diff_ctrl_conn = None
diff_ctrl_lock = threading.Lock()

music_state = {
    "concepts": [],
    "preset": "balanced",
    "target_block": 0.0,         # 0.0 = Авто-минимизация блока по умолчанию!
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
    "steps": 2,
    "guidance_scale": 1.0,
    "strength_low": 0.50,
    "strength_high": 0.85,
    "ema_smooth": 0.14,
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


def _recalc_and_push_buffer_locked(now: float):
    """Пересчитывает реальный остаток блоков и секунд (в очереди + в полете у браузера)."""
    ahead_sec = max(0.0, browser_playout_horizon - now)
    queue_sec = sum(len(c) for c in audio_pcm_queue) / (SAMPLE_RATE * 4.0)
    total_sec = queue_sec + ahead_sec
    in_flight = 1 if ahead_sec > 0.015 else 0
    total_chunks = len(audio_pcm_queue) + in_flight
    with music_state_lock:
        music_state["buffer_sec"] = round(total_sec, 3)
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
        global browser_playout_horizon
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

        elif clean_path in ['/stream.mjpg', '/live']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.end_headers()
            while True:
                with jpeg_lock:
                    frame = latest_jpeg_frame
                if frame is not None:
                    try:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode() + frame + b'\r\n')
                        self.wfile.flush()
                    except Exception:
                        break
                time.sleep(0.016)

        elif clean_path in ['/audio.wav', '/stream.wav']:
            self.send_response(200)
            self.send_header('Content-Type', 'audio/x-wav')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.end_headers()
            try:
                self.wfile.write(make_wav_header())
                self.wfile.flush()
            except Exception:
                return

            with audio_pcm_lock:
                browser_playout_horizon = time.time()

            # Отдаем чанки в браузер ровно по мере их реального проигрывания (не более 0.15с в полете),
            # чтобы в браузере НИКОГДА не накапливалось 88 секунд в одном блоке!
            while True:
                chunk = None
                now = time.time()
                with audio_pcm_lock:
                    ahead = browser_playout_horizon - now
                    if ahead <= 0.15 and len(audio_pcm_queue) > 0:
                        chunk = audio_pcm_queue.pop(0)
                        c_dur = len(chunk) / (SAMPLE_RATE * 4.0)
                        browser_playout_horizon = max(now, browser_playout_horizon) + c_dur
                        _recalc_and_push_buffer_locked(now)

                if chunk is not None:
                    try:
                        floats = np.frombuffer(chunk, dtype=np.float32)
                        int16s = (np.clip(floats, -1.0, 1.0) * 32767).astype(np.int16)
                        self.wfile.write(int16s.tobytes())
                        self.wfile.flush()
                        broadcast_full_state()
                    except Exception:
                        break
                else:
                    time.sleep(0.008)

        elif clean_path == '/api/state_stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            import queue
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
    global latest_jpeg_frame
    try:
        listener = Listener(('localhost', IPC_PORT), authkey=IPC_AUTHKEY)
    except Exception:
        return
    while True:
        try:
            conn = listener.accept()
            while True:
                data = conn.recv()
                if isinstance(data, bytes):
                    with jpeg_lock:
                        latest_jpeg_frame = data
        except Exception:
            time.sleep(0.5)


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

                    with audio_pcm_lock:
                        now = time.time()
                        max_blocks = max(1, int(music_state.get("target_buffer_count", 2)))
                        audio_pcm_queue.append(raw_bytes)
                        # Строгий лимит по числу блоков буфера (плюс 1 блок запаса),
                        # чтобы очередь НИКОГДА не копила 60 чанков (88 секунд)!
                        while len(audio_pcm_queue) > max_blocks + 1:
                            audio_pcm_queue.pop(0)
                        _recalc_and_push_buffer_locked(now)

                    broadcast_full_state()
        except Exception:
            time.sleep(0.5)
        finally:
            with music_conn_lock:
                active_music_conn = None


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
                        now = time.time()
                        if now - last_sse_push > 0.25:
                            last_sse_push = now
                            broadcast_full_state()
        except Exception:
            time.sleep(0.5)
        finally:
            with diff_ctrl_lock:
                active_diff_ctrl_conn = None


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
