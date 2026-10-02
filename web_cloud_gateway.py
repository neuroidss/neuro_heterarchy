#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY (PORT 8080 & IPC 6002/6003)
===================================================================================
- HTTP/1.1 с keep-alive (иначе Chrome упирается в ERR_INSUFFICIENT_RESOURCES).
- Web Bluetooth с throttling'ом fetch-запросов (один в полёте на девайс).
- MJPEG и WAV корректно пишут заголовки напрямую в wfile.
- Опциональный HTTPS.
===================================================================================
"""

import sys
import os
import ssl
import json
import time
import struct
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from pylsl import StreamInfo, StreamOutlet
import numpy as np

HTTP_PORT = 8080
IPC_PORT = 6002
IPC_AUTHKEY = b'canvas'
IPC_AUDIO_PORT = 6003
IPC_AUDIO_AUTHKEY = b'audio'
NUM_CHANNELS = 16
DEFAULT_SPS = 250.0
SAMPLE_RATE = 32000

CERT_PATH = "/app/cert.pem"
KEY_PATH = "/app/key.pem"

latest_jpeg_frame = None
jpeg_lock = threading.Lock()

audio_pcm_queue = []
audio_pcm_lock = threading.Lock()
AUDIO_QUEUE_MAX = 100

active_music_conn = None
music_conn_lock = threading.Lock()

current_music_config = {
    "target_block": 2.4,
    "target_context": 0.4,
    "target_buffer_count": 2,
    "temp": 1.0,
    "top_k": 250,
    "cfg_coef": 1.0,
}


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
                print(f"📡 [LSL ROUTER] Живой поток зарегистрирован: '{stream_name}' ({sps:.0f} Hz)")
            return self.outlets[stream_name]

    def push(self, user, role, samples, sps=DEFAULT_SPS):
        outlet = self.get_or_create_outlet(user, role, sps)
        for s in samples:
            if len(s) == NUM_CHANNELS:
                outlet.push_sample(s)


lsl_router = DynamicLSLRouter()


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>NeuroCanvas Cloud Terminal</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #080c12; color: #c8d6e5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; margin: 0; padding: 24px; }
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px solid #1f2d3d; padding-bottom: 12px; }
        h1 { font-size: 20px; margin: 0; color: #00ffc8; }
        .container { display: flex; gap: 24px; flex-wrap: wrap; }
        .canvas-box { border: 2px solid #223145; border-radius: 8px; overflow: hidden; background: #000; min-width: 512px; min-height: 384px; display: flex; align-items: center; justify-content: center; }
        .controls { background: #0f1622; padding: 20px; border-radius: 8px; border: 1px solid #223145; width: 440px; }
        h3 { color: #54a0ff; margin-top: 0; font-size: 15px; border-bottom: 1px solid #223145; padding-bottom: 8px; }
        .btn-ble { width: 100%; padding: 14px; background: #10ac84; cursor: pointer; font-weight: bold; border: none; border-radius: 6px; color: #fff; font-size: 14px; transition: 0.2s; margin-bottom: 16px; }
        .btn-ble:hover { background: #1dd1a1; }
        .btn-audio { width: 100%; padding: 14px; background: #ee5253; cursor: pointer; font-weight: bold; border: none; border-radius: 6px; color: #fff; font-size: 14px; transition: 0.2s; margin-bottom: 16px; }
        .btn-audio.playing { background: #10ac84; }
        .device-card { background: #182232; border: 1px solid #34495e; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
        .device-card h4 { margin: 0 0 10px 0; color: #00ffc8; font-size: 14px; display: flex; justify-content: space-between; }
        label { display: block; margin: 8px 0 4px; color: #8395a7; font-size: 12px; }
        select, input { width: 100%; padding: 8px; background: #0f1622; border: 1px solid #2c3e50; color: #fff; border-radius: 4px; font-family: monospace; font-size: 13px; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #10ac84; color: #fff; }
        .rate { font-size: 11px; color: #1dd1a1; margin-top: 6px; }
        .status { font-size: 11px; color: #8395a7; margin-top: 6px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 NeuroCanvas Cloud Terminal</h1>
        <span class="badge">PORT 8080 ACTIVE</span>
    </div>
    <div class="container">
        <div class="canvas-box">
            <img src="/stream.mjpg" width="768" height="576" alt="Ожидание кадров...">
        </div>
        <div class="controls">
            <h3>🔊 MusicGen Live Audio</h3>
            <button id="btnAudio" class="btn-audio" onclick="toggleAudio()">▶️ ВКЛЮЧИТЬ ЗВУК</button>
            <div id="audioStatus" class="status">Статус: выключен</div>
            <audio id="neuroAudio" preload="none"></audio>

            <h3 style="margin-top:20px;">📡 Bluetooth Подключение (FreeEEG16)</h3>
            <button class="btn-ble" onclick="connectNewBleDevice()">+ ПОДКЛЮЧИТЬ БЛЕ УСТРОЙСТВО</button>

            <div id="deviceList"></div>
            <div id="noDevicesMsg" style="color: #576574; font-size: 12px; text-align: center; margin-top: 20px;">
                Нет подключенных Bluetooth устройств.<br>Нажмите кнопку выше для прямого поиска плат FreeEEG16.
            </div>
        </div>
    </div>

    <script>
        const SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
        const DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
        const CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8";
        const UV_SCALE       = (1.2 / 4.0 / 8388607.0) * 1e6;
        const BATCH_SIZE     = 40;      // ~160 мс при 250 Гц = 6 POST/сек на девайс

        let activeDevices = [];
        let isAudioPlaying = false;

        function toggleAudio() {
            const audio = document.getElementById('neuroAudio');
            const btn   = document.getElementById('btnAudio');
            const st    = document.getElementById('audioStatus');

            if (!isAudioPlaying) {
                try { audio.pause(); } catch(e) {}
                audio.removeAttribute('src');
                try { audio.load(); } catch(e) {}

                audio.src = '/audio.wav?t=' + Date.now();
                const p = audio.play();
                if (p !== undefined) {
                    p.then(() => {
                        isAudioPlaying = true;
                        btn.innerText = '⏸️ ВЫКЛЮЧИТЬ ЗВУК';
                        btn.classList.add('playing');
                        st.innerText = 'Статус: 🟢 играет (32 kHz live)';
                        st.style.color = '#00ffc8';
                    }).catch(e => {
                        st.innerText = 'Ошибка: ' + e.message;
                        st.style.color = '#ff6b6b';
                    });
                }
            } else {
                try { audio.pause(); } catch(e) {}
                audio.removeAttribute('src');
                try { audio.load(); } catch(e) {}
                isAudioPlaying = false;
                btn.innerText = '▶️ ВКЛЮЧИТЬ ЗВУК';
                btn.classList.remove('playing');
                st.innerText = 'Статус: выключен';
                st.style.color = '#8395a7';
            }
        }

        async function connectNewBleDevice() {
            try {
                const device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [SERVICE_UUID] }]
                });
                const server  = await device.gatt.connect();
                const service = await server.getPrimaryService(SERVICE_UUID);
                const dataChar = await service.getCharacteristic(DATA_CHAR_UUID);
                const cmdChar  = await service.getCharacteristic(CMD_CHAR_UUID);

                await cmdChar.writeValue(new Uint8Array([0x04, 0x44, 0x44])).catch(()=>{});
                await cmdChar.writeValue(new Uint8Array([0x05, 0x44, 0x44])).catch(()=>{});

                const devObj = {
                    id: Date.now(),
                    name: device.name || "FreeEEG16",
                    device: device,
                    user: "User" + (activeDevices.length + 1),
                    role: activeDevices.length === 0 ? "AFz" : "Fpz",
                    packetCount: 0,
                    sampleBuffer: [],
                    sending: false,           // ← флаг занятости fetch'а
                    droppedBatches: 0
                };
                activeDevices.push(devObj);
                renderDeviceCards();

                await dataChar.startNotifications();
                dataChar.addEventListener('characteristicvaluechanged', (e) => {
                    const b = new Uint8Array(e.target.value.buffer);
                    if (b[0] !== 0xA0) return;
                    const sample = [];
                    for (let i = 0; i < 16; i++) {
                        let v = (b[2 + i * 3] << 16) | (b[3 + i * 3] << 8) | b[4 + i * 3];
                        if (v & 0x800000) v -= 0x1000000;
                        sample.push(v * UV_SCALE);
                    }
                    devObj.sampleBuffer.push(sample);
                    devObj.packetCount++;

                    // Защита от переполнения буфера, если сервер долго молчит
                    if (devObj.sampleBuffer.length > 400) {
                        devObj.sampleBuffer.splice(0, devObj.sampleBuffer.length - BATCH_SIZE);
                        devObj.droppedBatches++;
                    }

                    if (devObj.sampleBuffer.length >= BATCH_SIZE) {
                        sendBatch(devObj);
                    }
                });

                device.addEventListener('gattserverdisconnected', () => {
                    activeDevices = activeDevices.filter(d => d.id !== devObj.id);
                    renderDeviceCards();
                });

            } catch (err) {
                console.error("BLE Connect Error:", err);
                alert("Ошибка подключения Bluetooth: " + err.message);
            }
        }

        async function sendBatch(devObj) {
            if (devObj.sending) return;                       // один запрос за раз
            if (devObj.sampleBuffer.length === 0) return;

            const batch = devObj.sampleBuffer.splice(0, devObj.sampleBuffer.length);
            devObj.sending = true;
            try {
                await fetch('/api/eeg_push', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user: devObj.user,
                        role: devObj.role,
                        sps: 250.0,
                        samples: batch
                    })
                });
            } catch (e) {
                // тихо: сеть моргнула, батч потерян, не валим соединение
            } finally {
                devObj.sending = false;
                if (devObj.sampleBuffer.length >= BATCH_SIZE) {
                    sendBatch(devObj);
                }
            }
        }

        function updateDeviceField(idx, field, value) {
            if (idx >= 0 && idx < activeDevices.length) {
                activeDevices[idx][field] = value;
                const el = document.getElementById('rate_' + activeDevices[idx].id);
                if (el) el.innerText = `Поток: ${activeDevices[idx].user}_${activeDevices[idx].role}`;
            }
        }

        function renderDeviceCards() {
            const container = document.getElementById('deviceList');
            const noDev = document.getElementById('noDevicesMsg');
            container.innerHTML = '';
            if (activeDevices.length === 0) {
                noDev.style.display = 'block';
                return;
            }
            noDev.style.display = 'none';

            activeDevices.forEach((dev, idx) => {
                const card = document.createElement('div');
                card.className = 'device-card';
                card.innerHTML = `
                    <h4>${dev.name} <span class="badge">ONLINE</span></h4>
                    <label>Пользователь / Субъект:</label>
                    <input type="text" value="${dev.user}"
                           onchange="updateDeviceField(${idx}, 'user', this.value)">
                    <label>Кортикальная роль (10-20 EEG):</label>
                    <select onchange="updateDeviceField(${idx}, 'role', this.value)">
                        <option value="AFz" ${dev.role==='AFz'?'selected':''}>AFz — Контент / Позитивный смысл</option>
                        <option value="Fpz" ${dev.role==='Fpz'?'selected':''}>Fpz — BA 10 Вето / Волевой отказ</option>
                        <option value="F3"  ${dev.role==='F3' ?'selected':''}>F3  — Левый dlPFC (Бета-Порядок)</option>
                        <option value="F4"  ${dev.role==='F4' ?'selected':''}>F4  — Правый dlPFC (Бета-Хаос)</option>
                        <option value="FCz" ${dev.role==='FCz'?'selected':''}>FCz — SMA (4D Кинематика Камеры)</option>
                    </select>
                    <div class="rate" id="rate_${dev.id}">Частота: 250 Hz | Поток: ${dev.user}_${dev.role}</div>
                `;
                container.appendChild(card);
            });
        }

        setInterval(() => {
            activeDevices.forEach(dev => {
                const el = document.getElementById('rate_' + dev.id);
                if (el) {
                    const drop = dev.droppedBatches > 0 ? ` | потерь: ${dev.droppedBatches}` : '';
                    el.innerText = `Частота: ${dev.packetCount} Hz | Поток: ${dev.user}_${dev.role}${drop}`;
                }
                dev.packetCount = 0;
            });
        }, 1000);
    </script>
</body>
</html>
"""


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
    # HTTP/1.1 — включает keep-alive, критично для fetch на удалённом хосте
    protocol_version = "HTTP/1.1"

    def _send_json(self, code, payload: bytes):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        clean_path = self.path.split('?')[0].rstrip('/')

        if clean_path in ['', '/index.html']:
            body = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        elif clean_path in ['/stream.mjpg', '/live']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            # НЕТ Content-Length: это бесконечный стрим, соединение не закрывается
            self.end_headers()
            while True:
                with jpeg_lock:
                    frame = latest_jpeg_frame
                if frame is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    except Exception:
                        break
                time.sleep(0.016)

        elif clean_path in ['/audio.wav', '/stream.wav']:
            self.send_response(200)
            self.send_header('Content-Type', 'audio/x-wav')
            self.send_header('Cache-Control', 'no-cache, no-store')
            # Без Content-Length: бесконечный поток
            self.end_headers()
            try:
                self.wfile.write(make_wav_header())
                self.wfile.flush()
            except Exception:
                return

            while True:
                chunk = None
                with audio_pcm_lock:
                    if len(audio_pcm_queue) > 0:
                        chunk = audio_pcm_queue.pop(0)
                if chunk is not None:
                    try:
                        floats = np.frombuffer(chunk, dtype=np.float32)
                        int16s = (np.clip(floats, -1.0, 1.0) * 32767).astype(np.int16)
                        self.wfile.write(int16s.tobytes())
                        self.wfile.flush()
                    except Exception:
                        break
                else:
                    time.sleep(0.02)

        elif clean_path == '/api/eeg_push':
            self._send_json(200, b'{"status":"ready","method":"POST"}')

        else:
            self.send_error(404)

    def do_POST(self):
        clean_path = self.path.split('?')[0].rstrip('/')
        length = int(self.headers.get('Content-Length', 0))

        if clean_path == '/api/eeg_push':
            if length == 0:
                self._send_json(400, b'{"status":"empty"}')
                return
            try:
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode('utf-8'))
                lsl_router.push(
                    payload.get('user', 'User1'),
                    payload.get('role', 'AFz'),
                    payload.get('samples', []),
                    float(payload.get('sps', DEFAULT_SPS))
                )
                self._send_json(200, b'{"status":"ok"}')
            except Exception as e:
                self._send_json(400, b'{"status":"error"}')

        elif clean_path == '/api/music_config':
            try:
                data = json.loads(self.rfile.read(length).decode('utf-8')) if length else {}
                current_music_config.update(data)
                with music_conn_lock:
                    if active_music_conn is not None:
                        try:
                            active_music_conn.send({'cmd': 'update_config', 'config': current_music_config})
                        except Exception:
                            pass
                self._send_json(200, b'{"status":"ok"}')
            except Exception:
                self._send_json(400, b'{"status":"error"}')

        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


class ThreadedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    allow_reuse_address = True


def ipc_frame_receiver():
    global latest_jpeg_frame
    print(f"🧠 [GATEWAY IPC] Слушаю кадры диффузии на localhost:{IPC_PORT} (key: canvas)...")
    try:
        listener = Listener(('localhost', IPC_PORT), authkey=IPC_AUTHKEY)
    except Exception as e:
        print(f"❌ [IPC ERROR] Порт {IPC_PORT} недоступен: {e}")
        return
    while True:
        try:
            conn = listener.accept()
            print("🔗 [GATEWAY IPC] NeuroCanvas подключен!")
            while True:
                data = conn.recv()
                if isinstance(data, bytes):
                    with jpeg_lock:
                        latest_jpeg_frame = data
        except (EOFError, ConnectionResetError):
            pass
        except Exception:
            time.sleep(0.5)


def ipc_audio_receiver():
    global active_music_conn
    print(f"🔊 [GATEWAY IPC] Слушаю аудио MusicGen на localhost:{IPC_AUDIO_PORT} (key: audio)...")
    try:
        listener = Listener(('localhost', IPC_AUDIO_PORT), authkey=IPC_AUDIO_AUTHKEY)
    except Exception as e:
        print(f"❌ [AUDIO IPC ERROR] Порт {IPC_AUDIO_PORT} недоступен: {e}")
        return
    while True:
        try:
            conn = listener.accept()
            with music_conn_lock:
                active_music_conn = conn
            print("🔗 [GATEWAY IPC] MusicGen подключен!")
            while True:
                data = conn.recv()
                if isinstance(data, bytes):
                    with audio_pcm_lock:
                        audio_pcm_queue.append(data)
                        if len(audio_pcm_queue) > AUDIO_QUEUE_MAX:
                            audio_pcm_queue.pop(0)
        except (EOFError, ConnectionResetError):
            pass
        except Exception:
            time.sleep(0.5)
        finally:
            with music_conn_lock:
                active_music_conn = None


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else HTTP_PORT

    threading.Thread(target=ipc_frame_receiver, daemon=True).start()
    threading.Thread(target=ipc_audio_receiver, daemon=True).start()

    server = ThreadedHTTPServer(('0.0.0.0', port), UnifiedCloudHandler)

    scheme = "http"
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=CERT_PATH, keyfile=KEY_PATH)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            scheme = "https"
            print(f"🔒 [GATEWAY] TLS активирован")
        except Exception as e:
            print(f"⚠️ [GATEWAY] TLS не запустился: {e}. HTTP.")

    print("=" * 65)
    print(f"🌐 [GATEWAY] Web UI & Web BLE: {scheme}://0.0.0.0:{port}/")
    print(f"🎥 [GATEWAY] Видео-стрим:      {scheme}://0.0.0.0:{port}/stream.mjpg")
    print(f"🔊 [GATEWAY] Аудио-стрим:      {scheme}://0.0.0.0:{port}/audio.wav")
    print(f"🧠 [GATEWAY] IPC видео:        localhost:{IPC_PORT} (key: canvas)")
    print(f"🎵 [GATEWAY] IPC аудио:        localhost:{IPC_AUDIO_PORT} (key: audio)")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка шлюза.")


if __name__ == '__main__':
    main()
