#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY (PORT 8080)
===================================================================================
- Раздача видео /stream.mjpg и потока звука /audio.wav (32 kHz WAV с RIFF заголовком)
- Web Bluetooth подключение FreeEEG16 с передачей данных в LSL
- Редактор обоих файлов лора (world_lore.txt и music_lore.txt) прямо из браузера
- Интерактивный мониторинг подключенных плат и переключение их кортикальных ролей
===================================================================================
"""

import sys
import os
import json
import time
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from pylsl import StreamInfo, StreamOutlet
import numpy as np

HTTP_PORT = 8080
IPC_CANVAS_PORT = 6002
IPC_AUDIO_PORT = 6003

latest_jpeg_frame = None
jpeg_lock = threading.Lock()

audio_pcm_queue = []
audio_pcm_lock = threading.Lock()
active_music_conn = None
music_conn_lock = threading.Lock()

current_music_config = {
    "target_block": 2.4,
    "target_context": 0.4,
    "target_buffer_count": 2,
    "temp": 1.0,
    "top_k": 250,
    "cfg_coef": 1.0
}

def make_wav_header(sample_rate=32000, num_channels=1, bits_per_sample=16):
    total_data_len = 0x7FFFFFFF - 36
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 0x7FFFFFFF, b'WAVE', b'fmt ',
        16, 1, num_channels, sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),
        num_channels * (bits_per_sample // 8),
        bits_per_sample, b'data', total_data_len
    )

class DynamicLSLRouter:
    def __init__(self):
        self.outlets = {}
        self.lock = threading.Lock()

    def push(self, user: str, role: str, samples: list, sps: float = 250.0):
        clean_user = "".join(c for c in user if c.isalnum() or c == "_").capitalize() or "User1"
        clean_role = "".join(c for c in role if c.isalnum()).capitalize() or "Afz"
        role_map = {"Afz": "AFz", "Fpz": "Fpz", "Fcz": "FCz", "F3": "F3", "F4": "F4"}
        clean_role = role_map.get(clean_role, clean_role)
        stream_name = f"{clean_user}_{clean_role}"

        with self.lock:
            if stream_name not in self.outlets:
                info = StreamInfo(name=stream_name, type='EEG', channel_count=16,
                                  nominal_srate=float(sps), channel_format='float32',
                                  source_id=f"ble_{stream_name}_{int(time.time())}")
                self.outlets[stream_name] = StreamOutlet(info)
                print(f"📡 [LSL ROUTER] Создан живой поток: '{stream_name}' ({sps:.0f} Hz)", flush=True)
            outlet = self.outlets[stream_name]

        for s in samples:
            if len(s) == 16: outlet.push_sample(s)

lsl_router = DynamicLSLRouter()

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>NeuroCanvas Unified Master Terminal</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #080c12; color: #c8d6e5; font-family: monospace; margin: 0; padding: 20px; }
        .header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1f2d3d; padding-bottom: 10px; margin-bottom: 20px; }
        h1 { font-size: 20px; margin: 0; color: #00ffc8; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .canvas-box { border: 2px solid #223145; border-radius: 8px; overflow: hidden; background: #000; width: 640px; padding: 15px; }
        .controls { background: #0f1622; padding: 15px; border-radius: 8px; border: 1px solid #223145; width: 520px; }
        h3 { color: #54a0ff; margin-top: 0; font-size: 14px; border-bottom: 1px solid #223145; padding-bottom: 6px; }
        .slider-group { margin-bottom: 12px; }
        .slider-group label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: #8395a7; }
        .slider-group span { color: #00ffc8; font-weight: bold; }
        input[type="range"] { width: 100%; accent-color: #00ffc8; cursor: pointer; }
        .btn { width: 100%; padding: 10px; cursor: pointer; font-weight: bold; border: none; border-radius: 6px; font-size: 13px; margin-bottom: 8px; transition: 0.2s; }
        .btn-ble { background: #10ac84; color: #fff; }
        .btn-audio { background: #ee5253; color: #fff; padding: 14px; font-size: 14px; margin-bottom: 12px; }
        .device-card { background: #182232; border: 1px solid #34495e; border-radius: 6px; padding: 10px; margin-top: 8px; }
        select, textarea { width: 100%; padding: 8px; background: #0f1622; border: 1px solid #2c3e50; color: #fff; border-radius: 4px; font-family: monospace; }
        textarea { height: 80px; resize: vertical; margin-bottom: 6px; }
        .tab-btn { background: #1b263b; border: 1px solid #2c3e50; color: #fff; padding: 6px 12px; cursor: pointer; border-radius: 4px; margin-right: 4px; }
        .tab-btn.active { background: #00ffc8; color: #000; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 NeuroCanvas Master Cloud Terminal</h1>
        <span style="color:#00ffc8;">PORT 8080 READY</span>
    </div>

    <div class="container">
        <div class="canvas-box">
            <h3 style="color:#00ffc8; margin-top:0;">🔊 Живое Радио MusicGen (32 kHz)</h3>
            <button id="btnAudio" class="btn btn-audio" onclick="toggleAudio()">▶️ ВКЛЮЧИТЬ ЗВУК В БРАУЗЕРЕ</button>
            <audio id="neuroAudio" preload="none"></audio>
            <div id="audioStatus" style="font-size:11px; color:#8395a7; margin-bottom: 15px;">Статус: Выключен</div>

            <h3 style="color:#54a0ff;">🖼️ Холст Диффузии</h3>
            <img src="/stream.mjpg" width="100%" height="340" style="object-fit:cover; border-radius:4px; background:#111;" alt="Ожидание диффузии...">
        </div>

        <div class="controls">
            <h3>📡 Управление Платами и Ролями</h3>
            <button class="btn btn-ble" onclick="connectBleDevice()">+ ПОДКЛЮЧИТЬ ПЛАТУ FREEEEG16</button>
            <button id="btnSim" class="btn" style="background:#2e86de; color:#fff;" onclick="toggleSimStream()">🚀 ВКЛЮЧИТЬ СИМУЛЯТОР LSL</button>
            <div id="bleLog" style="font-size:11px; color:#8395a7;">Поиск подключений...</div>
            <div id="activeDevices"></div>

            <h3 style="margin-top:15px;">📝 Редактор Лора (Концептуальное Пространство)</h3>
            <div>
                <button class="tab-btn active" id="tabMusic" onclick="selectLoreTab('music')">🎵 Музыка (music_lore.txt)</button>
                <button class="tab-btn" id="tabVision" onclick="selectLoreTab('vision')">🖼️ Картинки (world_lore.txt)</button>
            </div>
            <textarea id="loreTextarea" style="margin-top:8px;"></textarea>
            <button class="btn" style="background:#f39c12; color:#fff;" onclick="saveLore()">💾 СОХРАНИТЬ И ПРИМЕНИТЬ ЛОР</button>
            <div id="loreSaveStatus" style="font-size:11px; color:#00ffc8;"></div>

            <h3 style="margin-top:15px;">🎛️ Параметры Генерации MusicGen</h3>
            <div class="slider-group">
                <label>Target Block (Длина чанка): <span id="v_block">2.4s</span></label>
                <input type="range" min="0.5" max="5.0" step="0.1" value="2.4" oninput="updateParam('target_block', this.value, 'v_block', 's')">
            </div>
            <div class="slider-group">
                <label>Target Context (Префикс памяти): <span id="v_ctx">0.4s</span></label>
                <input type="range" min="0.1" max="1.5" step="0.05" value="0.4" oninput="updateParam('target_context', this.value, 'v_ctx', 's')">
            </div>
            <div class="slider-group">
                <label>Temperature (Энтропия тембра): <span id="v_temp">1.00</span></label>
                <input type="range" min="0.2" max="2.0" step="0.05" value="1.0" oninput="updateParam('temp', this.value, 'v_temp', '')">
            </div>
        </div>
    </div>

    <script>
        const SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
        const DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
        const CMD_CHAR_UUID = "c0de0001-36e1-4688-b7f5-ea07361b26a8";
        const UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6;

        let activeDevs = [];
        let isAudioPlaying = false;
        let simInterval = null;
        let currentTab = 'music';

        function selectLoreTab(tab) {
            currentTab = tab;
            document.getElementById('tabMusic').className = tab === 'music' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('tabVision').className = tab === 'vision' ? 'tab-btn active' : 'tab-btn';
            loadLore();
        }

        function loadLore() {
            fetch('/api/get_lore?type=' + currentTab)
                .then(r => r.text())
                .then(txt => { document.getElementById('loreTextarea').value = txt; });
        }
        loadLore();

        function saveLore() {
            const txt = document.getElementById('loreTextarea').value;
            fetch('/api/save_lore', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ type: currentTab, content: txt })
            }).then(() => {
                document.getElementById('loreSaveStatus').innerText = "✅ Лор сохранен и обновлен на лету!";
                setTimeout(() => { document.getElementById('loreSaveStatus').innerText = ""; }, 3000);
            });
        }

        function toggleAudio() {
            const audio = document.getElementById('neuroAudio');
            const btn = document.getElementById('btnAudio');
            const status = document.getElementById('audioStatus');
            if (!isAudioPlaying) {
                audio.src = '/audio.wav?t=' + Date.now();
                audio.play().then(() => {
                    isAudioPlaying = true;
                    btn.innerText = "⏸️ ВЫКЛЮЧИТЬ ЗВУК";
                    btn.style.background = "#10ac84";
                    status.innerText = "Статус: 🟢 ИГРАЕТ (32 kHz Live Stream)";
                    status.style.color = "#00ffc8";
                }).catch(e => {
                    status.innerText = "Ошибка: " + e.message;
                    status.style.color = "#ff6b6b";
                });
            } else {
                audio.pause();
                audio.src = "";
                isAudioPlaying = false;
                btn.innerText = "▶️ ВКЛЮЧИТЬ ЗВУК В БРАУЗЕРЕ";
                btn.style.background = "#ee5253";
                status.innerText = "Статус: Выключен";
                status.style.color = "#8395a7";
            }
        }

        async function connectBleDevice() {
            const logEl = document.getElementById('bleLog');
            try {
                logEl.innerText = "Поиск Bluetooth...";
                const device = await navigator.bluetooth.requestDevice({ filters: [{ services: [SERVICE_UUID] }] });
                logEl.innerText = "Подключение к " + device.name + "...";
                const server = await device.gatt.connect();
                const service = await server.getPrimaryService(SERVICE_UUID);
                const dataChar = await service.getCharacteristic(DATA_CHAR_UUID);
                const cmdChar = await service.getCharacteristic(CMD_CHAR_UUID);

                await cmdChar.writeValue(new Uint8Array([0x04, 0x44, 0x44])).catch(()=>{});
                await cmdChar.writeValue(new Uint8Array([0x05, 0x44, 0x44])).catch(()=>{});

                const devObj = { id: Date.now(), name: device.name || "FreeEEG16", user: "User" + (activeDevs.length + 1), role: "AFz", samples: [] };
                activeDevs.push(devObj);
                renderDevs();

                await dataChar.startNotifications();
                dataChar.addEventListener('characteristicvaluechanged', (e) => {
                    const b = new Uint8Array(e.target.value.buffer);
                    if (b[0] === 0xA0) {
                        const s = [];
                        for (let i = 0; i < 16; i++) {
                            let v = (b[2 + i*3] << 16) | (b[3 + i*3] << 8) | b[4 + i*3];
                            if (v & 0x800000) v -= 0x1000000;
                            s.push(v * UV_SCALE);
                        }
                        devObj.samples.push(s);
                        if (devObj.samples.length >= 8) {
                            const batch = devObj.samples.splice(0, devObj.samples.length);
                            fetch('/api/eeg_push', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({ user: devObj.user, role: devObj.role, samples: batch, sps: 250.0 })
                            }).catch(()=>{});
                        }
                    }
                });

                device.addEventListener('gattserverdisconnected', () => {
                    activeDevs = activeDevs.filter(d => d.id !== devObj.id);
                    renderDevs();
                });
                logEl.innerText = "✅ " + device.name + " подключен!";
            } catch(e) { logEl.innerText = "❌ Ошибка BLE: " + e.message; }
        }

        function toggleSimStream() {
            const btn = document.getElementById('btnSim');
            if (simInterval) {
                clearInterval(simInterval);
                simInterval = null;
                btn.innerText = "🚀 ВКЛЮЧИТЬ СИМУЛЯТОР LSL";
                btn.style.background = "#2e86de";
            } else {
                let t = 0;
                simInterval = setInterval(() => {
                    t += 0.032;
                    const samples = [];
                    for (let step = 0; step < 8; step++) {
                        const s = [];
                        for (let ch = 0; ch < 16; ch++) {
                            s.push(Math.sin(2 * Math.PI * 6.0 * (t + step/250.0) + ch * 0.3) * 15.0);
                        }
                        samples.push(s);
                    }
                    fetch('/api/eeg_push', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ user: "SimUser", role: "AFz", samples: samples, sps: 250.0 })
                    }).catch(()=>{});
                }, 32);
                btn.innerText = "⏹️ ОСТАНОВИТЬ СИМУЛЯТОР";
                btn.style.background = "#e67e22";
            }
        }

        function changeDeviceRole(idx, newRole) {
            activeDevs[idx].role = newRole;
            fetch('/api/device_role', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ dev_idx: idx, role: newRole })
            }).catch(()=>{});
        }

        function renderDevs() {
            const c = document.getElementById('activeDevices');
            c.innerHTML = activeDevs.map((d, idx) => `
                <div class="device-card">
                    <b>${d.name}</b> (${d.user})
                    <select onchange="changeDeviceRole(${idx}, this.value)">
                        <option value="AFz" ${d.role==='AFz'?'selected':''}>AFz (Макро-лад и тоника)</option>
                        <option value="F3"  ${d.role==='F3'?'selected':''}>F3 (Ритмический синтаксис 2D)</option>
                        <option value="F4"  ${d.role==='F4'?'selected':''}>F4 (Тембр и энтропия)</option>
                        <option value="FCz" ${d.role==='FCz'?'selected':''}>FCz (4D моторный поток)</option>
                        <option value="Fpz" ${d.role==='Fpz'?'selected':''}>Fpz (The Drop / Ветвление)</option>
                    </select>
                </div>
            `).join('');
        }

        function updateParam(param, val, labelId, suffix) {
            document.getElementById(labelId).innerText = val + suffix;
            const p = {}; p[param] = parseFloat(val);
            fetch('/api/music_config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(p)
            }).catch(()=>{});
        }
    </script>
</body>
</html>
"""

class UnifiedHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        clean = self.path.split('?')[0].rstrip('/')
        if clean in ['', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif clean in ['/stream.mjpg', '/live']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with jpeg_lock: f = latest_jpeg_frame
                if f:
                    try:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(f)).encode() + b'\r\n\r\n' + f + b'\r\n')
                    except Exception: break
                time.sleep(0.016)

        elif clean in ['/audio.wav', '/stream.wav']:
            self.send_response(200)
            self.send_header('Content-Type', 'audio/x-wav')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.end_headers()

            header = make_wav_header(sample_rate=32000, num_channels=1, bits_per_sample=16)
            try:
                self.wfile.write(header)
                self.wfile.flush()
            except Exception: return

            while True:
                chunk = None
                with audio_pcm_lock:
                    if len(audio_pcm_queue) > 0: chunk = audio_pcm_queue.pop(0)

                if chunk is not None:
                    try:
                        floats = np.frombuffer(chunk, dtype=np.float32)
                        int16s = (np.clip(floats, -1.0, 1.0) * 32767).astype(np.int16)
                        self.wfile.write(int16s.tobytes())
                        self.wfile.flush()
                    except Exception: break
                else: time.sleep(0.02)

        elif clean == '/api/get_lore':
            lore_type = self.path.split('type=')[-1] if 'type=' in self.path else 'music'
            fn = "music_lore.txt" if lore_type == 'music' else "world_lore.txt"
            content = ""
            if os.path.exists(fn):
                with open(fn, 'r', encoding='utf-8') as f: content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        elif clean == '/api/music_config':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(current_music_config).encode())
        else: self.send_error(404)

    def do_POST(self):
        global active_music_conn
        clean = self.path.split('?')[0].rstrip('/')
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length).decode('utf-8')) if length > 0 else {}

        if clean == '/api/save_lore':
            lore_type = data.get('type', 'music')
            content = data.get('content', '')
            fn = "music_lore.txt" if lore_type == 'music' else "world_lore.txt"
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📝 [LORE SAVED]: Обновлен {fn} из браузера!", flush=True)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        elif clean == '/api/device_role':
            dev_idx = int(data.get('dev_idx', 0))
            role = data.get('role', None)
            with music_conn_lock:
                if active_music_conn:
                    try: active_music_conn.send({'cmd': 'set_role', 'payload': (dev_idx, role)})
                    except Exception: active_music_conn = None
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        elif clean == '/api/music_config':
            current_music_config.update(data)
            with music_conn_lock:
                if active_music_conn:
                    try: active_music_conn.send({'cmd': 'update_config', 'config': current_music_config})
                    except Exception: active_music_conn = None
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

        elif clean == '/api/eeg_push':
            lsl_router.push(data.get('user', 'User1'), data.get('role', 'AFz'), data.get('samples', []), float(data.get('sps', 250.0)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else: self.send_error(404)

    def log_message(self, format, *args): pass

def ipc_canvas_loop():
    global latest_jpeg_frame
    try: l = Listener(('localhost', IPC_CANVAS_PORT), authkey=b'canvas')
    except Exception: return
    while True:
        try:
            c = l.accept()
            while True:
                d = c.recv()
                if isinstance(d, bytes):
                    with jpeg_lock: latest_jpeg_frame = d
        except Exception: time.sleep(0.5)

def ipc_audio_loop():
    global active_music_conn
    try: l = Listener(('localhost', IPC_AUDIO_PORT), authkey=b'audio')
    except Exception: return
    while True:
        try:
            c = l.accept()
            with music_conn_lock: active_music_conn = c
            print("🔗 [GATEWAY] Аудио-движок MusicGen подключен!", flush=True)
            while True:
                d = c.recv()
                if isinstance(d, bytes):
                    with audio_pcm_lock:
                        audio_pcm_queue.append(d)
                        if len(audio_pcm_queue) > 60: audio_pcm_queue.pop(0)
        except Exception: time.sleep(0.5)

def main():
    threading.Thread(target=ipc_canvas_loop, daemon=True).start()
    threading.Thread(target=ipc_audio_loop, daemon=True).start()
    server = ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), UnifiedHandler)
    print("=" * 65)
    print(f"🌐 [GATEWAY] Master Web Terminal: http://localhost:{HTTP_PORT}/")
    print(f"🔊 [GATEWAY] Поток радио:        http://localhost:{HTTP_PORT}/audio.wav")
    print("=" * 65)
    try: server.serve_forever()
    except KeyboardInterrupt: pass

if __name__ == '__main__':
    main()
