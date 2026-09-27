#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY (PORT 8080 & IPC 6002)
===================================================================================
- Никаких фейковых потоков по умолчанию: только живые данные от BLE или API.
- IPC Слушатель (порт 6002, key: canvas) стартует строго в процессе шлюза.
- Web UI с Web Bluetooth API: подключение FreeEEG16 прямо из браузера.
===================================================================================
"""

import sys
import os
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from pylsl import StreamInfo, StreamOutlet

HTTP_PORT = 8080
IPC_PORT = 6002
IPC_AUTHKEY = b'canvas'
NUM_CHANNELS = 16
DEFAULT_SPS = 250.0

latest_jpeg_frame = None
jpeg_lock = threading.Lock()

# -----------------------------------------------------------------------------
# 1. МЕНЕДЖЕР ДИНАМИЧЕСКИХ LSL-ПОТОКОВ (ТОЛЬКО ДЛЯ РЕАЛЬНЫХ ДАННЫХ)
# -----------------------------------------------------------------------------
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
                    name=stream_name,
                    type='EEG',
                    channel_count=NUM_CHANNELS,
                    nominal_srate=float(sps),
                    channel_format='float32',
                    source_id=f"ble_{stream_name}_{int(time.time())}"
                )
                self.outlets[stream_name] = StreamOutlet(info)
                print(f"📡 [LSL ROUTER] Живой поток зарегистрирован: '{stream_name}' ({sps:.0f} Hz)")
            return self.outlets[stream_name]

    def push(self, user: str, role: str, samples: list, sps: float = DEFAULT_SPS):
        outlet = self.get_or_create_outlet(user, role, sps)
        for s in samples:
            if len(s) == NUM_CHANNELS:
                outlet.push_sample(s)

lsl_router = DynamicLSLRouter()

# -----------------------------------------------------------------------------
# 2. WEB UI (ТОЛЬКО РЕАЛЬНЫЙ BLE И МОНИТОРИНГ)
# -----------------------------------------------------------------------------
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
        .controls { background: #0f1622; padding: 20px; border-radius: 8px; border: 1px solid #223145; width: 420px; }
        h3 { color: #54a0ff; margin-top: 0; font-size: 15px; border-bottom: 1px solid #223145; padding-bottom: 8px; }
        .btn-ble { width: 100%; padding: 14px; background: #10ac84; cursor: pointer; font-weight: bold; border: none; border-radius: 6px; color: #fff; font-size: 14px; transition: 0.2s; margin-bottom: 16px; }
        .btn-ble:hover { background: #1dd1a1; }
        .device-card { background: #182232; border: 1px solid #34495e; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
        .device-card h4 { margin: 0 0 10px 0; color: #00ffc8; font-size: 14px; display: flex; justify-content: space-between; }
        label { display: block; margin: 8px 0 4px; color: #8395a7; font-size: 12px; }
        select, input { width: 100%; padding: 8px; background: #0f1622; border: 1px solid #2c3e50; color: #fff; border-radius: 4px; font-family: monospace; font-size: 13px; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #10ac84; color: #fff; }
        .rate { font-size: 11px; color: #1dd1a1; margin-top: 6px; }
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
            <h3>📡 Bluetooth Подключение (FreeEEG16)</h3>
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

        let activeDevices = [];

        async function connectNewBleDevice() {
            try {
                const device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [SERVICE_UUID] }]
                });

                const server = await device.gatt.connect();
                const service = await server.getPrimaryService(SERVICE_UUID);
                const dataChar = await service.getCharacteristic(DATA_CHAR_UUID);
                const cmdChar = await service.getCharacteristic(CMD_CHAR_UUID);

                // Усиление 16x для ADS131M08
                await cmdChar.writeValue(new Uint8Array([0x04, 0x44, 0x44])).catch(()=>{});
                await cmdChar.writeValue(new Uint8Array([0x05, 0x44, 0x44])).catch(()=>{});

                const devId = Date.now();
                const devObj = {
                    id: devId,
                    name: device.name || "FreeEEG16",
                    device: device,
                    user: "User" + (activeDevices.length + 1),
                    role: activeDevices.length === 0 ? "AFz" : "Fpz",
                    packetCount: 0,
                    sampleBuffer: []
                };

                activeDevices.push(devObj);
                renderDeviceCards();

                await dataChar.startNotifications();
                dataChar.addEventListener('characteristicvaluechanged', (e) => {
                    const b = new Uint8Array(e.target.value.buffer);
                    if (b[0] === 0xA0) {
                        const sample = [];
                        for (let i = 0; i < 16; i++) {
                            let v = (b[2 + i * 3] << 16) | (b[3 + i * 3] << 8) | b[4 + i * 3];
                            if (v & 0x800000) v -= 0x1000000;
                            sample.push(v * UV_SCALE);
                        }
                        devObj.sampleBuffer.push(sample);
                        devObj.packetCount++;

                        if (devObj.sampleBuffer.length >= 8) {
                            sendBatch(devObj);
                        }
                    }
                });

                device.addEventListener('gattserverdisconnected', () => {
                    activeDevices = activeDevices.filter(d => d.id !== devId);
                    renderDeviceCards();
                });

            } catch (err) {
                console.error("BLE Connect Error:", err);
                alert("Ошибка подключения Bluetooth: " + err.message);
            }
        }

        function sendBatch(devObj) {
            const batch = devObj.sampleBuffer.splice(0, devObj.sampleBuffer.length);
            fetch('/api/eeg_push', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user: devObj.user,
                    role: devObj.role,
                    sps: 250.0,
                    samples: batch
                })
            }).catch(()=>{});
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

            activeDevices.forEach((dev) => {
                const card = document.createElement('div');
                card.className = 'device-card';
                card.innerHTML = `
                    <h4>${dev.name} <span class="badge">ONLINE</span></h4>
                    <label>Пользователь / Субъект:</label>
                    <input type="text" value="${dev.user}" onchange="dev.user = this.value">
                    <label>Кортикальная роль (10-20 EEG):</label>
                    <select onchange="dev.role = this.value">
                        <option value="AFz" ${dev.role==='AFz'?'selected':''}>AFz — Контент / Позитивный смысл</option>
                        <option value="Fpz" ${dev.role==='Fpz'?'selected':''}>Fpz — BA 10 Вето / Волевой отказ</option>
                        <option value="F3"  ${dev.role==='F3'?'selected':''}>F3  — Левый dlPFC (Бета-Порядок)</option>
                        <option value="F4"  ${dev.role==='F4'?'selected':''}>F4  — Правый dlPFC (Бета-Хаос)</option>
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
                    el.innerText = `Частота: ${dev.packetCount} Hz | Поток: ${dev.user}_${dev.role}`;
                }
                dev.packetCount = 0;
            });
        }, 1000);
    </script>
</body>
</html>
"""

class UnifiedCloudHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        clean_path = self.path.split('?')[0].rstrip('/')

        if clean_path in ['', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif clean_path in ['/stream.mjpg', '/live']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            while True:
                with jpeg_lock:
                    frame = latest_jpeg_frame
                if frame is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    except Exception:
                        break
                time.sleep(0.016)

        elif clean_path == '/api/eeg_push':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "method": "POST"}, indent=2).encode('utf-8'))

        else:
            self.send_error(404)

    def do_POST(self):
        clean_path = self.path.split('?')[0].rstrip('/')
        if clean_path == '/api/eeg_push':
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self.send_response(400)
                self.end_headers()
                return
            try:
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                lsl_router.push(
                    payload.get('user', 'User1'),
                    payload.get('role', 'AFz'),
                    payload.get('samples', []),
                    float(payload.get('sps', DEFAULT_SPS))
                )
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

def ipc_frame_receiver():
    """Слушатель на порту 6002 запускается ТОЛЬКО когда скрипт запущен напрямую."""
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

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else HTTP_PORT
    
    # Запускаем фоновый IPC-приемник кадров
    threading.Thread(target=ipc_frame_receiver, daemon=True).start()

    server = ThreadingHTTPServer(('0.0.0.0', port), UnifiedCloudHandler)
    print("=" * 65)
    print(f"🌐 [GATEWAY] Web UI & Web BLE: http://localhost:{port}/")
    print(f"🎥 [GATEWAY] Видео-стрим:      http://localhost:{port}/stream.mjpg")
    print(f"🧠 [GATEWAY] IPC порт:         localhost:{IPC_PORT} (key: canvas)")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка шлюза.")

if __name__ == '__main__':
    main()
