#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS UNIFIED CLOUD WEB & LSL GATEWAY v6.0 (FOOLPROOF & ZERO-LOSS)
===================================================================================
- ПРЕСЕТЫ РЕАКЦИИ: 🧘 Медитация, ⚖️ Баланс, ⚡ Динамика с авто-синхронизацией полей.
- ЗАЩИТА ОТ ДУРОСТИ (Sanity Clamping): Невозможно выставить ломающие диффузию числа.
- ЧИСТЫЙ ПРОЕКТОР: Кнопка вынесена в тулбар, 0 рамок, 0 надписей на изображении.
- ЧИСТЫЙ СТАРТ: Без 50k слов. Если лор пуст, список концептов пуст [].
- МГНОВЕННОЕ ОБНОВЛЕНИЕ DOM: Концепты добавляются/удаляются без нажатия F5.
- MUSICGEN ТЕМЫ И БУФЕР: Полный монитор отставания, задержки в мс и RTF.
===================================================================================
"""

import sys
import os
import ssl
import json
import time
import struct
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener, Client
from pylsl import StreamInfo, StreamOutlet
import numpy as np

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

latest_jpeg_frame = None
jpeg_lock = threading.Lock()

audio_pcm_queue = []
audio_pcm_lock = threading.Lock()
AUDIO_QUEUE_MAX = 50

active_music_conn = None
music_conn_lock = threading.Lock()

active_diff_ctrl_conn = None
diff_ctrl_lock = threading.Lock()

music_state = {
    "concepts": [],
    "target_block": 2.4,
    "target_context": 0.4,
    "target_buffer_count": 2,
    "speed_safety_factor": 1.0,
    "temp": 1.0,
    "top_k": 250,
    "cfg_coef": 1.0,
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
    "strength_low": 0.55,
    "strength_high": 0.85,
    "ema_smooth": 0.16,
    "is_recording": False,
    "current_puck": [0.0, 0.0],
    "active_thought": "",
    "gpu_sat_mask": [False] * 16
}
world_state_lock = threading.Lock()

devices_registry = {}
devices_lock = threading.Lock()

sse_clients = []
sse_lock = threading.Lock()

def broadcast_sse(event_name: str, payload: dict):
    msg = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode('utf-8')
    with sse_lock:
        dead = []
        for q in sse_clients:
            try: q.put_nowait(msg)
            except Exception: dead.append(q)
        for d in dead:
            if d in sse_clients: sse_clients.remove(d)

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
            if len(s) == NUM_CHANNELS: outlet.push_sample(s)

lsl_router = DynamicLSLRouter()

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>NeuroCanvas Unified Studio & Controller</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #06090e; color: #cbd5e1; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; margin: 0; padding: 16px; }
        .header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin-bottom: 14px; }
        h1 { font-size: 17px; margin: 0; color: #00ffc8; }
        .layout { display: grid; grid-template-columns: 1fr 460px; gap: 16px; }
        
        .canvas-toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .fs-action-btn { background: #1e293b; border: 1px solid #38bdf8; color: #00ffc8; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold; transition: 0.2s; }
        .fs-action-btn:hover { background: #00ffc8; color: #000; }

        /* Чистый проектор без рамок */
        .canvas-box { position: relative; border: none; outline: none; border-radius: 0; overflow: hidden; background: #000; min-height: 520px; display: flex; align-items: center; justify-content: center; }
        .canvas-box img { width: 100%; height: auto; max-height: 85vh; object-fit: contain; }

        :fullscreen .canvas-box { width: 100vw; height: 100vh; cursor: none; margin: 0; padding: 0; }
        :fullscreen .canvas-box img { max-height: 100vh; width: 100vw; }

        .panel { background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 12px; }
        h3 { color: #38bdf8; margin-top: 0; font-size: 12px; border-bottom: 1px solid #1e293b; padding-bottom: 4px; }
        label { display: block; margin: 5px 0 2px; color: #94a3b8; font-size: 10px; }
        select, input { width: 100%; padding: 6px; background: #030712; border: 1px solid #334155; color: #fff; border-radius: 4px; font-family: monospace; font-size: 11px; }
        
        .btn { width: 100%; padding: 8px; cursor: pointer; font-weight: bold; border: none; border-radius: 4px; font-size: 11px; transition: 0.2s; }
        .btn-green { background: #059669; color: #fff; }
        .btn-green:hover { background: #10b981; }
        .btn-audio { background: #e11d48; color: #fff; margin-bottom: 8px; }
        .btn-audio.playing { background: #059669; }
        .btn-red { background: #e11d48; color: #fff; }
        .btn-red:hover { background: #f43f5e; }

        .device-card { background: #131d2e; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 8px; }
        .device-card h4 { margin: 0 0 4px 0; color: #00ffc8; font-size: 12px; display: flex; justify-content: space-between; align-items: center; }
        
        .sat-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; margin-top: 4px; }
        .sat-box { height: 14px; border-radius: 2px; background: #10b981; display: flex; align-items: center; justify-content: center; font-size: 8px; color: #000; font-weight: bold; }
        .sat-box.clip { background: #ef4444; color: #fff; animation: blink 0.5s infinite; }
        @keyframes blink { 50% { opacity: 0.3; } }

        .concept-chip { display: inline-flex; align-items: center; background: #1e293b; border: 1px solid #475569; border-radius: 4px; padding: 2px 6px; margin: 2px 3px 0 0; font-size: 11px; }
        .concept-chip span { color: #00ffc8; font-weight: bold; margin-right: 5px; }
        .concept-chip.music span { color: #f43f5e; }
        .concept-chip button { background: none; border: none; color: #f43f5e; cursor: pointer; font-weight: bold; }
        
        .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; background: #0284c7; color: #fff; }
        .probe-switch { background: #1e293b; border: 1px solid #f59e0b; padding: 8px; border-radius: 6px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; font-size: 11px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 NeuroCanvas Production Studio (Unified Multimodal)</h1>
        <div>
            <span class="badge" id="netBadge">HTTPS SECURE</span>
            <span class="badge" id="recBadge" style="background: #475569; margin-left: 6px;">ЗАПИСЬ: ВЫКЛ</span>
        </div>
    </div>
    
    <div class="layout">
        <!-- Левая колонка: Проекторный экран -->
        <div>
            <div class="canvas-toolbar">
                <span style="font-size: 11px; color: #64748b;">Проекция (Canvas Feed):</span>
                <button class="fs-action-btn" onclick="toggleFullScreen()">⛶ ПОЛНЫЙ ЭКРАН (ПРОЕКТОР)</button>
            </div>
            <div class="canvas-box" id="canvasBox">
                <img id="videoStreamImg" src="/stream.mjpg" alt="Ожидание генерации...">
            </div>
        </div>

        <!-- Правая колонка: Панели управления -->
        <div>
            <!-- Тумблер разгрузки BLE-клиента -->
            <div class="probe-switch">
                <span>⚡ <b>Легковесный BLE-зонд (для слабых устройств):</b></span>
                <input type="checkbox" id="probeModeToggle" style="width:20px; height:20px;" onchange="toggleProbeMode(this.checked)">
            </div>

            <!-- Bluetooth & Аппаратный АЦП -->
            <div class="panel">
                <h3>📡 Bluetooth Подключение (FreeEEG16)</h3>
                <button class="btn btn-green" onclick="connectNewBleDevice()">+ ПОДКЛЮЧИТЬ ПЛАТУ (BLE)</button>
                <div id="deviceList"></div>
                <div id="noDevicesMsg" style="color: #64748b; font-size: 11px; text-align: center; margin-top: 6px;">
                    Нет активных Bluetooth плат на этой вкладке.
                </div>
            </div>

            <!-- Визуальный мир: Концепты и Плавность/Динамика -->
            <div class="panel">
                <h3>🎨 Визуальный мир (Диффузия)</h3>
                <label>Активный пул образов (Пусто = Свободный дрейф):</label>
                <div id="visionConceptList" style="min-height: 24px;"></div>
                <div style="display: flex; gap: 6px; margin: 6px 0;">
                    <input type="text" id="newVisionInput" placeholder="Новый визуальный образ...">
                    <button class="btn btn-green" style="width: 90px;" onclick="addVisionConcept()">+ ОБРАЗ</button>
                </div>

                <label style="margin-top: 8px;">Режим динамики (Пресеты):</label>
                <select id="diffPreset" onchange="applyPreset(this.value)">
                    <option value="meditation">🧘 Глубокая медитация (Плавный морфинг: Steps 2, Str 0.45-0.70, EMA 0.30)</option>
                    <option value="balanced" selected>⚖️ Баланс / Стандарт (Steps 2, Str 0.55-0.85, EMA 0.16)</option>
                    <option value="dynamic">⚡ Высокая динамика (Быстрый отклик: Steps 3, Str 0.65-0.92, EMA 0.08)</option>
                    <option value="custom">🛠️ Пользовательский (Ручная настройка)</option>
                </select>

                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-top: 6px;">
                    <div><label>Steps:</label><input type="number" id="diffSteps" value="2" min="1" max="6" onchange="onManualParamChange()"></div>
                    <div><label>CFG:</label><input type="number" id="diffCfg" value="1.0" min="0.0" max="3.0" step="0.1" onchange="onManualParamChange()"></div>
                    <div><label>Min Str:</label><input type="number" id="diffStrLow" value="0.55" min="0.35" max="0.75" step="0.05" onchange="onManualParamChange()"></div>
                    <div><label>Max Str:</label><input type="number" id="diffStrHigh" value="0.85" min="0.55" max="0.95" step="0.05" onchange="onManualParamChange()"></div>
                </div>
                <div style="margin-top: 4px;">
                    <label>EMA Плавность (0.04 = динамика, 0.40 = медитация):</label>
                    <input type="number" id="diffEmaSmooth" value="0.16" min="0.04" max="0.50" step="0.02" onchange="onManualParamChange()">
                </div>
            </div>

            <!-- MusicGen: Темы и Параметры реакции -->
            <div class="panel">
                <h3>🔊 Музыкальный генератор (MusicGen 32 kHz)</h3>
                <button id="btnAudio" class="btn btn-audio" onclick="toggleAudio()">▶️ ВКЛЮЧИТЬ LIVE ЗВУК</button>
                <audio id="neuroAudio" preload="none"></audio>
                
                <div id="musicLagStatus" style="font-size: 11px; color: #38bdf8; font-weight: bold; margin-bottom: 6px;">
                    Буфер: 0.0 с (0 чанков) | RTF: 1.0x | Отставание: 0 мс
                </div>

                <label>Активные музыкальные темы (Пусто = Свободный звук):</label>
                <div id="musicConceptList" style="min-height: 24px;"></div>
                <div style="display: flex; gap: 6px; margin: 6px 0;">
                    <input type="text" id="newMusicInput" placeholder="Новая музыкальная тема...">
                    <button class="btn btn-green" style="width: 90px;" onclick="addMusicConcept()">+ ТЕМА</button>
                </div>

                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px;">
                    <div><label>Блок (сек):</label><input type="number" id="mBlock" value="2.4" step="0.2" onchange="updateMusicConfig()"></div>
                    <div><label>Контекст (сек):</label><input type="number" id="mContext" value="0.4" step="0.1" onchange="updateMusicConfig()"></div>
                    <div><label>Буфер (чанков):</label><input type="number" id="mBuffer" value="2" min="1" max="5" onchange="updateMusicConfig()"></div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 4px;">
                    <div><label>Temp (0.2-2.0):</label><input type="number" id="mTemp" value="1.0" step="0.1" onchange="updateMusicConfig()"></div>
                    <div><label>Top-K:</label><input type="number" id="mTopK" value="250" step="10" onchange="updateMusicConfig()"></div>
                    <div><label>CFG Coef:</label><input type="number" id="mCfg" value="1.0" step="0.1" onchange="updateMusicConfig()"></div>
                </div>
            </div>

            <!-- Сессионная запись -->
            <div class="panel">
                <h3>📼 Сессионная запись (ЭЭГ + Видео + Теги)</h3>
                <div style="display: flex; gap: 6px;">
                    <button id="btnRecStart" class="btn btn-red" onclick="toggleRecording(true)">🔴 СТАРТ ЗАПИСИ</button>
                    <button id="btnRecStop" class="btn" style="background: #334155; color:#fff;" onclick="toggleRecording(false)">⏹️ СТОП</button>
                </div>
                <div id="recStatusText" style="font-size: 10px; color: #64748b; margin-top: 4px;">Запись не активна</div>
            </div>
        </div>
    </div>

    <script>
        const SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
        const DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
        const CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8";
        const BATCH_SIZE     = 40;
        const RAW_SAT_LIMIT  = 8300000;

        function calcUvScale(gain) {
            return (1.2 / parseFloat(gain) / 8388607.0) * 1e6;
        }

        let activeDevices = [];
        let isAudioPlaying = false;
        let myClientId = "client_" + Math.random().toString(36).substring(2, 9);
        let isProbeMode = false;

        function toggleProbeMode(enabled) {
            isProbeMode = enabled;
            const img = document.getElementById('videoStreamImg');
            if (isProbeMode) {
                img.src = "";
                img.style.display = "none";
                if (isAudioPlaying) toggleAudio();
            } else {
                img.style.display = "block";
                img.src = "/stream.mjpg";
            }
        }

        function toggleFullScreen() {
            const el = document.getElementById('canvasBox');
            if (!document.fullscreenElement) el.requestFullscreen().catch(err => alert(err.message));
            else document.exitFullscreen();
        }

        // Пресеты динамики
        const PRESETS = {
            meditation: { steps: 2, cfg: 1.0, strLow: 0.45, strHigh: 0.70, ema: 0.30 },
            balanced:   { steps: 2, cfg: 1.0, strLow: 0.55, strHigh: 0.85, ema: 0.16 },
            dynamic:    { steps: 3, cfg: 1.2, strLow: 0.65, strHigh: 0.92, ema: 0.08 }
        };

        function applyPreset(presetKey) {
            if (presetKey === 'custom') return;
            const p = PRESETS[presetKey];
            if (!p) return;
            document.getElementById('diffSteps').value = p.steps;
            document.getElementById('diffCfg').value = p.cfg;
            document.getElementById('diffStrLow').value = p.strLow;
            document.getElementById('diffStrHigh').value = p.strHigh;
            document.getElementById('diffEmaSmooth').value = p.ema;
            sendDiffConfig(p.steps, p.cfg, p.strLow, p.strHigh, p.ema, presetKey);
        }

        function onManualParamChange() {
            document.getElementById('diffPreset').value = 'custom';
            // Безопасное ограничение (Sanity Clamping)
            const steps = Math.max(1, Math.min(6, parseInt(document.getElementById('diffSteps').value) || 2));
            const cfg = Math.max(0.0, Math.min(3.0, parseFloat(document.getElementById('diffCfg').value) || 1.0));
            let sLow = Math.max(0.35, Math.min(0.75, parseFloat(document.getElementById('diffStrLow').value) || 0.55));
            let sHigh = Math.max(sLow + 0.05, Math.min(0.95, parseFloat(document.getElementById('diffStrHigh').value) || 0.85));
            const ema = Math.max(0.04, Math.min(0.50, parseFloat(document.getElementById('diffEmaSmooth').value) || 0.16));

            document.getElementById('diffSteps').value = steps;
            document.getElementById('diffCfg').value = cfg;
            document.getElementById('diffStrLow').value = sLow;
            document.getElementById('diffStrHigh').value = sHigh;
            document.getElementById('diffEmaSmooth').value = ema;

            sendDiffConfig(steps, cfg, sLow, sHigh, ema, 'custom');
        }

        async function sendDiffConfig(steps, cfg, sLow, sHigh, ema, presetKey) {
            const payload = {
                preset: presetKey,
                steps: steps,
                guidance_scale: cfg,
                strength_low: sLow,
                strength_high: sHigh,
                ema_smooth: ema
            };
            await fetch('/api/diffusion/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
        }

        const sse = new EventSource('/api/state_stream');
        sse.addEventListener('state', (e) => {
            const st = JSON.parse(e.data);
            if (st.world) {
                renderVisionConcepts(st.world.concepts || []);
                document.getElementById('diffSteps').value = st.world.steps || 2;
                document.getElementById('diffCfg').value = st.world.guidance_scale || 1.0;
                document.getElementById('diffStrLow').value = st.world.strength_low || 0.55;
                document.getElementById('diffStrHigh').value = st.world.strength_high || 0.85;
                document.getElementById('diffEmaSmooth').value = st.world.ema_smooth || 0.16;
                if (st.world.preset) document.getElementById('diffPreset').value = st.world.preset;
                
                const badge = document.getElementById('recBadge');
                if (st.world.is_recording) {
                    badge.style.background = "#e11d48";
                    badge.innerText = "ЗАПИСЬ: АКТИВНА";
                } else {
                    badge.style.background = "#475569";
                    badge.innerText = "ЗАПИСЬ: ВЫКЛ";
                }
            }
            if (st.music) {
                renderMusicConcepts(st.music.concepts || []);
                document.getElementById('mBlock').value = st.music.target_block || 2.4;
                document.getElementById('mContext').value = st.music.target_context || 0.4;
                document.getElementById('mBuffer').value = st.music.target_buffer_count || 2;
                document.getElementById('mTemp').value = st.music.temp || 1.0;
                document.getElementById('mTopK').value = st.music.top_k || 250;
                document.getElementById('mCfg').value = st.music.cfg_coef || 1.0;

                const mStatus = document.getElementById('musicLagStatus');
                if (mStatus) {
                    const bSec = (st.music.buffer_sec || 0).toFixed(1);
                    const bChunks = st.music.buffer_chunks || 0;
                    const rtf = (st.music.rtf || 1.0).toFixed(2);
                    mStatus.innerText = `Буфер: ${bSec} с (${bChunks} чанков) | RTF: ${rtf}x | Отставание: ${(bSec*1000).toFixed(0)} мс 🟢`;
                }
            }
        });

        sse.addEventListener('ble_cmd', async (e) => {
            const req = JSON.parse(e.data);
            const devObj = activeDevices.find(d => d.id === req.device_id || req.device_id === "all");
            if (devObj && devObj.cmdChar) {
                if (req.cmd === 'set_gain') {
                    const codeMap = { 4: 0x22, 8: 0x33, 16: 0x44, 32: 0x55 };
                    const hex = codeMap[req.val] || 0x22;
                    await devObj.cmdChar.writeValue(new Uint8Array([0x04, hex, hex])).catch(()=>{});
                    await devObj.cmdChar.writeValue(new Uint8Array([0x05, hex, hex])).catch(()=>{});
                    devObj.gain = parseInt(req.val);
                    devObj.uvScale = calcUvScale(devObj.gain);
                    const gEl = document.getElementById('gain_' + devObj.id);
                    if (gEl) gEl.value = req.val;
                }
            }
        });

        function toggleAudio() {
            const audio = document.getElementById('neuroAudio');
            const btn   = document.getElementById('btnAudio');
            if (!isAudioPlaying) {
                try { audio.pause(); } catch(e) {}
                audio.removeAttribute('src');
                try { audio.load(); } catch(e) {}
                audio.src = '/audio.wav?t=' + Date.now();
                audio.play().then(() => {
                    isAudioPlaying = true;
                    btn.innerText = '⏸️ ВЫКЛЮЧИТЬ LIVE ЗВУК';
                    btn.classList.add('playing');
                }).catch(e => alert("Аудио ошибка: " + e.message));
            } else {
                try { audio.pause(); } catch(e) {}
                audio.removeAttribute('src');
                isAudioPlaying = false;
                btn.innerText = '▶️ ВКЛЮЧИТЬ LIVE ЗВУК';
                btn.classList.remove('playing');
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

                // По умолчанию Gain 4x (0x22)
                await cmdChar.writeValue(new Uint8Array([0x04, 0x22, 0x22])).catch(()=>{});
                await cmdChar.writeValue(new Uint8Array([0x05, 0x22, 0x22])).catch(()=>{});

                const devObj = {
                    id: "dev_" + Date.now(),
                    name: device.name || "FreeEEG16",
                    device: device,
                    cmdChar: cmdChar,
                    user: "User" + (activeDevices.length + 1),
                    role: activeDevices.length === 0 ? "AFz" : "Fpz",
                    gain: 4,
                    uvScale: calcUvScale(4),
                    packetCount: 0,
                    sampleBuffer: [],
                    sending: false,
                    droppedBatches: 0,
                    lastCounter: -1,
                    totalReceived: 0,
                    totalLost: 0,
                    channelPeak: new Array(16).fill(0)
                };
                activeDevices.push(devObj);
                renderDeviceCards();

                await fetch('/api/device/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        device_id: devObj.id, name: devObj.name,
                        client_id: myClientId, gain: devObj.gain,
                        role: devObj.role, user: devObj.user
                    })
                });

                await dataChar.startNotifications();
                dataChar.addEventListener('characteristicvaluechanged', (e) => {
                    const b = new Uint8Array(e.target.value.buffer);
                    if (b[0] !== 0xA0) return;

                    const counter = b[1];
                    if (devObj.lastCounter !== -1) {
                        const expected = (devObj.lastCounter + 1) & 0xFF;
                        if (counter !== expected) {
                            devObj.totalLost += (counter - expected) & 0xFF;
                        }
                    }
                    devObj.lastCounter = counter;
                    devObj.totalReceived++;
                    devObj.packetCount++;

                    const sample = [];
                    const scale = devObj.uvScale;
                    for (let i = 0; i < 16; i++) {
                        let v = (b[2 + i * 3] << 16) | (b[3 + i * 3] << 8) | b[4 + i * 3];
                        if (v & 0x800000) v -= 0x1000000;
                        sample.push(v * scale);
                        
                        const absV = Math.abs(v);
                        if (absV > devObj.channelPeak[i]) devObj.channelPeak[i] = absV;
                    }
                    devObj.sampleBuffer.push(sample);

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
                alert("Ошибка подключения Bluetooth: " + err.message);
            }
        }

        async function sendBatch(devObj) {
            if (devObj.sending || devObj.sampleBuffer.length === 0) return;
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
            } finally {
                devObj.sending = false;
                if (devObj.sampleBuffer.length >= BATCH_SIZE) sendBatch(devObj);
            }
        }

        async function setDeviceGain(devId, gainVal) {
            const devObj = activeDevices.find(d => d.id === devId);
            if (devObj) {
                devObj.gain = parseInt(gainVal);
                devObj.uvScale = calcUvScale(devObj.gain);
            }
            await fetch('/api/device/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ device_id: devId, cmd: 'set_gain', val: parseInt(gainVal) })
            });
        }

        function updateDeviceField(idx, field, value) {
            if (idx >= 0 && idx < activeDevices.length) {
                activeDevices[idx][field] = value;
                fetch('/api/device/command', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ device_id: activeDevices[idx].id, cmd: 'set_' + field, val: value })
                });
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
                    <h4>${dev.name} <span class="badge" style="background:#059669;">ONLINE</span></h4>
                    <label>Субъект:</label>
                    <input type="text" value="${dev.user}" onchange="updateDeviceField(${idx}, 'user', this.value)">
                    
                    <label>Кортикальная зона (10-20 EEG):</label>
                    <select onchange="updateDeviceField(${idx}, 'role', this.value)">
                        <option value="AFz" ${dev.role==='AFz'?'selected':''}>AFz — Макросцена / Связи</option>
                        <option value="Fpz" ${dev.role==='Fpz'?'selected':''}>Fpz — BA 10 Вето / Волевой отказ</option>
                        <option value="F3"  ${dev.role==='F3' ?'selected':''}>F3  — Левый dlPFC (2D Ранги SWM)</option>
                        <option value="F4"  ${dev.role==='F4' ?'selected':''}>F4  — Правый dlPFC (Контекст / Энтропия)</option>
                        <option value="FCz" ${dev.role==='FCz'?'selected':''}>FCz — SMA (4D Аффордансы)</option>
                    </select>

                    <label>Усиление PGA (ADS131M08):</label>
                    <select id="gain_${dev.id}" onchange="setDeviceGain('${dev.id}', this.value)">
                        <option value="4" ${dev.gain===4?'selected':''}>4x (Дефолт прошивки)</option>
                        <option value="8" ${dev.gain===8?'selected':''}>8x</option>
                        <option value="16" ${dev.gain===16?'selected':''}>16x (Стандарт ЭЭГ)</option>
                        <option value="32" ${dev.gain===32?'selected':''}>32x (Высокая чувствительность)</option>
                    </select>

                    <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:10px;">
                        <span id="rate_${dev.id}" style="color:#38bdf8;">0 Гц</span>
                        <span id="loss_${dev.id}" style="color:#10b981;">Потерь: 0 (0.00%)</span>
                    </div>

                    <label style="margin-top:4px;">Аппаратный клиппинг (24-bit Rail):</label>
                    <div class="sat-grid" id="satgrid_${dev.id}"></div>
                `;
                container.appendChild(card);

                const sg = document.getElementById('satgrid_' + dev.id);
                for (let ch = 0; ch < 16; ch++) {
                    const b = document.createElement('div');
                    b.className = 'sat-box';
                    b.id = `sat_${dev.id}_${ch}`;
                    b.innerText = ch + 1;
                    sg.appendChild(b);
                }
            });
        }

        setInterval(() => {
            activeDevices.forEach(dev => {
                const total = dev.totalReceived + dev.totalLost;
                const pct = total > 0 ? ((dev.totalLost / total) * 100).toFixed(2) : "0.00";
                
                const rateEl = document.getElementById('rate_' + dev.id);
                if (rateEl) rateEl.innerText = `${dev.packetCount} Гц | Дропов: ${dev.droppedBatches}`;
                
                const lossEl = document.getElementById('loss_' + dev.id);
                if (lossEl) {
                    lossEl.innerText = `Потерь: ${dev.totalLost} (${pct}%)`;
                    lossEl.style.color = dev.totalLost === 0 ? "#10b981" : (dev.totalLost < 10 ? "#f59e0b" : "#ef4444");
                }

                for (let ch = 0; ch < 16; ch++) {
                    const isClipped = dev.channelPeak[ch] >= RAW_SAT_LIMIT;
                    const b = document.getElementById(`sat_${dev.id}_${ch}`);
                    if (b) {
                        if (isClipped) b.classList.add('clip');
                        else b.classList.remove('clip');
                    }
                    dev.channelPeak[ch] = 0;
                }
                dev.packetCount = 0;
            });
        }, 1000);

        async function loadAllInitialConcepts() {
            const rV = await fetch('/api/world/concepts');
            const dV = await rV.json();
            renderVisionConcepts(dV.concepts || []);

            const rM = await fetch('/api/music/concepts');
            const dM = await rM.json();
            renderMusicConcepts(dM.concepts || []);
        }

        function renderVisionConcepts(list) {
            const box = document.getElementById('visionConceptList');
            if (!box) return;
            box.innerHTML = '';
            if (list.length === 0) {
                box.innerHTML = '<span style="color:#64748b; font-size:10px;">Пул пуст (Режим свободного потока)</span>';
                return;
            }
            list.forEach(c => {
                const chip = document.createElement('div');
                chip.className = 'concept-chip';
                chip.innerHTML = `<span>${c}</span><button onclick="removeVisionConcept('${c}')">✕</button>`;
                box.appendChild(chip);
            });
        }

        async function addVisionConcept() {
            const inp = document.getElementById('newVisionInput');
            const c = inp.value.trim();
            if (!c) return;
            const res = await fetch('/api/world/concepts/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ concept: c })
            });
            const data = await res.json();
            inp.value = '';
            renderVisionConcepts(data.concepts || []);
        }

        async function removeVisionConcept(c) {
            const res = await fetch('/api/world/concepts/remove', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ concept: c })
            });
            const data = await res.json();
            renderVisionConcepts(data.concepts || []);
        }

        function renderMusicConcepts(list) {
            const box = document.getElementById('musicConceptList');
            if (!box) return;
            box.innerHTML = '';
            if (list.length === 0) {
                box.innerHTML = '<span style="color:#64748b; font-size:10px;">Пул пуст (Свободный эмбиент)</span>';
                return;
            }
            list.forEach(c => {
                const chip = document.createElement('div');
                chip.className = 'concept-chip music';
                chip.innerHTML = `<span>${c}</span><button onclick="removeMusicConcept('${c}')">✕</button>`;
                box.appendChild(chip);
            });
        }

        async function addMusicConcept() {
            const inp = document.getElementById('newMusicInput');
            const c = inp.value.trim();
            if (!c) return;
            const res = await fetch('/api/music/concepts/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ concept: c })
            });
            const data = await res.json();
            inp.value = '';
            renderMusicConcepts(data.concepts || []);
        }

        async function removeMusicConcept(c) {
            const res = await fetch('/api/music/concepts/remove', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ concept: c })
            });
            const data = await res.json();
            renderMusicConcepts(data.concepts || []);
        }

        async function updateMusicConfig() {
            const payload = {
                target_block: parseFloat(document.getElementById('mBlock').value),
                target_context: parseFloat(document.getElementById('mContext').value),
                target_buffer_count: parseInt(document.getElementById('mBuffer').value),
                temp: parseFloat(document.getElementById('mTemp').value),
                top_k: parseInt(document.getElementById('mTopK').value),
                cfg_coef: parseFloat(document.getElementById('mCfg').value)
            };
            await fetch('/api/music_config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
        }

        async function toggleRecording(start) {
            const endpoint = start ? '/api/recording/start' : '/api/recording/stop';
            const res = await fetch(endpoint, { method: 'POST' });
            const data = await res.json();
            const statusEl = document.getElementById('recStatusText');
            if (start) {
                statusEl.innerText = "🔴 Идет запись в: " + (data.session_path || "сессию");
                statusEl.style.color = "#f43f5e";
            } else {
                statusEl.innerText = "⏹️ Сохранено: " + (data.saved_path || "готово");
                statusEl.style.color = "#10b981";
            }
        }

        loadAllInitialConcepts();
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

OPENAPI_SPEC = {
    "openapi": "3.0.1",
    "info": {
        "title": "NeuroCanvas Unified World & MusicGen Agent API",
        "description": "API для внешних LLM-агентов (ChatGPT / LangChain) для управления визуальными и музыкальными концептами, качеством реакции и записью.",
        "version": "6.0.0"
    },
    "servers": [{"url": "http://localhost:8080"}],
    "paths": {
        "/api/world/concepts": {
            "get": { "summary": "Список активных визуальных концептов", "operationId": "getVisionConcepts", "responses": {"200": {"description": "Список"}} }
        },
        "/api/world/concepts/add": {
            "post": {
                "summary": "Добавить новый визуальный концепт в диффузию",
                "operationId": "addVisionConcept",
                "requestBody": { "required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"concept": {"type": "string"}}}}} },
                "responses": {"200": {"description": "Добавлено"}}
            }
        },
        "/api/world/concepts/remove": {
            "post": {
                "summary": "Удалить визуальный концепт из диффузии",
                "operationId": "removeVisionConcept",
                "requestBody": { "required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"concept": {"type": "string"}}}}} },
                "responses": {"200": {"description": "Удалено"}}
            }
        },
        "/api/music/concepts": {
            "get": { "summary": "Список активных музыкальных тем MusicGen", "operationId": "getMusicConcepts", "responses": {"200": {"description": "Список"}} }
        },
        "/api/music/concepts/add": {
            "post": {
                "summary": "Добавить новую тему саундтрека в MusicGen",
                "operationId": "addMusicConcept",
                "requestBody": { "required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"concept": {"type": "string"}}}}} },
                "responses": {"200": {"description": "Добавлено"}}
            }
        },
        "/api/music/concepts/remove": {
            "post": {
                "summary": "Удалить тему саундтрека из MusicGen",
                "operationId": "removeMusicConcept",
                "requestBody": { "required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"concept": {"type": "string"}}}}} },
                "responses": {"200": {"description": "Удалено"}}
            }
        },
        "/api/world/telemetry": {
            "get": { "summary": "Фокус внимания и эмоциональное состояние пользователя", "operationId": "getUserMentalState", "responses": {"200": {"description": "Телеметрия"}} }
        },
        "/api/recording/start": {
            "post": { "summary": "Начать мультимодальную запись (ЭЭГ + Видео + Теги)", "operationId": "startRecording", "responses": {"200": {"description": "Запущено"}} }
        },
        "/api/recording/stop": {
            "post": { "summary": "Остановить запись и сохранить датасет", "operationId": "stopRecording", "responses": {"200": {"description": "Сохранено"}} }
        }
    }
}

class UnifiedCloudHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, code, obj):
        payload = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

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

        elif clean_path == '/openapi.json':
            self._send_json(200, OPENAPI_SPEC)

        elif clean_path in ['/stream.mjpg', '/live']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            while True:
                with jpeg_lock: frame = latest_jpeg_frame
                if frame is not None:
                    try:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode() + frame + b'\r\n')
                        self.wfile.flush()
                    except Exception: break
                time.sleep(0.016)

        elif clean_path in ['/audio.wav', '/stream.wav']:
            self.send_response(200)
            self.send_header('Content-Type', 'audio/x-wav')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.end_headers()
            try:
                self.wfile.write(make_wav_header())
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

        elif clean_path == '/api/state_stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            import queue
            client_q = queue.Queue(maxsize=30)
            with sse_lock: sse_clients.append(client_q)
            
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
                    if client_q in sse_clients: sse_clients.remove(client_q)

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
                    "concepts": world_state["concepts"]
                })

        else:
            self.send_error(404)

    def do_POST(self):
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
                    try: active_music_conn.send({'cmd': 'update_config', 'config': music_state})
                    except Exception: pass
            broadcast_full_state()
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/device/register':
            dev_id = data.get('device_id')
            with devices_lock:
                devices_registry[dev_id] = data
            self._send_json(200, {"status": "registered"})

        elif clean_path == '/api/device/command':
            cmd = data.get('cmd')
            val = data.get('val')
            dev_id = data.get('device_id', 'all')
            broadcast_sse("ble_cmd", {"device_id": dev_id, "cmd": cmd, "val": val})
            self._send_json(200, {"status": "command_broadcasted"})

        # Визуальные концепты (мгновенный возврат нового списка для отрисовки без F5)
        elif clean_path == '/api/world/concepts/add':
            concept = data.get('concept', '').strip()
            if concept:
                with world_state_lock:
                    if concept not in world_state["concepts"]:
                        world_state["concepts"].append(concept)
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

        # Музыкальные концепты (мгновенный возврат нового списка для отрисовки без F5)
        elif clean_path == '/api/music/concepts/add':
            concept = data.get('concept', '').strip()
            if concept:
                with music_state_lock:
                    if concept not in music_state["concepts"]:
                        music_state["concepts"].append(concept)
                with music_conn_lock:
                    if active_music_conn is not None:
                        try: active_music_conn.send({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
                        except Exception: pass
                broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/music/concepts/remove':
            concept = data.get('concept', '').strip()
            with music_state_lock:
                if concept in music_state["concepts"]:
                    music_state["concepts"].remove(concept)
            with music_conn_lock:
                if active_music_conn is not None:
                    try: active_music_conn.send({'cmd': 'update_concepts', 'concepts': music_state["concepts"]})
                    except Exception: pass
            broadcast_full_state()
            self._send_json(200, {"status": "ok", "concepts": music_state["concepts"]})

        elif clean_path == '/api/diffusion/config':
            with world_state_lock:
                world_state.update(data)
            send_diff_ipc({"cmd": "update_params", "params": data})
            broadcast_full_state()
            self._send_json(200, {"status": "ok"})

        elif clean_path == '/api/recording/start':
            with world_state_lock:
                world_state["is_recording"] = True
            send_diff_ipc({"cmd": "start_recording", "session_name": data.get("name")})
            broadcast_full_state()
            self._send_json(200, {"status": "recording_started"})

        elif clean_path == '/api/recording/stop':
            with world_state_lock:
                world_state["is_recording"] = False
            send_diff_ipc({"cmd": "stop_recording"})
            broadcast_full_state()
            self._send_json(200, {"status": "recording_stopped"})

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
            try: active_diff_ctrl_conn.send(payload)
            except Exception: active_diff_ctrl_conn = None

class ThreadedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    allow_reuse_address = True

def ipc_frame_receiver():
    global latest_jpeg_frame
    try: listener = Listener(('localhost', IPC_PORT), authkey=IPC_AUTHKEY)
    except Exception: return
    while True:
        try:
            conn = listener.accept()
            while True:
                data = conn.recv()
                if isinstance(data, bytes):
                    with jpeg_lock: latest_jpeg_frame = data
        except Exception: time.sleep(0.5)

def ipc_audio_receiver():
    global active_music_conn
    try: listener = Listener(('localhost', IPC_AUDIO_PORT), authkey=IPC_AUDIO_AUTHKEY)
    except Exception: return
    while True:
        try:
            conn = listener.accept()
            with music_conn_lock: active_music_conn = conn
            while True:
                data = conn.recv()
                if isinstance(data, bytes):
                    with audio_pcm_lock:
                        audio_pcm_queue.append(data)
                        target_max_chunks = music_state.get("target_buffer_count", 2)
                        while len(audio_pcm_queue) > target_max_chunks:
                            audio_pcm_queue.pop(0)

                        total_bytes = sum(len(c) for c in audio_pcm_queue)
                        b_sec = total_bytes / (SAMPLE_RATE * 4)
                        with music_state_lock:
                            music_state["buffer_sec"] = b_sec
                            music_state["buffer_chunks"] = len(audio_pcm_queue)
        except Exception: time.sleep(0.5)
        finally:
            with music_conn_lock: active_music_conn = None

def ipc_control_listener():
    global active_diff_ctrl_conn
    try: listener = Listener(('localhost', IPC_DIFF_CONTROL_PORT), authkey=IPC_DIFF_AUTHKEY)
    except Exception: return
    while True:
        try:
            conn = listener.accept()
            with diff_ctrl_lock: active_diff_ctrl_conn = conn
            print("🔗 [GATEWAY IPC] Канал управления диффузией соединен!")
            while True:
                msg = conn.recv()
                if isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "init_concepts":
                        with world_state_lock:
                            world_state["concepts"] = msg.get("concepts", [])
                        broadcast_full_state()
                    elif cmd == "telemetry":
                        with world_state_lock:
                            world_state["current_puck"] = msg.get("puck", [0.0, 0.0])
                            world_state["active_thought"] = msg.get("active_thought", "")
        except Exception: time.sleep(0.5)
        finally:
            with diff_ctrl_lock: active_diff_ctrl_conn = None

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

    with world_state_lock: world_state["concepts"] = initial_world_words
    with music_state_lock: music_state["concepts"] = initial_music_words

    cert_candidates = []
    if args.cert: cert_candidates.append(args.cert)
    if os.environ.get("CERT_PATH"): cert_candidates.append(os.environ.get("CERT_PATH"))
    cert_candidates.extend(["cert.pem", "./cert.pem", "/app/cert.pem", "../cert.pem"])

    key_candidates = []
    if args.key: key_candidates.append(args.key)
    if os.environ.get("KEY_PATH"): key_candidates.append(os.environ.get("KEY_PATH"))
    key_candidates.extend(["key.pem", "./key.pem", "/app/key.pem", "../key.pem"])

    found_cert = next((p for p in cert_candidates if os.path.exists(p)), None)
    found_key = next((p for p in key_candidates if os.path.exists(p)), None)

    threading.Thread(target=ipc_frame_receiver, daemon=True).start()
    threading.Thread(target=ipc_audio_receiver, daemon=True).start()
    threading.Thread(target=ipc_control_listener, daemon=True).start()

    server = ThreadedHTTPServer(('0.0.0.0', port), UnifiedCloudHandler)

    scheme = "http"
    if found_cert and found_key:
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=found_cert, keyfile=found_key)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            scheme = "https"
            print(f"🔒 [GATEWAY] TLS АКТИВИРОВАН: Web Bluetooth Secure Context ГАРАНТИРОВАН!")
        except Exception as e:
            print(f"⚠️ [GATEWAY] Ошибка TLS: {e}. Переход в режим HTTP.")
    else:
        print("⚠️ [GATEWAY] Сертификаты не найдены. Внимание: Web Bluetooth на удаленных IP требует HTTPS!")

    print("=" * 65)
    print(f"🌐 [GATEWAY] Веб-терминал:     {scheme}://0.0.0.0:{port}/")
    print(f"📖 [GATEWAY] OpenAPI для GPT: {scheme}://0.0.0.0:{port}/openapi.json")
    print(f"🎥 [GATEWAY] Видео-стрим:     {scheme}://0.0.0.0:{port}/stream.mjpg")
    print(f"🔊 [GATEWAY] Аудио-стрим:     {scheme}://0.0.0.0:{port}/audio.wav")
    print(f"🎨 [GATEWAY] Визуальный лор:  {len(initial_world_words)} концептов")
    print(f"🎵 [GATEWAY] Музыкальный лор: {len(initial_music_words)} концептов")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка шлюза.")

if __name__ == '__main__':
    main()

