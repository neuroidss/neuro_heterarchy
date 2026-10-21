const SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
const DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
const CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8";
const BATCH_SIZE     = 40;
const RAW_SAT_LIMIT  = 8300000;
const AUDIO_SR       = 32000;

function calcUvScale(gain) {
    return (1.2 / parseFloat(gain) / 8388607.0) * 1e6;
}

let activeDevices = [];
let isAudioPlaying = false;
let audioCtx = null;
let audioAbortCtrl = null;
const myClientId = "client_" + Math.random().toString(36).substring(2, 9);
let isProbeMode = false;

const dom = {
    videoStreamImg: document.getElementById('videoStreamImg'),
    canvasBox: document.getElementById('canvasBox'),
    fsToggleBtn: document.getElementById('fsToggleBtn'),
    probeModeToggle: document.getElementById('probeModeToggle'),
    btnConnectBle: document.getElementById('btnConnectBle'),
    deviceList: document.getElementById('deviceList'),
    noDevicesMsg: document.getElementById('noDevicesMsg'),

    ladderStageSelect: document.getElementById('ladderStageSelect'),
    montyStatusBar: document.getElementById('montyStatusBar'),
    
    visionConceptList: document.getElementById('visionConceptList'),
    newVisionInput: document.getElementById('newVisionInput'),
    btnAddVision: document.getElementById('btnAddVision'),
    diffPreset: document.getElementById('diffPreset'),
    diffSteps: document.getElementById('diffSteps'),
    diffCfg: document.getElementById('diffCfg'),
    diffStrLow: document.getElementById('diffStrLow'),
    diffStrHigh: document.getElementById('diffStrHigh'),
    diffEmaSmooth: document.getElementById('diffEmaSmooth'),
    
    btnAudio: document.getElementById('btnAudio'),
    neuroAudio: document.getElementById('neuroAudio'),
    musicLagStatus: document.getElementById('musicLagStatus'),
    musicConceptList: document.getElementById('musicConceptList'),
    newMusicInput: document.getElementById('newMusicInput'),
    btnAddMusic: document.getElementById('btnAddMusic'),
    musicPreset: document.getElementById('musicPreset'),
    mStrLow: document.getElementById('mStrLow'),
    mStrHigh: document.getElementById('mStrHigh'),
    mEmaSmooth: document.getElementById('mEmaSmooth'),
    mBlock: document.getElementById('mBlock'),
    mContext: document.getElementById('mContext'),
    mBuffer: document.getElementById('mBuffer'),
    mTemp: document.getElementById('mTemp'),
    mTopK: document.getElementById('mTopK'),
    mCfg: document.getElementById('mCfg'),
    
    btnRecStart: document.getElementById('btnRecStart'),
    btnRecStop: document.getElementById('btnRecStop'),
    recStatusText: document.getElementById('recStatusText'),
    recBadge: document.getElementById('recBadge'),
    
    deviceCardTemplate: document.getElementById('deviceCardTemplate'),
    conceptChipTemplate: document.getElementById('conceptChipTemplate')
};

function toggleProbeMode(enabled) {
    isProbeMode = enabled;
    if (isProbeMode) {
        dom.videoStreamImg.src = "";
        dom.videoStreamImg.style.display = "none";
        if (isAudioPlaying) toggleAudio();
    } else {
        dom.videoStreamImg.style.display = "block";
        dom.videoStreamImg.src = "/stream.mjpg";
    }
}

function toggleFullScreen() {
    if (!document.fullscreenElement) {
        dom.canvasBox.requestFullscreen().catch(err => alert("Fullscreen error: " + err.message));
    } else {
        document.exitFullscreen();
    }
}

function readRegister(devObj, reg) {
    return new Promise((resolve) => {
        devObj.pendingReadReg = reg;
        devObj.pendingReadResolve = resolve;
        devObj.cmdChar.writeValue(new Uint8Array([reg])).catch(() => {
            devObj.pendingReadResolve = null;
            resolve(-1);
        });
        setTimeout(() => {
            if (devObj.pendingReadResolve) {
                devObj.pendingReadResolve = null;
                resolve(-1);
            }
        }, 350);
    });
}

function decodeGainFromRegister(rawVal) {
    if (rawVal < 0) return null;
    const b = rawVal & 0xFF;
    if (b === 0x22) return 4;
    if (b === 0x33) return 8;
    if (b === 0x44) return 16;
    if (b === 0x55) return 32;
    if (b === 0x11) return 2;
    if (b === 0x00) return 1;
    return null;
}

async function writeAndVerifyGain(devObj, targetGain, maxRetries = 8) {
    const codeMap = { 4: 0x22, 8: 0x33, 16: 0x44, 32: 0x55 };
    const hex = codeMap[targetGain] || 0x22;
    const expected = (hex << 8) | hex;

    if (devObj.domRefs) {
        devObj.domRefs.gainBadge.textContent = "⏳ Запись...";
        devObj.domRefs.gainBadge.style.color = "#f59e0b";
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        await devObj.cmdChar.writeValue(new Uint8Array([0x04, hex, hex])).catch(() => {});
        await new Promise(r => setTimeout(r, 45));
        const val1 = await readRegister(devObj, 0x04);

        await devObj.cmdChar.writeValue(new Uint8Array([0x05, hex, hex])).catch(() => {});
        await new Promise(r => setTimeout(r, 45));
        const val2 = await readRegister(devObj, 0x05);

        if (val1 === expected && val2 === expected) {
            devObj.gain = targetGain;
            devObj.uvScale = calcUvScale(targetGain);
            if (devObj.domRefs) {
                devObj.domRefs.gainBadge.textContent = `✓ Железо подтвердило (${targetGain}x)`;
                devObj.domRefs.gainBadge.style.color = "#10b981";
                devObj.domRefs.gainSelect.value = targetGain;
            }
            return true;
        }
        await new Promise(r => setTimeout(r, 60));
    }

    const actualRaw = await readRegister(devObj, 0x04);
    const actualGain = decodeGainFromRegister(actualRaw);
    if (actualGain) {
        devObj.gain = actualGain;
        devObj.uvScale = calcUvScale(actualGain);
        if (devObj.domRefs) devObj.domRefs.gainSelect.value = actualGain;
    }
    if (devObj.domRefs) {
        devObj.domRefs.gainBadge.textContent = `⚠️ Ошибка! В чипе: ${devObj.gain}x`;
        devObj.domRefs.gainBadge.style.color = "#ef4444";
    }
    return false;
}

const PRESETS = {
    meditation: { steps: 2, cfg: 1.0, strLow: 0.45, strHigh: 0.70, ema: 0.30 },
    balanced:   { steps: 2, cfg: 1.0, strLow: 0.50, strHigh: 0.85, ema: 0.14 },
    dynamic:    { steps: 3, cfg: 1.2, strLow: 0.60, strHigh: 0.92, EMA: 0.08 }
};

const MUSIC_PRESETS = {
    meditation: { block: 1.8, ctx: 3.6, strLow: 0.12, strHigh: 0.45, ema: 0.35, temp: 0.95 },
    balanced:   { block: 1.8, ctx: 2.4, strLow: 0.25, strHigh: 0.75, ema: 0.20, temp: 1.0 },
    dynamic:    { block: 1.4, ctx: 1.6, strLow: 0.35, strHigh: 0.92, ema: 0.10, temp: 1.05 }
};

function applyPreset(presetKey) {
    if (presetKey === 'custom') return;
    const p = PRESETS[presetKey];
    if (!p) return;
    dom.diffSteps.value = p.steps;
    dom.diffCfg.value = p.cfg;
    dom.diffStrLow.value = p.strLow;
    dom.diffStrHigh.value = p.strHigh;
    dom.diffEmaSmooth.value = p.ema || 0.08;
    sendDiffConfig(p.steps, p.cfg, p.strLow, p.strHigh, p.ema || 0.08, presetKey);
}

function applyMusicPreset(presetKey) {
    if (presetKey === 'custom') return;
    const p = MUSIC_PRESETS[presetKey];
    if (!p) return;
    dom.mBlock.value = p.block;
    dom.mContext.value = p.ctx;
    dom.mStrLow.value = p.strLow;
    dom.mStrHigh.value = p.strHigh;
    dom.mEmaSmooth.value = p.ema;
    dom.mTemp.value = p.temp;
    updateMusicConfig();
}

function onManualParamChange() {
    dom.diffPreset.value = 'custom';
    const steps = Math.max(1, Math.min(6, parseInt(dom.diffSteps.value) || 2));
    const cfg = Math.max(0.0, Math.min(3.0, parseFloat(dom.diffCfg.value) || 1.0));
    let sLow = Math.max(0.35, Math.min(0.75, parseFloat(dom.diffStrLow.value) || 0.50));
    let sHigh = Math.max(sLow + 0.05, Math.min(0.95, parseFloat(dom.diffStrHigh.value) || 0.85));
    const ema = Math.max(0.04, Math.min(0.50, parseFloat(dom.diffEmaSmooth.value) || 0.14));

    dom.diffSteps.value = steps;
    dom.diffCfg.value = cfg;
    dom.diffStrLow.value = sLow;
    dom.diffStrHigh.value = sHigh;
    dom.diffEmaSmooth.value = ema;

    sendDiffConfig(steps, cfg, sLow, sHigh, ema, 'custom');
}

async function sendDiffConfig(steps, cfg, sLow, sHigh, ema, presetKey) {
    await fetch('/api/diffusion/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            preset: presetKey,
            steps: steps,
            guidance_scale: cfg,
            strength_low: sLow,
            strength_high: sHigh,
            ema_smooth: ema
        })
    });
}

async function onLadderStageChange() {
    const stg = parseInt(dom.ladderStageSelect.value) || 4;
    await fetch('/api/preset/ladder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ stage: stg })
    });
}

const sse = new EventSource('/api/state_stream');

sse.addEventListener('state', (e) => {
    const st = JSON.parse(e.data);
    if (st.world) {
        renderVisionConcepts(st.world.concepts || []);
        dom.diffSteps.value = st.world.steps || 2;
        dom.diffCfg.value = st.world.guidance_scale || 1.0;
        dom.diffStrLow.value = st.world.strength_low || 0.50;
        dom.diffStrHigh.value = st.world.strength_high || 0.85;
        dom.diffEmaSmooth.value = st.world.ema_smooth || 0.14;
        if (st.world.preset) dom.diffPreset.value = st.world.preset;
        if (st.world.ladder_stage) dom.ladderStageSelect.value = st.world.ladder_stage;

        const mLoss = (st.world.monty_loss || 0.0).toFixed(4);
        const pers = (st.world.persistence || 0.0).toFixed(2);
        const surp = (st.world.surprise || 0.0).toFixed(2);
        if (dom.montyStatusBar) {
            dom.montyStatusBar.textContent = `Loss (MSE): ${mLoss} | Persistence: ${pers} | Surprise: ${surp}`;
        }

        if (st.world.is_recording) {
            dom.recBadge.style.background = "#e11d48";
            dom.recBadge.textContent = "ЗАПИСЬ: АКТИВНА";
        } else {
            dom.recBadge.style.background = "#475569";
            dom.recBadge.textContent = "ЗАПИСЬ: ВЫКЛ";
        }
    }
    if (st.music) {
        renderMusicConcepts(st.music.concepts || []);
        if (document.activeElement !== dom.mBlock) dom.mBlock.value = st.music.target_block ?? 1.8;
        if (document.activeElement !== dom.mContext) dom.mContext.value = st.music.target_context ?? 2.4;
        if (document.activeElement !== dom.mBuffer) dom.mBuffer.value = st.music.target_buffer_count ?? 2;
        if (document.activeElement !== dom.mTemp) dom.mTemp.value = st.music.temp ?? 1.0;
        if (document.activeElement !== dom.mTopK) dom.mTopK.value = st.music.top_k ?? 250;
        if (document.activeElement !== dom.mCfg) dom.mCfg.value = st.music.cfg_coef ?? 1.0;
        if (document.activeElement !== dom.mStrLow) dom.mStrLow.value = st.music.strength_low ?? 0.25;
        if (document.activeElement !== dom.mStrHigh) dom.mStrHigh.value = st.music.strength_high ?? 0.75;
        if (document.activeElement !== dom.mEmaSmooth) dom.mEmaSmooth.value = st.music.ema_smooth ?? 0.20;

        const bSec = (st.music.buffer_sec || 0).toFixed(1);
        const bChunks = st.music.buffer_chunks || 0;
        const rtf = (st.music.rtf || 1.0).toFixed(2);
        const dStr = (st.music.dyn_strength || 0.25).toFixed(2);
        dom.musicLagStatus.textContent = `Буфер: ${bSec} с (${bChunks} чанков) | RTF: ${rtf}x | Сила WM (DynStr): ${dStr} 🟢`;
    }
});

sse.addEventListener('ble_cmd', async (e) => {
    const req = JSON.parse(e.data);
    const devObj = activeDevices.find(d => d.id === req.device_id || req.device_id === "all");
    if (!devObj) return;
    if (req.cmd === 'set_gain' && devObj.cmdChar) {
        await writeAndVerifyGain(devObj, parseInt(req.val));
    } else if (req.cmd === 'set_role') {
        devObj.role = req.val;
        if (devObj.domRefs) devObj.domRefs.roleSelect.value = req.val;
    } else if (req.cmd === 'set_user') {
        devObj.user = req.val;
        if (devObj.domRefs) devObj.domRefs.userInput.value = req.val;
    }
});

// =============================================================================
// БЕСШОВНЫЙ WEB AUDIO API ПЛЕЕР (УСТРАНЯЕТ МИКРОПАУЗЫ БРАУЗЕРА МЕЖДУ БЛОКАМИ)
// =============================================================================
async function startWebAudioStream() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass({ sampleRate: AUDIO_SR });
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

    audioAbortCtrl = new AbortController();
    const response = await fetch('/audio.wav?t=' + Date.now(), { signal: audioAbortCtrl.signal });
    const reader = response.body.getReader();

    let headerBytesRemaining = 44; // Пропускаем 44-байтный WAV-заголовок
    let leftoverByte = null;
    let nextPlayTime = 0;
    let pendingSamples = [];
    let hasStarted = false;
    const prebufferSamples = Math.floor(AUDIO_SR * 0.45); // 0.45с стартовый запас против микропауз
    const sliceSamples = Math.floor(AUDIO_SR * 0.16);     // Планируем по 160 мс

    function scheduleSlice(float32Arr) {
        if (!audioCtx || !isAudioPlaying) return;
        const now = audioCtx.currentTime;
        if (nextPlayTime < now + 0.02) {
            // Если вдруг буфер просел — ставим мягкий запас 180 мс без щелчка
            nextPlayTime = now + 0.18;
        }
        const ahead = nextPlayTime - now;
        // Плавная автоподстройка скорости в пределах ±2%, чтобы буфер никогда не пустел и не рос
        let rate = 1.0;
        if (ahead < 0.28) rate = 0.975;
        else if (ahead > 2.2) rate = 1.03;

        const audioBuf = audioCtx.createBuffer(1, float32Arr.length, AUDIO_SR);
        audioBuf.getChannelData(0).set(float32Arr);

        const src = audioCtx.createBufferSource();
        src.buffer = audioBuf;
        src.playbackRate.value = 1.0;
        src.connect(audioCtx.destination);
        src.start(nextPlayTime);
        nextPlayTime += (float32Arr.length / AUDIO_SR);
    }

    (async () => {
        try {
            while (isAudioPlaying) {
                const { done, value } = await reader.read();
                if (done || !isAudioPlaying) break;
                let chunk = value;

                if (headerBytesRemaining > 0) {
                    if (chunk.byteLength <= headerBytesRemaining) {
                        headerBytesRemaining -= chunk.byteLength;
                        continue;
                    } else {
                        chunk = chunk.subarray(headerBytesRemaining);
                        headerBytesRemaining = 0;
                    }
                }

                if (leftoverByte !== null) {
                    const merged = new Uint8Array(chunk.byteLength + 1);
                    merged[0] = leftoverByte;
                    merged.set(chunk, 1);
                    chunk = merged;
                    leftoverByte = null;
                }

                if (chunk.byteLength % 2 !== 0) {
                    leftoverByte = chunk[chunk.byteLength - 1];
                    chunk = chunk.subarray(0, chunk.byteLength - 1);
                }

                const numSamples = chunk.byteLength >> 1;
                if (numSamples === 0) continue;

                const view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
                for (let i = 0; i < numSamples; i++) {
                    const s16 = view.getInt16(i * 2, true);
                    pendingSamples.push(s16 / 32768.0);
                }

                if (!hasStarted) {
                    if (pendingSamples.length >= prebufferSamples) {
                        hasStarted = true;
                        nextPlayTime = audioCtx.currentTime + 0.08;
                        while (pendingSamples.length >= sliceSamples) {
                            scheduleSlice(new Float32Array(pendingSamples.splice(0, sliceSamples)));
                        }
                    }
                } else {
                    while (pendingSamples.length >= sliceSamples) {
                        scheduleSlice(new Float32Array(pendingSamples.splice(0, sliceSamples)));
                    }
                }
            }
        } catch (err) {
            if (isAudioPlaying) console.warn("WebAudio stream interrupted:", err);
        }
    })();
}

function stopWebAudioStream() {
    if (audioAbortCtrl) {
        try { audioAbortCtrl.abort(); } catch (e) {}
        audioAbortCtrl = null;
    }
    if (audioCtx) {
        try { audioCtx.close(); } catch (e) {}
        audioCtx = null;
    }
}

function toggleAudio() {
    if (!isAudioPlaying) {
        isAudioPlaying = true;
        dom.btnAudio.textContent = '⏸️ ВЫКЛЮЧИТЬ LIVE ЗВУК';
        dom.btnAudio.classList.add('playing');
        startWebAudioStream().catch(e => {
            isAudioPlaying = false;
            dom.btnAudio.textContent = '▶️ ВКЛЮЧИТЬ LIVE ЗВУК';
            dom.btnAudio.classList.remove('playing');
            alert("Аудио ошибка: " + e.message);
        });
    } else {
        isAudioPlaying = false;
        stopWebAudioStream();
        dom.btnAudio.textContent = '▶️ ВКЛЮЧИТЬ LIVE ЗВУК';
        dom.btnAudio.classList.remove('playing');
    }
}

function createDeviceCardDOM(devObj) {
    const clone = dom.deviceCardTemplate.content.cloneNode(true);
    const cardEl = clone.querySelector('.device-card');
    
    const titleName = cardEl.querySelector('.device-title-name');
    const userInput = cardEl.querySelector('.device-user-input');
    const roleSelect = cardEl.querySelector('.device-role-select');
    const gainBadge = cardEl.querySelector('.gain-badge');
    const gainSelect = cardEl.querySelector('.device-gain-select');
    const rateText = cardEl.querySelector('.device-rate-text');
    const lossText = cardEl.querySelector('.device-loss-text');
    const satGrid = cardEl.querySelector('.sat-grid');

    titleName.textContent = devObj.name;
    userInput.value = devObj.user;
    roleSelect.value = devObj.role;
    gainSelect.value = devObj.gain;

    const satBoxes = [];
    for (let ch = 0; ch < 16; ch++) {
        const box = document.createElement('div');
        box.className = 'sat-box';
        box.textContent = (ch + 1).toString();
        satGrid.appendChild(box);
        satBoxes.push(box);
    }

    userInput.addEventListener('change', () => {
        devObj.user = userInput.value;
        sendDeviceCommand(devObj.id, 'set_user', devObj.user);
    });

    roleSelect.addEventListener('change', () => {
        devObj.role = roleSelect.value;
        sendDeviceCommand(devObj.id, 'set_role', devObj.role);
    });

    gainSelect.addEventListener('change', async () => {
        const targetGain = parseInt(gainSelect.value);
        await writeAndVerifyGain(devObj, targetGain);
        sendDeviceCommand(devObj.id, 'set_gain', targetGain);
    });

    devObj.domRefs = {
        cardEl, userInput, roleSelect, gainBadge,
        gainSelect, rateText, lossText, satBoxes
    };

    dom.deviceList.appendChild(cardEl);
    dom.noDevicesMsg.style.display = 'none';
}

async function sendDeviceCommand(deviceId, cmd, val) {
    await fetch('/api/device/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ device_id: deviceId, cmd: cmd, val: val })
    });
}

async function connectNewBleDevice() {
    try {
        const device = await navigator.bluetooth.requestDevice({
            filters: [{ services: [SERVICE_UUID] }]
        });
        const server   = await device.gatt.connect();
        const service  = await server.getPrimaryService(SERVICE_UUID);
        const dataChar = await service.getCharacteristic(DATA_CHAR_UUID);
        const cmdChar  = await service.getCharacteristic(CMD_CHAR_UUID);

        const devObj = {
            id: "dev_" + Date.now(),
            name: device.name || "FreeEEG16",
            device: device,
            cmdChar: cmdChar,
            user: "User" + (activeDevices.length + 1),
            role: activeDevices.length === 0 ? "AFz" : "FCz",
            gain: 4,
            uvScale: calcUvScale(4),
            packetCount: 0,
            sampleBuffer: [],
            sending: false,
            droppedBatches: 0,
            lastCounter: -1,
            totalReceived: 0,
            totalLost: 0,
            channelPeak: new Array(16).fill(0),
            pendingReadReg: -1,
            pendingReadResolve: null,
            domRefs: null
        };

        activeDevices.push(devObj);
        createDeviceCardDOM(devObj);

        await cmdChar.startNotifications();
        cmdChar.addEventListener('characteristicvaluechanged', (e) => {
            const b = new Uint8Array(e.target.value.buffer);
            if (b.length >= 3) {
                const reg = b[0];
                const val = (b[1] << 8) | b[2];
                if (devObj.pendingReadResolve && devObj.pendingReadReg === reg) {
                    devObj.pendingReadResolve(val);
                    devObj.pendingReadResolve = null;
                }
            }
        });

        await writeAndVerifyGain(devObj, 4);

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
            if (devObj.domRefs && devObj.domRefs.cardEl) {
                devObj.domRefs.cardEl.remove();
            }
            activeDevices = activeDevices.filter(d => d.id !== devObj.id);
            if (activeDevices.length === 0) {
                dom.noDevicesMsg.style.display = 'block';
            }
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

setInterval(() => {
    activeDevices.forEach(dev => {
        if (!dev.domRefs) return;
        const total = dev.totalReceived + dev.totalLost;
        const pct = total > 0 ? ((dev.totalLost / total) * 100).toFixed(2) : "0.00";

        dev.domRefs.rateText.textContent = `${dev.packetCount} Гц | Дропов: ${dev.droppedBatches}`;
        dev.domRefs.lossText.textContent = `Потерь: ${dev.totalLost} (${pct}%)`;
        dev.domRefs.lossText.style.color = dev.totalLost === 0 ? "#10b981" : (dev.totalLost < 10 ? "#f59e0b" : "#ef4444");

        for (let ch = 0; ch < 16; ch++) {
            const isClipped = dev.channelPeak[ch] >= RAW_SAT_LIMIT;
            const b = dev.domRefs.satBoxes[ch];
            if (b) {
                if (isClipped) b.classList.add('clip');
                else b.classList.remove('clip');
            }
            dev.channelPeak[ch] = 0;
        }
        dev.packetCount = 0;
    });
}, 1000);

function renderConceptList(containerEl, list, isMusic, onRemove) {
    containerEl.replaceChildren();
    if (list.length === 0) {
        const empty = document.createElement('span');
        empty.className = 'empty-concepts';
        empty.textContent = isMusic ? 'Пул пуст (Свободный дрейф по словарю T5)' : 'Пул пуст (Свободный латентный дрейф)';
        containerEl.appendChild(empty);
        return;
    }

    list.forEach(c => {
        const clone = dom.conceptChipTemplate.content.cloneNode(true);
        const chip = clone.querySelector('.concept-chip');
        if (isMusic) chip.classList.add('music');
        
        const txt = chip.querySelector('.concept-text');
        txt.textContent = c;

        const rmBtn = chip.querySelector('.concept-remove-btn');
        rmBtn.addEventListener('click', () => onRemove(c));

        containerEl.appendChild(chip);
    });
}

function renderVisionConcepts(list) {
    renderConceptList(dom.visionConceptList, list, false, removeVisionConcept);
}

function renderMusicConcepts(list) {
    renderConceptList(dom.musicConceptList, list, true, removeMusicConcept);
}

async function addVisionConcept() {
    const c = dom.newVisionInput.value.trim();
    if (!c) return;
    const res = await fetch('/api/world/concepts/add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ concept: c })
    });
    const data = await res.json();
    dom.newVisionInput.value = '';
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

async function addMusicConcept() {
    const c = dom.newMusicInput.value.trim();
    if (!c) return;
    const res = await fetch('/api/music/concepts/add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ concept: c })
    });
    const data = await res.json();
    dom.newMusicInput.value = '';
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
    let sLow = Math.max(0.05, Math.min(0.85, parseFloat(dom.mStrLow.value) || 0.25));
    let sHigh = Math.max(sLow + 0.05, Math.min(0.98, parseFloat(dom.mStrHigh.value) || 0.75));
    const ema = Math.max(0.04, Math.min(0.60, parseFloat(dom.mEmaSmooth.value) || 0.20));

    dom.mStrLow.value = sLow;
    dom.mStrHigh.value = sHigh;
    dom.mEmaSmooth.value = ema;

    await fetch('/api/music_config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            target_block: parseFloat(dom.mBlock.value) || 1.8,
            target_context: parseFloat(dom.mContext.value) || 2.4,
            target_buffer_count: parseInt(dom.mBuffer.value) || 2,
            temp: parseFloat(dom.mTemp.value) || 1.0,
            top_k: parseInt(dom.mTopK.value) || 250,
            cfg_coef: parseFloat(dom.mCfg.value) || 1.0,
            strength_low: sLow,
            strength_high: sHigh,
            ema_smooth: ema
        })
    });
}

async function toggleRecording(start) {
    const endpoint = start ? '/api/recording/start' : '/api/recording/stop';
    const res = await fetch(endpoint, { method: 'POST' });
    const data = await res.json();
    if (start) {
        dom.recStatusText.textContent = "🔴 Идет запись в: " + (data.session_path || "сессию");
        dom.recStatusText.style.color = "#f43f5e";
    } else {
        dom.recStatusText.textContent = "⏹️ Сохранено: " + (data.saved_path || "готово");
        dom.recStatusText.style.color = "#10b981";
    }
}

window.addEventListener('DOMContentLoaded', async () => {
    dom.fsToggleBtn.addEventListener('click', toggleFullScreen);
    dom.probeModeToggle.addEventListener('change', (e) => toggleProbeMode(e.target.checked));
    dom.btnConnectBle.addEventListener('click', connectNewBleDevice);
    dom.btnAudio.addEventListener('click', toggleAudio);
    
    dom.btnAddVision.addEventListener('click', addVisionConcept);
    dom.btnAddMusic.addEventListener('click', addMusicConcept);

    if (dom.ladderStageSelect) {
        dom.ladderStageSelect.addEventListener('change', onLadderStageChange);
    }

    dom.diffPreset.addEventListener('change', (e) => applyPreset(e.target.value));
    [dom.diffSteps, dom.diffCfg, dom.diffStrLow, dom.diffStrHigh, dom.diffEmaSmooth].forEach(inp => {
        inp.addEventListener('change', onManualParamChange);
    });

    if (dom.musicPreset) {
        dom.musicPreset.addEventListener('change', (e) => applyMusicPreset(e.target.value));
    }
    [dom.mStrLow, dom.mStrHigh, dom.mEmaSmooth, dom.mBlock, dom.mContext, dom.mBuffer, dom.mTemp, dom.mTopK, dom.mCfg].forEach(inp => {
        if (inp) {
            inp.addEventListener('change', () => {
                if (dom.musicPreset) dom.musicPreset.value = 'custom';
                updateMusicConfig();
            });
        }
    });

    dom.btnRecStart.addEventListener('click', () => toggleRecording(true));
    dom.btnRecStop.addEventListener('click', () => toggleRecording(false));

    try {
        const [rV, rM] = await Promise.all([
            fetch('/api/world/concepts'),
            fetch('/api/music/concepts')
        ]);
        const dV = await rV.json();
        const dM = await rM.json();
        renderVisionConcepts(dV.concepts || []);
        renderMusicConcepts(dM.concepts || []);
    } catch (err) {
        console.error("Ошибка начальной загрузки:", err);
    }
});
