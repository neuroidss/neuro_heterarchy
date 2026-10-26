const SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
const DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
const CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8";
const BATCH_SIZE = 15; // 60 мс вместо 160 мс — мгновенная доставка ЭЭГ в облако!
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

let videoStreamAbortCtrl = null;
let latestJpegBytes = null;
let latestFrameSeq = 0;
let renderedFrameSeq = 0;
let isImgDecoding = false;
let currentBlobUrl = null;
let clientRenderCount = 0;
let clientScreenFps = 0.0;
let clientDroppedFrames = 0;

// Локальный кэш порегионных секций 3D-DiT видео-мира
let cachedVideoSections = {
    FPZ: [], AFZ: [], F3: [], F4: [], FCZ: [], NEGATIVE: []
};
let cachedActiveConceptIdx = { Fpz: 0, AFz: 0, F3: 0, F4: 0, FCz: 0 };

const dom = {
    diffModel: document.getElementById('diffModel'),
    musicModel: document.getElementById('musicModel'),

    videoStreamImg: document.getElementById('videoStreamImg'),
    canvasBox: document.getElementById('canvasBox'),
    fsToggleBtn: document.getElementById('fsToggleBtn'),
    probeModeToggle: document.getElementById('probeModeToggle'),
    btnConnectBle: document.getElementById('btnConnectBle'),
    deviceList: document.getElementById('deviceList'),
    noDevicesMsg: document.getElementById('noDevicesMsg'),

    // Элементы панели 3D-DiT Видео-Трансформера
    videoLagStatus: document.getElementById('videoLagStatus'),
    videoLoreSelect: document.getElementById('videoLoreSelect'),
    btnVideoResetGenesis: document.getElementById('btnVideoResetGenesis'),
    afzStageTxt: document.getElementById('afzStageTxt'),
    afzStageBar: document.getElementById('afzStageBar'),
    fpzPsiTxt: document.getElementById('fpzPsiTxt'),
    fpzPsiBar: document.getElementById('fpzPsiBar'),
    vBoxFpz: document.getElementById('vBoxFpz'),
    vBoxAfz: document.getElementById('vBoxAfz'),
    vBoxF3: document.getElementById('vBoxF3'),
    vBoxF4: document.getElementById('vBoxF4'),
    vBoxFcz: document.getElementById('vBoxFcz'),
    vGainFpz: document.getElementById('vGainFpz'),
    vGainAfz: document.getElementById('vGainAfz'),
    vGainF3: document.getElementById('vGainF3'),
    vGainF4: document.getElementById('vGainF4'),
    vGainFcz: document.getElementById('vGainFcz'),
    vAutoBoost: document.getElementById('vAutoBoost'),
    vActiveUserGain: document.getElementById('vActiveUserGain'),
    vInactiveBaseGain: document.getElementById('vInactiveBaseGain'),
    videoRegionSelect: document.getElementById('videoRegionSelect'),
    videoRegionConceptList: document.getElementById('videoRegionConceptList'),
    newVideoRegionInput: document.getElementById('newVideoRegionInput'),
    btnAddVideoConcept: document.getElementById('btnAddVideoConcept'),
    vNumFrames: document.getElementById('vNumFrames'),
    vContextRing: document.getElementById('vContextRing'),
    vGeodesicLock: document.getElementById('vGeodesicLock'),
    vSteps: document.getElementById('vSteps'),
    vCrossBleed: document.getElementById('vCrossBleed'),
    vCascadeRate: document.getElementById('vCascadeRate'),
    vPsiThresh: document.getElementById('vPsiThresh'),

    ladderStageSelect: document.getElementById('ladderStageSelect'),
    montyStatusBar: document.getElementById('montyStatusBar'),

    diffLagStatus: document.getElementById('diffLagStatus'),
    visionConceptList: document.getElementById('visionConceptList'),
    newVisionInput: document.getElementById('newVisionInput'),
    btnAddVision: document.getElementById('btnAddVision'),
    diffPreset: document.getElementById('diffPreset'),
    diffResolution: document.getElementById('diffResolution'),
    diffTaesd: document.getElementById('diffTaesd'),
    diffSteps: document.getElementById('diffSteps'),
    diffCfg: document.getElementById('diffCfg'),
    diffStrLow: document.getElementById('diffStrLow'),
    diffStrHigh: document.getElementById('diffStrHigh'),
    diffWorldContext: document.getElementById('diffWorldContext'),
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

// =============================================================================
// HIGH-FPS SERVER-PUSH STREAMER
// =============================================================================
function paintLatestFrameFromRam() {
    if (isProbeMode || isImgDecoding || latestFrameSeq === renderedFrameSeq || !latestJpegBytes) {
        return;
    }
    isImgDecoding = true;
    const skipped = (latestFrameSeq - renderedFrameSeq) - 1;
    if (skipped > 0) clientDroppedFrames += skipped;
    renderedFrameSeq = latestFrameSeq;

    const blob = new Blob([latestJpegBytes], { type: 'image/jpeg' });
    const nextUrl = URL.createObjectURL(blob);

    dom.videoStreamImg.onload = () => {
        if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = nextUrl;
        isImgDecoding = false;
        clientRenderCount++;
        if (latestFrameSeq !== renderedFrameSeq) {
            requestAnimationFrame(paintLatestFrameFromRam);
        }
    };

    dom.videoStreamImg.onerror = () => {
        URL.revokeObjectURL(nextUrl);
        isImgDecoding = false;
    };

    dom.videoStreamImg.src = nextUrl;
}

async function startPushVideoStream() {
    stopPushVideoStream();
    if (isProbeMode || !dom.videoStreamImg) return;

    videoStreamAbortCtrl = new AbortController();
    const signal = videoStreamAbortCtrl.signal;

    try {
        const response = await fetch('/stream.mjpg?t=' + Date.now(), {
            signal: signal,
            cache: 'no-store'
        });
        if (!response.body) {
            dom.videoStreamImg.src = '/stream.mjpg';
            return;
        }

        const reader = response.body.getReader();
        let buffer = new Uint8Array(0);

        while (!isProbeMode && !signal.aborted) {
            const { done, value } = await reader.read();
            if (done || signal.aborted) break;
            if (!value || value.byteLength === 0) continue;

            const merged = new Uint8Array(buffer.byteLength + value.byteLength);
            merged.set(buffer, 0);
            merged.set(value, buffer.byteLength);
            buffer = merged;

            // Ищем последний полный JPEG с КОНЦА буфера (через нативный быстрый поиск),
            // чтобы при облачном стриминге на 40 FPS мгновенно рисовать самый свежий кадр!
            let lastEoi = -1;
            for (let i = buffer.byteLength - 2; i >= 2; i--) {
                if (buffer[i] === 0xFF && buffer[i + 1] === 0xD9) {
                    lastEoi = i + 2;
                    break;
                }
            }

            if (lastEoi !== -1) {
                let lastSoi = -1;
                for (let i = lastEoi - 4; i >= 0; i--) {
                    if (buffer[i] === 0xFF && buffer[i + 1] === 0xD8) {
                        lastSoi = i;
                        break;
                    }
                }
                if (lastSoi !== -1) {
                    latestJpegBytes = buffer.slice(lastSoi, lastEoi);
                    latestFrameSeq++;
                    buffer = buffer.slice(lastEoi);
                    paintLatestFrameFromRam();
                } else if (buffer.byteLength > 1024 * 1024) {
                    buffer = new Uint8Array(0);
                }
            } else if (buffer.byteLength > 2 * 1024 * 1024) {
                buffer = new Uint8Array(0);
            }
        }
    } catch (err) {
        if (!signal.aborted && !isProbeMode) {
            setTimeout(startPushVideoStream, 500);
        }
    }
}

function stopPushVideoStream() {
    if (videoStreamAbortCtrl) {
        try { videoStreamAbortCtrl.abort(); } catch (e) {}
        videoStreamAbortCtrl = null;
    }
    isImgDecoding = false;
}

function toggleProbeMode(enabled) {
    isProbeMode = enabled;
    if (isProbeMode) {
        stopPushVideoStream();
        dom.videoStreamImg.src = "";
        dom.videoStreamImg.style.display = "none";
        if (isAudioPlaying) toggleAudio();
    } else {
        dom.videoStreamImg.style.display = "block";
        startPushVideoStream();
    }
}

function toggleFullScreen() {
    if (!document.fullscreenElement) {
        dom.canvasBox.requestFullscreen().catch(err => alert("Fullscreen error: " + err.message));
    } else {
        document.exitFullscreen();
    }
}

// =============================================================================
// ЧТЕНИЕ И ЗАПИСЬ РЕГИСТРОВ PGA ADS131M08
// =============================================================================
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

// =============================================================================
// УПРАВЛЕНИЕ 3D-DiT ВИДЕО-ТРАНСФОРМЕРОМ (ОТВЯЗАННОЕ ОТ 2D И MUSICGEN)
// =============================================================================
function renderSelectedVideoRegionConcepts() {
    if (!dom.videoRegionSelect || !dom.videoRegionConceptList) return;
    const regKey = dom.videoRegionSelect.value.toUpperCase();
    const list = cachedVideoSections[regKey] || [];
    const roleMap = { FPZ: "Fpz", AFZ: "AFz", F3: "F3", F4: "F4", FCZ: "FCz" };
    const activeIdx = cachedActiveConceptIdx[roleMap[regKey] || "FCz"] || 0;

    dom.videoRegionConceptList.replaceChildren();
    if (list.length === 0) {
        const empty = document.createElement('span');
        empty.className = 'empty-concepts';
        empty.textContent = `В регионе [${regKey}] нет концептов`;
        dom.videoRegionConceptList.appendChild(empty);
        return;
    }

    list.forEach((c, idx) => {
        const clone = dom.conceptChipTemplate.content.cloneNode(true);
        const chip = clone.querySelector('.concept-chip');
        if (idx === activeIdx) {
            chip.classList.add('video-active');
        }
        const txt = chip.querySelector('.concept-text');
        txt.textContent = `${idx === activeIdx ? '★ ' : ''}[${regKey}#${idx + 1}] ${c.length > 44 ? c.slice(0, 44) + '…' : c}`;
        txt.title = c;

        const rmBtn = chip.querySelector('.concept-remove-btn');
        rmBtn.addEventListener('click', () => removeVideoRegionConcept(regKey, c));

        dom.videoRegionConceptList.appendChild(chip);
    });
}

async function addVideoRegionConcept() {
    const regKey = dom.videoRegionSelect.value.toUpperCase();
    const val = dom.newVideoRegionInput.value.trim();
    if (!val) return;
    const res = await fetch('/api/video/region_concepts', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ region: regKey, action: 'add', concept: val })
    });
    const d = await res.json();
    dom.newVideoRegionInput.value = '';
    if (d.sections) {
        cachedVideoSections = d.sections;
        renderSelectedVideoRegionConcepts();
    }
}

async function removeVideoRegionConcept(regKey, conceptStr) {
    const res = await fetch('/api/video/region_concepts', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ region: regKey, action: 'remove', concept: conceptStr })
    });
    const d = await res.json();
    if (d.sections) {
        cachedVideoSections = d.sections;
        renderSelectedVideoRegionConcepts();
    }
}

async function switchVideoLoreFile() {
    const loreFile = dom.videoLoreSelect.value;
    const res = await fetch('/api/video/switch_lore', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ lore_file: loreFile })
    });
    const d = await res.json();
    if (d.sections) {
        cachedVideoSections = d.sections;
        renderSelectedVideoRegionConcepts();
    }
}

async function resetVideoGenesis() {
    await fetch('/api/video/reset_genesis', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    });
}

async function updateVideoTransformerConfig() {
    const payload = {
        region_gains: {
            Fpz: parseFloat(dom.vGainFpz.value) || 0.06,
            AFz: parseFloat(dom.vGainAfz.value) || 0.08,
            F3:  parseFloat(dom.vGainF3.value)  || 0.08,
            F4:  parseFloat(dom.vGainF4.value)  || 0.10,
            FCz: parseFloat(dom.vGainFcz.value) || 0.92,
        },
        auto_active_boost: dom.vAutoBoost.value === "1",
        active_user_gain: parseFloat(dom.vActiveUserGain.value) || 0.92,
        inactive_base_gain: parseFloat(dom.vInactiveBaseGain.value) || 0.06,
        cross_region_bleed: parseFloat(dom.vCrossBleed.value) || 0.25,
        cascade_rate: parseFloat(dom.vCascadeRate.value) || 1.0,
        psi_threshold: parseFloat(dom.vPsiThresh.value) || 0.55,
        geodesic_lock: parseFloat(dom.vGeodesicLock.value) || 0.18,
        num_frames: parseInt(dom.vNumFrames.value) || 5,
        context_ring_len: parseInt(dom.vContextRing.value) || 16,
        steps: parseInt(dom.vSteps.value) || 4,
    };
    await fetch('/api/video/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
}

async function uploadVideoInitImage() {
    const fileInp = document.getElementById('videoInitImageInput');
    if (!fileInp || !fileInp.files || fileInp.files.length === 0) {
        alert("Выберите файл картинки (PNG/JPG)!");
        return;
    }
    const file = fileInp.files[0];
    const reader = new FileReader();
    reader.onload = async () => {
        await fetch('/api/video/init_image', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ image_base64: reader.result })
        });
    };
    reader.readAsDataURL(file);
}

// =============================================================================
// ПРЕСЕТЫ И НАСТРОЙКИ 2D-ДИФФУЗИИ И МУЗЫКИ
// =============================================================================
const PRESETS = {
    meditation: { steps: 0, cfg: 1.0, strLow: 0.45, strHigh: 0.70, ctx: 0.30, ema: 0.28 },
    balanced:   { steps: 0, cfg: 1.0, strLow: 0.50, strHigh: 0.85, ctx: 0.18, ema: 0.12 },
    dynamic:    { steps: 2, cfg: 1.1, strLow: 0.60, strHigh: 0.92, ctx: 0.05, ema: 0.05 }
};

const MUSIC_PRESETS = {
    meditation: { block: 0.0, ctx: 3.6, strLow: 0.12, strHigh: 0.45, ema: 0.35, temp: 0.95 },
    balanced:   { block: 0.0, ctx: 3.6, strLow: 0.25, strHigh: 0.75, ema: 0.20, temp: 1.00 },
    dynamic:    { block: 0.0, ctx: 2.4, strLow: 0.35, strHigh: 0.92, ema: 0.10, temp: 1.05 }
};

function applyPreset(presetKey) {
    if (presetKey === 'custom') return;
    const p = PRESETS[presetKey];
    if (!p) return;
    dom.diffSteps.value = p.steps;
    dom.diffCfg.value = p.cfg;
    dom.diffStrLow.value = p.strLow;
    dom.diffStrHigh.value = p.strHigh;
    if (dom.diffWorldContext) dom.diffWorldContext.value = p.ctx;
    dom.diffEmaSmooth.value = p.ema;
    onManualParamChange(presetKey);
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

function onManualParamChange(presetOverride = 'custom') {
    if (typeof presetOverride !== 'string') presetOverride = 'custom';
    dom.diffPreset.value = presetOverride;

    const rawSteps = parseInt(dom.diffSteps.value);
    const steps = Number.isNaN(rawSteps) ? 0 : Math.max(0, Math.min(8, rawSteps));
    const cfg = Math.max(0.0, Math.min(3.0, parseFloat(dom.diffCfg.value) || 1.0));
    let sLow = Math.max(0.30, Math.min(0.80, parseFloat(dom.diffStrLow.value) || 0.50));
    let sHigh = Math.max(sLow + 0.05, Math.min(0.95, parseFloat(dom.diffStrHigh.value) || 0.85));

    const rawCtx = parseFloat(dom.diffWorldContext ? dom.diffWorldContext.value : 0.18);
    const wCtx = Number.isNaN(rawCtx) ? 0.18 : Math.max(0.0, Math.min(0.60, rawCtx));
    const ema = Math.max(0.02, Math.min(0.50, parseFloat(dom.diffEmaSmooth.value) || 0.12));

    const res = dom.diffResolution ? dom.diffResolution.value : "512x384";
    const useTaesd = dom.diffTaesd ? (dom.diffTaesd.value === "1") : true;

    dom.diffSteps.value = steps;
    dom.diffCfg.value = cfg;
    dom.diffStrLow.value = sLow;
    dom.diffStrHigh.value = sHigh;
    if (dom.diffWorldContext) dom.diffWorldContext.value = wCtx;
    dom.diffEmaSmooth.value = ema;

    fetch('/api/diffusion/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            mode: dom.diffModel ? dom.diffModel.value : 'lcm',
            preset: presetOverride,
            steps: steps,
            guidance_scale: cfg,
            strength_low: sLow,
            strength_high: sHigh,
            world_context: wCtx,
            ema_smooth: ema,
            resolution: res,
            use_taesd: useTaesd,
            use_color: useTaesd
        })
    });
}

async function onLadderStageChange() {
    const stg = parseInt(dom.ladderStageSelect.value) || 1;
    await fetch('/api/preset/ladder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ stage: stg })
    });
}

// =============================================================================
// SSE ПОТОК ТЕЛЕМЕТРИИ И СОСТОЯНИЯ
// =============================================================================
const sse = new EventSource('/api/state_stream');

sse.addEventListener('state', (e) => {
    const st = JSON.parse(e.data);

    // 1. Обновление специфической панели 3D-DiT Видео-Трансформера
    if (st.video) {
        const v = st.video;
        if (v.sections) {
            cachedVideoSections = v.sections;
        }
        if (v.active_concept_idx) {
            cachedActiveConceptIdx = v.active_concept_idx;
        }
        renderSelectedVideoRegionConcepts();

        if (dom.videoLoreSelect && Array.isArray(v.available_lores)) {
            const currentOpts = Array.from(dom.videoLoreSelect.options).map(o => o.value);
            if (JSON.stringify(currentOpts) !== JSON.stringify(v.available_lores)) {
                dom.videoLoreSelect.replaceChildren();
                v.available_lores.forEach(pathStr => {
                    const opt = document.createElement('option');
                    opt.value = pathStr;
                    opt.textContent = pathStr ? pathStr : '[Пусто — Свободный 4096-D дрейф без лор-файла]';
                    dom.videoLoreSelect.appendChild(opt);
                });
            }
            if (document.activeElement !== dom.videoLoreSelect && v.lore_file !== undefined) {
                dom.videoLoreSelect.value = v.lore_file;
            }
        }

        const stagePct = Math.round((v.afz_stage_progress || 0) * 100);
        const psiPct = Math.round((v.fpz_meta_charge || 0) * 100);
        if (dom.afzStageTxt) dom.afzStageTxt.textContent = `${stagePct}%`;
        if (dom.afzStageBar) dom.afzStageBar.style.width = `${stagePct}%`;
        if (dom.fpzPsiTxt) dom.fpzPsiTxt.textContent = `${psiPct}%`;
        if (dom.fpzPsiBar) dom.fpzPsiBar.style.width = `${psiPct}%`;

        if (v.region_gains) {
            if (document.activeElement !== dom.vGainFpz) dom.vGainFpz.value = v.region_gains.Fpz ?? 0.06;
            if (document.activeElement !== dom.vGainAfz) dom.vGainAfz.value = v.region_gains.AFz ?? 0.08;
            if (document.activeElement !== dom.vGainF3)  dom.vGainF3.value  = v.region_gains.F3  ?? 0.08;
            if (document.activeElement !== dom.vGainF4)  dom.vGainF4.value  = v.region_gains.F4  ?? 0.10;
            if (document.activeElement !== dom.vGainFcz) dom.vGainFcz.value = v.region_gains.FCz ?? 0.92;
        }

        if (v.active_region_mask) {
            dom.vBoxFpz.classList.toggle('user-on', !!v.active_region_mask.Fpz);
            dom.vBoxAfz.classList.toggle('user-on', !!v.active_region_mask.AFz);
            dom.vBoxF3.classList.toggle('user-on',  !!v.active_region_mask.F3);
            dom.vBoxF4.classList.toggle('user-on',  !!v.active_region_mask.F4);
            dom.vBoxFcz.classList.toggle('user-on', !!v.active_region_mask.FCz);
        }

        if (dom.videoLagStatus) {
            if (v.connected) {
                const vFps = (v.fps || 0).toFixed(1);
                const vMs = (v.gen_ms || 0).toFixed(0);
                dom.videoLagStatus.textContent = `3D-DiT: ${vFps} FPS (${vMs} мс/блок) | ${v.debug_tokens || 'ONLINE'} 🟢`;
            } else {
                dom.videoLagStatus.textContent = `3D-DiT: Автономный режим / Ожидание порта 6007`;
            }
        }
    }

    // 2. Обновление панели 2D-Диффузии и глобальной телеметрии
    if (st.world) {
        renderVisionConcepts(st.world.concepts || []);
        if (document.activeElement !== dom.diffSteps) dom.diffSteps.value = st.world.steps ?? 0;
        if (document.activeElement !== dom.diffCfg) dom.diffCfg.value = st.world.guidance_scale ?? 1.0;
        if (document.activeElement !== dom.diffStrLow) dom.diffStrLow.value = st.world.strength_low ?? 0.50;
        if (document.activeElement !== dom.diffStrHigh) dom.diffStrHigh.value = st.world.strength_high ?? 0.85;
        if (dom.diffWorldContext && document.activeElement !== dom.diffWorldContext) {
            dom.diffWorldContext.value = st.world.world_context ?? 0.18;
        }
        if (document.activeElement !== dom.diffEmaSmooth) dom.diffEmaSmooth.value = st.world.ema_smooth ?? 0.12;
        if (dom.diffResolution && document.activeElement !== dom.diffResolution && st.world.resolution) {
            dom.diffResolution.value = st.world.resolution;
        }
        if (dom.diffTaesd && document.activeElement !== dom.diffTaesd && st.world.use_taesd !== undefined) {
            dom.diffTaesd.value = st.world.use_taesd ? "1" : "0";
        }
        if (st.world.preset && document.activeElement !== dom.diffPreset) {
            dom.diffPreset.value = st.world.preset;
        }
        if (st.world.ladder_stage && document.activeElement !== dom.ladderStageSelect) {
            dom.ladderStageSelect.value = st.world.ladder_stage;
        }

        if (dom.diffLagStatus) {
            const gpuFps = (st.world.fps || 0).toFixed(1);
            const scrFps = clientScreenFps.toFixed(1);
            const genMs = (st.world.gen_ms || 0).toFixed(0);
            const actSteps = st.world.actual_steps || st.world.steps || 2;
            const res = st.world.resolution || "512x384";
            dom.diffLagStatus.textContent = `Экран: ${scrFps} FPS | Поток: ${gpuFps} FPS (${genMs} мс) | Шаги: ${actSteps} | ${res} 🟢`;
        }

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

    // 3. Обновление панели MusicGen
    if (st.music) {
        renderMusicConcepts(st.music.concepts || []);
        if (document.activeElement !== dom.mBlock) dom.mBlock.value = st.music.target_block ?? 0;
        if (document.activeElement !== dom.mContext) dom.mContext.value = st.music.target_context ?? 3.6;
        if (document.activeElement !== dom.mBuffer) dom.mBuffer.value = st.music.target_buffer_count ?? 2;
        if (document.activeElement !== dom.mTemp) dom.mTemp.value = st.music.temp ?? 1.0;
        if (document.activeElement !== dom.mTopK) dom.mTopK.value = st.music.top_k ?? 250;
        if (document.activeElement !== dom.mCfg) dom.mCfg.value = st.music.cfg_coef ?? 1.0;
        if (document.activeElement !== dom.mStrLow) dom.mStrLow.value = st.music.strength_low ?? 0.25;
        if (document.activeElement !== dom.mStrHigh) dom.mStrHigh.value = st.music.strength_high ?? 0.75;
        if (document.activeElement !== dom.mEmaSmooth) dom.mEmaSmooth.value = st.music.ema_smooth ?? 0.20;

        const bSec = (st.music.buffer_sec || 0).toFixed(2);
        const bChunks = st.music.buffer_chunks || 0;
        const maxChunks = st.music.target_buffer_count || 2;
        const actBlk = (st.music.actual_block || 0.16).toFixed(2);
        const spd = (st.music.play_speed || 1.0).toFixed(2);
        const rtf = (st.music.rtf || 1.0).toFixed(2);
        const dStr = (st.music.dyn_strength || 0.25).toFixed(2);
        dom.musicLagStatus.textContent = `Буфер: ${bSec}с (${bChunks}/${maxChunks} бл) | Блок: ${actBlk}с | Speed: ${spd}x | RTF: ${rtf}x | WM: ${dStr} 🟢`;
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
// ПРЯМОЙ ПРИЕМНИК ГОТОВОГО АУДИО С СЕРВЕРА
// =============================================================================
async function startWebAudioStream() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass({ sampleRate: AUDIO_SR });
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

    audioAbortCtrl = new AbortController();
    const response = await fetch('/audio.wav?t=' + Date.now(), {
        signal: audioAbortCtrl.signal,
        cache: 'no-store'
    });
    const reader = response.body.getReader();

    let headerBytesRemaining = 44;
    let leftoverByte = null;
    let nextPlayTime = 0;

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

                const float32Arr = new Float32Array(numSamples);
                const view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
                for (let i = 0; i < numSamples; i++) {
                    float32Arr[i] = view.getInt16(i * 2, true) / 32768.0;
                }

                if (!audioCtx || !isAudioPlaying) break;
                const now = audioCtx.currentTime;
                if (nextPlayTime < now) {
                    nextPlayTime = now + 0.06;
                }

                const audioBuf = audioCtx.createBuffer(1, float32Arr.length, AUDIO_SR);
                audioBuf.getChannelData(0).set(float32Arr);

                const src = audioCtx.createBufferSource();
                src.buffer = audioBuf;
                src.connect(audioCtx.destination);
                src.start(nextPlayTime);
                nextPlayTime += float32Arr.length / AUDIO_SR;
            }
        } catch (err) {
            if (isAudioPlaying) console.warn("WebAudio stream closed:", err);
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

// =============================================================================
// BLUETOOTH BLE (FREEEEG16/24)
// =============================================================================
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
        // Синхронно переключаем выбранный регион в редакторе 3D-DiT!
        if (dom.videoRegionSelect) {
            dom.videoRegionSelect.value = devObj.role.toUpperCase();
            renderSelectedVideoRegionConcepts();
        }
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
            role: activeDevices.length === 0 ? "FCz" : "AFz",
            gain: 32,
            uvScale: calcUvScale(32),
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

        await writeAndVerifyGain(devObj, 32);

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
    if (devObj.sampleBuffer.length === 0) return;
    if ((devObj.inFlightCount || 0) >= 4) return; // До 4 пакетов параллельно в полете через интернет!

    const batch = devObj.sampleBuffer.splice(0, devObj.sampleBuffer.length);
    devObj.inFlightCount = (devObj.inFlightCount || 0) + 1;

    fetch('/api/eeg_push', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user: devObj.user,
            role: devObj.role,
            sps: 250.0,
            samples: batch
        }),
        keepalive: true
    }).catch(() => {}).finally(() => {
        devObj.inFlightCount = Math.max(0, (devObj.inFlightCount || 1) - 1);
        if (devObj.sampleBuffer.length >= BATCH_SIZE) {
            sendBatch(devObj);
        }
    });
}

setInterval(() => {
    clientScreenFps = clientRenderCount;
    clientRenderCount = 0;

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

// =============================================================================
// УПРАВЛЕНИЕ 2D-КОНЦЕПТАМИ, МУЗЫКОЙ И ЗАПИСЬЮ СЕССИЙ
// =============================================================================
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

    const parsedBlock = parseFloat(dom.mBlock.value);
    const targetBlock = Number.isNaN(parsedBlock) ? 0.0 : Math.max(0.0, parsedBlock);

    dom.mStrLow.value = sLow;
    dom.mStrHigh.value = sHigh;
    dom.mEmaSmooth.value = ema;
    dom.mBlock.value = targetBlock;

    await fetch('/api/music_config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            model: dom.musicModel ? dom.musicModel.value : 'facebook/musicgen-small',
            target_block: targetBlock,
            target_context: parseFloat(dom.mContext.value) || 3.6,
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

    // Обработчики специфической панели 3D-DiT Видео-Трансформера
    document.getElementById('btnUploadInitImage')?.addEventListener('click', uploadVideoInitImage);
    if (dom.videoLoreSelect) {
        dom.videoLoreSelect.addEventListener('change', switchVideoLoreFile);
    }
    if (dom.btnVideoResetGenesis) {
        dom.btnVideoResetGenesis.addEventListener('click', resetVideoGenesis);
    }
    if (dom.videoRegionSelect) {
        dom.videoRegionSelect.addEventListener('change', renderSelectedVideoRegionConcepts);
    }
    if (dom.btnAddVideoConcept) {
        dom.btnAddVideoConcept.addEventListener('click', addVideoRegionConcept);
    }
    [
        dom.vGainFpz, dom.vGainAfz, dom.vGainF3, dom.vGainF4, dom.vGainFcz,
        dom.vAutoBoost, dom.vActiveUserGain, dom.vInactiveBaseGain,
        dom.vNumFrames, dom.vContextRing, dom.vGeodesicLock, dom.vSteps,
        dom.vCrossBleed, dom.vCascadeRate, dom.vPsiThresh
    ].forEach(inp => {
        if (inp) inp.addEventListener('change', updateVideoTransformerConfig);
    });

    dom.btnAddVision.addEventListener('click', addVisionConcept);
    dom.btnAddMusic.addEventListener('click', addMusicConcept);

    if (dom.ladderStageSelect) {
        dom.ladderStageSelect.addEventListener('change', onLadderStageChange);
    }

    dom.diffPreset.addEventListener('change', (e) => applyPreset(e.target.value));
    [
        dom.diffSteps, dom.diffCfg, dom.diffStrLow, dom.diffStrHigh,
        dom.diffWorldContext, dom.diffEmaSmooth, dom.diffResolution, dom.diffTaesd
    ].forEach(inp => {
        if (inp) {
            inp.addEventListener('change', () => onManualParamChange('custom'));
        }
    });

    if (dom.musicPreset) {
        dom.musicPreset.addEventListener('change', (e) => applyMusicPreset(e.target.value));
    }
    [
        dom.mStrLow, dom.mStrHigh, dom.mEmaSmooth, dom.mBlock,
        dom.mContext, dom.mBuffer, dom.mTemp, dom.mTopK, dom.mCfg
    ].forEach(inp => {
        if (inp) {
            inp.addEventListener('change', () => {
                if (dom.musicPreset) dom.musicPreset.value = 'custom';
                updateMusicConfig();
            });
        }
    });

    dom.btnRecStart.addEventListener('click', () => toggleRecording(true));
    dom.btnRecStop.addEventListener('click', () => toggleRecording(false));

    startPushVideoStream();

    try {
        const [rV, rM, rVid] = await Promise.all([
            fetch('/api/world/concepts'),
            fetch('/api/music/concepts'),
            fetch('/api/video/state')
        ]);
        const dV = await rV.json();
        const dM = await rM.json();
        const dVid = await rVid.json();
        renderVisionConcepts(dV.concepts || []);
        renderMusicConcepts(dM.concepts || []);
        if (dVid && dVid.sections) {
            cachedVideoSections = dVid.sections;
            renderSelectedVideoRegionConcepts();
        }
    } catch (err) {
        console.error("Ошибка начальной загрузки:", err);
    }
});

