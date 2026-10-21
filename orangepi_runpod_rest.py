#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ROBUST RUNPOD ORCHESTRATOR v6.2
===================================================================================
Строгая последовательность запуска:
  1. Проверка SSH доступности
  2. Проверка репозитория: git clone (если нет) или git reset --hard (если есть)
  3. Проверка зависимостей (cv2)
  4. Запуск 3 процессов в реальном времени с выводом в /workspace/backend.log
===================================================================================
"""

from __future__ import annotations
import os
import sys
import glob
import time
import json
import base64
import subprocess
import threading
from pathlib import Path
import requests
import urllib3
from flask import Flask, request, jsonify, render_template_string, send_from_directory

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RUNPOD_API_KEY    = os.environ.get("RUNPOD_API_KEY", "").strip()
GQL_URL           = "https://api.runpod.io/graphql"

DEFAULT_SSH_HOST  = os.environ.get("RUNPOD_SSH_HOST", "47.47.180.20").strip()
DEFAULT_SSH_PORT  = int(os.environ.get("RUNPOD_SSH_PORT", "21155"))
SSH_USER          = os.environ.get("RUNPOD_SSH_USER", "root").strip()
SSH_PRIV_KEY      = os.path.expanduser(os.environ.get("SSH_KEY_PATH", "~/.ssh/id_ed25519"))

GIT_REPO_URL      = os.environ.get("NEURO_GIT_REPO", "https://github.com/neuroidss/neuro_heterarchy.git").strip()
LOCAL_SESSIONS    = Path(os.environ.get("LOCAL_SESSIONS_DIR", "./downloaded_sessions")).resolve()
LOCAL_SESSIONS.mkdir(parents=True, exist_ok=True)

IMAGE_MAP = {
    "all":    "neuroidss/neuro_heterarchy_all:latest",
    "music":  "neuroidss/neuro_heterarchy_musicgen:latest",
    "vision": "neuroidss/neuro_heterarchy_diffusion:latest"
}


def ensure_https_certificates(cert_path="cert.pem", key_path="key.pem"):
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-days", "365", "-nodes", "-subj", "/CN=orangepi.local"
        ], check=False)
    return cert_path, key_path


def ensure_ssh_keypair() -> str:
    pub_path = SSH_PRIV_KEY + ".pub"
    if not os.path.exists(SSH_PRIV_KEY):
        os.makedirs(os.path.dirname(SSH_PRIV_KEY), exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", SSH_PRIV_KEY, "-N", ""], check=False)
    if os.path.exists(pub_path):
        with open(pub_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def gather_all_local_pub_keys() -> str:
    ensure_ssh_keypair()
    ssh_dir = os.path.expanduser("~/.ssh")
    pub_lines = []
    for p in sorted(glob.glob(os.path.join(ssh_dir, "*.pub"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                line = f.read().strip()
                if line and line not in pub_lines:
                    pub_lines.append(line)
        except Exception:
            pass
    return "\n".join(pub_lines)


class UnifiedRunPodOrchestrator:
    def __init__(self):
        self.lock = threading.Lock()
        self.api_key = RUNPOD_API_KEY
        self.target_pod_id = None
        self.active_pod = None
        
        self.ssh_host = DEFAULT_SSH_HOST
        self.ssh_port = DEFAULT_SSH_PORT
        self.ssh_user = SSH_USER
        
        self.all_pods = []
        self.studio_url = None
        self.live_logs = "Ожидание подключения к поду..."
        self.status_msg = "Готов к работе."
        self.is_busy = False
        self.auth_ok = False
        self.stack_mode = "music"

        ensure_ssh_keypair()
        threading.Thread(target=self._poll_cloud_loop, daemon=True).start()

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _gql(self, query: str, variables: dict | None = None) -> tuple[bool, any]:
        if not self.api_key:
            return False, "Не задан RUNPOD_API_KEY"
        try:
            r = requests.post(
                f"{GQL_URL}?api_key={self.api_key}",
                headers=self._headers(),
                json={"query": query, "variables": variables or {}},
                timeout=12.0
            )
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
            data = r.json()
            if "errors" in data and data["errors"]:
                return False, data["errors"][0].get("message", "GQL Error")
            return True, data.get("data", {})
        except Exception as e:
            return False, str(e)

    def run_ssh(self, cmd: str, timeout: int = 25) -> tuple[bool, str]:
        if not self.ssh_host or not self.ssh_port:
            return False, "SSH Host/Port не задан"
        env = os.environ.copy()
        env["SSH_AUTH_SOCK"] = ""
        full_cmd = [
            "ssh",
            "-i", SSH_PRIV_KEY,
            "-p", str(self.ssh_port),
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=6",
            "-o", "BatchMode=yes",
            f"{self.ssh_user}@{self.ssh_host}",
            cmd
        ]
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout, env=env)
            if r.returncode == 0:
                return True, r.stdout.strip()
            err_msg = r.stderr.strip() or r.stdout.strip() or f"Exit {r.returncode}"
            return False, err_msg
        except Exception as e:
            return False, str(e)

    def _poll_cloud_loop(self):
        query = """
        query {
            myself {
                id
                pods {
                    id
                    name
                    desiredStatus
                    costPerHr
                    runtime {
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                        }
                    }
                }
            }
        }
        """
        while True:
            if self.api_key:
                ok, res = self._gql(query)
                if ok:
                    self.auth_ok = True
                    pods = (res.get("myself") or {}).get("pods") or []
                    pod_list = []
                    for p in pods:
                        s_ip, s_port = None, None
                        for pt in ((p.get("runtime") or {}).get("ports") or []):
                            if pt.get("privatePort") == 22 and pt.get("isIpPublic"):
                                s_ip, s_port = pt.get("ip"), int(pt.get("publicPort"))
                                break
                        pod_list.append({
                            "id": p["id"],
                            "name": p.get("name", p["id"]),
                            "status": (p.get("desiredStatus") or "UNKNOWN").upper(),
                            "cost": float(p.get("costPerHr") or 0.0),
                            "ssh_ip": s_ip,
                            "ssh_port": s_port,
                            "studio_url": f"https://{p['id']}-8080.proxy.runpod.net"
                        })
                    self.all_pods = pod_list

                    if self.target_pod_id:
                        matched = next((p for p in pod_list if p["id"] == self.target_pod_id), None)
                        if matched:
                            self.active_pod = matched
                            if matched["ssh_ip"] and matched["ssh_port"]:
                                self.ssh_host = matched["ssh_ip"]
                                self.ssh_port = matched["ssh_port"]
                            self.studio_url = matched["studio_url"]
                else:
                    self.auth_ok = False

            # Безопасный опрос реалтайм-лога (touch гарантирует существование)
            if self.ssh_host and self.ssh_port:
                read_log_cmd = "touch /workspace/backend.log && tail -n 60 /workspace/backend.log 2>/dev/null"
                ok_log, out_log = self.run_ssh(read_log_cmd, timeout=5)
                if ok_log and out_log:
                    with self.lock:
                        self.live_logs = out_log

            time.sleep(2.5)

    def deploy_and_start_pipeline(self, mode: str = "music"):
        """
        Последовательное выполнение: SSH -> Git -> Deps -> Запуск всех 3 процессов.
        """
        if self.is_busy:
            return
        self.is_busy = True
        self.stack_mode = mode
        self.status_msg = f"⏳ [1/4] Проверка связи по SSH ({self.ssh_host}:{self.ssh_port})..."

        # 1. Проверка SSH
        ok_ssh, out_ssh = self.run_ssh("echo PING", timeout=8)
        if not ok_ssh:
            self.status_msg = f"❌ SSH недоступен: {out_ssh}"
            self.is_busy = False
            return

        # 2. Обновление или клонирование репозитория
        self.status_msg = "⏳ [2/4] Синхронизация кода с GitHub..."
        git_prep_cmd = f"""
        mkdir -p /workspace/hf_cache /workspace/logs /workspace/neuro_heterarchy/sessions /workspace/neuro_heterarchy/lore
        touch /workspace/backend.log
        if [ ! -d "/workspace/neuro_heterarchy/.git" ]; then
            echo "[DEPLOY] Клонирование {GIT_REPO_URL}..." >> /workspace/backend.log
            rm -rf /workspace/neuro_heterarchy
            git clone {GIT_REPO_URL} /workspace/neuro_heterarchy
        else
            echo "[DEPLOY] Обновление существующего репозитория Git..." >> /workspace/backend.log
            cd /workspace/neuro_heterarchy
            git fetch --all 2>&1
            git reset --hard origin/HEAD 2>&1
        fi
        """
        ok_git, out_git = self.run_ssh(git_prep_cmd, timeout=45)
        if not ok_git:
            self.status_msg = f"❌ Сбой Git: {out_git[:120]}"
            self.is_busy = False
            return

        # 3. Проверка зависимостей
        self.status_msg = "⏳ [3/4] Проверка зависимостей окружения..."
        deps_cmd = """
        python3 -c "import cv2" 2>/dev/null || pip install --no-cache-dir opencv-python-headless 2>/dev/null || true
        """
        self.run_ssh(deps_cmd, timeout=30)

        # 4. Запуск всех 3 процессов с перенаправлением в /workspace/backend.log
        self.status_msg = f"🚀 [4/4] Запуск 3 процессов ({mode.upper()})..."

        run_script = f"""
        set -e
        cd /workspace/neuro_heterarchy

        # Очистка старых процессов
        pkill -9 -f 'musicgen_server.py' 2>/dev/null || true
        pkill -9 -f 'brain_server.py' 2>/dev/null || true
        pkill -9 -f 'web_cloud_gateway.py' 2>/dev/null || true
        pkill -9 -f 'neuro_heterarchy_musicgen.py' 2>/dev/null || true
        pkill -9 -f 'neuro_heterarchy_diffusion.py' 2>/dev/null || true

        # Сертификаты HTTPS
        if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
            openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
        fi

        export PYTHONUNBUFFERED=1
        if [ -f /opt/venv_musicgen/bin/python3 ]; then
            PY_MUSIC="/opt/venv_musicgen/bin/python3"
        else
            PY_MUSIC="python3"
        fi
        PY_VISION="python3"

        echo "=== [1/3] СТАРТ СЕРВЕРА МОДЕЛИ ===" >> /workspace/backend.log
        """

        if mode == "music":
            run_script += """
            nohup stdbuf -oL -eL $PY_MUSIC -u musicgen_server.py --port 6005 --model facebook/musicgen-small < /dev/null 2>&1 \
                | stdbuf -oL sed -u 's/^/[SERVER] /' >> /workspace/backend.log &
            """
        else:
            run_script += """
            nohup stdbuf -oL -eL $PY_VISION -u brain_server.py --mode lcm < /dev/null 2>&1 \
                | stdbuf -oL sed -u 's/^/[SERVER] /' >> /workspace/backend.log &
            """

        run_script += """
        echo "=== [2/3] СТАРТ WEB & LSL ШЛЮЗА (:8080) ===" >> /workspace/backend.log
        nohup stdbuf -oL -eL python3 -u web_cloud_gateway.py --port 8080 --cert ./cert.pem --key ./key.pem \
            --music-lore lore/music_lore.txt --lore lore/world_lore.txt < /dev/null 2>&1 \
            | stdbuf -oL sed -u 's/^/[GATEWAY] /' >> /workspace/backend.log &

        sleep 3

        echo "=== [3/3] СТАРТ ДВИЖКА ГЕТЕРАРХИИ ===" >> /workspace/backend.log
        """

        if mode == "music":
            run_script += """
            nohup stdbuf -oL -eL $PY_MUSIC -u neuro_heterarchy_musicgen.py --headless --mode semantic \
                --lore lore/music_lore.txt --gateway-port 6003 < /dev/null 2>&1 \
                | stdbuf -oL sed -u 's/^/[ENGINE] /' >> /workspace/backend.log &
            """
        else:
            run_script += """
            nohup stdbuf -oL -eL $PY_VISION -u neuro_heterarchy_diffusion.py --headless --mode lcm \
                --speed quality --steps 2 --lore lore/world_lore.txt < /dev/null 2>&1 \
                | stdbuf -oL sed -u 's/^/[ENGINE] /' >> /workspace/backend.log &
            """

        run_script += """
        echo "✅ ВСЕ 3 ПРОЦЕССА УСПЕШНО ЗАПУЩЕНЫ В РЕАЛЬНОМ ВРЕМЕНИ!" >> /workspace/backend.log
        """

        b64 = base64.b64encode(run_script.encode("utf-8")).decode("ascii")
        # Закрываем файловые дескрипторы для мгновенного выхода SSH
        ssh_cmd = f"echo {b64} | base64 -d | nohup bash > /dev/null 2>&1 &"
        ok_launch, out_launch = self.run_ssh(ssh_cmd, timeout=12)

        if ok_launch:
            studio = self.studio_url or f"https://{self.ssh_host}:8080"
            self.status_msg = f"🟢 Стек успешно запущен! Студия: {studio}"
        else:
            self.status_msg = f"⚠️ Ошибка вызова запуска: {out_launch}"

        self.is_busy = False

    def git_pull_explicit(self) -> str:
        """Принудительный Git Pull по отдельной кнопке"""
        self.status_msg = "⏳ Выполняю Git Pull на поде..."
        cmd = """
        mkdir -p /workspace/neuro_heterarchy
        touch /workspace/backend.log
        if [ ! -d "/workspace/neuro_heterarchy/.git" ]; then
            git clone """ + GIT_REPO_URL + """ /workspace/neuro_heterarchy 2>&1
        else
            cd /workspace/neuro_heterarchy && git fetch --all 2>&1 && git reset --hard origin/HEAD 2>&1
        fi
        echo "✅ Код на поде синхронизирован с GitHub" >> /workspace/backend.log
        """
        ok, out = self.run_ssh(cmd, timeout=30)
        self.status_msg = "✅ Код на поде обновлен!" if ok else f"❌ Ошибка: {out}"
        return out

    def stop_stack(self):
        cmd = """
        touch /workspace/backend.log
        pkill -9 -f 'musicgen_server.py' 2>/dev/null || true
        pkill -9 -f 'brain_server.py' 2>/dev/null || true
        pkill -9 -f 'web_cloud_gateway.py' 2>/dev/null || true
        pkill -9 -f 'neuro_heterarchy_musicgen.py' 2>/dev/null || true
        pkill -9 -f 'neuro_heterarchy_diffusion.py' 2>/dev/null || true
        echo '⏹️ Все 3 процесса стека остановлены' >> /workspace/backend.log
        """
        self.run_ssh(cmd, timeout=10)
        self.status_msg = "⏹️ Стек остановлен."

    def sync_sessions(self) -> int:
        self.status_msg = "🔍 Поиск сессий на RunPod..."
        ok, out = self.run_ssh("find /workspace/neuro_heterarchy/sessions -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null", timeout=10)
        if not ok or not out.strip():
            self.status_msg = "ℹ️ Папка sessions на поде пуста."
            return 0

        sessions = [s.strip() for s in out.splitlines() if s.strip()]
        downloaded = 0
        for s in sessions:
            remote_tar = f"/tmp/{s}.tar.gz"
            local_tar = LOCAL_SESSIONS / f"{s}.tar.gz"
            pack_cmd = f"tar --exclude='frames' --exclude='*.jpg' -czf {remote_tar} -C /workspace/neuro_heterarchy/sessions {s}"
            self.run_ssh(pack_cmd, timeout=30)

            env = os.environ.copy()
            env["SSH_AUTH_SOCK"] = ""
            scp_cmd = [
                "scp", "-i", SSH_PRIV_KEY, "-P", str(self.ssh_port),
                "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{self.ssh_user}@{self.ssh_host}:{remote_tar}", str(local_tar)
            ]
            r = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=90, env=env)
            if r.returncode == 0 and local_tar.exists():
                subprocess.run(["tar", "-xzf", str(local_tar), "-C", str(LOCAL_SESSIONS)], check=False)
                s_dir = LOCAL_SESSIONS / s
                if s_dir.exists() and Path("analyze_session.py").exists():
                    subprocess.run([sys.executable, "analyze_session.py", str(s_dir)], check=False)
                downloaded += 1

        self.status_msg = f"✅ Скачано сессий: {downloaded} шт."
        return downloaded

    def get_sessions_catalog(self) -> list[dict]:
        res = []
        for d in sorted(LOCAL_SESSIONS.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").exists():
                try:
                    with open(d / "manifest.json", "r", encoding="utf-8") as f:
                        man = json.load(f)
                    res.append({
                        "name": d.name,
                        "duration_sec": round(float(man.get("duration_seconds", 0.0)), 1),
                        "video_sec": round(float(man.get("video_duration_seconds", 0.0)), 1),
                        "eeg_samples": int(man.get("eeg_samples_count", 0)),
                        "tar_file": f"{d.name}.tar.gz" if (LOCAL_SESSIONS / f"{d.name}.tar.gz").exists() else "",
                        "has_plot": (d / "verified_sync_alignment.png").exists(),
                        "has_video": (d / "video_timeline.mp4").exists()
                    })
                except Exception:
                    pass
        return res


orchestrator = UnifiedRunPodOrchestrator()
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>Orange Pi × RunPod Controller v6.2</title>
    <style>
        body { background: #06090e; color: #cbd5e1; font-family: monospace; margin: 0; padding: 14px; }
        .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; margin-bottom: 12px; max-width: 940px; margin-left: auto; margin-right: auto; }
        h2 { margin-top: 0; color: #00ffc8; font-size: 14px; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
        .stat { font-size: 13px; margin: 5px 0; }
        .row { display: flex; gap: 8px; margin-top: 6px; }
        .btn { display: inline-block; width: 100%; padding: 10px; font-size: 12px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; font-family: monospace; text-align: center; box-sizing: border-box; text-decoration: none; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-green  { background: #059669; color: #fff; }
        .btn-blue   { background: #0284c7; color: #fff; }
        .btn-orange { background: #d97706; color: #fff; }
        .btn-red    { background: #e11d48; color: #fff; }
        .btn-gray   { background: #334155; color: #cbd5e1; }
        .log-box { background: #020617; color: #38bdf8; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; font-size: 11px; white-space: pre-wrap; height: 320px; overflow-y: auto; }
        .pod-row { display: flex; justify-content: space-between; align-items: center; background: #131d2e; padding: 8px 10px; border-radius: 4px; margin: 5px 0; font-size: 11px; border: 1px solid #334155; }
        .pod-row.selected { border-color: #00ffc8; background: #102436; }
        input, select { padding: 7px; background: #030712; border: 1px solid #334155; color: #fff; border-radius: 4px; font-family: monospace; font-size: 11px; width: 100%; box-sizing: border-box; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🍊 Orange Pi × RunPod Controller (Реалтайм Стек & Управление Подами)</h2>
        <div class="stat">🔑 <b>RunPod API:</b> <span id="apiStatus">...</span> | <b>Выбранный Pod:</b> <code id="activePodId">...</code></div>
        <div class="stat">🌐 <b>Текущий SSH:</b> <span id="sshTarget">...</span></div>
        <div class="stat">🔗 <b>Студия (HTTPS :8080):</b> <span id="studioLink">Ожидание...</span></div>
        <div class="stat" style="color:#fbbf24; margin-top:6px;" id="statusMsg">...</div>

        <div class="row">
            <input type="text" id="manualHost" placeholder="Прямой SSH IP (например, 47.47.180.20)">
            <input type="number" id="manualPort" placeholder="SSH Port (например, 21155)" style="width:160px;">
            <button class="btn btn-blue" id="btnConnect" style="width:200px;" onclick="saveManualSsh()">🔌 Подключиться</button>
        </div>
    </div>

    <div class="card">
        <h2>📦 Существующие поды в вашем аккаунте RunPod</h2>
        <div id="podList">Загрузка подов...</div>
    </div>

    <div class="card">
        <h2>⚡ Запуск полного стека (SSH ➔ Git Clone/Reset ➔ Deps ➔ Старт 3 процессов)</h2>
        <div class="row">
            <button class="btn btn-green" id="btnMusic" onclick="runAction('/stack/start_music', this)">
                🎵 СТАРТ MUSICGEN (Server :6005 + Gateway :8080 + Engine :6003)
            </button>
            <button class="btn btn-blue" id="btnVision" onclick="runAction('/stack/start_vision', this)">
                🎨 СТАРТ DIFFUSION (Server :6000 + Gateway :8080 + Engine :6002)
            </button>
        </div>
        <div class="row">
            <button class="btn btn-orange" id="btnStop" onclick="runAction('/stack/stop', this)">⏹️ ОСТАНОВИТЬ ВСЕ 3 ПРОЦЕССА</button>
            <button class="btn btn-gray" id="btnPull" onclick="runAction('/stack/git_pull', this)">🔄 GIT PULL НА ПОДЕ</button>
        </div>
    </div>

    <div class="card">
        <h2>📜 Единый живой поток логов всех трёх процессов ([SERVER], [GATEWAY], [ENGINE])</h2>
        <div class="log-box" id="liveLogs">Загрузка логов...</div>
    </div>

    <div class="card">
        <h2>📂 Менеджер сессий на Orange Pi (Видео 1:1 + ЭЭГ)</h2>
        <button class="btn btn-blue" id="btnSync" onclick="runAction('/sessions/sync', this)">
            📥 СИНХРОНИЗИРОВАТЬ СЕССИИ С RUNPOD НА ORANGE PI СЕЙЧАС
        </button>
        <div id="sessionsList" style="margin-top:10px;">Загрузка списка...</div>
        <div id="viewer" style="display:none; margin-top:12px; background:#020617; padding:12px; border-radius:6px; border:1px solid #38bdf8;">
            <h3 id="viewerTitle" style="color:#00ffc8; margin-top:0;">Просмотр</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                <div>
                    <label>Видео 1:1 (video_timeline.mp4):</label>
                    <video id="viewerVid" controls style="width:100%; border-radius:4px; background:#000;"></video>
                </div>
                <div>
                    <label>График синхронизации (analyze_session.py):</label>
                    <img id="viewerImg" style="width:100%; border-radius:4px; background:#111;" alt="График ЭЭГ">
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runAction(url, btnEl) {
            const oldText = btnEl ? btnEl.textContent : "";
            if (btnEl) {
                btnEl.disabled = true;
                btnEl.textContent = "⏳ Выполняется...";
            }
            try {
                const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
                const d = await res.json();
                if (d.status_msg) document.getElementById('statusMsg').textContent = d.status_msg;
            } catch(e) {
                alert("Ошибка запроса: " + e.message);
            } finally {
                if (btnEl) {
                    btnEl.disabled = false;
                    btnEl.textContent = oldText;
                }
                poll();
            }
        }
        async function saveManualSsh() {
            const host = document.getElementById('manualHost').value.trim();
            const port = parseInt(document.getElementById('manualPort').value);
            if (host && port) {
                await fetch('/direct_ssh/set', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({host: host, port: port})
                });
                poll();
            }
        }
        function openViewer(name) {
            document.getElementById('viewer').style.display = 'block';
            document.getElementById('viewerTitle').textContent = 'Сессия: ' + name;
            document.getElementById('viewerVid').src = `/session_file/${name}/video_timeline.mp4?t=` + Date.now();
            document.getElementById('viewerImg').src = `/session_file/${name}/verified_sync_alignment.png?t=` + Date.now();
        }
        async function poll() {
            try {
                const r = await fetch('/status');
                const d = await r.json();
                document.getElementById('apiStatus').textContent = d.auth_ok ? '✅ Авторизован' : '⚠️ API ключ не задан';
                document.getElementById('activePodId').textContent = d.target_pod_id || 'Прямое подключение';
                document.getElementById('sshTarget').textContent = `${d.ssh_user}@${d.ssh_host}:${d.ssh_port}`;
                document.getElementById('statusMsg').textContent = d.status_msg;

                const link = d.studio_url || `https://${d.ssh_host}:8080`;
                document.getElementById('studioLink').innerHTML = `<a href="${link}" target="_blank" style="color:#00ffc8;">${link}</a>`;

                const lBox = document.getElementById('liveLogs');
                lBox.textContent = d.live_logs;
                lBox.scrollTop = lBox.scrollHeight;

                const pList = document.getElementById('podList');
                if (!d.all_pods || d.all_pods.length === 0) {
                    pList.innerHTML = '<div style="color:#64748b;">Нет обнаруженных подов через API. Используйте прямой ввод IP выше.</div>';
                } else {
                    pList.innerHTML = d.all_pods.map(p => {
                        const sel = (p.id === d.target_pod_id);
                        return `
                        <div class="pod-row ${sel ? 'selected' : ''}">
                            <span><b>${p.name}</b> (<code>${p.id}</code>) | ${p.status} ($${p.cost}/ч)</span>
                            <button class="btn btn-blue" style="width:auto; padding:4px 8px; margin:0;" onclick="fetch('/pod/select', {method:'POST', body:JSON.stringify({id:'${p.id}'})}).then(poll)">Выбрать</button>
                        </div>`;
                    }).join('');
                }

                const sList = document.getElementById('sessionsList');
                if (!d.sessions || d.sessions.length === 0) {
                    sList.innerHTML = '<div style="color:#64748b;">Нет сохранённых сессий на Orange Pi.</div>';
                } else {
                    sList.innerHTML = d.sessions.map(s => `
                        <div class="pod-row">
                            <span><b>${s.name}</b> | Видео: ${s.video_sec}с | ЭЭГ: ${s.eeg_samples} сэмплов</span>
                            <div style="display:flex; gap:6px;">
                                <button class="btn btn-blue" style="width:auto; padding:4px 8px; margin:0;" onclick="openViewer('${s.name}')">🎬 Видео + ЭЭГ</button>
                                ${s.tar_file ? `<a class="btn btn-green" style="width:auto; padding:4px 8px; margin:0;" href="/downloads/${s.tar_file}">⬇️ .tar.gz</a>` : ''}
                            </div>
                        </div>`).join('');
                }
            } catch(e) {}
        }
        setInterval(poll, 2500);
        poll();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/status")
def status():
    return jsonify({
        "auth_ok": orchestrator.auth_ok,
        "target_pod_id": orchestrator.target_pod_id,
        "ssh_host": orchestrator.ssh_host,
        "ssh_port": orchestrator.ssh_port,
        "ssh_user": orchestrator.ssh_user,
        "studio_url": orchestrator.studio_url,
        "status_msg": orchestrator.status_msg,
        "live_logs": orchestrator.live_logs,
        "all_pods": orchestrator.all_pods,
        "sessions": orchestrator.get_sessions_catalog()
    })


@app.route("/direct_ssh/set", methods=["POST"])
def direct_ssh_set():
    data = request.get_json(force=True)
    if data.get("host"):
        orchestrator.ssh_host = data["host"].strip()
    if data.get("port"):
        orchestrator.ssh_port = int(data["port"])
    orchestrator.status_msg = f"🔧 Переключено на SSH: {orchestrator.ssh_host}:{orchestrator.ssh_port}"
    return jsonify({"status": "ok"})


@app.route("/pod/select", methods=["POST"])
def pod_select():
    data = request.get_json(force=True)
    orchestrator.target_pod_id = data.get("id")
    matched = next((p for p in orchestrator.all_pods if p["id"] == orchestrator.target_pod_id), None)
    if matched:
        orchestrator.active_pod = matched
        if matched["ssh_ip"] and matched["ssh_port"]:
            orchestrator.ssh_host = matched["ssh_ip"]
            orchestrator.ssh_port = matched["ssh_port"]
        orchestrator.studio_url = matched["studio_url"]
        orchestrator.status_msg = f"👉 Выбран под {matched['name']} ({orchestrator.ssh_host}:{orchestrator.ssh_port})"
    return jsonify({"status": "selected"})


@app.route("/stack/start_music", methods=["POST"])
def stack_start_music():
    threading.Thread(target=orchestrator.deploy_and_start_pipeline, args=("music",), daemon=True).start()
    return jsonify({"status": "starting_music", "status_msg": "Запущен конвейер старта MusicGen..."})


@app.route("/stack/start_vision", methods=["POST"])
def stack_start_vision():
    threading.Thread(target=orchestrator.deploy_and_start_pipeline, args=("vision",), daemon=True).start()
    return jsonify({"status": "starting_vision", "status_msg": "Запущен конвейер старта Diffusion..."})


@app.route("/stack/stop", methods=["POST"])
def stack_stop():
    threading.Thread(target=orchestrator.stop_stack, daemon=True).start()
    return jsonify({"status": "stopping", "status_msg": "Остановка процессов..."})


@app.route("/stack/git_pull", methods=["POST"])
def stack_git_pull():
    out = orchestrator.git_pull_explicit()
    return jsonify({"status": "pulled", "output": out, "status_msg": orchestrator.status_msg})


@app.route("/sessions/sync", methods=["POST"])
def sessions_sync():
    threading.Thread(target=orchestrator.sync_sessions, daemon=True).start()
    return jsonify({"status": "syncing", "status_msg": "Синхронизация сессий..."})


@app.route("/downloads/<path:filename>")
def download_file(filename):
    return send_from_directory(str(LOCAL_SESSIONS), filename, as_attachment=True)


@app.route("/session_file/<session_name>/<path:filename>")
def serve_session_file(session_name, filename):
    s_dir = LOCAL_SESSIONS / session_name
    return send_from_directory(str(s_dir), filename)


if __name__ == "__main__":
    cert_f, key_f = ensure_https_certificates()
    print("=" * 70)
    print("🍊 [ORANGE PI RUNPOD REST v6.2] Запущен: https://0.0.0.0:5000")
    print(f"👉 Подключение к поду: {DEFAULT_SSH_HOST}:{DEFAULT_SSH_PORT}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, ssl_context=(cert_f, key_f), threaded=True)
