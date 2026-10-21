#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: ORANGE PI RUNPOD REST & BULLETPROOF ORCHESTRATOR v8.4
===================================================================================
- ВОССТАНОВЛЕН ОБРАЗ DIFFUSION: В меню снова доступен "Diffusion Only" (Python 3.10),
  который нативно поддерживает RTX 5090 без дополнительных патчей.
- ИСПРАВЛЕНИЕ GNUTLS HANDSHAKE: Если Git на поде зависает из-за провайдера (AT&T),
  под сам мгновенно скачивает репозиторий через curl + tar.
- ВОЗВРАЩЕНЫ ВСЕ ФИЧИ: Управление подами, менеджер сессий, автопилот, живые логи.
- МГНОВЕННЫЙ ЗАПУСК: Все процессы отвязаны от SSH, таймаутов больше нет.
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
from collections import deque
from pathlib import Path
import requests
import urllib3
from flask import Flask, request, jsonify, render_template_string, send_from_directory

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RUNPOD_API_KEY    = os.environ.get("RUNPOD_API_KEY", "").strip()
GQL_URL           = "https://api.runpod.io/graphql"

DEFAULT_SSH_HOST  = os.environ.get("RUNPOD_SSH_HOST", "").strip()
DEFAULT_SSH_PORT  = int(os.environ.get("RUNPOD_SSH_PORT", "22"))
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
        self.live_logs = "Ожидание запуска процессов..."
        self.cmd_logs = deque(maxlen=120)
        self.status_msg = "Готов к работе."
        self.is_busy = False
        self.auth_ok = False
        self.stack_mode = "music"

        ensure_ssh_keypair()
        self._log_event("Оркестратор запущен и готов к работе.")
        threading.Thread(target=self._poll_cloud_loop, daemon=True).start()

    def _log_event(self, text: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        print(line, flush=True)
        with self.lock:
            self.cmd_logs.appendleft(line)

    def get_cmd_logs_text(self) -> str:
        with self.lock:
            return "\n".join(self.cmd_logs)

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

    def refresh_pods_sync(self):
        query = """
        query {
            myself {
                id
                pods {
                    id
                    name
                    desiredStatus
                    costPerHr
                    imageName
                    machine { dataCenterId }
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
        if not self.api_key:
            return
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
                    "datacenter": (p.get("machine") or {}).get("dataCenterId") or "Cloud",
                    "image": p.get("imageName", ""),
                    "ssh_ip": s_ip,
                    "ssh_port": s_port,
                    "studio_url": f"https://{p['id']}-8080.proxy.runpod.net"
                })
            self.all_pods = pod_list

            if self.target_pod_id:
                matched = next((p for p in pod_list if p["id"] == self.target_pod_id), None)
                if matched:
                    self.active_pod = matched
                    self.studio_url = matched["studio_url"]
                    if matched["status"] == "RUNNING" and matched["ssh_ip"] and matched["ssh_port"]:
                        self.ssh_host = matched["ssh_ip"]
                        self.ssh_port = matched["ssh_port"]
                    elif matched["status"] != "RUNNING":
                        self.ssh_host = ""
                        self.ssh_port = 0

    def run_ssh(self, cmd: str, timeout: int = 25, silent: bool = False) -> tuple[bool, str]:
        if not self.ssh_host or not self.ssh_port:
            return False, "SSH недоступен (под остановлен или порт 22 еще не открыт)"
        if not silent:
            short_cmd = cmd.strip().replace("\n", " ; ")
            if len(short_cmd) > 140:
                short_cmd = short_cmd[:140] + "..."
            self._log_event(f"👉 SSH CMD: {short_cmd}")

        env = os.environ.copy()
        env["SSH_AUTH_SOCK"] = ""
        full_cmd = [
            "ssh",
            "-i", SSH_PRIV_KEY,
            "-p", str(self.ssh_port),
            "-o", "KexAlgorithms=curve25519-sha256",
            "-o", "IPQoS=throughput",
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{self.ssh_user}@{self.ssh_host}",
            cmd
        ]
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout, env=env)
            out_combined = (r.stdout.strip() + "\n" + r.stderr.strip()).strip()
            if r.returncode == 0:
                if not silent and out_combined:
                    self._log_event(f"💬 OUTPUT: {out_combined[:300]}")
                return True, r.stdout.strip()
            err_msg = out_combined or f"Exit {r.returncode}"
            if not silent:
                self._log_event(f"💥 ОШИБКА SSH: {err_msg[:300]}")
            return False, err_msg
        except Exception as e:
            if not silent:
                self._log_event(f"💥 ТАЙМАУТ/СБОЙ SSH: {e}")
            return False, str(e)

    def _poll_cloud_loop(self):
        while True:
            try:
                self.refresh_pods_sync()
                if self.ssh_host and self.ssh_port:
                    read_log_cmd = (
                        "mkdir -p /workspace/logs && touch /workspace/backend.log && "
                        "echo '=== [SERVER LOG] ===' && tail -n 20 /workspace/logs/server.log 2>/dev/null || true; "
                        "echo '=== [GATEWAY LOG] ===' && tail -n 12 /workspace/logs/gateway.log 2>/dev/null || true; "
                        "echo '=== [ENGINE LOG] ===' && tail -n 20 /workspace/logs/engine.log 2>/dev/null || true; "
                        "tail -n 25 /workspace/backend.log 2>/dev/null || true"
                    )
                    ok_log, out_log = self.run_ssh(read_log_cmd, timeout=8, silent=True)
                    if ok_log and out_log.strip():
                        with self.lock:
                            self.live_logs = out_log.strip()
            except Exception:
                pass
            time.sleep(2.5)

    def create_pod(self, gpu_type: str, mode: str, cloud_type: str = "ALL", volume_gb: int = 60, disk_gb: int = 80):
        if self.is_busy:
            return
        self.is_busy = True
        try:
            target_image = IMAGE_MAP.get(mode, IMAGE_MAP["all"])
            all_pubs = gather_all_local_pub_keys()
            b64_keys = base64.b64encode(all_pubs.encode("utf-8")).decode("ascii")

            start_script = (
                "bash -c '"
                "mkdir -p /var/run/sshd ~/.ssh /workspace/hf_cache /workspace/logs ; "
                f"echo \"{b64_keys}\" | base64 -d >> ~/.ssh/authorized_keys ; "
                "chmod 700 ~/.ssh ; chmod 600 ~/.ssh/authorized_keys ; "
                "/usr/sbin/sshd -D"
                "'"
            )

            m_create = """
            mutation($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    name
                    desiredStatus
                    imageName
                }
            }
            """
            variables = {
                "input": {
                    "cloudType": cloud_type,
                    "gpuCount": 1,
                    "volumeInGb": int(volume_gb),
                    "containerDiskInGb": int(disk_gb),
                    "minVcpuCount": 4,
                    "minMemoryInGb": 16,
                    "gpuTypeId": gpu_type,
                    "name": f"neurocanvas-{mode}",
                    "imageName": target_image,
                    "dockerArgs": start_script,
                    "ports": "8080/http,22/tcp",
                    "volumeMountPath": "/workspace",
                    "supportPublicIp": True,
                    "env": [
                        {"key": "PUBLIC_KEY", "value": all_pubs},
                        {"key": "HF_HOME", "value": "/workspace/hf_cache"},
                        {"key": "PYTHONUNBUFFERED", "value": "1"}
                    ]
                }
            }

            self.status_msg = f"🚀 Создание Pod ({gpu_type}, Image: {mode})..."
            self._log_event(self.status_msg)
            ok, res = self._gql(m_create, variables)
            if not ok:
                raise RuntimeError(str(res))

            created = res.get("podFindAndDeployOnDemand") or {}
            new_id = created.get("id")
            if not new_id:
                raise RuntimeError(f"RunPod вернул пустой ответ: {res}")

            self.target_pod_id = new_id
            self.status_msg = f"✅ Pod {new_id} создан! Ожидание RUNNING..."
            self._log_event(self.status_msg)
            self.refresh_pods_sync()
        except Exception as e:
            self.status_msg = f"❌ Ошибка создания пода: {e}"
            self._log_event(self.status_msg)
        finally:
            self.is_busy = False

    def resume_pod_and_wait(self, pod_id: str) -> bool:
        self.status_msg = f"🚀 Возобновление пода {pod_id}..."
        self._log_event(self.status_msg)
        m_resume = """
        mutation($input: PodResumeInput!) {
            podResume(input: $input) { id desiredStatus }
        }
        """
        ok, res = self._gql(m_resume, {"input": {"podId": pod_id, "gpuCount": 1}})
        if not ok:
            self.status_msg = f"❌ Ошибка podResume: {res}"
            self._log_event(self.status_msg)
            return False

        self.target_pod_id = pod_id
        for sec in range(45):
            time.sleep(2.0)
            self.refresh_pods_sync()
            matched = next((p for p in self.all_pods if p["id"] == pod_id), None)
            if matched:
                self.status_msg = f"⏳ Запуск GPU ({matched['status']})... {sec*2}с"
                if matched["status"] == "RUNNING" and matched["ssh_ip"] and matched["ssh_port"]:
                    self.active_pod = matched
                    self.ssh_host = matched["ssh_ip"]
                    self.ssh_port = matched["ssh_port"]
                    self.studio_url = matched["studio_url"]
                    self.status_msg = f"🟢 Под {pod_id} запущен! SSH: {self.ssh_host}:{self.ssh_port}"
                    self._log_event(self.status_msg)
                    return True

        self.status_msg = f"⚠️ Под {pod_id} запущен, ждем выделения порта SSH..."
        self._log_event(self.status_msg)
        return False

    def select_existing_pod(self, pod_id: str):
        self.target_pod_id = pod_id
        self.refresh_pods_sync()
        matched = next((p for p in self.all_pods if p["id"] == pod_id), None)
        if matched:
            self.active_pod = matched
            self.studio_url = matched["studio_url"]
            if matched["status"] == "RUNNING" and matched["ssh_ip"]:
                self.ssh_host = matched["ssh_ip"]
                self.ssh_port = matched["ssh_port"]
                self.status_msg = f"👉 Выбран активный под {matched['name']} ({self.ssh_host}:{self.ssh_port})"
            else:
                self.ssh_host = ""
                self.ssh_port = 0
                self.status_msg = f"⚠️ Выбран под {matched['name']} [СТАТУС: {matched['status']}]. Нажмите '▶️ Запустить', чтобы разбудить его."
            self._log_event(self.status_msg)

    def deploy_and_start_pipeline(self, mode: str = "music"):
        if self.is_busy:
            return
        self.is_busy = True
        self.stack_mode = mode
        try:
            if self.active_pod and self.active_pod.get("status") != "RUNNING":
                pid = self.active_pod["id"]
                if not self.resume_pod_and_wait(pid):
                    return

            if not self.ssh_host or not self.ssh_port:
                self.status_msg = "❌ Ошибка: SSH-адрес пода не найден. Убедитесь, что под в RUNNING."
                self._log_event(self.status_msg)
                return

            # ШАГ 1: ПРОВЕРКА SSH
            self.status_msg = f"⏳ [1/4] Проверка SSH связи ({self.ssh_host}:{self.ssh_port})..."
            ok_ssh, out_ssh = self.run_ssh("echo PING", timeout=10)
            if not ok_ssh:
                self.status_msg = f"❌ SSH не отвечает: {out_ssh}"
                return

            # ШАГ 2: СИНХРОНИЗАЦИЯ КОДА (Бронебойный метод: скачивание через CURL прямо на поде)
            self.status_msg = "⏳ [2/4] Синхронизация кода с GitHub..."
            self._log_event("Синхронизация репозитория /workspace/neuro_heterarchy...")
            git_prep_cmd = f"""
            mkdir -p /workspace/hf_cache /workspace/logs /workspace/neuro_heterarchy/sessions /workspace/neuro_heterarchy/lore
            touch /workspace/backend.log
            
            git config --global http.version HTTP/1.1
            git config --global http.sslVerify false
            
            if [ ! -d "/workspace/neuro_heterarchy/.git" ]; then
                echo "[DEPLOY] Скачивание {GIT_REPO_URL}..." >> /workspace/backend.log
                rm -rf /workspace/neuro_heterarchy
                git clone {GIT_REPO_URL} /workspace/neuro_heterarchy 2>>/workspace/backend.log || \\
                ( curl -kL {GIT_REPO_URL.replace('.git', '')}/archive/refs/heads/main.tar.gz | tar -xzC /workspace && mv /workspace/neuro_heterarchy-main /workspace/neuro_heterarchy )
            else
                echo "[DEPLOY] Обновление репозитория Git..." >> /workspace/backend.log
                cd /workspace/neuro_heterarchy
                git fetch --all 2>>/workspace/backend.log || true
                git reset --hard origin/HEAD 2>>/workspace/backend.log || true
            fi
            cd /workspace/neuro_heterarchy && git log -1 --oneline >> /workspace/backend.log || echo "Repo synced via curl" >> /workspace/backend.log
            """
            ok_git, out_git = self.run_ssh(git_prep_cmd, timeout=45)
            if not ok_git:
                self.status_msg = f"❌ Ошибка загрузки кода: {out_git[:120]}"
                return

            # ШАГ 3: ЗАВИСИМОСТИ И АВТО-ПАТЧ ДЛЯ RTX 5090
            self.status_msg = "⏳ [3/4] Проверка GPU, PyTorch и зависимостей..."
            deps_cmd = """
            timeout 60 python3 -c "import cv2" 2>/dev/null || timeout 120 pip install --default-timeout=15 --no-cache-dir "opencv-python-headless==4.9.0.80" "numpy<2.0" 2>/dev/null || true
            
            if [ -f /opt/venv_musicgen/bin/python3 ]; then
                GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\\n' || true)
                if [[ "$GPU_NAME" == *"5090"* ]]; then
                    echo "🔥 Обнаружена RTX 5090 (Blackwell)! Обновление PyTorch 2.6 Nightly для поддержки sm_120..." >> /workspace/backend.log
                    /opt/venv_musicgen/bin/python3 -c "import torch; assert int(torch.__version__.split('.')[0]) >= 2 and int(torch.__version__.split('.')[1]) >= 5" 2>/dev/null || \
                    timeout 300 /opt/venv_musicgen/bin/pip install --default-timeout=15 --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 || true
                    
                    echo "Удаление xformers (переход на нативный FlashAttention SDPA)..." >> /workspace/backend.log
                    /opt/venv_musicgen/bin/pip uninstall -y xformers || true
                    
                    python3 -c "
import os
f = '/workspace/neuro_heterarchy/musicgen_server.py'
if os.path.exists(f):
    with open(f, 'r') as file: content = file.read()
    content = content.replace('from xformers import ops', 'try:\\n    from xformers import ops\\nexcept ImportError:\\n    ops = None')
    with open(f, 'w') as file: file.write(content)
" || true
                fi
            fi
            """
            self.run_ssh(deps_cmd, timeout=300)

            # ШАГ 4: ЗАПУСК 3 ПРОЦЕССОВ БЕЗ БЛОКИРОВКИ SSH
            self.status_msg = f"🚀 [4/4] Запуск 3 процессов ({mode.upper()})..."

            run_script = f"""#!/bin/bash
cd /workspace/neuro_heterarchy
> /workspace/logs/server.log
> /workspace/logs/gateway.log
> /workspace/logs/engine.log
> /workspace/backend.log

rm -f cert.pem key.pem /workspace/neuro_heterarchy/cert.pem /workspace/neuro_heterarchy/key.pem

pkill -9 -f 'musicgen_server.py' 2>/dev/null || true
pkill -9 -f 'brain_server.py' 2>/dev/null || true
pkill -9 -f 'web_cloud_gateway.py' 2>/dev/null || true
pkill -9 -f 'neuro_heterarchy_musicgen.py' 2>/dev/null || true
pkill -9 -f 'neuro_heterarchy_diffusion.py' 2>/dev/null || true

export PYTHONUNBUFFERED=1
if [ -f /opt/venv_musicgen/bin/python3 ]; then
    PY_MUSIC="/opt/venv_musicgen/bin/python3"
else
    PY_MUSIC="python3"
fi
PY_VISION="python3"
"""
            if mode == "music":
                run_script += """
nohup stdbuf -oL -eL $PY_MUSIC -u /workspace/neuro_heterarchy/musicgen_server.py --port 6005 --model facebook/musicgen-small </dev/null >>/workspace/logs/server.log 2>&1 &
nohup python3 -u /workspace/neuro_heterarchy/web_cloud_gateway.py --port 8080 --music-lore lore/music_lore.txt --lore lore/world_lore.txt </dev/null >>/workspace/logs/gateway.log 2>&1 &
nohup $PY_MUSIC -u /workspace/neuro_heterarchy/neuro_heterarchy_musicgen.py --headless --mode semantic --lore lore/music_lore.txt --gateway-port 6003 </dev/null >>/workspace/logs/engine.log 2>&1 &
"""
            else:
                run_script += """
nohup stdbuf -oL -eL $PY_VISION -u /workspace/neuro_heterarchy/brain_server.py --mode lcm </dev/null >>/workspace/logs/server.log 2>&1 &
nohup python3 -u /workspace/neuro_heterarchy/web_cloud_gateway.py --port 8080 --music-lore lore/music_lore.txt --lore lore/world_lore.txt </dev/null >>/workspace/logs/gateway.log 2>&1 &
nohup $PY_VISION -u /workspace/neuro_heterarchy/neuro_heterarchy_diffusion.py --headless --mode lcm --steps 2 --lore lore/world_lore.txt </dev/null >>/workspace/logs/engine.log 2>&1 &
"""
            run_script += "\necho 'LAUNCHED_OK'\n"

            b64 = base64.b64encode(run_script.encode("utf-8")).decode("ascii")
            ssh_cmd = f"echo {b64} | base64 -d > /tmp/start_neuro.sh && chmod +x /tmp/start_neuro.sh && nohup /tmp/start_neuro.sh > /dev/null 2>&1 &"
            
            ok_launch, out_launch = self.run_ssh(ssh_cmd, timeout=15)

            if ok_launch:
                studio = self.studio_url or f"https://{self.ssh_host}:8080"
                self.status_msg = f"🟢 Стек запущен! Проверьте логи ниже. Студия: {studio}"
                self._log_event(self.status_msg)
            else:
                self.status_msg = f"⚠️ Ошибка вызова запуска: {out_launch}"
        finally:
            self.is_busy = False

    def git_pull_explicit(self) -> str:
        self.status_msg = "⏳ Выполняю Git Pull на поде..."
        cmd = f"""
        mkdir -p /workspace/neuro_heterarchy
        touch /workspace/backend.log
        git config --global http.version HTTP/1.1
        git config --global http.sslVerify false
        if [ ! -d "/workspace/neuro_heterarchy/.git" ]; then
            git clone {GIT_REPO_URL} /workspace/neuro_heterarchy 2>&1 || \\
            ( curl -kL {GIT_REPO_URL.replace('.git', '')}/archive/refs/heads/main.tar.gz | tar -xzC /workspace && mv /workspace/neuro_heterarchy-main /workspace/neuro_heterarchy )
        else
            cd /workspace/neuro_heterarchy && git fetch --all 2>&1 && git reset --hard origin/HEAD 2>&1
        fi
        echo "✅ Код на поде синхронизирован с GitHub" >> /workspace/backend.log
        """
        ok, out = self.run_ssh(cmd, timeout=30)
        self.status_msg = "✅ Код на поде обновлен!" if ok else f"❌ Ошибка: {out}"
        return out

    def stop_stack(self):
        cmd = (
            "pkill -9 -f 'musicgen_server.py' 2>/dev/null || true; "
            "pkill -9 -f 'brain_server.py' 2>/dev/null || true; "
            "pkill -9 -f 'web_cloud_gateway.py' 2>/dev/null || true; "
            "pkill -9 -f 'neuro_heterarchy_musicgen.py' 2>/dev/null || true; "
            "pkill -9 -f 'neuro_heterarchy_diffusion.py' 2>/dev/null || true; "
            "echo 'STOPPED'"
        )
        self.run_ssh(cmd, timeout=10)
        self.status_msg = "⏹️ Все 3 процесса остановлены."

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
                "scp",
                "-i", SSH_PRIV_KEY,
                "-P", str(self.ssh_port),
                "-o", "KexAlgorithms=curve25519-sha256",
                "-o", "IPQoS=throughput",
                "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
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
    <title>Orange Pi × RunPod Controller v8.4</title>
    <style>
        body { background: #06090e; color: #cbd5e1; font-family: monospace; margin: 0; padding: 14px; }
        .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; margin-bottom: 12px; max-width: 980px; margin-left: auto; margin-right: auto; }
        h2 { margin-top: 0; color: #00ffc8; font-size: 14px; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
        .stat { font-size: 13px; margin: 5px 0; }
        .row { display: flex; gap: 8px; margin-top: 6px; }
        .btn { display: inline-block; padding: 10px; font-size: 12px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; font-family: monospace; text-align: center; box-sizing: border-box; text-decoration: none; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-green  { background: #059669; color: #fff; }
        .btn-blue   { background: #0284c7; color: #fff; }
        .btn-orange { background: #d97706; color: #fff; }
        .btn-red    { background: #e11d48; color: #fff; }
        .btn-gray   { background: #334155; color: #cbd5e1; }
        .log-box { background: #020617; color: #38bdf8; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; font-size: 11px; white-space: pre-wrap; height: 260px; overflow-y: auto; }
        .cmd-box { background: #020617; color: #a7f3d0; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; font-size: 11px; white-space: pre-wrap; height: 180px; overflow-y: auto; }
        .pod-row { display: flex; justify-content: space-between; align-items: center; background: #131d2e; padding: 8px 10px; border-radius: 4px; margin: 6px 0; font-size: 11px; border: 1px solid #334155; }
        .pod-row.selected { border-color: #00ffc8; background: #0e2a38; }
        input, select { padding: 7px; background: #030712; border: 1px solid #334155; color: #fff; border-radius: 4px; font-family: monospace; font-size: 11px; width: 100%; box-sizing: border-box; }
        .badge { padding: 3px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
        .b-running { background: #059669; color: #fff; }
        .b-stopped { background: #d97706; color: #fff; }
        .b-exited  { background: #64748b; color: #fff; }
        label { font-size: 10px; color: #94a3b8; display: block; margin-bottom: 2px; }
        .mt-4 { margin-top: 4px; }
        .mt-6 { margin-top: 6px; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🍊 Orange Pi × RunPod Controller v8.4 (С поддержкой RTX 5090)</h2>
        <div class="stat">🔑 <b>RunPod API:</b> <span id="apiStatus">...</span></div>
        <div class="stat">🎯 <b>Активный Pod:</b> <b id="activePodName" style="color:#00ffc8;">Не выбран</b> (<code><span id="activePodId">...</span></code>)</div>
        <div class="stat">🌐 <b>SSH Эндпоинт:</b> <span id="sshTarget">...</span></div>
        <div class="stat">🔗 <b>Студия (HTTP/HTTPS :8080):</b> <span id="studioLink">Ожидание...</span></div>
        <div class="stat" style="color:#fbbf24; margin-top:6px;" id="statusMsg">...</div>

        <div class="row">
            <input type="text" id="manualHost" placeholder="Прямой SSH IP (например, 68.164.57.7)">
            <input type="number" id="manualPort" placeholder="SSH Port (например, 16741)" style="width:160px;">
            <button class="btn btn-blue" id="btnConnect" style="width:200px;" onclick="saveManualSsh()">🔌 Подключиться</button>
        </div>
    </div>

    <div class="card">
        <h2>➕ Создать НОВЫЙ Pod</h2>
        <div class="row">
            <div style="width:50%;">
                <label>Режим и образ контейнера:</label>
                <select id="createMode">
                    <option value="all" selected>🚀 All-in-One (Общий образ: Diffusion + MusicGen)</option>
                    <option value="music">🎵 MusicGen Only (Python 3.9, torch 2.1 cu121)</option>
                    <option value="vision">🎨 Diffusion Only (Python 3.10)</option>
                </select>
            </div>
            <div style="width:50%;">
                <label>Модель GPU:</label>
                <select id="createGpu">
                    <option value="NVIDIA GeForce RTX 5090" selected>🚀 RTX 5090 (32GB | $1.19/h — Авто-апгрейд PyTorch)</option>
                    <option value="NVIDIA GeForce RTX 4090">⚡ RTX 4090 (24GB | $0.89/h — Стандарт)</option>
                    <option value="RTX 6000 Ada generation">💎 RTX 6000 Ada (48GB | $0.99/h — Серверный 4090)</option>
                    <option value="NVIDIA L40S">⚙️ L40S (48GB | $1.09/h)</option>
                    <option value="NVIDIA GeForce RTX 3090">🐢 RTX 3090 (24GB | $0.50/h)</option>
                    <option value="NVIDIA A100 80GB PCIe">🔥 A100 PCIe (80GB | $1.79/h)</option>
                </select>
            </div>
        </div>
        <div class="row mt-4">
            <div style="width:34%;">
                <label>Тип облака:</label>
                <select id="createCloud">
                    <option value="SECURE" selected>Secure Cloud (Надёжные датацентры)</option>
                    <option value="ALL">Любое (ALL)</option>
                    <option value="COMMUNITY">Community Cloud</option>
                </select>
            </div>
            <div style="width:33%;">
                <label>Постоянный диск (/workspace):</label>
                <input type="number" id="createVolume" value="60" min="20" max="500">
            </div>
            <div style="width:33%;">
                <label>Диск контейнера:</label>
                <input type="number" id="createDisk" value="80" min="20" max="500">
            </div>
        </div>
        <button class="btn btn-green mt-6" onclick="createPodAction(this)">🚀 Создать и Развернуть Под в RunPod</button>
    </div>

    <div class="card">
        <h2>📦 Поды в вашем аккаунте RunPod</h2>
        <div id="podList">Загрузка подов...</div>
    </div>

    <div class="card">
        <h2>⚡ Запуск полного стека на ВЫБРАННОМ поде</h2>
        <div class="row">
            <button class="btn btn-green" style="width:50%;" onclick="runAction('/stack/start_music', this)">
                🎵 СТАРТ MUSICGEN (Server :6005 + Gateway :8080 + Engine :6003)
            </button>
            <button class="btn btn-blue" style="width:50%;" onclick="runAction('/stack/start_vision', this)">
                🎨 СТАРТ DIFFUSION (Server :6000 + Gateway :8080 + Engine :6002)
            </button>
        </div>
        <div class="row">
            <button class="btn btn-orange" style="width:50%;" onclick="runAction('/stack/stop', this)">⏹️ ОСТАНОВИТЬ 3 ПРОЦЕССА</button>
            <button class="btn btn-gray" style="width:50%;" onclick="runAction('/stack/git_pull', this)">🔄 РУЧНОЙ GIT PULL</button>
        </div>
    </div>

    <div class="card">
        <h2>📡 Журнал SSH-команд и ответов оркестратора (Реалтайм)</h2>
        <div class="cmd-box" id="cmdLogs">Загрузка журнала команд...</div>
    </div>

    <div class="card">
        <h2>📜 Единый живой поток логов 3 процессов ([SERVER], [GATEWAY], [ENGINE])</h2>
        <div class="log-box" id="liveLogs">Загрузка логов...</div>
    </div>

    <div class="card">
        <h2>📂 Менеджер сессий на Orange Pi (Видео 1:1 + ЭЭГ)</h2>
        <button class="btn btn-blue" id="btnSync" onclick="runAction('/sessions/sync', this)">
            📥 СИНХРОНИЗИРОВАТЬ СЕССИИ С ПОДА НА ORANGE PI
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
            if (btnEl) { btnEl.disabled = true; btnEl.textContent = "⏳ Выполняется..."; }
            try {
                const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
                const d = await res.json();
                if (d.status_msg) document.getElementById('statusMsg').textContent = d.status_msg;
            } catch(e) { alert("Ошибка запроса: " + e.message); } 
            finally { if (btnEl) { btnEl.disabled = false; btnEl.textContent = oldText; } poll(); }
        }
        async function createPodAction(btnEl) {
            btnEl.disabled = true; btnEl.textContent = "⏳ Отправка запроса...";
            try {
                const payload = {
                    mode: document.getElementById('createMode').value,
                    gpu: document.getElementById('createGpu').value,
                    cloud: document.getElementById('createCloud').value,
                    volume: parseInt(document.getElementById('createVolume').value) || 60,
                    disk: parseInt(document.getElementById('createDisk').value) || 80
                };
                const res = await fetch('/pod/create', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
                const d = await res.json();
                if (d.status_msg) document.getElementById('statusMsg').textContent = d.status_msg;
            } catch(e) {} finally { btnEl.disabled = false; btnEl.textContent = "🚀 Создать и Развернуть Под в RunPod"; poll(); }
        }
        async function saveManualSsh() {
            const host = document.getElementById('manualHost').value.trim();
            const port = parseInt(document.getElementById('manualPort').value);
            if (host && port) { await fetch('/direct_ssh/set', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({host: host, port: port})}); poll(); }
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
                document.getElementById('activePodId').textContent = d.target_pod_id || 'Прямой ввод';
                document.getElementById('activePodName').textContent = d.active_pod_name || 'Прямой ввод';
                document.getElementById('sshTarget').textContent = d.ssh_host ? `${d.ssh_user}@${d.ssh_host}:${d.ssh_port}` : 'Ожидание...';
                document.getElementById('statusMsg').textContent = d.status_msg;

                const link = d.studio_url || (d.ssh_host ? `https://${d.ssh_host}:8080` : '');
                document.getElementById('studioLink').innerHTML = link ? `<a href="${link}" target="_blank" style="color:#00ffc8;">${link}</a>` : 'Оффлайн';
                document.getElementById('cmdLogs').textContent = d.cmd_logs || '...';

                const lBox = document.getElementById('liveLogs');
                lBox.textContent = d.live_logs; lBox.scrollTop = lBox.scrollHeight;

                const pList = document.getElementById('podList');
                if (!d.all_pods || d.all_pods.length === 0) { pList.innerHTML = '<div style="color:#64748b;">Нет подов.</div>'; } 
                else {
                    pList.innerHTML = d.all_pods.map(p => {
                        const sel = (p.id === d.target_pod_id);
                        const isRun = (p.status === 'RUNNING');
                        const badgeClass = isRun ? 'b-running' : (p.status === 'STOPPED' ? 'b-stopped' : 'b-exited');
                        return `
                        <div class="pod-row ${sel ? 'selected' : ''}">
                            <span><b>${p.name}</b> (<code>${p.id}</code>) | <span class="badge ${badgeClass}">${p.status}</span> | <b>$${p.cost}/ч</b> | ${p.ssh_ip ? p.ssh_ip + ':' + p.ssh_port : 'Нет SSH'} ${sel ? ' <b style="color:#00ffc8;">[АКТИВЕН]</b>' : ''}</span>
                            <div style="display:flex; gap:6px;">
                                <button class="btn btn-blue" style="padding:4px 8px; margin:0;" onclick="runAction('/pod/select?id=' + '${p.id}', this)">👉 Выбрать</button>
                                ${!isRun ? `<button class="btn btn-green" style="padding:4px 8px; margin:0;" onclick="runAction('/pod/resume?id=' + '${p.id}', this)">▶️ Запустить</button>` : `<button class="btn btn-orange" style="padding:4px 8px; margin:0;" onclick="runAction('/pod/pause?id=' + '${p.id}', this)">⏸️ Стоп</button>`}
                                <button class="btn btn-red" style="padding:4px 8px; margin:0;" onclick="if(confirm('Удалить под навсегда?')) runAction('/pod/terminate?id=' + '${p.id}', this)">🗑️</button>
                            </div>
                        </div>`;
                    }).join('');
                }
            } catch(e) {}
        }
        setInterval(poll, 2500); poll();
    </script>
</body>
</html>
"""


@app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)

@app.route("/status")
def status():
    return jsonify({
        "auth_ok": orchestrator.auth_ok, "target_pod_id": orchestrator.target_pod_id,
        "active_pod_name": orchestrator.active_pod.get("name") if orchestrator.active_pod else "",
        "ssh_host": orchestrator.ssh_host, "ssh_port": orchestrator.ssh_port, "ssh_user": orchestrator.ssh_user,
        "studio_url": orchestrator.studio_url, "status_msg": orchestrator.status_msg,
        "cmd_logs": orchestrator.get_cmd_logs_text(), "live_logs": orchestrator.live_logs,
        "all_pods": orchestrator.all_pods
    })

@app.route("/direct_ssh/set", methods=["POST"])
def direct_ssh_set():
    data = request.get_json(force=True)
    if data.get("host"):
        orchestrator.target_pod_id, orchestrator.active_pod = None, None
        orchestrator.ssh_host = data["host"].strip()
    if data.get("port"): orchestrator.ssh_port = int(data["port"])
    orchestrator.status_msg = f"🔧 Переключено на ручной SSH: {orchestrator.ssh_host}:{orchestrator.ssh_port}"
    orchestrator._log_event(orchestrator.status_msg)
    return jsonify({"status": "ok"})

@app.route("/pod/select", methods=["POST"])
def pod_select():
    pid = request.args.get("id") or request.get_json(force=True, silent=True).get("id")
    if pid: orchestrator.select_existing_pod(pid)
    return jsonify({"status": "selected"})

@app.route("/pod/create", methods=["POST"])
def pod_create():
    data = request.get_json(force=True)
    threading.Thread(target=orchestrator.create_pod, args=(data.get("gpu", "NVIDIA GeForce RTX 4090"), data.get("mode", "all"), data.get("cloud", "SECURE"), 60, 80), daemon=True).start()
    return jsonify({"status": "creating"})

@app.route("/pod/resume", methods=["POST"])
def pod_resume():
    pid = request.args.get("id") or request.get_json(force=True, silent=True).get("id") or orchestrator.target_pod_id
    if pid: threading.Thread(target=orchestrator.resume_pod_and_wait, args=(pid,), daemon=True).start()
    return jsonify({"status": "resuming"})

@app.route("/pod/pause", methods=["POST"])
def pod_pause():
    pid = request.args.get("id") or request.get_json(force=True, silent=True).get("id") or orchestrator.target_pod_id
    if pid:
        orchestrator._gql("mutation($input: PodStopInput!) { podStop(input: $input) { id } }", {"input": {"podId": pid}})
        orchestrator.status_msg = f"⏸️ Команда остановки отправлена поду {pid}"; orchestrator._log_event(orchestrator.status_msg); orchestrator.refresh_pods_sync()
    return jsonify({"status": "paused"})

@app.route("/pod/terminate", methods=["POST"])
def pod_terminate():
    pid = request.args.get("id") or request.get_json(force=True, silent=True).get("id") or orchestrator.target_pod_id
    if pid:
        orchestrator._gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": pid}})
        orchestrator.status_msg = f"🗑️ Под {pid} удален"; orchestrator._log_event(orchestrator.status_msg); orchestrator.refresh_pods_sync()
    return jsonify({"status": "terminated"})

@app.route("/stack/start_music", methods=["POST"])
def stack_start_music():
    threading.Thread(target=orchestrator.deploy_and_start_pipeline, args=("music",), daemon=True).start()
    return jsonify({"status": "starting_music"})

@app.route("/stack/start_vision", methods=["POST"])
def stack_start_vision():
    threading.Thread(target=orchestrator.deploy_and_start_pipeline, args=("vision",), daemon=True).start()
    return jsonify({"status": "starting_vision"})

@app.route("/stack/stop", methods=["POST"])
def stack_stop():
    threading.Thread(target=orchestrator.stop_stack, daemon=True).start()
    return jsonify({"status": "stopping"})

@app.route("/stack/git_pull", methods=["POST"])
def stack_git_pull():
    orchestrator.git_pull_explicit()
    return jsonify({"status": "pulled"})

@app.route("/sessions/sync", methods=["POST"])
def sessions_sync():
    threading.Thread(target=orchestrator.sync_sessions, daemon=True).start()
    return jsonify({"status": "syncing"})

@app.route("/downloads/<path:filename>")
def download_file(filename):
    return send_from_directory(str(LOCAL_SESSIONS), filename, as_attachment=True)

@app.route("/session_file/<session_name>/<path:filename>")
def serve_session_file(session_name, filename):
    s_dir = LOCAL_SESSIONS / session_name
    return send_from_directory(str(s_dir), filename)


if __name__ == "__main__":
    cert_f, key_f = ensure_https_certificates()
    print("=" * 70); print("🍊 [ORANGE PI RUNPOD REST v8.4] Запущен: https://0.0.0.0:5000"); print("=" * 70)
    app.run(host="0.0.0.0", port=5000, ssl_context=(cert_f, key_f), threaded=True)
