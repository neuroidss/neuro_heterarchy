#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS ORANGE PI HTTPS CLOUD ORCHESTRATOR v3.1 (WITH SESSION MANAGER & VIEWER)
===================================================================================
- Исправлен таймаут SSH (фон запускается через nohup ... </dev/null >/dev/null 2>&1 &
  и возвращает управление за 0.15 сек).
- Полноценный МЕНЕДЖЕР СЕССИЙ на Orange Pi (./downloaded_sessions/):
  * Скачивает каждую сессию отдельно (video_timeline.mp4, eeg_raw.npz, events.csv,
    manifest.json — БЕЗ тысяч мелких картинок frames/*.jpg).
  * Автоматически запускает analyze_session.py и строит график синхронизации ЭЭГ.
  * Позволяет прямо в браузере (даже когда облако полностью удалено в 0 ₽/ч!)
    смотреть видео сессии, график ЭЭГ и скачивать .tar.gz любой выбранной сессии.
===================================================================================
"""

import os
import sys
import time
import json
import base64
import random
import shutil
import ipaddress
import subprocess
import threading
from pathlib import Path
import requests
import urllib3
import openstack
from flask import Flask, Response, request, jsonify, render_template_string, send_from_directory

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AUTH_URL   = os.environ.get("OS_AUTH_URL", "https://api.immers.cloud:5000/v3/").rstrip("/")
USERNAME   = os.environ.get("OS_USERNAME", "").strip()
PASSWORD   = os.environ.get("OS_PASSWORD", "").strip()
PROJECT_ID = os.environ.get("OS_PROJECT_ID", "").strip()
REGION     = os.environ.get("OS_REGION_NAME", "RegionOne").strip()

FLAVOR_SUBSTRING = os.environ.get("NEURO_FLAVOR", "4090").strip()
IMAGE_SUBSTRING  = os.environ.get("NEURO_IMAGE", "Ubuntu").strip()
NETWORK_SUBSTR   = os.environ.get("NEURO_NET", "immers").strip()
VOL_TYPE_SUBSTR  = os.environ.get("NEURO_VOL_TYPE", "ssd").strip()
VOLUME_SIZE_GB   = int(os.environ.get("NEURO_VOL_GB", "80"))
GIT_REPO_URL     = os.environ.get("NEURO_GIT_REPO", "https://github.com/neuroidss/neuro_heterarchy.git").strip()

VOLUME_NAME      = os.environ.get("NEURO_VOL_NAME", "neuro_heterarchy_ssd_vol").strip()
SERVER_NAME      = os.environ.get("NEURO_SRV_NAME", "neuro_rtx4090_worker").strip()
KEYPAIR_NAME     = "orangepi_neuro_key"

SSH_PUB_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519.pub")
SSH_PRIV_KEY     = os.path.expanduser("~/.ssh/id_ed25519")
LOCAL_SESSIONS   = Path(os.environ.get("LOCAL_SESSIONS_DIR", "./downloaded_sessions")).resolve()
LOCAL_SESSIONS.mkdir(parents=True, exist_ok=True)
CUSTOM_PIPELINE_FILE = Path(os.path.expanduser("~/.neurocanvas_custom_pipeline.json"))


def is_private_ip(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).is_private
    except Exception:
        return True


def ensure_https_certificates(cert_path="cert.pem", key_path="key.pem"):
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        print("🔒 Генерация SSL-сертификата (cert.pem, key.pem) для HTTPS...")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-days", "365", "-nodes", "-subj", "/CN=orangepi.local"
        ], check=True)
    return cert_path, key_path


def ensure_ssh_keypair() -> str:
    if not os.path.exists(SSH_PUB_KEY_PATH):
        os.makedirs(os.path.dirname(SSH_PUB_KEY_PATH), exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", SSH_PRIV_KEY, "-N", ""], check=False)
    with open(SSH_PUB_KEY_PATH, "r") as f:
        return f.read().strip()


def load_initial_custom_pipeline() -> list[dict]:
    if CUSTOM_PIPELINE_FILE.exists():
        try:
            with open(CUSTOM_PIPELINE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        import run_meditation_timeline
        if hasattr(run_meditation_timeline, "TIMELINE_STAGES"):
            return list(run_meditation_timeline.TIMELINE_STAGES)
    except Exception:
        pass
    return [
        {"minute": 0.0, "mode": "set", "name": "Стадия 1: Базовый фокус", "concepts": ["silver shining ball of light"]}
    ]


DEPLOY_SHELL_COMMANDS = """#!/bin/bash
exec > /var/log/neuro_deploy.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/5] Проверка Docker и NVIDIA Container Runtime ==="
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
fi

if ! docker compose version &> /dev/null; then
    apt-get update && (apt-get install -y docker-compose-v2 || apt-get install -y docker-compose-plugin || true)
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "Настройка NVIDIA Container Toolkit..."
    if ! command -v nvidia-ctk &> /dev/null; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || true
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list || true
        apt-get update && apt-get install -y nvidia-container-toolkit || true
    fi
    nvidia-ctk runtime configure --runtime=docker || true
    systemctl restart docker || true
fi

echo "=== [2/5] Обновление кода с GitHub (git pull в /opt/neuro_heterarchy) ==="
mkdir -p /opt /opt/hf_cache
if [ ! -d "/opt/neuro_heterarchy/.git" ]; then
    rm -rf /opt/neuro_heterarchy
    git clone "__GIT_REPO__" /opt/neuro_heterarchy
else
    cd /opt/neuro_heterarchy
    git fetch --all || true
    git reset --hard origin/HEAD || git pull || true
fi

cd /opt/neuro_heterarchy
mkdir -p sessions lore /opt/hf_cache

sed -i 's|hf_cache:/root/.cache/huggingface|/opt/hf_cache:/root/.cache/huggingface|g' docker-compose.yml || true

echo "=== [3/5] Генерация сертификатов HTTPS ==="
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=neurocanvas.cloud"
fi

echo "=== [4/5] Проверка готового образа neuroidss/neuro_heterarchy_diffusion:latest ==="
if ! docker image inspect neuroidss/neuro_heterarchy_diffusion:latest &> /dev/null; then
    docker pull neuroidss/neuro_heterarchy_diffusion:latest
else
    echo "Образ уже сохранен на SSD-диске!"
fi

echo "=== [5/5] Очистка старых контейнеров и чистый запуск docker-compose.yml (--no-build) ==="
docker compose -f docker-compose.yml down --remove-orphans || true
docker rm -f neuro_brain_server neuro_web_gateway neuro_canvas_engine neuro_musicgen_server neuro_musicgen_engine 2>/dev/null || true

cat << 'EOF' > /etc/systemd/system/neuro-heterarchy.service
[Unit]
Description=NeuroCanvas RTX 4090 Heterarchy Stack (Zero-Build Direct Mount)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/neuro_heterarchy
ExecStartPre=-/usr/bin/git fetch --all
ExecStartPre=-/usr/bin/git reset --hard origin/HEAD
ExecStartPre=-/bin/sed -i 's|hf_cache:/root/.cache/huggingface|/opt/hf_cache:/root/.cache/huggingface|g' docker-compose.yml
ExecStartPre=-/usr/bin/docker rm -f neuro_brain_server neuro_web_gateway neuro_canvas_engine
ExecStart=/usr/bin/docker compose -f docker-compose.yml up -d --no-build --remove-orphans
ExecStop=/usr/bin/docker compose -f docker-compose.yml stop
ExecStopPost=/bin/sync
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable neuro-heterarchy.service
docker compose -f docker-compose.yml up -d --no-build --remove-orphans
sync
echo "=== [ГОТОВО] ВСЕ КОНТЕЙНЕРЫ ЗАПУЩЕНЫ БЕЗ БИЛДА! ==="
"""


def build_cloud_init_script(ssh_pub_key: str, git_repo: str) -> str:
    body = DEPLOY_SHELL_COMMANDS.replace("__GIT_REPO__", git_repo)
    script = f"""#!/bin/bash
if [ -d "/home/ubuntu" ]; then
    mkdir -p /home/ubuntu/.ssh
    echo "{ssh_pub_key}" >> /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
    chmod 600 /home/ubuntu/.ssh/authorized_keys
fi
{body}
"""
    return base64.b64encode(script.encode("utf-8")).decode("utf-8")


class CustomPipelineRunner:
    def __init__(self, cloud_mgr):
        self.mgr = cloud_mgr
        self.stages = load_initial_custom_pipeline()
        self.is_running = False
        self.stop_event = threading.Event()
        self.current_stage_idx = 0
        self.elapsed_sec = 0.0
        self.total_duration_min = 25.0
        self.override_mode = "as_defined"
        self.max_concepts_per_stage = 0
        self.random_sample = False
        self.status_text = "Протокол не запущен"
        self.active_session_name = ""

    def save_stages(self, new_stages: list[dict]):
        self.stages = new_stages
        try:
            with open(CUSTOM_PIPELINE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.stages, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _vm_api(self, endpoint: str, method: str = "GET", payload: dict = None) -> dict:
        ip = self.mgr.vm_ip
        if not ip:
            return {}
        url = f"https://{ip}:8080{endpoint}"
        try:
            if method == "POST":
                r = requests.post(url, json=payload or {}, verify=False, timeout=4.0)
            else:
                r = requests.get(url, verify=False, timeout=4.0)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def apply_stage_concepts(self, stage: dict):
        raw_concepts = list(stage.get("concepts", []))
        if self.max_concepts_per_stage > 0 and len(raw_concepts) > self.max_concepts_per_stage:
            if self.random_sample:
                target_concepts = random.sample(raw_concepts, self.max_concepts_per_stage)
            else:
                target_concepts = raw_concepts[:self.max_concepts_per_stage]
        else:
            target_concepts = raw_concepts

        mode = stage.get("mode", "set")
        if self.override_mode == "always_set":
            mode = "set"
        elif self.override_mode == "always_add":
            mode = "add"

        if mode == "set":
            res = self._vm_api("/api/world/concepts/set", "POST", {"concepts": target_concepts})
            if not res:
                current_pool = self._vm_api("/api/world/concepts", "GET").get("concepts", [])
                for c in current_pool:
                    if c not in target_concepts:
                        self._vm_api("/api/world/concepts/remove", "POST", {"concept": c})
                        time.sleep(0.04)
                for c in target_concepts:
                    if c not in current_pool:
                        self._vm_api("/api/world/concepts/add", "POST", {"concept": c})
                        time.sleep(0.04)
        else:
            current_pool = self._vm_api("/api/world/concepts", "GET").get("concepts", [])
            for c in target_concepts:
                if c not in current_pool:
                    self._vm_api("/api/world/concepts/add", "POST", {"concept": c})
                    time.sleep(0.04)

        self.status_text = f"Активна [{stage.get('name', 'Фаза')}]: {', '.join(target_concepts[:4])}"

    def start_protocol(self, session_name: str = "meditation", speedup: float = 1.0,
                       override_mode: str = "as_defined", max_pick: int = 0, random_pick: bool = False):
        if self.is_running:
            return
        if not self.mgr.vm_ip or not self.mgr.endpoints_status.get("gateway_8080"):
            self.status_text = "❌ Сначала дождитесь готовности HTTPS Студии :8080!"
            return

        self.override_mode = override_mode
        self.max_concepts_per_stage = max(0, int(max_pick))
        self.random_sample = bool(random_pick)
        self.active_session_name = session_name or "meditation"
        self.stop_event.clear()
        self.is_running = True

        threading.Thread(target=self._run_loop, args=(max(0.1, float(speedup)),), daemon=True).start()

    def stop_protocol(self):
        self.stop_event.set()

    def trigger_next_stage_now(self):
        if self.is_running and self.current_stage_idx + 1 < len(self.stages):
            self.current_stage_idx += 1
            self.apply_stage_concepts(self.stages[self.current_stage_idx])

    def reshuffle_current_stage(self):
        if self.stages and 0 <= self.current_stage_idx < len(self.stages):
            self.apply_stage_concepts(self.stages[self.current_stage_idx])

    def _run_loop(self, speedup: float):
        try:
            rec_res = self._vm_api("/api/recording/start", "POST", {"name": self.active_session_name})
            s_path = rec_res.get("session_path", "sessions/active")
            self.status_text = f"🔴 Запись включена ({s_path}). Старт протокола..."

            t_start = time.time()
            last_minute = self.stages[-1].get("minute", 21.0) if self.stages else 21.0
            self.total_duration_min = last_minute + 4.0

            applied_indices = set()
            while not self.stop_event.is_set():
                self.elapsed_sec = (time.time() - t_start) * speedup
                cur_min = self.elapsed_sec / 60.0

                for idx, stg in enumerate(self.stages):
                    if cur_min >= float(stg.get("minute", 0.0)) and idx not in applied_indices:
                        applied_indices.add(idx)
                        self.current_stage_idx = idx
                        self.apply_stage_concepts(stg)

                if cur_min >= self.total_duration_min:
                    break
                time.sleep(0.5)
        finally:
            self._vm_api("/api/recording/stop", "POST")
            self.is_running = False
            self.status_text = "✅ Протокол завершен! Автоматическое скачивание сессии на Orange Pi..."
            time.sleep(1.0)
            self.mgr.sync_sessions_from_vm()


class OpenStackNeuroManager:
    def __init__(self):
        self.auth_url = AUTH_URL
        self.username = USERNAME
        self.password = PASSWORD
        self.project_id = PROJECT_ID
        self.region = REGION

        self.flavor_substr = FLAVOR_SUBSTRING
        self.image_substr = IMAGE_SUBSTRING
        self.net_substr = NETWORK_SUBSTR
        self.vol_type_substr = VOL_TYPE_SUBSTR

        self.status_msg = "Инициализация..."
        self.vm_ip = None
        self.vm_status = "UNKNOWN"
        self.vol_status = "UNKNOWN"
        self.is_busy = False
        self.auth_ok = False

        self.billable_items = []
        self.remote_logs = "Ожидание подключения к виртуальной машине..."
        self.endpoints_status = {"gateway_8080": False, "ssh_22": False}
        self.pipeline = CustomPipelineRunner(self)

        threading.Thread(target=self._poll_cloud_loop, daemon=True).start()
        threading.Thread(target=self._poll_ssh_logs_loop, daemon=True).start()

    def get_local_sessions_catalog(self) -> list[dict]:
        """Возвращает список всех скачанных на Orange Pi сессий с их метаданными для Менеджера Сессий"""
        catalog = []
        for d in sorted(LOCAL_SESSIONS.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").exists():
                try:
                    with open(d / "manifest.json", "r", encoding="utf-8") as f:
                        man = json.load(f)
                    tar_name = f"{d.name}.tar.gz"
                    has_tar = (LOCAL_SESSIONS / tar_name).exists()
                    has_plot = (d / "verified_sync_alignment.png").exists()
                    has_vid = (d / "video_timeline.mp4").exists()
                    catalog.append({
                        "name": d.name,
                        "duration_sec": round(float(man.get("duration_seconds", 0.0)), 1),
                        "video_sec": round(float(man.get("video_duration_seconds", 0.0)), 1),
                        "eeg_samples": int(man.get("eeg_samples_count", 0)),
                        "frames": int(man.get("unique_generated_frames", 0)),
                        "created_at": man.get("created_at", ""),
                        "tar_file": tar_name if has_tar else "",
                        "has_plot": has_plot,
                        "has_video": has_vid
                    })
                except Exception:
                    pass
        return catalog

    def discover_project_id_if_needed(self) -> str:
        if self.project_id and self.project_id != "ВАШ_PROJECT_ID":
            return self.project_id
        if not self.username or not self.password:
            raise RuntimeError("Не заданы OS_USERNAME и OS_PASSWORD")

        payload = {
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {
                        "user": {
                            "name": self.username,
                            "domain": {"id": "default"},
                            "password": self.password
                        }
                    }
                }
            }
        }
        r = requests.post(f"{self.auth_url}/auth/tokens", json=payload, verify=False, timeout=10.0)
        if r.status_code == 401:
            payload["auth"]["identity"]["password"]["user"]["domain"] = {"name": "Default"}
            r = requests.post(f"{self.auth_url}/auth/tokens", json=payload, verify=False, timeout=10.0)

        if r.status_code not in (200, 201):
            raise RuntimeError(f"Ошибка авторизации HTTP {r.status_code}")

        token = r.headers.get("X-Subject-Token")
        body = r.json().get("token", {})
        if "project" in body and "id" in body["project"]:
            self.project_id = body["project"]["id"]
            return self.project_id

        rp = requests.get(f"{self.auth_url}/auth/projects", headers={"X-Auth-Token": token}, verify=False, timeout=10.0)
        if rp.status_code == 200:
            enabled = [p for p in rp.json().get("projects", []) if p.get("enabled", True)]
            if enabled:
                self.project_id = enabled[0]["id"]
                return self.project_id
        raise RuntimeError("Не найдено активных проектов в аккаунте")

    def _get_conn(self):
        pid = self.discover_project_id_if_needed()
        return openstack.connect(
            auth_url=self.auth_url,
            username=self.username,
            password=self.password,
            project_id=pid,
            user_domain_id="default",
            project_domain_id="default",
            region_name=self.region,
            verify=False
        )

    def set_credentials(self, username: str, password: str, net_substr: str = "", vol_type: str = ""):
        if username.strip():
            self.username = username.strip()
        if password.strip():
            self.password = password.strip()
            self.project_id = ""
        if net_substr.strip():
            self.net_substr = net_substr.strip()
        if vol_type.strip():
            self.vol_type_substr = vol_type.strip()
        self.auth_ok = False
        self.status_msg = "🔄 Применение новых настроек..."

    def _extract_ips(self, server) -> tuple[str | None, str | None]:
        pub_ip, priv_ip = None, None
        if not server or not server.addresses:
            return None, None
        for _, addr_list in server.addresses.items():
            for a in addr_list:
                if a.get("version") == 4:
                    addr = a.get("addr", "")
                    if not is_private_ip(addr) or a.get("OS-EXT-IPS:type") == "floating":
                        pub_ip = addr
                    else:
                        priv_ip = addr
        return pub_ip, priv_ip

    def _select_best_network(self, conn):
        nets = list(conn.network.networks())
        if not nets:
            raise RuntimeError("В проекте нет доступных сетей!")
        if self.net_substr:
            matched = [n for n in nets if self.net_substr.lower() in n.name.lower()]
            if matched:
                return matched[0]
        ext_nets = [n for n in nets if getattr(n, "is_router_external", False)]
        if ext_nets:
            for en in ext_nets:
                if "immers" in en.name.lower():
                    return en
            return ext_nets[0]
        return nets[0]

    def _select_ssd_volume_type(self, conn) -> str | None:
        try:
            vtypes = list(conn.block_storage.types())
            if not vtypes:
                return None
            target_sub = (self.vol_type_substr or "ssd").lower()
            for vt in vtypes:
                name_low = (vt.name or "").lower()
                desc_low = (getattr(vt, "description", "") or "").lower()
                if target_sub in name_low or target_sub in desc_low:
                    return vt.name
            return vtypes[0].name
        except Exception:
            return None

    def _ensure_security_group_ports(self, conn):
        try:
            for sg in conn.network.security_groups(project_id=self.project_id):
                if sg.name == "default":
                    for port_num in (22, 8080):
                        try:
                            conn.network.create_security_group_rule(
                                security_group_id=sg.id,
                                direction="ingress",
                                ethertype="IPv4",
                                protocol="tcp",
                                port_range_min=port_num,
                                port_range_max=port_num,
                                remote_ip_prefix="0.0.0.0/0"
                            )
                        except Exception:
                            pass
        except Exception:
            pass

    def _check_gateway_alive(self, ip: str) -> bool:
        if not ip:
            return False
        try:
            r = requests.get(f"https://{ip}:8080/api/world/concepts", verify=False, timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _run_ssh_ubuntu(self, ip: str, command: str, timeout: int = 12) -> tuple[bool, str]:
        try:
            res = subprocess.run([
                "ssh", "-i", SSH_PRIV_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=4",
                "-o", "BatchMode=yes",
                f"ubuntu@{ip}", command
            ], capture_output=True, text=True, timeout=timeout)
            if res.returncode == 0:
                return True, res.stdout
            return False, f"SSH Exit {res.returncode}: {res.stderr.strip()} {res.stdout.strip()}"
        except Exception as e:
            return False, str(e)

    def _poll_cloud_loop(self):
        while True:
            try:
                conn = self._get_conn()
                self.auth_ok = True

                items = []
                servers = list(conn.compute.servers())
                volumes = list(conn.block_storage.volumes())
                fips = list(conn.network.ips(project_id=self.project_id))

                for s in servers:
                    pub, priv = self._extract_ips(s)
                    items.append({
                        "type": "server", "id": s.id, "name": s.name,
                        "detail": f"Статус: {s.status} | Белый IP: {pub or 'НЕТ'} {f'(Серый: {priv})' if priv else ''}"
                    })
                for v in volumes:
                    v_type = getattr(v, "volume_type", "") or "disk"
                    items.append({
                        "type": "volume", "id": v.id, "name": v.name or v.id[:8],
                        "detail": f"Тип: {v_type} | Размер: {v.size} ГБ | Статус: {v.status.upper()}"
                    })
                for f in fips:
                    items.append({
                        "type": "fip", "id": f.id, "name": f.floating_ip_address,
                        "detail": f"Порт: {'Привязан' if f.port_id else 'СВОБОДЕН (тарифицируется)'}"
                    })
                self.billable_items = items

                srv = next((s for s in servers if s.name == SERVER_NAME), servers[0] if servers else None)
                vol = next((v for v in volumes if v.name == VOLUME_NAME), volumes[0] if volumes else None)

                if vol:
                    v_t = getattr(vol, "volume_type", "")
                    self.vol_status = f"{vol.status.upper()} ({vol.size} ГБ, {v_t})"
                else:
                    self.vol_status = "ОТСУТСТВУЕТ (0 ₽/ч)"

                if srv:
                    pub_ip, priv_ip = self._extract_ips(srv)
                    self.vm_ip = pub_ip or priv_ip
                    gw_ok = self._check_gateway_alive(self.vm_ip)
                    self.endpoints_status["gateway_8080"] = gw_ok

                    if srv.status in ("SHELVED", "SHELVED_OFFLOADED"):
                        self.vm_status = f"❄️ В ПАУЗЕ ({srv.status} — GPU 0 ₽/ч)"
                    elif gw_ok:
                        self.vm_status = "🟢 ГОТОВА К РАБОТЕ (RTX 4090 АКТИВНА)"
                        if not self.is_busy:
                            self.status_msg = f"🟢 Студия активна: https://{self.vm_ip}:8080"
                    else:
                        ip_tag = f"Белый IP: {pub_ip}" if pub_ip else f"Серый IP: {priv_ip}"
                        self.vm_status = f"⏳ СТАРТ КОНТЕЙНЕРОВ ({srv.status}) | {ip_tag}"
                else:
                    self.vm_ip = None
                    self.endpoints_status = {"gateway_8080": False, "ssh_22": False}
                    self.vm_status = "ОТСУТСТВУЕТ (GPU 0 ₽/ч)"
                    if not self.is_busy:
                        if not items:
                            self.status_msg = "💤 В облаке 0 ресурсов (Тарификация: 0.00 ₽/час)"
                        elif len(volumes) > 0 and all(v.status == "deleting" for v in volumes):
                            self.status_msg = "⏳ Диск доудаляется в облаке (скоро станет 0.00 ₽/час)..."
            except Exception as e:
                self.auth_ok = False
                if not self.is_busy:
                    self.status_msg = f"⚠️ Ошибка OpenStack: {e}"
            time.sleep(3.0)

    def _poll_ssh_logs_loop(self):
        while True:
            ip = self.vm_ip
            if ip and not is_private_ip(ip) and "SHELVED" not in self.vm_status:
                ssh_cmd = (
                    "echo '=== 1. КОНТЕЙНЕРЫ DOCKER & ЗАПИСАННЫЕ СЕССИИ НА VM ==='; "
                    "sudo docker ps -a --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null || echo 'Docker еще устанавливается...'; "
                    "echo 'Папка сессий на VM:'; ls -lh /opt/neuro_heterarchy/sessions 2>/dev/null || echo 'Пусто'; "
                    "echo 'Кэш весов (/opt/hf_cache):'; sudo du -sh /opt/hf_cache 2>/dev/null || true; "
                    "echo ''; echo '=== 2. GPU NVIDIA-SMI ==='; "
                    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'GPU инициализируется...'; "
                    "echo ''; echo '=== 3. ЛОГИ КОНТЕЙНЕРОВ (ПОСЛЕДНИЕ СТРОКИ) ==='; "
                    "cd /opt/neuro_heterarchy 2>/dev/null && sudo docker compose -f docker-compose.yml logs --tail=12 2>/dev/null || echo 'Контейнеры еще не запущены'; "
                    "echo ''; echo '=== 4. ЖИВОЙ ЛОГ УСТАНОВКИ (/var/log/neuro_deploy.log) ==='; "
                    "sudo tail -n 16 /var/log/neuro_deploy.log 2>/dev/null || echo 'Ожидание лога...'"
                )
                ok, out = self._run_ssh_ubuntu(ip, ssh_cmd, timeout=8)
                self.endpoints_status["ssh_22"] = ok
                if ok and out.strip():
                    self.remote_logs = out.strip()
            else:
                self.endpoints_status["ssh_22"] = False
                self.remote_logs = "Виртуалка выключена или в паузе Shelve (SSH недоступен, когда VM на паузе)..."
            time.sleep(3.5)

    def repair_or_git_pull_on_live_vm(self):
        """
        Отправляет скрипт на VM и запускает его через nohup с закрытием дескрипторов
        (</dev/null >/dev/null 2>&1 &), чтобы SSH возвращался мгновенно за 0.15 сек без таймаута!
        """
        if not self.vm_ip or is_private_ip(self.vm_ip) or "SHELVED" in self.vm_status:
            self.status_msg = "❌ Сначала разбудите виртуалку (в режиме Shelve SSH выключен)!"
            return
        self.status_msg = f"🔄 Обновление с GitHub (git pull) и чистый перезапуск контейнеров на {self.vm_ip}..."
        remote_script = DEPLOY_SHELL_COMMANDS.replace("__GIT_REPO__", GIT_REPO_URL)
        b64_cmd = base64.b64encode(remote_script.encode("utf-8")).decode("utf-8")

        launch_cmd = (
            f"echo {b64_cmd} | base64 -d | sudo tee /tmp/neuro_deploy.sh > /dev/null && "
            f"sudo chmod +x /tmp/neuro_deploy.sh && "
            f"sudo nohup bash /tmp/neuro_deploy.sh > /var/log/neuro_deploy.log 2>&1 < /dev/null &"
        )
        ok, err = self._run_ssh_ubuntu(self.vm_ip, launch_cmd, timeout=10)
        if ok:
            self.status_msg = f"✅ Код с GitHub обновлен на {self.vm_ip} (без билда)! Смотрите лог ниже..."
        else:
            self.status_msg = f"⚠️ Ошибка SSH к ubuntu@{self.vm_ip}: {err}"

    def sync_sessions_from_vm(self) -> bool:
        """
        Скачивает все сессии с 4090 на Orange Pi (./downloaded_sessions/),
        создает по-сессионные архивы .tar.gz (без папки frames/) и строит график analyze_session.py!
        """
        if not self.vm_ip or is_private_ip(self.vm_ip):
            self.status_msg = "❌ Нет активного белого IP у виртуалки для скачивания"
            return False
        if "SHELVED" in self.vm_status:
            self.status_msg = "❌ Виртуалка сейчас в паузе Shelve! Сначала разбудите её кнопкой '1. Запустить', чтобы забрать файлы."
            return False

        self.status_msg = f"🔍 Поиск записанных сессий на ubuntu@{self.vm_ip}..."
        list_cmd = "sudo find /opt/neuro_heterarchy/sessions -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null"
        ok, out = self._run_ssh_ubuntu(self.vm_ip, list_cmd, timeout=10)
        if not ok:
            self.status_msg = f"❌ Ошибка SSH при проверке сессий: {out}"
            return False

        session_names = [line.strip() for line in out.splitlines() if line.strip().startswith("session") or line.strip()]
        if not session_names:
            self.status_msg = "ℹ️ Папка /opt/neuro_heterarchy/sessions на VM пуста."
            return False

        downloaded_now = 0
        for s_name in session_names:
            remote_tar = f"/tmp/{s_name}.tar.gz"
            local_tar = LOCAL_SESSIONS / f"{s_name}.tar.gz"
            self.status_msg = f"📥 Упаковка и скачивание сессии '{s_name}' (без картинок frames/)..."

            pack_one = (
                f"sudo tar --exclude='frames' --exclude='*.jpg' -czf {remote_tar} -C /opt/neuro_heterarchy/sessions {s_name} && "
                f"sudo chmod 644 {remote_tar}"
            )
            ok_p, _ = self._run_ssh_ubuntu(self.vm_ip, pack_one, timeout=25)
            if not ok_p:
                continue

            scp_cmd = [
                "scp", "-i", SSH_PRIV_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=4",
                "-o", "BatchMode=yes",
                f"ubuntu@{self.vm_ip}:{remote_tar}",
                str(local_tar)
            ]
            r = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and local_tar.exists():
                subprocess.run(["tar", "-xzf", str(local_tar), "-C", str(LOCAL_SESSIONS)], check=False)
                # Автоматически запускаем analyze_session.py для генерации графика синхронизации ЭЭГ и видео!
                s_dir = LOCAL_SESSIONS / s_name
                if s_dir.exists() and Path("analyze_session.py").exists():
                    subprocess.run([sys.executable, "analyze_session.py", str(s_dir)], check=False)
                downloaded_now += 1

        self.status_msg = f"✅ Синхронизировано сессий на Orange Pi: {downloaded_now} шт. Доступны в Менеджере Сессий ниже!"
        return downloaded_now > 0

    def _safe_flush_vm_before_stop(self):
        if self.vm_ip and not is_private_ip(self.vm_ip) and "SHELVED" not in self.vm_status:
            self.status_msg = "💾 Остановка контейнеров и сброс кэша весов на SSD (sync)..."
            self._run_ssh_ubuntu(
                self.vm_ip,
                "cd /opt/neuro_heterarchy && sudo docker compose -f docker-compose.yml stop && sudo sync",
                timeout=20
            )

    def start_or_warm_boot_vm(self):
        if self.is_busy:
            return
        self.is_busy = True
        try:
            conn = self._get_conn()
            self._ensure_security_group_ports(conn)

            pub_key = ensure_ssh_keypair()
            if not conn.compute.find_keypair(KEYPAIR_NAME, ignore_missing=True):
                conn.compute.create_keypair(name=KEYPAIR_NAME, public_key=pub_key)

            srv = conn.compute.find_server(SERVER_NAME, ignore_missing=True)
            if srv and srv.status in ("SHELVED", "SHELVED_OFFLOADED"):
                self.status_msg = "🚀 Пробуждение RTX 4090 из паузы Shelve..."
                conn.compute.unshelve_server(srv)
                srv = conn.compute.wait_for_server(srv, status="ACTIVE", interval=3, wait=300)
                pub_ip, priv_ip = self._extract_ips(srv)
                self.vm_ip = pub_ip or priv_ip
                self.status_msg = f"🟢 Сервер разбужен ({self.vm_ip})! Ждем SSH и обновляем с GitHub..."
                for _ in range(25):
                    time.sleep(3.0)
                    ok, _ = self._run_ssh_ubuntu(self.vm_ip, "echo READY", timeout=4)
                    if ok:
                        self.repair_or_git_pull_on_live_vm()
                        break
                return

            if srv:
                pub_ip, priv_ip = self._extract_ips(srv)
                self.vm_ip = pub_ip or priv_ip
                self.repair_or_git_pull_on_live_vm()
                return

            flavors = list(conn.compute.flavors())
            flavor = next((f for f in flavors if self.flavor_substr.lower() in f.name.lower()), None)
            if not flavor:
                raise RuntimeError(f"Флейвор '{self.flavor_substr}' не найден!")

            net = self._select_best_network(conn)

            vol = conn.block_storage.find_volume(VOLUME_NAME, ignore_missing=True)
            if not vol:
                images = list(conn.compute.images())
                img = next((im for im in images if self.image_substr.lower() in im.name.lower()), None)
                if not img:
                    raise RuntimeError(f"Образ '{self.image_substr}' не найден!")
                req_gb = max(VOLUME_SIZE_GB, getattr(img, "min_disk", 0) or 0, getattr(flavor, "disk", 0) or 0)
                ssd_type = self._select_ssd_volume_type(conn)

                self.status_msg = f"🌅 Создание быстрого SSD-тома ({ssd_type or 'default'}, {req_gb} ГБ)..."
                vol_kwargs = {"name": VOLUME_NAME, "size": req_gb, "image_id": img.id}
                if ssd_type:
                    vol_kwargs["volume_type"] = ssd_type

                vol = conn.block_storage.create_volume(**vol_kwargs)
                conn.block_storage.wait_for_status(vol, status="available", failures=["error"], interval=2, wait=600)
                self.status_msg = f"✅ SSD-том создан! Запускаем RTX 4090 в сети '{net.name}'..."
            else:
                self.status_msg = f"🚀 Быстрый старт (Warm Boot) из готового SSD-тома в сети '{net.name}'..."

            user_data_b64 = build_cloud_init_script(pub_key, GIT_REPO_URL)
            bdm = [{
                "uuid": vol.id,
                "source_type": "volume",
                "destination_type": "volume",
                "boot_index": 0,
                "delete_on_termination": False
            }]

            srv = conn.compute.create_server(
                name=SERVER_NAME,
                flavor_id=flavor.id,
                networks=[{"uuid": net.id}],
                key_name=KEYPAIR_NAME,
                block_device_mapping_v2=bdm,
                user_data=user_data_b64
            )
            srv = conn.compute.wait_for_server(srv, interval=2, wait=600)

            pub_ip, priv_ip = self._extract_ips(srv)
            self.vm_ip = pub_ip or priv_ip
            self.status_msg = f"🟢 VM запущена ({self.vm_ip})! Ждем SSH и запускаем контейнеры..."

            for _ in range(25):
                time.sleep(3.0)
                ok, _ = self._run_ssh_ubuntu(self.vm_ip, "echo READY", timeout=4)
                if ok:
                    self.repair_or_git_pull_on_live_vm()
                    break
        except Exception as e:
            self.status_msg = f"❌ Ошибка запуска: {e}"
        finally:
            self.is_busy = False

    def shelve_vm_free_pause(self):
        if self.is_busy:
            return
        self.is_busy = True
        try:
            if "SHELVED" not in self.vm_status:
                self.sync_sessions_from_vm()
                self._safe_flush_vm_before_stop()
            conn = self._get_conn()
            srv = conn.compute.find_server(SERVER_NAME, ignore_missing=True)
            if srv and srv.status == "ACTIVE":
                self.status_msg = "❄️ Перевод в бесплатную паузу (Shelve)..."
                conn.compute.shelve_server(srv)
                self.status_msg = "❄️ Сервер переведен в Shelve (GPU 0 ₽/ч, кэш весов /opt/hf_cache сохранен)."
        except Exception as e:
            self.status_msg = f"❌ Ошибка Shelve: {e}"
        finally:
            self.is_busy = False

    def pause_delete_vm_keep_volume(self):
        if self.is_busy:
            return
        self.is_busy = True
        try:
            if "SHELVED" not in self.vm_status:
                self.sync_sessions_from_vm()
                self._safe_flush_vm_before_stop()
            conn = self._get_conn()
            for srv in conn.compute.servers():
                self.status_msg = f"⏸️ Удаление VM {srv.name}..."
                conn.compute.delete_server(srv, ignore_missing=True)
                conn.compute.wait_for_delete(srv, interval=2, wait=180)
            self.vm_ip = None
            self.status_msg = "✅ VM удалена, SSD-том (с весами в /opt/hf_cache) сохранен!"
        except Exception as e:
            self.status_msg = f"❌ Ошибка паузы: {e}"
        finally:
            self.is_busy = False

    def force_nuke_all(self, skip_sync: bool = False):
        self.is_busy = True
        try:
            if not skip_sync and "SHELVED" not in self.vm_status:
                self.sync_sessions_from_vm()

            conn = self._get_conn()
            for srv in list(conn.compute.servers()):
                self.status_msg = f"💥 Удаление VM {srv.name}..."
                conn.compute.delete_server(srv, ignore_missing=True, force=True)
            for srv in list(conn.compute.servers()):
                try:
                    conn.compute.wait_for_delete(srv, interval=2, wait=120)
                except Exception:
                    pass

            for fip in list(conn.network.ips(project_id=self.project_id)):
                try:
                    conn.network.delete_ip(fip)
                except Exception:
                    pass

            for vol in list(conn.block_storage.volumes()):
                self.status_msg = f"💥 Удаление тома {vol.name or vol.id[:8]}..."
                for _ in range(15):
                    v_cur = conn.block_storage.get_volume(vol.id)
                    if v_cur.status in ("available", "error", "error_deleting"):
                        break
                    time.sleep(1.5)
                try:
                    conn.block_storage.delete_volume(vol, ignore_missing=True, force=True)
                except Exception:
                    conn.block_storage.delete_volume(vol, ignore_missing=True)

            for _ in range(25):
                rem_vols = list(conn.block_storage.volumes())
                if not rem_vols:
                    break
                self.status_msg = f"⏳ Ожидание завершения удаления тома в Cinder ({rem_vols[0].status})..."
                time.sleep(2.0)

            self.vm_ip = None
            self.remote_logs = "Все ресурсы удалены."
            self.status_msg = "💤 ЧИСТО! Все VM и SSD-тома полностью удалены (0.00 ₽/час)."
        except Exception as e:
            self.status_msg = f"❌ Ошибка очистки: {e}"
        finally:
            self.is_busy = False

    def delete_single_item(self, item_type: str, item_id: str):
        try:
            conn = self._get_conn()
            if item_type == "server":
                conn.compute.delete_server(item_id, ignore_missing=True, force=True)
            elif item_type == "volume":
                conn.block_storage.delete_volume(item_id, ignore_missing=True, force=True)
            elif item_type == "fip":
                conn.network.delete_ip(item_id)
            self.status_msg = f"🗑️ Отправлено удаление {item_type}:{item_id[:8]}"
        except Exception as e:
            self.status_msg = f"❌ Не удалось удалить {item_type}: {e}"


manager = OpenStackNeuroManager()
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>Orange Pi HTTPS — Контроллер Облака и Сессий</title>
    <style>
        body { background: #070b12; color: #e2e8f0; font-family: monospace; margin: 0; padding: 16px; }
        .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-bottom: 14px; max-width: 880px; margin-left: auto; margin-right: auto; }
        h2 { margin-top: 0; color: #00ffc8; font-size: 15px; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
        .stat { font-size: 13px; margin: 5px 0; }
        .btn { display: block; width: 100%; padding: 11px; margin: 7px 0; font-size: 13px; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; font-family: monospace; text-align: center; text-decoration: none; box-sizing: border-box; }
        .btn-start  { background: #059669; color: #fff; }
        .btn-repair { background: #4f46e5; color: #fff; }
        .btn-sync   { background: #0d9488; color: #fff; }
        .btn-shelve { background: #0284c7; color: #fff; }
        .btn-pause  { background: #d97706; color: #fff; }
        .btn-nuke   { background: #e11d48; color: #fff; }
        .btn-panic  { background: #991b1b; color: #fef08a; border: 2px solid #ef4444; }
        .log-box { background: #020617; color: #38bdf8; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; font-size: 11px; white-space: pre-wrap; max-height: 280px; overflow-y: auto; }
        .inv-row { display: flex; justify-content: space-between; align-items: center; background: #131d2e; padding: 8px 10px; border-radius: 4px; margin: 5px 0; font-size: 12px; gap: 8px; }
        .btn-sm { background: #0284c7; color: #fff; border: none; padding: 5px 9px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 11px; text-decoration: none; }
        .btn-del-sm { background: #ef4444; color: #fff; border: none; padding: 5px 9px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 11px; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .b-ok { background: #059669; color: #fff; }
        .b-wait { background: #475569; color: #cbd5e1; }
        .pipe-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px; margin: 8px 0; }
        input, select, textarea { padding: 7px; background: #030712; border: 1px solid #334155; color: #fff; border-radius: 4px; font-family: monospace; font-size: 11px; width: 100%; box-sizing: border-box; }
        a { color: #00ffc8; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🍊 Orange Pi HTTPS × immers.cloud (RTX 4090 & Session Manager)</h2>
        <div class="stat">🔑 <b>Проект OpenStack:</b> <span id="authStat">...</span></div>
        <div class="stat">🖥️ <b>Статус GPU VM:</b> <span id="vmStat">...</span></div>
        <div class="stat">💾 <b>Статус Cinder Тома:</b> <span id="volStat">...</span></div>
        <div class="stat">🌐 <b>Белый IP Студии:</b> <span id="vmIp">...</span></div>
        <div class="stat">📡 <b>Эндпоинты VM:</b> 
            <span class="badge b-wait" id="epSsh">SSH :22</span>
            <span class="badge b-wait" id="epGw">HTTPS Студия :8080</span>
        </div>
        <div class="stat" style="color:#fbbf24; margin-top:8px;" id="msgBox">...</div>
    </div>

    <div class="card">
        <h2>🎬 Авто-Пилот Протокола Сессии (С Авто-Записью Видео 1:1 и ЭЭГ)</h2>
        <div class="stat" style="color:#00ffc8;" id="pipeStat">Протокол не запущен</div>
        <div class="pipe-grid">
            <div>
                <label style="font-size:10px; color:#94a3b8;">Имя сессии:</label>
                <input type="text" id="pName" value="meditation_session">
            </div>
            <div>
                <label style="font-size:10px; color:#94a3b8;">Режим смены концептов:</label>
                <select id="pMode">
                    <option value="as_defined">По сценарию (SET / ++)</option>
                    <option value="always_set">Всегда ЗАМЕНА (SET)</option>
                    <option value="always_add">Всегда НАКОПЛЕНИЕ (++)</option>
                </select>
            </div>
            <div>
                <label style="font-size:10px; color:#94a3b8;">Сколько брать из фазы (0=все):</label>
                <input type="number" id="pMaxPick" value="0" min="0" max="20">
            </div>
            <div>
                <label style="font-size:10px; color:#94a3b8;">Выборка при K &gt; 0:</label>
                <select id="pRand">
                    <option value="0">Первые K по порядку</option>
                    <option value="1">Случайные K из фазы</option>
                </select>
            </div>
        </div>
        <div style="display:flex; gap:8px;">
            <button class="btn btn-start" onclick="startPipeline()">▶️ СТАРТ ПРОТОКОЛА + АВТО-ЗАПИСЬ СЕССИИ</button>
            <button class="btn btn-nuke" style="width:240px;" onclick="apiPost('/pipeline/stop')">⏹️ СТОП (СОХРАНИТЬ СЕССИЮ)</button>
        </div>
        <div style="display:flex; gap:8px;">
            <button class="btn btn-shelve" onclick="apiPost('/pipeline/next')">⏭️ Следующая фаза сейчас</button>
            <button class="btn btn-repair" onclick="apiPost('/pipeline/reshuffle')">🎲 Перевыбрать K концептов текущей фазы</button>
        </div>
    </div>

    <div class="card">
        <h2>📂 Менеджер Сессий на Orange Pi (Доступны даже после удаления облака!)</h2>
        <button class="btn btn-sync" onclick="apiPost('/cloud/sync')">
            📥 СИНХРОНИЗИРОВАТЬ НОВЫЕ СЕССИИ С 4090 НА ORANGE PI СЕЙЧАС (Видео + ЭЭГ + CSV, без картинок)
        </button>
        <div id="sessionsCatalog">Загрузка списка сохраненных сессий...</div>
        <div id="sessionViewerBox" style="display:none; margin-top:12px; background:#020617; padding:12px; border-radius:8px; border:1px solid #38bdf8;">
            <h3 id="viewerTitle" style="margin-top:0; color:#00ffc8;">Просмотр сессии</h3>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                <div>
                    <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;">Видео таймлайн 1:1 (video_timeline.mp4):</div>
                    <video id="viewerVideo" controls style="width:100%; border-radius:6px; background:#000;"></video>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;">Синхронизация ЭЭГ и кадров (analyze_session.py):</div>
                    <img id="viewerPlot" style="width:100%; border-radius:6px; background:#111;" alt="График ЭЭГ строится...">
                </div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>📜 Живой прогресс установки и логи контейнеров Docker (ubuntu@VM)</h2>
        <div class="log-box" id="logBox">Подключение к логам виртуальной машины...</div>
    </div>

    <div class="card">
        <h2>💰 Тарифицируемые ресурсы в проекте (Обновляется каждые 3 сек)</h2>
        <div id="invList">Загрузка списка ресурсов...</div>
    </div>

    <div class="card">
        <h2>⚡ Управление жизненным циклом и GitHub</h2>
        <button class="btn btn-start" onclick="apiPost('/cloud/start')">
            🚀 1. ЗАПУСТИТЬ / РАЗБУДИТЬ RTX 4090 (Сеть 'immers' + Быстрый SSD Том + Кэш Весов)
        </button>
        <button class="btn btn-repair" onclick="apiPost('/cloud/repair_docker')">
            🔄 ОБНОВИТЬ КОД С GITHUB (git pull + перезапуск за 3 сек БЕЗ docker build и без потери весов!)
        </button>
        <button class="btn btn-shelve" onclick="apiPost('/cloud/shelve')">
            ❄️ 2А. БЕСПЛАТНАЯ ПАУЗА SHELVE (Скачать сессии на Orange Pi + Заморозить GPU в 0 ₽/ч)
        </button>
        <button class="btn btn-pause" onclick="apiPost('/cloud/pause')">
            ⏸️ 2Б. УДАЛИТЬ VM, ОСТАВИТЬ SSD-ТОМ С ВЕСАМИ (~2 ₽/ч, быстрый старт за 40с)
        </button>
        <button class="btn btn-nuke" onclick="apiPost('/cloud/nuke')">
            🌙 3. КОНЕЦ ДНЯ: СКАЧАТЬ СЕССИИ НА ORANGE PI И УДАЛИТЬ ВСЁ В 0 ₽/ЧАС
        </button>
        <button class="btn btn-panic" onclick="if(confirm('Экстренно убить все VM и SSD-тома без ожидания архивации?')) apiPost('/cloud/panic_nuke')">
            🚨 ЭКСТРЕННЫЙ СБРОС ВСЕГО В 0 ₽/ЧАС (МГНОВЕННО, БЕЗ ОЖИДАНИЯ)
        </button>
    </div>

    <script>
        async function apiPost(url, body={}) {
            await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
            pollStatus();
        }
        async function startPipeline() {
            await apiPost('/pipeline/start', {
                name: document.getElementById('pName').value,
                override_mode: document.getElementById('pMode').value,
                max_pick: parseInt(document.getElementById('pMaxPick').value) || 0,
                random_pick: document.getElementById('pRand').value === "1"
            });
        }
        function openSessionViewer(sName) {
            const box = document.getElementById('sessionViewerBox');
            box.style.display = 'block';
            document.getElementById('viewerTitle').textContent = '📊 Просмотр сессии: ' + sName;
            document.getElementById('viewerVideo').src = `/session_file/${sName}/video_timeline.mp4?t=` + Date.now();
            document.getElementById('viewerPlot').src = `/session_file/${sName}/verified_sync_alignment.png?t=` + Date.now();
        }
        async function delLocalSession(sName) {
            if (!confirm('Удалить локальную копию сессии ' + sName + ' с Orange Pi?')) return;
            await apiPost('/local_session/delete', {name: sName});
        }
        async function delItem(t, id) {
            await apiPost('/cloud/delete_item', {type: t, id: id});
        }
        async function pollStatus() {
            try {
                const r = await fetch('/cloud/status');
                const d = await r.json();
                document.getElementById('authStat').textContent = d.auth_ok ? `✅ OK (${d.project_id})` : '❌ Ошибка авторизации';
                document.getElementById('vmStat').textContent = d.vm_status;
                document.getElementById('volStat').textContent = d.vol_status;
                document.getElementById('vmIp').innerHTML = d.vm_ip ? `<a href="https://${d.vm_ip}:8080" target="_blank">https://${d.vm_ip}:8080 (Открыть студию)</a>` : 'Оффлайн';
                document.getElementById('msgBox').textContent = d.status_msg;
                document.getElementById('logBox').textContent = d.remote_logs;
                document.getElementById('pipeStat').textContent = d.pipeline_status;

                const epSsh = document.getElementById('epSsh');
                epSsh.className = 'badge ' + (d.endpoints.ssh_22 ? 'b-ok' : 'b-wait');
                epSsh.textContent = 'SSH :22 ' + (d.endpoints.ssh_22 ? '🟢 OK' : '⏳');

                const epGw = document.getElementById('epGw');
                epGw.className = 'badge ' + (d.endpoints.gateway_8080 ? 'b-ok' : 'b-wait');
                epGw.textContent = 'HTTPS Студия :8080 ' + (d.endpoints.gateway_8080 ? '🟢 ГОТОВА' : '⏳');

                const cat = document.getElementById('sessionsCatalog');
                if (!d.sessions_catalog || d.sessions_catalog.length === 0) {
                    cat.innerHTML = '<div style="color:#64748b; margin-top:6px;">На Orange Pi пока нет скачанных сессий.</div>';
                } else {
                    cat.innerHTML = d.sessions_catalog.map(s => `
                        <div class="inv-row">
                            <span>
                                <b>${s.name}</b> | Видео: ${s.video_sec}с | ЭЭГ: ${s.eeg_samples} сэмплов | Кадров: ${s.frames}
                            </span>
                            <div style="display:flex; gap:6px;">
                                <button class="btn-sm" onclick="openSessionViewer('${s.name}')">🎬 Видео + ЭЭГ</button>
                                ${s.tar_file ? `<a class="btn-sm" style="background:#059669;" href="/downloads/${s.tar_file}">⬇️ .tar.gz</a>` : ''}
                                <button class="btn-del-sm" onclick="delLocalSession('${s.name}')">🗑️</button>
                            </div>
                        </div>
                    `).join('');
                }

                const inv = document.getElementById('invList');
                if (!d.billable_items || d.billable_items.length === 0) {
                    inv.innerHTML = '<div style="color:#10b981;">В облаке 0 ресурсов (Расход: 0.00 ₽/час)</div>';
                } else {
                    inv.innerHTML = d.billable_items.map(it => `
                        <div class="inv-row">
                            <span><b>[${it.type.toUpperCase()}]</b> ${it.name} — ${it.detail}</span>
                            <button class="btn-del-sm" onclick="delItem('${it.type}', '${it.id}')">🗑️ Удалить</button>
                        </div>
                    `).join('');
                }
            } catch(e) {}
        }
        setInterval(pollStatus, 2000);
        pollStatus();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/downloads/<path:filename>")
def download_archive(filename):
    return send_from_directory(str(LOCAL_SESSIONS), filename, as_attachment=True)


@app.route("/session_file/<session_name>/<path:filename>")
def serve_session_file(session_name, filename):
    s_dir = LOCAL_SESSIONS / session_name
    if filename == "verified_sync_alignment.png" and not (s_dir / filename).exists():
        if Path("analyze_session.py").exists():
            subprocess.run([sys.executable, "analyze_session.py", str(s_dir)], check=False)
    return send_from_directory(str(s_dir), filename)


@app.route("/local_session/delete", methods=["POST"])
def delete_local_session():
    data = request.get_json(force=True)
    s_name = data.get("name", "")
    if s_name and "/" not in s_name and ".." not in s_name:
        shutil.rmtree(LOCAL_SESSIONS / s_name, ignore_errors=True)
        tar_f = LOCAL_SESSIONS / f"{s_name}.tar.gz"
        if tar_f.exists():
            tar_f.unlink()
    return jsonify({"status": "deleted"})


@app.route("/cloud/status")
def cloud_status():
    p = manager.pipeline
    pipe_str = (
        f"🟢 ИДЕТ СЕССИЯ ({p.elapsed_sec/60.0:.1f}/{p.total_duration_min:.1f} мин) | {p.status_text}"
        if p.is_running else f"⏹️ {p.status_text}"
    )
    return jsonify({
        "auth_ok": manager.auth_ok,
        "project_id": manager.project_id[:10] + "..." if manager.project_id else "",
        "vm_ip": manager.vm_ip,
        "vm_status": manager.vm_status,
        "vol_status": manager.vol_status,
        "status_msg": manager.status_msg,
        "remote_logs": manager.remote_logs,
        "endpoints": manager.endpoints_status,
        "billable_items": manager.billable_items,
        "sessions_catalog": manager.get_local_sessions_catalog(),
        "pipeline_status": pipe_str
    })


@app.route("/pipeline/start", methods=["POST"])
def pipeline_start():
    data = request.get_json(force=True)
    manager.pipeline.start_protocol(
        session_name=data.get("name", "meditation"),
        speedup=float(data.get("speedup", 1.0)),
        override_mode=data.get("override_mode", "as_defined"),
        max_pick=int(data.get("max_pick", 0)),
        random_pick=bool(data.get("random_pick", False))
    )
    return jsonify({"status": "started"})


@app.route("/pipeline/stop", methods=["POST"])
def pipeline_stop():
    manager.pipeline.stop_protocol()
    return jsonify({"status": "stopping"})


@app.route("/pipeline/next", methods=["POST"])
def pipeline_next():
    manager.pipeline.trigger_next_stage_now()
    return jsonify({"status": "next_triggered"})


@app.route("/pipeline/reshuffle", methods=["POST"])
def pipeline_reshuffle():
    manager.pipeline.reshuffle_current_stage()
    return jsonify({"status": "reshuffled"})


@app.route("/cloud/start", methods=["POST"])
def cloud_start():
    threading.Thread(target=manager.start_or_warm_boot_vm, daemon=True).start()
    return jsonify({"status": "starting"})


@app.route("/cloud/repair_docker", methods=["POST"])
def cloud_repair_docker():
    threading.Thread(target=manager.repair_or_git_pull_on_live_vm, daemon=True).start()
    return jsonify({"status": "repairing"})


@app.route("/cloud/sync", methods=["POST"])
def cloud_sync():
    threading.Thread(target=manager.sync_sessions_from_vm, daemon=True).start()
    return jsonify({"status": "syncing"})


@app.route("/cloud/shelve", methods=["POST"])
def cloud_shelve():
    threading.Thread(target=manager.shelve_vm_free_pause, daemon=True).start()
    return jsonify({"status": "shelving"})


@app.route("/cloud/pause", methods=["POST"])
def cloud_pause():
    threading.Thread(target=manager.pause_delete_vm_keep_volume, daemon=True).start()
    return jsonify({"status": "pausing"})


@app.route("/cloud/nuke", methods=["POST"])
def cloud_nuke():
    threading.Thread(target=manager.force_nuke_all, args=(False,), daemon=True).start()
    return jsonify({"status": "nuking"})


@app.route("/cloud/panic_nuke", methods=["POST"])
def cloud_panic_nuke():
    threading.Thread(target=manager.force_nuke_all, args=(True,), daemon=True).start()
    return jsonify({"status": "panic_nuking"})


@app.route("/cloud/delete_item", methods=["POST"])
def cloud_delete_item():
    data = request.get_json(force=True)
    threading.Thread(target=manager.delete_single_item, args=(data.get("type"), data.get("id")), daemon=True).start()
    return jsonify({"status": "deleting"})


if __name__ == "__main__":
    cert_f, key_f = ensure_https_certificates()
    print("=" * 70)
    print("🔒 HTTPS Оркестратор v3.1 запущен: https://0.0.0.0:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, ssl_context=(cert_f, key_f), threaded=True)
