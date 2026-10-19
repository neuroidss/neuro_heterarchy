#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS ORANGE PI HTTPS CLOUD ORCHESTRATOR v2.4 (IMMERS.CLOUD NATIVE)
===================================================================================
1. Веса моделей сохраняются на SSD-диске в /opt/hf_cache (никогда не качаются заново
   после паузы Shelve или пересоздания VM из тома).
2. Автоматический git pull при каждом пробуждении VM + кнопка "🔄 Обновить с GitHub"
   (обновляет код внутри контейнеров за 5 секунд без пересоздания тома!).
3. Мгновенное скачивание сессий одним .tar.gz архивом БЕЗ папки frames/*.jpg
   (только video_timeline.mp4, eeg_raw.npz, events.csv, manifest.json) + ссылка
   на скачивание архива прямо из браузера.
===================================================================================
"""

import os
import sys
import time
import json
import base64
import ipaddress
import subprocess
import threading
from pathlib import Path
import requests
import urllib3
import openstack
from flask import Flask, request, jsonify, render_template_string, send_from_directory

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


# Единый скрипт развертывания, сохранения кэша весов в /opt/hf_cache и обновления с GitHub
DEPLOY_SHELL_COMMANDS = """
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
    echo "Настройка NVIDIA Container Toolkit для Docker..."
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

echo "=== [2/5] Клонирование и свежее обновление с GitHub (git pull) ==="
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

# Привязываем постоянную папку /opt/hf_cache на SSD-диске ко всем контейнерам,
# чтобы веса HuggingFace НИКОГДА не качались заново после паузы или ребута!
sed -i 's|hf_cache:/root/.cache/huggingface|/opt/hf_cache:/root/.cache/huggingface|g' docker-compose.fast.yml || true
sed -i 's|hf_cache:/root/.cache/huggingface|/opt/hf_cache:/root/.cache/huggingface|g' docker-compose.yml || true

echo "=== [3/5] Проверка HTTPS сертификатов ==="
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=neurocanvas.cloud"
fi

echo "=== [4/5] Проверка базового образа neuroidss/neuro_heterarchy_diffusion:latest ==="
if ! docker image inspect neuroidss/neuro_heterarchy_diffusion:latest &> /dev/null; then
    docker pull neuroidss/neuro_heterarchy_diffusion:latest
else
    echo "Базовый образ уже на SSD-диске! Пропускаем долгое скачивание."
fi

echo "=== [5/5] Быстрая сборка кода с GitHub и старт контейнеров ==="
cat << 'EOF' > /etc/systemd/system/neuro-heterarchy.service
[Unit]
Description=NeuroCanvas RTX 4090 Heterarchy Stack
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/neuro_heterarchy
ExecStartPre=-/usr/bin/git fetch --all
ExecStartPre=-/usr/bin/git reset --hard origin/HEAD
ExecStartPre=-/bin/sed -i 's|hf_cache:/root/.cache/huggingface|/opt/hf_cache:/root/.cache/huggingface|g' docker-compose.fast.yml
ExecStart=/usr/bin/docker compose -f docker-compose.fast.yml up -d --build
ExecStop=/usr/bin/docker compose -f docker-compose.fast.yml stop
ExecStopPost=/bin/sync
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable neuro-heterarchy.service
docker compose -f docker-compose.fast.yml up -d --build
sync
echo "=== [ГОТОВО] ВСЕ КОНТЕЙНЕРЫ ЗАПУЩЕНЫ И ВЕСА НА SSD СОХРАНЕНЫ! ==="
"""


def build_cloud_init_script(ssh_pub_key: str, git_repo: str) -> str:
    body = DEPLOY_SHELL_COMMANDS.replace("__GIT_REPO__", git_repo)
    script = f"""#!/bin/bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo "{ssh_pub_key}" >> /root/.ssh/authorized_keys
sed -i 's/.*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config || true
chmod 600 /root/.ssh/authorized_keys

if [ -d "/home/ubuntu" ]; then
    mkdir -p /home/ubuntu/.ssh
    echo "{ssh_pub_key}" >> /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
    chmod 600 /home/ubuntu/.ssh/authorized_keys
fi
{body}
"""
    return base64.b64encode(script.encode("utf-8")).decode("utf-8")


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
        self.last_archive_name = ""
        self.last_sync_count = self._count_local_sessions()

        self.billable_items = []
        self.remote_logs = "Ожидание подключения к виртуальной машине..."
        self.endpoints_status = {"gateway_8080": False, "ssh_22": False}

        threading.Thread(target=self._poll_cloud_loop, daemon=True).start()
        threading.Thread(target=self._poll_ssh_logs_loop, daemon=True).start()

    def _count_local_sessions(self) -> int:
        return len([d for d in LOCAL_SESSIONS.iterdir() if d.is_dir() and d.name.startswith("session")])

    def _get_archives_list(self) -> list[str]:
        return sorted([f.name for f in LOCAL_SESSIONS.glob("*.tar.gz")], reverse=True)

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

    def _run_ssh_cmd(self, ip: str, command: str, timeout: int = 12) -> tuple[bool, str]:
        """Универсальный вызов SSH (сначала пробует ubuntu с sudo, затем root)"""
        for ssh_user in ("ubuntu", "root"):
            try:
                res = subprocess.run([
                    "ssh", "-i", SSH_PRIV_KEY,
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ConnectTimeout=4",
                    "-o", "BatchMode=yes",
                    f"{ssh_user}@{ip}", command
                ], capture_output=True, text=True, timeout=timeout)
                if res.returncode == 0 and "Please login as the user" not in res.stdout:
                    return True, res.stdout
            except Exception:
                pass
        return False, ""

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
            if ip and not is_private_ip(ip):
                ssh_cmd = (
                    "echo '=== 1. КОНТЕЙНЕРЫ DOCKER & КЭШ ВЕСОВ (/opt/hf_cache) ==='; "
                    "sudo docker ps -a --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null || echo 'Docker еще устанавливается...'; "
                    "sudo du -sh /opt/hf_cache /opt/neuro_heterarchy/sessions 2>/dev/null || true; "
                    "echo ''; echo '=== 2. GPU NVIDIA-SMI ==='; "
                    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'GPU инициализируется...'; "
                    "echo ''; echo '=== 3. ЛОГИ КОНТЕЙНЕРОВ (ПОСЛЕДНИЕ СТРОКИ) ==='; "
                    "cd /opt/neuro_heterarchy 2>/dev/null && sudo docker compose -f docker-compose.fast.yml logs --tail=12 2>/dev/null || echo 'Контейнеры еще не запущены'; "
                    "echo ''; echo '=== 4. ЖИВОЙ ЛОГ УСТАНОВКИ (/var/log/neuro_deploy.log) ==='; "
                    "sudo tail -n 18 /var/log/neuro_deploy.log 2>/dev/null || echo 'Ожидание лога...'"
                )
                ok, out = self._run_ssh_cmd(ip, ssh_cmd, timeout=8)
                self.endpoints_status["ssh_22"] = ok
                if ok and out.strip():
                    self.remote_logs = out.strip()
            else:
                self.remote_logs = "Виртуалка выключена или в паузе Shelve..."
            time.sleep(3.5)

    def repair_or_git_pull_on_live_vm(self):
        """Обновляет код с GitHub (git pull) и перезапускает контейнеры за 5 секунд с сохранением /opt/hf_cache!"""
        if not self.vm_ip or is_private_ip(self.vm_ip):
            self.status_msg = "❌ Нет белого IP у виртуалки для SSH-обновления"
            return
        self.status_msg = f"🔄 Обновление кода с GitHub и перезапуск контейнеров на {self.vm_ip}..."
        remote_script = DEPLOY_SHELL_COMMANDS.replace("__GIT_REPO__", GIT_REPO_URL)
        b64_cmd = base64.b64encode(remote_script.encode("utf-8")).decode("utf-8")
        ok, _ = self._run_ssh_cmd(self.vm_ip, f"echo {b64_cmd} | base64 -d | sudo bash &", timeout=10)
        if ok:
            self.status_msg = f"✅ Команда git pull + docker compose build отправлена на {self.vm_ip}! Смотрите лог ниже..."
        else:
            self.status_msg = f"⚠️ Не удалось подключиться по SSH к {self.vm_ip}"

    def sync_sessions_from_vm(self) -> bool:
        """
        Упаковывает все сессии на VM в единый .tar.gz архив БЕЗ папки frames/ и БЕЗ *.jpg
        (только video_timeline.mp4, eeg_raw.npz, events.csv, manifest.json) и качает за 1 секунду!
        """
        if not self.vm_ip or is_private_ip(self.vm_ip):
            return False
        self.status_msg = f"📦 Архивация сессий на {self.vm_ip} (только Видео + ЭЭГ + CSV, без картинок frames/)..."

        pack_cmd = (
            "if [ -d /opt/neuro_heterarchy/sessions ] && [ \"$(ls -A /opt/neuro_heterarchy/sessions 2>/dev/null)\" ]; then "
            "  sudo tar --exclude='frames' --exclude='*.jpg' -czf /tmp/neuro_sessions_clean.tar.gz -C /opt/neuro_heterarchy sessions && "
            "  sudo chmod 644 /tmp/neuro_sessions_clean.tar.gz && echo 'ARCHIVE_READY'; "
            "else "
            "  echo 'NO_SESSIONS'; "
            "fi"
        )
        ok, out = self._run_ssh_cmd(self.vm_ip, pack_cmd, timeout=25)
        if not ok or "NO_SESSIONS" in out:
            self.status_msg = "ℹ️ На виртуалке пока нет записанных сессий в /opt/neuro_heterarchy/sessions"
            return False

        ts_str = time.strftime("%Y%m%d_%H%M%S")
        archive_filename = f"sessions_clean_{ts_str}.tar.gz"
        local_archive_path = LOCAL_SESSIONS / archive_filename

        for ssh_user in ("ubuntu", "root"):
            scp_cmd = [
                "scp", "-i", SSH_PRIV_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=4",
                "-o", "BatchMode=yes",
                f"{ssh_user}@{self.vm_ip}:/tmp/neuro_sessions_clean.tar.gz",
                str(local_archive_path)
            ]
            try:
                r = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
                if r.returncode == 0 and local_archive_path.exists() and local_archive_path.stat().st_size > 100:
                    # Распаковываем рядом в downloaded_sessions/
                    subprocess.run(["tar", "-xzf", str(local_archive_path), "--strip-components=1", "-C", str(LOCAL_SESSIONS)], check=False)
                    self.last_archive_name = archive_filename
                    self.last_sync_count = self._count_local_sessions()
                    sz_mb = local_archive_path.stat().st_size / (1024 * 1024)
                    self.status_msg = f"✅ Архив {archive_filename} ({sz_mb:.1f} МБ, без картинок!) скачан на Orange Pi!"
                    return True
            except Exception:
                pass

        self.status_msg = "⚠️ Не удалось скачать архив сессий по SCP"
        return False

    def _safe_flush_vm_before_stop(self):
        """Перед паузой Shelve или удалением VM аккуратно сбрасывает кэш диска (sync), чтобы не бились веса!"""
        if self.vm_ip and not is_private_ip(self.vm_ip):
            self.status_msg = "💾 Сохранение дискового кэша весов на VM (sync)..."
            self._run_ssh_cmd(
                self.vm_ip,
                "cd /opt/neuro_heterarchy && sudo docker compose -f docker-compose.fast.yml stop && sudo sync",
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
                self.status_msg = f"🟢 Сервер разбужен ({self.vm_ip})! Ждем SSH и запускаем git pull + Docker..."
                # Ждем появления SSH и автоматически обновляем код с GitHub + поднимаем контейнеры!
                for _ in range(25):
                    time.sleep(3.0)
                    ok, _ = self._run_ssh_cmd(self.vm_ip, "echo READY", timeout=4)
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
            self.status_msg = f"🟢 VM запущена ({self.vm_ip})! Проверяем SSH и запускаем git pull + Docker..."

            # На случай если это Warm Boot из старого тома (где cloud-init уже не крутится сам) — толкаем по SSH!
            for _ in range(25):
                time.sleep(3.0)
                ok, _ = self._run_ssh_cmd(self.vm_ip, "echo READY", timeout=4)
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
            self.sync_sessions_from_vm()
            self._safe_flush_vm_before_stop()
            conn = self._get_conn()
            srv = conn.compute.find_server(SERVER_NAME, ignore_missing=True)
            if srv and srv.status == "ACTIVE":
                self.status_msg = "❄️ Перевод в бесплатную паузу (Shelve)..."
                conn.compute.shelve_server(srv)
                self.status_msg = "❄️ Сервер переведен в Shelve (оплата за GPU остановлена, кэш весов сохранен)."
        except Exception as e:
            self.status_msg = f"❌ Ошибка Shelve: {e}"
        finally:
            self.is_busy = False

    def pause_delete_vm_keep_volume(self):
        if self.is_busy:
            return
        self.is_busy = True
        try:
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
            if not skip_sync:
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
                self.status_msg = f"💥 Отправка команды удаления тома {vol.name or vol.id[:8]}..."
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
    <title>Orange Pi HTTPS — Контроллер Облака и Докеров</title>
    <style>
        body { background: #070b12; color: #e2e8f0; font-family: monospace; margin: 0; padding: 16px; }
        .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-bottom: 14px; max-width: 840px; margin-left: auto; margin-right: auto; }
        h2 { margin-top: 0; color: #00ffc8; font-size: 15px; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
        .stat { font-size: 13px; margin: 5px 0; }
        .btn { display: block; width: 100%; padding: 11px; margin: 7px 0; font-size: 13px; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; font-family: monospace; }
        .btn-start  { background: #059669; color: #fff; }
        .btn-repair { background: #4f46e5; color: #fff; }
        .btn-sync   { background: #0d9488; color: #fff; }
        .btn-shelve { background: #0284c7; color: #fff; }
        .btn-pause  { background: #d97706; color: #fff; }
        .btn-nuke   { background: #e11d48; color: #fff; }
        .btn-panic  { background: #991b1b; color: #fef08a; border: 2px solid #ef4444; }
        .log-box { background: #020617; color: #38bdf8; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; font-size: 11px; white-space: pre-wrap; max-height: 340px; overflow-y: auto; }
        .inv-row { display: flex; justify-content: space-between; align-items: center; background: #131d2e; padding: 6px 10px; border-radius: 4px; margin: 4px 0; font-size: 12px; }
        .btn-del-sm { background: #ef4444; color: #fff; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 11px; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .b-ok { background: #059669; color: #fff; }
        .b-wait { background: #475569; color: #cbd5e1; }
        .input-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr auto; gap: 6px; margin-top: 8px; }
        input { padding: 7px; background: #030712; border: 1px solid #334155; color: #fff; border-radius: 4px; font-family: monospace; font-size: 11px; }
        a { color: #00ffc8; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🍊 Orange Pi HTTPS × immers.cloud (RTX 4090 & SSD Orchestrator)</h2>
        <div class="stat">🔑 <b>Проект OpenStack:</b> <span id="authStat">...</span></div>
        <div class="stat">🖥️ <b>Статус GPU VM:</b> <span id="vmStat">...</span></div>
        <div class="stat">💾 <b>Статус Cinder Тома:</b> <span id="volStat">...</span></div>
        <div class="stat">🌐 <b>Белый IP Студии:</b> <span id="vmIp">...</span></div>
        <div class="stat">📦 <b>Сессий на Orange Pi:</b> <span id="syncCnt">0</span> <span id="archLinks"></span></div>
        <div class="stat">📡 <b>Эндпоинты VM:</b> 
            <span class="badge b-wait" id="epSsh">SSH :22</span>
            <span class="badge b-wait" id="epGw">HTTPS Студия :8080</span>
        </div>
        <div class="stat" style="color:#fbbf24; margin-top:8px;" id="msgBox">...</div>

        <details style="margin-top:10px;">
            <summary style="cursor:pointer; color:#94a3b8; font-size:12px;">⚙️ Настройки подключения, Сети (immers) и Диска (ssd)</summary>
            <div class="input-row">
                <input type="text" id="osUser" placeholder="Логин (пусто = не менять)">
                <input type="password" id="osPass" placeholder="Пароль (пусто = не менять)">
                <input type="text" id="osNet" value="immers" placeholder="Сеть (immers)">
                <input type="text" id="osVolType" value="ssd" placeholder="Тип тома (ssd)">
                <button class="btn-del-sm" style="background:#0284c7;" onclick="saveCreds()">ПРИМЕНИТЬ</button>
            </div>
        </details>
    </div>

    <div class="card">
        <h2>📜 Живой прогресс установки и логи контейнеров Docker (по SSH)</h2>
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
            🔄 ОБНОВИТЬ КОД С GITHUB (git pull + пересборка контейнеров за 5 сек без потери весов!)
        </button>
        <button class="btn btn-sync" onclick="apiPost('/cloud/sync')">
            📦 СКАЧАТЬ СЕССИИ СЕЙЧАС АРХИВОМ .TAR.GZ (Видео + ЭЭГ + CSV, БЕЗ картинок frames!)
        </button>
        <button class="btn btn-shelve" onclick="apiPost('/cloud/shelve')">
            ❄️ 2А. БЕСПЛАТНАЯ ПАУЗА SHELVE (Скачать архив сессий + Сохранить кэш + Заморозить GPU)
        </button>
        <button class="btn btn-pause" onclick="apiPost('/cloud/pause')">
            ⏸️ 2Б. УДАЛИТЬ VM, ОСТАВИТЬ SSD-ТОМ С ВЕСАМИ (~2 ₽/ч, быстрый старт за 40с)
        </button>
        <button class="btn btn-nuke" onclick="apiPost('/cloud/nuke')">
            🌙 3. КОНЕЦ ДНЯ: СКАЧАТЬ АРХИВ СЕССИЙ И УДАЛИТЬ ВСЁ В 0 ₽/ЧАС
        </button>
        <button class="btn btn-panic" onclick="if(confirm('Экстренно убить все VM и SSD-тома без ожидания архивации?')) apiPost('/cloud/panic_nuke')">
            🚨 ЭКСТРЕННЫЙ СБРОС ВСЕГО В 0 ₽/ЧАС (МГНОВЕННО, БЕЗ RSYNC/SCP)
        </button>
    </div>

    <script>
        async function apiPost(url, body={}) {
            await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
            pollStatus();
        }
        async function saveCreds() {
            const u = document.getElementById('osUser').value;
            const p = document.getElementById('osPass').value;
            const net = document.getElementById('osNet').value;
            const vt = document.getElementById('osVolType').value;
            await apiPost('/cloud/credentials', {username: u, password: p, net_substr: net, vol_type: vt});
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
                document.getElementById('syncCnt').textContent = d.last_sync_count;
                if (d.archives && d.archives.length > 0) {
                    document.getElementById('archLinks').innerHTML = ' | Последний архив: <a href="/downloads/' + d.archives[0] + '">⬇️ ' + d.archives[0] + '</a>';
                }
                document.getElementById('msgBox').textContent = d.status_msg;
                document.getElementById('logBox').textContent = d.remote_logs;

                const epSsh = document.getElementById('epSsh');
                epSsh.className = 'badge ' + (d.endpoints.ssh_22 ? 'b-ok' : 'b-wait');
                epSsh.textContent = 'SSH :22 ' + (d.endpoints.ssh_22 ? '🟢 OK' : '⏳');

                const epGw = document.getElementById('epGw');
                epGw.className = 'badge ' + (d.endpoints.gateway_8080 ? 'b-ok' : 'b-wait');
                epGw.textContent = 'HTTPS Студия :8080 ' + (d.endpoints.gateway_8080 ? '🟢 ГОТОВА' : '⏳');

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

PROJECTOR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NeuroCanvas Projector Display</title>
    <style>
        body, html { margin: 0; padding: 0; width: 100vw; height: 100vh; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; color: #334155; font-family: monospace; }
        img { width: 100vw; height: 100vh; object-fit: contain; display: none; }
    </style>
</head>
<body>
    <div id="waitMsg">🧠 ОЖИДАНИЕ ПРОБУЖДЕНИЯ ОБЛАЧНОЙ КОРЫ (RTX 4090)...</div>
    <img id="streamImg" alt="">
    <script>
        let currentIp = null;
        const img = document.getElementById('streamImg');
        const msg = document.getElementById('waitMsg');

        setInterval(async () => {
            try {
                const r = await fetch('/cloud/status');
                const d = await r.json();
                if (d.vm_ip && d.endpoints && d.endpoints.gateway_8080) {
                    if (currentIp !== d.vm_ip) {
                        currentIp = d.vm_ip;
                        img.src = `https://${currentIp}:8080/stream.mjpg?t=` + Date.now();
                        img.style.display = 'block';
                        msg.style.display = 'none';
                    }
                } else {
                    currentIp = null;
                    img.style.display = 'none';
                    msg.style.display = 'block';
                    msg.textContent = "⏳ " + (d.status_msg || "Ожидание запуска сессии");
                }
            } catch(e) {}
        }, 2500);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/projector")
def projector():
    return render_template_string(PROJECTOR_HTML)


@app.route("/downloads/<path:filename>")
def download_archive(filename):
    return send_from_directory(str(LOCAL_SESSIONS), filename, as_attachment=True)


@app.route("/cloud/status")
def cloud_status():
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
        "last_sync_count": manager.last_sync_count,
        "archives": manager._get_archives_list()
    })


@app.route("/cloud/credentials", methods=["POST"])
def cloud_credentials():
    data = request.get_json(force=True)
    manager.set_credentials(
        data.get("username", ""),
        data.get("password", ""),
        data.get("net_substr", ""),
        data.get("vol_type", "")
    )
    return jsonify({"status": "updated"})


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


@app.route("/cloud/proxy_ladder", methods=["POST"])
def proxy_ladder():
    if not manager.vm_ip:
        return jsonify({"error": "VM offline"}), 503
    data = request.get_json(force=True)
    try:
        r = requests.post(f"https://{manager.vm_ip}:8080/api/preset/ladder", json=data, verify=False, timeout=3.0)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/eeg_push", methods=["POST"])
def proxy_eeg_push():
    if not manager.vm_ip:
        return jsonify({"status": "vm_offline"}), 200
    data = request.get_data()
    try:
        requests.post(
            f"https://{manager.vm_ip}:8080/api/eeg_push",
            data=data,
            headers={"Content-Type": "application/json"},
            verify=False,
            timeout=0.8
        )
        return jsonify({"status": "ok"})
    except Exception:
        return jsonify({"status": "dropped"}), 200


if __name__ == "__main__":
    cert_f, key_f = ensure_https_certificates()
    print("=" * 70)
    print("🔒 HTTPS Оркестратор запущен: https://0.0.0.0:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, ssl_context=(cert_f, key_f), threaded=True)
