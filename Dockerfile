FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy

# Системные зависимости для Python, OpenCV headless, Pygame и сети
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libasound2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Обновляем pip
RUN pip3 install --no-cache-dir --upgrade pip

# Зависимости Python (pylsl сам подтянет нужные бинарники liblsl)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Открываем порты:
# 6000 - Brain Server IPC
# 6002 - Web Gateway Canvas IPC
# 8080 - Web Cloud Gateway (HTTP / MJPEG)
EXPOSE 6000 6002 8080
