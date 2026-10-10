#!/usr/bin/env python3
"""
===================================================================================
NEUROCANVAS: AI QUEST AGENT SIMULATION SCRIPT (OPENAI PROTOCOL COMPLIANT)
===================================================================================
Имитирует внешнего автономного агента-проводника (ChatGPT / Custom GPT Action).
Сценарий:
  1. Старт с нулевого уровня (0 концептов, чистый свободный латентный дрейф).
  2. Запуск сессионной записи (ЭЭГ + Видео 1:1 + Маркеры).
  3. Акт 1 (Водная стихия): добавление двух концептов ["Wave", "Whirlpool"].
  4. Считывание телеметрии мозга: опрос фокуса внимания и 2D-координат SWM.
  5. Акт 2 (Огненная трансформация): удаление двух старых и ввод двух новых ["Flame tongue", "Lava flow"].
  6. Музыкальная модуляция: переключение саундтрека на "deep resonance drone".
  7. Финал: удаление концептов (возврат к свободному дрейфу) и остановка записи.
===================================================================================
"""

import time
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8080"  # Или https://<IP>:8080 при включенном SSL

def api_call(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"❌ [API ОШИБКА]: {endpoint} -> {e}")
        return {}

def log_agent(speaker: str, text: str):
    print(f"\n[{time.strftime('%H:%M:%S')}] 🗣️  <{speaker}>: {text}")

def main():
    print("=" * 70)
    print("🤖 СИМУЛЯЦИЯ ИИ-ПРОВОДНИКА КВЕСТА / МЕДИАЦИИ (OPENAI PROTOCOL)")
    print(f"🔗 Подключение к шлюзу: {BASE_URL}")
    print("=" * 70)

    # 0. Проверка доступности API
    current_concepts = api_call("/api/world/concepts").get("concepts", [])
    print(f"📋 Текущий пул концептов мира: {current_concepts}")

    # 1. Старт сессионной записи
    log_agent("System", "Инициализация квестовой сессии. Запуск записи...")
    rec_resp = api_call("/api/recording/start", method="POST", payload={"name": "dream_quest_act1"})
    print(f"🔴 Статус записи: {rec_resp.get('status')} | Папка: {rec_resp.get('session_path')}")

    # 2. Акт 1: Погружение в Водную Стихию (Добавление первых 2 концептов)
    log_agent("Паймон-Гид", "Здравствуй, сновидец. Начнём погружение с глубинных вод. Вглядись в пену волн и вращение водоворота...")
    
    time.sleep(1.5)
    api_call("/api/world/concepts/add", method="POST", payload={"concept": "Wave"})
    time.sleep(0.5)
    api_call("/api/world/concepts/add", method="POST", payload={"concept": "Whirlpool"})
    api_call("/api/music/concepts/add", method="POST", payload={"concept": "flowing water ambient"})

    print("✅ Добавлены концепты: ['Wave', 'Whirlpool']")
    print(f"📋 Активный пул мира: {api_call('/api/world/concepts').get('concepts')}")

    # 3. Фаза созерцания и опрос телеметрии мозга
    log_agent("Паймон-Гид", "Я считываю твои мозговые волны... Удерживай внимание на вращении.")
    for step in range(3):
        time.sleep(3.0)
        telem = api_call("/api/world/telemetry")
        puck = telem.get("puck", [0.0, 0.0])
        thought = telem.get("active_thought", "...")
        print(f"   🧠 [Телеметрия мозга] Доминирующий образ: '{thought}' | 2D Ранги SWM: [{puck[0]:+.2f}, {puck[1]:+.2f}]")

    # 4. Акт 2: Фазовый переход в Огонь (Удаление 2 старых, добавление 2 новых)
    log_agent("Паймон-Гид", "Вода испаряется! Из глубины прорывается жар. Забудь о волнах — перед тобой языки пламени и раскаленная лава!")

    # Удаление концептов Воды
    api_call("/api/world/concepts/remove", method="POST", payload={"concept": "Wave"})
    api_call("/api/world/concepts/remove", method="POST", payload={"concept": "Whirlpool"})
    print("🗑️ Удалены концепты Воды: ['Wave', 'Whirlpool']")

    # Добавление концептов Огня
    api_call("/api/world/concepts/add", method="POST", payload={"concept": "Flame tongue"})
    time.sleep(0.5)
    api_call("/api/world/concepts/add", method="POST", payload={"concept": "Lava flow"})
    
    # Смена саундтрека
    api_call("/api/music/concepts/remove", method="POST", payload={"concept": "flowing water ambient"})
    api_call("/api/music/concepts/add", method="POST", payload={"concept": "crackling fire hypnotic pulses"})

    print("✅ Добавлены концепты Огня: ['Flame tongue', 'Lava flow']")
    print(f"📋 Активный пул мира: {api_call('/api/world/concepts').get('concepts')}")

    # 5. Опрос реакции коры на новую стихию
    for step in range(3):
        time.sleep(3.0)
        telem = api_call("/api/world/telemetry")
        puck = telem.get("puck", [0.0, 0.0])
        thought = telem.get("active_thought", "...")
        print(f"   🧠 [Телеметрия мозга] Доминирующий образ: '{thought}' | 2D Ранги SWM: [{puck[0]:+.2f}, {puck[1]:+.2f}]")

    # 6. Завершение медитации (Очистка в свободный поток)
    log_agent("Паймон-Гид", "Огонь затихает. Мы отпускаем формы и растворяем образы. Твой разум возвращается в свободное парение...")
    api_call("/api/world/concepts/remove", method="POST", payload={"concept": "Flame tongue"})
    api_call("/api/world/concepts/remove", method="POST", payload={"concept": "Lava flow"})
    print(f"📋 Пул полностью очищен (Свободный латентный дрейф): {api_call('/api/world/concepts').get('concepts')}")

    time.sleep(3.0)
    stop_resp = api_call("/api/recording/stop", method="POST")
    print(f"\n⏹️ Запись завершена! Сохранённый датасет: {stop_resp.get('saved_path')}")
    print("🎯 Сценарий успешно выполнен без единого перезапуска сервисов!")

if __name__ == '__main__':
    main()
