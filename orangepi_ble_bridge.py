#!/usr/bin/env python3
"""
===================================================================================
ORANGE PI AUTONOMOUS BLE-TO-CLOUD FORWARDER (ZERO UI PAIRING NEEDED)
===================================================================================
Ловит плату FreeEEG16/24 по Bluetooth в зале галереи и шлет батчи по 250 Гц
на локальный оркестратор Orange Pi (http://localhost:5000/api/eeg_push), который
автоматически перенаправляет их на текущий IP облачной виртуалки RTX 4090!
===================================================================================
"""

import asyncio
import argparse
import requests
from bleak import BleakScanner, BleakClient

SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8"


async def run_ble_forwarder(role: str, gain: int, proxy_url: str):
    uv_scale = (1.2 / float(gain) / 8388607.0) * 1e6
    gain_hex_map = {4: 0x22, 8: 0x33, 16: 0x44, 32: 0x55}
    g_hex = gain_hex_map.get(gain, 0x22)

    session = requests.Session()
    batch = []

    def on_data(sender, data: bytearray):
        if len(data) != 51 or data[0] != 0xA0 or data[50] != 0xC0:
            return
        sample = []
        for i in range(16):
            k = 2 + i * 3
            val = (data[k] << 16) | (data[k+1] << 8) | data[k+2]
            if val & 0x800000:
                val -= 0x1000000
            sample.append(val * uv_scale)
        batch.append(sample)

    while True:
        print("🔍 [ORANGE PI BLE] Поиск включенной платы FreeEEG16 в зале...")
        devices = await BleakScanner.discover(timeout=3.0, service_uuids=[SERVICE_UUID])
        if not devices:
            await asyncio.sleep(1.0)
            continue

        target = devices[0]
        print(f"✅ Найдена плата {target.address}! Подключение...")
        try:
            async with BleakClient(target.address, timeout=15.0) as client:
                await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x04, g_hex, g_hex]), response=False)
                await asyncio.sleep(0.05)
                await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x05, g_hex, g_hex]), response=False)
                await asyncio.sleep(0.05)

                await client.start_notify(DATA_CHAR_UUID, on_data)
                print(f"🚀 Поток ЭЭГ ({role}, Gain {gain}x) идет в облако через {proxy_url}!")

                while client.is_connected:
                    await asyncio.sleep(0.08)
                    if len(batch) >= 20:
                        to_send = batch[:]
                        batch.clear()
                        try:
                            session.post(proxy_url, json={
                                "user": "User1",
                                "role": role,
                                "sps": 250.0,
                                "samples": to_send
                            }, timeout=0.5)
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️ Переподключение BLE: {e}")
            await asyncio.sleep(1.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, default="AFz", choices=["AFz", "FCz", "F3", "F4", "Fpz"])
    parser.add_argument("--gain", type=int, default=4, choices=[4, 8, 16, 32])
    parser.add_argument("--proxy", type=str, default="http://localhost:5000/api/eeg_push")
    args = parser.parse_args()
    asyncio.run(run_ble_forwarder(args.role, args.gain, args.proxy))
