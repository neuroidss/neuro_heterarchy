#!/usr/bin/env python3
import serial
import time
import argparse
from pylsl import StreamInfo, StreamOutlet

# ================= НАСТРОЙКИ =================
SERIAL_PORT = '/dev/ttyACM0'  # Укажите ваш порт
BAUD_RATE = 2000000
CHANNELS_PER_NODE = 16
PACKET_SIZE = 57  # 1(A0) + 1(Count) + 48(Data) + 6(MAC) + 1(C0)
# =============================================

class Node:
    def __init__(self, mac_str):
        self.mac_str = mac_str
        self.last_counter = -1
        self.packets_received = 0
        self.packets_lost = 0
        
        clean_mac = self.mac_str.replace(":", "").replace("-", "")
        print(f"\n[NEW DEVICE] MAC: {mac_str}. Creating LSL Stream 'FreeEEG_{clean_mac}'...")
        info = StreamInfo(f'FreeEEG_{clean_mac}', 'EEG', CHANNELS_PER_NODE, 
                          250.0, 'int32', f'uid_{mac_str}')
        self.outlet = StreamOutlet(info)

def parse_24bit_to_int32(raw_bytes):
    channels = []
    for i in range(16):
        idx = i * 3
        val = (raw_bytes[idx] << 16) | (raw_bytes[idx+1] << 8) | raw_bytes[idx+2]
        if val & 0x800000:
            val -= 0x1000000
        channels.append(val)
    return channels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain', type=int, default=16, choices=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--autochop', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    print(f"Connecting to Dongle on {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"Failed to open {SERIAL_PORT}: {e}")
        return

    time.sleep(1.5) 
    ser.write(f"GAIN={args.gain}\n".encode('utf-8'))
    time.sleep(0.1)
    ser.write(f"CHOP={args.autochop}\n".encode('utf-8'))
    ser.reset_input_buffer()

    active_nodes = {}
    buffer = bytearray()
    last_print_time = time.time()
    
    print("\n[READY] Listening for streams... Press Ctrl+C to stop.")
    
    while True:
        chunk = ser.read(8192)
        if chunk:
            buffer.extend(chunk)
            
        while len(buffer) >= PACKET_SIZE:
            if buffer[0] == 0xA0 and buffer[PACKET_SIZE - 1] == 0xC0:
                counter = buffer[1]
                adc_data = buffer[2:50]
                mac_bytes = buffer[50:56]
                
                mac_str = ':'.join(f'{b:02X}' for b in mac_bytes)
                if mac_str not in active_nodes: active_nodes[mac_str] = Node(mac_str)
                node = active_nodes[mac_str]
                
                if node.last_counter != -1:
                    expected_counter = (node.last_counter + 1) % 256
                    if counter != expected_counter:
                        node.packets_lost += (counter - expected_counter) % 256
                        
                node.last_counter = counter
                node.packets_received += 1
                
                channel_data = parse_24bit_to_int32(adc_data)
                node.outlet.push_sample(channel_data)
                del buffer[:PACKET_SIZE]
            else:
                # ВМЕСТО УДАЛЕНИЯ, ПЕЧАТАЕМ ЛОГИ С ДОНГЛА!
                next_a0 = buffer.find(0xA0, 1)
                if next_a0 == -1:
                    text = buffer.decode('ascii', 'ignore').replace('\r', '').replace('\n', ' ')
                    if text.strip(): print(f"[DONGLE LOG]: {text.strip()}")
                    del buffer[:]
                else:
                    text = buffer[:next_a0].decode('ascii', 'ignore').replace('\r', '').replace('\n', ' ')
                    if text.strip(): print(f"[DONGLE LOG]: {text.strip()}")
                    del buffer[:next_a0]

        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            if active_nodes:
                print("\n--- LSL Bridge Stats (250 Hz Expected) ---")
                for mac, n in active_nodes.items():
                    print(f"Device [{mac}]: Rx = {n.packets_received:3d} pkts/s | Lost Total = {n.packets_lost}")
                    n.packets_received = 0 
            last_print_time = current_time

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: print("\nStopped.")
