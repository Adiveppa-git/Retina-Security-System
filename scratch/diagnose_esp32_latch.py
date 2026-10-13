import serial
import time

def send_to_esp32(command: str):
    esp32 = None
    try:
        esp32 = serial.Serial()
        esp32.port = "COM9"
        esp32.baudrate = 115200
        esp32.dtr = False
        esp32.rts = False
        esp32.timeout = 1
        esp32.open()

        esp32.write(command.encode("utf-8"))
        esp32.flush()
        print(f"[send_to_esp32] Sent '{command}'")
    except Exception as e:
        print(f"[send_to_esp32 ERROR] {e}")
    finally:
        if esp32 and esp32.is_open:
            try:
                esp32.close()
            except Exception:
                pass

def open_serial():
    s = serial.Serial()
    s.port = "COM9"
    s.baudrate = 115200
    s.dtr = False
    s.rts = False
    s.timeout = 1
    s.open()
    return s

print("=== STEP 1: Sending 'U' via send_to_esp32 ===")
send_to_esp32("U")
time.sleep(1.0)

print("\n=== STEP 2: Checking if port can be read or if alarm is active ===")
s = open_serial()
lines = []
t0 = time.time()
while time.time() - t0 < 1.0:
    line = s.readline().decode('utf-8', errors='ignore').strip()
    if line:
        lines.append(line)
s.close()
print("Output captured during alarm:", lines)

print("\n=== STEP 3: Sending 'A' via send_to_esp32 ===")
send_to_esp32("A")
time.sleep(0.5)

print("\n=== STEP 4: Checking output after 'A' ===")
s = open_serial()
lines_after = []
t0 = time.time()
while time.time() - t0 < 1.0:
    line = s.readline().decode('utf-8', errors='ignore').strip()
    if line:
        lines_after.append(line)
s.close()
print("Output captured after 'A':", lines_after)
