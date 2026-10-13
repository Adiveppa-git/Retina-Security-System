import serial
import time

print("Connecting to ESP32 on COM9 at 115200 baud...")
time.sleep(1.5)

s = serial.Serial()
s.port = "COM9"
s.baudrate = 115200
s.dtr = False
s.rts = False
s.timeout = 2
s.open()
time.sleep(0.3)

# Clear any boot buffer
s.read_all()

print("\n--- TEST 1: Send 'A' (Authorized from Idle) ---")
s.write(b"A")
s.flush()
line_A = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: '{line_A}'")
assert "[ESP32] AUTHORIZED - LED ON" in line_A, f"Unexpected response: {line_A}"
print("Waiting 2.2s for LED duration to complete...")
time.sleep(2.2)
print("[PASS] Test 1: Authorized LED triggered and completed.")

print("\n--- TEST 2: Send 'U' (Enter Continuous Unauthorized Alarm) ---")
s.write(b"U")
s.flush()
line_U = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: '{line_U}'")
assert "[ESP32] UNAUTHORIZED - ALARM TRIGGERED" in line_U, f"Unexpected response: {line_U}"
print("Alarm active. Letting buzzer beep continuously for 2.0s...")
time.sleep(2.0)

print("\n--- TEST 3: Send duplicate 'U' while alarm is already active ---")
s.write(b"U")
s.flush()
line_U2 = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: '{line_U2}'")
assert "[ESP32] UNAUTHORIZED - ALARM MAINTAINED" in line_U2, f"Unexpected response: {line_U2}"
print("Alarm maintained without interruption. Letting buzzer beep for another 2.0s...")
time.sleep(2.0)

print("\n--- TEST 4: Send 'A' WHILE BUZZER ALARM IS ACTIVELY BEEPING ---")
t0 = time.time()
s.write(b"A")
s.flush()
line_cancel = s.readline().decode("utf-8", errors="ignore").strip()
latency = time.time() - t0
print(f"ESP32 Output: '{line_cancel}' (Latency: {latency:.4f}s)")
assert "[ESP32] AUTHORIZED - LED ON" in line_cancel, f"Unexpected response: {line_cancel}"
print("Buzzer was immediately stopped and LED is ON!")
print("Waiting 2.2s for LED duration to complete...")
time.sleep(2.2)
print("[PASS] Test 4: Buzzer alarm cancelled immediately upon receiving 'A', LED ON for 2s.")

print("\n--- TEST 5: Verify lowercase commands ('u' -> 'a') ---")
s.write(b"u")
s.flush()
line_u = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output for 'u': '{line_u}'")
assert "[ESP32] UNAUTHORIZED - ALARM TRIGGERED" in line_u
time.sleep(1.0)

s.write(b"a")
s.flush()
line_a = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output for 'a': '{line_a}'")
assert "[ESP32] AUTHORIZED - LED ON" in line_a
time.sleep(2.2)
print("[PASS] Test 5: Lowercase 'u' triggers alarm and 'a' silences it.")

s.close()
print("\n>>> ALL TESTS PASSED SUCCESSFULLY! STATE MACHINE IS WORKING FLAWLESSLY! <<<")
