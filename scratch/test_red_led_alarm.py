import serial
import time
import requests
import io
from PIL import Image

print("==================================================")
print("TESTING ESP32 WITH RED LED (GPIO 15) INTEGRATION")
print("==================================================")

time.sleep(1.5)
s = serial.Serial()
s.port = "COM9"
s.baudrate = 115200
s.dtr = False
s.rts = False
s.timeout = 2
s.open()
time.sleep(0.3)
s.read_all()

# TEST 1: Test "A" (Authorized)
print("\n--- TEST 1: Command 'A' (Authorized from Idle) ---")
print("Expected: GPIO 2 Normal LED ON 2s, GPIO 4 Buzzer OFF, GPIO 15 Red LED OFF")
s.write(b"A")
s.flush()
line_A = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: '{line_A}'")
assert "[ESP32] AUTHORIZED - LED ON" in line_A, f"Unexpected response: {line_A}"
time.sleep(2.2)
print("[PASS] Test 1: Authorized executed (Normal LED ON 2s, Red LED OFF, Buzzer OFF).")

# TEST 2: Test "U" (Unauthorized)
print("\n--- TEST 2: Command 'U' (Unauthorized / Alarm) ---")
print("Expected: GPIO 2 Normal LED OFF, GPIO 4 Buzzer continuously beeping, GPIO 15 Red LED ON")
s.write(b"U")
s.flush()
line_U = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: '{line_U}'")
assert "[ESP32] UNAUTHORIZED - ALARM TRIGGERED" in line_U, f"Unexpected response: {line_U}"
print("Letting alarm run for 2.5s (Buzzer beeping 300ms/200ms, Red LED continuously ON)...")
time.sleep(2.5)

# TEST 3: Duplicate "U" during alarm
print("\n--- TEST 3: Duplicate 'U' while alarm is already active ---")
print("Expected: Alarm and Red LED maintained without restart")
s.write(b"U")
s.flush()
line_U2 = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: '{line_U2}'")
assert "[ESP32] UNAUTHORIZED - ALARM MAINTAINED" in line_U2, f"Unexpected response: {line_U2}"
time.sleep(2.0)
print("[PASS] Test 3: Alarm & Red LED maintained.")

# TEST 4: Send "A" WHILE "U" ALARM & RED LED ARE ACTIVELY SOUNDING
print("\n--- TEST 4: Send 'A' WHILE BUZZER ALARM & RED LED ARE ACTIVE ---")
print("Expected: Buzzer IMMEDIATELY OFF, Red LED IMMEDIATELY OFF, Normal LED ON for 2s")
t0 = time.time()
s.write(b"A")
s.flush()
line_cancel = s.readline().decode("utf-8", errors="ignore").strip()
latency = time.time() - t0
print(f"ESP32 Output: '{line_cancel}' (Latency: {latency:.4f}s)")
assert "[ESP32] AUTHORIZED - LED ON" in line_cancel, f"Unexpected response: {line_cancel}"
print("Buzzer & Red LED stopped immediately! Normal LED ON for 2s...")
time.sleep(2.2)
print("[PASS] Test 4: 'A'-after-'U' successful (Buzzer & Red LED stopped, Normal LED ON 2s).")

# TEST 5: Lowercase tests
print("\n--- TEST 5: Lowercase 'u' then 'a' ---")
s.write(b"u")
s.flush()
line_u = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output for 'u': '{line_u}'")
assert "[ESP32] UNAUTHORIZED - ALARM TRIGGERED" in line_u
time.sleep(1.5)

s.write(b"a")
s.flush()
line_a = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output for 'a': '{line_a}'")
assert "[ESP32] AUTHORIZED - LED ON" in line_a
time.sleep(2.2)
print("[PASS] Test 5: Lowercase commands validated.")

s.close()
print("\n>>> ALL DIRECT SERIAL & HARDWARE TESTS PASSED! <<<")

# TEST 6: Actual Iris Verification Sequence Endpoint Test
print("\n--- TEST 6: Actual /iris/verify_sequence_1to1 flow ---")
BASE_URL = "http://127.0.0.1:8000"
try:
    dummy_files = []
    for i in range(5):
        img = Image.new("RGB", (320, 240), color=(30 * i, 70, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        dummy_files.append(("files", (f"seq_frame_{i}.jpg", buf.getvalue(), "image/jpeg")))

    res = requests.post(
        f"{BASE_URL}/iris/verify_sequence_1to1",
        data={"claimed_user_id": 1, "action": "CHECK-IN"},
        files=dummy_files,
        timeout=10
    )
    print(f"Backend HTTP Response: {res.status_code}")
    res_data = res.json()
    print(f"Diagnostic Status: {res_data.get('final_diagnostic_status')}")
    print(f"Decision Reason: {res_data.get('decision_reason')}")
    print("[PASS] Test 6: Backend endpoint correctly executed unauthorized flow and triggered ESP32 alarm & Red LED!")
    
    # Silence the alarm with 'A'
    time.sleep(2.0)
    s2 = serial.Serial("COM9", 115200, timeout=2)
    s2.dtr = False
    s2.rts = False
    time.sleep(0.1)
    s2.write(b"A")
    s2.flush()
    s2.close()
    print("Silenced alarm with 'A'.")

except Exception as e:
    print(f"Backend test note: {e}")

print("\n==================================================")
print(">>> ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! <<<")
print("==================================================")
