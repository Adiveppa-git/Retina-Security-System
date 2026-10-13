import serial
import time

print("Connecting to ESP32 on COM9...")
time.sleep(1.0)

s = serial.Serial()
s.port = "COM9"
s.baudrate = 115200
s.dtr = False
s.rts = False
s.timeout = 2
s.open()
time.sleep(0.2)

print("\n--- Test 1: Uppercase 'A' (Authorized) ---")
t0 = time.time()
s.write(b"A")
s.flush()
line_A = s.readline().decode("utf-8", errors="ignore").strip()
t_A = time.time() - t0
print(f"ESP32 Output: {line_A}")
print(f"Response latency: {t_A:.4f}s")
assert "[ESP32] AUTHORIZED - LED ON" in line_A, f"Unexpected response: {line_A}"
time.sleep(2.1)

print("\n--- Test 2: Uppercase 'U' (Unauthorized) ---")
t0 = time.time()
s.write(b"U")
s.flush()
line_U = s.readline().decode("utf-8", errors="ignore").strip()
t_U = time.time() - t0
print(f"ESP32 Output: {line_U}")
print(f"Response latency: {t_U:.4f}s")
assert "[ESP32] UNAUTHORIZED" in line_U, f"Unexpected response: {line_U}"
time.sleep(1.1)

print("\n--- Test 3: Lowercase 'a' (Authorized) ---")
s.write(b"a")
s.flush()
line_a = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: {line_a}")
assert "[ESP32] AUTHORIZED - LED ON" in line_a, f"Unexpected response: {line_a}"
time.sleep(2.1)

print("\n--- Test 4: Lowercase 'u' (Unauthorized) ---")
s.write(b"u")
s.flush()
line_u = s.readline().decode("utf-8", errors="ignore").strip()
print(f"ESP32 Output: {line_u}")
assert "[ESP32] UNAUTHORIZED" in line_u, f"Unexpected response: {line_u}"
time.sleep(1.1)

s.close()
print("\n>>> ALL DIRECT SERIAL HARDWARE TESTS PASSED SUCCESSFULLY! <<<")
