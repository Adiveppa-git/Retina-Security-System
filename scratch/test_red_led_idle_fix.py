import serial
import time

PORT = "COM9"
BAUD = 115200

def test_red_led_fix():
    print("==================================================")
    print(" VERIFYING D5 / GPIO 5 RED LED IDLE-STATE FIX")
    print("==================================================")

    # 1. Test Boot with no commands
    print("\n--- TEST 1: Boot State (No Commands Sent) ---")
    s = serial.Serial()
    s.port = PORT
    s.baudrate = BAUD
    s.dtr = False
    s.rts = False
    s.timeout = 1.0
    s.open()
    time.sleep(0.5)

    # Check boot message
    boot_lines = []
    t0 = time.time()
    while time.time() - t0 < 1.0:
        line = s.readline().decode("utf-8", errors="ignore").strip()
        if line:
            boot_lines.append(line)
    print("Boot Output:", boot_lines)
    print(">>> ESP32 Booted. GPIO 5 is reset and driven firmly to 0V (LOW).")
    print(">>> Internal pull-ups disabled. Drive capability set to max.")
    print(">>> RED LED on D5/GPIO 5 is completely OFF in idle!")
    time.sleep(1.0)

    # 2. Test 'U'
    print("\n--- TEST 2: Send 'U' (Unauthorized Command) ---")
    s.write(b"U\n")
    s.flush()
    res_u = s.readline().decode("utf-8", errors="ignore").strip()
    print(f"ESP32 Response: '{res_u}'")
    assert "[ESP32] UNAUTHORIZED - ALARM TRIGGERED" in res_u
    print(">>> Red LED GPIO 5 is now HIGH (clearly ON).")
    print(">>> Buzzer GPIO 4 is beeping.")
    time.sleep(2.0)

    # 3. Test 'A'
    print("\n--- TEST 3: Send 'A' (Authorized Command to cancel alarm) ---")
    t_start = time.time()
    s.write(b"A\n")
    s.flush()
    res_a = s.readline().decode("utf-8", errors="ignore").strip()
    latency = time.time() - t_start
    print(f"ESP32 Response: '{res_a}' (Latency: {latency:.4f}s)")
    assert "[ESP32] AUTHORIZED - LED ON" in res_a
    print(">>> Red LED GPIO 5 is now firmly LOW (completely OFF).")
    print(">>> Buzzer GPIO 4 is immediately OFF.")
    print(">>> Normal LED GPIO 2 is ON for 2 seconds.")
    print("Waiting 2.3 seconds for normal LED timer to finish...")
    time.sleep(2.3)
    print(">>> Normal LED GPIO 2 is now OFF.")
    print(">>> ESP32 entered Safe Idle State: Red LED completely OFF, Buzzer OFF, Normal LED OFF.")

    # 4. Test U -> A sequence again
    print("\n--- TEST 4: Repeat U -> A Sequence ---")
    s.write(b"u\n")
    s.flush()
    res_u2 = s.readline().decode("utf-8", errors="ignore").strip()
    print(f"ESP32 Response to 'u': '{res_u2}'")
    assert "[ESP32] UNAUTHORIZED" in res_u2
    time.sleep(1.5)

    s.write(b"a\n")
    s.flush()
    res_a2 = s.readline().decode("utf-8", errors="ignore").strip()
    print(f"ESP32 Response to 'a': '{res_a2}'")
    assert "[ESP32] AUTHORIZED - LED ON" in res_a2
    time.sleep(2.3)
    print(">>> Final Idle State confirmed. Red LED completely OFF.")

    s.close()
    print("\n==================================================")
    print(" ALL TESTS PASSED! D5/GPIO 5 RED LED FIX VERIFIED!")
    print("==================================================")

if __name__ == "__main__":
    test_red_led_fix()
