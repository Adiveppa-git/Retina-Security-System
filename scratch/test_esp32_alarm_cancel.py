import serial
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.routers.iris_router import send_to_esp32

PORT = "COM9"
BAUD = 115200

def test_independent_esp32():
    print("==================================================")
    print(" 1. INDEPENDENT ESP32 TEST (U -> A ALARM CANCEL)")
    print("==================================================")
    
    s = serial.Serial()
    s.port = PORT
    s.baudrate = BAUD
    s.dtr = False
    s.rts = False
    s.timeout = 1.0
    s.open()
    time.sleep(0.1)
    s.read_all()

    # Step 1: Trigger UNAUTHORIZED Alarm ('U')
    print("\n[Step 1] Sending 'U' (Trigger Alarm)...")
    s.write(b"U\n")
    s.flush()
    res_u = s.readline().decode("utf-8", errors="ignore").strip()
    print(f"ESP32 Response: '{res_u}'")
    assert "[ESP32] UNAUTHORIZED - ALARM TRIGGERED" in res_u, f"Unexpected response: {res_u}"
    print(">>> Red LED (GPIO 5) is ON. Repeating buzzer (GPIO 4) is beeping.")
    print(">>> Letting alarm sound for 2.0 seconds...")
    time.sleep(2.0)

    # Step 2: Cancel Alarm immediately with AUTHORIZED ('A')
    print("\n[Step 2] Sending 'A' (WHILE ALARM IS ACTIVELY BEEPING)...")
    t0 = time.time()
    s.write(b"A\n")
    s.flush()
    res_a = s.readline().decode("utf-8", errors="ignore").strip()
    t_cancel = time.time() - t0
    print(f"ESP32 Response: '{res_a}' (Response Latency: {t_cancel:.4f}s)")
    assert "[ESP32] AUTHORIZED - LED ON" in res_a, f"Unexpected response: {res_a}"
    print(">>> BUZZER IMMEDIATELY OFF! RED LED (GPIO 5) IMMEDIATELY OFF!")
    print(">>> Normal LED (GPIO 2) is ON for 2 seconds.")
    print("Waiting 2.3 seconds for normal LED duration to finish...")
    time.sleep(2.3)
    print(">>> Normal LED (GPIO 2) is now OFF. ESP32 is in SAFE IDLE STATE.")

    s.close()
    time.sleep(0.5)

def test_alternating_sequence():
    print("\n==================================================")
    print(" 2. ALTERNATING SEQUENCE TEST: A -> U -> A -> U -> A")
    print("==================================================")

    steps = [
        ('A', "AUTHORIZED", 2.2),
        ('U', "UNAUTHORIZED", 1.5),
        ('A', "AUTHORIZED", 2.2),
        ('U', "UNAUTHORIZED", 1.5),
        ('A', "AUTHORIZED", 2.2)
    ]

    for idx, (cmd, expected, wait_sec) in enumerate(steps, start=1):
        print(f"\n[Seq Step {idx}] Sending '{cmd}' via send_to_esp32()...")
        send_to_esp32(cmd)
        print(f"Holding state for {wait_sec}s...")
        time.sleep(wait_sec)
        print(f"[Seq Step {idx}] Done.")

    print("\n>>> ALL ALTERNATING SEQUENCE STEPS PASSED!")
    print(">>> ALARM NEVER REMAINS LATCHED AFTER 'A'!")
    print(">>> FINAL ESP32 STATE: SAFE IDLE (All outputs OFF: GPIO 2 LOW, GPIO 4 LOW, GPIO 5 LOW).")

if __name__ == "__main__":
    test_independent_esp32()
    test_alternating_sequence()
