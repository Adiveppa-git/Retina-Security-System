def hardware_action(final_decision: str):
    """
    Hardware Access Control State Dispatcher:
    STRICT SECURITY RULE:
    Door unlocks ONLY AND EXCLUSIVELY when final_decision == "ACCESS_GRANTED".
    All other decisions (ACCESS_DENIED, SPOOF_DETECTED, NO_EYES_DETECTED, NO_FACE_DETECTED)
    MUST lock the door, set LED to Red, and trigger the Buzzer Alarm.
    """
    if final_decision == "ACCESS_GRANTED":
        print("\n========== HARDWARE ACCESS CONTROL ==========")
        print("STATUS : ACCESS GRANTED [ALLOWED]")
        print("LED    : GREEN")
        print("DOOR   : UNLOCKED")
        print("BUZZER : OFF")
        print("=============================================\n")

        return {
            "door": "Unlocked",
            "led": "Green",
            "buzzer": "Off"
        }
    else:
        print("\n========== HARDWARE ACCESS CONTROL ==========")
        print(f"STATUS : ACCESS DENIED [BLOCKED] ({final_decision})")
        print("LED    : RED")
        print("DOOR   : LOCKED")
        print("BUZZER : ON (SECURITY ALARM)")
        print("=============================================\n")

        return {
            "door": "Locked",
            "led": "Red",
            "buzzer": "On"
        }