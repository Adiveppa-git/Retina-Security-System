def hardware_action(person):

    if person != "Unknown":

        print("\n========== HARDWARE ==========")
        print("LED : GREEN")
        print("Door : UNLOCKED")
        print("Buzzer : OFF")
        print("==============================\n")

        return {
            "door": "Unlocked",
            "led": "Green",
            "buzzer": "Off"
        }

    else:

        print("\n========== HARDWARE ==========")
        print("LED : RED")
        print("Door : LOCKED")
        print("Buzzer : ON")
        print("==============================\n")

        return {
            "door": "Locked",
            "led": "Red",
            "buzzer": "On"
        }