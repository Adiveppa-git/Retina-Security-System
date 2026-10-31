#include <Arduino.h>
#include "driver/gpio.h"

#define LED_PIN 2        // Normal Green/Blue LED (GPIO 2)
#define BUZZER_PIN 4     // MH-FMD Active Buzzer Module (GPIO 4)
#define RED_LED_PIN 5    // Red Alarm LED (D5 / GPIO 5)

// Active-LOW polarity for MH-FMD "Low level trigger" buzzer module:
// LOW = Transistor conducts, Buzzer ON
// HIGH = Transistor cut off, Buzzer OFF
#define BUZZER_ACTIVE   LOW
#define BUZZER_INACTIVE HIGH

// State Machine Definitions
enum SystemState {
  STATE_IDLE,
  STATE_AUTHORIZED,
  STATE_UNAUTHORIZED
};

SystemState currentState = STATE_IDLE;

// Configurable timing parameters (non-blocking)
const unsigned long GREEN_DURATION_MS = 2500;              // 2.5 seconds green LED indicator (2-3s)
const unsigned long SUCCESS_BEEP_MS = 3000;                 // 3 seconds audible beep on authorized (3s)
const unsigned long UNAUTHORIZED_BUZZER_DURATION_MS = 10000;// 10 seconds buzzer duration on unauthorized (10s)
const unsigned long RED_ALARM_DURATION_MS = 15000;         // 15 seconds denied alarm (configurable 10-30s)

// Option C: Double-beep pattern for unauthorized alarm (non-blocking):
// 150ms ON -> 120ms OFF -> 150ms ON -> 600ms OFF (Total cycle = 1020ms)
const unsigned long BEEP1_ON_MS = 150;
const unsigned long BEEP1_OFF_MS = 120;
const unsigned long BEEP2_ON_MS = 150;
const unsigned long BEEP2_OFF_MS = 600;
const unsigned long DOUBLE_BEEP_CYCLE_MS = BEEP1_ON_MS + BEEP1_OFF_MS + BEEP2_ON_MS + BEEP2_OFF_MS; // 1020ms

unsigned long greenSequenceStartTime = 0;
unsigned long unauthorizedStartTime = 0;
bool isBeeping = false;
unsigned long beepStartTime = 0;

void setRedLed(bool state) {
  if (state) {
    digitalWrite(RED_LED_PIN, HIGH);
    gpio_set_level(GPIO_NUM_5, 1);
  } else {
    digitalWrite(RED_LED_PIN, LOW);
    gpio_set_level(GPIO_NUM_5, 0);
  }
}

void enterIdleState() {
  currentState = STATE_IDLE;
  setRedLed(false);
  digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
  digitalWrite(LED_PIN, LOW);
  isBeeping = false;
}

void enterUnauthorizedState() {
  currentState = STATE_UNAUTHORIZED;

  // Immediately stop green LED & success beep
  digitalWrite(LED_PIN, LOW);
  isBeeping = false;

  // Reset unauthorized alarm timer
  unauthorizedStartTime = millis();

  // Red LED continuously ON for the alarm duration, and start double-beep cycle immediately (Beep 1 ON)
  setRedLed(true);
  digitalWrite(BUZZER_PIN, BUZZER_ACTIVE);

  Serial.println("[ESP32] UNAUTHORIZED - RED & BUZZER ACTIVE (AUTO TIMEOUT 15s)");
}

void enterAuthorizedState() {
  currentState = STATE_AUTHORIZED;

  // Immediately clear unauthorized alert & reset timers
  setRedLed(false);

  // Turn green LED ON
  digitalWrite(LED_PIN, HIGH);
  greenSequenceStartTime = millis();

  // Short success beep
  digitalWrite(BUZZER_PIN, BUZZER_ACTIVE);
  beepStartTime = millis();
  isBeeping = true;

  Serial.println("[ESP32] AUTHORIZED - GREEN ON + SUCCESS BEEP");
}

void setup() {
  // 1. Initialize pin modes
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);

  // 2. Hardware-level clean reset for GPIO 4 (D4 / BUZZER_PIN):
  // Configure push-pull output with maximum drive strength (GPIO_DRIVE_CAP_3 ~40mA)
  // Ensure buzzer starts in silent/inactive state (HIGH for active-LOW trigger)
  gpio_reset_pin(GPIO_NUM_4);
  gpio_set_direction(GPIO_NUM_4, GPIO_MODE_OUTPUT);
  gpio_pullup_dis(GPIO_NUM_4);
  gpio_pulldown_dis(GPIO_NUM_4);
  gpio_set_level(GPIO_NUM_4, 1);
  gpio_set_drive_capability(GPIO_NUM_4, GPIO_DRIVE_CAP_3);
  digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);

  // 3. Hardware-level clean reset for GPIO 5 (D5 / RED_LED_PIN):
  // Disconnect any ROM strapping / VSPI CS defaults, disable internal pull-ups,
  // enforce push-pull output with maximum drive strength, and drive firmly to 0V (LOW).
  gpio_reset_pin(GPIO_NUM_5);
  gpio_set_direction(GPIO_NUM_5, GPIO_MODE_OUTPUT);
  gpio_pullup_dis(GPIO_NUM_5);
  gpio_pulldown_dis(GPIO_NUM_5);
  gpio_set_level(GPIO_NUM_5, 0);
  gpio_set_drive_capability(GPIO_NUM_5, GPIO_DRIVE_CAP_3);
  digitalWrite(RED_LED_PIN, LOW);

  // Initialize to IDLE state (all outputs OFF)
  enterIdleState();

  Serial.begin(115200);
  Serial.setTimeout(100);

  Serial.println("ESP32 READY");
}

void loop() {
  // 1. Process all incoming Serial characters continuously
  while (Serial.available() > 0) {
    char c = Serial.read();

    // Ignore newline, carriage return, space, and tab
    if (c == '\r' || c == '\n' || c == ' ' || c == '\t' || c == '\0') {
      continue;
    }

    if (c == 'A' || c == 'a') {
      enterAuthorizedState();
    } else if (c == 'U' || c == 'u') {
      if (currentState != STATE_UNAUTHORIZED) {
        enterUnauthorizedState();
      } else {
        // If already unauthorized, maintain the alarm without resetting the 15-second countdown timer!
        Serial.println("[ESP32] UNAUTHORIZED - ALARM MAINTAINED");
      }
    }
  }

  unsigned long currentMillis = millis();

  // 2. State Machine Loop Actions
  switch (currentState) {
    case STATE_UNAUTHORIZED:
      // Red LED remains ON until alarm duration expires (15s)
      setRedLed(true);
      digitalWrite(LED_PIN, LOW);

      // Buzzer sounds for 10 seconds with Option C Double-Beep pattern:
      // 150ms ON -> 120ms OFF -> 150ms ON -> 600ms OFF (repeating non-blocking cycle)
      if (currentMillis - unauthorizedStartTime < UNAUTHORIZED_BUZZER_DURATION_MS) {
        unsigned long cycleTime = (currentMillis - unauthorizedStartTime) % DOUBLE_BEEP_CYCLE_MS;
        if (cycleTime < BEEP1_ON_MS) {
          digitalWrite(BUZZER_PIN, BUZZER_ACTIVE);
        } else if (cycleTime < BEEP1_ON_MS + BEEP1_OFF_MS) {
          digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
        } else if (cycleTime < BEEP1_ON_MS + BEEP1_OFF_MS + BEEP2_ON_MS) {
          digitalWrite(BUZZER_PIN, BUZZER_ACTIVE);
        } else {
          digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
        }
      } else {
        digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
      }

      if (currentMillis - unauthorizedStartTime >= RED_ALARM_DURATION_MS) {
        enterIdleState();
        Serial.println("[ESP32] IDLE - RED & BUZZER TIMED OUT");
      }
      break;

    case STATE_AUTHORIZED:
      // Red LED remains OFF
      setRedLed(false);

      // Handle green LED duration (2.5s)
      if (currentMillis - greenSequenceStartTime >= GREEN_DURATION_MS) {
        digitalWrite(LED_PIN, LOW);
      } else {
        digitalWrite(LED_PIN, HIGH);
      }

      // Handle double-beep pattern for authorized access (3.0s duration):
      // 150ms ON -> 120ms OFF -> 150ms ON -> 600ms OFF (repeating non-blocking cycle)
      if (isBeeping) {
        if (currentMillis - beepStartTime < SUCCESS_BEEP_MS) {
          unsigned long cycleTime = (currentMillis - beepStartTime) % DOUBLE_BEEP_CYCLE_MS;
          if (cycleTime < BEEP1_ON_MS) {
            digitalWrite(BUZZER_PIN, BUZZER_ACTIVE);
          } else if (cycleTime < BEEP1_ON_MS + BEEP1_OFF_MS) {
            digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
          } else if (cycleTime < BEEP1_ON_MS + BEEP1_OFF_MS + BEEP2_ON_MS) {
            digitalWrite(BUZZER_PIN, BUZZER_ACTIVE);
          } else {
            digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
          }
        } else {
          digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
          isBeeping = false;
        }
      } else {
        digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
      }

      // Return to normal idle state once authorized sequence is complete (3.0s beep and 2.5s LED)
      if (!isBeeping && (currentMillis - greenSequenceStartTime >= GREEN_DURATION_MS)) {
        enterIdleState();
        Serial.println("[ESP32] IDLE - GREEN OFF");
      }
      break;

    case STATE_IDLE:
    default:
      setRedLed(false);
      digitalWrite(BUZZER_PIN, BUZZER_INACTIVE);
      digitalWrite(LED_PIN, LOW);
      break;
  }
}