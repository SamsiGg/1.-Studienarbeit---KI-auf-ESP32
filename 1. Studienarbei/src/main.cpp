#include <Arduino.h>

// Die eingebaute LED beim LOLIN D32 ist mit dem Pin GPIO 5 verbunden.
#define LED_BUILTIN 5

void setup() {
  // Setzt den LED-Pin als Ausgang.
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // Schaltet die LED ein (HIGH ist der Spannungspegel)
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000); // Wartet eine Sekunde

  // Schaltet die LED aus, indem die Spannung auf LOW gesetzt wird
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000); // Wartet eine Sekunde
}