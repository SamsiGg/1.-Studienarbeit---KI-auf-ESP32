/*
 * main.cpp
 *
 * Dieser Code verwendet die korrekte Arduino setup() und loop() Struktur.
 * Er wird NICHT überhitzen.
 */

#include <Arduino.h>
#include "internally_implemented.h" // Für Benchmark-Funktionen
#include "submitter_implemented.h" // Für th_getchar()

void setup() {
  // WICHTIG: Wir nutzen die MLPerf-Initialisierung.
  // Diese Funktion ruft intern DEINE 'th_serialport_initialize' auf.
  // Das sorgt dafür, dass Serial2 (9600) und Serial (115200) gestartet werden.
  // Außerdem sendet sie am Ende automatisch "m-ready".
  ee_benchmark_initialize();
}

void loop() {
  // 1. Zeichen empfangen
  // Wir rufen DEINE th_getchar() Funktion auf.
  // Die enthält jetzt deine Debug-Punkte (...) und lauscht
  // automatisch am richtigen Port (Serial2 im Energy Mode).
  char c = th_getchar(); 

  // 2. Zeichen verarbeiten
  // Das MLPerf Framework kümmert sich um den Rest.
  ee_serial_callback(c);
}

/*#include <Arduino.h>
// Minimaler Echo-Test für Teensy 4.0 an Serial2
void setup() {
  // LED zur visuellen Kontrolle
  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH); // LED an = Start

  // Serial2 ist Pin 7 (RX) und Pin 8 (TX)
  Serial2.begin(9600); 
  Serial.begin(115200); // Für Debugging
}

void loop() {
  if (Serial2.available()) {
    char c = Serial2.read();
    
    // Visuelles Feedback: LED kurz ausschalten bei Empfang
    digitalWrite(13, LOW);
    delay(5);
    digitalWrite(13, HIGH);
    
    // Zeichen direkt zurücksenden
    Serial2.write(c);
    Serial.print("Echoed: ");
    
  }
}*/