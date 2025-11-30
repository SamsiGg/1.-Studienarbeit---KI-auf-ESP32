/*
 * main.cpp
 *
 * Dieser Code verwendet die korrekte Arduino setup() und loop() Struktur.
 * Er wird NICHT überhitzen.
 */

#include <Arduino.h>
#include "internally_implemented.h" // Für Benchmark-Funktionen
#include "submitter_implemented.h" // Für th_getchar()

// setup() wird einmal beim Start aufgerufen (NACHDEM Arduino die Hardware init hat)
void setup() {
  // 1. STARTE die serielle Schnittstelle so früh wie möglich
  Serial.begin(115200);

  // 2. WARTE hier, bis der PC (Monitor) verbunden ist
  //    Dies ist der sichere Weg, um Überhitzung zu vermeiden
  while (!Serial){
    delay(10); // Warte 10 ms und prüfe erneut
  }
  // Initialisiert Serial, TFLM, und sendet "m-ready"
  ee_benchmark_initialize();
}

// loop() wird kontinuierlich aufgerufen
void loop() {
  // Prüfe, ob ein Zeichen vom Host-Computer gesendet wurde
  if (Serial.available() > 0) {
    
    // HINWEIS: Wir rufen th_getchar() NICHT mehr auf,
    // da th_getchar() die blockierende Schleife enthält, die
    // wir hier nicht wollen. Wir lesen direkt.
    char c = Serial.read();
    
    // Gib das Zeichen an den internen Callback-Parser weiter
    ee_serial_callback(c);
  }
}