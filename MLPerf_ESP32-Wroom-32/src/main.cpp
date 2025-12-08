/*
 * main.cpp
 *
 * Die saubere ESP32 Main für MLPerf Tiny.
 * Vertraut auf die Logik in submitter_implemented.cpp.
 */

#include "internally_implemented.h" 
#include "submitter_implemented.h"

// Wichtig für C++ Projekte in ESP-IDF
extern "C" void app_main(void) {
    
    // 1. Initialisierung
    // WICHTIG: Das ruft intern th_serialport_initialize() auf!
    // Dadurch wird automatisch die richtige Baudrate (9600 vs 115200) 
    // gesetzt, je nachdem was wir in submitter_implemented.cpp definiert haben.
    ee_benchmark_initialize();

    // 2. Endlosschleife
    while (1) {
        // Wir nutzen DEINE Funktion th_getchar().
        // Die kümmert sich um UART-Reads und Timeouts.
        char c = th_getchar(); 
        
        // Verarbeite das Zeichen im MLPerf Framework
        ee_serial_callback(c);
    }
}