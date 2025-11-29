/*
 * main.cpp
 *
 * Portierung für ESP32 (ESP-IDF) auf Lolin D32.
 * Ersetzt Arduino setup() und loop() durch app_main().
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "sdkconfig.h"

// Deine Header-Dateien
#include "internally_implemented.h" 
#include "submitter_implemented.h"

// Konfiguration für den Serial Port (UART0 ist Standard für Logging/Konsole)
#define EX_UART_NUM UART_NUM_0
#define BUF_SIZE (1024)

/**
 * Initialisiert den UART Treiber (Ersatz für Serial.begin)
 */
void init_serial_communication() {
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    // Konfiguriere UART Parameter
    ESP_ERROR_CHECK(uart_param_config(EX_UART_NUM, &uart_config));

    // Setze Pins (UART_PIN_NO_CHANGE behält die Standard-Pins TX=1, RX=3 bei)
    ESP_ERROR_CHECK(uart_set_pin(EX_UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    // Installiere den Treiber (wir brauchen keinen Event-Queue hier)
    ESP_ERROR_CHECK(uart_driver_install(EX_UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0));
}

// app_main ist der Einstiegspunkt in ESP-IDF (Äquivalent zu setup, aber läuft nur einmal)
extern "C" void app_main(void) {
    
    // 1. STARTE die serielle Schnittstelle
    init_serial_communication();

    // Kurze Pause zur Stabilisierung
    vTaskDelay(pdMS_TO_TICKS(100));

    // Initialisiert TFLM und sendet "m-ready"
    // Dies entspricht deinem Aufruf in setup()
    ee_benchmark_initialize();

    // Puffer für empfangene Daten
    uint8_t data;

    // 2. ENDLOSSCHLEIFE (Äquivalent zu loop())
    while (1) {
        // Versuche 1 Byte zu lesen. 
        // Der letzte Parameter (20 / portTICK_PERIOD_MS) ist der Timeout.
        // Das verhindert, dass die CPU zu 100% blockiert und lässt den Watchdog 'atmen'.
        int len = uart_read_bytes(EX_UART_NUM, &data, 1, 10 / portTICK_PERIOD_MS);

        if (len > 0) {
            ee_serial_callback((char)data);
        }
    }
}