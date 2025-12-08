/*
 * main.cpp - Hybrid-Version für ESP32-S3
 * * Performance Mode: Nutzt Native USB (usb_serial_jtag) für High-Speed.
 * Energy Mode: Nutzt UART via th_getchar() für Joulescope-Kompatibilität.
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "internally_implemented.h" 
#include "submitter_implemented.h"

// ============================================================
// PERFORMANCE MODE LOGIK (USB-SERIAL-JTAG)
// ============================================================
#if !EE_CFG_ENERGY_MODE

// Nur einbinden, wenn wir NICHT im Energiemodus sind
#include "driver/usb_serial_jtag.h" 

#define BUF_SIZE (1024)

void init_usb_communication() {
    usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
        .tx_buffer_size = BUF_SIZE,
        .rx_buffer_size = BUF_SIZE,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_serial_jtag_config));
}

#endif 
// ============================================================


extern "C" void app_main(void) {
    
    // --------------------------------------------------------
    // FALL 1: PERFORMANCE MODE (USB)
    // --------------------------------------------------------
    #if !EE_CFG_ENERGY_MODE
        
        // 1. Starte Native USB
        init_usb_communication();

        // 2. Warte kurz, bis USB verbunden ist (wichtig beim S3!)
        vTaskDelay(pdMS_TO_TICKS(1500));

        // 3. Init Benchmark
        ee_benchmark_initialize();

        // 4. Loop: Lese direkt vom USB-Treiber
        uint8_t data;
        while (1) {
            // Nicht-blockierendes Lesen mit kurzem Timeout
            int len = usb_serial_jtag_read_bytes(&data, 1, 10 / portTICK_PERIOD_MS);

            if (len > 0) {
                ee_serial_callback((char)data);
            }
        }

    // --------------------------------------------------------
    // FALL 2: ENERGY MODE (UART)
    // --------------------------------------------------------
    #else
        
        // 1. Init Benchmark (Ruft intern th_serialport_initialize auf -> UART Init!)
        ee_benchmark_initialize();

        // 2. Loop: Nutze th_getchar() (welches UART liest)
        while (1) {
            // th_getchar() in submitter_implemented.cpp kümmert sich um UART Read & Timeout
            char c = th_getchar(); 
            ee_serial_callback(c);
        }

    #endif
}