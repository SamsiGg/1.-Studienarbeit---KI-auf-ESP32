/*
 * main.cpp - Angepasst für ESP32-S3 (Native USB)
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
// WICHTIG: Neuer Treiber für S3 USB
#include "driver/usb_serial_jtag.h" 
#include "sdkconfig.h"

#include "internally_implemented.h" 
#include "submitter_implemented.h"

#define BUF_SIZE (1024)

void init_serial_communication() {
    // Konfiguration für den USB-Serial-JTAG Treiber
    usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
        .tx_buffer_size = BUF_SIZE,
        .rx_buffer_size = BUF_SIZE,
    };
    
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_serial_jtag_config));
}

extern "C" void app_main(void) {
    
    init_serial_communication();

    // WICHTIG: Kurz warten, damit USB sich verbinden kann
    vTaskDelay(pdMS_TO_TICKS(1500));

    ee_benchmark_initialize();

    uint8_t data;
    while (1) {
        // Lese von USB (nicht blockierend, Timeout kurz)
        int len = usb_serial_jtag_read_bytes(&data, 1, 10 / portTICK_PERIOD_MS);

        if (len > 0) {
            ee_serial_callback((char)data);
        }
    }
}