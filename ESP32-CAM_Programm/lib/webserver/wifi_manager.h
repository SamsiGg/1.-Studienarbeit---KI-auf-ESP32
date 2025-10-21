#pragma once
#include "esp_err.h"

/**
 * @brief Verbindet den ESP32 als STA mit dem WLAN.
 * Die Zugangsdaten werden aus credentials.h gelesen.
 * * @return ESP_OK bei Erfolg.
 */
esp_err_t wifi_connect_sta();