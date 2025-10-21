#pragma once
#include "esp_camera.h"

/**
 * @brief Initialisiert die Kamera mit den AI-Thinker Pins.
 * * @return ESP_OK bei Erfolg, sonst ESP_FAIL.
 */
esp_err_t init_camera();