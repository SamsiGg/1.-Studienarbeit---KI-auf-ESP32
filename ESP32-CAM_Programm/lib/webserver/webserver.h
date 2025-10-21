#pragma once
#include "esp_err.h"
#include "freertos/semphr.h"

/**
 * @brief Startet den MJPEG-Stream-Webserver.
 * Registriert auch den IP-Event-Handler, um die IP-Adresse anzuzeigen.
 * * @param camera_mutex Der Mutex für den exklusiven Kamerazugriff.
 * @return ESP_OK bei Erfolg.
 */
esp_err_t start_camera_webserver(SemaphoreHandle_t camera_mutex);