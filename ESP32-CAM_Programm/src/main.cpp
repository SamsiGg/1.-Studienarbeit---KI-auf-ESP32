#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"

#include "camera_init.h"
#include "person_detection.h"

// Diese Module werden nur eingebunden, wenn die Flag gesetzt ist
#if ENABLE_WEBSERVER
#include "wifi_manager.h"
#include "webserver.h"
#endif

static const char *TAG = "APP_MAIN";
static SemaphoreHandle_t camera_mutex;

extern "C" void app_main(void) {
    // 1. Initialisierung (NVS, TCP/IP, Event Loop)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // 2. Erstelle den Mutex
    camera_mutex = xSemaphoreCreateMutex();
    if (camera_mutex == NULL) {
        ESP_LOGE(TAG, "Mutex konnte nicht erstellt werden");
        return;
    }

    // 3. Hardware und Modelle initialisieren (laufen immer)
    if (init_camera() != ESP_OK) {
        ESP_LOGE(TAG, "Kamera-Init fehlgeschlagen!");
        return;
    }
    if (setup_person_detection_model() != ESP_OK) {
        ESP_LOGE(TAG, "TFLite-Setup fehlgeschlagen!");
        return;
    }

    // 4. Starte den Webserver-Teil (NUR wenn konfiguriert)
    // Dieser ganze Code-Block wird vom Compiler entfernt,
    // wenn die Flag in platformio.ini auskommentiert ist.
    #if ENABLE_WEBSERVER
        ESP_LOGI(TAG, "Webserver ist AKTIVIERT.");
        ESP_ERROR_CHECK(wifi_connect_sta());
        ESP_ERROR_CHECK(start_camera_webserver(camera_mutex));
    #else
        ESP_LOGI(TAG, "Webserver ist DEAKTIVIERT. Nur Inferenz läuft.");
    #endif

    // 5. Starte den Inferenz-Task (läuft immer)
    start_detection_task(camera_mutex);

    ESP_LOGI(TAG, "app_main() abgeschlossen, Tasks laufen.");
}