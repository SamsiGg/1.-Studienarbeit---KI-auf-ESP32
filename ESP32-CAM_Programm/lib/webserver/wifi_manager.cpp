// --- Projekt-Header ---
#include "wifi_manager.h"
#include "credentials.h" // Für WIFI_NAME und WIFI_PASSWORD

// --- ESP-IDF Framework ---
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_netif.h"

static const char *TAG = "WIFI";

// Event-Handler NUR für Wi-Fi-Verbindungs-Events
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
        ESP_LOGI(TAG, "Verbinde mit WLAN...");
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WLAN-Verbindung getrennt. Versuche erneut...");
        esp_wifi_connect();
    }
}

esp_err_t wifi_connect_sta(void) {
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // Nur den Wi-Fi-Handler registrieren.
    // Den IP-Handler registriert das Modul, das die IP *braucht* (der Webserver).
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_NAME,
            .password = WIFI_PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WLAN-Initialisierung abgeschlossen.");
    return ESP_OK;
}