#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h" // Wichtig: Für den Mutex
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "esp_camera.h"
#include "esp_http_server.h"

#include "credentials.h" // Enthält WIFI_NAME und WIFI_PASSWORD



static const char *TAG = "CAMERA_WEBSERVER";

SemaphoreHandle_t camera_mutex;

void run_person_detection(camera_fb_t *fb) {
    // 1. Hole das Bild aus fb->buf
    // 2. Konvertiere es in das TFLite-Format (GetImage-Funktion)
    // 3. Führe Inferenz aus (Invoke)
    // 4. Lese das Ergebnis
    
    // Beispiel-Ausgabe:
    // if (person_detected) {
    //   ESP_LOGI(TAG, "Person erkannt!");
    // } else {
    //   ESP_LOGI(TAG, "Keine Person.");
    // }
    ESP_LOGI(TAG, "Inferenz ausgeführt (Platzhalter)");
}

// Das ist deine neue Task
void person_detection_task(void *pvParameter) {
    ESP_LOGI(TAG, "Starte Person Detection Task...");
    
    while (true) {
        // 1. Versuche, den Mutex zu bekommen (mit Wartezeit)
        if (xSemaphoreTake(camera_mutex, (TickType_t)50) == pdTRUE) {
            
            // Wir haben das Schloss -> wir dürfen die Kamera benutzen
            camera_fb_t *fb = esp_camera_fb_get();
            if (!fb) {
                ESP_LOGE(TAG, "Inferenz: Kamera-Capture fehlgeschlagen");
            } else {
                // Führe die Erkennung auf dem Bild aus
                run_person_detection(fb);
                
                // Bild-Buffer wieder freigeben
                esp_camera_fb_return(fb);
            }
            
            // Das Schloss wieder freigeben
            xSemaphoreGive(camera_mutex);

        } else {
            // Konnten den Mutex nicht bekommen (Stream ist aktiv und schnell)
            // -> Einfach überspringen und in einer Sekunde erneut versuchen
            ESP_LOGW(TAG, "Inferenz übersprungen, Kamera blockiert.");
        }

        // Warte 1 Sekunde bis zum nächsten Versuch
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// HTTP Stream Handler
esp_err_t stream_handler(httpd_req_t *req){
    // Funktion zur Stream-Implementierung (wird vom Beispiel übernommen)
    // ... Dieser Teil ist sehr umfangreich und wird hier aus Gründen der Übersichtlichkeit
    // weggelassen. Der Standard-Webserver-Handler aus dem IDF-Beispiel funktioniert hier.
    // Ein vereinfachtes Beispiel:
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    char * part_buf[64];

    // HTTP Header für den Stream setzen
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while(true){
        // Versuche, das Schloss zu bekommen
        if (xSemaphoreTake(camera_mutex, (TickType_t)50) == pdTRUE) {
            
            // Wir haben das Schloss
            fb = esp_camera_fb_get();
            if (!fb) {
                ESP_LOGE(TAG, "Stream: Kamera-Capture fehlgeschlagen");
                res = ESP_FAIL;
            } else {
                res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
            }
            esp_camera_fb_return(fb);
            
            // Schloss freigeben, damit die Inferenz-Task eine Chance hat
            xSemaphoreGive(camera_mutex);

        } else {
            // Konnten das Schloss nicht bekommen (passiert selten)
            // Kurz warten und erneut versuchen
            vTaskDelay(pdMS_TO_TICKS(5));
        }

        if(res != ESP_OK){
            break;
        }
    }
    return res;
}

// Webserver starten
void start_webserver(void){
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    ESP_LOGI(TAG, "Starting server on port: '%d'", config.server_port);
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t stream_uri = {
            .uri       = "/",
            .method    = HTTP_GET,
            .handler   = stream_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &stream_uri);
    }
}

// WLAN-Verbindung herstellen
void wifi_init_sta(void) {
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_NAME,
            .password = WIFI_PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");
}

void app_main(void) {
    // Initialisierung von NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialisierung von TCP/IP und Event Loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // ERSTELLE DEN MUTEX
    camera_mutex = xSemaphoreCreateMutex();
    if (camera_mutex == NULL) {
        ESP_LOGE(TAG, "Mutex konnte nicht erstellt werden");
        return; // Abbruch
    }

    // Kamera initialisieren
    ESP_ERROR_CHECK(esp_camera_init(NULL));

    // WLAN verbinden
    wifi_init_sta();

    // Webserver starten
    start_webserver();

    xTaskCreate(
        person_detection_task,  // Funktion, die ausgeführt wird
        "detection_task",       // Name der Task (für Debugging)
        4096,                   // Stack-Größe (TFLite braucht viel!)
        NULL,                   // Parameter für die Task (hier keine)
        5,                      // Priorität (5 ist Standard)
        NULL                    // Task-Handle (hier nicht nötig)
    );
}