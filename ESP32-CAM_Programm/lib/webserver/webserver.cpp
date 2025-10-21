#if ENABLE_WEBSERVER

// --- C/C++ Standard-Bibliotheken ---
#include <cstdio>   // Für snprintf
#include <cstring>  // Für strlen

// --- ESP-IDF Framework ---
#include "esp_log.h"
#include "esp_err.h"         // Für esp_err_t, ESP_ERROR_CHECK
#include "esp_http_server.h"
#include "esp_camera.h"
#include "esp_event.h"
#include "esp_netif.h"       // Für IPSTR, IP2STR

// --- FreeRTOS ---
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// --- Projekt-Header ---
#include "webserver.h"

#include "img_converters.h"
#include "person_detection.h" 
#include "model_settings.h"

static const char *TAG = "WEBSERVER";

#define PART_BOUNDARY "_--FRAME--"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// HTTP Stream Handler (KORRIGIERT FÜR MJPEG)
esp_err_t stream_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    char part_hdr_buf[64]; // Wir brauchen einen echten Puffer für den Header

    SemaphoreHandle_t camera_mutex = (SemaphoreHandle_t)req->user_ctx;

    // HTTP Header für den Stream setzen (NEUER HEADER)
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    // Schleife starten
    while(true){
        // Versuche, das Schloss zu bekommen
        if (xSemaphoreTake(camera_mutex, (TickType_t)50) == pdTRUE) {
            
            // Wir haben das Schloss
            fb = esp_camera_fb_get();
            if (!fb) {
                ESP_LOGE(TAG, "Stream: Kamera-Capture fehlgeschlagen");
                res = ESP_FAIL;
            } else {
                // Wir haben ein Bild (fb)
                
                // 1. Sende den Header für DIESES Bild (Typ + Länge)
                size_t hlen = snprintf(part_hdr_buf, 64, _STREAM_PART, fb->len);
                res = httpd_resp_send_chunk(req, part_hdr_buf, hlen);

                // 2. Sende die eigentlichen Bilddaten
                if (res == ESP_OK) {
                    res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
                }

                // 3. Sende den Boundary-Marker (Trenner zum nächsten Bild)
                if (res == ESP_OK) {
                    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
                }
            }
            
            // Bild-Buffer freigeben, egal ob Senden klappte oder nicht
            if (fb) {
                esp_camera_fb_return(fb);
                fb = NULL; // Wichtig: Zeiger zurücksetzen
            }
            
            // Schloss freigeben
            xSemaphoreGive(camera_mutex);

            vTaskDelay(pdMS_TO_TICKS(10));

        } else {
            // Konnten das Schloss nicht bekommen (passiert selten)
            // Kurz warten und erneut versuchen
            vTaskDelay(pdMS_TO_TICKS(5));
        }

        // Wenn der Client die Verbindung getrennt hat, Schleife beenden
        if(res != ESP_OK){
            break;
        }
    } // Ende while(true)
    
    return res;
}

// HTTP Stream Handler (KORRIGIERT FÜR MJPEG)
esp_err_t debug_stream_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    char part_hdr_buf[64]; // Wir brauchen einen echten Puffer für den Header

    SemaphoreHandle_t camera_mutex = (SemaphoreHandle_t)req->user_ctx;

    // HTTP Header für den Stream setzen (NEUER HEADER)
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    // Schleife starten
    while(true){
        // Versuche, das Schloss zu bekommen
        if (xSemaphoreTake(camera_mutex, (TickType_t)50) == pdTRUE) {
            
            // Wir haben das Schloss
            fb = esp_camera_fb_get();
            if (!fb) {
                ESP_LOGE(TAG, "Stream: Kamera-Capture fehlgeschlagen");
                res = ESP_FAIL;
            } else {
                // --- START DEBUG-STREAM ANPASSUNG ---

                // 1. Führe die Inferenz aus, um model_input_buffer zu füllen
                //    (Stelle sicher, dass run_person_detection und model_input_buffer
                //     in person_detection.h als 'extern' deklariert sind)
                run_person_detection(fb);

                // 2. Prüfe, ob die Puffer bereit sind
                //if (model_input_buffer && temp_gray_buf) {
                if (rgb888_image_full) {
                    
                    /*// 3. Konvertiere int8_t [-128, 127] zu uint8_t [0, 255]
                    for (int i = 0; i < kNumCols * kNumRows; i++) {
                        temp_gray_buf[i] = (uint8_t)(model_input_buffer[i] + 128);
                    }

                    // 4. Konvertiere das 96x96 Graustufenbild (uint8_t) zu JPEG
                    uint8_t *jpeg_buf = NULL;
                    size_t jpeg_len = 0;
                    bool ok = fmt2jpg(temp_gray_buf,         // Quelle
                                     kNumCols * kNumRows,   // Quellenlänge
                                     kNumCols, kNumRows,    // Breite, Höhe
                                     PIXFORMAT_GRAYSCALE,   // Quellformat
                                     12,                    // Qualität
                                     &jpeg_buf, &jpeg_len); // Output*/

                    // 3. Konvertiere das 160x120 RGB565-Bild direkt zu JPEG
                    uint8_t *jpeg_buf = NULL;
                    size_t jpeg_len = 0;
                    bool ok = fmt2jpg((uint8_t*)rgb888_image_full, // Quelle
                                     full_width * full_height * 3, // Quellenlänge (WICHTIG: * 2 Bytes)
                                     full_width, full_height, // Breite, Höhe
                                     PIXFORMAT_RGB888,   // Quellformat (WICHTIG!)
                                     12,                 // Qualität
                                     &jpeg_buf, &jpeg_len); // Output

                    if (ok) {
                        // 5. Sende das 96x96 JPEG
                        size_t hlen = snprintf(part_hdr_buf, 64, _STREAM_PART, jpeg_len);
                        res = httpd_resp_send_chunk(req, part_hdr_buf, hlen);

                        if (res == ESP_OK) {
                            res = httpd_resp_send_chunk(req, (const char *)jpeg_buf, jpeg_len);
                        }
                        if (res == ESP_OK) {
                            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
                        }
                        
                        free(jpeg_buf); // Wichtig! JPEG-Puffer freigeben
                    } else {
                        ESP_LOGE(TAG, "fmt2jpg Konvertierung fehlgeschlagen");
                        res = ESP_FAIL;
                    }

                } else {
                    ESP_LOGE(TAG, "model_input_buffer oder temp_gray_buf ist NULL");
                    res = ESP_FAIL;
                }
                
                // --- ENDE DEBUG-STREAM ANPASSUNG ---
            }
            
            // Bild-Buffer freigeben, egal ob Senden klappte oder nicht
            if (fb) {
                esp_camera_fb_return(fb); // Gib das Originalbild frei
                fb = NULL; // Wichtig: Zeiger zurücksetzen
            }
            
            // Schloss freigeben
            xSemaphoreGive(camera_mutex);

            // WICHTIG: Kurze Pause, damit der Inferenz-Task (falls er
            // parallel läuft) auch eine Chance hat.
            vTaskDelay(pdMS_TO_TICKS(10)); 

        } else {
            // Konnten das Schloss nicht bekommen (passiert selten)
            // Kurz warten und erneut versuchen
            vTaskDelay(pdMS_TO_TICKS(5));
        }

        // Wenn der Client die Verbindung getrennt hat, Schleife beenden
        if(res != ESP_OK){
            break;
        }
    } // Ende while(true)

    // free(temp_gray_buf); // Lassen wir allokiert, da der Server weiterlaufen könnte
    
    return res;
}

// Event-Handler NUR für die IP-Adresse (gehört logisch zum Webserver)
static void ip_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data)
{
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "===================================");
        ESP_LOGI(TAG, "Webserver gestartet! Zugriff unter:");
        ESP_LOGI(TAG, "http://" IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "===================================");
    }
}

// Webserver starten
esp_err_t start_camera_webserver(SemaphoreHandle_t camera_mutex){
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    // Registriere den IP-Handler, um die URL anzuzeigen
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &ip_event_handler, NULL));

    ESP_LOGI(TAG, "Starting server on port: '%d'", config.server_port);
    if (httpd_start(&server, &config) == ESP_OK) {

        // 1. Der NORMALE Stream auf der Hauptseite ("/")
        httpd_uri_t stream_uri_main = {
            .uri       = "/",
            .method    = HTTP_GET,
            .handler   = stream_handler, // Der Original-Handler
            .user_ctx  = (void*)camera_mutex
        };
        httpd_register_uri_handler(server, &stream_uri_main);

        // 2. Der DEBUG Stream auf "/debug"
        httpd_uri_t stream_uri_debug = {
            .uri       = "/debug",
            .method    = HTTP_GET,
            .handler   = debug_stream_handler, // Dein neuer Debug-Handler
            .user_ctx  = (void*)camera_mutex
        };
        httpd_register_uri_handler(server, &stream_uri_debug);

        return ESP_OK;
    }
    return ESP_FAIL;
}
#endif // ENABLE_WEBSERVER