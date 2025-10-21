// --- C/C++ Standard-Bibliotheken ---
#include <cstdint>
#include <cstddef>
#include <cstdio>   // Für printf
#include <cstring>  // Für memcpy
#include <cmath>

// --- ESP-IDF Framework ---
#include "esp_heap_caps.h" // Für heap_caps_malloc
#include "esp_log.h"       // Für ESP_LOGI, ESP_LOGE
#include "esp_err.h"       // Für esp_err_t, ESP_OK, ESP_FAIL
#include "esp_camera.h"    // Für camera_fb_t und Kamera-Funktionen

// --- FreeRTOS ---
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// --- TensorFlow Lite Micro ---
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h" // Für kTfLiteOk

// --- Projekt-spezifische Bibliotheken & Header ---
#include "img_converters.h"   // Für jpg2rgb565
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "person_detection.h" // Der eigene Header dieser Datei

// --- HILFSFUNKTIONEN (von esp-face kopiert) ---
#define DL_IMAGE_MIN(A, B) ((A) < (B) ? (A) : (B))
#define DL_IMAGE_MAX(A, B) ((A) < (B) ? (B) : (A))
static void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h)
{
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    int dst_stride = dst_c * dst_w;
    int src_stride = dst_c * src_w;
    for (int y = 0; y < dst_h; y++)
    {
        float fy[2];
        fy[0] = (float)((y + 0.5) * scale_y - 0.5); // y
        int src_y = (int)fy[0];                     // y1
        fy[0] -= src_y;                             // y - y1
        fy[1] = 1 - fy[0];                          // y2 - y
        src_y = DL_IMAGE_MAX(0, src_y);
        src_y = DL_IMAGE_MIN(src_y, src_h - 2);

        for (int x = 0; x < dst_w; x++)
        {
            float fx[2];
            fx[0] = (float)((x + 0.5) * scale_x - 0.5); // x
            int src_x = (int)fx[0];                     // x1
            fx[0] -= src_x;                             // x - x1
            if (src_x < 0)
            {
                fx[0] = 0;
                src_x = 0;
            }
            if (src_x > src_w - 2)
            {
                fx[0] = 0;
                src_x = src_w - 2;
            }
            fx[1] = 1 - fx[0]; // x2 - x

            for (int c = 0; c < dst_c; c++)
            {
                dst_image[y * dst_stride + x * dst_c + c] = round(src_image[src_y * src_stride + src_x * dst_c + c] * fx[1] * fy[1] + src_image[src_y * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[1] + src_image[(src_y + 1) * src_stride + src_x * dst_c + c] * fx[1] * fy[0] + src_image[(src_y + 1) * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[0]);
            }
        }
    }
}

static const char *TAG = "PERSON_DETECTION";

const int full_width = 320;
const int full_height = 240;
uint8_t* rgb888_image_full = NULL; // Puffer für 320x240x3
uint8_t* rgb888_image_96x96 = NULL;
int8_t* model_input_buffer = NULL;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
constexpr int kTensorArenaSize = 160 * 1024;
static uint8_t *tensor_arena;

esp_err_t setup_person_detection_model(){

    // Stelle sicher, dass der model_input_buffer allokiert ist
    model_input_buffer = (int8_t*) heap_caps_malloc(kNumRows * kNumCols * kNumChannels, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!model_input_buffer) {
        ESP_LOGE(TAG, "Konnte TFLite-Input-Buffer nicht allokieren");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Input-Puffer (%.1f KB) allokiert.", (kNumRows * kNumCols * kNumChannels)/1024.0);

    // Stelle sicher, dass der rgb565_image_160x120 Puffer allokiert ist
    rgb888_image_full = (uint8_t*) heap_caps_malloc(full_width * full_height * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb888_image_full) {
        ESP_LOGE(TAG, "Speicher für 320x240 RGB888 Puffer konnte nicht allokiert werden");
        heap_caps_free(model_input_buffer);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "RGB888 320x240 Puffer (%.1f KB) allokiert.", (full_width * full_height * 3)/1024.0);

    // Stelle sicher, dass der rgb565_image_96x96 Puffer allokiert ist
    rgb888_image_96x96 = (uint8_t*) heap_caps_malloc(kNumCols * kNumRows * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb888_image_96x96) {
        ESP_LOGE(TAG, "Speicher für 96x96 RGB888 Puffer konnte nicht allokiert werden");
        heap_caps_free(model_input_buffer);
        heap_caps_free(rgb888_image_full);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "RGB888 96x96 Puffer (%.1f KB) allokiert.", (kNumCols * kNumRows * 3)/1024.0);

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_person_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        heap_caps_free(model_input_buffer);
        heap_caps_free(rgb888_image_full);
        heap_caps_free(rgb888_image_96x96);
        return ESP_FAIL;
    }

    if (tensor_arena == NULL) {
        //tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        // Benutze PSRAM (SPIRAM) statt internem RAM
        tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        heap_caps_free(model_input_buffer);
        heap_caps_free(rgb888_image_full);
        heap_caps_free(rgb888_image_96x96);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Tensor-Arena (%.1f KB) allokiert.", kTensorArenaSize/1024.0);

    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        heap_caps_free(model_input_buffer);
        heap_caps_free(rgb888_image_full);
        heap_caps_free(rgb888_image_96x96);
        return ESP_FAIL;
    }

    // Get information about the memory area to use for the model's input.
    input = interpreter->input(0);

    ESP_LOGI(TAG, "===================================");
    ESP_LOGI(TAG, "TFLite-Setup erfolgreich abgeschlossen!");
    ESP_LOGI(TAG, "Alle Puffer und Tensoren sind bereit.");
    ESP_LOGI(TAG, "===================================");
    
    return ESP_OK;
}

void run_person_detection(camera_fb_t *fb) {

    // 1. JPEG (320x240) -> RGB888 (320x240)
    // Wir benutzen die Funktion 'fmt2rgb888', die JPEG dekodieren kann
    bool conversion_ok = fmt2rgb888(fb->buf, fb->len, 
                                  PIXFORMAT_JPEG, // Quellformat
                                  rgb888_image_full); // Ausgabe-Puffer
    
    if (!conversion_ok) {
        ESP_LOGE(TAG, "JPEG -> RGB888 Dekodierung fehlgeschlagen");
        return;
    }
    
    // 2. Skaliere von 320x240 auf 96x96 (Deine Idee!)
    // Wir nutzen die RGB888-Puffer (3 Kanäle)
    image_resize_linear(rgb888_image_96x96,  // 1. Ziel-Puffer (96x96x3)
                        rgb888_image_full,   // 2. Quell-Puffer (320x240x3)
                        kNumCols,            // 3. Ziel-Breite (96)
                        kNumRows,            // 4. Ziel-Höhe (96)
                        3,                   // 5. Ziel-Kanäle (RGB = 3)
                        full_width,          // 6. Quell-Breite (320)
                        full_height);        // 7. Quell-Höhe (240)

    // 3. RGB888 (96x96) -> Grayscale (96x96) für das Modell
    for (int i = 0; i < kNumRows * kNumCols; i++) {
        // Lese die 3 Bytes (R, G, B) aus dem 96x96 RGB-Puffer
        uint8_t r = rgb888_image_96x96[i * 3 + 0];
        uint8_t g = rgb888_image_96x96[i * 3 + 1];
        uint8_t b = rgb888_image_96x96[i * 3 + 2];

        // Gamma-korrigierte RGB-zu-Grayscale-Formel und Quantisierung
        int8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;

        // Schreibe in den TFLite-Input-Puffer
        model_input_buffer[i] = grey_pixel;
    }

    // 7. Setze den Input-Puffer des Modells
    memcpy(input->data.int8, model_input_buffer, kMaxImageSize);

    // 8. Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != interpreter->Invoke()) {
        ESP_LOGE(TAG, "Invoke failed.");
    }
    
    // 9. Lese das Ergebnis
    TfLiteTensor* output = interpreter->output(0);

    // 10. Hole die Scores aus dem Tensor
    // Das Modell gibt 2 Werte im int8_t-Format aus (von -128 bis 127).
    // Wir nutzen die Indizes aus deiner 'model_settings.h'
    int8_t person_score = output->data.int8[kPersonIndex];
    int8_t not_a_person_score = output->data.int8[kNotAPersonIndex];

    // 11. Vergleiche die Scores und gib eine sinnvolle Meldung aus
    // Ein höherer Wert bedeutet eine höhere Wahrscheinlichkeit.
    if (person_score > 50) {
        // ESP_LOGI (aus "esp_log.h") ist besser für die Ausgabe im ESP-IDF
        ESP_LOGI(TAG, "Person erkannt! (Score: %d vs %d)", person_score, not_a_person_score);
    } else {
        ESP_LOGI(TAG, "Keine Person. (Score: %d vs %d)", person_score, not_a_person_score);
    }
}

// Diese Funktion ist der Task, der die Inferenz ausführt.
static void person_detection_task(void *pvParameter) {
    // Der Mutex wird als Parameter übergeben
    SemaphoreHandle_t camera_mutex = (SemaphoreHandle_t)pvParameter;
    ESP_LOGI(TAG, "Starte Person Detection Task...");
    
    while (true) {
        if (xSemaphoreTake(camera_mutex, (TickType_t)50) == pdTRUE) {
            
            camera_fb_t *fb = esp_camera_fb_get();
            if (!fb) {
                ESP_LOGE(TAG, "Inferenz: Kamera-Capture fehlgeschlagen");
            } else {
                run_person_detection(fb);
                esp_camera_fb_return(fb);
            }
            xSemaphoreGive(camera_mutex);

        } else {
            ESP_LOGW(TAG, "Inferenz übersprungen, Kamera blockiert.");
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// Öffentliche Funktion zum Starten des Tasks
void start_detection_task(SemaphoreHandle_t camera_mutex)
{
    xTaskCreate(
        person_detection_task,  // Die Task-Funktion
        "detection_task",       // Name
        8192,                   // Stack (TFLite braucht viel!)
        (void*)camera_mutex,    // Parameter: Der Mutex!
        5,                      // Priorität
        NULL                    // Task-Handle
    );
}