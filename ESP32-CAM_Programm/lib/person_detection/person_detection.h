#pragma once

#include <esp_camera.h>
#include "esp_err.h"
#include "freertos/semphr.h"

esp_err_t setup_person_detection_model();

/**
 * @brief Startet den FreeRTOS-Task für die Personenerkennung.
 * * @param camera_mutex Der Mutex für den exklusiven Kamerazugriff.
 */
void start_detection_task(SemaphoreHandle_t camera_mutex);
void run_person_detection(camera_fb_t *fb);

extern int8_t* model_input_buffer;
extern uint8_t* rgb888_image_full;
extern uint8_t* rgb888_image_96x96;
extern const int full_width;
extern const int full_height;