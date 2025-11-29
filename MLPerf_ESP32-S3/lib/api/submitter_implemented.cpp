/*
 * submitter_implemented.cpp
 *
 * Portierung für Arduino Nano ESP32 (ESP32-S3).
 * - Nutzt Native USB (USB Serial JTAG) statt UART
 * - Nutzt MALLOC_CAP_SPIRAM für 8MB PSRAM Zugriff
 */

// 1. API-Header
#include "submitter_implemented.h"
#include "internally_implemented.h" 

// 2. ESP-IDF & C Standard-Bibliotheken
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
// WICHTIG: USB Treiber für S3 Native USB
#include "driver/usb_serial_jtag.h" 
#include "driver/gpio.h"
#include "esp_timer.h"
#include "rom/ets_sys.h"
#include "esp_heap_caps.h" // Für PSRAM Zugriff

// 3. TFLM-Header
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===================================================================
// DEINE MODELL-KONFIGURATION
// ===================================================================

#if TH_MODEL_VERSION == EE_MODEL_VERSION_IC01
  #include "ic01_model_data.h" 
  const unsigned char* g_model = pretrainedResnet_quant_tflite;
  constexpr size_t kTensorArenaSize = 200 * 1024; 

#elif TH_MODEL_VERSION == EE_MODEL_VERSION_KWS01
  #include "kws01_model_data.h"
  const unsigned char* g_model = kws_ref_model_tflite;
  constexpr size_t kTensorArenaSize = 150 * 1024; 

#elif TH_MODEL_VERSION == EE_MODEL_VERSION_VWW01
  #include "vww01_model_data.h" 
  const unsigned char* g_model = vww_96_int8_tflite;
  // WICHTIG: Wir nutzen PSRAM, also gönnen wir uns 1 MB Arena!
  // Das verhindert jegliche Engpässe.
  constexpr size_t kTensorArenaSize = 300 * 1024; 

#elif TH_MODEL_VERSION == EE_MODEL_VERSION_AD01
  #include "ad01_model_data.h" 
  const unsigned char* g_model = ad01_int8_tflite;
  constexpr size_t kTensorArenaSize = 100 * 1024; 

#elif TH_MODEL_VERSION == EE_MODEL_VERSION_STRWW01
  #include "strww01_model_data.h" 
  const unsigned char* g_model = str_ww_ref_model_tflite;
  constexpr size_t kTensorArenaSize = 100 * 1024; 

#else
  #error "TH_MODEL_VERSION wurde nicht auf ein gültiges Modell gesetzt!"
#endif

// ===================================================================
// GLOBALE VARIABLEN
// ===================================================================

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
tflite::MicroOpResolver* op_resolver = nullptr;

// Zeiger für dynamische Arena im PSRAM
uint8_t* tensor_arena = nullptr;

#if EE_CFG_ENERGY_MODE
  const gpio_num_t TH_GPIO_TIMESTAMP_PIN = GPIO_NUM_4;
#endif
} // namespace

// HINWEIS: DebugLog kommt aus esp-tflite-micro Lib.

/**
 * @brief Fügt TFLM-Operatoren hinzu
 */
void AddOpsToResolver() {
  #if TH_MODEL_VERSION == EE_MODEL_VERSION_IC01 
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddAdd(); 
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    op_resolver = &micro_op_resolver;

  #elif TH_MODEL_VERSION == EE_MODEL_VERSION_KWS01 
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    op_resolver = &micro_op_resolver;

  #elif TH_MODEL_VERSION == EE_MODEL_VERSION_VWW01 
    // Resolver für VWW01
    static tflite::MicroMutableOpResolver<7> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddMean(); 
    op_resolver = &micro_op_resolver;

  #elif TH_MODEL_VERSION == EE_MODEL_VERSION_AD01 
    static tflite::MicroMutableOpResolver<2> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddRelu();
    op_resolver = &micro_op_resolver;

  #elif TH_MODEL_VERSION == EE_MODEL_VERSION_STRWW01 
    static tflite::MicroMutableOpResolver<4> micro_op_resolver;
    micro_op_resolver.AddUnidirectionalSequenceLSTM(); 
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();
    op_resolver = &micro_op_resolver;

  #else
    #error "TH_MODEL_VERSION nicht erkannt in AddOpsToResolver()"
  #endif
}

void th_load_tensor() {
  size_t input_size_bytes = model_input->bytes;
  static uint8_t temp_host_buffer[MAX_DB_INPUT_SIZE];
  
  size_t host_buffer_size = ee_get_buffer(temp_host_buffer, input_size_bytes);

  if (host_buffer_size != input_size_bytes) {
    th_printf("FEHLER: Host-Puffer (%d) passt nicht zur Tensor-Groesse (%d)!\r\n", 
              (int)host_buffer_size, (int)input_size_bytes);
  }

  if (model_input->type == kTfLiteInt8) {
    int8_t* tensor_data = model_input->data.int8;

#if (TH_MODEL_VERSION == EE_MODEL_VERSION_IC01) || (TH_MODEL_VERSION == EE_MODEL_VERSION_VWW01)
    for (size_t i = 0; i < input_size_bytes; i++) {
      int16_t temp = (int16_t)temp_host_buffer[i] - 128;
      tensor_data[i] = (int8_t)temp;
    }
#else
    memcpy(tensor_data, temp_host_buffer, input_size_bytes);
#endif
  } 
  else if (model_input->type == kTfLiteUInt8) {
    uint8_t* tensor_data = model_input->data.uint8;
    memcpy(tensor_data, temp_host_buffer, input_size_bytes);
  }
  else {
    th_printf("FEHLER: Unbekannter Input-Tensor-Typ!");
  }
}

void th_results() {
  th_printf("m-results-[");

  TfLiteTensor* output = model_output;
  size_t output_size = output->dims->data[output->dims->size - 1];
  float scale = output->params.scale;
  int32_t zero_point = output->params.zero_point;
  int8_t* output_data = output->data.int8;

  for (size_t i = 0; i < output_size; i++) {
    float float_val = ((float)output_data[i] - (float)zero_point) * scale;
    th_printf("%f", float_val);
    if (i < output_size - 1) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

void th_infer() {
  if (interpreter->Invoke() != kTfLiteOk) {
    th_printf("FEHLER: interpreter->Invoke() ist fehlgeschlagen!");
  }
}

void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 1);
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 0);
  ets_delay_us(2);
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 1);
#else 
  th_printf(EE_MSG_TIMESTAMP, (unsigned long)esp_timer_get_time());
#endif
}

// ÄNDERUNG: Nutzt Native USB Serial JTAG
void th_printf(const char *fmt, ...) {
  char buffer[256]; // Größerer Buffer schadet im PSRAM nicht
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  
  usb_serial_jtag_write_bytes(buffer, strlen(buffer), portMAX_DELAY);
}

// ÄNDERUNG: Nutzt Native USB Serial JTAG
char th_getchar() {
  uint8_t data = 0;
  int len = 0;
  while (len <= 0) {
    len = usb_serial_jtag_read_bytes(&data, 1, 10 / portTICK_PERIOD_MS);
    if (len <= 0) {
        taskYIELD(); 
    }
  }
  th_printf("%c", (char)data);
  return (char)data;
}

void th_serialport_initialize(void) {
  // WICHTIG: Native USB braucht Zeit zum Initialisieren auf dem Host (PC/Mac).
  // Ohne diese Pause gehen die ersten Print-Ausgaben verloren, 
  // weil der Port am PC noch nicht offen ist.
  
  // Warte 2 Sekunden (2000 ms)
  vTaskDelay(pdMS_TO_TICKS(2000));
  
  // Optional: Leere den Input-Buffer
  usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
      .tx_buffer_size = 1024,
      .rx_buffer_size = 1024,
  };
  // Treiber installieren, falls noch nicht in main geschehen (zur Sicherheit)
  if (usb_serial_jtag_driver_install(&usb_serial_jtag_config) == ESP_OK) {
      // War schon installiert? Egal.
  }
}

void th_timestamp_initialize(void) {
}

/**
 * @brief Init TFLM mit intelligenter Speicherwahl (SRAM -> PSRAM Fallback)
 */
void th_final_initialize(void) {
  
  // 1. Diagnose: Zeige freien Speicher (Intern & PSRAM)
  size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  
  th_printf("DEBUG: Freier Interner RAM: %d Bytes\r\n", free_internal);
  th_printf("DEBUG: Freier PSRAM (Ext): %d Bytes\r\n", free_psram);
  th_printf("DEBUG: Benoetigte Arena: %d Bytes\r\n", kTensorArenaSize);

  // 2. Speicher allokieren (Priorität: Intern > Extern)
  if (tensor_arena == nullptr) {
      
      // VERSUCH 1: Interner SRAM (Schnell & Geringe Latenz)
      // MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT zwingt die Allokation in den internen RAM.
      tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

      if (tensor_arena != nullptr) {
          th_printf("ERFOLG: Arena liegt im INTERNEN SRAM (Max Performance!).\r\n");
      } 
      else {
          // VERSUCH 2: PSRAM (Fallback, wenn intern kein Platz ist)
          th_printf("WARNUNG: Interner RAM nicht ausreichend/fragmentiert. Wechsle auf PSRAM...\r\n");
          
          // MALLOC_CAP_SPIRAM zwingt die Allokation in den externen Chip.
          tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
          
          if (tensor_arena != nullptr) {
              th_printf("ERFOLG: Arena liegt im PSRAM (Sicher, aber etwas langsamer).\r\n");
          }
      }
  }

  // 3. Prüfen ob Allocation überhaupt irgendwo erfolgreich war
  if (tensor_arena == nullptr) {
      th_printf("FATAL ERROR: Speicherallokation fehlgeschlagen! Weder im SRAM noch im PSRAM war Platz.\r\n");
      return;
  }

  model = tflite::GetModel(g_model);
  AddOpsToResolver();
 
  static tflite::MicroInterpreter static_interpreter(
      model, *op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    th_printf("FEHLER: AllocateTensors() fehlgeschlagen.\n");
    return;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  
  #if EE_CFG_ENERGY_MODE
  gpio_config_t io_conf = {};
  io_conf.intr_type = GPIO_INTR_DISABLE;
  io_conf.mode = GPIO_MODE_OUTPUT;
  io_conf.pin_bit_mask = (1ULL << TH_GPIO_TIMESTAMP_PIN);
  gpio_config(&io_conf);
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 1);
  th_printf("DEBUG: Energie-Modus init. GPIO %d\r\n", TH_GPIO_TIMESTAMP_PIN);
  #else
  th_printf("DEBUG: Performance-Modus init.\r\n");
  #endif
}

void th_pre() {}
void th_post() {}
void th_command_ready(char volatile *msg) {
  ee_serial_command_parser_callback((char*) msg);
}

// LIBC HOOKS (unverändert)
int th_strncmp(const char *str1, const char *str2, size_t n) { return strncmp(str1, str2, n); }
char *th_strncpy(char *dest, const char *src, size_t n) { return strncpy(dest, src, n); }
size_t th_strnlen(const char *str, size_t maxlen) { return strnlen(str, maxlen); }
char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }
char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }
int th_atoi(const char *str) { return atoi(str); }
void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }
void *th_memcpy(void *dst, const void *src, size_t n) { return memcpy(dst, src, n); }
int th_vprintf(const char *format, va_list ap) {
  char buffer[256];
  int ret = vsnprintf(buffer, sizeof(buffer), format, ap);
  usb_serial_jtag_write_bytes(buffer, ret, 0);
  return ret;
}