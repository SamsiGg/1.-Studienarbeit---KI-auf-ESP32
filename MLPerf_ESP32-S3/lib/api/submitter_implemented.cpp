/*
 * submitter_implemented.cpp
 *
 * Portierung für Arduino Nano ESP32 (ESP32-S3).
 * - PERFORMANCE MODE: Nutzt Native USB (USB Serial JTAG)
 * - ENERGY MODE: Nutzt Hardware UART (Pin 43/44 laut ESP32-S3 Mapping auf Nano)
 * - Speicher: Nutzt PSRAM Fallback
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
#include "driver/gpio.h"
#include "esp_timer.h"
#include "rom/ets_sys.h"
#include "esp_heap_caps.h" 

// TREIBER-WEICHE
#if EE_CFG_ENERGY_MODE
    #include "driver/uart.h"
    // Beim Nano ESP32 sind TX0/RX0 auf GPIO 43 und 44 gemappt (siehe Pinout unten)
    // Wir nutzen UART_NUM_0
    #define EX_UART_NUM UART_NUM_0
    #define TX_PIN GPIO_NUM_43 // Nano Pin D1 (TX)
    #define RX_PIN GPIO_NUM_44 // Nano Pin D0 (RX)
#else
    #include "driver/usb_serial_jtag.h" 
#endif

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

uint8_t* tensor_arena = nullptr;

#if EE_CFG_ENERGY_MODE
  // WICHTIG: Pin D2 auf dem Nano ESP32 ist GPIO 5? 
  // BITTE PRÜFEN: Arduino Nano ESP32 Pinout!
  // D2 = GPIO 5, D3 = GPIO 6, D4 = GPIO 7.
  // Wir nehmen D2 (GPIO 5) für den Timestamp.
  const gpio_num_t TH_GPIO_TIMESTAMP_PIN = GPIO_NUM_5;
#endif
} // namespace


// --- RESOLVER & TENSOR LOAD (Unverändert) ---
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
  #endif
}

void th_load_tensor() {
  size_t input_size_bytes = model_input->bytes;
  static uint8_t temp_host_buffer[MAX_DB_INPUT_SIZE];
  
  size_t host_buffer_size = ee_get_buffer(temp_host_buffer, input_size_bytes);

  if (host_buffer_size != input_size_bytes) {
    th_printf("FEHLER: Buffer Size Mismatch!\r\n");
  }

  if (model_input->type == kTfLiteInt8) {
    int8_t* tensor_data = model_input->data.int8;
#if (TH_MODEL_VERSION == EE_MODEL_VERSION_IC01) || (TH_MODEL_VERSION == EE_MODEL_VERSION_VWW01)
    for (size_t i = 0; i < input_size_bytes; i++) {
      tensor_data[i] = (int8_t)((int16_t)temp_host_buffer[i] - 128);
    }
#else
    memcpy(tensor_data, temp_host_buffer, input_size_bytes);
#endif
  } 
  else if (model_input->type == kTfLiteUInt8) {
    memcpy(model_input->data.uint8, temp_host_buffer, input_size_bytes);
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
    if (i < output_size - 1) th_printf(",");
  }
  th_printf("]\r\n");
}

void th_infer() {
  if (interpreter->Invoke() != kTfLiteOk) {
    th_printf("FEHLER: interpreter->Invoke() failed!\r\n");
  }
}

void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE
  // Kurzer Impuls für Joulescope
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 1);
  ets_delay_us(100); // 100us ist sicher für Joulescope 10kHz+
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 0);
#else 
  th_printf(EE_MSG_TIMESTAMP, (unsigned long)esp_timer_get_time());
#endif
}

// -----------------------------------------------------------
// HYBRID COMMUNICATION LAYER (USB JTAG vs UART)
// -----------------------------------------------------------

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE
    // --- ENERGY MODE (UART) ---
    uart_config_t uart_config = {
        .baud_rate = 9600, // Langsam für IO Manager
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB,
    };
    // Installiere UART Treiber (Puffergröße 512)
    ESP_ERROR_CHECK(uart_driver_install(EX_UART_NUM, 512, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(EX_UART_NUM, &uart_config));
    
    // Setze Pins: TX=43 (D1), RX=44 (D0) auf Nano ESP32
    ESP_ERROR_CHECK(uart_set_pin(EX_UART_NUM, TX_PIN, RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    
#else
    // --- PERFORMANCE MODE (USB) ---
    // Warte kurz, damit PC Treiber ready ist
    vTaskDelay(pdMS_TO_TICKS(1500));
    
    // Treiber ist oft schon in main installiert, aber sicher ist sicher:
    usb_serial_jtag_driver_config_t usb_config = {
        .tx_buffer_size = 1024, .rx_buffer_size = 1024
    };
    if (usb_serial_jtag_driver_install(&usb_config) != ESP_OK) {
        // Ignoriere Fehler, falls schon installiert
    }
#endif
}

void th_printf(const char *fmt, ...) {
  char buffer[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  
#if EE_CFG_ENERGY_MODE
  // Sende via UART
  uart_write_bytes(EX_UART_NUM, buffer, strlen(buffer));
#else
  // Sende via USB
  usb_serial_jtag_write_bytes(buffer, strlen(buffer), portMAX_DELAY);
#endif
}

char th_getchar() {
  uint8_t data = 0;
  int len = 0;
  while (len <= 0) {
#if EE_CFG_ENERGY_MODE
    // Lese von UART (Blockierend mit Timeout/Yield)
    len = uart_read_bytes(EX_UART_NUM, &data, 1, 10 / portTICK_PERIOD_MS);
#else
    // Lese von USB
    len = usb_serial_jtag_read_bytes(&data, 1, 10 / portTICK_PERIOD_MS);
#endif
    if (len <= 0) taskYIELD(); 
  }
#if !EE_CFG_ENERGY_MODE
  // Echo nur im Performance Mode (USB), im Energy Mode stört es den Runner
  th_printf("%c", (char)data);
#endif
  return (char)data;
}

// -----------------------------------------------------------
// INITIALISIERUNG
// -----------------------------------------------------------

void th_timestamp_initialize(void) { }

void th_final_initialize(void) {
  
  // Speicher Allocation (SRAM -> PSRAM Fallback)
  if (tensor_arena == nullptr) {
      // Versuch 1: Intern
      tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      
      if (tensor_arena == nullptr) {
          th_printf("WARN: Interner RAM voll. Versuche PSRAM...\r\n");
          // Versuch 2: Extern (PSRAM)
          tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
      }
  }

  if (tensor_arena == nullptr) {
      th_printf("FATAL: Out of Memory!\r\n");
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
  // Init Timestamp Pin
  gpio_reset_pin(TH_GPIO_TIMESTAMP_PIN);
  gpio_set_direction(TH_GPIO_TIMESTAMP_PIN, GPIO_MODE_OUTPUT);
  gpio_set_level(TH_GPIO_TIMESTAMP_PIN, 0);
  th_printf("DEBUG: Energy Mode Ready. Timestamp GPIO %d\r\n", (int)TH_GPIO_TIMESTAMP_PIN);
  #else
  th_printf("DEBUG: Performance Mode Ready (USB).\r\n");
  #endif
}

void th_pre() {}
void th_post() {}
void th_command_ready(char volatile *msg) {
  ee_serial_command_parser_callback((char*) msg);
}

// LIBC HOOKS (Für printf Wrappers etc.)
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
#if EE_CFG_ENERGY_MODE
  uart_write_bytes(EX_UART_NUM, buffer, ret);
#else
  usb_serial_jtag_write_bytes(buffer, ret, 0);
#endif
  return ret;
}