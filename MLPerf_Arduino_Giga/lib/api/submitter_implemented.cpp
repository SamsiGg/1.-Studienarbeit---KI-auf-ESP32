/*
 * Copyright 2024 The MLPerf Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 *
 * Adapated for Arduino GIGA R1 WiFi (Cortex-M7) by Gemini.
 * UPDATED: Hybrid Allocation (Try Internal RAM -> Fallback to SDRAM)
 */

// 1. API-Header
#include "submitter_implemented.h"
#include "internally_implemented.h" 

// 2. Arduino & C Standard-Bibliotheken
#include <Arduino.h>  
#include <stdarg.h>   
#include <stdio.h>    
#include <string.h>   
#include <SDRAM.h> // Für den Fallback nötig

// 3. TFLM-Header
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===================================================================
// MODELL-KONFIGURATION
// ===================================================================

#if TH_MODEL_VERSION == EE_MODEL_VERSION_IC01
  #include "ic01_model_data.h"
  const unsigned char* g_model = pretrainedResnet_quant_tflite;
  // ResNet ist groß, wird wahrscheinlich im SDRAM landen
  constexpr size_t kTensorArenaSize = 1024 * 1024; 

#elif TH_MODEL_VERSION == EE_MODEL_VERSION_KWS01
  #include "kws01_model_data.h"
  const unsigned char* g_model = kws_ref_model_tflite;
  // Könnte in den internen RAM passen
  constexpr size_t kTensorArenaSize = 100 * 1024;

#elif TH_MODEL_VERSION == EE_MODEL_VERSION_VWW01
  #include "vww01_model_data.h"
  const unsigned char* g_model = vww_96_int8_tflite;
  // Wir fordern genug Platz an. Wenn intern voll -> SDRAM.
  constexpr size_t kTensorArenaSize = 350 * 1024; 

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

// Pointer statt statisches Array. Wir entscheiden zur Laufzeit wohin.
uint8_t* tensor_arena = nullptr;

#if EE_CFG_ENERGY_MODE
  constexpr int TH_GPIO_TIMESTAMP_PIN = 5;
#endif
} // namespace

// ===================================================================
// HILFSFUNKTIONEN
// ===================================================================

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

  #else
    #error "TH_MODEL_VERSION nicht erkannt in AddOpsToResolver()"
  #endif
}

// ===================================================================
// MLPERF API IMPLEMENTIERUNG
// ===================================================================

void th_load_tensor() {
  size_t input_size_bytes = model_input->bytes;
  static uint8_t temp_host_buffer[MAX_DB_INPUT_SIZE];
  
  size_t host_buffer_size = ee_get_buffer(temp_host_buffer, input_size_bytes);

  if (host_buffer_size != input_size_bytes) {
    th_printf("FEHLER: Host-Puffer Groesse (%d) != Tensor Groesse (%d)!\r\n", 
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
  } else {
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
    if (i < output_size - 1) th_printf(",");
  }
  th_printf("]\r\n");
}

void th_infer() {
  if (interpreter->Invoke() != kTfLiteOk) {
    th_printf("FEHLER: interpreter->Invoke() fehlgeschlagen!\r\n");
  }
}

void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE
  digitalWrite(TH_GPIO_TIMESTAMP_PIN, HIGH);
  delayMicroseconds(500);
  digitalWrite(TH_GPIO_TIMESTAMP_PIN, LOW);
#else 
  th_printf(EE_MSG_TIMESTAMP, micros());
#endif
}

void th_printf(const char *fmt, ...) {
  char buffer[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

#if EE_CFG_ENERGY_MODE
  Serial2.print(buffer);
  Serial2.flush(); // Sicherstellen, dass alles rausgeht
  Serial.print(buffer);
  Serial.flush(); // <--- FIX: Warten bis Daten gesendet sind, bevor CPU belastet wird
#else
  Serial.print(buffer);
  Serial.flush(); // <--- FIX: Warten bis Daten gesendet sind, bevor CPU belastet wird
#endif
}

char th_getchar() {
#if EE_CFG_ENERGY_MODE
  HardwareSerial& active_serial = Serial2;
#else
  Stream& active_serial = Serial;
#endif

  while (!active_serial.available()) {
    yield(); 
  }
  char message = active_serial.read();
  //th_printf("%c", message); 
  return message;
}

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE
  Serial2.begin(9600); 
  Serial.begin(115200); 
#else 
  Serial.begin(115200); 
#endif
  
  long start = micros();
  while (!Serial && (micros() - start < 2000000)) { 
    yield();
  }
}

void th_timestamp_initialize(void) {
}

void th_final_initialize(void) {
  model = tflite::GetModel(g_model);
  
  AddOpsToResolver();
 
  // --- INTELLIGENTE SPEICHERZUWEISUNG ---
  
  // Größe + Alignment Puffer
  size_t alloc_size = kTensorArenaSize + 16;
  void* raw_mem = nullptr;
  bool using_sdram = false;

  // 1. Versuch: Interner RAM (malloc)
  // Mbed OS malloc nutzt den verfügbaren Heap im AXI SRAM
  raw_mem = malloc(alloc_size);

  if (raw_mem != nullptr) {
    th_printf("DEBUG: Interner RAM erfolgreich zugewiesen!\r\n");
  } else {
    // 2. Fallback: SDRAM
    th_printf("DEBUG: Interner RAM voll. Versuche SDRAM...\r\n");
    SDRAM.begin();
    raw_mem = SDRAM.malloc(alloc_size);
    using_sdram = true;
  }

  if (raw_mem == nullptr) {
    th_printf("FEHLER: Speicher voll! Weder RAM noch SDRAM verfuegbar.\r\n");
    return;
  }

  // Manuelles Alignment auf 16 Byte
  tensor_arena = (uint8_t*)((uintptr_t)((uint8_t*)raw_mem + 15) & ~15);

  th_printf("Arena Adresse: 0x%X (%s)\r\n", (uintptr_t)tensor_arena, using_sdram ? "SDRAM" : "INTERNAL");

  static tflite::MicroInterpreter static_interpreter(
      model, *op_resolver, tensor_arena, kTensorArenaSize);
      
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    th_printf("FEHLER: AllocateTensors() fehlgeschlagen.\r\n");
    th_printf("Benötigt: %d\r\n", interpreter->arena_used_bytes());
    return;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  th_printf("DEBUG: Initialisierung abgeschlossen.\r\n");
  th_printf("Arena Used Bytes: %d\r\n", interpreter->arena_used_bytes());
  
  #if EE_CFG_ENERGY_MODE
    pinMode(TH_GPIO_TIMESTAMP_PIN, OUTPUT);
    digitalWrite(TH_GPIO_TIMESTAMP_PIN, LOW);
  #endif
}

void th_pre() {}
void th_post() {}
void th_command_ready(char volatile *msg) {
  ee_serial_command_parser_callback((char*) msg);
}

// Linker Hooks
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
  vsnprintf(buffer, sizeof(buffer), format, ap);
  Serial.print(buffer);
  return strlen(buffer);
}