# Registriere die Quellen (Sources)
# Wir fügen main.cpp, submitter_implemented.cpp und internally_implemented.cpp hinzu.
# WICHTIG: Wenn deine Model-Daten (z.B. vww01_model_data.cc) auch kompiliert werden müssen,
# müssen sie hier auch gelistet oder als Bibliothek eingebunden werden.

# Suche nach allen .cpp und .c Dateien im src Ordner
file(GLOB_RECURSE SRC_FILES *.cpp *.c)

# Suche nach Model-Dateien in lib/model_data (falls sie dort liegen)
# Passe den Pfad an, falls deine Model-Daten woanders liegen!
file(GLOB_RECURSE MODEL_FILES "../lib/model_data/*.cc" "../lib/model_data/*.cpp" "../lib/model_data/*.c")

idf_component_register(
    SRCS 
        ${SRC_FILES} 
        ${MODEL_FILES}
    
    INCLUDE_DIRS 
        "." 
        "../lib/api"
        "../lib/model_data"
    
    # Hier definieren wir die Abhängigkeiten zu ESP-IDF Komponenten
    # 'esp-tflite-micro' ist der Name des Ordners in 'components/'
    REQUIRES 
        esp-tflite-micro 
        driver 
        esp_timer
        log
)