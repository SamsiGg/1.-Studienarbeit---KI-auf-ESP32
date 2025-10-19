# src/CMakeLists.txt

idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS ".")

set(REQUIRES esp_camera esp_httpd)