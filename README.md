# Analyse von TinyML-Inferenz auf dem ESP32 vs. High-Performance-Mikrocontrollern

Dieses Repository beinhaltet den Quellcode, die Messdaten und die Auswertungsskripte meiner Studienarbeit an der DHBW Mannheim.

Ziel der Arbeit war es, die Eignung des kostengünstigen **ESP32** (und ESP32-S3) für Machine-Learning-Anwendungen im Vergleich zu leistungsstarken **ARM Cortex-M7** Systemen zu evaluieren. Hierfür wurde ein Benchmarking auf Basis des Industriestandards **MLPerf Tiny (Closed Division)** durchgeführt.

📄 **Die vollständige schriftliche Ausarbeitung (PDF) ist ebenfalls in diesem Repository verfügbar.**

## 📂 Struktur des Repositories

Das Repository ist als Workspace organisiert. Die wichtigsten Ordner sind die jeweiligen `MLPerf`-Implementierungen, welche als eigenständige **PlatformIO**-Umgebungen angelegt sind.

### Hauptprojekte (Benchmark)
Diese Ordner enthalten den vollständigen Code, um die MLPerf-Benchmarks (Keyword Spotting, Image Classification, Visual Wake Words) auf der jeweiligen Hardware auszuführen. Sie können direkt mit PlatformIO geöffnet und auf den Mikrocontroller geflasht werden.

* `📂 MLPerf_ESP32-Wroom-32` - Implementierung für den generischen ESP32 (Xtensa LX6).
* `📂 MLPerf_ESP32-S3` - Optimierte Implementierung für den ESP32-S3 (Xtensa LX7 mit Vektor-Instruktionen).
* `📂 MLPerf Teensy 4.0` - Referenz-Implementierung für den Teensy 4.0 (Cortex-M7).
* `📂 MLPerf_Arduino_Giga` - Implementierung für den Arduino Giga R1 (Cortex-M7).

### Hilfsprojekte & Tools
Zusätzlich zu den Benchmarks befinden sich hier Projekte, die zum Verständnis der Materie oder zur Datenauswertung erstellt wurden:

* `📂 Python Modellerstellung` - Python-Skripte zur Aufbereitung der Messdaten und Erstellung der Diagramme für die Arbeit.
* `📂 ESP32-CAM_Programm` & `📂 ESP32-Wroom-32_Programm` - Kleinere Hilfsprojekte und "Playgrounds", die zur Einarbeitung in die Thematik und zum Testen von Einzelkomponenten dienten.

## 🚀 Nutzung & Konfiguration

⚠️ **Voraussetzung:** Der **EEMBC Runner** (die Host-Software zur Steuerung des Benchmarks) ist **nicht** in diesem Repository enthalten. Er ist im offiziellen MLCommons Repository zu finden:
👉 [https://github.com/mlcommons/tiny](https://github.com/mlcommons/tiny)

📦 **Plug & Play:** Da es sich um PlatformIO-Projekte handelt, sind **keine manuellen Bibliotheks-Installationen** notwendig. Alle Abhängigkeiten werden automatisch durch die Projektkonfiguration verwaltet.

### 1. Modellauswahl (`platformio.ini`)
Das zu testende neuronale Netz wird über ein Define in der `platformio.ini` Datei des jeweiligen Projekts festgelegt. Um das Modell zu wechseln, muss das entsprechende Flag gesetzt werden (die anderen sollten auskommentiert oder entfernt sein):

* `EE_MODEL_VERSION_KWS01` - Keyword Spotting
* `EE_MODEL_VERSION_IC01`  - Image Classification
* `EE_MODEL_VERSION_VWW01` - Visual Wake Words

### 2. Test-Modus (Environment)
Für die verschiedenen Messarten (Performance, Energie) sind in PlatformIO separate Umgebungen (**Environments**) vorkonfiguriert. Wähle vor dem Kompilieren/Flashen einfach die selbsterklärende Umgebung aus der Liste aus (z.B. `env:teensy40_perf`, `env:teensy40_energy`).

## 🛠 Hardware & Software Stack

**Untersuchte Hardware:**
* **Espressif:** Lolin D32 (ESP32), Arduino Nano ESP32 (S3)
* **ARM Cortex-M7:** Teensy 4.0, Arduino Giga R1

**Software:**
* **Framework:** TensorFlow Lite for Microcontrollers (TFLM)
* **IDE/Build System:** VS Code mit PlatformIO
* **Messung:** EEMBC MLPerf Runner & Joulescope (JS220)

## 📊 Zusammenfassung der Ergebnisse

Die Untersuchung hat gezeigt, dass der Cortex-M7 (Teensy 4.0) sowohl in der Geschwindigkeit als auch der Energieeffizienz führend ist. Der **ESP32-S3** konnte jedoch durch seine Vektor-Instruktionen die Lücke deutlich verkleinern und bietet einen guten Kompromiss aus Preis und Leistung. Der klassische **ESP32** stößt bei komplexen Vision-Modellen an Speichergrenzen, eignet sich aber weiterhin hervorragend für kostensensitive Audio-Anwendungen (z.B. Keyword Spotting).

---
*Erstellt von Samuel Geffert im Rahmen der Studienarbeit, Januar 2026.*
