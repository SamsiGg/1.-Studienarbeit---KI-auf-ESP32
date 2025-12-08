import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURATION ---
DATA_DIR = "Data"         # Dein Ordnername
OUTPUT_DIR = "."          # Wo die Bilder gespeichert werden
SAVE_FORMAT = "png"       # Dateiformat der Bilder

# Feste Reihenfolge für die Diagramme
MODEL_ORDER = ["kws", "ic", "vww"]
MCU_ORDER   = ["esp32", "s3", "giga", "teensy"]

def parse_energy_txt(filepath):
    """
    Liest die results.txt ein und sucht nach gültigen Power und Energy Werten.
    Ignoriert 'nan' und fehlende Werte.
    """
    valid_power = []
    valid_energy = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Regex Suche nach Zahlen hinter "Power :" und "Energy/Inf :"
        # Findet z.B. "Power : 146.5858 mW" oder "Power : nan mW"
        power_matches = re.findall(r"Power\s*:\s*([0-9\w\.-]+)\s*mW", content)
        energy_matches = re.findall(r"Energy/Inf\s*:\s*([0-9\w\.-]+)\s*uJ/inf", content)
        
        # Power Werte konvertieren
        for val in power_matches:
            try:
                f_val = float(val)
                if not np.isnan(f_val):
                    valid_power.append(f_val)
            except ValueError:
                continue # War wohl 'nan' oder Text
                
        # Energie Werte konvertieren
        for val in energy_matches:
            try:
                f_val = float(val)
                if not np.isnan(f_val):
                    valid_energy.append(f_val)
            except ValueError:
                continue

    except Exception as e:
        print(f"Warnung: Konnte {filepath} nicht lesen: {e}")
        
    return valid_power, valid_energy

def extract_metrics():
    results_list = []

    print(f"Scanne Ordner '{DATA_DIR}'...")

    for root, dirs, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        parts = folder_name.split('_')

        # Bestimme Modus (p oder e)
        mode = None
        if 'p' in parts:
            mode = 'p'
            mode_idx = parts.index('p')
        elif 'e' in parts:
            mode = 'e'
            mode_idx = parts.index('e')
        else:
            continue # Ordner ohne p/e Label ignorieren

        # Bestimme MCU und Modell
        try:
            model = parts[mode_idx - 1]
            mcu = "_".join(parts[:mode_idx - 1])
        except IndexError:
            continue

        # --- FALL 1: PERFORMANCE (JSON) ---
        if mode == 'p':
            if "results.json" in files:
                filepath = os.path.join(root, "results.json")
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list): iterations = data
                    else: iterations = [data]

                    valid_tp = []
                    for run in iterations:
                        infer = run.get('infer', {})
                        val = infer.get('throughput')
                        if val is not None and not np.isnan(val):
                            valid_tp.append(val)
                    
                    if valid_tp:
                        results_list.append({
                            "MCU": mcu, "Model": model,
                            "Metric": "Throughput (FPS)",
                            "Value": np.mean(valid_tp)
                        })
                except Exception as e:
                    print(f"Fehler in {filepath}: {e}")

        # --- FALL 2: ENERGIE & POWER (TXT) ---
        elif mode == 'e':
            # Hier nutzen wir jetzt results.txt statt json
            if "results.txt" in files:
                filepath = os.path.join(root, "results.txt")
                powers, energies = parse_energy_txt(filepath)
                
                # Wenn wir gültige Werte gefunden haben, speichern wir den Durchschnitt
                if energies:
                    results_list.append({
                        "MCU": mcu, "Model": model,
                        "Metric": "Energy (uJ)",
                        "Value": np.mean(energies)
                    })
                if powers:
                    results_list.append({
                        "MCU": mcu, "Model": model,
                        "Metric": "Power (mW)",
                        "Value": np.mean(powers)
                    })

    return pd.DataFrame(results_list)

def plot_results(df):
    if df.empty:
        print("Keine gültigen Daten gefunden! (Prüfe Ordnerstruktur/Dateinamen)")
        return

    sns.set_theme(style="whitegrid")
    
    # Welche Metriken haben wir gefunden?
    available_metrics = df["Metric"].unique()
    
    for metric in available_metrics:
        subset = df[df["Metric"] == metric]
        
        plt.figure(figsize=(10, 6))
        
        # Erstelle den Plot
        # errorbar=None entfernt den schwarzen Strich
        sns.barplot(
            data=subset,
            x="Model",
            y="Value",
            hue="MCU",
            palette="viridis",
            order=MODEL_ORDER,    # X-Achse Sortierung
            hue_order=MCU_ORDER,  # Legende Sortierung
            errorbar=None         # <--- WICHTIG: Keine Fehlerbalken/Striche
        )
        
        plt.title(f"Vergleich: {metric}")
        plt.ylabel(metric)
        plt.xlabel("Modell")
        
        # Legende nach außen verschieben
        plt.legend(title="MCU", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Dateiname bereinigen (Leerzeichen weg etc.)
        filename = f"{metric.split()[0]}_chart.{SAVE_FORMAT}"
        plt.savefig(filename)
        print(f"Grafik erstellt: {filename}")
        plt.close()

if __name__ == "__main__":
    df = extract_metrics()
    
    if not df.empty:
        print("\nExtrahierte Durchschnittswerte (Vorschau):")
        print(df.groupby(['MCU', 'Model', 'Metric'])['Value'].mean())
        print("-" * 30)
        plot_results(df)
    else:
        print("Es konnten keine Daten extrahiert werden.")