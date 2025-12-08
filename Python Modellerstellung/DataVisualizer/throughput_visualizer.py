import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguration
DATA_DIR = "Data"
OUTPUT_DIR = "." # Oder "Results", wenn der Ordner existiert

def extract_metrics():
    results_list = []

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file == "results.json":
                folder_name = os.path.basename(root)
                parts = folder_name.split('_')

                # Nur 'p' Modus relevant
                if 'p' in parts:
                    mode_idx = parts.index('p')
                else:
                    continue

                try:
                    # Extrahiere Model und MCU basierend auf der Position von 'p'
                    model = parts[mode_idx - 1]
                    mcu = "_".join(parts[:mode_idx - 1])
                except IndexError:
                    print(f"Konnte Ordner nicht parsen: {folder_name}")
                    continue

                # Datei laden
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Fehler beim Lesen von {filepath}: {e}")
                    continue

                # Daten sammeln
                valid_values = []
                
                # Handling für Listen oder einzelne Objekte
                if isinstance(data, list):
                    iterations = data
                else:
                    iterations = [data]

                for run in iterations:
                    infer = run.get('infer', {})
                    if not infer: continue

                    # Nur Throughput
                    val = infer.get('throughput')
                    if val is not None and not np.isnan(val):
                        valid_values.append(val)
                
                if valid_values:
                    results_list.append({
                        "MCU": mcu, 
                        "Model": model, 
                        "Metric": "Throughput (FPS)", 
                        "Value": np.mean(valid_values)
                    })

    return pd.DataFrame(results_list)

def plot_results(df):
    if df.empty:
        print("Keine Daten für Performance gefunden!")
        return

    sns.set_theme(style="whitegrid")

    # --- HIER IST DIE REIHENFOLGE DEFINIERT ---
    model_order = ["kws", "ic", "vww"]
    mcu_order = ["esp32", "s3", "giga", "teensy"]

    plt.figure(figsize=(10, 6))
    
    # Barplot mit erzwungener Reihenfolge (order & hue_order)
    sns.barplot(
        data=df, 
        x="Model", 
        y="Value", 
        hue="MCU", 
        palette="viridis",
        order=model_order,     # Sortierung X-Achse
        hue_order=mcu_order    # Sortierung Legende/Balken
    )
    
    plt.title("Performance Vergleich: Throughput")
    plt.ylabel("Throughput (FPS)")
    plt.xlabel("Modell")
    plt.legend(title="MCU", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = "Throughput_chart.png"
    plt.savefig(filename)
    print(f"Grafik gespeichert: {filename}")

if __name__ == "__main__":
    print("Starte Extraktion...")
    df = extract_metrics()
    print("Daten extrahiert.")
    plot_results(df)