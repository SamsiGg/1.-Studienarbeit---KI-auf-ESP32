import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURATION ---
DATA_DIR = "Data"
OUTPUT_DIR = "."
SAVE_FORMAT = "png"

# --- PREISE (in Euro) ---
PRICES = {
    "esp32": 6.00,
    "s3": 20.00,
    "teensy": 25.00,
    "giga": 75.00
}

# Reihenfolge für Diagramme
MODEL_ORDER = ["kws", "ic", "vww"]
MCU_ORDER   = ["esp32", "s3", "giga", "teensy"]

def parse_energy_txt(filepath):
    """Liest Energie aus results.txt (ignoriert nan)"""
    valid_energy = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Suche nach Zahlen hinter Energy/Inf
        matches = re.findall(r"Energy/Inf\s*:\s*([0-9\w\.-]+)\s*uJ/inf", content)
        for val in matches:
            try:
                f_val = float(val)
                if not np.isnan(f_val):
                    valid_energy.append(f_val)
            except ValueError:
                continue
    except:
        pass
    return valid_energy

def extract_data_with_price():
    results = []
    print(f"Scanne {DATA_DIR}...")

    for root, dirs, files in os.walk(DATA_DIR):
        folder = os.path.basename(root)
        parts = folder.split('_')

        # Modus bestimmen (p oder e)
        if 'p' in parts:
            mode = 'p'; idx = parts.index('p')
        elif 'e' in parts:
            mode = 'e'; idx = parts.index('e')
        else:
            continue

        # MCU & Modell bestimmen
        try:
            model = parts[idx-1]
            mcu = "_".join(parts[:idx-1])
        except: continue

        # Preis abrufen (Fallback 0, falls MCU Name abweicht)
        price = PRICES.get(mcu, 0)
        if price == 0:
            print(f"Warnung: Kein Preis für '{mcu}' gefunden.")

        # --- Daten extrahieren ---
        
        # 1. Throughput (JSON)
        if mode == 'p' and "results.json" in files:
            try:
                with open(os.path.join(root, "results.json"), 'r') as f:
                    data = json.load(f)
                if isinstance(data, list): runs = data
                else: runs = [data]
                
                valid_tp = []
                for run in runs:
                    val = run.get('infer', {}).get('throughput')
                    if val is not None and not np.isnan(val):
                        valid_tp.append(val)
                
                if valid_tp:
                    results.append({
                        "MCU": mcu, "Model": model, "Metric": "Throughput", 
                        "Value": np.mean(valid_tp), "Price": price
                    })
            except Exception as e:
                print(f"Error {mcu} {model}: {e}")

        # 2. Energy (TXT)
        elif mode == 'e' and "results.txt" in files:
            vals = parse_energy_txt(os.path.join(root, "results.txt"))
            if vals:
                results.append({
                    "MCU": mcu, "Model": model, "Metric": "Energy", 
                    "Value": np.mean(vals), "Price": price
                })

    return pd.DataFrame(results)

def plot_price_performance(df):
    if df.empty:
        print("Keine Daten!")
        return

    sns.set_theme(style="whitegrid")

    # 1. Scatter Plot: Preis vs. Throughput
    # Zeigt: Bekommt man für mehr Geld auch mehr Leistung?
    df_p = df[df["Metric"] == "Throughput"]
    if not df_p.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_p, x="Price", y="Value", 
            hue="MCU", style="Model", 
            palette="viridis", s=150, # s=Größe der Punkte
            hue_order=MCU_ORDER
        )
        plt.title("Preis vs. Performance (Throughput)")
        plt.xlabel("Preis (€)")
        plt.ylabel("Throughput (FPS)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"Scatter_Price_Throughput.{SAVE_FORMAT}")
        print("Erstellt: Scatter_Price_Throughput.png")

        # 2. Balkendiagramm: FPS pro Euro (Der Preis-Leistungs-Sieger)
        df_p = df_p.copy()
        df_p["FPS_per_Euro"] = df_p["Value"] / df_p["Price"]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_p, x="Model", y="FPS_per_Euro", 
            hue="MCU", palette="viridis",
            order=MODEL_ORDER, hue_order=MCU_ORDER,
            errorbar=None
        )
        plt.title("Preis-Leistung: FPS pro Euro")
        plt.ylabel("FPS / € (Höher ist besser)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"Bar_FPS_per_Euro.{SAVE_FORMAT}")
        print("Erstellt: Bar_FPS_per_Euro.png")

    # 3. Scatter Plot: Preis vs. Energie
    df_e = df[df["Metric"] == "Energy"]
    if not df_e.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_e, x="Price", y="Value", 
            hue="MCU", style="Model", 
            palette="viridis", s=150,
            hue_order=MCU_ORDER
        )
        plt.title("Preis vs. Energieverbrauch")
        plt.xlabel("Preis (€)")
        plt.ylabel("Energie pro Inferenz (uJ)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"Scatter_Price_Energy.{SAVE_FORMAT}")
        print("Erstellt: Scatter_Price_Energy.png")

if __name__ == "__main__":
    df = extract_data_with_price()
    plot_price_performance(df)