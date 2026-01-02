import os
import json
import re
import pandas as pd
import numpy as np

# --- KONFIGURATION ---
DATA_DIR = "Data"
# Keine Output-Dir nötig, da wir nur printen

# --- PREISE (in Euro) ---
PRICES = {
    "esp32": 10.00,
    "s3": 19.80,
    "teensy": 29.50,
    "giga": 63.80
}

def parse_energy_txt(filepath):
    """Liest Energie aus results.txt (ignoriert nan)"""
    valid_energy = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
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

def extract_data():
    results = []
    print(f"--- Scanne Verzeichnis '{DATA_DIR}' ---\n")

    for root, dirs, files in os.walk(DATA_DIR):
        folder = os.path.basename(root)
        parts = folder.split('_')

        if 'p' in parts:
            mode = 'p'; idx = parts.index('p')
        elif 'e' in parts:
            mode = 'e'; idx = parts.index('e')
        else:
            continue

        try:
            model = parts[idx-1]
            mcu = "_".join(parts[:idx-1])
        except: continue

        price = PRICES.get(mcu, 0)
        
        # 1. Throughput
        if mode == 'p' and "results.json" in files:
            try:
                with open(os.path.join(root, "results.json"), 'r') as f:
                    data = json.load(f)
                runs = data if isinstance(data, list) else [data]
                valid_tp = [r.get('infer', {}).get('throughput') for r in runs 
                            if r.get('infer', {}).get('throughput') is not None]
                
                if valid_tp:
                    results.append({
                        "MCU": mcu, "Model": model, "Metric": "Throughput", 
                        "Value": np.mean(valid_tp), "Price": price
                    })
            except: pass

        # 2. Energy
        elif mode == 'e' and "results.txt" in files:
            vals = parse_energy_txt(os.path.join(root, "results.txt"))
            if vals:
                results.append({
                    "MCU": mcu, "Model": model, "Metric": "Energy", 
                    "Value": np.mean(vals), "Price": price
                })

    return pd.DataFrame(results)

def print_tikz_ready_data(df):
    if df.empty:
        print("Keine Daten gefunden!")
        return

    # Pivotisieren: Wir wollen Spalten für Throughput und Energy nebeneinander
    # Index: MCU, Model, Price. Spalten: Metric (Throughput, Energy)
    df_pivot = df.pivot_table(
        index=['MCU', 'Model', 'Price'], 
        columns='Metric', 
        values='Value'
    ).reset_index()

    # Spaltennamen bereinigen (falls Metriken fehlen, füllen wir mit NaN)
    if 'Throughput' not in df_pivot.columns: df_pivot['Throughput'] = np.nan
    if 'Energy' not in df_pivot.columns: df_pivot['Energy'] = np.nan

    # Berechnete Metrik: FPS pro Euro
    df_pivot['FPS_per_Euro'] = df_pivot['Throughput'] / df_pivot['Price']

    # Sortierung für schöne Ausgabe
    df_pivot.sort_values(by=['Model', 'MCU'], inplace=True)

    # --- AUSGABE 1: Alles in einer Tabelle ---
    print("-" * 60)
    print("ROHDATEN ÜBERSICHT (Kopierbar für TikZ/pgfplots)")
    print("-" * 60)
    
    # Wir formatieren es so, dass es wie eine .dat Datei aussieht
    header = f"{'MCU':<10} {'Model':<10} {'Price(Eur)':<12} {'FPS':<10} {'Energy(uJ)':<12} {'FPS/Euro':<10}"
    print(header)
    
    for _, row in df_pivot.iterrows():
        mcu = row['MCU']
        model = row['Model']
        price = row['Price']
        fps = row['Throughput']
        nrg = row['Energy']
        fpe = row['FPS_per_Euro']
        
        # Formatierung: .2f für Floats, 'NaN' falls leer
        s_price = f"{price:.2f}"
        s_fps   = f"{fps:.2f}" if not np.isnan(fps) else "nan"
        s_nrg   = f"{nrg:.2f}" if not np.isnan(nrg) else "nan"
        s_fpe   = f"{fpe:.4f}" if not np.isnan(fpe) else "nan"

        print(f"{mcu:<10} {model:<10} {s_price:<12} {s_fps:<10} {s_nrg:<12} {s_fpe:<10}")

    print("\n" + "-" * 60)
    print("NACH MODELL GRUPPIERT (Für separate Plots)")
    print("-" * 60)

    # --- AUSGABE 2: Gruppiert nach Modell (nützlich für Bar Charts) ---
    models = df_pivot['Model'].unique()
    for m in models:
        print(f"\nModel: {m}")
        print(f"MCU         FPS       Energy    FPS/Euro")
        subset = df_pivot[df_pivot['Model'] == m]
        for _, row in subset.iterrows():
            print(f"{row['MCU']:<10} {row['Throughput']:<9.2f} {row['Energy']:<9.2f} {row['FPS_per_Euro']:.4f}")

if __name__ == "__main__":
    df = extract_data()
    print_tikz_ready_data(df)