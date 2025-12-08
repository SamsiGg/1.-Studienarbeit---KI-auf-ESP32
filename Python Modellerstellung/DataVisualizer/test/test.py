import json

def drucke_struktur(element, ebene=0):
    """
    Gibt die Struktur (Keys) eines JSON-Objekts rekursiv aus.
    """
    einrueckung = "  " * ebene
    
    if isinstance(element, dict):
        for key, value in element.items():
            print(f"{einrueckung}- {key}")
            # Rekursiv weitermachen
            drucke_struktur(value, ebene + 1)
            
    elif isinstance(element, list):
        if len(element) > 0:
            print(f"{einrueckung}[ Liste mit {len(element)} Einträgen ]")
            # Wir schauen uns nur das erste Element der Liste an, 
            # um die Struktur zu zeigen (angenommen alle Items sind ähnlich)
            print(f"{einrueckung}  (Struktur eines Elements:)")
            drucke_struktur(element[0], ebene + 2)
        else:
            print(f"{einrueckung}[] (Leere Liste)")

# --- Hauptprogramm ---
def main():
    dateiname = 'results.json'

    try:
        with open(dateiname, 'r', encoding='utf-8') as f:
            daten = json.load(f)
        
        print(f"Struktur von '{dateiname}':")
        drucke_struktur(daten)

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()