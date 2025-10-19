from PIL import Image

def image_to_c_array(image_path, array_name="my_digit"):
    """
    Konvertiert ein 28x28 Bild in ein normalisiertes C-Array.

    Args:
        image_path (str): Der Pfad zum PNG-Bild.
        array_name (str): Der gewünschte Name für die C-Array-Variable.
    """
    try:
        # Bild öffnen und in Graustufen umwandeln
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{image_path}' wurde nicht gefunden.")
        return

    # Sicherstellen, dass das Bild die richtige Größe hat
    if img.size != (28, 28):
        print(f"Warnung: Das Bild hat die Größe {img.size}, "
              "es wird auf 28x28 skaliert.")
        img = img.resize((28, 28))

    # Pixeldaten auslesen (Werte von 0-255)
    pixels = list(img.getdata())

    # Normalisieren (Werte zwischen 0.0 und 1.0)
    normalized_pixels = [p / 255.0 for p in pixels]

    # C-Array-String formatieren
    c_array = f"float {array_name}[{len(normalized_pixels)}] = {{\n    "
    
    # Füge jeden Pixelwert zum String hinzu, mit Zeilenumbruch alle 14 Werte
    for i, p in enumerate(normalized_pixels):
        c_array += f"{p:.6f}f, "
        if (i + 1) % 14 == 0:
            c_array += "\n    "
            
    # Entferne das letzte Komma und Leerzeichen und schließe das Array
    c_array = c_array.strip()[:-1] + "\n};\n"

    print("Hier ist dein C-Array (kopieren und einfügen):\n")
    print("--------------------------------------------------")
    print(c_array)
    print("--------------------------------------------------")


# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    # Gib hier den Namen deiner Bilddatei an
    IMAGE_FILE = "sample7.png"
    
    # Gib hier den gewünschten Variablennamen für dein C-Array an
    ARRAY_VARIABLE_NAME = "sample_digit_7"
    
    image_to_c_array(IMAGE_FILE, ARRAY_VARIABLE_NAME)