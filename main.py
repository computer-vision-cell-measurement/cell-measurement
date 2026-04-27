import cv2
import numpy as np
import os
import glob

def preprocess_images(input_path, output_path):
    # Tworzenie folderu wyjściowego, jeśli nie istnieje
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Szukamy plików .tif
    image_files = glob.glob(os.path.join(input_path, "*.tif"))

    if not image_files:
        print("Nie znaleziono plików .tif w podanym folderze!")
        return

    # Inicjalizacja CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img_path in image_files:
        img_16bit = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            print(f"Błąd podczas wczytywania: {img_path}")
            continue

        # Normalizacja z 16-bit do 8-bit
        img_8bit = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX)
        img_8bit = img_8bit.astype('uint8')

        # Zastosowanie CLAHE
        final_img = clahe.apply(img_8bit)

        # Zmiana rozszerzenia na .png
        base_name = os.path.basename(img_path) # pobiera 'obraz.tif'
        file_name_no_ext = os.path.splitext(base_name)[0] # wycina 'obraz'
        new_filename = f"{file_name_no_ext}.png" # tworzy 'obraz.png'

        # Zapis pliku
        cv2.imwrite(os.path.join(output_path, new_filename), final_img)
        
    print(f"Przetworzono pomyślnie {len(image_files)} obrazów i zapisano jako .png.")

# Ścieżki (pamiętaj, żeby były poprawne na Twoim kompie)
INPUT_DIR = 'C:/Users/User/Desktop/Wizja/Images'
OUTPUT_DIR = 'data/Processed_images_png'

if __name__ == "__main__":
    preprocess_images(INPUT_DIR, OUTPUT_DIR)