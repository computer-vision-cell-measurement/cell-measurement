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


def save_debug_image(image, step_name, is_first_image, debug_dir):
    if not is_first_image:
        return
        
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        
    filename = f"{step_name}.png"
    cv2.imwrite(os.path.join(debug_dir, filename), image)


def segment_images(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_files = glob.glob(os.path.join(input_path, "*.png"))

    if not image_files:
        print("Nie znaleziono plików .png w podanym folderze do segmentacji!")
        return

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)

    for i, img_path in enumerate(image_files):
        is_first = (i == 0)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Błąd podczas wczytywania: {img_path}")
            continue
            
        save_debug_image(image, "01_oryginal_skala_szarosci", is_first, 'segmentation-and-filtration-test')

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        save_debug_image(blurred, "02_rozmycie_gaussa", is_first, 'segmentation-and-filtration-test')
        
        ret, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        milder_threshold = ret * 0.6
        _, binary_mask = cv2.threshold(blurred, milder_threshold, 255, cv2.THRESH_BINARY)
        save_debug_image(binary_mask, "03_binaryzacja_otsu", is_first, 'segmentation-and-filtration-test')

        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)
        save_debug_image(binary_mask, "04_otwarcie_morfologiczne", is_first, 'segmentation-and-filtration-test')
        
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)
        save_debug_image(binary_mask, "05_zamkniecie_morfologiczne", is_first, 'segmentation-and-filtration-test')

        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_path, filename), binary_mask)

    print(f"Segmentacja Otsu zakończona: {len(image_files)} obrazów zapisano do {output_path}.")


INPUT_DIR = 'images'
OUTPUT_DIR = 'data/Processed_images_png'
SEGMENTED_DIR = 'data/otsu_segmented_masks'


if __name__ == "__main__":
    preprocess_images(INPUT_DIR, OUTPUT_DIR)
    segment_images(OUTPUT_DIR, SEGMENTED_DIR)