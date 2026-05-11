"""
Rozdziela stykające się i nakładające się komórki/ziarna
przy użyciu transformaty odległościowej (distance transform)
i algorytmu Watershed.

Wejście : data/otsu_segmented_masks/         (obrazy binarne PNG/TIFF)
Wyjście : data/masks_watershed/              (maski etykiet   watershed_masks.png)
          data/visualizations_watershed/     (kolorowe wizualizacje watershed_colored.png)
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

#Ścieżki
INPUT_DIR        = Path("data/otsu_segmented_masks")
OUTPUT_MASK_DIR  = Path("data/masks_watershed")
OUTPUT_COLOR_DIR = Path("data/visualizations_watershed") 

# Tworzenie folderów, jeśli nie istnieją
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_COLOR_DIR.mkdir(parents=True, exist_ok=True)

#Parametry 
MIN_DISTANCE   = 20    # jądra U2OS ~30-50px średnicy → promień ~17px
FOOTPRINT_SIZE = 7     # większe okno = mniej fałszywych szczytów
KERNEL_OPEN    = 5     # mocniejsze czyszczenie artefaktów po Otsu
THRESHOLD_REL  = 0.4   # ignoruj szczyty DT < 40% globalnego max

#Opcje wizualizacji  
DRAW_BORDERS      = True   # rysuj białe granice między komórkami
BORDER_THICKNESS  = 1      # grubość granicy [px]

EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


#Kolory

def make_colormap_hsv(n: int) -> np.ndarray:
    
    colors = np.zeros((n + 1, 3), dtype=np.uint8)  
    for i in range(1, n + 1):
        hue = int(((i - 1) / n) * 179)             
        sat = 220
        val = 230
        hsv_px = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
        colors[i] = bgr
    return colors


def labels_to_color(labels: np.ndarray) -> np.ndarray:
    """Zamienia mapę etykiet (uint16) na obraz BGR."""
    n = int(labels.max())
    if n == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)

    colors = make_colormap_hsv(n)
    return colors[labels]


def draw_borders(color_img: np.ndarray, labels: np.ndarray,
                  thickness: int = 1) -> np.ndarray:
    """Rysuje białe kontury na granicach między komórkami."""
    out = color_img.copy()
    n = int(labels.max())
    for label_id in range(1, n + 1):
        mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), thickness)
    return out


#Segmentacja 

def process_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Przyjmuje maskę binarną, zwraca mapę etykiet (uint16)"""
    binary = (mask_bin > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_OPEN, KERNEL_OPEN)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = ndi.distance_transform_edt(binary)

    coords = peak_local_max(
        dist,
        min_distance=MIN_DISTANCE,
        threshold_rel=THRESHOLD_REL,
        footprint=np.ones((FOOTPRINT_SIZE, FOOTPRINT_SIZE)),
        labels=binary,
    )
    marker_mask = np.zeros(dist.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)

    labels = watershed(-dist, markers, mask=binary)
    return labels.astype(np.uint16)


#Zapis

def save_label_png(labels: np.ndarray, path: Path) -> None:
    n = labels.max()
    # Jeśli mamy mało obiektów, zapisujemy jako 8-bit
    # Jeśli > 254, musimy zapisujemy jako 16-bit
    out = labels.astype(np.uint8) if n <= 254 else labels.astype(np.uint16)
    cv2.imwrite(str(path), out)


def save_color_png(labels: np.ndarray, path: Path) -> None:
    color_img = labels_to_color(labels)
    if DRAW_BORDERS:
        color_img = draw_borders(color_img, labels, BORDER_THICKNESS)
    cv2.imwrite(str(path), color_img)


#Main

def main() -> None:
    files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in EXTENSIONS]

    if not files:
        print(f"Brak plików w '{INPUT_DIR}'")
        return

    print(f"Znaleziono {len(files)} masek w '{INPUT_DIR}'")
    print(f"Zapis masek do: {OUTPUT_MASK_DIR}")
    print(f"Zapis kolorów do: {OUTPUT_COLOR_DIR}\n")

    for fpath in sorted(files):
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Nie można wczytać: {fpath.name}")
            continue

        labels    = process_mask(img)
        n_objects = int(labels.max())

        stem = fpath.stem
        
        save_label_png(labels, OUTPUT_MASK_DIR / f"{stem}_watershed_masks.png")
        save_color_png(labels, OUTPUT_COLOR_DIR / f"{stem}_watershed_colored.png")

        print(f"{fpath.name:40s}  →  {n_objects:3d} komórek")

    print(f"\n Gotowe!")

if __name__ == "__main__":
    main()