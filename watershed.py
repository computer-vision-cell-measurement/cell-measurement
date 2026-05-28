import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

INPUT_DIR        = Path("data/otsu_segmented_masks")
OUTPUT_MASK_DIR  = Path("data/masks_watershed")
OUTPUT_COLOR_DIR = Path("data/visualizations_watershed")
OUTPUT_MARKERS_DIR = Path("data/visualizations_markers")
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_COLOR_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MARKERS_DIR.mkdir(parents=True, exist_ok=True)

# Aktualne 
MIN_DISTANCE     = 11  
THRESHOLD_ABS    = 5  
DRAW_BORDERS     = True
EXTENSIONS       = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}

def make_colormap_hsv(n):
    colors = np.zeros((n + 1, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        hue = int(((i - 1) / n) * 179)
        hsv_px = np.uint8([[[hue, 220, 230]]])
        colors[i] = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
    return colors

def labels_to_color(labels):
    n = int(labels.max())
    if n == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    return make_colormap_hsv(n)[labels]

def draw_borders(color_img, labels):
    out = color_img.copy()
    for label_id in range(1, int(labels.max()) + 1):
        mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out

def process_mask(mask_bin):
    binary = (mask_bin > 127).astype(np.uint8)
    dist   = ndi.distance_transform_edt(binary)

    coords = peak_local_max(
        dist,
        min_distance=MIN_DISTANCE,
        threshold_abs=THRESHOLD_ABS, 
        labels=binary,
    )

    marker_mask = np.zeros(dist.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)

    labels = watershed(-dist, markers, mask=binary).astype(np.uint16)
    
    
    return labels, coords

def main():
    files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in EXTENSIONS]
    if not files:
        print(f"Brak plików w '{INPUT_DIR}'")
        return

    for fpath in sorted(files):
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Nie można wczytać: {fpath.name}")
            continue

        labels, coords = process_mask(img)
        stem   = fpath.stem

    
        cv2.imwrite(str(OUTPUT_MASK_DIR  / f"{stem}_watershed_masks.png"), labels.astype(np.uint8))
        color = labels_to_color(labels)
        if DRAW_BORDERS:
            color = draw_borders(color, labels)
        cv2.imwrite(str(OUTPUT_COLOR_DIR / f"{stem}_watershed_colored.png"), color)

        
        # Konwertujemy oryginalną maskę na BGR, aby narysować punkty
        marker_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Funkcja peak_local_max zwraca współrzędne w formacie (Y, X). 
        
        for y, x in coords:
            cv2.circle(marker_vis, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            
        cv2.imwrite(str(OUTPUT_MARKERS_DIR / f"{stem}_markers.png"), marker_vis)

        print(f"{fpath.name:40s}  →  {int(labels.max()):3d} komórek")

if __name__ == "__main__":
    main()