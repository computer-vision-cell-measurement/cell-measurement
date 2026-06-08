import cv2
import numpy as np
import pandas as pd
from pathlib import Path

WATERSHED_MASKS = Path("data/masks_watershed")
ORIGINAL_IMAGES = Path("data/Processed_images_png")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_objects():
    mask_files = list(WATERSHED_MASKS.glob("*_watershed_masks.png"))
    
    all_stats_list = []
    
    for mask_path in mask_files:
        labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        orig_name = mask_path.name.replace("_watershed_masks.png", ".png")
        orig_img = cv2.imread(str(ORIGINAL_IMAGES / orig_name))
        
        if labels is None or orig_img is None:
            print(f"Pominięto {orig_name} - brak maski lub oryginału.")
            continue

        unique_labels = np.unique(labels)
        obiekty_na_obrazie = 0
        
        for label_id in unique_labels:
            if label_id == 0: continue 
            
            mask = np.uint8(labels == label_id)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            area = np.sum(mask) 
            perimeter = cv2.arcLength(contours[0], True)
            
            circularity = 0
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            diameter = np.sqrt(4 * area / np.pi)
            
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue

            all_stats_list.append({
                "nazwa obrazu": orig_name,
                "id_komorki": label_id,
                "pole(px)": area,
                "kołowatość": round(circularity, 2),
                "średnica": round(diameter, 2),
                "srodek x": cx,
                "srodek y": cy
            })
            obiekty_na_obrazie += 1

            cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 1)
            cv2.putText(orig_img, str(label_id), (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imwrite(str(RESULTS_DIR / f"{mask_path.stem}_final_viz.png"), orig_img)
        print(f"Plik: {orig_name} -> Znaleziono {obiekty_na_obrazie} obiektów.")

    if all_stats_list:
        df_all = pd.DataFrame(all_stats_list)
        excel_path = RESULTS_DIR / "zbiorcze_wyniki_analizy.xlsx"
        df_all.to_excel(excel_path, index=False)
        print(f"\nZAKOŃCZONO! Wyniki dla {len(mask_files)} obrazów zapisano do: {excel_path}")
    else:
        print("\nNie znaleziono żadnych obiektów na obrazach.")

if __name__ == "__main__":
    analyze_objects()