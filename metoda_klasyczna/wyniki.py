import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import math

WATERSHED_MASKS = Path("data/masks_watershed")
ORIGINAL_IMAGES = Path("data/Processed_images_png")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Rzeczywista liczba komórek (ground truth) - ustaw ręcznie lub wczytaj z pliku
GROUND_TRUTH_COUNT = None  # np. 3135 - jeśli znasz, wpisz tutaj

def generate_report(all_stats_list, total_detected, total_gt, elapsed_time, num_images):
    areas = [s["pole(px)"] for s in all_stats_list]
    circularities = [s["kołowatość"] for s in all_stats_list]
    diameters = [s["średnica"] for s in all_stats_list]

    fps = num_images / elapsed_time if elapsed_time > 0 else 0
    avg_detected = total_detected / num_images if num_images > 0 else 0

    if total_gt:
        avg_error = abs(total_detected - total_gt) / num_images
        relative_error = abs(total_detected - total_gt) / total_gt * 100
    else:
        avg_error = None
        relative_error = None

    lines = []
    lines.append("=" * 50)
    lines.append("  RAPORT KLASYCZNY: METRYKI I STATYSTYKI KOMÓREK")
    lines.append("=" * 50)
    lines.append(f"Testowanych obrazów: {num_images}")
    lines.append(f"Zidentyfikowano obiektów łącznie: {total_detected}")
    if total_gt:
        lines.append(f"Całkowita liczba rzeczywistych komórek: {total_gt}")
    lines.append("-" * 50)

    lines.append("")
    lines.append("1. METRYKI ZLICZANIA:")
    lines.append(f"   * Średnia wykrytych/obraz:     {avg_detected:.1f} komórek")
    if avg_error is not None:
        lines.append(f"   * Średni błąd liczenia:        {avg_error:.2f} komórek/obraz")
    if relative_error is not None:
        lines.append(f"   * Względny błąd liczenia:      {relative_error:.2f}%")
    lines.append(f"   * Wydajność przetwarzania:     {fps:.2f} FPS")
    lines.append("-" * 50)

    lines.append("")
    lines.append("2. STATYSTYKI FIZYCZNE KOMÓREK (MORFOLOGIA):")
    lines.append(f"   * Najmniejsza pow.:  {min(areas)} pikseli kwadratowych")
    lines.append(f"   * Największa pow.:   {max(areas)} pikseli kwadratowych")
    lines.append(f"   * Średnia pow.:      {np.mean(areas):.1f} pikseli kwadratowych")
    lines.append(f"   * Mediana pow.:      {np.median(areas):.1f} pikseli kwadratowych")
    lines.append(f"   * Odch. std pow.:    {np.std(areas):.1f} pikseli kwadratowych")
    lines.append("-" * 50)

    lines.append("")
    lines.append("3. STATYSTYKI KOŁOWATOŚCI:")
    lines.append(f"   * Minimalna:   {min(circularities):.3f}")
    lines.append(f"   * Maksymalna:  {max(circularities):.3f}")
    lines.append(f"   * Średnia:     {np.mean(circularities):.3f}")
    lines.append(f"   * Mediana:     {np.median(circularities):.3f}")
    lines.append("-" * 50)

    lines.append("")
    lines.append("4. STATYSTYKI ŚREDNICY:")
    lines.append(f"   * Minimalna:   {min(diameters):.2f} px")
    lines.append(f"   * Maksymalna:  {max(diameters):.2f} px")
    lines.append(f"   * Średnia:     {np.mean(diameters):.2f} px")
    lines.append(f"   * Mediana:     {np.median(diameters):.2f} px")
    lines.append("=" * 50)

    report_text = "\n".join(lines)

    report_path = RESULTS_DIR / "raport_ewaluacji_klasyczny.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nRaport zapisano do: {report_path}")


def analyze_objects():
    mask_files = list(WATERSHED_MASKS.glob("*_watershed_masks.png"))
    all_stats_list = []
    total_detected = 0
    start_time = time.time()

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
            if label_id == 0:
                continue

            mask = np.uint8(labels == label_id)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            area = int(np.sum(mask))
            perimeter = cv2.arcLength(contours[0], True)
            circularity = round((4 * math.pi * area) / (perimeter ** 2), 2) if perimeter > 0 else 0
            diameter = round(math.sqrt(4 * area / math.pi), 2)

            M = cv2.moments(mask)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            all_stats_list.append({
                "nazwa obrazu": orig_name,
                "id_komorki": label_id,
                "pole(px)": area,
                "kołowatość": circularity,
                "średnica": diameter,
                "srodek x": cx,
                "srodek y": cy
            })
            obiekty_na_obrazie += 1

            cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 1)
            cv2.putText(orig_img, str(label_id), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imwrite(str(RESULTS_DIR / f"{mask_path.stem}_final_viz.png"), orig_img)
        total_detected += obiekty_na_obrazie
        print(f"Plik: {orig_name} -> Znaleziono {obiekty_na_obrazie} obiektów.")

    elapsed_time = time.time() - start_time

    if all_stats_list:
        df_all = pd.DataFrame(all_stats_list)
        excel_path = RESULTS_DIR / "zbiorcze_wyniki_analizy.xlsx"
        df_all.to_excel(excel_path, index=False)
        print(f"\nZAKOŃCZONO! Wyniki dla {len(mask_files)} obrazów zapisano do: {excel_path}")

        generate_report(
            all_stats_list=all_stats_list,
            total_detected=total_detected,
            total_gt=GROUND_TRUTH_COUNT,
            elapsed_time=elapsed_time,
            num_images=len(mask_files)
        )
    else:
        print("\nNie znaleziono żadnych obiektów na obrazach.")


if __name__ == "__main__":
    analyze_objects()