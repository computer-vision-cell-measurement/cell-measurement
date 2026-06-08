import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import math # <-- NOWE: Do obliczeń liczby Pi i pierwiastków
import cv2  # <-- NOWE: Do pomiaru obwodu komórek (krawędzi maski)
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CellDataset
from model import get_model_instance_segmentation
from scipy.spatial.distance import pdist, squareform

def collate_fn(batch):
    return tuple(zip(*batch))

def calculate_pixel_metrics(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    dice = 1.0 if pred_mask.sum() + true_mask.sum() == 0 else 2. * intersection / (pred_mask.sum() + true_mask.sum())
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou

def calculate_object_f1_and_counts(pred_masks, true_masks, iou_threshold=0.5):
    pred_count = len(pred_masks)
    true_count = len(true_masks)
    if true_count == 0 and pred_count == 0:
        return 0, 0, 0, pred_count, true_count
    
    tp = 0
    matched_true_indices = []
    for p_mask in pred_masks:
        best_iou = 0
        best_t_idx = -1
        for t_idx, t_mask in enumerate(true_masks):
            if t_idx in matched_true_indices:
                continue
            intersection = np.logical_and(p_mask, t_mask).sum()
            union = np.logical_or(p_mask, t_mask).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_t_idx = t_idx
        if best_iou >= iou_threshold:
            tp += 1
            matched_true_indices.append(best_t_idx)

    fp = pred_count - tp
    fn = true_count - tp
    return tp, fp, fn, pred_count, true_count

def extract_spatial_features(masks):
    if len(masks) == 0:
        return [], []

    areas = []
    centroids = []

    for mask in masks:
        areas.append(mask.sum())
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            centroids.append([x_indices.mean(), y_indices.mean()])

    distances_to_nearest = []
    if len(centroids) > 1:
        dist_matrix = squareform(pdist(centroids))
        np.fill_diagonal(dist_matrix, np.inf)
        distances_to_nearest = dist_matrix.min(axis=1).tolist()

    return areas, distances_to_nearest

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Ewaluacja uruchomiona na: {device}")

    test_dataset = CellDataset(images_dir='dane/test/images', masks_dir='dane/test/masks')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load('mask_rcnn_bbbc039.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval() 

    total_dice, total_iou = 0.0, 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    total_count_error, total_true_cells = 0.0, 0
    
    global_areas = []
    global_distances = []
    
    # --- NOWE: Dodano nagłówki Kolowatosc i Srednica ---
    csv_data = [["Nazwa_Obrazu", "ID_Komorki", "Pole_Powierzchni_px", "Srodek_X", "Srodek_Y", "Kolowatosc_0_1", "Srednica_Ekwiwalentna_px"]]
    global_img_idx = 0

    print("\nRozpoczynam testowanie na zbiorze walidacyjnym...")
    start_time = time.time()
    
    with torch.no_grad():
        loop = tqdm(test_loader, leave=True, desc="Ewaluacja")
        
        for images, targets in loop:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i in range(len(images)):
                image_name = test_dataset.imgs[global_img_idx]
                global_img_idx += 1
                
                true_masks = targets[i]['masks'].cpu().numpy()
                true_global_mask = np.any(true_masks, axis=0) 
                
                out_scores = outputs[i]['scores'].cpu().numpy()
                out_masks = outputs[i]['masks'].cpu().numpy()
                
                valid_indices = out_scores > 0.5
                pred_masks_soft = out_masks[valid_indices, 0]
                pred_masks_bin = pred_masks_soft > 0.5
                pred_global_mask = np.any(pred_masks_bin, axis=0) if len(pred_masks_bin) > 0 else np.zeros_like(true_global_mask)

                # Metryki podstawowe
                dice, iou = calculate_pixel_metrics(pred_global_mask, true_global_mask)
                total_dice += dice
                total_iou += iou
                
                tp, fp, fn, p_count, t_count = calculate_object_f1_and_counts(pred_masks_bin, true_masks)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_count_error += abs(p_count - t_count)
                total_true_cells += t_count
                
                # Ekstrakcja danych globalnych (do konsoli)
                areas, distances = extract_spatial_features(pred_masks_bin)
                global_areas.extend(areas)
                global_distances.extend(distances)
                
                # --- Zbieranie szczegółowych danych do CSV (Wzbogacone) ---
                for cell_id, mask in enumerate(pred_masks_bin):
                    area = mask.sum()
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        cx, cy = x_indices.mean(), y_indices.mean()
                        
                        # 1. Średnica ekwiwalentna
                        diameter = 2 * math.sqrt(area / math.pi)
                        
                        # 2. Kołowatość
                        mask_uint8 = mask.astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        circularity = 0.0
                        if len(contours) > 0:
                            # Szukamy największego konturu (ignorujemy śmieciowe artefakty)
                            c = max(contours, key=cv2.contourArea)
                            perimeter = cv2.arcLength(c, True)
                            if perimeter > 0:
                                circularity = (4 * math.pi * area) / (perimeter ** 2)
                                # Kwadratowe piksele czasem dają wynik lekko > 1.0, obcinamy do perfekcyjnego koła
                                circularity = min(circularity, 1.0)
                        
                        csv_data.append([
                            image_name, 
                            cell_id + 1, 
                            area, 
                            round(cx, 2), 
                            round(cy, 2), 
                            round(circularity, 4), 
                            round(diameter, 2)
                        ])

    end_time = time.time()
    num_images = len(test_dataset)
    
    if len(global_areas) > 0:
        min_area, max_area = np.min(global_areas), np.max(global_areas)
        mean_area, median_area = np.mean(global_areas), np.median(global_areas)
    else:
        min_area = max_area = mean_area = median_area = 0

    mean_neighbor_dist = np.mean(global_distances) if len(global_distances) > 0 else 0

    mean_dice = total_dice / num_images
    mean_iou = total_iou / num_images
    mean_count_error = total_count_error / num_images
    relative_count_error = (total_count_error / total_true_cells) * 100 if total_true_cells > 0 else 0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fps = num_images / (end_time - start_time)

    raport = f"""==================================================
      RAPORT AI: METRYKI I STATYSTYKI KOMÓREK
==================================================
Testowanych obrazów: {num_images}
Zidentyfikowano obiektów łącznie: {len(global_areas)}
Całkowita liczba rzeczywistych komórek: {total_true_cells}
--------------------------------------------------
 1. METRYKI SEGMENTACJI (JAKOŚĆ SIECI):
   * Pixel Dice:       {mean_dice:.4f}
   * Pixel IoU:        {mean_iou:.4f}
   * Object F1:        {f1_score:.4f}
   * Precision:        {precision:.4f}
   * Recall:           {recall:.4f}
   * Średni błąd liczenia:   {mean_count_error:.2f} komórek/obraz
   * Względny błąd liczenia: {relative_count_error:.2f}%
   * Wydajność GPU:          {fps:.2f} FPS
--------------------------------------------------
 2. STATYSTYKI FIZYCZNE KOMÓREK (MORFOLOGIA):
   * Najmniejsza pow.: {min_area:.0f} pikseli kwadratowych
   * Największa pow.:  {max_area:.0f} pikseli kwadratowych
   * Średnia pow.:     {mean_area:.1f} pikseli kwadratowych
   * Mediana pow.:     {median_area:.1f} pikseli kwadratowych
   * Średni dystans do najbliższego sąsiada: {mean_neighbor_dist:.2f} pikseli
=================================================="""

    print("\n" + raport)

    with open("raport_ewaluacji.txt", "w", encoding="utf-8") as file:
        file.write(raport)
    print("=> Zapisano raport tekstowy jako 'raport_ewaluacji.txt'")

    with open("wymiary_komorek.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(csv_data)
    print("=> Zapisano szczegółową bazę danych do excela jako 'wymiary_komorek.csv'")

    metryki_nazwy = ['Pixel Dice', 'Pixel IoU', 'Object F1', 'Precision', 'Recall']
    metryki_wartosci = [mean_dice, mean_iou, f1_score, precision, recall]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metryki_nazwy, metryki_wartosci, color=['#2ca02c', '#98df8a', '#1f77b4', '#aec7e8', '#ff7f0e'])
    
    plt.ylim(0, 1.1)
    plt.title('Zaawansowane metryki ewaluacji modelu Mask R-CNN', fontsize=14, pad=20)
    plt.ylabel('Wartość metryki (0.0 - 1.0)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f"{height:.4f}", ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('wykres_ewaluacji_zaawansowany.png', dpi=300)
    plt.close()
    print("=> Zapisano nowy wykres podsumowujący jako 'wykres_ewaluacji_zaawansowany.png'\n")

if __name__ == '__main__':
    main()