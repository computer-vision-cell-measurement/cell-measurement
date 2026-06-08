import pandas as pd
from pathlib import Path

df_klasyczny = pd.read_excel("data/results/zbiorcze_wyniki_analizy.xlsx")
df_yolo = pd.read_excel("data/results/wymiary_komorek_excel.xlsx")  # Upewnij się, że nazwa pliku jest poprawna

KOLUMNA_OBRAZ_YOLO = "Nazwa_Obrazu"
KOLUMNA_POLE_YOLO = "Pole_Powierzchni_px"
KOLUMNA_OBRAZ_KLASYCZNY = "nazwa obrazu"

df_klasyczny['czysta_nazwa'] = df_klasyczny[KOLUMNA_OBRAZ_KLASYCZNY].apply(lambda x: Path(str(x)).stem)
df_yolo['czysta_nazwa'] = df_yolo[KOLUMNA_OBRAZ_YOLO].apply(lambda x: Path(str(x)).stem)

wspolne_obrazy = df_yolo['czysta_nazwa'].unique()

df_klasyczny_filtrowany = df_klasyczny[df_klasyczny['czysta_nazwa'].isin(wspolne_obrazy)]

print("=" * 50)
print("   PORÓWNANIE WYNIKÓW (TYLKO DLA WSPÓLNYCH 40 ZDJĘĆ)   ")
print("=" * 50)
print(f"Liczba analizowanych zdjęć: {len(wspolne_obrazy)}")
print("-" * 50)

print(f"Liczba wykrytych obiektów (Watershed): {len(df_klasyczny_filtrowany)}")
print(f"Liczba wykrytych obiektów (Mask R-CNN):    {len(df_yolo)}")
print("-" * 50)

print(f"Średnie pole (Watershed): {df_klasyczny_filtrowany['pole(px)'].mean():.2f} px")
print(f"Średnie pole (Mask R-CNN):    {df_yolo[KOLUMNA_POLE_YOLO].mean():.2f} px")
print("-" * 50)

print(f"Mediana pola (Watershed): {df_klasyczny_filtrowany['pole(px)'].median():.2f} px")
print(f"Mediana pola (Mask R-CNN):    {df_yolo[KOLUMNA_POLE_YOLO].median():.2f} px")
print("=" * 50)