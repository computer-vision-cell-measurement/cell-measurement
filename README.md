# Automatyczna Segmentacja i Analiza Morfologiczna Komórek Biologicznych (Zbiór BBBC039)

Projekt realizowany w ramach przedmiotu **Wizja Komputerowa** (Projekt nr 7: *Pomiar wielkości „komórek”/ziaren na obrazie*). Celem projektu jest budowa, walidacja oraz bezpośrednie porównanie dwóch niezależnych systemów segmentacji instancji: potoku klasycznego (opartego na algorytmie wododziałowym) oraz modelu głębokiego uczenia (Mask R-CNN).

## Autorzy
* **Konrad Krupa**(177111)
* **Jan Lochman** (177119)
* **Mateusz Zajdel**(177191)
* **Jakub Kucharski** (177112)
* **Przemysław Krężel** (177110)

**Uczelnia:** Politechnika Rzeszowska im. Ignacego Łukasiewicza  
**Katedra:** Katedra Informatyki i Automatyki  
**Grupa:** AA4  
**Data wykonania:** 10.06.2026

---

## O zbiorze danych
Projekty wykorzystuje referencyjny zbiór danych **BBBC039** pochodzący z repozytorium *Broad Bioimage Benchmark Collection*.
* **Specyfika biologiczna:** Fluorescencyjne obrazy mikroskopowe komórek raka piersi u ludzi (linia komórkowa MCF-7) z wybarwionymi jądrami komórkowymi (barwnik Hoechst 33342).
* **Format:** Surowe obrazy 16-bitowe o wysokiej rozdzielczości tonalnej zapisane w formacie `.tiff` o wymiarach $520 \times 696$ pikseli.
* **Próba testowa:** Ewaluacja porównawcza obu metod została przeprowadzona na zunifikowanym, zamkniętym zbiorze **40 obrazów testowych**, zawierających łącznie **3179 rzeczywistych komórek** oznaczonych przez ekspertów (Ground Truth).

---

## Architektura Systemu

### 1. Potok Klasyczny (Classical Pipeline)
Zaimplementowany w pełni z wykorzystaniem operacji przetwarzania obrazów i morfologii matematycznej:
* Wyrównanie kontrastu metodą **CLAHE**.
* Filtrowanie gausowskie w celu redukcji szumów wysokoczęstotliwościowych.
* Progowanie adaptacyjne metodą **Otsu**.
* Operacje morfologiczne (otwarcie, zamknięcie) usuwające artefakty tła.
* Transformacja odległościowa (**Distance Transform**) do wyznaczenia lokalnych maksimów.
* Algorytm wododziałowy (**Watershed**) sterowany znacznikami do separacji stykających się obiektów.

### 2. Głębokie Uczenie (Deep Learning Pipeline)
* Model **Mask R-CNN** oparty na architekturze szkieletowej **ResNet-50-FPN**.
* Wykorzystanie techniki **Transfer Learningu** (wagi zainicjalizowane na zbiorze COCO).
* Detekcja obiektów (Bounding Boxes) połączona z równoległym generowaniem masek na poziomie pikseli dla każdej instancji jądra komórkowego.

---

## 📂 Struktura projektu

```text
├── data/
│   └── BBBC039/            # Surowe zdjęcia mikroskopowe oraz eksperckie maski (.tiff)
├── outputs/
│   ├── csv_results/        # Wygenerowane pliki CSV z wyliczonymi cechami komórek
│   └── plots/              # Wykresy porównawcze i wizualizacje masek segmentacji
├── src/
│   ├── classical_watershed.py  # Implementacja potoku klasycznego (Otsu + Watershed)
│   ├── mask_rcnn_eval.py       # Skrypt do inferencji i ewaluacji modelu Mask R-CNN
│   └── morphology_features.py  # Moduł obliczający cechy geometryczne obiektów
├── main.py                 # Główny punkt wejścia do aplikacji (CLI)
└── requirements.txt        # Lista zależności projektowych
```

## Instalacja i konfiguracja
 * **1. Sklonuj repozytorium na swój dysk lokalny:**

* **2. Zainstaluj wymagane pakiety i biblioteki za pomocą menedżera pip:**
    pip install -r requirements.txt

Wymagane środowisko: Python 3.8 lub nowszy, PyTorch (z obsługą CUDA dla akceleracji GPU przy Mask R-CNN), OpenCV, Scikit-Image, Pandas, Matplotlib.


## Uruchomienie programu (Interfejs CLI)
Główny skrypt main.py obsługuje parametry wiersza poleceń poprzez flagę --mode, sterującą pracą całego systemu.

* **1. Uruchomienie segmentacji klasycznej (Watershed)**
Przetwarza zbiór testowy potokiem klasycznym, zapisuje mapy kontrolne w przestrzeni HSV z naniesionymi konturami oraz generuje raporty cech fizycznych.
python main.py --mode watershed

* **2. Uruchomienie inferencji AI (Mask R-CNN)Uruchamia ewaluację zoptymalizowanego modelu głębokiego uczenia PyTorch, dokonując predykcji masek obiektów piksel po pikselu.**
python main.py --mode mask_rcnn

* **3. Generowanie analizy porównawczejDokonuje bezpośredniego zestawienia obu potoków, porównując ich błędy zliczania, odchylenia pól powierzchni oraz generując zbiorcze wykresy statystyczne w katalogu outputs/plots/.**
  python main.py --mode compare 

## Wyniki Eksperymentalne i Ewaluacja
 Bezpośrednie starcie obu systemów na próbie testowej (40 obrazów, 3179 komórek Ground Truth) wykazało unikalną charakterystykę zalet i ograniczeń każdej z metod:  
 Metryki Jakości Segmentacji AI (Mask R-CNN):Skuteczność detekcji (Object-level): F1-Score: 0.9520 (Precyzja: 0.9778, Czułość: 0.9276).  
 Precyzja masek (Pixel-level): Współczynnik Dice'a: 0.9394, Indeks Jaccarda (IoU): 0.8875.  
 
## Wnioski z porównania metod (compare):
* **1. Zliczanie obiektów:**
Algorytm klasyczny (Watershed) wykazał się bardzo niskim globalnym błędem względnym zliczania wynoszącym zaledwie 1.23% (efekt kompensacji pominięć i fałszywych detekcji szumu). Model Mask R-CNN wykazał podejście bardziej konserwatywne (błąd niedoszacowania o -178 komórek), kładąc nacisk na maksymalizację precyzji kosztem czułości.

* **2. Dokładność przestrzenna:**
Metoda klasyczna cierpi na problem tzw. poświaty wokół jąder komórkowych, przez co średnia powierzchnia obiektów w potoku klasycznym wyniosła aż 834.38 px² (zawyżenie o ok. 31% względem Ground Truth). Model Mask R-CNN osiągnął średnią 698.0 px², idealnie odzwierciedlając rzeczywiste, biologiczne granice obiektów.  


# Ekstrakcja Danych Analitycznych (Format .csv)
Moduł automatycznej kwantyfikacji generuje dla każdego potoku pliki .csv w outputs/csv_results/. Każdy rekord odpowiada pojedynczej, fizycznie zidentyfikowanej komórce i zawiera następujące parametry geometryczno-topologiczne:  
* **ID:** Unikalny identyfikator instancji oraz klatki źródłowej.**
* **Centroid (X, Y):** Współrzędne środka ciężkości jądra w przestrzeni dwuwymiarowej.**
* **Pole powierzchni (Area):** Całkowita liczba pikseli zajmowana przez komórkę.**  
* **Średnica ekwiwalentna (Equivalent Diameter):** Średnica idealnego okręgu o identycznym polu powierzchni.**
* **Kołowatość (Circularity):** Współczynnik określający regularność i gładkość kształtu struktury, wyliczany według wzoru: $$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$**
* **Dystans do Najbliższego Sąsiada (Nearest Neighbor Distance):** Najkrótszy dystans euklidesowy do najbliższej sąsiadującej komórki.**


