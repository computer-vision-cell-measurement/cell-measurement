import cv2
import numpy as np
from dataset import CellDataset

def main():
    print("Ładowanie pierwszego zdjęcia ze zbioru treningowego...")

    dataset = CellDataset('metoda_ai/dane/train/images', 'metoda_ai/dane/train/masks')
    img_tensor, target = dataset[0]

    img = img_tensor.permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = target['boxes'].numpy()
    print(f"\n=> Program znalazł {len(boxes)} ważnych komórek na tym zdjęciu.")

    for box in boxes:
        xmin, ymin, xmax, ymax = box.astype(int)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    cv2.imwrite("diagnostyka_ramek.png", img)
    print("=> Zapisano zdjęcie 'diagnostyka_ramek.png' w folderze projektu.")

if __name__ == "__main__":
    main()