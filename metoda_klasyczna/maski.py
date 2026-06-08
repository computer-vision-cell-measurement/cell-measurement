import cv2
import numpy as np
import os
import glob


INPUT_MASKS = "masks"
OUTPUT_MASKS = "data/processed_masks"

os.makedirs(OUTPUT_MASKS, exist_ok=True)

def process_masks():
    mask_files = glob.glob(os.path.join(INPUT_MASKS, "*.*"))

    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None: continue
        
        binary_mask = np.where(mask > 0, 255, 0).astype('uint8')

        filename = os.path.basename(mask_path)
        cv2.imwrite(os.path.join(OUTPUT_MASKS, filename), binary_mask)

    print(f"Maski gotowe w: {OUTPUT_MASKS}")

if __name__ == "__main__":
    process_masks()