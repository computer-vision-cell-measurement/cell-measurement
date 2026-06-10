import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import torchvision
from PIL import Image

class CellDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(images_dir)))
        self.masks = list(sorted(os.listdir(masks_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.imgs[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        # Wczytanie obrazu głównego przez PIL (wymuszenie 3 kanałów RGB)
        img = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        
        # Wczytanie maski przez PIL i konwersja na tablicę Numpy
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)



        if len(mask_array.shape) == 3 and mask_array.shape[2] == 4:
            mask_array = mask_array[:, :, :3]

        
        if len(mask_array.shape) == 3:
            mask_2d = mask_array[:, :, 0]
        else:
            mask_2d = mask_array

        unique_vals = np.unique(mask_2d)


        
        if len(unique_vals) == 1 and unique_vals[0] == 0:
             labels = mask_2d


        elif np.max(unique_vals) == 255 or len(unique_vals) < 10:

            binary_mask = (mask_2d > 0).astype(np.uint8) * 255
            num_labels, labels = cv2.connectedComponents(binary_mask)
        else:

            labels = mask_2d
        
  
        obj_ids = np.unique(labels)
        obj_ids = obj_ids[obj_ids > 0]

        boxes = []
        valid_masks = []
        
        for i in obj_ids:

            pos = np.where(labels == i)
            if pos[0].size > 0 and pos[1].size > 0:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                

                if xmax > xmin and ymax > ymin:
                    area = (xmax - xmin) * (ymax - ymin)
                    
                    # Filtr szumu
                    if area > 30: 
                        boxes.append([xmin, ymin, xmax, ymax])
                        valid_masks.append(labels == i)

        # Budowanie Tensorów PyTorch
        if len(boxes) == 0:
            # Puste tensory, jeśli zdjęcie jest kompletnie puste
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, mask_2d.shape[0], mask_2d.shape[1]), dtype=torch.uint8)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            # Wypełnione tensory
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)
            masks_tensor = torch.as_tensor(np.array(valid_masks, dtype=np.uint8), dtype=torch.uint8)
            area_tensor = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": image_id,
            "area": area_tensor,
            "iscrowd": iscrowd_tensor
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)