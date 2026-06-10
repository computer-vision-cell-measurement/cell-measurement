import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CellDataset
from model import get_model_instance_segmentation


def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Trening uruchomiony na: {device}")

    dataset = CellDataset('metoda_ai/dane/train/images', 'metoda_ai/dane/train/masks')
    
    data_loader = DataLoader(
        dataset, 
        batch_size=2,          
        shuffle=True, 
        num_workers=0,         
        collate_fn=collate_fn
    )


    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 15

    print("\nRozpoczynam trening...")
    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        
        epoch_loss = 0
        valid_batches = 0
        
        loop = tqdm(data_loader, leave=True, desc=f"Epoka {epoch+1}/{num_epochs}")
        
        for i, (images, targets) in enumerate(loop):
            
            valid_images = []
            valid_targets = []
            for img, tgt in zip(images, targets):
                if len(tgt['boxes']) > 0:
                    valid_images.append(img.to(device))
                    valid_targets.append({k: v.to(device) for k, v in tgt.items()})

            if len(valid_images) == 0:
                continue


            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if valid_batches == 0:
                print("\nSkładowe straty:", {k: round(v.item(), 3) for k, v in loss_dict.items()})
            

            optimizer.zero_grad()
            losses.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            # ----------------------------------------
            
            epoch_loss += losses.item()
            valid_batches += 1
            loop.set_postfix(strata=losses.item())


        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
        print(f"Koniec epoki {epoch+1}. Średnia strata: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), 'mask_rcnn_bbbc039.pth')
    print("Trening zakończony! Model zapisano jako 'mask_rcnn_bbbc039.pth'")

if __name__ == '__main__':
    main()