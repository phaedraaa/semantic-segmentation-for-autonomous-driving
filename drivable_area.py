import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================
TRAIN_IMG_DIR = "/bdd100k_images_10k/10k/train" 
TRAIN_MASK_DIR = "/bdd100k_drivable_maps/labels/train" 

# SAVE PATHS
MODELS_DIR = "models_area"
RESULTS_DIR = "results_area"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 360  
IMAGE_WIDTH = 640   

# ==========================================
# DATA TRANSFORMS
# ==========================================
def get_train_transforms():
    return A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

# ==========================================
# PYTORCH DATASET CLASS (WITH FILTER)
# ==========================================
class BDDDrivableDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        all_images = os.listdir(images_dir)
        self.valid_images = []
        
        for img_name in all_images:
            if not img_name.endswith('.jpg'):
                continue
                
            mask_name = img_name.replace('.jpg', '_drivable_id.png')
            mask_path = os.path.join(self.masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.valid_images.append(img_name)
                
        print(f"Dataset Info: Βρέθηκαν {len(self.valid_images)} έγκυρα ζευγάρια εικόνας-μάσκας.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        mask_name = img_name.replace('.jpg', '_drivable_id.png')
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask[mask == 255] = 0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        mask = mask.to(torch.long)

        return image, mask

# ==========================================
# TRAINING LOOP
# ==========================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

# ==========================================
# VISUALIZATION
# ==========================================
def save_predictions_as_images(loader, model, epoch, folder="training_results", device="cuda"):
    model.eval() # vazoume to modelo se leitourgia aksiologhshs
    
    # pairnoume to prwto batch apo ton loader gia to paradeigma
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            # to modelo vgazei pithanotites gia 3 classes. pairnoume thn megalyterh (argmax)
            preds = torch.argmax(preds, dim=1) 
        
        # metafora sth CPU gia na mporesei h matplotlib na ta zografisei
        x = x.cpu()
        y = y.cpu()
        preds = preds.cpu()

        # dialegoume th prwth eikona tou batch [0]
        img = x[0].permute(1, 2, 0).numpy() # apo (Channels, H, W) se (H, W, Channels)
        
        # epeidh eixame kanei normalize, prepei na kanoume un-normalize gia na fainetai to RGB
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        mask_gt = y[0].numpy()
        mask_pred = preds[0].numpy()

        # ftiaxnoume to plot (1 seira, 3 sthles)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # xrisimopoioume colormap 'viridis' gia na xromatisei tis klaseis 0,1,2
        axes[1].imshow(mask_gt, cmap='viridis', vmin=0, vmax=2)
        axes[1].set_title("GT (Ground Truth)")
        axes[1].axis("off")

        axes[2].imshow(mask_pred, cmap='viridis', vmin=0, vmax=2)
        axes[2].set_title(f"Pred (Epoch {epoch})")
        axes[2].axis("off")

        # save img
        save_path = os.path.join(folder, f"result_epoch_{epoch}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        break # theloume mono mia eikona opote break loop
        
    model.train() # epistrefoume se leitourgia ekpaidefshs

# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    print(f"Χρησιμοποιείται: {DEVICE}")

    train_dataset = BDDDrivableDataset(
        images_dir=TRAIN_IMG_DIR,
        masks_dir=TRAIN_MASK_DIR,
        transform=get_train_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        print(f"\n--- Εποχή {epoch+1}/{EPOCHS} ---")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Μέσο Loss εποχής: {avg_loss:.4f}")

        # save model
        model_path = os.path.join(MODELS_DIR, f"unet_drivable_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Το μοντέλο αποθηκεύτηκε στο: {model_path}")

        # save results
        save_predictions_as_images(train_loader, model, epoch+1, folder=RESULTS_DIR, device=DEVICE)
        print(f"Το οπτικό αποτέλεσμα αποθηκεύτηκε στον φάκελο {RESULTS_DIR}")
