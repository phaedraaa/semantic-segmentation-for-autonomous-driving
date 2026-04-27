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
from unet_train import UNet
from deeplab_train import create_deeplab_model
from bisenet_train import BiSeNet

# ==========================================
# ΕΠΙΛΟΓΗ ΜΟΝΤΕΛΟΥ 
# ==========================================
#MODEL_TO_TRAIN = "UNet"  
#MODEL_TO_TRAIN = "create_deeplab_model"  
MODEL_TO_TRAIN = "BiSeNet"  

# ==========================================
# PATHS & HYPERPARAMETERS
# ==========================================
TRAIN_IMG_DIR = "/home/faidbogi/ads/bdd100k_images_10k/10k/train" 
TRAIN_MASK_DIR = "/home/faidbogi/ads/bdd100k_drivable_maps/labels/train" 

MODELS_DIR = f"models_area_bisenet/{MODEL_TO_TRAIN}"
RESULTS_DIR = f"results_area_bisenet/{MODEL_TO_TRAIN}"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#IMAGE_HEIGHT = 360  
#IMAGE_WIDTH = 640   
IMAGE_HEIGHT = 352  
IMAGE_WIDTH = 640   
# ==========================================
# METRICS CLASS
# ==========================================
class SegmentationMetrics:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.total_intersections = np.zeros(num_classes)
        self.total_unions = np.zeros(num_classes)
        self.total_correct_pixels = 0
        self.total_valid_pixels = 0

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()

        for cls in range(self.num_classes):
            pred_inds = preds == cls
            target_inds = targets == cls
            
            intersection = np.logical_and(pred_inds, target_inds).sum()
            union = np.logical_or(pred_inds, target_inds).sum()

            self.total_intersections[cls] += intersection
            self.total_unions[cls] += union

        self.total_correct_pixels += (preds == targets).sum()
        self.total_valid_pixels += targets.size

    def compute(self):
        ious = self.total_intersections / np.maximum(self.total_unions, 1e-10)
        miou = np.mean(ious)
        pixel_accuracy = self.total_correct_pixels / self.total_valid_pixels
        return ious, miou, pixel_accuracy

# ==========================================
# DATA & DATASET
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

class BDDDrivableDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        all_images = os.listdir(images_dir)
        self.valid_images = []
        
        for img_name in all_images:
            if not img_name.endswith('.jpg'): continue
            mask_name = img_name.replace('.jpg', '_drivable_id.png')
            if os.path.exists(os.path.join(self.masks_dir, mask_name)):
                self.valid_images.append(img_name)
                
        print(f"Dataset Info: Βρέθηκαν {len(self.valid_images)} έγκυρα ζευγάρια.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        mask_name = img_name.replace('.jpg', '_drivable_id.png')
        
        image = cv2.imread(os.path.join(self.images_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(self.masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        
        if mask is None: mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else: mask[mask == 255] = 0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            
        return image, mask.to(torch.long)

# ==========================================
# TRAINING
# ==========================================
def train_fn(loader, model, optimizer, loss_fn, scaler, metrics):
    model.train()
    loop = tqdm(loader, desc=f"Training {MODEL_TO_TRAIN}")
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)

            if isinstance(predictions, dict):
                predictions = predictions['out']
            elif isinstance(predictions, tuple):
                predictions = predictions[0]

            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        metrics.update(predictions.detach(), targets.detach())
        loop.set_postfix(loss=loss.item())
        
    ious, miou, pixel_acc = metrics.compute()
    return total_loss / len(loader), ious, miou, pixel_acc

# ==========================================
# VISUALIZATION
# ==========================================
def save_predictions_as_images(loader, model, epoch, folder="training_results", device="cuda"):
    model.eval() 
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            
            if isinstance(preds, dict): preds = preds['out']
            elif isinstance(preds, tuple): preds = preds[0]
                
            preds = torch.argmax(preds, dim=1) 
        
        x, y, preds = x.cpu(), y.cpu(), preds.cpu()
        img = x[0].permute(1, 2, 0).numpy() 
        img = np.clip(np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406]), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(y[0].numpy(), cmap='viridis', vmin=0, vmax=2); axes[1].set_title("GT"); axes[1].axis("off")
        axes[2].imshow(preds[0].numpy(), cmap='viridis', vmin=0, vmax=2); axes[2].set_title(f"Pred (Epoch {epoch})"); axes[2].axis("off")

        plt.savefig(os.path.join(folder, f"result_epoch_{epoch}.png"), bbox_inches='tight')
        plt.close() 
        break 
    model.train() 

# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    print(f"Χρησιμοποιείται: {DEVICE} για εκπαίδευση του {MODEL_TO_TRAIN}")

    train_dataset = BDDDrivableDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=get_train_transforms())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    if MODEL_TO_TRAIN == "UNet":
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3).to(DEVICE)
    elif MODEL_TO_TRAIN == "create_deeplab_model":
        model = smp.DeepLabV3(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3).to(DEVICE)
    elif MODEL_TO_TRAIN == "BiSeNet":
        model = BiSeNet(num_classes=3).to(DEVICE) 
    else:
        raise ValueError("Άγνωστο μοντέλο! Επίλεξε UNet, DeepLabV3 ή BiSeNet.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        print(f"\n--- Εποχή {epoch+1}/{EPOCHS} ---")
        
        epoch_metrics = SegmentationMetrics(num_classes=3)
        avg_loss, ious, miou, pixel_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch_metrics)
        
        print(f"Μέσο Loss: {avg_loss:.4f}")
        print(f"Pixel Accuracy: {pixel_acc * 100:.2f}% | mIoU: {miou * 100:.2f}%")
        print(f"IoU -> Backgr: {ious[0]*100:.1f}%, Direct: {ious[1]*100:.1f}%, Alt: {ious[2]*100:.1f}%")

        save_path = os.path.join(MODELS_DIR, f"{MODEL_TO_TRAIN.lower()}_drivable_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        
        save_predictions_as_images(train_loader, model, epoch+1, folder=RESULTS_DIR, device=DEVICE)