import os
import glob
import random
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt

# ---------------------------
# PATHS
# ---------------------------
IMG_TRAIN_DIR = "/bdd100k_images_10k/10k/train"
IMG_VAL_DIR   = "/bdd100k_images_10k/10k/val"
MASK_TRAIN_DIR = "/bdd100k_seg_maps/labels/train"
MASK_VAL_DIR   = "/bdd100k_seg_maps/labels/val"

SAVE_DIR = "./results_unet"
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_DIR = "./models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# PARAMETERS
# ---------------------------
IMG_SIZE = (512, 256)   # (width, height) 
BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 20
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

# ---------------------------
# Utility: build mapping from original label IDs -> consecutive indices
# BDD100K doesnt have consequtive indices -> Problem with Pytorch
# keep 255 as IGNORE (do not map it)
# ---------------------------
def build_label_map(mask_dir, sample_limit=None):
    """
    Scan mask files to find unique label IDs and create mapping
    excluding 255 (ignore). Returns:
      id_to_idx: {orig_id: new_idx}
      classes: [orig_id1, orig_id2, ...] in new_idx order
    """
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if sample_limit:
        files = files[:sample_limit]
    uniq = set()
    print("Scanning masks to build label map (this may take a bit)...")
    for p in tqdm(files):
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        vals = np.unique(m)
        for v in vals:
            uniq.add(int(v))
    uniq = sorted(list(uniq))
    print("Unique IDs found (sample):", uniq[:50])
    # exclude ignore id 255 from mapping (we will keep it as ignore_index)
    ids = [i for i in uniq if i != 255]
    id_to_idx = {orig: idx for idx, orig in enumerate(ids)}
    print("Mapped classes (orig_id -> new_idx):", id_to_idx)
    return id_to_idx, ids

# Build mapping from training masks
id_to_idx, classes_orig = build_label_map(MASK_TRAIN_DIR, sample_limit=2000)
NUM_CLASSES = len(classes_orig)
IGNORE_LABEL = 255
print("Number of classes (after remap):", NUM_CLASSES)

# ---------------------------
# Dataset
# ---------------------------
class BDDSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, id_to_idx, img_size=(512,256), augment=False):
        self.id_to_idx = id_to_idx
        self.img_size = img_size
        self.augment = augment

        # Load image and mask paths
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        # Build image dict: stem → full path
        img_map = {}
        for p in self.img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            img_map[stem] = p

        # Build mask dict: strip "_train_id" suffix
        mask_map = {}
        for p in self.mask_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.endswith("_train_id"):
                stem = stem[:-len("_train_id")]
            mask_map[stem] = p

        # Find matching pairs
        common = sorted(set(img_map.keys()).intersection(set(mask_map.keys())))
        self.pairs = [(img_map[k], mask_map[k]) for k in common]

        print(f"Found {len(self.pairs)} image-mask pairs in {img_dir}")

        # Define transforms
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def remap_mask(self, mask):
        """
        mask: numpy array HxW with original ids
        returns: np.array HxW with values 0..C-1 for classes, and 255 for ignore
        """
        out = np.full_like(mask, IGNORE_LABEL, dtype=np.uint8)
        for orig, newidx in self.id_to_idx.items():
            out[mask == orig] = newidx
        # any pixel that was not in id_to_idx remains 255
        return out

    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]
        # load image
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # load mask
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Mask failed to load: {mask_p}")
        # resize both
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = self.remap_mask(mask)
        # augmentation: simple horizontal flip
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        img_t = self.to_tensor(img)  # tensor float CHW
        mask_t = torch.from_numpy(mask.astype(np.int64))  # HxW long
        return img_t, mask_t

# ---------------------------
# U-Net model
# ---------------------------
# Conv2d -> Batch Normalization -> ReLU -> ...
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.enc1 = DoubleConv(3,64)
        self.enc2 = DoubleConv(64,128)
        self.enc3 = DoubleConv(128,256)
        self.enc4 = DoubleConv(256,512)
        self.pool = nn.MaxPool2d(2) #reduce img resolution in half everytime we go down a scale

        self.bottleneck = DoubleConv(512,1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024,512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512,256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256,128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128,64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = torch.cat([u4, c4], dim=1)
        c5 = self.dec4(u4)

        u3 = self.up3(c5)
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.dec3(u3)

        u2 = self.up2(c6)
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.dec2(u2)

        u1 = self.up1(c7)
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.dec1(u1)

        out = self.final(c8) 
        return out

# ---------------------------
# Metrics: per-class IoU
# ---------------------------
def per_class_iou(pred, target, n_classes, ignore_index=255):
    """
    pred: HxW numpy array (predicted class indices)
    target: HxW numpy array (true labels with ignore)
    returns: ious list length n_classes (float)
    """
    ious = []
    for cls in range(n_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        # exclude ignore pixels
        valid = (target != ignore_index)
        inter = np.logical_and(pred_mask, target_mask) & valid
        union = np.logical_or(pred_mask, target_mask) & valid
        inter_sum = inter.sum()
        union_sum = union.sum()
        if union_sum == 0:
            ious.append(float('nan'))  # no instances of this class in gt
        else:
            ious.append(inter_sum / union_sum)
    return ious

# ---------------------------
# Train / Validation loops
# ---------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(imgs)  # B x C x H x W
        loss = F.cross_entropy(preds, masks, ignore_index=IGNORE_LABEL)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, device, n_classes):
    model.eval()
    total_loss = 0.0
    iou_sum = np.zeros(n_classes, dtype=float)
    iou_count = np.zeros(n_classes, dtype=int)
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = F.cross_entropy(preds, masks, ignore_index=IGNORE_LABEL)
            total_loss += loss.item() * imgs.size(0)
            preds_arg = preds.argmax(dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for p, t in zip(preds_arg, masks_np):
                ious = per_class_iou(p, t, n_classes, ignore_index=IGNORE_LABEL)
                for i, val in enumerate(ious):
                    if not np.isnan(val):
                        iou_sum[i] += val
                        iou_count[i] += 1
    avg_loss = total_loss / len(loader.dataset)
    mean_iou_per_class = np.divide(iou_sum, np.maximum(iou_count, 1))
    mean_iou = np.nanmean([v for v in mean_iou_per_class if not np.isnan(v)])
    return avg_loss, mean_iou, mean_iou_per_class

# ---------------------------
# Main
# ---------------------------
def main():
    # datasets and loaders
    train_ds = BDDSegDataset(IMG_TRAIN_DIR, MASK_TRAIN_DIR, id_to_idx, img_size=IMG_SIZE, augment=True)
    val_ds   = BDDSegDataset(IMG_VAL_DIR, MASK_VAL_DIR, id_to_idx, img_size=IMG_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_mIoU = 0.0
    for epoch in range(1, EPOCHS+1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_mIoU, per_class = validate(model, val_loader, DEVICE, NUM_CLASSES)
        print(f"Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val mIoU: {val_mIoU:.4f}")
        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "id_to_idx": id_to_idx,
            "classes_orig": classes_orig
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch}.pth"))
        # save best
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"unet_best.pth"))
            print("Saved new best model.")
        # save a few sample predictions for inspection
        save_sample_predictions(model, val_loader, DEVICE, epoch)
    print("Training finished.")

def save_sample_predictions(model, loader, device, epoch, n=3):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        preds = model(imgs).argmax(dim=1).cpu().numpy()
    imgs_np = imgs.cpu().numpy()
    masks_np = masks.numpy()
    for i in range(min(n, imgs_np.shape[0])):
        img = imgs_np[i].transpose(1,2,0)  # CHW -> HWC
        # undo normalization
        img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
        img = np.clip(img, 0, 1)
        pred = preds[i]
        gt = masks_np[i]
        # map pred indices back to original id colors for visualization
        vis_pred = colorize_mask(pred)
        vis_gt = colorize_mask(gt)
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(img)
        axs[0].set_title("Image"); axs[0].axis("off")
        axs[1].imshow(vis_gt)
        axs[1].set_title("GT"); axs[1].axis("off")
        axs[2].imshow(vis_pred)
        axs[2].set_title("Pred"); axs[2].axis("off")
        out_p = os.path.join(SAVE_DIR, f"epoch{epoch}_sample{i}.png")
        fig.savefig(out_p, bbox_inches='tight')
        plt.close(fig)

# ---------------------------
# Visualization helpers
# ---------------------------
# build reverse mapping: new_idx -> orig_id (for coloring)
idx_to_orig = {v:k for k,v in id_to_idx.items()}

# Simple color palette (random deterministic colors)
palette = {}
random.seed(1234)
for new_idx in range(NUM_CLASSES):
    palette[new_idx] = tuple([int(255 * x) for x in np.random.rand(3)])

def colorize_mask(mask_idx):
    """
    mask_idx: HxW numpy with values 0..C-1 or 255 ignore
    returns HxW x 3 uint8 image with colors assigned to each class, ignore -> black
    """
    h,w = mask_idx.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for idx,orig in idx_to_orig.items():
        col = palette[idx]
        out[mask_idx == idx] = col
    out[mask_idx == IGNORE_LABEL] = (0,0,0)
    return out

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    main()
