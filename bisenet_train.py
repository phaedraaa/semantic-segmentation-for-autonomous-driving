import os
import glob
import random
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt

# ---------------------------
# PATHS
# ---------------------------
IMG_TRAIN_DIR = "/bdd100k_images_10k/10k/train"
IMG_VAL_DIR   = "/bdd100k_images_10k/10k/val"

MASK_TRAIN_DIR = "/bdd100k_seg_maps/labels/train"
MASK_VAL_DIR   = "/bdd100k_seg_maps/labels/val"

SAVE_DIR       = "./results_bisenet"
CHECKPOINT_DIR = "./models_bisenet"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# HYPERPARAMETERS
# ---------------------------
IMG_SIZE   = (640, 320)  # (width, height)
BATCH_SIZE = 4           
LR         = 1e-3
EPOCHS     = 25
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
IGNORE_LABEL = 255

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)

# ---------------------------
# LABEL MAP
# ---------------------------
def build_label_map(mask_dir, sample_limit=1000):
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))[:sample_limit]
    uniq = set()
    print("Scanning masks for label IDs...")
    for p in tqdm(files):
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        vals = np.unique(m).astype(int).tolist()
        uniq.update(vals)
    uniq = sorted(list(uniq))
    print("Unique IDs:", uniq)
    ids = [u for u in uniq if u != IGNORE_LABEL]
    id_to_idx = {orig: idx for idx, orig in enumerate(ids)}
    print("Mapped classes:", id_to_idx)
    return id_to_idx, ids

id_to_idx, classes_orig = build_label_map(MASK_TRAIN_DIR)
NUM_CLASSES = len(classes_orig)
print("NUM_CLASSES =", NUM_CLASSES)

# ---------------------------
# DATASET
# ---------------------------
class BDDSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, id_to_idx, img_size=(640, 320), augment=False):
        self.augment   = augment
        self.id_to_idx = id_to_idx
        self.img_size  = img_size

        self.img_paths  = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        img_map = {}
        for p in self.img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            img_map[stem] = p

        mask_map = {}
        for p in self.mask_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.endswith("_train_id"):
                stem = stem[:-len("_train_id")]
            mask_map[stem] = p

        common = sorted(set(img_map.keys()).intersection(mask_map.keys()))
        self.pairs = [(img_map[k], mask_map[k]) for k in common]
        print(f"Loaded {len(self.pairs)} pairs from {img_dir}")

        if augment:
            self.img_transform = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def remap_mask(self, mask):
        out = np.full_like(mask, IGNORE_LABEL, dtype=np.uint8)
        for orig, new in self.id_to_idx.items():
            out[mask == orig] = new
        return out

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]

        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)

        # crop bottom 65% 
        h, w = img.shape[:2]
        crop_start = int(h * 0.35)
        img  = img[crop_start:]
        mask = mask[crop_start:]

        if self.augment and random.random() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        img  = cv2.resize(img,  self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        mask = self.remap_mask(mask)

        img_t  = self.img_transform(img)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t

# ---------------------------
# METRICS
# ---------------------------
def per_class_iou(pred, target, n_classes, ignore_index=255):
    ious = []
    for cls in range(n_classes):
        pred_mask   = (pred == cls)
        target_mask = (target == cls)
        valid       = (target != ignore_index)
        inter = (pred_mask & target_mask & valid).sum()
        union = ((pred_mask | target_mask) & valid).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return ious

# ---------------------------
# BiSeNet-like ARCHITECTURE
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class SpatialPath(nn.Module):
    """
    Spatial path: 3 conv blocks με stride=2 για να κρατήσουμε spatial info
    τελικό feature map: 1/8 της ανάλυσης
    """
    def __init__(self):
        super().__init__()
        self.layer1 = ConvBNReLU(3,   64, k=7, s=2, p=3)  # H/2
        self.layer2 = ConvBNReLU(64,  64, k=3, s=2, p=1)  # H/4
        self.layer3 = ConvBNReLU(64, 128, k=3, s=2, p=1)  # H/8

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B,128,H/8,W/8)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv(x)
        attn = self.attention(feat)
        return feat * attn

class FeatureFusionModule(nn.Module):
    def __init__(self, sp_ch, ct_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(sp_ch + ct_ch, out_ch, k=1, s=1, p=0)
        # channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, sp, ct):
        x = torch.cat([sp, ct], dim=1)
        x = self.conv(x)
        attn = self.attention(x)
        x = x + x * attn
        return x

class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # backbone for context path: ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # kratame mexri layer3, layer4
        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )  # -> 1/4
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32

        self.spatial_path = SpatialPath()

        # ARMs for C3 (layer3) and C4 (layer4)
        self.arm16 = AttentionRefinementModule(256, 256)  # layer3 out_ch=256
        self.arm32 = AttentionRefinementModule(512, 256)  # layer4 out_ch=512 -> 256 out


        # global context for the deepest feature
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )


        # upsample layers
        self.up32_to_16 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up16_to_8  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # prosarmogh channels from context path to fusion
        self.conv_ct16 = ConvBNReLU(256, 128, k=3, s=1, p=1)
        self.conv_ct8  = ConvBNReLU(128, 128, k=3, s=1, p=1)

        self.ffm = FeatureFusionModule(sp_ch=128, ct_ch=128, out_ch=256)

        # teliko classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        # upsample to original size
        self.up8_to_1 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # spatial path
        sp = self.spatial_path(x)  # (B,128,H/8,W/8)

        # context path (resnet18)
        x0 = self.conv1(x)   # 1/4
        x1 = self.layer1(x0) # 1/4
        x2 = self.layer2(x1) # 1/8
        x3 = self.layer3(x2) # 1/16
        x4 = self.layer4(x3) # 1/32

        # ARM on 1/32
        cx4 = self.arm32(x4)                       # (B,512,H/32,W/32)
        gp  = self.global_pool(x4)                 # (B,512,1,1)
        cx4 = cx4 + gp                             # global context
        cx4 = self.up32_to_16(cx4)                 # -> 1/16

        # ARM on 1/16
        cx3 = self.arm16(x3)                       # (B,256,H/16,W/16)
        cx3 = cx3 + cx4                            # fuse

        # upsample to 1/8
        cx3_up = self.up16_to_8(cx3)               # (B,256,H/8,W/8)
        cx3_up = self.conv_ct16(cx3_up)            # (B,128,H/8,W/8)

        # context feature σε 1/8
        ct = self.conv_ct8(cx3_up)                 # (B,128,H/8,W/8)

        # Feature fusion
        feat = self.ffm(sp, ct)                    # (B,256,H/8,W/8)

        # classifier + upsample
        out = self.classifier(feat)                # (B,C,H/8,W/8)
        out = self.up8_to_1(out)                   # (B,C,H,W) approx

        # final resize to input size
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

# ---------------------------
# TRAIN / VAL LOOPS
# ---------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = F.cross_entropy(out, masks, ignore_index=IGNORE_LABEL)
        loss.backward()
        optimizer.step()

        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

def validate(model, loader):
    model.eval()
    total = 0.0
    iou_sum   = np.zeros(NUM_CLASSES)
    iou_count = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out  = model(imgs)
            loss = F.cross_entropy(out, masks, ignore_index=IGNORE_LABEL)
            total += loss.item() * imgs.size(0)

            preds    = out.argmax(1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for p, t in zip(preds, masks_np):
                ious = per_class_iou(p, t, NUM_CLASSES, ignore_index=IGNORE_LABEL)
                for i, v in enumerate(ious):
                    if not np.isnan(v):
                        iou_sum[i]   += v
                        iou_count[i] += 1

    mean_iou_per_class = iou_sum / np.maximum(iou_count, 1)
    mean_iou = np.nanmean(mean_iou_per_class)
    return total / len(loader.dataset), mean_iou

# ---------------------------
# VISUALIZATION
# ---------------------------
def save_sample_predictions(model, dataset, epoch, n=2):
    model.eval()
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.random.seed(123)
    colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

    with torch.no_grad():
        for i in range(n):
            img_t, mask_t = dataset[i]
            img_in = img_t.unsqueeze(0).to(DEVICE)
            out = model(img_in)
            pred = out.argmax(1)[0].cpu().numpy()
            mask_np = mask_t.numpy()

            img = img_t.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            mask_color = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            pred_color = np.zeros_like(mask_color)
            for c in range(NUM_CLASSES):
                mask_color[mask_np == c] = colors[c]
                pred_color[pred == c]    = colors[c]

            fig = plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img);        plt.axis("off")
            plt.subplot(1,3,2); plt.title("GT");    plt.imshow(mask_color); plt.axis("off")
            plt.subplot(1,3,3); plt.title("Pred");  plt.imshow(pred_color); plt.axis("off")
            plt.tight_layout()
            out_path = os.path.join(SAVE_DIR, f"epoch{epoch}_sample{i}.png")
            fig.savefig(out_path)
            plt.close(fig)

# ---------------------------
# MAIN
# ---------------------------
def main():
    train_ds = BDDSegDataset(IMG_TRAIN_DIR, MASK_TRAIN_DIR, id_to_idx, img_size=IMG_SIZE, augment=True)
    val_ds   = BDDSegDataset(IMG_VAL_DIR,  MASK_VAL_DIR,  id_to_idx, img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = BiSeNet(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_iou = validate(model, val_loader)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val mIoU:   {val_iou:.4f}")
        print("Current LR:", optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"bisenet_epoch{epoch}.pth"))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "bisenet_best.pth"))
            print("==> New best model saved")

        save_sample_predictions(model, val_ds, epoch, n=2)

    print("\nTraining finished. Best mIoU:", best_iou)

if __name__ == "__main__":
    main()
