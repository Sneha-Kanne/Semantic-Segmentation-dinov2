"""
train_segmentation_v2.py
Improved training — targets 85%+ IoU
Drop-in replacement for train_segmentation.py (original files unchanged)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm

plt.switch_backend('Agg')

# ── Reuse value_map exactly as original ──────────────────────────────────────
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


# ── IMPROVEMENT 1: Data Augmentation ─────────────────────────────────────────
class AugmentedMaskDataset(Dataset):
    def __init__(self, data_dir, img_size, augment=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.img_size = img_size  # (H, W)
        self.augment = augment
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask  = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))

        h, w = self.img_size
        image = TF.resize(image, [h, w])
        mask  = TF.resize(mask,  [h, w], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            # Random vertical flip
            if random.random() > 0.3:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
            # Random crop + resize
            if random.random() > 0.5:
                i, j, ch, cw = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.6, 1.0), ratio=(4/3, 16/9))
                image = TF.resized_crop(image, i, j, ch, cw, [h, w])
                mask  = TF.resized_crop(mask,  i, j, ch, cw, [h, w],
                                        interpolation=TF.InterpolationMode.NEAREST)
            # Color jitter (image only)
            image = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)(image)

        image = self.normalize(TF.to_tensor(image))
        mask  = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ── IMPROVEMENT 2: Deeper Segmentation Head ──────────────────────────────────
class ImprovedSegHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.stem  = _block(in_channels, 256)
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block1 = _block(256, 128)
        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block2 = _block(128, 64)
        self.head  = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(self.up1(x))
        x = self.block2(self.up2(x))
        return self.head(x)


# ── IMPROVEMENT 3: Combined Loss (CE + Dice) ─────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        inter = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.cw, self.dw = ce_weight, dice_weight

    def forward(self, logits, targets):
        return self.cw * self.ce(logits, targets) + self.dw * self.dice(logits, targets)


# ── Metrics (same logic as original) ─────────────────────────────────────────
def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        ious.append((inter / union).item() if union > 0 else float('nan'))
    return np.nanmean(ious)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    batch_size = 4         # increase if VRAM allows
    n_epochs   = 60        # was 10
    lr         = 3e-4
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir    = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(script_dir, 'train_stats_v2')
    os.makedirs(output_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    trainset = AugmentedMaskDataset(data_dir, (h, w), augment=True)
    valset   = AugmentedMaskDataset(val_dir,  (h, w), augment=False)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(trainset)}  Val: {len(valset)}")

    # ── IMPROVEMENT 4: Larger backbone (ViT-Base) ────────────────────────────
    print("Loading DINOv2 ViT-Base backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval()
    backbone.to(device)

    # ── IMPROVEMENT 5: Fine-tune last 4 transformer blocks ───────────────────
    for name, param in backbone.named_parameters():
        param.requires_grad = False          # freeze all first
    for name, param in backbone.named_parameters():
        if any(f"blocks.{i}" in name for i in [8, 9, 10, 11]):
            param.requires_grad = True       # unfreeze last 4 blocks

    # Get embedding dim
    with torch.no_grad():
        sample, _ = trainset[0]
        feat = backbone.forward_features(sample.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_emb = feat.shape[2]
    print(f"Embedding dim: {n_emb}")

    # ── Model / Loss / Optimizer ──────────────────────────────────────────────
    classifier = ImprovedSegHead(n_emb, n_classes, w // 14, h // 14).to(device)
    criterion  = CombinedLoss(ce_weight=0.5, dice_weight=0.5)

    # Separate LRs: backbone gets 10× smaller LR
    optimizer = optim.AdamW([
        {'params': [p for p in backbone.parameters()  if p.requires_grad], 'lr': lr / 10},
        {'params': classifier.parameters(), 'lr': lr},
    ], weight_decay=1e-4)

    # ── IMPROVEMENT 6: Cosine LR schedule with warmup ────────────────────────
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr / 10, lr],
        epochs=n_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05
    )

    best_iou   = 0.0
    best_path  = os.path.join(script_dir, "segmentation_head_v2_best.pth")

    print("\nStarting training...")
    for epoch in range(n_epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        classifier.train()
        backbone.train()       # allows grad for unfrozen layers
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # backbone: partial grad
            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ──────────────────────────────────────────────────────────
        classifier.eval()
        backbone.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                val_ious.append(compute_iou(outputs, labels))

        mean_val_iou = np.nanmean(val_ious)
        print(f"Epoch {epoch+1:3d} | Loss {np.mean(train_losses):.4f} | Val IoU {mean_val_iou:.4f}")

        # ── Save best model ───────────────────────────────────────────────────
        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            torch.save(classifier.state_dict(), best_path)
            print(f"  ✓ New best IoU: {best_iou:.4f} — saved to {best_path}")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    # Also save final weights compatible with original test_segmentation.py name
    torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head.pth"))
    print("Saved final weights as segmentation_head.pth (compatible with test_segmentation.py)")


if __name__ == "__main__":
    main()