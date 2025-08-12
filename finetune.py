# ---- 0) Load your previously saved best frozen-head model ----
import torch
from torchvision import models, datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import random

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Configuration
DATA_DIR = "data/fruit_subset"
BATCH_SIZE = 32
NUM_WORKERS = 0  # Disable multiprocessing for macOS compatibility
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLD_CKPT = "model/best_resnet18.pt"            # from your previous run
NEW_CKPT = "model/best_resnet18_finetune.pt"   # we'll only write this if val improves

# Data transforms (same as training)
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Load datasets
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tfms)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=eval_tfms)
test_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=eval_tfms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classes = train_dataset.classes
print(f"Classes: {classes}")

ckpt = torch.load(OLD_CKPT, map_location=DEVICE)

# Rebuild the same architecture
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, len(classes))
model.load_state_dict(ckpt["model_state"])     # start from your best weights
model.to(DEVICE)

# ---- 1) Unfreeze only the last block for light fine-tuning ----
for p in model.parameters():
    p.requires_grad = False
for p in model.layer4.parameters():            # last residual block
    p.requires_grad = True
for p in model.fc.parameters():                # and the classifier head
    p.requires_grad = True

# ---- 2) Optimizer / loss / (optional) regularization ----
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # small smoothing often helps
optimizer = optim.Adam(
    [
        {"params": model.fc.parameters(),     "lr": 1e-3, "weight_decay": 1e-4},
        {"params": model.layer4.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
    ]
)

# Optional: reduce LR if val loss plateaus
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

# ---- 3) train/val epoch helper (reuse yours if you already have it) ----
def run_epoch(loader, train_mode=True):
    model.train() if train_mode else model.eval()
    import torch
    total_loss, total_correct, total_seen = 0.0, 0, 0
    torch.set_grad_enabled(train_mode)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train_mode:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total_seen += y.size(0)
    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)

# ---- 4) Early stopping + best-on-val checkpointing ----
EPOCHS = 10                  # you can set this higher; early stopping will halt when needed
PATIENCE = 4                 # stop if val acc doesn’t improve for 4 epochs
best_val_acc = 0.0           # track best validation accuracy during fine-tune
epochs_without_improve = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader,   False)

    # Step LR scheduler on validation loss trend
    scheduler.step(va_loss)

    improved = va_acc > best_val_acc + 1e-4   # tiny epsilon avoids float ties
    if improved:
        best_val_acc = va_acc
        epochs_without_improve = 0
        torch.save({"model_state": model.state_dict(), "classes": classes}, NEW_CKPT)
        print(f"[✓] Epoch {epoch}: val acc improved to {va_acc:.3f} -> saved {NEW_CKPT}")
    else:
        epochs_without_improve += 1
        print(f"[i] Epoch {epoch}: no val acc improvement ({epochs_without_improve}/{PATIENCE})")

    print(f"Epoch {epoch} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")

    if epochs_without_improve >= PATIENCE:
        print(f"[STOP] Early stopping (no improvement for {PATIENCE} epochs).")
        break

# ---- 5) Decide which checkpoint to use after fine-tune ----
import os

if os.path.exists(NEW_CKPT):
    # Fine-tune produced a better model; evaluate it on test
    final_ckpt_path = NEW_CKPT
    print(f"Using fine-tuned checkpoint: {NEW_CKPT}")
else:
    # No improvement -> keep your original best
    final_ckpt_path = OLD_CKPT
    print(f"No fine-tune improvement; keeping original: {OLD_CKPT}")

# Load the chosen checkpoint and do a final test evaluation
final_ckpt = torch.load(final_ckpt_path, map_location=DEVICE)
model.load_state_dict(final_ckpt["model_state"])
test_loss, test_acc = run_epoch(test_loader, False)
print(f"Final test accuracy from [{final_ckpt_path}]: {test_acc:.3f}")