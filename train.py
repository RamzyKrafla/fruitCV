import os
# Provides small utilities for filesystem operations. We will use it to create a "model" folder.

import random
# Python's RNG; we seed it so that any randomness (like shuffling) is repeatable across runs.

import torch
# Core PyTorch library: tensors, autograd, device management (CPU/GPU).

from torch import nn, optim
# nn: neural network building blocks (Linear layers, losses).
# optim: optimizers that update parameters (Adam, SGD, etc.).

from torch.utils.data import DataLoader
# DataLoader batches data and optionally shuffles it so we train efficiently.

from torchvision import datasets, transforms, models
# datasets: ready-made dataset interfaces like ImageFolder.
# transforms: image preprocessing and augmentation operators.
# models: pretrained CNN architectures such as ResNet.

# --------------------------
# 0) Reproducibility and basic configuration
# --------------------------

random.seed(42)
# Fixes Python's RNG so shuffles are repeatable. "42" is a common choice, any integer works.

torch.manual_seed(42)
# Fixes PyTorch's RNG (used in weight init or some transforms) for repeatability.

DATA_DIR = "data/fruit_subset"
# Root folder that must contain "train", "val", and "test" subfolders.
# Inside each split, you must have one subfolder per class (apple, banana, mango).

BATCH_SIZE = 32
# Number of images processed per optimizer step.
# 32 is a practical default: big enough for stable gradients, small enough to fit in typical laptop GPU/CPU RAM.

EPOCHS = 15
# How many full passes over the training set.
# With 300 images per class and 3 classes, 8 epochs is a sensible starting point for a frozen backbone.

LR_HEAD = 1e-3
# Learning rate for the small classifier head we will train.
# 1e-3 works well with Adam on a tiny head; large enough to learn in a few epochs, small enough to stay stable.

NUM_WORKERS = 0
# How many background processes load data.
# 0 disables multiprocessing to avoid issues on macOS; single-threaded loading is fine for this dataset size.

IMAGE_SIZE = 224
# Most ImageNet-pretrained CNNs expect 224x224 inputs; we resize or crop to that size.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# Channel-wise normalization constants used during ImageNet pretraining.
# We normalize with the same statistics so the pretrained weights see inputs in the distribution they expect.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# If a CUDA-capable GPU is available, use it for speed; otherwise run on CPU.

# --------------------------
# 1) Image transforms (preprocessing and augmentation)
# --------------------------

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    # Randomly crops a region of the image and resizes to 224x224.
    # scale=(0.7, 1.0) means the crop will cover between 70% and 100% of the original area.
    # This teaches robustness to framing and slight zoom changes.

    transforms.RandomHorizontalFlip(),
    # Randomly flips images left-right with 50% probability.
    # Many fruits look similar when mirrored, so this augmentation is safe and increases diversity.

    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
    # Slight random changes to brightness, contrast, and saturation.
    # These simulate different lighting or camera conditions so the model does not overfit to a single look.

    transforms.ToTensor(),
    # Converts a PIL image (H x W x C in [0,255]) into a PyTorch tensor (C x H x W in [0,1]).

    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # Standardizes each channel so that inputs match the distribution seen during ImageNet pretraining.
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # For validation and test, we do not randomize.
    # We deterministically resize to 224x224 so evaluation is consistent.

    transforms.ToTensor(),
    # Same conversion to tensor.

    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # Same normalization so the pretrained network sees familiar statistics.
])

# --------------------------
# 2) Datasets and data loaders
# --------------------------

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
# ImageFolder assumes a directory per class inside DATA_DIR/train.
# It assigns integer labels in alphabetical order of folder names, and applies train_tfms to each image.

val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tfms)
# Same for validation data, but with deterministic transforms so accuracy is measured fairly.

test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tfms)
# Same for the final test set.

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
# Creates mini-batches for training.
# shuffle=True randomizes sample order each epoch, which helps generalization.

val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# Validation loader does not shuffle.
# We want stable measurements across epochs.

test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# Test loader also does not shuffle, for the same reason.

classes = train_ds.classes
# List of class names in the order ImageFolder assigned them (alphabetical by folder name).
# Example: ['apple', 'banana', 'mango'].

num_classes = len(classes)
# 3 in this project, used to size the final classifier layer.

print("Classes:", classes)
# Quick sanity check that the classes are detected correctly.

# --------------------------
# 3) Load a pretrained CNN and adapt it to our classes
# --------------------------

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Loads a ResNet-18 with weights pretrained on ImageNet.
# ResNet-18 is a convolutional network that turns pixels into a rich feature representation.

for p in model.parameters():
    p.requires_grad = False
# Freeze all parameters so the pretrained feature extractor stays fixed.
# This is classic transfer learning: keep the general visual features, only train a small classifier on top.

in_feats = model.fc.in_features
# Read how many input features the original final fully connected (fc) layer expects.
# In ResNet-18 this is typically 512.

model.fc = nn.Linear(in_feats, num_classes)
# Replace the original 1000-class ImageNet head with a new Linear layer that outputs "num_classes" scores.
# This is the classifier head that we will actually train.

model = model.to(DEVICE)
# Move the entire model to the selected device (GPU if available, else CPU).

# --------------------------
# 4) Loss function and optimizer
# --------------------------

criterion = nn.CrossEntropyLoss()
# Cross-entropy compares the model's raw class scores (logits) to the true class index.
# It combines a softmax with negative log-likelihood internally, so we do not apply softmax ourselves.

optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)
# Adam is an adaptive optimizer that usually converges quickly on small heads.
# We pass only model.fc.parameters() because everything else is frozen; this prevents wasted computation and keeps training fast.

# --------------------------
# 5) One epoch helper: trains or evaluates depending on "train_mode"
# --------------------------

def run_epoch(loader, train_mode=True):
    # Defines a function that loops through one DataLoader and returns average loss and accuracy.

    if train_mode:
        model.train()
    else:
        model.eval()
    # model.train() enables training-specific layers like dropout and tells batch norm to update running stats.
    # model.eval() disables those behaviors for deterministic evaluation.

    total_loss = 0.0
    # Accumulates loss across all samples so we can report an average at the end.

    total_correct = 0
    # Counts how many predictions match the true labels.

    total_seen = 0
    # Counts how many samples we processed; used to compute averages.

    torch.set_grad_enabled(train_mode)
    # Enables gradient tracking only during training to save memory and compute during evaluation.

    for x, y in loader:
        # Iterate over mini-batches from the loader.
        # x has shape [batch_size, 3, H, W], y has shape [batch_size] with integer labels.

        x, y = x.to(DEVICE), y.to(DEVICE)
        # Move both images and labels to the target device (GPU or CPU).

        if train_mode:
            optimizer.zero_grad()
        # Clears gradients from the previous step.
        # Gradients accumulate by default in PyTorch, so we must zero them each iteration when training.

        logits = model(x)
        # Forward pass: the model maps images to raw class scores (logits) with shape [batch_size, num_classes].

        loss = criterion(logits, y)
        # Compute the cross-entropy loss between logits and the ground-truth labels.

        if train_mode:
            loss.backward()
            # Backpropagation: computes gradients of the loss with respect to the trainable parameters (only the new head).

            optimizer.step()
            # Applies the gradients to update the head's weights.

        total_loss += loss.item() * x.size(0)
        # Add the batch loss multiplied by the batch size so that averaging later is correct even if the last batch is smaller.

        preds = logits.argmax(dim=1)
        # Predicted class per image: index of the largest logit along the class dimension.

        total_correct += (preds == y).sum().item()
        # Count how many predictions matched the labels in this batch.

        total_seen += y.size(0)
        # Increase the total number of samples processed.

    avg_loss = total_loss / max(1, total_seen)
    # Average loss per sample across the entire loader. max(1, ...) avoids division by zero on empty sets.

    avg_acc  = total_correct / max(1, total_seen)
    # Accuracy across the entire loader.

    return avg_loss, avg_acc
    # Return metrics so the training loop can log them and decide whether to save a checkpoint.

# --------------------------
# 6) Training loop with validation and checkpointing
# --------------------------

if __name__ == '__main__':
    os.makedirs("model", exist_ok=True)
    # Ensure we have a folder to save the best model checkpoint.

    best_val_acc = 0.0
    # Track the best validation accuracy seen so far so we can save the best checkpoint.

    for epoch in range(1, EPOCHS + 1):
        # Loop over epochs; each epoch trains once over the train set and evaluates once on the validation set.

        tr_loss, tr_acc = run_epoch(train_loader, train_mode=True)
        # Train for one epoch: updates only the classifier head.

        va_loss, va_acc = run_epoch(val_loader,   train_mode=False)
        # Evaluate on validation set without updating weights: gives an unbiased signal for early stopping or model selection.

        if va_acc > best_val_acc:
            # If this epoch improved validation accuracy, keep this model.

            best_val_acc = va_acc
            # Update the running best so future epochs must beat this to be saved.

            torch.save({
                "model_state": model.state_dict(),
                "classes": classes
            }, "model/best_resnet18.pt")
            # Save a checkpoint that contains:
            # - model_state: only the weights (compact and portable)
            # - classes: the class name order so prediction code can map indices back to labels.

        print(f"Epoch {epoch} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f}  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")
        # Log training and validation metrics so you can monitor learning progress.

    # --------------------------
    # 7) Final evaluation on held-out test set
    # --------------------------

    ckpt = torch.load("model/best_resnet18.pt", map_location=DEVICE)
    # Load the best checkpoint chosen by validation accuracy.
    # map_location ensures it loads on whatever device we are using now.

    model.load_state_dict(ckpt["model_state"])
    # Put the saved weights back into the model.

    test_loss, test_acc = run_epoch(test_loader, train_mode=False)
    # Evaluate once on the test set, which was never used for training or model selection.

    print(f"Test accuracy: {test_acc:.3f}")
    # Report the final generalization score. This is the performance you would expect on new fruit images.