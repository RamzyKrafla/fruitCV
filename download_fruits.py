import os
import requests
import random
import shutil
from pathlib import Path
from duckduckgo_search import DDGS
import time

random.seed(42)

CLASSES = {
    "apple": "apple fruit",
    "banana": "banana fruit", 
    "mango": "mango fruit",
}

RAW = Path("data/raw")
OUT = Path("data/fruit_subset")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Create directories
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Clear existing data
print("Clearing existing dataset...")
if RAW.exists():
    shutil.rmtree(RAW)
    RAW.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

def download_image(url, filepath):
    """Download an image from URL to filepath"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def download_images_for_class(class_name, query, limit=50):
    """Download images for a specific class"""
    class_dir = RAW / class_name
    class_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {limit} images for {class_name}...")
    
    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=limit*2))  # Get more to account for failures
    
    downloaded = 0
    for i, result in enumerate(results):
        if downloaded >= limit:
            break
            
        url = result['image']
        filepath = class_dir / f"{class_name}_{downloaded:03d}.jpg"
        
        if download_image(url, filepath):
            downloaded += 1
            print(f"  Downloaded {downloaded}/{limit}: {filepath.name}")
        
        time.sleep(0.5)  # Be nice to the servers
    
    print(f"Successfully downloaded {downloaded} images for {class_name}")
    return downloaded

# 1) Download images for each class
for cls, query in CLASSES.items():
    download_images_for_class(cls, query, limit=300)

# 2) Make split folders
for split in SPLITS:
    for cls in CLASSES:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

def is_img(p): 
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}

# 3) Split each class into train/val/test and copy
for cls in CLASSES:
    files = [p for p in (RAW / cls).rglob("*") if p.is_file() and is_img(p)]
    if not files:
        print(f"Warning: No images found for {cls}")
        continue
        
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])
    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train+n_val],
        "test":  files[n_train+n_val:],
    }
    
    idx = {"train": 0, "val": 0, "test": 0}
    for split, paths in splits.items():
        for p in paths:
            dst = OUT / split / cls / f"{cls}_{idx[split]}{p.suffix.lower()}"
            shutil.copy2(p, dst)
            idx[split] += 1
    
    print(f"Split {cls}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

print("\nDone! Your dataset is in data/fruit_subset/ with train/val/test splits.")
print("Dataset structure:")
for split in SPLITS:
    for cls in CLASSES:
        split_dir = OUT / split / cls
        if split_dir.exists():
            count = len(list(split_dir.glob("*")))
            print(f"  {split}/{cls}: {count} images")
