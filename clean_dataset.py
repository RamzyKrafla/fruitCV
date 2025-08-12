import os
from pathlib import Path
from PIL import Image
import shutil

def is_valid_image(file_path):
    """Check if an image file is valid and can be opened"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image
        return True
    except Exception:
        return False

def clean_dataset(data_dir):
    """Remove corrupted images from the dataset"""
    data_path = Path(data_dir)
    removed_count = 0
    
    # Walk through all subdirectories
    for split_dir in data_path.iterdir():
        if split_dir.is_dir():
            print(f"Checking {split_dir.name}...")
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    print(f"  Checking {class_dir.name}...")
                    for img_file in class_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                            if not is_valid_image(img_file):
                                print(f"    Removing corrupted file: {img_file}")
                                img_file.unlink()
                                removed_count += 1
    
    print(f"\nRemoved {removed_count} corrupted images.")
    return removed_count

if __name__ == '__main__':
    clean_dataset("data/fruit_subset")
