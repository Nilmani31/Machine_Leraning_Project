"""
Split training_data into train/test folders (80/20 split)
Run this ONCE before training to create a proper test set
"""
import os
import shutil
import random
from pathlib import Path

def create_train_test_split(source_dir='training_data', 
                            train_dir='training_data_train', 
                            test_dir='training_data_test',
                            test_split=0.2):
    """
    Split images from source_dir into train and test directories
    test_split: fraction of data to use for testing (default 0.2 = 20%)
    """
    
    if not os.path.exists(source_dir):
        print(f"❌ Error: {source_dir} not found!")
        return False
    
    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📊 CREATING TRAIN/TEST SPLIT")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Train Dir: {train_dir} (80%)")
    print(f"Test Dir: {test_dir} (20%)")
    print(f"{'='*70}\n")
    
    disease_folders = [d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d))]
    
    total_train = 0
    total_test = 0
    
    for disease in sorted(disease_folders):
        source_path = os.path.join(source_dir, disease)
        train_path = os.path.join(train_dir, disease)
        test_path = os.path.join(test_dir, disease)
        
        # Create disease folders
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(source_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        if not image_files:
            print(f"⚠️  No images found in {disease}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        split_point = int(len(image_files) * (1 - test_split))
        train_images = image_files[:split_point]
        test_images = image_files[split_point:]
        
        # Copy to train folder
        for img in train_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(train_path, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        # Copy to test folder
        for img in test_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(test_path, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        print(f"✅ {disease:45} | Train: {len(train_images):4} | Test: {len(test_images):4}")
        total_train += len(train_images)
        total_test += len(test_images)
    
    print(f"\n{'='*70}")
    print(f"✅ SPLIT COMPLETE!")
    print(f"{'='*70}")
    print(f"Total Training Images: {total_train}")
    print(f"Total Test Images:     {total_test}")
    print(f"\n✏️  NEXT STEPS:")
    print(f"1. Update train_both_models.py: Change data_dir='training_data_train'")
    print(f"2. Run: python train_both_models.py")
    print(f"3. In UI, upload images from: {test_dir}/")
    print(f"   This ensures you test on UNSEEN data!")
    print(f"{'='*70}\n")
    
    return True

if __name__ == "__main__":
    # Run the split
    create_train_test_split()
